"""
verity.core.types
=================
The movement. All data contracts for the Verity engine.

This file is the heartbeat of the system. It defines what everything is —
no logic, no I/O, no imports from anywhere else in the package. Every other
module imports from here. Nothing here imports from Verity.

Design discipline (the ETA 2824 principle):
  - Frozen dataclasses for all outputs — immutable after construction
  - Mutable dataclasses only for inputs built incrementally
  - Type aliases for all identifier strings — no ambiguous bare str
  - Every categorical value is an enum — no magic strings
  - The three-axis weight is a first-class type, not three loose floats
  - Decay parameters are a first-class type, not scattered constants
  - The module interface is a first-class type, not an implicit convention
  - Every field documented inline — the type file is also the spec

The five invariants enforced structurally here:
  1. uncertainty   — float field on ContextBundle, never Optional
  2. reasoning_trace — tuple field, never Optional, never empty by contract
  3. excluded      — tuple field, empty tuple not None when nothing excluded
  4. audit_id      — AuditRef (int) on every output touching the Merkle chain
  5. consent_ref   — ConsentRef on every operation touching the graph

Third-party module authors: ModuleManifest is your contract.
Backend implementors: GraphStore Protocol is in core.protocols.
Everything else builds on what is defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


# ── Type aliases ──────────────────────────────────────────────────────────────
#
# These are the lug-width standard. Every attachment point in the system
# speaks these types. No ambiguous bare strings for identifiers.

EntityId      = str   # Format: "{namespace}:{local_id}"  e.g. "patient:abc123"
ConsentRef    = str   # Format: "consent:{uuid4}"
ProvenanceRef = str   # Format: "prov:{uuid4}"
SessionId     = str   # Format: "session:{uuid4}"
ModuleId      = str   # Format: "verity_{domain}"         e.g. "verity_fhir"
AuditRef      = int   # Monotonically increasing sequence. Never reused.


# ── Enumerations ──────────────────────────────────────────────────────────────


class DataClassification(StrEnum):
    """
    Sensitivity classification for facts, edges, and context bundles.
    Inherits str so values serialize cleanly to/from RDF literals and JSON.

    Applied at ingestion — never inferred at query time.
    Classifications escalate only — PHI cannot become INTERNAL.
    Conservative default for unclassified facts: INTERNAL.

    Domain modules map their vocabularies to these tiers:
      FHIR sensitivity codes       → PHI / PII
      GDPR special categories      → PHI
      FIBO non-public information  → FINANCIAL
      LKIF legal privilege         → LEGAL
    """

    PUBLIC       = "public"        # No restriction. Safe in any context.
    INTERNAL     = "internal"      # Internal use. Not for external exposure.
    CONFIDENTIAL = "confidential"  # Restricted. Requires purpose justification.
    PHI          = "phi"           # Protected Health Info. HIPAA-regulated.
    PII          = "pii"           # Personally Identifiable Info. GDPR-regulated.
    FINANCIAL    = "financial"     # Financial data. SOX / GLBA / DORA-regulated.
    LEGAL        = "legal"         # Legal privilege or regulatory filing.

    @property
    def requires_consent(self) -> bool:
        """True for classifications that always require an explicit consent record."""
        return self in (
            DataClassification.PHI,
            DataClassification.PII,
            DataClassification.FINANCIAL,
            DataClassification.LEGAL,
        )

    @property
    def audit_on_access(self) -> bool:
        """True for classifications that generate an audit event on every access."""
        return self in (
            DataClassification.PHI,
            DataClassification.PII,
            DataClassification.FINANCIAL,
            DataClassification.LEGAL,
            DataClassification.CONFIDENTIAL,
        )

    @classmethod
    def escalate(
        cls,
        a: "DataClassification",
        b: "DataClassification",
    ) -> "DataClassification":
        """
        Return the higher of two classifications.
        Used when an edge inherits classification from its endpoints.
        Order: PUBLIC < INTERNAL < CONFIDENTIAL < PII < FINANCIAL < LEGAL < PHI
        """
        order = [
            cls.PUBLIC,
            cls.INTERNAL,
            cls.CONFIDENTIAL,
            cls.PII,
            cls.FINANCIAL,
            cls.LEGAL,
            cls.PHI,
        ]
        return a if order.index(a) >= order.index(b) else b


class Completeness(str):
    """
    How complete the assembled ContextBundle is relative to what exists
    in the graph for this query and purpose. Inherits str for serialization.

    Drives LLM reasoning calibration — an agent receiving PARTIAL
    should reason differently from one receiving SUFFICIENT.
    """
    EMPTY      = "empty"      # No relevant facts found.
    PARTIAL    = "partial"    # Relevant facts exist but significant gaps remain.
    SUFFICIENT = "sufficient" # Enough facts for the stated purpose. Normal case.
    SATURATED  = "saturated"  # All available relevant facts included.


class CheckpointDecision(str):
    """
    The outcome of a GOVERN checkpoint. Inherits str for serialization.

    VETOED is the default — timeout = veto, not approval.
    This is a non-negotiable behavioral invariant verified by canary
    tests at every engine boot.
    """
    APPROVED  = "approved"   # Human actively chose to proceed.
    VETOED    = "vetoed"     # Human declined, OR timeout elapsed. THE DEFAULT.
    MODIFIED  = "modified"   # Human approved with modifications.
    DEFERRED  = "deferred"   # Human needs more time. Action blocked until resolved.


class TrustSource(str):
    """
    Origin of a fact. Used to calculate initial trust_score at ingestion.
    Domain modules may define granular sources that map to these tiers.
    Inherits str for serialization.
    """
    HUMAN_VERIFIED   = "human_verified"    # Human entered or confirmed.    → 0.95
    INSTITUTIONAL    = "institutional"     # EHR, financial system, filing. → 0.90
    ALGORITHMIC_HIGH = "algorithmic_high"  # Model output, validated.       → 0.75
    ALGORITHMIC_LOW  = "algorithmic_low"   # Model output, unvalidated.     → 0.50
    INFERRED         = "inferred"          # Derived by the graph engine.   → 0.40
    UNKNOWN          = "unknown"           # Source not established.        → 0.20

    SCORES: dict[str, float] = {
        "human_verified":   0.95,
        "institutional":    0.90,
        "algorithmic_high": 0.75,
        "algorithmic_low":  0.50,
        "inferred":         0.40,
        "unknown":          0.20,
    }


class AuditEventType(str):
    """
    The type of event appended to the Merkle-chained audit trail.
    Every value here produces an immutable record. Inherits str for RDF.
    """
    INGEST                = "ingest"
    CONTEXT_ASSEMBLED     = "context_assembled"
    CHECKPOINT_PRESENTED  = "checkpoint_presented"
    CHECKPOINT_DECIDED    = "checkpoint_decided"
    CONSENT_GRANTED       = "consent_granted"
    CONSENT_REVOKED       = "consent_revoked"
    DECAY_APPLIED         = "decay_applied"
    ERASURE_REQUESTED     = "erasure_requested"   # GDPR Art.17
    ERASURE_COMPLETED     = "erasure_completed"   # GDPR Art.17
    CRISIS_DETECTED       = "crisis_detected"     # Absolute barrier fired
    PRINCIPLES_VERIFIED   = "principles_verified" # Boot-time check passed
    CANARY_PASSED         = "canary_passed"
    CANARY_FAILED         = "canary_failed"        # Engine halts on this
    SESSION_OPENED        = "session_opened"
    SESSION_CLOSED        = "session_closed"


# ── The movement: core scientific types ──────────────────────────────────────
#
# These types encode the scientific basis of the engine.
# They are not implementation details — they are the claims the system makes.


@dataclass(frozen=True)
class ThreeAxisWeight:
    """
    The three orthogonal axes of edge relevance.

    Independently derived from three fields — this is not a design choice,
    it is the scientific basis of the system:
      Distance   — information retrieval (BM25, cosine similarity literature)
      Complexity — graph theory (betweenness centrality, path length)
      Size       — cognitive science (working memory, Miller 1956)

    These three axes cannot be collapsed to one without losing the orthogonal
    information each carries. This is documented in principles.yaml under
    decay_scientific_basis and is a behavioral invariant verified at boot.

    All values are [0.0, 1.0]. The engine computes effective_weight from
    these using DecayParameters — this type stores the base measurements.
    """
    distance: float    # How far this fact is from the query concept.
    complexity: float  # How many inferential steps connect them.
    size: float        # Cognitive load — how much this fact demands of the reader.

    def __post_init__(self) -> None:
        for axis, value in [
            ("distance", self.distance),
            ("complexity", self.complexity),
            ("size", self.size),
        ]:
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"ThreeAxisWeight.{axis} must be in [0.0, 1.0], got {value}"
                )


@dataclass(frozen=True)
class DecayParameters:
    """
    The scientific constants governing power-law edge weight decay.

    These are not configuration values — they have specific scientific
    citations that justify each number. Overriding them requires
    documented justification in the domain module's manifest.

    Default values represent the general-purpose baseline.
    Domain modules may provide their own DecayParameters via ModuleManifest
    when their domain's empirical literature supports different calibration.
    """
    exponent: float = 0.5
    # Power-law exponent. effective_weight = base x (1 + days)^(-exponent)
    # Source: Wixted (2004), Jost's Law. Default 0.5 is empirically validated.

    sensitive_multiplier: float = 1.4
    # Multiplier on exponent for sensitive edges (PHI/PII/FINANCIAL/LEGAL).
    # Source: Nolen-Hoeksema (1991) on emotional memory salience.

    spacing_cap: float = 2.0
    # Maximum reinforcement bonus from spaced repetition.
    # factor = min(spacing_cap, 1.0 + days_since_last / 30.0)
    # Source: Cepeda et al. (2006) on optimal spacing intervals.

    prune_threshold: float = 0.05
    # Edges with effective_weight below this are removed from the graph.

    def __post_init__(self) -> None:
        if not 0.0 < self.exponent <= 5.0:
            raise ValueError(f"exponent must be in (0.0, 5.0], got {self.exponent}")
        if self.sensitive_multiplier < 1.0:
            raise ValueError(
                f"sensitive_multiplier must be >= 1.0, got {self.sensitive_multiplier}"
            )
        if self.spacing_cap < 1.0:
            raise ValueError(f"spacing_cap must be >= 1.0, got {self.spacing_cap}")
        if not 0.0 < self.prune_threshold < 1.0:
            raise ValueError(f"prune_threshold must be in (0.0, 1.0)")


# ── Module manifest — the complication interface ──────────────────────────────


@dataclass(frozen=True)
class ModuleManifest:
    """
    The formal contract between a domain module and the Verity engine.

    This is the standard attachment point — the lug width that every
    complication must conform to. A module that provides a valid
    ModuleManifest can be loaded by any Verity engine, anywhere,
    without the engine knowing anything about the domain.

    Module authors: implement this and register via setuptools entry_points:

        [project.entry-points."verity.modules"]
        fhir_r4 = "verity_fhir:manifest"

    The engine discovers modules at startup via entry_points discovery.
    No engine code changes required to add a new module.
    """
    module_id: ModuleId
    version: str
    display_name: str
    classifications: tuple[DataClassification, ...]
    entry_point: str
    # Fully qualified Python class path. Must satisfy ContextModule Protocol.
    # "verity_fhir:FHIRModule"

    decay_parameters: DecayParameters | None = None
    # None = use engine defaults. Must document empirical basis if overriding.

    shacl_shapes_path: str | None = None
    ontology_path: str | None = None
    prompt_template_path: str | None = None

    requires_consent_for: tuple[DataClassification, ...] = ()
    # Added to engine defaults — never subtracts from them.

    checkpoint_purposes: tuple[str, ...] = ()
    # Purpose strings that always require a GOVERN checkpoint in this domain.


# ── Consent ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ConsentRecord:
    """
    A record of consent granted by a subject for a specific purpose.

    The consent gate checks this before any graph traversal. Consent is
    per-purpose — "clinical_decision_support" does not authorize
    "research_deidentified". Enforced by canary tests at every boot.

    Stored in the consent Named Graph (urn:verity:consent) with OR-Set
    CRDT semantics — concurrent grants and revocations resolve correctly.
    """
    consent_ref: ConsentRef
    subject_id: EntityId
    granted_by: str
    granted_at: datetime
    purpose: str
    classifications: tuple[DataClassification, ...]
    audit_id: AuditRef

    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    revoked_by: str | None = None
    revocation_audit_id: AuditRef | None = None

    @property
    def is_active(self) -> bool:
        """True if granted, not revoked, and not expired."""
        now = datetime.now(timezone.utc)
        if self.revoked_at is not None:
            return False
        if self.expires_at is not None and now > self.expires_at:
            return False
        return True


# ── Core fact types ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TypedFact:
    """
    A single typed entity in the knowledge graph.

    The fundamental unit of knowledge in Verity. Unlike keyword-based
    systems (YAKE extracts "anxiety"), TypedFact carries the entity's
    ontological type, its classification, its provenance, and its trust score.

    domain_properties is the module's private store — the engine never
    reads it directly. The movement doesn't need to know the complication.
    """
    entity_id: EntityId
    entity_type: str
    classification: DataClassification
    trust_score: float
    provenance_ref: ProvenanceRef
    created_at: datetime
    source: str

    domain_properties: dict[str, Any] = field(default_factory=dict)
    domain_module: ModuleId | None = None
    external_id: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.trust_score <= 1.0:
            raise ValueError(
                f"trust_score must be in [0.0, 1.0], got {self.trust_score}"
            )


@dataclass(frozen=True)
class WeightedEdge:
    """
    A typed, three-axis weighted, decay-adjusted relationship between two facts.

    base_weight stores the original ThreeAxisWeight at ingestion — never lost.
    effective_weight is the pre-computed scalar after power-law decay — used
    for ranking. Both stored so decay history is fully auditable.
    """
    edge_id: str
    source_id: EntityId
    target_id: EntityId
    relationship_type: str

    base_weight: ThreeAxisWeight
    effective_weight: float

    last_reinforced: datetime
    reinforcement_count: int
    is_sensitive: bool
    classification: DataClassification
    created_at: datetime
    provenance_ref: ProvenanceRef

    def __post_init__(self) -> None:
        if not 0.0 <= self.effective_weight <= 1.0:
            raise ValueError(
                f"effective_weight must be in [0.0, 1.0], got {self.effective_weight}"
            )


# ── Request / response contracts ──────────────────────────────────────────────


@dataclass
class ContextRequest:
    """
    What the caller asks the engine to assemble. Mutable — built incrementally.
    The consent gate validates consent_ref before any traversal begins.
    """
    query: str
    purpose: str
    consent_ref: ConsentRef

    max_facts: int = 20
    min_weight: float = 0.1
    include_classifications: tuple[DataClassification, ...] = (
        DataClassification.PUBLIC,
        DataClassification.INTERNAL,
    )
    max_tokens: int | None = None
    domain_module: ModuleId | None = None
    session_id: SessionId | None = None


@dataclass(frozen=True)
class ExclusionNote:
    """
    Why a fact was considered but not included in a ContextBundle.
    ContextBundle.excluded is an empty tuple when nothing was excluded — never None.
    """
    entity_id: EntityId
    entity_type: str
    classification: DataClassification
    reason: str
    # "below_weight_threshold" | "consent_not_granted" | "purpose_mismatch"
    # "token_limit" | "classification_excluded"


@dataclass(frozen=True)
class ContextBundle:
    """
    The primary output of the Verity engine.

    Every field except checkpoint_context is mandatory and enforced
    at construction — not just documented. The agent_prompt field is
    the product: pre-assembled, domain-aware, uncertainty-annotated
    context ready for direct LLM injection.
    """
    facts: tuple[TypedFact, ...]
    edges: tuple[WeightedEdge, ...]

    uncertainty: float                    # [0.0, 1.0] — NEVER omitted
    completeness: Completeness
    excluded: tuple[ExclusionNote, ...]   # Empty tuple, not None
    reasoning_trace: tuple[str, ...]      # Every inference step — NEVER empty

    consent_ref: ConsentRef
    purpose: str
    assembled_at: datetime
    audit_id: AuditRef
    session_id: SessionId | None

    agent_prompt: str
    agent_prompt_tokens: int

    checkpoint_required: bool
    checkpoint_context: str | None

    schema_version: str = "2.0"

    def __post_init__(self) -> None:
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError(
                f"uncertainty must be [0.0, 1.0], got {self.uncertainty}"
            )
        if len(self.reasoning_trace) == 0:
            raise ValueError(
                "reasoning_trace must not be empty. "
                "A ContextBundle with no reasoning trace is non-compliant."
            )
        if self.checkpoint_required and self.checkpoint_context is None:
            raise ValueError(
                "checkpoint_context must be set when checkpoint_required is True."
            )


# ── GOVERN contracts ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProposedAction:
    """An action proposed for human review via the GOVERN checkpoint."""
    action_type: str
    affects: tuple[EntityId, ...]
    classification: DataClassification
    reversible: bool
    description: str
    proposed_by: str

    proposed_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    domain_properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CheckpointResult:
    """
    The outcome of a GOVERN checkpoint. Always written to the audit trail.
    VETOED is the default — timeout is a veto, not an approval.
    """
    decision: CheckpointDecision
    decided_by: str               # Human identifier, or "timeout"
    decided_at: datetime
    audit_id: AuditRef

    proposed_action: ProposedAction
    context_audit_id: AuditRef

    rationale: str | None = None
    modifications: dict[str, Any] = field(default_factory=dict)


# ── RELATE contracts ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RelateResult:
    """What RELATE returns after ingesting data into the knowledge graph."""
    facts_added: tuple[TypedFact, ...]
    edges_added: tuple[WeightedEdge, ...]
    facts_updated: tuple[TypedFact, ...]
    edges_updated: tuple[WeightedEdge, ...]

    crisis_detected: bool
    # When True: nothing written to graph. Crisis routing happened instead.

    audit_id: AuditRef
    session_id: SessionId | None
    concepts: tuple[str, ...]
    validation_passed: bool
    validation_errors: tuple[str, ...] = ()


# ── REMEMBER contracts ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuditEvent:
    """
    A single immutable record in the Merkle-chained audit trail.
    Append-only. Written once. Never modified. Never deleted.
    Any tampering breaks all subsequent hashes.
    """
    sequence: AuditRef
    event_type: AuditEventType
    timestamp: datetime

    actor: str
    session_id: SessionId | None
    consent_ref: ConsentRef | None

    payload: dict[str, Any]
    # Event-specific data. Never contains raw PHI/PII — references only.

    content_hash: str
    previous_hash: str | None     # None = genesis record
    chain_valid: bool


# ── Session contract ──────────────────────────────────────────────────────────


@dataclass
class SessionState:
    """
    The consent and audit boundary for a sequence of engine operations.
    Opened via engine.session(consent_ref=...) as an async context manager.
    """
    session_id: SessionId
    consent_ref: ConsentRef
    opened_at: datetime
    domain_module: ModuleId | None

    facts_ingested: int = 0
    contexts_assembled: int = 0
    checkpoints_presented: int = 0
    checkpoints_approved: int = 0
    checkpoints_vetoed: int = 0

    is_open: bool = True
    closed_at: datetime | None = None
    closing_audit_id: AuditRef | None = None


# ── Engine defaults ───────────────────────────────────────────────────────────

DEFAULT_DECAY_PARAMETERS = DecayParameters()
# The engine uses these when no domain module provides its own.
# Wixted (2004) / Jost's Law baseline. Do not change without citation.
