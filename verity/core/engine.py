"""
verity.core.engine
==================
The Verity engine. Four functions. One session boundary. One audit trail.

  RELATE   — ingest typed facts into the knowledge graph
  NAVIGATE — traverse the graph, assemble a ContextBundle
  GOVERN   — present a proposed action to a human checkpoint
  REMEMBER — append an immutable event to the Merkle chain

VERIFY is not a fifth function — it is the immune system that runs
inside every other function. SOMA is not a function — it is the
assembly step inside NAVIGATE that produces agent_prompt.

Usage:
    engine = await Engine.start()

    async with engine.session(consent_ref="consent:abc123") as s:
        await s.ingest(text, source="manual_entry")
        context = await s.context(
            query="recent observations",
            purpose="clinical_decision_support",
        )
        print(context.agent_prompt)

Design discipline:
  - Crisis barrier runs FIRST, before anything else in RELATE
  - Consent gate runs SECOND, before any graph traversal in NAVIGATE
  - GOVERN veto is the default — timeout = veto, not approval
  - Every operation produces an AuditEvent — no silent operations
  - The engine never knows which backend is running beneath it
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from verity.core.crisis import check_and_raise, get_crisis_resources
from verity.core.exceptions import (
    CanaryError,
    ConsentRequiredError,
    ConsentExpiredError,
    ConsentRevokedError,
    EngineNotStartedError,
    PurposeMismatchError,
    SessionClosedError,
)
from verity.core.graph_store import GraphStore
from verity.core.graph_store.registry import get_graph_store
from verity.core.principles import LoadedPrinciples, verify_principles
from verity.core.types import (
    AuditEvent,
    AuditEventType,
    AuditRef,
    CheckpointDecision,
    CheckpointResult,
    Completeness,
    ConsentRecord,
    ConsentRef,
    ContextBundle,
    ContextRequest,
    DataClassification,
    DecayParameters,
    EntityId,
    ExclusionNote,
    ModuleId,
    ModuleManifest,
    ProposedAction,
    RelateResult,
    SessionId,
    SessionState,
    ThreeAxisWeight,
    TrustSource,
    TypedFact,
    WeightedEdge,
    DEFAULT_DECAY_PARAMETERS,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# BFS traversal switches to Personalized PageRank above this node count
_PPR_THRESHOLD = 200

# Token estimation: conservative 4 chars per token
_CHARS_PER_TOKEN = 4

# Checkpoint timeout in seconds — veto fires after this
_CHECKPOINT_TIMEOUT_SECONDS = 300


# ── SOMA: agent_prompt assembly ───────────────────────────────────────────────

def _assemble_agent_prompt(
    facts: list[TypedFact],
    edges: list[WeightedEdge],
    excluded: list[ExclusionNote],
    uncertainty: float,
    completeness: Completeness,
    purpose: str,
    domain_module: ModuleId | None,
) -> str:
    """
    SOMA — assemble the agent_prompt from facts, edges, and metadata.

    Produces a structured, uncertainty-annotated context string ready
    for direct LLM injection. Domain modules may provide their own
    templates via prompt_template_path in ModuleManifest.

    This is the generic template — used when no domain module is active.
    """
    lines: list[str] = []

    lines.append(f"[CONTEXT — {purpose}]")
    lines.append(
        f"Based on {len(facts)} verified observation(s) "
        f"(uncertainty: {uncertainty:.0%}):"
    )
    lines.append("")

    if not facts:
        lines.append("No relevant facts found for this query and purpose.")
    else:
        for fact in facts:
            lines.append(
                f"• [{fact.entity_type}] {fact.entity_id} "
                f"(trust: {fact.trust_score:.2f} | "
                f"source: {fact.source} | "
                f"classification: {fact.classification})"
            )
            if fact.domain_properties:
                for k, v in list(fact.domain_properties.items())[:3]:
                    lines.append(f"    {k}: {v}")

    lines.append("")

    if excluded:
        exclusion_summary: dict[str, int] = {}
        for note in excluded:
            exclusion_summary[note.reason] = exclusion_summary.get(note.reason, 0) + 1
        parts = [f"{count} {reason}" for reason, count in exclusion_summary.items()]
        lines.append(f"[EXCLUDED: {' | '.join(parts)}]")

    lines.append(
        f"[UNCERTAINTY NOTE: This context is {completeness}. "
        f"Missing data ≠ absence of fact. Uncertainty: {uncertainty:.0%}]"
    )

    return "\n".join(lines)


def _estimate_tokens(text: str) -> int:
    """Conservative token estimate: len(text) / 4."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ── Session ───────────────────────────────────────────────────────────────────

class Session:
    """
    A consent-and-audit-bounded sequence of engine operations.

    Opened via engine.session(consent_ref=...) as an async context manager.
    All operations within the session share:
      - The active ConsentRecord
      - The session_id (linked across all audit events)
      - Automatic session-close audit event on __aexit__

    Do not instantiate directly. Use engine.session().
    """

    def __init__(
        self,
        engine: "Engine",
        consent_ref: ConsentRef,
        domain_module: ModuleId | None = None,
    ) -> None:
        self._engine = engine
        self._state = SessionState(
            session_id=f"session:{uuid.uuid4()}",
            consent_ref=consent_ref,
            opened_at=datetime.now(timezone.utc),
            domain_module=domain_module,
        )

    @property
    def session_id(self) -> SessionId:
        return self._state.session_id

    @property
    def state(self) -> SessionState:
        return self._state

    def _require_open(self) -> None:
        if not self._state.is_open:
            raise SessionClosedError(
                f"Session {self.session_id} is closed. Open a new session."
            )

    # ── RELATE ────────────────────────────────────────────────────────────────

    async def ingest(
        self,
        text: str,
        source: str = "manual_entry",
        classification: DataClassification = DataClassification.INTERNAL,
        trust_source: str = TrustSource.UNKNOWN,
        domain_properties: dict[str, Any] | None = None,
    ) -> RelateResult:
        """
        RELATE — ingest text or structured data into the knowledge graph.

        Crisis barrier runs first, unconditionally. If crisis is detected,
        CrisisBarrierError is raised and nothing is written to the graph.

        Args:
            text:               Raw text or stringified data to ingest
            source:             Where this data came from
            classification:     Sensitivity classification (default: INTERNAL)
            trust_source:       Origin of this data (affects trust_score)
            domain_properties:  Module-specific key-value pairs

        Returns:
            RelateResult — what was written, or crisis_detected=True

        Raises:
            CrisisBarrierError  — crisis content detected, nothing written
            SessionClosedError  — session is not open
        """
        self._require_open()
        return await self._engine.relate(
            text=text,
            source=source,
            classification=classification,
            trust_source=trust_source,
            domain_properties=domain_properties or {},
            session_id=self.session_id,
            consent_ref=self._state.consent_ref,
        )

    # ── NAVIGATE ──────────────────────────────────────────────────────────────

    async def context(
        self,
        query: str,
        purpose: str,
        max_facts: int = 20,
        min_weight: float = 0.1,
        include_classifications: tuple[DataClassification, ...] | None = None,
        max_tokens: int | None = None,
    ) -> ContextBundle:
        """
        NAVIGATE — assemble a ContextBundle for this query and purpose.

        Consent gate validates before any traversal begins.
        Returns a fully assembled ContextBundle with agent_prompt ready
        for LLM injection.

        Raises:
            ConsentRequiredError  — consent not valid for this purpose
            SessionClosedError    — session is not open
        """
        self._require_open()

        request = ContextRequest(
            query=query,
            purpose=purpose,
            consent_ref=self._state.consent_ref,
            max_facts=max_facts,
            min_weight=min_weight,
            include_classifications=include_classifications or (
                DataClassification.PUBLIC,
                DataClassification.INTERNAL,
            ),
            max_tokens=max_tokens,
            domain_module=self._state.domain_module,
            session_id=self.session_id,
        )

        bundle = await self._engine.navigate(request)
        self._state.contexts_assembled += 1
        return bundle

    # ── GOVERN ────────────────────────────────────────────────────────────────

    async def checkpoint(
        self,
        action: ProposedAction,
        context: ContextBundle,
        timeout_seconds: int = _CHECKPOINT_TIMEOUT_SECONDS,
    ) -> CheckpointResult:
        """
        GOVERN — present a proposed action to the human checkpoint.

        Veto is the default. If no response is received within
        timeout_seconds, the action is automatically vetoed.

        The decision and the context it was made in are recorded
        immutably in the audit trail regardless of outcome.

        Raises:
            SessionClosedError — session is not open
        """
        self._require_open()
        result = await self._engine.govern(
            action=action,
            context=context,
            session_id=self.session_id,
            timeout_seconds=timeout_seconds,
        )
        self._state.checkpoints_presented += 1
        if result.decision == CheckpointDecision.APPROVED:
            self._state.checkpoints_approved += 1
        else:
            self._state.checkpoints_vetoed += 1
        return result

    # ── Internal close ────────────────────────────────────────────────────────

    async def _close(self) -> None:
        """Record session close audit event and mark session closed."""
        if not self._state.is_open:
            return

        self._state.closed_at = datetime.now(timezone.utc)
        self._state.is_open = False

        audit_id = await self._engine.remember(AuditEvent(
            sequence=0,  # Assigned by append_audit
            event_type=AuditEventType.SESSION_CLOSED,
            timestamp=self._state.closed_at,
            actor="session",
            session_id=self.session_id,
            consent_ref=self._state.consent_ref,
            payload={
                "event":              "session_closed",
                "facts_ingested":     self._state.facts_ingested,
                "contexts_assembled": self._state.contexts_assembled,
                "checkpoints":        self._state.checkpoints_presented,
                "approved":           self._state.checkpoints_approved,
                "vetoed":             self._state.checkpoints_vetoed,
            },
            content_hash="",    # Computed by append_audit
            previous_hash=None,
            chain_valid=True,
        ))
        self._state.closing_audit_id = audit_id
        logger.info(
            f"Session closed | id={self.session_id} | "
            f"facts={self._state.facts_ingested} | "
            f"contexts={self._state.contexts_assembled}"
        )


# ── Engine ────────────────────────────────────────────────────────────────────

class Engine:
    """
    The Verity context engine.

    Entry point: Engine.start()
    Session entry point: engine.session(consent_ref=...)

    The engine:
      1. Verifies principles (cryptographic + behavioral canaries)
      2. Initializes the graph store backend
      3. Loads requested domain modules
      4. Records a PRINCIPLES_VERIFIED audit event
      5. Accepts sessions

    The engine does not know which graph store backend is running.
    The engine does not know the internals of any domain module.
    That is the Machine Test boundary.
    """

    def __init__(
        self,
        store: GraphStore,
        principles: LoadedPrinciples,
        decay_parameters: DecayParameters = DEFAULT_DECAY_PARAMETERS,
        modules: dict[ModuleId, ModuleManifest] | None = None,
    ) -> None:
        self._store = store
        self._principles = principles
        self._decay = decay_parameters
        self._modules: dict[ModuleId, ModuleManifest] = modules or {}
        self._started = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @classmethod
    async def start(
        cls,
        modules: list[str] | None = None,
        decay_parameters: DecayParameters = DEFAULT_DECAY_PARAMETERS,
    ) -> "Engine":
        """
        Start the Verity engine.

        1. Verifies principles.yaml (signature + canary tests)
        2. Initializes the graph store
        3. Loads domain modules
        4. Records PRINCIPLES_VERIFIED audit event

        Args:
            modules:           List of domain module IDs to load.
                               Example: ["fhir_r4", "gdpr"]
            decay_parameters:  Override decay constants (uses defaults if None)

        Returns:
            A started Engine ready to accept sessions.

        Raises:
            PrinciplesError — principles verification failed, engine will not start
            CanaryError     — behavioral canary failed, engine will not start
        """
        logger.info("Starting Verity engine...")

        # Step 1: Verify principles — halt if either check fails
        principles = verify_principles()

        # Step 2: Initialize graph store
        store = get_graph_store(decay_parameters=decay_parameters)
        await store.initialize()

        # Step 3: Load domain modules
        loaded_modules = cls._load_modules(modules or [])

        # Step 4: Build engine
        engine = cls(
            store=store,
            principles=principles,
            decay_parameters=decay_parameters,
            modules=loaded_modules,
        )
        engine._started = True

        # Step 5: Record principles verified event
        await engine.remember(AuditEvent(
            sequence=0,
            event_type=AuditEventType.PRINCIPLES_VERIFIED,
            timestamp=datetime.now(timezone.utc),
            actor="engine",
            session_id=None,
            consent_ref=None,
            payload={
                "principles_version": principles.version,
                "principles_sequence": principles.sequence,
                "content_hash": principles.content_hash,
                "modules_loaded": list(loaded_modules.keys()),
            },
            content_hash="",
            previous_hash=None,
            chain_valid=True,
        ))

        logger.info(
            f"Verity engine started | "
            f"modules={list(loaded_modules.keys()) or 'none'} | "
            f"backend={type(store).__name__}"
        )
        return engine

    @staticmethod
    def _load_modules(
        module_ids: list[str],
    ) -> dict[ModuleId, ModuleManifest]:
        """
        Load domain modules via setuptools entry points.

        Modules register themselves as:
            [project.entry-points."verity.modules"]
            fhir_r4 = "verity_fhir:manifest"

        Unknown module IDs are warned and skipped — not a fatal error.
        A missing module should surface as a warning, not halt the engine.
        """
        from importlib.metadata import entry_points

        loaded: dict[ModuleId, ModuleManifest] = {}
        available = {
            ep.name: ep
            for ep in entry_points(group="verity.modules")
        }

        for module_id in module_ids:
            if module_id not in available:
                logger.warning(
                    f"Domain module '{module_id}' not found. "
                    f"Install it: pip install verity-{module_id.replace('_', '-')}"
                )
                continue
            try:
                manifest = available[module_id].load()
                if not isinstance(manifest, ModuleManifest):
                    logger.error(
                        f"Module '{module_id}' entry point did not return "
                        f"a ModuleManifest. Skipping."
                    )
                    continue
                loaded[module_id] = manifest
                logger.info(f"Loaded module: {manifest.display_name} v{manifest.version}")
            except Exception as e:
                logger.error(f"Failed to load module '{module_id}': {e}")

        return loaded

    async def stop(self) -> None:
        """Cleanly shut down the engine and release backend resources."""
        await self._store.close()
        self._started = False
        logger.info("Verity engine stopped.")

    def _require_started(self) -> None:
        if not self._started:
            raise EngineNotStartedError(
                "Engine not started. Call await Engine.start() first."
            )

    # ── Session context manager ───────────────────────────────────────────────

    @asynccontextmanager
    async def session(
        self,
        consent_ref: ConsentRef,
        domain_module: ModuleId | None = None,
    ) -> AsyncIterator[Session]:
        """
        Open a session as an async context manager.

        Usage:
            async with engine.session(consent_ref="consent:abc123") as s:
                await s.ingest(text)
                context = await s.context(query="...", purpose="...")

        The session records a close event and commits all pending audit
        events when the context manager exits — whether normally or on
        exception. Veto any open checkpoints before closing.
        """
        self._require_started()
        s = Session(self, consent_ref=consent_ref, domain_module=domain_module)
        logger.info(
            f"Session opened | id={s.session_id} | consent={consent_ref}"
        )
        try:
            yield s
        finally:
            await s._close()

    # ── RELATE ────────────────────────────────────────────────────────────────

    async def relate(
        self,
        text: str,
        source: str,
        classification: DataClassification,
        trust_source: str,
        domain_properties: dict[str, Any],
        session_id: SessionId | None,
        consent_ref: ConsentRef,
    ) -> RelateResult:
        """
        RELATE — ingest data into the knowledge graph.

        Order of operations (non-negotiable):
          1. Crisis barrier — absolute, runs first, no exceptions
          2. Extract entities (domain module recognizer or YAKE fallback)
          3. Build TypedFacts and WeightedEdges
          4. Write to graph
          5. REMEMBER audit event

        Returns RelateResult with crisis_detected=True if the barrier fired.
        CrisisBarrierError is re-raised after the audit event is written.
        """
        self._require_started()

        # ── Step 1: Crisis barrier — FIRST, ALWAYS ────────────────────────────
        try:
            check_and_raise(text=text, actor=source, session_id=session_id)
        except Exception as crisis_exc:
            # Record the crisis event before re-raising
            await self._store.append_audit(AuditEvent(
                sequence=0,
                event_type=AuditEventType.CRISIS_DETECTED,
                timestamp=datetime.now(timezone.utc),
                actor=source,
                session_id=session_id,
                consent_ref=consent_ref,
                payload={
                    "crisis_detected": True,
                    "source": source,
                    "resources_offered": [
                        r.name for r in get_crisis_resources()
                    ],
                },
                content_hash="",
                previous_hash=None,
                chain_valid=True,
            ))
            raise  # Always re-raise — never swallow a crisis barrier

        # ── Step 2: Extract entities ──────────────────────────────────────────
        trust_score = TrustSource.SCORES.get(trust_source, 0.20)
        facts, edges = self._extract_entities(
            text=text,
            source=source,
            classification=classification,
            trust_score=trust_score,
            domain_properties=domain_properties,
            session_id=session_id,
        )

        # ── Step 3 & 4: Write to graph ────────────────────────────────────────
        facts_added: list[TypedFact] = []
        facts_updated: list[TypedFact] = []
        edges_added: list[WeightedEdge] = []
        edges_updated: list[WeightedEdge] = []

        for fact in facts:
            is_new = await self._store.write_fact(fact, session_id=session_id)
            if is_new:
                facts_added.append(fact)
            else:
                facts_updated.append(fact)

        for edge in edges:
            is_new = await self._store.write_edge(edge, session_id=session_id)
            if is_new:
                edges_added.append(edge)
            else:
                edges_updated.append(edge)

        # ── Step 5: REMEMBER ──────────────────────────────────────────────────
        audit_id = await self._store.append_audit(AuditEvent(
            sequence=0,
            event_type=AuditEventType.INGEST,
            timestamp=datetime.now(timezone.utc),
            actor=source,
            session_id=session_id,
            consent_ref=consent_ref,
            payload={
                "facts_added":    len(facts_added),
                "facts_updated":  len(facts_updated),
                "edges_added":    len(edges_added),
                "edges_updated":  len(edges_updated),
                "source":         source,
                "classification": classification,
            },
            content_hash="",
            previous_hash=None,
            chain_valid=True,
        ))

        return RelateResult(
            facts_added=tuple(facts_added),
            edges_added=tuple(edges_added),
            facts_updated=tuple(facts_updated),
            edges_updated=tuple(edges_updated),
            crisis_detected=False,
            audit_id=audit_id,
            session_id=session_id,
            concepts=tuple(f.entity_id for f in facts_added),
            validation_passed=True,
        )

    def _extract_entities(
        self,
        text: str,
        source: str,
        classification: DataClassification,
        trust_score: float,
        domain_properties: dict[str, Any],
        session_id: SessionId | None,
    ) -> tuple[list[TypedFact], list[WeightedEdge]]:
        """
        Extract typed entities from text.

        Uses the active domain module's recognizer if available.
        Falls back to YAKE keyword extraction for personal tier.
        Wraps YAKE keywords as TypedFacts with entity_type="verity:Keyword".
        """
        # Domain module recognizer (future: loaded from manifest entry_point)
        # For now: YAKE fallback for all inputs
        return self._yake_fallback(
            text=text,
            source=source,
            classification=classification,
            trust_score=trust_score,
            domain_properties=domain_properties,
        )

    def _yake_fallback(
        self,
        text: str,
        source: str,
        classification: DataClassification,
        trust_score: float,
        domain_properties: dict[str, Any],
    ) -> tuple[list[TypedFact], list[WeightedEdge]]:
        """
        YAKE keyword extraction fallback for personal tier.

        Extracts keywords from text and wraps them as TypedFacts
        with entity_type="verity:Keyword". Edges are created between
        co-occurring keywords with equal ThreeAxisWeight.
        """
        try:
            import yake
            kw_extractor = yake.KeywordExtractor(
                lan="en", n=2, dedupLim=0.7, top=10
            )
            keywords = kw_extractor.extract_keywords(text)
        except ImportError:
            logger.warning("YAKE not available — falling back to simple word split")
            words = [w.strip(".,!?;:") for w in text.split() if len(w) > 4]
            keywords = [(w.lower(), 0.5) for w in words[:10]]
        except Exception as e:
            logger.warning(f"YAKE extraction failed: {e}")
            keywords = []

        now = datetime.now(timezone.utc)
        prov_ref = f"prov:{uuid.uuid4()}"
        facts: list[TypedFact] = []
        edges: list[WeightedEdge] = []

        for keyword, score in keywords:
            # YAKE score is lower = more important; invert for trust
            keyword_trust = max(0.1, min(0.9, 1.0 - score)) * trust_score
            entity_id = f"kw:{keyword.replace(' ', '_')}"

            fact = TypedFact(
                entity_id=entity_id,
                entity_type="verity:Keyword",
                classification=classification,
                trust_score=keyword_trust,
                provenance_ref=prov_ref,
                created_at=now,
                source=source,
                domain_properties=domain_properties,
                domain_module=None,
            )
            facts.append(fact)

        # Create co-occurrence edges between consecutive keywords
        for i in range(len(facts) - 1):
            a, b = facts[i], facts[i + 1]
            edge_id = f"edge:{a.entity_id}:{b.entity_id}"
            base_weight = ThreeAxisWeight(
                distance=0.8,    # Adjacent keywords are close
                complexity=0.5,  # Moderate inferential distance
                size=0.3,        # Low cognitive load
            )
            edge = WeightedEdge(
                edge_id=edge_id,
                source_id=a.entity_id,
                target_id=b.entity_id,
                relationship_type="verity:coOccurs",
                base_weight=base_weight,
                effective_weight=(base_weight.distance +
                                  base_weight.complexity +
                                  base_weight.size) / 3.0,
                last_reinforced=now,
                reinforcement_count=1,
                is_sensitive=classification in (
                    DataClassification.PHI,
                    DataClassification.PII,
                    DataClassification.FINANCIAL,
                    DataClassification.LEGAL,
                ),
                classification=classification,
                created_at=now,
                provenance_ref=prov_ref,
            )
            edges.append(edge)

        return facts, edges

    # ── NAVIGATE ──────────────────────────────────────────────────────────────

    async def navigate(self, request: ContextRequest) -> ContextBundle:
        """
        NAVIGATE — traverse the graph and assemble a ContextBundle.

        Order of operations:
          1. Consent gate — validate consent before any traversal
          2. Seed search — find facts matching the query
          3. BFS traversal (or PPR if graph is large)
          4. Rank, filter, and assemble
          5. SOMA — assemble agent_prompt
          6. REMEMBER audit event

        Raises:
            ConsentRequiredError — consent not valid for this purpose
        """
        self._require_started()

        # ── Step 1: Consent gate ──────────────────────────────────────────────
        consent = await self._validate_consent(request)

        reasoning_trace: list[str] = [
            f"Consent validated: {request.consent_ref} for purpose '{request.purpose}'",
        ]

        # ── Step 2: Seed search ───────────────────────────────────────────────
        seed_facts = await self._store.search_facts(request.query, request)
        reasoning_trace.append(
            f"Seed search '{request.query}': {len(seed_facts)} seed fact(s) found"
        )

        # ── Step 3: BFS traversal ─────────────────────────────────────────────
        store_stats = await self._store.stats()
        node_count = store_stats.get("facts", 0)

        all_facts: list[TypedFact] = list(seed_facts)
        all_edges: list[WeightedEdge] = []

        if seed_facts:
            if node_count >= _PPR_THRESHOLD:
                reasoning_trace.append(
                    f"Graph size {node_count} ≥ {_PPR_THRESHOLD}: "
                    f"PPR traversal not yet implemented, falling back to BFS"
                )
            traversed_facts, traversed_edges = await self._bfs_traverse(
                seed_facts=seed_facts,
                request=request,
                reasoning_trace=reasoning_trace,
            )
            # Deduplicate facts by entity_id, keeping the one with higher trust_score
            merged: dict[str, TypedFact] = {}
            for f in [*seed_facts, *traversed_facts]:
                if f.entity_id not in merged or f.trust_score > merged[f.entity_id].trust_score:
                    merged[f.entity_id] = f
            duplicates = len(seed_facts) + len(traversed_facts) - len(merged)
            if duplicates:
                reasoning_trace.append(
                    f"Deduplication: {duplicates} duplicate fact(s) merged "
                    f"(kept higher trust_score)"
                )
            all_facts = list(merged.values())
            all_edges = traversed_edges

        # ── Step 4: Rank, filter, assemble ───────────────────────────────────
        included_facts, included_edges, excluded_notes, reasoning_trace = self._rank_and_filter(
            facts=all_facts,
            edges=all_edges,
            request=request,
            reasoning_trace=reasoning_trace,
        )

        # Truncate to max_facts
        if len(included_facts) > request.max_facts:
            for excess in included_facts[request.max_facts:]:
                excluded_notes.append(ExclusionNote(
                    entity_id=excess.entity_id,
                    entity_type=excess.entity_type,
                    classification=excess.classification,
                    reason="token_limit",
                ))
            included_facts = included_facts[:request.max_facts]
            reasoning_trace.append(
                f"Truncated to {request.max_facts} facts (max_facts limit)"
            )

        # ── Step 5: SOMA — assemble agent_prompt ──────────────────────────────
        uncertainty = self._calculate_uncertainty(
            facts=included_facts,
            excluded=excluded_notes,
            total_found=len(all_facts),
        )
        completeness = self._calculate_completeness(
            included=len(included_facts),
            excluded=len(excluded_notes),
            total=len(all_facts),
        )

        reasoning_trace.append(
            f"Assembled {len(included_facts)} fact(s) | "
            f"excluded={len(excluded_notes)} | "
            f"uncertainty={uncertainty:.2f} | "
            f"completeness={completeness}"
        )

        agent_prompt = _assemble_agent_prompt(
            facts=included_facts,
            edges=included_edges,
            excluded=excluded_notes,
            uncertainty=uncertainty,
            completeness=completeness,
            purpose=request.purpose,
            domain_module=request.domain_module,
        )
        agent_prompt_tokens = _estimate_tokens(agent_prompt)

        # Token limit truncation — remove lowest-trust facts until prompt fits,
        # recording each dropped fact in excluded so the caller knows what was cut
        if request.max_tokens and agent_prompt_tokens > request.max_tokens:
            while agent_prompt_tokens > request.max_tokens and included_facts:
                removed = included_facts.pop()  # facts are sorted desc by trust; pop removes the lowest
                excluded_notes.append(ExclusionNote(
                    entity_id=removed.entity_id,
                    entity_type=removed.entity_type,
                    classification=removed.classification,
                    reason="token_limit",
                ))
                agent_prompt = _assemble_agent_prompt(
                    facts=included_facts,
                    edges=included_edges,
                    excluded=excluded_notes,
                    uncertainty=uncertainty,
                    completeness=completeness,
                    purpose=request.purpose,
                    domain_module=request.domain_module,
                )
                agent_prompt_tokens = _estimate_tokens(agent_prompt)
            reasoning_trace.append(
                f"Token limit ({request.max_tokens}): "
                f"{len(included_facts)} fact(s) kept, "
                f"{sum(1 for n in excluded_notes if n.reason == 'token_limit')} dropped"
            )

        # Checkpoint required?
        checkpoint_required, checkpoint_context = self._assess_checkpoint(
            facts=included_facts,
            uncertainty=uncertainty,
            completeness=completeness,
            request=request,
        )

        # ── Step 6: REMEMBER ──────────────────────────────────────────────────
        audit_id = await self._store.append_audit(AuditEvent(
            sequence=0,
            event_type=AuditEventType.CONTEXT_ASSEMBLED,
            timestamp=datetime.now(timezone.utc),
            actor="engine.navigate",
            session_id=request.session_id,
            consent_ref=request.consent_ref,
            payload={
                "query":            request.query,
                "purpose":          request.purpose,
                "facts_included":   len(included_facts),
                "facts_excluded":   len(excluded_notes),
                "uncertainty":      uncertainty,
                "completeness":     completeness,
                "checkpoint_required": checkpoint_required,
            },
            content_hash="",
            previous_hash=None,
            chain_valid=True,
        ))

        return ContextBundle(
            facts=tuple(included_facts),
            edges=tuple(included_edges),
            uncertainty=uncertainty,
            completeness=completeness,
            excluded=tuple(excluded_notes),
            reasoning_trace=tuple(reasoning_trace),
            consent_ref=request.consent_ref,
            purpose=request.purpose,
            assembled_at=datetime.now(timezone.utc),
            audit_id=audit_id,
            session_id=request.session_id,
            agent_prompt=agent_prompt,
            agent_prompt_tokens=agent_prompt_tokens,
            checkpoint_required=checkpoint_required,
            checkpoint_context=checkpoint_context,
        )

    async def _validate_consent(self, request: ContextRequest) -> ConsentRecord:
        """
        Validate the consent record for this request.

        Raises ConsentRequiredError subclasses on failure.
        Never silently permits access — always raises on invalid consent.
        """
        consent = await self._store.get_consent(request.consent_ref)

        if consent is None:
            raise ConsentRequiredError(
                operation="context_query",
                consent_ref=request.consent_ref,
                purpose=request.purpose,
            )

        if consent.revoked_at is not None:
            raise ConsentRevokedError(
                operation="context_query",
                consent_ref=request.consent_ref,
                purpose=request.purpose,
            )

        now = datetime.now(timezone.utc)
        if consent.expires_at and now > consent.expires_at:
            raise ConsentExpiredError(
                operation="context_query",
                consent_ref=request.consent_ref,
                purpose=request.purpose,
            )

        if consent.purpose != request.purpose:
            raise PurposeMismatchError(
                requested_purpose=request.purpose,
                consented_purpose=consent.purpose,
                consent_ref=request.consent_ref,
            )

        return consent

    async def _bfs_traverse(
        self,
        seed_facts: list[TypedFact],
        request: ContextRequest,
        reasoning_trace: list[str],
    ) -> tuple[list[TypedFact], list[WeightedEdge]]:
        """
        Breadth-first traversal from seed facts.
        Follows edges above min_weight, up to max_facts depth.
        """
        visited: set[EntityId] = {f.entity_id for f in seed_facts}
        queue: list[EntityId] = [f.entity_id for f in seed_facts]
        all_facts: list[TypedFact] = []
        all_edges: list[WeightedEdge] = []
        depth = 0
        max_depth = 3  # Configurable in future via ContextRequest

        while queue and len(all_facts) < request.max_facts:
            next_queue: list[EntityId] = []
            depth += 1

            for entity_id in queue:
                edges = await self._store.get_edges(
                    entity_id, min_weight=request.min_weight
                )
                for edge in edges:
                    all_edges.append(edge)
                    neighbor_id = (
                        edge.target_id
                        if edge.source_id == entity_id
                        else edge.source_id
                    )
                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        neighbor = await self._store.get_fact(neighbor_id)
                        if neighbor:
                            # Classification filter
                            if neighbor.classification in request.include_classifications:
                                all_facts.append(neighbor)
                                next_queue.append(neighbor_id)

            reasoning_trace.append(
                f"BFS depth {depth}: {len(next_queue)} new fact(s) discovered"
            )
            queue = next_queue
            if depth >= max_depth:
                break

        return all_facts, all_edges

    def _rank_and_filter(
        self,
        facts: list[TypedFact],
        edges: list[WeightedEdge],
        request: ContextRequest,
        reasoning_trace: list[str],
    ) -> tuple[list[TypedFact], list[WeightedEdge], list[ExclusionNote], list[str]]:
        """
        Rank facts by trust_score and filter by classification/weight.
        Also filters edges to only those connecting included facts.
        Returns (included_facts, included_edges, excluded, updated_reasoning_trace).
        """
        included: list[TypedFact] = []
        excluded: list[ExclusionNote] = []

        # Build edge weight lookup for ranking
        edge_weights: dict[EntityId, float] = {}
        for edge in edges:
            edge_weights[edge.source_id] = max(
                edge_weights.get(edge.source_id, 0.0), edge.effective_weight
            )
            edge_weights[edge.target_id] = max(
                edge_weights.get(edge.target_id, 0.0), edge.effective_weight
            )

        for fact in facts:
            # Classification filter
            if fact.classification not in request.include_classifications:
                excluded.append(ExclusionNote(
                    entity_id=fact.entity_id,
                    entity_type=fact.entity_type,
                    classification=fact.classification,
                    reason="classification_excluded",
                ))
                reasoning_trace.append(
                    f"Excluded {fact.entity_id}: classification {fact.classification} "
                    f"not in include_classifications"
                )
                continue

            # Weight filter
            effective_weight = edge_weights.get(fact.entity_id, fact.trust_score)
            if effective_weight < request.min_weight:
                excluded.append(ExclusionNote(
                    entity_id=fact.entity_id,
                    entity_type=fact.entity_type,
                    classification=fact.classification,
                    reason="below_weight_threshold",
                ))
                continue

            included.append(fact)

        # Rank by trust_score descending
        included.sort(key=lambda f: f.trust_score, reverse=True)

        # Filter edges: only keep edges where both endpoints are in included facts
        # and the edge itself passes classification and weight thresholds
        included_ids = {f.entity_id for f in included}
        included_edges = [
            e for e in edges
            if e.source_id in included_ids
            and e.target_id in included_ids
            and e.classification in request.include_classifications
            and e.effective_weight >= request.min_weight
        ]

        return included, included_edges, excluded, reasoning_trace

    def _calculate_uncertainty(
        self,
        facts: list[TypedFact],
        excluded: list[ExclusionNote],
        total_found: int,
    ) -> float:
        """
        Calculate uncertainty for the assembled context.

        Factors:
          - Average trust_score of included facts (lower = more uncertain)
          - Exclusion ratio (more exclusions = more uncertain)
          - Empty result (maximum uncertainty)
        """
        if not facts:
            return 1.0  # No facts = maximum uncertainty

        avg_trust = sum(f.trust_score for f in facts) / len(facts)
        trust_uncertainty = 1.0 - avg_trust

        total = len(facts) + len(excluded)
        exclusion_ratio = len(excluded) / total if total > 0 else 0.0

        # Weighted combination: trust (60%) + exclusions (40%)
        uncertainty = (trust_uncertainty * 0.6) + (exclusion_ratio * 0.4)
        return round(max(0.0, min(1.0, uncertainty)), 4)

    def _calculate_completeness(
        self,
        included: int,
        excluded: int,
        total: int,
    ) -> Completeness:
        """Map included/excluded/total counts to a Completeness value."""
        if total == 0:
            return Completeness.EMPTY
        if included == 0:
            return Completeness.EMPTY
        ratio = included / total
        if ratio >= 0.9:
            return Completeness.SATURATED
        if ratio >= 0.6:
            return Completeness.SUFFICIENT
        return Completeness.PARTIAL

    def _assess_checkpoint(
        self,
        facts: list[TypedFact],
        uncertainty: float,
        completeness: Completeness,
        request: ContextRequest,
    ) -> tuple[bool, str | None]:
        """
        Determine if a GOVERN checkpoint is required before acting.

        Triggers when:
          - Any included fact carries PHI/PII/FINANCIAL/LEGAL classification
          - Uncertainty > 0.7
          - Completeness is EMPTY or PARTIAL
          - Active domain module specifies this purpose requires checkpoint
        """
        reasons: list[str] = []

        sensitive_classifications = {
            DataClassification.PHI,
            DataClassification.PII,
            DataClassification.FINANCIAL,
            DataClassification.LEGAL,
        }
        if any(f.classification in sensitive_classifications for f in facts):
            reasons.append("context includes sensitive classified data")

        if uncertainty > 0.7:
            reasons.append(f"high uncertainty ({uncertainty:.0%})")

        if completeness in (Completeness.EMPTY, Completeness.PARTIAL):
            reasons.append(f"context completeness is {completeness}")

        # Domain module checkpoint purposes
        if request.domain_module and request.domain_module in self._modules:
            manifest = self._modules[request.domain_module]
            if request.purpose in manifest.checkpoint_purposes:
                reasons.append(
                    f"domain module '{request.domain_module}' "
                    f"requires checkpoint for purpose '{request.purpose}'"
                )

        if reasons:
            return True, "Checkpoint required: " + "; ".join(reasons)
        return False, None

    # ── GOVERN ────────────────────────────────────────────────────────────────

    async def govern(
        self,
        action: ProposedAction,
        context: ContextBundle,
        session_id: SessionId | None = None,
        timeout_seconds: int = _CHECKPOINT_TIMEOUT_SECONDS,
    ) -> CheckpointResult:
        """
        GOVERN — present a proposed action to the human checkpoint.

        Presents the action and context to the human interface.
        Waits up to timeout_seconds for a response.
        VETOED is the default — timeout = veto.

        Both the presentation and the decision are recorded in the audit trail.
        """
        self._require_started()
        now = datetime.now(timezone.utc)

        # Record: checkpoint presented
        presentation_audit_id = await self._store.append_audit(AuditEvent(
            sequence=0,
            event_type=AuditEventType.CHECKPOINT_PRESENTED,
            timestamp=now,
            actor=action.proposed_by,
            session_id=session_id,
            consent_ref=context.consent_ref,
            payload={
                "action_type":       action.action_type,
                "affects":           list(action.affects),
                "classification":    action.classification,
                "reversible":        action.reversible,
                "description":       action.description,
                "context_audit_id":  context.audit_id,
                "uncertainty":       context.uncertainty,
                "completeness":      context.completeness,
            },
            content_hash="",
            previous_hash=None,
            chain_valid=True,
        ))

        # Present to human and await response
        # Current implementation: automation bias warning + console input
        # Future: pluggable UI adapter via ModuleManifest
        decision, decided_by, rationale = await self._await_checkpoint_response(
            action=action,
            context=context,
            timeout_seconds=timeout_seconds,
        )

        decided_at = datetime.now(timezone.utc)

        # Record: decision made
        decision_audit_id = await self._store.append_audit(AuditEvent(
            sequence=0,
            event_type=AuditEventType.CHECKPOINT_DECIDED,
            timestamp=decided_at,
            actor=decided_by,
            session_id=session_id,
            consent_ref=context.consent_ref,
            payload={
                "decision":             decision,
                "decided_by":           decided_by,
                "rationale":            rationale,
                "presentation_audit_id": presentation_audit_id,
                "context_audit_id":     context.audit_id,
            },
            content_hash="",
            previous_hash=None,
            chain_valid=True,
        ))

        return CheckpointResult(
            decision=decision,
            decided_by=decided_by,
            decided_at=decided_at,
            audit_id=decision_audit_id,
            proposed_action=action,
            context_audit_id=context.audit_id,
            rationale=rationale,
        )

    async def _await_checkpoint_response(
        self,
        action: ProposedAction,
        context: ContextBundle,
        timeout_seconds: int,
    ) -> tuple[CheckpointDecision, str, str | None]:
        """
        Display the checkpoint to a human and await their decision.

        Current implementation: stdout display + asyncio input with timeout.
        VETOED if timeout elapses. Automation bias warning always displayed.

        Future: pluggable adapter for web UI, mobile, Slack, etc.
        """
        print("\n" + "═" * 60)
        print("⚠️  VERITY CHECKPOINT — HUMAN REVIEW REQUIRED")
        print("═" * 60)
        print(f"Action:       {action.action_type}")
        print(f"Description:  {action.description}")
        print(f"Affects:      {', '.join(action.affects)}")
        print(f"Reversible:   {'Yes' if action.reversible else 'NO — IRREVERSIBLE'}")
        print(f"Classification: {action.classification}")
        print(f"Uncertainty:  {context.uncertainty:.0%}")
        print(f"Completeness: {context.completeness}")
        print()
        print("⚠️  AUTOMATION BIAS WARNING: AI systems can be confidently wrong.")
        print("   Review the context carefully before approving.")
        print()
        print(f"You have {timeout_seconds} seconds to respond.")
        print("Options: [A]pprove  [V]eto  (default: Veto)")
        print("═" * 60)

        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, input, "Decision: "),
                timeout=float(timeout_seconds),
            )
            response = response.strip().upper()

            if response.startswith("A"):
                return CheckpointDecision.APPROVED, "human", None
            else:
                return CheckpointDecision.VETOED, "human", "Human chose not to proceed."

        except asyncio.TimeoutError:
            print(f"\n⏱  Timeout after {timeout_seconds}s — action VETOED by default.")
            return CheckpointDecision.VETOED, "timeout", "No response within timeout."

    # ── REMEMBER ──────────────────────────────────────────────────────────────

    async def remember(self, event: AuditEvent) -> AuditRef:
        """
        REMEMBER — append an event to the immutable Merkle-chained audit trail.

        This is the only write path to the provenance graph.
        The sequence number is assigned by the store — callers do not
        control it. The chain is extended atomically.

        Returns the AuditRef (sequence number) assigned to this event.
        """
        self._require_started()
        return await self._store.append_audit(event)

    # ── Maintenance ───────────────────────────────────────────────────────────

    async def apply_decay(self) -> dict[str, int]:
        """
        Run the power-law decay cycle across all edges.

        Should be run on a schedule (e.g. daily).
        Records a DECAY_APPLIED audit event.
        """
        self._require_started()
        result = await self._store.apply_decay()

        await self._store.append_audit(AuditEvent(
            sequence=0,
            event_type=AuditEventType.DECAY_APPLIED,
            timestamp=datetime.now(timezone.utc),
            actor="engine.decay",
            session_id=None,
            consent_ref=None,
            payload=result,
            content_hash="",
            previous_hash=None,
            chain_valid=True,
        ))

        return result

    async def stats(self) -> dict[str, Any]:
        """Return engine and backend statistics."""
        self._require_started()
        store_stats = await self._store.stats()
        return {
            **store_stats,
            "modules": list(self._modules.keys()),
            "principles_version": self._principles.version,
            "principles_sequence": self._principles.sequence,
        }
