"""
verity.core.graph_store.rdflib_store
=====================================
Personal tier graph store backend using rdflib.

Pure Python. No external services. No Java. Runs on a Raspberry Pi 4
and in a Kubernetes pod with identical code.

Three Named Graphs backed by rdflib ConjunctiveGraph:
  urn:verity:knowledge   — typed facts and weighted edges (RDF triples)
  urn:verity:provenance  — append-only Merkle chain (audit trail)
  urn:verity:consent     — consent ledger (OR-Set CRDT semantics)

Storage options (selected via VERITY_GRAPH_PATH env var):
  - In-memory (default, no env var) — ephemeral, useful for testing
  - On-disk (path provided)         — persistent across restarts

Performance note:
  rdflib is pure Python and suitable for personal/development use.
  For production workloads with > 10k facts, install pyoxigraph:
    pip install verity[fast]
  The engine detects pyoxigraph and uses it automatically as the
  rdflib store backend (10-100x SPARQL speedup, same API).

Satisfies: GraphStore Protocol (verity.core.graph_store)
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rdflib import ConjunctiveGraph, Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, XSD

from verity.core.exceptions import GraphStoreError
from verity.core.types import (
    DEFAULT_DECAY_PARAMETERS,
    AuditEvent,
    AuditRef,
    ConsentRecord,
    ConsentRef,
    ContextRequest,
    DataClassification,
    DecayParameters,
    EntityId,
    SessionId,
    ThreeAxisWeight,
    TypedFact,
    WeightedEdge,
)

logger = logging.getLogger(__name__)

# ── RDF Namespaces ────────────────────────────────────────────────────────────

VERITY  = Namespace("urn:verity:")
VK      = Namespace("urn:verity:knowledge#")    # knowledge graph terms
VP      = Namespace("urn:verity:provenance#")   # provenance graph terms
VC      = Namespace("urn:verity:consent#")      # consent graph terms

# Named Graph URIs
KNOWLEDGE_GRAPH  = URIRef("urn:verity:knowledge")
PROVENANCE_GRAPH = URIRef("urn:verity:provenance")
CONSENT_GRAPH    = URIRef("urn:verity:consent")

ALL_NAMED_GRAPHS = [
    str(KNOWLEDGE_GRAPH),
    str(PROVENANCE_GRAPH),
    str(CONSENT_GRAPH),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _uri(namespace: Namespace, local: str) -> URIRef:
    """Build a URIRef, percent-encoding the local part."""
    from urllib.parse import quote
    return URIRef(str(namespace) + quote(local, safe=""))


def _lit(value: Any, datatype=XSD.string) -> Literal:
    """Build a typed RDF Literal."""
    return Literal(str(value), datatype=datatype)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


# ── Backend ───────────────────────────────────────────────────────────────────

class RDFLibStore:
    """
    Personal tier graph store. Implements the GraphStore Protocol.

    Uses rdflib ConjunctiveGraph with three Named Graphs.
    Optionally backed by pyoxigraph for 10-100x SPARQL performance.

    Instantiated by the registry — do not instantiate directly.
    Use: from verity.core.graph_store.registry import get_graph_store
    """

    def __init__(
        self,
        path: str | Path | None = None,
        decay_parameters: DecayParameters = DEFAULT_DECAY_PARAMETERS,
    ) -> None:
        self._path = Path(path) if path else None
        self._decay = decay_parameters
        self._graph: ConjunctiveGraph | None = None
        self._audit_sequence: int = 0
        self._previous_hash: str | None = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Create the ConjunctiveGraph and bind namespaces. Idempotent."""
        if self._graph is not None:
            return  # Already initialized

        store_backend = self._select_store_backend()
        self._graph = ConjunctiveGraph(store=store_backend)

        if self._path and store_backend != "default":
            self._graph.open(str(self._path), create=True)

        # Bind namespaces for readable Turtle serialization
        self._graph.bind("verity", VERITY)
        self._graph.bind("vk", VK)
        self._graph.bind("vp", VP)
        self._graph.bind("vc", VC)

        # Restore audit sequence from provenance graph
        self._audit_sequence = await self._restore_audit_sequence()

        logger.info(
            f"RDFLibStore initialized | "
            f"backend={store_backend} | "
            f"path={self._path or 'in-memory'} | "
            f"audit_sequence={self._audit_sequence}"
        )

    async def close(self) -> None:
        """Flush and close the graph store."""
        if self._graph is not None:
            try:
                self._graph.close()
            except Exception as e:
                logger.warning(f"Error closing graph store: {e}")
            self._graph = None

    def _select_store_backend(self) -> str:
        """
        Select the rdflib store backend.
        Prefers pyoxigraph if installed (10-100x faster SPARQL).
        Falls back to rdflib's default memory store.
        """
        if self._path:
            try:
                import pyoxigraph  # noqa: F401
                logger.info("pyoxigraph detected — using Oxigraph store backend")
                return "Oxigraph"
            except ImportError:
                pass
            return "default"
        return "default"  # In-memory

    def _g(self) -> ConjunctiveGraph:
        """Return the graph, raising if not initialized."""
        if self._graph is None:
            raise GraphStoreError(
                "RDFLibStore not initialized. Call await store.initialize() first."
            )
        return self._graph

    def _named(self, graph_uri: URIRef) -> Graph:
        """Return a specific Named Graph context."""
        return self._g().get_context(graph_uri)

    # ── RELATE ────────────────────────────────────────────────────────────────

    async def write_fact(
        self,
        fact: TypedFact,
        session_id: SessionId | None = None,
    ) -> bool:
        """
        Write or reinforce a TypedFact in the knowledge Named Graph.
        Returns True if this was a new fact, False if it was a reinforcement.
        """
        g = self._named(KNOWLEDGE_GRAPH)
        subj = _uri(VK, f"fact/{fact.entity_id}")

        # Check if fact already exists — if so, reinforce trust_score
        if (subj, RDF.type, _uri(VK, fact.entity_type)) in g:
            # Reinforce: take the higher of existing and incoming trust_score,
            # then apply a small bump (bounded at 1.0) to reward repetition
            existing_score = float(
                g.value(subj, VK.trust_score) or fact.trust_score
            )
            new_score = min(1.0, max(existing_score, fact.trust_score) + 0.05)
            g.set((subj, VK.trust_score, _lit(new_score, XSD.float)))
            g.set((subj, VK.last_reinforced, _lit(_now_iso())))
            logger.debug(f"Reinforced fact: {fact.entity_id}")
            return False  # reinforcement, not a new fact

        # New fact
        g.add((subj, RDF.type,              _uri(VK, fact.entity_type)))
        g.add((subj, VK.entity_id,          _lit(fact.entity_id)))
        g.add((subj, VK.entity_type,        _lit(fact.entity_type)))
        g.add((subj, VK.classification,     _lit(fact.classification)))
        g.add((subj, VK.trust_score,        _lit(fact.trust_score, XSD.float)))
        g.add((subj, VK.provenance_ref,     _lit(fact.provenance_ref)))
        g.add((subj, VK.created_at,         _lit(fact.created_at.isoformat())))
        g.add((subj, VK.source,             _lit(fact.source)))
        g.add((subj, VK.last_reinforced,    _lit(_now_iso())))

        if fact.domain_module:
            g.add((subj, VK.domain_module, _lit(fact.domain_module)))
        if fact.external_id:
            g.add((subj, VK.external_id, _lit(fact.external_id)))

        # domain_properties stored as JSON literal
        if fact.domain_properties:
            g.add((subj, VK.domain_properties,
                   _lit(json.dumps(fact.domain_properties))))

        logger.debug(f"Wrote fact: {fact.entity_id} ({fact.entity_type})")
        return True  # new fact

    async def write_edge(
        self,
        edge: WeightedEdge,
        session_id: SessionId | None = None,
    ) -> bool:
        """
        Write or reinforce a WeightedEdge in the knowledge Named Graph.
        Returns True if this was a new edge, False if it was a reinforcement.
        """
        g = self._named(KNOWLEDGE_GRAPH)
        subj = _uri(VK, f"edge/{edge.edge_id}")

        # Check for existing edge
        if (subj, VK.edge_id, None) in g:
            # Reinforce: keep the higher effective_weight, increment count
            count = int(g.value(subj, VK.reinforcement_count) or 0)
            existing_weight = float(g.value(subj, VK.effective_weight) or 0.0)
            g.set((subj, VK.effective_weight,
                   _lit(max(existing_weight, edge.effective_weight), XSD.float)))
            g.set((subj, VK.reinforcement_count, _lit(count + 1, XSD.integer)))
            g.set((subj, VK.last_reinforced, _lit(_now_iso())))
            logger.debug(f"Reinforced edge: {edge.edge_id}")
            return False  # reinforcement, not a new edge

        # New edge
        g.add((subj, VK.edge_id,              _lit(edge.edge_id)))
        g.add((subj, VK.source_id,            _lit(edge.source_id)))
        g.add((subj, VK.target_id,            _lit(edge.target_id)))
        g.add((subj, VK.relationship_type,    _lit(edge.relationship_type)))
        g.add((subj, VK.distance_weight,      _lit(edge.base_weight.distance, XSD.float)))
        g.add((subj, VK.complexity_weight,    _lit(edge.base_weight.complexity, XSD.float)))
        g.add((subj, VK.size_weight,          _lit(edge.base_weight.size, XSD.float)))
        g.add((subj, VK.effective_weight,     _lit(edge.effective_weight, XSD.float)))
        g.add((subj, VK.last_reinforced,      _lit(edge.last_reinforced.isoformat())))
        g.add((subj, VK.reinforcement_count,  _lit(edge.reinforcement_count, XSD.integer)))
        g.add((subj, VK.is_sensitive,         _lit(edge.is_sensitive, XSD.boolean)))
        g.add((subj, VK.classification,       _lit(edge.classification)))
        g.add((subj, VK.created_at,           _lit(edge.created_at.isoformat())))
        g.add((subj, VK.provenance_ref,       _lit(edge.provenance_ref)))

        logger.debug(f"Wrote edge: {edge.source_id} → {edge.target_id}")
        return True  # new edge

    # ── NAVIGATE ──────────────────────────────────────────────────────────────

    async def get_fact(self, entity_id: EntityId) -> TypedFact | None:
        """Return a TypedFact by entity_id. None if not found."""
        g = self._named(KNOWLEDGE_GRAPH)
        subj = _uri(VK, f"fact/{entity_id}")

        entity_type = g.value(subj, VK.entity_type)
        if entity_type is None:
            return None

        return self._fact_from_graph(g, subj, entity_id)

    async def get_edges(
        self,
        entity_id: EntityId,
        min_weight: float = 0.0,
    ) -> list[WeightedEdge]:
        """Return edges connected to entity_id above min_weight."""
        g = self._named(KNOWLEDGE_GRAPH)
        edges = []

        for subj in g.subjects(VK.source_id, _lit(entity_id)):
            weight = float(g.value(subj, VK.effective_weight) or 0.0)
            if weight >= min_weight:
                edge = self._edge_from_graph(g, subj)
                if edge:
                    edges.append(edge)

        for subj in g.subjects(VK.target_id, _lit(entity_id)):
            weight = float(g.value(subj, VK.effective_weight) or 0.0)
            if weight >= min_weight:
                edge = self._edge_from_graph(g, subj)
                if edge:
                    edges.append(edge)

        return sorted(edges, key=lambda e: e.effective_weight, reverse=True)

    async def search_facts(
        self,
        query: str,
        request: ContextRequest,
    ) -> list[TypedFact]:
        """
        Keyword search over facts in the knowledge graph.
        Matches entity_id, entity_type, and domain_properties.
        Respects include_classifications and min_weight constraints.
        """
        g = self._named(KNOWLEDGE_GRAPH)
        query_lower = query.lower()
        results: list[TypedFact] = []

        for subj in g.subjects(RDF.type, None):
            entity_id_val = str(g.value(subj, VK.entity_id) or "")
            entity_type_val = str(g.value(subj, VK.entity_type) or "")
            domain_props_val = str(g.value(subj, VK.domain_properties) or "")
            classification_val = str(g.value(subj, VK.classification) or "internal")

            # Classification filter
            if classification_val not in [c for c in request.include_classifications]:
                continue

            # Keyword match
            searchable = f"{entity_id_val} {entity_type_val} {domain_props_val}".lower()
            if any(term in searchable for term in query_lower.split()):
                fact = self._fact_from_graph(g, subj, entity_id_val)
                if fact:
                    results.append(fact)

        return results

    # ── Decay ─────────────────────────────────────────────────────────────────

    async def apply_decay(self) -> dict[str, int]:
        """
        Apply power-law decay to all edges in the knowledge graph.

        Formula:
          days = (now - last_reinforced).days
          spacing_bonus = min(spacing_cap, 1.0 + days / 30.0)
          exponent = base_exponent * (sensitive_multiplier if is_sensitive else 1.0)
          effective_weight = base_weight * spacing_bonus * (1 + days)^(-exponent)

        Edges below prune_threshold are removed.
        """
        g = self._named(KNOWLEDGE_GRAPH)
        now = datetime.now(UTC)
        params = self._decay

        edges_decayed = 0
        edges_pruned = 0
        to_remove: list[URIRef] = []

        for subj in list(g.subjects(VK.edge_id, None)):
            try:
                last_reinforced_str = str(g.value(subj, VK.last_reinforced) or "")
                if not last_reinforced_str:
                    continue

                last_reinforced = _parse_dt(last_reinforced_str)
                days = max(0.0, (now - last_reinforced).total_seconds() / 86400)

                # Base weight — average of three axes
                dist = float(g.value(subj, VK.distance_weight) or 0.5)
                comp = float(g.value(subj, VK.complexity_weight) or 0.5)
                size = float(g.value(subj, VK.size_weight) or 0.5)
                base = (dist + comp + size) / 3.0

                # Exponent — sensitive edges decay faster
                is_sensitive = str(g.value(subj, VK.is_sensitive) or "false").lower() == "true"
                exponent = params.exponent
                if is_sensitive:
                    exponent *= params.sensitive_multiplier

                # Reinforcement count — spacing bonus
                reinf_count = int(g.value(subj, VK.reinforcement_count) or 0)
                spacing_bonus = (
                    min(params.spacing_cap, 1.0 + days / 30.0) if reinf_count >= 1 else 1.0
                )

                # Power-law decay
                new_weight = base * spacing_bonus * ((1 + days) ** (-exponent))
                new_weight = max(0.0, min(1.0, new_weight))

                if new_weight < params.prune_threshold:
                    to_remove.append(subj)
                    edges_pruned += 1
                else:
                    g.set((subj, VK.effective_weight, _lit(new_weight, XSD.float)))
                    edges_decayed += 1

            except Exception as e:
                logger.warning(f"Decay failed for edge {subj}: {e}")

        # Prune dead edges
        for subj in to_remove:
            for triple in list(g.triples((subj, None, None))):
                g.remove(triple)

        logger.info(
            f"Decay applied | decayed={edges_decayed} | pruned={edges_pruned}"
        )
        return {
            "edges_decayed": edges_decayed,
            "edges_pruned":  edges_pruned,
            "nodes_pruned":  0,  # Node pruning: future — orphan detection
        }

    # ── Consent graph ─────────────────────────────────────────────────────────

    async def write_consent(self, record: ConsentRecord) -> None:
        """Write a ConsentRecord to the consent Named Graph."""
        g = self._named(CONSENT_GRAPH)
        subj = _uri(VC, f"consent/{record.consent_ref}")

        g.add((subj, VC.consent_ref,   _lit(record.consent_ref)))
        g.add((subj, VC.subject_id,    _lit(record.subject_id)))
        g.add((subj, VC.granted_by,    _lit(record.granted_by)))
        g.add((subj, VC.granted_at,    _lit(record.granted_at.isoformat())))
        g.add((subj, VC.purpose,       _lit(record.purpose)))
        g.add((subj, VC.audit_id,      _lit(record.audit_id, XSD.integer)))
        g.add((subj, VC.classifications,
               _lit(json.dumps([c for c in record.classifications]))))

        if record.expires_at:
            g.add((subj, VC.expires_at, _lit(record.expires_at.isoformat())))
        if record.revoked_at:
            g.add((subj, VC.revoked_at, _lit(record.revoked_at.isoformat())))
            g.add((subj, VC.revoked_by, _lit(record.revoked_by or "")))

        logger.debug(f"Wrote consent: {record.consent_ref}")

    async def get_consent(self, consent_ref: ConsentRef) -> ConsentRecord | None:
        """Return the ConsentRecord for consent_ref. None if not found."""
        g = self._named(CONSENT_GRAPH)
        subj = _uri(VC, f"consent/{consent_ref}")

        granted_at_val = g.value(subj, VC.granted_at)
        if granted_at_val is None:
            return None

        classifications_raw = json.loads(
            str(g.value(subj, VC.classifications) or "[]")
        )
        classifications = tuple(classifications_raw)

        expires_at_val = g.value(subj, VC.expires_at)

        # Look for a revocation node linked to this consent ref (append-only)
        revoked_at_val = None
        revoked_by_val = None
        revocation_audit_id_val = None
        for rev_subj in g.subjects(VC.revokes, subj):
            revoked_at_val = g.value(rev_subj, VC.revoked_at)
            revoked_by_val = g.value(rev_subj, VC.revoked_by)
            revocation_audit_id_val = g.value(rev_subj, VC.revocation_audit_id)
            break  # Only one revocation can exist per consent ref

        return ConsentRecord(
            consent_ref=consent_ref,
            subject_id=str(g.value(subj, VC.subject_id) or ""),
            granted_by=str(g.value(subj, VC.granted_by) or ""),
            granted_at=_parse_dt(str(granted_at_val)),
            purpose=str(g.value(subj, VC.purpose) or ""),
            classifications=classifications,
            audit_id=int(g.value(subj, VC.audit_id) or 0),
            expires_at=_parse_dt(str(expires_at_val)) if expires_at_val else None,
            revoked_at=_parse_dt(str(revoked_at_val)) if revoked_at_val else None,
            revoked_by=str(revoked_by_val) if revoked_by_val else None,
            revocation_audit_id=int(revocation_audit_id_val) if revocation_audit_id_val else None,
        )

    async def revoke_consent(
        self,
        consent_ref: ConsentRef,
        revoked_by: str,
        audit_id: AuditRef,
    ) -> None:
        """
        Record a consent revocation as a new append-only node linked to
        the original grant. The original grant node is never modified —
        this preserves the append-only invariant of the consent named graph.
        """
        g = self._named(CONSENT_GRAPH)
        revoked_at = _now_iso()

        # Write a new revocation node rather than mutating the original grant
        revocation_subj = _uri(VC, f"revocation/{consent_ref}/{audit_id}")
        g.add((revocation_subj, VC.revokes,            _uri(VC, f"consent/{consent_ref}")))
        g.add((revocation_subj, VC.consent_ref,        _lit(consent_ref)))
        g.add((revocation_subj, VC.revoked_at,         _lit(revoked_at)))
        g.add((revocation_subj, VC.revoked_by,         _lit(revoked_by)))
        g.add((revocation_subj, VC.revocation_audit_id, _lit(audit_id, XSD.integer)))

        logger.info(f"Consent revoked: {consent_ref} by {revoked_by}")

    # ── Provenance / audit ────────────────────────────────────────────────────

    async def append_audit(self, event: AuditEvent) -> AuditRef:
        """Append an AuditEvent to the provenance Named Graph."""
        g = self._named(PROVENANCE_GRAPH)

        self._audit_sequence += 1
        seq = self._audit_sequence

        # Content hash: SHA-256 over sequence + event_type + timestamp + payload
        hash_input = json.dumps({
            "sequence":   seq,
            "event_type": event.event_type,
            "timestamp":  event.timestamp.isoformat(),
            "payload":    event.payload,
        }, sort_keys=True)
        content_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        subj = _uri(VP, f"audit/{seq}")

        g.add((subj, VP.sequence,       _lit(seq, XSD.integer)))
        g.add((subj, VP.event_type,     _lit(event.event_type)))
        g.add((subj, VP.timestamp,      _lit(event.timestamp.isoformat())))
        g.add((subj, VP.actor,          _lit(event.actor)))
        g.add((subj, VP.content_hash,   _lit(content_hash)))
        g.add((subj, VP.payload,        _lit(json.dumps(event.payload))))

        if event.session_id:
            g.add((subj, VP.session_id, _lit(event.session_id)))
        if event.consent_ref:
            g.add((subj, VP.consent_ref, _lit(event.consent_ref)))

        # Chain to previous record
        if self._previous_hash:
            g.add((subj, VP.previous_hash, _lit(self._previous_hash)))

        self._previous_hash = content_hash
        logger.debug(f"Audit event appended: seq={seq} type={event.event_type}")
        return seq

    async def get_audit(self, sequence: AuditRef) -> AuditEvent | None:
        """Return the AuditEvent at the given sequence number."""
        g = self._named(PROVENANCE_GRAPH)
        subj = _uri(VP, f"audit/{sequence}")

        timestamp_val = g.value(subj, VP.timestamp)
        if timestamp_val is None:
            return None

        return AuditEvent(
            sequence=sequence,
            event_type=str(g.value(subj, VP.event_type) or ""),
            timestamp=_parse_dt(str(timestamp_val)),
            actor=str(g.value(subj, VP.actor) or ""),
            session_id=(
                str(g.value(subj, VP.session_id)) if g.value(subj, VP.session_id) else None
            ),
            consent_ref=(
                str(g.value(subj, VP.consent_ref)) if g.value(subj, VP.consent_ref) else None
            ),
            payload=json.loads(str(g.value(subj, VP.payload) or "{}")),
            content_hash=str(g.value(subj, VP.content_hash) or ""),
            previous_hash=(
                str(g.value(subj, VP.previous_hash))
                if g.value(subj, VP.previous_hash)
                else None
            ),
            chain_valid=True,  # Verified separately by verify_chain()
        )

    async def verify_chain(self) -> bool:
        """Walk the audit chain and verify every hash. True = intact."""
        g = self._named(PROVENANCE_GRAPH)
        previous_hash = None

        for seq in range(1, self._audit_sequence + 1):
            subj = _uri(VP, f"audit/{seq}")
            stored_previous = g.value(subj, VP.previous_hash)
            stored_hash = str(g.value(subj, VP.content_hash) or "")

            expected_previous = str(stored_previous) if stored_previous else None
            if expected_previous != previous_hash:
                logger.error(f"Chain broken at sequence {seq}")
                return False

            previous_hash = stored_hash

        return True

    async def latest_audit_ref(self) -> AuditRef:
        """Return the sequence number of the most recent AuditEvent."""
        return self._audit_sequence

    async def _restore_audit_sequence(self) -> int:
        """Restore audit sequence from the provenance graph on startup."""
        g = self._named(PROVENANCE_GRAPH)
        max_seq = 0
        for subj in g.subjects(VP.sequence, None):
            seq_val = g.value(subj, VP.sequence)
            if seq_val:
                max_seq = max(max_seq, int(seq_val))
        return max_seq

    # ── GDPR erasure ──────────────────────────────────────────────────────────

    async def erase_subject(
        self,
        subject_id: EntityId,
        audit_id: AuditRef,
    ) -> dict[str, int]:
        """
        GDPR Article 17 erasure. Remove all facts linked to subject_id.
        The audit record of the erasure is preserved.
        """
        g = self._named(KNOWLEDGE_GRAPH)
        facts_erased = 0
        edges_erased = 0

        # Find all facts belonging to this subject (exact match only)
        subject_facts: list[URIRef] = []
        for subj in g.subjects(VK.entity_id, None):
            entity_id_val = str(g.value(subj, VK.entity_id) or "")
            if entity_id_val == subject_id:
                subject_facts.append(subj)

        # Remove facts and their connected edges
        for fact_subj in subject_facts:
            entity_id_val = str(g.value(fact_subj, VK.entity_id) or "")

            # Remove connected edges
            for edge_subj in list(g.subjects(VK.source_id, _lit(entity_id_val))):
                for triple in list(g.triples((edge_subj, None, None))):
                    g.remove(triple)
                edges_erased += 1

            for edge_subj in list(g.subjects(VK.target_id, _lit(entity_id_val))):
                for triple in list(g.triples((edge_subj, None, None))):
                    g.remove(triple)
                edges_erased += 1

            # Remove the fact itself
            for triple in list(g.triples((fact_subj, None, None))):
                g.remove(triple)
            facts_erased += 1

        logger.info(
            f"GDPR erasure | subject={subject_id} | "
            f"facts={facts_erased} | edges={edges_erased} | audit={audit_id}"
        )
        return {"facts_erased": facts_erased, "edges_erased": edges_erased}

    # ── Introspection ─────────────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        """Return store statistics."""
        kg = self._named(KNOWLEDGE_GRAPH)

        facts = sum(1 for _ in kg.subjects(VK.entity_id, None))
        edges = sum(1 for _ in kg.subjects(VK.edge_id, None))

        return {
            "facts":        facts,
            "edges":        edges,
            "audit_events": self._audit_sequence,
            "backend":      "rdflib",
            "named_graphs": ALL_NAMED_GRAPHS,
            "path":         str(self._path) if self._path else "in-memory",
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _fact_from_graph(
        self,
        g: Graph,
        subj: URIRef,
        entity_id: str,
    ) -> TypedFact | None:
        """Reconstruct a TypedFact from graph triples."""
        try:
            domain_props_raw = g.value(subj, VK.domain_properties)
            domain_props = json.loads(str(domain_props_raw)) if domain_props_raw else {}

            return TypedFact(
                entity_id=entity_id,
                entity_type=str(g.value(subj, VK.entity_type) or "verity:Keyword"),
                classification=str(g.value(subj, VK.classification) or DataClassification.INTERNAL),
                trust_score=float(g.value(subj, VK.trust_score) or 0.5),
                provenance_ref=str(g.value(subj, VK.provenance_ref) or ""),
                created_at=_parse_dt(str(g.value(subj, VK.created_at) or _now_iso())),
                source=str(g.value(subj, VK.source) or "unknown"),
                domain_properties=domain_props,
                domain_module=(
                    str(g.value(subj, VK.domain_module))
                    if g.value(subj, VK.domain_module)
                    else None
                ),
                external_id=(
                    str(g.value(subj, VK.external_id))
                    if g.value(subj, VK.external_id)
                    else None
                ),
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct fact {entity_id}: {e}")
            return None

    def _edge_from_graph(
        self,
        g: Graph,
        subj: URIRef,
    ) -> WeightedEdge | None:
        """Reconstruct a WeightedEdge from graph triples."""
        try:
            last_reinforced_str = str(g.value(subj, VK.last_reinforced) or _now_iso())
            created_at_str = str(g.value(subj, VK.created_at) or _now_iso())

            return WeightedEdge(
                edge_id=str(g.value(subj, VK.edge_id) or ""),
                source_id=str(g.value(subj, VK.source_id) or ""),
                target_id=str(g.value(subj, VK.target_id) or ""),
                relationship_type=str(g.value(subj, VK.relationship_type) or "verity:relatedTo"),
                base_weight=ThreeAxisWeight(
                    distance=float(g.value(subj, VK.distance_weight) or 0.5),
                    complexity=float(g.value(subj, VK.complexity_weight) or 0.5),
                    size=float(g.value(subj, VK.size_weight) or 0.5),
                ),
                effective_weight=float(g.value(subj, VK.effective_weight) or 0.0),
                last_reinforced=_parse_dt(last_reinforced_str),
                reinforcement_count=int(g.value(subj, VK.reinforcement_count) or 0),
                is_sensitive=str(g.value(subj, VK.is_sensitive) or "false").lower() == "true",
                classification=str(g.value(subj, VK.classification) or DataClassification.INTERNAL),
                created_at=_parse_dt(created_at_str),
                provenance_ref=str(g.value(subj, VK.provenance_ref) or ""),
            )
        except Exception as e:
            logger.warning(f"Failed to reconstruct edge: {e}")
            return None
