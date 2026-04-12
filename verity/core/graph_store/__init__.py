"""
verity.core.graph_store
=======================
The graph store abstraction layer.

This is the Machine Test boundary. The four core functions (RELATE /
NAVIGATE / GOVERN / REMEMBER) call this Protocol. They have no knowledge
of what backend is running beneath it. Swapping backends requires zero
changes to the engine.

Backends implemented:
  - rdflib     — Personal tier. Pure Python, no external services.
                 Three Named Graphs in memory or on disk.
                 Default backend. Runs on a Raspberry Pi.

  - pgvector   — Team tier. PostgreSQL + pgvector + rdflib.
                 Multi-user, persistent, vector-similarity search.
                 Install: pip install verity[team]

  - jena       — Enterprise tier. Apache Jena Fuseki via HTTP.
                 Full SPARQL 1.1, OWL 2 reasoning, Named Graphs.
                 Install: pip install verity[enterprise]

The Protocol is runtime_checkable — isinstance() works for testing.
Any class that implements all methods satisfies the Protocol.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from verity.core.types import (
    AuditEvent,
    AuditRef,
    ConsentRecord,
    ConsentRef,
    ContextRequest,
    EntityId,
    RelateResult,
    SessionId,
    TypedFact,
    WeightedEdge,
)


@runtime_checkable
class GraphStore(Protocol):
    """
    The storage backend interface for the Verity engine.

    Machine Test: the four core functions must be able to swap any
    implementation of this Protocol without modification.

    Brain Test: contextual flow — accumulated facts, edges, and their
    decay state — must be preserved correctly across all operations.

    Three Named Graphs, regardless of backend:
      urn:verity:knowledge   — typed facts and weighted edges
      urn:verity:provenance  — append-only Merkle chain
      urn:verity:consent     — consent ledger (OR-Set CRDT semantics)
    """

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Create schema and Named Graphs. Idempotent — safe at every boot.
        Called by Engine.start() before any other operation.
        """
        ...

    async def close(self) -> None:
        """
        Release backend resources cleanly.
        Called by Engine.stop() and session context manager __aexit__.
        """
        ...

    # ── RELATE ────────────────────────────────────────────────────────────────

    async def write_fact(
        self,
        fact: TypedFact,
        session_id: SessionId | None = None,
    ) -> None:
        """
        Write a TypedFact to the knowledge graph.
        If a fact with the same entity_id exists, reinforce it.
        """
        ...

    async def write_edge(
        self,
        edge: WeightedEdge,
        session_id: SessionId | None = None,
    ) -> None:
        """
        Write a WeightedEdge to the knowledge graph.
        If an edge with the same edge_id exists, update effective_weight
        and increment reinforcement_count.
        """
        ...

    # ── NAVIGATE ──────────────────────────────────────────────────────────────

    async def get_fact(self, entity_id: EntityId) -> TypedFact | None:
        """Return a single TypedFact by entity_id. None if not found."""
        ...

    async def get_edges(
        self,
        entity_id: EntityId,
        min_weight: float = 0.0,
    ) -> list[WeightedEdge]:
        """
        Return all edges connected to entity_id with effective_weight
        above min_weight. Used by BFS traversal in NAVIGATE.
        """
        ...

    async def search_facts(
        self,
        query: str,
        request: ContextRequest,
    ) -> list[TypedFact]:
        """
        Find facts relevant to the query string.

        Personal tier: keyword matching against entity_id and domain_properties.
        Team tier: pgvector similarity search.
        Enterprise tier: SPARQL full-text search against Jena.

        Always respects request.include_classifications and request.min_weight.
        """
        ...

    # ── Decay ─────────────────────────────────────────────────────────────────

    async def apply_decay(self) -> dict[str, int]:
        """
        Apply power-law decay to all edges.
        Prune edges below DecayParameters.prune_threshold.

        Returns {"edges_decayed": int, "edges_pruned": int, "nodes_pruned": int}

        The decay formula is:
          effective_weight = base x (1 + days_since_reinforced)^(-exponent)
          sensitive edges use: exponent x sensitive_multiplier
          with spacing bonus: min(spacing_cap, 1.0 + days / 30.0)
        """
        ...

    # ── Consent graph ─────────────────────────────────────────────────────────

    async def write_consent(self, record: ConsentRecord) -> None:
        """Write a ConsentRecord to the consent Named Graph."""
        ...

    async def get_consent(self, consent_ref: ConsentRef) -> ConsentRecord | None:
        """Return the ConsentRecord for consent_ref. None if not found."""
        ...

    async def revoke_consent(
        self,
        consent_ref: ConsentRef,
        revoked_by: str,
        audit_id: AuditRef,
    ) -> None:
        """
        Mark a ConsentRecord as revoked.
        The record is never deleted — revocation is recorded immutably.
        """
        ...

    # ── Provenance / audit ────────────────────────────────────────────────────

    async def append_audit(self, event: AuditEvent) -> AuditRef:
        """
        Append an AuditEvent to the provenance Named Graph.
        Returns the sequence number assigned to this event.
        The Merkle chain is extended — previous_hash is set here.
        """
        ...

    async def get_audit(self, sequence: AuditRef) -> AuditEvent | None:
        """Return the AuditEvent at the given sequence number."""
        ...

    async def verify_chain(self) -> bool:
        """
        Walk the entire audit chain and verify every hash.
        Returns True if the chain is intact, False if any record is tampered.
        """
        ...

    async def latest_audit_ref(self) -> AuditRef:
        """Return the sequence number of the most recent AuditEvent."""
        ...

    # ── GDPR erasure ──────────────────────────────────────────────────────────

    async def erase_subject(
        self,
        subject_id: EntityId,
        audit_id: AuditRef,
    ) -> dict[str, int]:
        """
        GDPR Article 17 erasure. Remove all facts linked to subject_id
        from the knowledge graph. The audit record of the erasure itself
        is preserved — you can prove the erasure happened.

        Returns {"facts_erased": int, "edges_erased": int}

        The audit event must be written BEFORE erasure begins.
        If erasure fails partway, the partial erasure is recorded.
        """
        ...

    # ── Introspection ─────────────────────────────────────────────────────────

    async def stats(self) -> dict[str, Any]:
        """
        Return backend statistics.
        Minimum keys: {"facts": int, "edges": int, "audit_events": int,
                        "backend": str, "named_graphs": list[str]}
        """
        ...
