"""
tests/test_graph_store.py
=========================
Tests for the rdflib graph store backend.

Machine Test: the GraphStore Protocol is satisfied.
Brain Test: facts, edges, and decay state survive read/write cycles.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock

from verity.core.graph_store import GraphStore
from verity.core.graph_store.rdflib_store import RDFLibStore
from verity.core.graph_store.registry import get_graph_store
from verity.core.types import (
    AuditEvent,
    AuditEventType,
    ConsentRecord,
    DataClassification,
    DecayParameters,
    ThreeAxisWeight,
    TypedFact,
    WeightedEdge,
    DEFAULT_DECAY_PARAMETERS,
    ContextRequest,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
async def store() -> RDFLibStore:
    """Fresh in-memory store for each test."""
    s = RDFLibStore(path=None, decay_parameters=DEFAULT_DECAY_PARAMETERS)
    await s.initialize()
    return s


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fact(entity_id: str = "kw:test", classification: str = DataClassification.INTERNAL) -> TypedFact:
    return TypedFact(
        entity_id=entity_id,
        entity_type="verity:Keyword",
        classification=classification,
        trust_score=0.8,
        provenance_ref="prov:test",
        created_at=_now(),
        source="test",
    )


def _edge(source: str = "kw:a", target: str = "kw:b") -> WeightedEdge:
    return WeightedEdge(
        edge_id=f"edge:{source}:{target}",
        source_id=source,
        target_id=target,
        relationship_type="verity:relatedTo",
        base_weight=ThreeAxisWeight(distance=0.7, complexity=0.5, size=0.3),
        effective_weight=0.6,
        last_reinforced=_now(),
        reinforcement_count=1,
        is_sensitive=False,
        classification=DataClassification.INTERNAL,
        created_at=_now(),
        provenance_ref="prov:test",
    )


def _consent(
    consent_ref: str = "consent:test",
    purpose: str = "test_purpose",
) -> ConsentRecord:
    return ConsentRecord(
        consent_ref=consent_ref,
        subject_id="patient:abc123",
        granted_by="clinician:xyz",
        granted_at=_now(),
        purpose=purpose,
        classifications=(DataClassification.INTERNAL,),
        audit_id=1,
    )


def _audit_event(event_type: str = AuditEventType.INGEST) -> AuditEvent:
    return AuditEvent(
        sequence=0,
        event_type=event_type,
        timestamp=_now(),
        actor="test",
        session_id=None,
        consent_ref=None,
        payload={"test": True},
        content_hash="",
        previous_hash=None,
        chain_valid=True,
    )


# ── Protocol compliance ───────────────────────────────────────────────────────

class TestProtocolCompliance:
    """Machine Test: RDFLibStore satisfies the GraphStore Protocol."""

    def test_isinstance_graph_store(self):
        store = RDFLibStore()
        assert isinstance(store, GraphStore)

    def test_has_all_protocol_methods(self):
        store = RDFLibStore()
        required = [
            "initialize", "close",
            "write_fact", "write_edge",
            "get_fact", "get_edges", "search_facts",
            "apply_decay",
            "write_consent", "get_consent", "revoke_consent",
            "append_audit", "get_audit", "verify_chain", "latest_audit_ref",
            "erase_subject", "stats",
        ]
        for method in required:
            assert hasattr(store, method), f"RDFLibStore missing method: {method}"


# ── Lifecycle ─────────────────────────────────────────────────────────────────

class TestLifecycle:

    async def test_initialize_idempotent(self, store):
        """initialize() called twice does not raise."""
        await store.initialize()
        await store.initialize()

    async def test_close(self, store):
        await store.close()
        assert store._graph is None

    async def test_stats_after_init(self, store):
        stats = await store.stats()
        assert "facts" in stats
        assert "edges" in stats
        assert "audit_events" in stats
        assert stats["backend"] == "rdflib"
        assert len(stats["named_graphs"]) == 3


# ── Facts ─────────────────────────────────────────────────────────────────────

class TestFacts:

    async def test_write_and_read_fact(self, store):
        fact = _fact("kw:diabetes")
        await store.write_fact(fact)
        result = await store.get_fact("kw:diabetes")
        assert result is not None
        assert result.entity_id == "kw:diabetes"
        assert result.entity_type == "verity:Keyword"

    async def test_get_nonexistent_fact(self, store):
        result = await store.get_fact("kw:nonexistent")
        assert result is None

    async def test_fact_classification_preserved(self, store):
        fact = _fact("kw:phi_test", classification=DataClassification.PHI)
        await store.write_fact(fact)
        result = await store.get_fact("kw:phi_test")
        assert result is not None
        assert result.classification == DataClassification.PHI

    async def test_fact_trust_score_preserved(self, store):
        fact = _fact("kw:trust_test")
        await store.write_fact(fact)
        result = await store.get_fact("kw:trust_test")
        assert result is not None
        assert abs(result.trust_score - 0.8) < 0.01

    async def test_fact_reinforcement(self, store):
        """Writing the same fact twice reinforces the trust_score."""
        fact = _fact("kw:reinforce")
        await store.write_fact(fact)
        await store.write_fact(fact)
        result = await store.get_fact("kw:reinforce")
        assert result is not None
        # Reinforced trust score should be >= original
        assert result.trust_score >= 0.8


# ── Edges ─────────────────────────────────────────────────────────────────────

class TestEdges:

    async def test_write_and_read_edge(self, store):
        edge = _edge("kw:a", "kw:b")
        await store.write_edge(edge)
        edges = await store.get_edges("kw:a")
        assert len(edges) == 1
        assert edges[0].source_id == "kw:a"
        assert edges[0].target_id == "kw:b"

    async def test_get_edges_by_target(self, store):
        """get_edges returns edges where entity is the target too."""
        edge = _edge("kw:a", "kw:b")
        await store.write_edge(edge)
        edges = await store.get_edges("kw:b")
        assert len(edges) == 1

    async def test_edge_min_weight_filter(self, store):
        edge = _edge("kw:a", "kw:b")  # effective_weight=0.6
        await store.write_edge(edge)

        # Above threshold — returns result
        edges = await store.get_edges("kw:a", min_weight=0.5)
        assert len(edges) == 1

        # At threshold — returns result
        edges = await store.get_edges("kw:a", min_weight=0.6)
        assert len(edges) == 1

        # Above effective_weight — no results
        edges = await store.get_edges("kw:a", min_weight=0.9)
        assert len(edges) == 0

    async def test_edge_effective_weight_preserved(self, store):
        edge = _edge("kw:x", "kw:y")
        await store.write_edge(edge)
        edges = await store.get_edges("kw:x")
        assert abs(edges[0].effective_weight - 0.6) < 0.01


# ── Search ────────────────────────────────────────────────────────────────────

class TestSearch:

    async def test_search_finds_matching_fact(self, store):
        fact = _fact("kw:diabetes")
        await store.write_fact(fact)
        request = ContextRequest(
            query="diabetes",
            purpose="test",
            consent_ref="consent:test",
            include_classifications=(DataClassification.INTERNAL,),
        )
        results = await store.search_facts("diabetes", request)
        assert any(f.entity_id == "kw:diabetes" for f in results)

    async def test_search_respects_classification_filter(self, store):
        phi_fact = _fact("kw:phi_data", classification=DataClassification.PHI)
        await store.write_fact(phi_fact)
        request = ContextRequest(
            query="phi",
            purpose="test",
            consent_ref="consent:test",
            include_classifications=(DataClassification.INTERNAL,),  # PHI excluded
        )
        results = await store.search_facts("phi_data", request)
        assert not any(f.entity_id == "kw:phi_data" for f in results)

    async def test_search_empty_graph(self, store):
        request = ContextRequest(
            query="anything",
            purpose="test",
            consent_ref="consent:test",
        )
        results = await store.search_facts("anything", request)
        assert results == []


# ── Decay ─────────────────────────────────────────────────────────────────────

class TestDecay:

    async def test_decay_returns_expected_keys(self, store):
        result = await store.apply_decay()
        assert "edges_decayed" in result
        assert "edges_pruned" in result
        assert "nodes_pruned" in result

    async def test_decay_empty_graph(self, store):
        """Decay on empty graph does not raise."""
        result = await store.apply_decay()
        assert result["edges_decayed"] == 0
        assert result["edges_pruned"] == 0

    async def test_decay_reduces_weight_over_time(self, store):
        """
        Edges written with a past last_reinforced date should decay.
        This test writes an edge as if it was reinforced 90 days ago.
        """
        from datetime import timedelta
        old_date = _now() - timedelta(days=90)

        old_edge = WeightedEdge(
            edge_id="edge:old:a:b",
            source_id="kw:old_a",
            target_id="kw:old_b",
            relationship_type="verity:relatedTo",
            base_weight=ThreeAxisWeight(distance=0.8, complexity=0.5, size=0.5),
            effective_weight=0.6,
            last_reinforced=old_date,
            reinforcement_count=1,
            is_sensitive=False,
            classification=DataClassification.INTERNAL,
            created_at=old_date,
            provenance_ref="prov:test",
        )
        await store.write_edge(old_edge)

        result = await store.apply_decay()
        # The old edge should either be decayed or pruned
        assert result["edges_decayed"] + result["edges_pruned"] >= 1


# ── Consent ───────────────────────────────────────────────────────────────────

class TestConsent:

    async def test_write_and_read_consent(self, store):
        record = _consent("consent:abc", "clinical_decision_support")
        await store.write_consent(record)
        result = await store.get_consent("consent:abc")
        assert result is not None
        assert result.consent_ref == "consent:abc"
        assert result.purpose == "clinical_decision_support"

    async def test_get_nonexistent_consent(self, store):
        result = await store.get_consent("consent:nonexistent")
        assert result is None

    async def test_consent_is_active(self, store):
        record = _consent()
        await store.write_consent(record)
        result = await store.get_consent("consent:test")
        assert result is not None
        assert result.is_active is True

    async def test_revoke_consent(self, store):
        record = _consent("consent:revoke_me")
        await store.write_consent(record)
        await store.revoke_consent("consent:revoke_me", revoked_by="admin", audit_id=1)
        result = await store.get_consent("consent:revoke_me")
        assert result is not None
        assert result.revoked_at is not None
        assert result.is_active is False


# ── Audit chain ───────────────────────────────────────────────────────────────

class TestAuditChain:

    async def test_append_and_retrieve(self, store):
        event = _audit_event()
        seq = await store.append_audit(event)
        assert seq == 1
        retrieved = await store.get_audit(seq)
        assert retrieved is not None
        assert retrieved.event_type == AuditEventType.INGEST

    async def test_sequence_increments(self, store):
        seq1 = await store.append_audit(_audit_event())
        seq2 = await store.append_audit(_audit_event())
        seq3 = await store.append_audit(_audit_event())
        assert seq1 == 1
        assert seq2 == 2
        assert seq3 == 3

    async def test_latest_audit_ref(self, store):
        assert await store.latest_audit_ref() == 0
        await store.append_audit(_audit_event())
        assert await store.latest_audit_ref() == 1
        await store.append_audit(_audit_event())
        assert await store.latest_audit_ref() == 2

    async def test_verify_chain_empty(self, store):
        assert await store.verify_chain() is True

    async def test_verify_chain_valid(self, store):
        for _ in range(5):
            await store.append_audit(_audit_event())
        assert await store.verify_chain() is True


# ── GDPR erasure ──────────────────────────────────────────────────────────────

class TestErasure:

    async def test_erase_removes_facts(self, store):
        fact = _fact("patient:john_smith")
        await store.write_fact(fact)

        result = await store.erase_subject("patient:john_smith", audit_id=1)
        assert result["facts_erased"] >= 1

        retrieved = await store.get_fact("patient:john_smith")
        assert retrieved is None

    async def test_erase_nonexistent_subject(self, store):
        """Erasing a subject that doesn't exist does not raise."""
        result = await store.erase_subject("patient:nobody", audit_id=1)
        assert result["facts_erased"] == 0
        assert result["edges_erased"] == 0


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:

    def test_default_backend_is_rdflib(self, monkeypatch):
        monkeypatch.delenv("VERITY_GRAPH_BACKEND", raising=False)
        store = get_graph_store()
        assert isinstance(store, RDFLibStore)

    def test_unknown_backend_falls_back(self, monkeypatch):
        import warnings
        monkeypatch.setenv("VERITY_GRAPH_BACKEND", "quantum_database")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            store = get_graph_store()
            assert isinstance(store, RDFLibStore)
            assert len(w) == 1
            assert "quantum_database" in str(w[0].message)

    def test_explicit_rdflib_backend(self, monkeypatch):
        monkeypatch.setenv("VERITY_GRAPH_BACKEND", "rdflib")
        store = get_graph_store()
        assert isinstance(store, RDFLibStore)
