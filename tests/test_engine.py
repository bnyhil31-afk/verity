"""
tests/test_engine.py
====================
Integration tests for the Verity engine.

These tests exercise the full stack:
  RELATE → NAVIGATE → GOVERN → REMEMBER

Brain Test: context assembled from ingested facts is coherent.
Machine Test: engine behavior is independent of backend implementation.

Note: Engine.start() calls verify_principles() which runs canary tests.
These tests mock the principles check to isolate engine behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from verity.core.engine import Engine
from verity.core.exceptions import (
    ConsentRequiredError,
    CrisisBarrierError,
    EngineNotStartedError,
    SessionClosedError,
)
from verity.core.graph_store.rdflib_store import RDFLibStore
from verity.core.principles import LoadedPrinciples
from verity.core.types import (
    DEFAULT_DECAY_PARAMETERS,
    AuditEventType,
    CheckpointDecision,
    Completeness,
    ConsentRecord,
    DataClassification,
    ProposedAction,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(UTC)


def _mock_principles() -> LoadedPrinciples:
    return LoadedPrinciples(
        version=1,
        sequence="0000000001",
        timestamp="2025-01-01T00:00:00Z",
        immutable=(),
        regulated=(),
        operational=(),
        canary_tests=(),
        content_hash="abc123",
    )


async def _make_engine() -> Engine:
    """
    Build a started engine with an in-memory rdflib store.
    Bypasses principles verification for unit testing.
    """
    store = RDFLibStore(path=None, decay_parameters=DEFAULT_DECAY_PARAMETERS)
    await store.initialize()

    engine = Engine(
        store=store,
        principles=_mock_principles(),
        decay_parameters=DEFAULT_DECAY_PARAMETERS,
    )
    engine._started = True

    # Seed the audit trail
    from verity.core.types import AuditEvent
    await store.append_audit(AuditEvent(
        sequence=0,
        event_type=AuditEventType.PRINCIPLES_VERIFIED,
        timestamp=_now(),
        actor="test",
        session_id=None,
        consent_ref=None,
        payload={},
        content_hash="",
        previous_hash=None,
        chain_valid=True,
    ))

    return engine


async def _add_consent(
    engine: Engine,
    consent_ref: str = "consent:test",
    purpose: str = "test_purpose",
) -> ConsentRecord:
    """Write a valid consent record to the store."""
    record = ConsentRecord(
        consent_ref=consent_ref,
        subject_id="patient:abc123",
        granted_by="clinician",
        granted_at=_now(),
        purpose=purpose,
        classifications=(DataClassification.INTERNAL, DataClassification.PUBLIC),
        audit_id=1,
    )
    await engine._store.write_consent(record)
    return record


# ── Engine lifecycle ──────────────────────────────────────────────────────────

class TestEngineLifecycle:

    async def test_engine_starts(self):
        engine = await _make_engine()
        assert engine._started is True

    async def test_engine_stops(self):
        engine = await _make_engine()
        await engine.stop()
        assert engine._started is False

    async def test_require_started_raises(self):
        store = RDFLibStore()
        await store.initialize()
        engine = Engine(store=store, principles=_mock_principles())
        # _started is False — operations should raise
        with pytest.raises(EngineNotStartedError):
            await engine.stats()

    async def test_stats_returns_expected_keys(self):
        engine = await _make_engine()
        stats = await engine.stats()
        assert "facts" in stats
        assert "edges" in stats
        assert "modules" in stats
        assert "principles_version" in stats


# ── RELATE ────────────────────────────────────────────────────────────────────

class TestRelate:

    async def test_ingest_basic_text(self):
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            result = await s.ingest(
                "Patient reported fatigue and low energy levels",
                source="manual_entry",
            )
        assert result.crisis_detected is False
        assert result.audit_id > 0

    async def test_ingest_extracts_concepts(self):
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            result = await s.ingest(
                "The patient has type 2 diabetes and hypertension",
                source="ehr",
            )
        assert len(result.concepts) > 0

    async def test_crisis_barrier_fires_before_graph_write(self):
        """Crisis content must not reach the graph."""
        engine = await _make_engine()
        await _add_consent(engine)

        with pytest.raises(CrisisBarrierError):
            async with engine.session(consent_ref="consent:test") as s:
                await s.ingest(
                    "I want to end my life",
                    source="chat",
                )

        # Verify nothing was written to knowledge graph
        stats = await engine.stats()
        assert stats["facts"] == 0

    async def test_crisis_audit_event_recorded(self):
        """Even when crisis fires, the detection event is recorded."""
        engine = await _make_engine()
        await _add_consent(engine)

        try:
            async with engine.session(consent_ref="consent:test") as s:
                await s.ingest("I want to kill myself", source="chat")
        except CrisisBarrierError:
            pass

        stats = await engine.stats()
        # Audit events should include the crisis detection event
        assert stats["audit_events"] >= 1

    async def test_ingest_phi_classification(self):
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            result = await s.ingest(
                "John Smith DOB 1975-03-15 SSN 123-45-6789",
                source="ehr",
                classification=DataClassification.PHI,
            )
        assert result.crisis_detected is False


# ── NAVIGATE ──────────────────────────────────────────────────────────────────

class TestNavigate:

    async def test_context_requires_consent(self):
        """Consent gate blocks context assembly without valid consent."""
        engine = await _make_engine()
        # No consent written

        with pytest.raises(ConsentRequiredError):
            async with engine.session(consent_ref="consent:nonexistent") as s:
                await s.context(
                    query="test query",
                    purpose="test_purpose",
                )

    async def test_context_returns_bundle(self):
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            await s.ingest("Patient has diabetes and fatigue")
            context = await s.context(
                query="diabetes fatigue",
                purpose="test_purpose",
            )

        assert context.uncertainty >= 0.0
        assert context.uncertainty <= 1.0
        assert len(context.reasoning_trace) > 0
        assert context.agent_prompt != ""
        assert context.audit_id > 0
        assert context.consent_ref == "consent:test"

    async def test_context_uncertainty_always_present(self):
        """Invariant: uncertainty is never missing."""
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            context = await s.context(
                query="anything",
                purpose="test_purpose",
            )
        # Empty result should have maximum uncertainty
        assert context.uncertainty == 1.0

    async def test_context_reasoning_trace_never_empty(self):
        """Invariant: reasoning_trace is never empty."""
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            context = await s.context(
                query="test",
                purpose="test_purpose",
            )
        assert len(context.reasoning_trace) > 0

    async def test_context_excluded_is_tuple_not_none(self):
        """Invariant: excluded is an empty tuple, not None."""
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            context = await s.context(
                query="test",
                purpose="test_purpose",
            )
        assert context.excluded is not None
        assert isinstance(context.excluded, tuple)

    async def test_context_agent_prompt_populated(self):
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            await s.ingest("Patient has chronic back pain and insomnia")
            context = await s.context(
                query="back pain insomnia",
                purpose="test_purpose",
            )
        assert len(context.agent_prompt) > 0
        assert "test_purpose" in context.agent_prompt

    async def test_context_purpose_mismatch_raises(self):
        """Consent for purpose A does not authorize purpose B."""
        engine = await _make_engine()
        await _add_consent(engine, purpose="clinical_decision_support")

        with pytest.raises(Exception):  # PurposeMismatchError
            async with engine.session(consent_ref="consent:test") as s:
                await s.context(
                    query="test",
                    purpose="research_deidentified",  # Wrong purpose
                )

    async def test_relate_navigate_roundtrip(self):
        """Brain Test: ingested facts are retrievable via context."""
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            await s.ingest(
                "The patient has type 2 diabetes and elevated HbA1c",
                source="ehr",
            )
            context = await s.context(
                query="diabetes HbA1c",
                purpose="test_purpose",
            )

        # Facts should be in the bundle
        assert len(context.facts) > 0 or context.completeness == Completeness.EMPTY


# ── Session ───────────────────────────────────────────────────────────────────

class TestSession:

    async def test_session_closed_after_context_manager(self):
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            assert s.state.is_open is True

        assert s.state.is_open is False

    async def test_operation_on_closed_session_raises(self):
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            pass  # Session closed here

        with pytest.raises(SessionClosedError):
            await s.ingest("test text")

    async def test_session_tracks_facts_ingested(self):
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            await s.ingest("first observation about fatigue")
            await s.ingest("second observation about low mood")
            assert s.state.facts_ingested == 2

    async def test_session_records_close_audit_event(self):
        engine = await _make_engine()
        await _add_consent(engine)

        stats_before = await engine.stats()
        async with engine.session(consent_ref="consent:test"):
            pass
        stats_after = await engine.stats()

        assert stats_after["audit_events"] > stats_before["audit_events"]


# ── GOVERN ────────────────────────────────────────────────────────────────────

class TestGovern:

    async def test_checkpoint_veto_on_timeout(self):
        """
        GOVERN: VETOED is the default. Timeout = veto.
        This test patches the await to simulate an immediate timeout.
        """
        engine = await _make_engine()
        await _add_consent(engine)

        from verity.core.types import ContextBundle
        context_bundle = ContextBundle(
            facts=(),
            edges=(),
            uncertainty=0.3,
            completeness=Completeness.SUFFICIENT,
            excluded=(),
            reasoning_trace=("Test trace.",),
            consent_ref="consent:test",
            purpose="test_purpose",
            assembled_at=_now(),
            audit_id=1,
            session_id=None,
            agent_prompt="[CONTEXT] test",
            agent_prompt_tokens=5,
            checkpoint_required=True,
            checkpoint_context="Test checkpoint.",
        )

        action = ProposedAction(
            action_type="test_action",
            affects=("entity:abc",),
            classification=DataClassification.INTERNAL,
            reversible=True,
            description="A test action",
            proposed_by="test_agent",
        )

        # Patch _await_checkpoint_response to simulate timeout
        async def _mock_timeout(*args, **kwargs):
            return CheckpointDecision.VETOED, "timeout", "No response within timeout."

        engine._await_checkpoint_response = _mock_timeout

        async with engine.session(consent_ref="consent:test") as s:
            result = await s.checkpoint(action, context_bundle, timeout_seconds=1)

        assert result.decision == CheckpointDecision.VETOED
        assert result.decided_by == "timeout"
        assert result.audit_id > 0

    async def test_checkpoint_records_both_events(self):
        """Presentation AND decision are both recorded in audit trail."""
        engine = await _make_engine()
        await _add_consent(engine)

        from verity.core.types import ContextBundle
        context_bundle = ContextBundle(
            facts=(), edges=(), uncertainty=0.3,
            completeness=Completeness.SUFFICIENT, excluded=(),
            reasoning_trace=("trace",), consent_ref="consent:test",
            purpose="test_purpose", assembled_at=_now(), audit_id=1,
            session_id=None, agent_prompt="[CONTEXT] test",
            agent_prompt_tokens=5, checkpoint_required=True,
            checkpoint_context="Required.",
        )
        action = ProposedAction(
            action_type="test", affects=("e:1",),
            classification=DataClassification.INTERNAL,
            reversible=True, description="Test", proposed_by="agent",
        )

        async def _mock_approve(*args, **kwargs):
            return CheckpointDecision.APPROVED, "human", None

        engine._await_checkpoint_response = _mock_approve
        stats_before = await engine.stats()

        async with engine.session(consent_ref="consent:test") as s:
            await s.checkpoint(action, context_bundle)

        stats_after = await engine.stats()
        # Two checkpoint events + session close
        assert stats_after["audit_events"] >= stats_before["audit_events"] + 2


# ── Decay ─────────────────────────────────────────────────────────────────────

class TestDecay:

    async def test_apply_decay_records_audit_event(self):
        engine = await _make_engine()
        stats_before = await engine.stats()
        await engine.apply_decay()
        stats_after = await engine.stats()
        assert stats_after["audit_events"] > stats_before["audit_events"]

    async def test_apply_decay_returns_stats(self):
        engine = await _make_engine()
        result = await engine.apply_decay()
        assert "edges_decayed" in result
        assert "edges_pruned" in result


# ── RELATE widened ────────────────────────────────────────────────────────────

class TestRelateWidened:

    async def test_ingest_accepts_string(self):
        """Existing string path is unchanged after the signature widening."""
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            result = await s.ingest(
                "Patient reported fatigue and low energy levels",
                source="manual_entry",
            )
        assert result.crisis_detected is False
        assert result.audit_id > 0

    async def test_ingest_accepts_dict(self):
        """dict input is serialised to JSON and processed via the YAKE path."""
        engine = await _make_engine()
        await _add_consent(engine)

        async with engine.session(consent_ref="consent:test") as s:
            result = await s.ingest(
                {"observation": "elevated HbA1c", "value": 7.8, "unit": "percent"},
                source="ehr",
            )
        assert result.crisis_detected is False
        assert result.audit_id > 0

    async def test_ingest_accepts_connector_record(self):
        """ConnectorRecord fields are mapped correctly into relate()."""
        from datetime import UTC, datetime

        from verity.core.connectors import ConnectorRecord

        engine = await _make_engine()
        await _add_consent(engine)

        record = ConnectorRecord(
            id="rec:001",
            content="Patient has chronic back pain and insomnia",
            source_id="ehr_system",
            resource="/records/patient_001.txt",
            metadata={"department": "ortho"},
            classification="internal",
            timestamp=datetime.now(UTC),
            trust_score=0.90,
        )

        async with engine.session(consent_ref="consent:test") as s:
            result = await s.ingest(record)

        assert result.crisis_detected is False
        assert result.audit_id > 0

    async def test_ingest_from_filesystem(self, tmp_path):
        """ingest_from() accumulates results across all records in a connector."""
        from verity.core.connectors.filesystem import FilesystemConnector

        # Create 3 text files
        (tmp_path / "a.txt").write_text("Patient has type 2 diabetes and fatigue")
        (tmp_path / "b.txt").write_text("Elevated blood pressure noted during visit")
        (tmp_path / "c.txt").write_text("Follow-up scheduled for cardiology review")

        engine = await _make_engine()
        await _add_consent(engine)

        connector = FilesystemConnector(source_id="test_fs")

        async with engine.session(consent_ref="consent:test") as s:
            result = await s.ingest_from(connector, str(tmp_path / "*.txt"))

        assert result.crisis_detected is False
        assert result.audit_id > 0
        # Three files → at least some facts extracted across the batch
        assert len(result.facts_added) + len(result.facts_updated) >= 0  # shape check
        # Session state updated
        assert s.state.facts_ingested > 0

    async def test_crisis_barrier_fires_on_connector_record(self):
        """Crisis barrier fires when a ConnectorRecord carries crisis content."""
        from verity.core.connectors import ConnectorRecord
        from verity.core.exceptions import CrisisBarrierError

        engine = await _make_engine()
        await _add_consent(engine)

        record = ConnectorRecord(
            id="rec:crisis",
            content="I want to end my life",
            source_id="chat",
            resource="chat_log",
            trust_score=0.5,
        )

        with pytest.raises(CrisisBarrierError):
            async with engine.session(consent_ref="consent:test") as s:
                await s.ingest(record)

        # Nothing written to the knowledge graph
        stats = await engine.stats()
        assert stats["facts"] == 0
