"""
tests/test_types.py
===================
Tests for verity.core.types — the contracts.

These tests verify that the structural invariants are enforced at
construction, not just documented. If a ContextBundle can be built
with an empty reasoning_trace, the architecture has failed.
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from verity.core.types import (
    CheckpointDecision,
    Completeness,
    ContextBundle,
    ContextRequest,
    DataClassification,
    DecayParameters,
    ExclusionNote,
    ModuleManifest,
    ThreeAxisWeight,
    TrustSource,
    TypedFact,
    WeightedEdge,
    DEFAULT_DECAY_PARAMETERS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fact(
    entity_id: str = "kw:test",
    classification: str = DataClassification.INTERNAL,
    trust_score: float = 0.8,
) -> TypedFact:
    return TypedFact(
        entity_id=entity_id,
        entity_type="verity:Keyword",
        classification=classification,
        trust_score=trust_score,
        provenance_ref="prov:abc123",
        created_at=_now(),
        source="test",
    )


def _edge(
    source: str = "kw:a",
    target: str = "kw:b",
    effective_weight: float = 0.6,
) -> WeightedEdge:
    return WeightedEdge(
        edge_id=f"edge:{source}:{target}",
        source_id=source,
        target_id=target,
        relationship_type="verity:relatedTo",
        base_weight=ThreeAxisWeight(distance=0.7, complexity=0.5, size=0.3),
        effective_weight=effective_weight,
        last_reinforced=_now(),
        reinforcement_count=1,
        is_sensitive=False,
        classification=DataClassification.INTERNAL,
        created_at=_now(),
        provenance_ref="prov:abc123",
    )


def _minimal_bundle(**overrides) -> dict:
    """Return keyword args for a minimal valid ContextBundle."""
    defaults = dict(
        facts=(_fact(),),
        edges=(_edge(),),
        uncertainty=0.3,
        completeness=Completeness.SUFFICIENT,
        excluded=(),
        reasoning_trace=("Test reasoning step.",),
        consent_ref="consent:abc123",
        purpose="test_purpose",
        assembled_at=_now(),
        audit_id=1,
        session_id=None,
        agent_prompt="[CONTEXT] test",
        agent_prompt_tokens=5,
        checkpoint_required=False,
        checkpoint_context=None,
    )
    defaults.update(overrides)
    return defaults


# ── ThreeAxisWeight ───────────────────────────────────────────────────────────

class TestThreeAxisWeight:

    def test_valid_construction(self):
        w = ThreeAxisWeight(distance=0.5, complexity=0.3, size=0.8)
        assert w.distance == 0.5
        assert w.complexity == 0.3
        assert w.size == 0.8

    def test_boundary_values(self):
        w = ThreeAxisWeight(distance=0.0, complexity=1.0, size=0.0)
        assert w.distance == 0.0
        assert w.complexity == 1.0

    def test_distance_out_of_range(self):
        with pytest.raises(ValueError, match="distance"):
            ThreeAxisWeight(distance=1.1, complexity=0.5, size=0.5)

    def test_complexity_negative(self):
        with pytest.raises(ValueError, match="complexity"):
            ThreeAxisWeight(distance=0.5, complexity=-0.1, size=0.5)

    def test_size_out_of_range(self):
        with pytest.raises(ValueError, match="size"):
            ThreeAxisWeight(distance=0.5, complexity=0.5, size=2.0)

    def test_frozen(self):
        w = ThreeAxisWeight(distance=0.5, complexity=0.5, size=0.5)
        with pytest.raises((AttributeError, TypeError)):
            w.distance = 0.9  # type: ignore


# ── DecayParameters ───────────────────────────────────────────────────────────

class TestDecayParameters:

    def test_defaults(self):
        p = DecayParameters()
        assert p.exponent == 0.5
        assert p.sensitive_multiplier == 1.4
        assert p.spacing_cap == 2.0
        assert p.prune_threshold == 0.05

    def test_default_singleton(self):
        assert DEFAULT_DECAY_PARAMETERS.exponent == 0.5

    def test_invalid_exponent_zero(self):
        with pytest.raises(ValueError, match="exponent"):
            DecayParameters(exponent=0.0)

    def test_invalid_exponent_negative(self):
        with pytest.raises(ValueError, match="exponent"):
            DecayParameters(exponent=-1.0)

    def test_sensitive_multiplier_below_one(self):
        """Sensitive edges must decay FASTER, not slower."""
        with pytest.raises(ValueError, match="sensitive_multiplier"):
            DecayParameters(sensitive_multiplier=0.9)

    def test_spacing_cap_below_one(self):
        with pytest.raises(ValueError, match="spacing_cap"):
            DecayParameters(spacing_cap=0.5)

    def test_prune_threshold_zero(self):
        with pytest.raises(ValueError, match="prune_threshold"):
            DecayParameters(prune_threshold=0.0)

    def test_frozen(self):
        p = DecayParameters()
        with pytest.raises((AttributeError, TypeError)):
            p.exponent = 1.0  # type: ignore


# ── DataClassification ────────────────────────────────────────────────────────

class TestDataClassification:

    def test_requires_consent_phi(self):
        assert DataClassification.PHI.requires_consent is True

    def test_requires_consent_pii(self):
        assert DataClassification.PII.requires_consent is True

    def test_requires_consent_financial(self):
        assert DataClassification.FINANCIAL.requires_consent is True

    def test_requires_consent_legal(self):
        assert DataClassification.LEGAL.requires_consent is True

    def test_no_consent_required_public(self):
        assert DataClassification.PUBLIC.requires_consent is False

    def test_no_consent_required_internal(self):
        assert DataClassification.INTERNAL.requires_consent is False

    def test_audit_on_access_phi(self):
        assert DataClassification.PHI.audit_on_access is True

    def test_audit_on_access_public_false(self):
        assert DataClassification.PUBLIC.audit_on_access is False

    def test_escalate_phi_wins(self):
        result = DataClassification.escalate(
            DataClassification.PHI,
            DataClassification.INTERNAL,
        )
        assert result == DataClassification.PHI

    def test_escalate_symmetric(self):
        a = DataClassification.escalate(DataClassification.PII, DataClassification.FINANCIAL)
        b = DataClassification.escalate(DataClassification.FINANCIAL, DataClassification.PII)
        assert a == b

    def test_escalate_same(self):
        result = DataClassification.escalate(
            DataClassification.INTERNAL,
            DataClassification.INTERNAL,
        )
        assert result == DataClassification.INTERNAL


# ── TypedFact ─────────────────────────────────────────────────────────────────

class TestTypedFact:

    def test_valid_construction(self):
        f = _fact()
        assert f.entity_id == "kw:test"
        assert f.trust_score == 0.8

    def test_trust_score_above_one(self):
        with pytest.raises(ValueError, match="trust_score"):
            _fact(trust_score=1.1)

    def test_trust_score_negative(self):
        with pytest.raises(ValueError, match="trust_score"):
            _fact(trust_score=-0.1)

    def test_trust_score_boundary_zero(self):
        f = _fact(trust_score=0.0)
        assert f.trust_score == 0.0

    def test_trust_score_boundary_one(self):
        f = _fact(trust_score=1.0)
        assert f.trust_score == 1.0

    def test_frozen(self):
        f = _fact()
        with pytest.raises((AttributeError, TypeError)):
            f.trust_score = 0.5  # type: ignore

    def test_domain_properties_default_empty(self):
        f = _fact()
        assert f.domain_properties == {}

    def test_domain_module_default_none(self):
        f = _fact()
        assert f.domain_module is None


# ── WeightedEdge ──────────────────────────────────────────────────────────────

class TestWeightedEdge:

    def test_valid_construction(self):
        e = _edge()
        assert e.source_id == "kw:a"
        assert e.target_id == "kw:b"
        assert e.effective_weight == 0.6

    def test_effective_weight_above_one(self):
        with pytest.raises(ValueError, match="effective_weight"):
            _edge(effective_weight=1.1)

    def test_effective_weight_negative(self):
        with pytest.raises(ValueError, match="effective_weight"):
            _edge(effective_weight=-0.1)

    def test_frozen(self):
        e = _edge()
        with pytest.raises((AttributeError, TypeError)):
            e.effective_weight = 0.9  # type: ignore


# ── ContextBundle — invariants ────────────────────────────────────────────────

class TestContextBundleInvariants:
    """
    The most important tests in the file.
    These verify that the five invariants are enforced at construction.
    """

    def test_valid_bundle(self):
        bundle = ContextBundle(**_minimal_bundle())
        assert bundle.uncertainty == 0.3
        assert len(bundle.reasoning_trace) == 1

    def test_uncertainty_above_one_rejected(self):
        with pytest.raises(ValueError, match="uncertainty"):
            ContextBundle(**_minimal_bundle(uncertainty=1.1))

    def test_uncertainty_negative_rejected(self):
        with pytest.raises(ValueError, match="uncertainty"):
            ContextBundle(**_minimal_bundle(uncertainty=-0.1))

    def test_empty_reasoning_trace_rejected(self):
        """A ContextBundle with no reasoning trace is non-compliant."""
        with pytest.raises(ValueError, match="reasoning_trace"):
            ContextBundle(**_minimal_bundle(reasoning_trace=()))

    def test_checkpoint_required_without_context_rejected(self):
        """checkpoint_context must be set when checkpoint_required is True."""
        with pytest.raises(ValueError, match="checkpoint_context"):
            ContextBundle(**_minimal_bundle(
                checkpoint_required=True,
                checkpoint_context=None,
            ))

    def test_checkpoint_required_with_context_valid(self):
        bundle = ContextBundle(**_minimal_bundle(
            checkpoint_required=True,
            checkpoint_context="High uncertainty.",
        ))
        assert bundle.checkpoint_required is True

    def test_excluded_empty_tuple_not_none(self):
        """excluded must be an empty tuple, not None."""
        bundle = ContextBundle(**_minimal_bundle(excluded=()))
        assert bundle.excluded == ()

    def test_frozen(self):
        bundle = ContextBundle(**_minimal_bundle())
        with pytest.raises((AttributeError, TypeError)):
            bundle.uncertainty = 0.9  # type: ignore

    def test_schema_version_default(self):
        bundle = ContextBundle(**_minimal_bundle())
        assert bundle.schema_version == "2.0"


# ── ModuleManifest ────────────────────────────────────────────────────────────

class TestModuleManifest:

    def test_minimal_manifest(self):
        m = ModuleManifest(
            module_id="verity_test",
            version="1.0.0",
            display_name="Test Module",
            classifications=(DataClassification.INTERNAL,),
            entry_point="verity_test:TestModule",
        )
        assert m.module_id == "verity_test"
        assert m.decay_parameters is None
        assert m.shacl_shapes_path is None
        assert m.requires_consent_for == ()
        assert m.checkpoint_purposes == ()

    def test_with_decay_override(self):
        custom_decay = DecayParameters(exponent=0.7)
        m = ModuleManifest(
            module_id="verity_fhir",
            version="1.0.0",
            display_name="FHIR R4",
            classifications=(DataClassification.PHI,),
            entry_point="verity_fhir:FHIRModule",
            decay_parameters=custom_decay,
        )
        assert m.decay_parameters.exponent == 0.7

    def test_frozen(self):
        m = ModuleManifest(
            module_id="verity_test",
            version="1.0.0",
            display_name="Test",
            classifications=(),
            entry_point="verity_test:Module",
        )
        with pytest.raises((AttributeError, TypeError)):
            m.module_id = "other"  # type: ignore


# ── TrustSource scores ────────────────────────────────────────────────────────

class TestTrustSource:

    def test_human_verified_highest(self):
        assert TrustSource.SCORES["human_verified"] == 0.95

    def test_unknown_lowest(self):
        assert TrustSource.SCORES["unknown"] == 0.20

    def test_institutional_above_algorithmic(self):
        assert (
            TrustSource.SCORES["institutional"]
            > TrustSource.SCORES["algorithmic_high"]
        )

    def test_all_scores_in_range(self):
        for source, score in TrustSource.SCORES.items():
            assert 0.0 <= score <= 1.0, f"{source} score {score} out of range"
