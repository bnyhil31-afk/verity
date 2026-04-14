"""
tests.test_cognitive.test_types
================================
Contract tests for verity.cognitive.types.

Covers:
- All dataclasses construct correctly
- bayesian_confidence returns alpha / (alpha + beta)
- confidence_interval_width narrows as alpha + beta grows
- ConfidenceTier serializes correctly as string
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from verity.cognitive.types import (
    ConfidenceTier,
    ImportanceWeights,
    MemoryEntry,
    MemoryTier,
    RetrievalResult,
    SleepCycleResult,
    TemporalModelType,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(UTC)


def _make_entry(**overrides) -> MemoryEntry:
    defaults = dict(
        memory_id="test-uuid-001",
        content="The sky is blue.",
        user_id="default",
        tier=MemoryTier.FAST,
        confidence_tier=ConfidenceTier.MODIFIABLE,
        importance=0.5,
        strength=0.8,
        created_at=_now(),
        last_accessed=_now(),
        access_count=1,
        source_count=1,
        alpha=2.0,
        beta=1.0,
    )
    defaults.update(overrides)
    return MemoryEntry(**defaults)


# ── MemoryTier ─────────────────────────────────────────────────────────────────

class TestMemoryTier:
    def test_fast_value(self):
        assert MemoryTier.FAST == "fast"

    def test_slow_value(self):
        assert MemoryTier.SLOW == "slow"

    def test_is_str(self):
        assert isinstance(MemoryTier.FAST, str)

    def test_all_members(self):
        members = {m.value for m in MemoryTier}
        assert members == {"fast", "slow"}


# ── ConfidenceTier ─────────────────────────────────────────────────────────────

class TestConfidenceTier:
    def test_immutable_serializes_as_string(self):
        assert ConfidenceTier.IMMUTABLE == "immutable"
        assert str(ConfidenceTier.IMMUTABLE) == "immutable"

    def test_protected_serializes_as_string(self):
        assert ConfidenceTier.PROTECTED == "protected"
        assert str(ConfidenceTier.PROTECTED) == "protected"

    def test_modifiable_serializes_as_string(self):
        assert ConfidenceTier.MODIFIABLE == "modifiable"
        assert str(ConfidenceTier.MODIFIABLE) == "modifiable"

    def test_labile_serializes_as_string(self):
        assert ConfidenceTier.LABILE == "labile"
        assert str(ConfidenceTier.LABILE) == "labile"

    def test_all_are_str_instances(self):
        for tier in ConfidenceTier:
            assert isinstance(tier, str)

    def test_four_members(self):
        assert len(list(ConfidenceTier)) == 4

    def test_roundtrip_from_value(self):
        for tier in ConfidenceTier:
            assert ConfidenceTier(tier.value) is tier


# ── TemporalModelType ──────────────────────────────────────────────────────────

class TestTemporalModelType:
    def test_exponential_value(self):
        assert TemporalModelType.EXPONENTIAL == "exponential"

    def test_renewal_value(self):
        assert TemporalModelType.RENEWAL == "renewal"

    def test_hawkes_value(self):
        assert TemporalModelType.HAWKES == "hawkes"

    def test_all_are_str_instances(self):
        for model in TemporalModelType:
            assert isinstance(model, str)


# ── MemoryEntry ────────────────────────────────────────────────────────────────

class TestMemoryEntry:
    def test_basic_construction(self):
        entry = _make_entry()
        assert entry.memory_id == "test-uuid-001"
        assert entry.content == "The sky is blue."
        assert entry.user_id == "default"
        assert entry.tier is MemoryTier.FAST
        assert entry.confidence_tier is ConfidenceTier.MODIFIABLE

    def test_defaults(self):
        entry = _make_entry()
        assert entry.alpha == 2.0
        assert entry.beta == 1.0
        assert entry.metadata == {}
        assert entry.embedding is None

    def test_metadata_not_shared(self):
        a = _make_entry()
        b = _make_entry()
        a.metadata["x"] = 1
        assert "x" not in b.metadata

    def test_embedding_can_be_set(self):
        entry = _make_entry(embedding=[0.1, 0.2, 0.3])
        assert entry.embedding == [0.1, 0.2, 0.3]

    # ── bayesian_confidence ────────────────────────────────────────────────────

    def test_bayesian_confidence_formula(self):
        entry = _make_entry(alpha=3.0, beta=1.0)
        assert entry.bayesian_confidence == pytest.approx(3.0 / 4.0)

    def test_bayesian_confidence_equal_alpha_beta(self):
        entry = _make_entry(alpha=5.0, beta=5.0)
        assert entry.bayesian_confidence == pytest.approx(0.5)

    def test_bayesian_confidence_high(self):
        entry = _make_entry(alpha=19.0, beta=1.0)
        assert entry.bayesian_confidence == pytest.approx(0.95)

    def test_bayesian_confidence_low(self):
        entry = _make_entry(alpha=1.0, beta=3.0)
        assert entry.bayesian_confidence == pytest.approx(0.25)

    def test_bayesian_confidence_default_alpha_beta(self):
        # Default: alpha=2.0, beta=1.0 → 2/3 ≈ 0.667
        entry = _make_entry()
        assert entry.bayesian_confidence == pytest.approx(2.0 / 3.0)

    # ── confidence_interval_width ──────────────────────────────────────────────

    def test_interval_width_formula(self):
        entry = _make_entry(alpha=2.0, beta=2.0)
        n = 4.0
        p = 0.5
        expected = 2 * 1.96 * math.sqrt(p * (1 - p) / n)
        assert entry.confidence_interval_width == pytest.approx(expected)

    def test_interval_width_narrows_as_n_grows(self):
        """More evidence (higher alpha+beta) → narrower interval."""
        small = _make_entry(alpha=2.0, beta=2.0)      # n=4
        medium = _make_entry(alpha=5.0, beta=5.0)     # n=10
        large = _make_entry(alpha=50.0, beta=50.0)    # n=100

        assert large.confidence_interval_width < medium.confidence_interval_width
        assert medium.confidence_interval_width < small.confidence_interval_width

    def test_interval_width_positive(self):
        entry = _make_entry(alpha=3.0, beta=1.0)
        assert entry.confidence_interval_width > 0

    def test_interval_width_near_zero_for_extreme_confidence(self):
        """Very high n with balanced p → small but non-zero width."""
        entry = _make_entry(alpha=500.0, beta=500.0)
        assert entry.confidence_interval_width < 0.1


# ── ImportanceWeights ──────────────────────────────────────────────────────────

class TestImportanceWeights:
    def test_default_construction(self):
        w = ImportanceWeights()
        assert w.surprise_weight == pytest.approx(0.35)
        assert w.recency_weight == pytest.approx(0.30)
        assert w.reference_weight == pytest.approx(0.20)
        assert w.relevance_weight == pytest.approx(0.15)

    def test_weights_sum_to_one(self):
        w = ImportanceWeights()
        total = (
            w.surprise_weight
            + w.recency_weight
            + w.reference_weight
            + w.relevance_weight
        )
        assert total == pytest.approx(1.0)

    def test_custom_weights(self):
        w = ImportanceWeights(surprise_weight=0.5, recency_weight=0.5,
                              reference_weight=0.0, relevance_weight=0.0)
        assert w.surprise_weight == pytest.approx(0.5)
        assert w.reference_weight == pytest.approx(0.0)


# ── SleepCycleResult ───────────────────────────────────────────────────────────

class TestSleepCycleResult:
    def test_construction(self):
        ts = _now()
        result = SleepCycleResult(
            memories_decayed=10,
            memories_pruned=3,
            memories_merged=2,
            abstractions_created=1,
            duration_seconds=0.42,
            cycle_timestamp=ts,
        )
        assert result.memories_decayed == 10
        assert result.memories_pruned == 3
        assert result.memories_merged == 2
        assert result.abstractions_created == 1
        assert result.duration_seconds == pytest.approx(0.42)
        assert result.cycle_timestamp is ts

    def test_zero_result(self):
        result = SleepCycleResult(
            memories_decayed=0,
            memories_pruned=0,
            memories_merged=0,
            abstractions_created=0,
            duration_seconds=0.0,
            cycle_timestamp=_now(),
        )
        assert result.memories_decayed == 0


# ── RetrievalResult ────────────────────────────────────────────────────────────

class TestRetrievalResult:
    def test_construction(self):
        entry = _make_entry()
        result = RetrievalResult(memory=entry, score=0.87, position=1)
        assert result.memory is entry
        assert result.score == pytest.approx(0.87)
        assert result.position == 1

    def test_position_is_one_indexed(self):
        entry = _make_entry()
        r = RetrievalResult(memory=entry, score=0.5, position=1)
        assert r.position >= 1

    def test_multiple_results_distinct_positions(self):
        entries = [_make_entry(memory_id=f"id-{i}") for i in range(3)]
        results = [
            RetrievalResult(memory=e, score=1.0 - i * 0.1, position=i + 1)
            for i, e in enumerate(entries)
        ]
        positions = [r.position for r in results]
        assert positions == [1, 2, 3]
