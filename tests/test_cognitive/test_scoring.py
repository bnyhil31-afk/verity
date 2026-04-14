"""
tests.test_cognitive.test_scoring
===================================
Tests for ImportanceScorer — composite importance scoring with
prediction error as a dopamine/norepinephrine proxy.

Coverage:
- surprise() returns 1.0 for completely orthogonal embedding
- surprise() returns ~0.0 for identical embedding
- surprise() returns 0.5 when no centroid initialized
- recency_decay() decreases as hours increase
- recency_decay() returns ~1.0 for a just-accessed memory
- recency_decay() returns < 0.5 for a memory accessed 200 hours ago
  (0.995^200 ≈ 0.37; the half-life of the 0.995^hours formula is ~138 h)
- reference_boost(0)  == 0.5  (1.0 / 2.0, normalized)
- reference_boost(10) == 1.0  (2.0 / 2.0, capped then normalized)
- reference_boost() increases with access_count
- score() returns value in [0.0, 1.0] with and without embedding
- score() with query_embedding biases result toward similar entries
- update_centroid() shifts the centroid toward the new embedding
- update_centroid() keeps the centroid at unit length
- signal_score() returns 0.5 for unknown memory_id (no signals)
- record_signal("correction") pulls signal_score below 0.5
- record_signal("recall") pushes signal_score above 0.5
- get_signals() returns empty list for unknown id
- record_signal() + get_signals() round-trip

Numpy-dependent tests are skipped automatically when numpy is not installed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from verity.cognitive.scoring import ImportanceScorer
from verity.cognitive.types import (
    ConfidenceTier,
    ImportanceWeights,
    MemoryEntry,
    MemoryTier,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    *,
    embedding: list[float] | None = None,
    access_count: int = 0,
    last_accessed: datetime | None = None,
) -> MemoryEntry:
    now = datetime.now(UTC)
    return MemoryEntry(
        memory_id="test-id",
        content="test content",
        user_id="test-user",
        tier=MemoryTier.FAST,
        confidence_tier=ConfidenceTier.LABILE,
        importance=0.5,
        strength=1.0,
        created_at=now,
        last_accessed=last_accessed if last_accessed is not None else now,
        access_count=access_count,
        source_count=1,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# surprise()
# ---------------------------------------------------------------------------


class TestSurprise:
    def test_no_centroid_returns_half(self) -> None:
        """Before any update_centroid call, surprise must be 0.5."""
        scorer = ImportanceScorer()
        result = scorer.surprise([1.0, 0.0, 0.0])
        assert result == pytest.approx(0.5)

    def test_orthogonal_embedding_returns_one(self) -> None:
        """Embedding perpendicular to centroid → cosine=0 → surprise=1.0."""
        pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        scorer.update_centroid([1.0, 0.0, 0.0])
        result = scorer.surprise([0.0, 1.0, 0.0])
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_identical_embedding_returns_zero(self) -> None:
        """Embedding identical to centroid → cosine=1 → surprise=0.0."""
        pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        scorer.update_centroid([1.0, 0.0, 0.0])
        result = scorer.surprise([1.0, 0.0, 0.0])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_no_numpy_returns_half(self) -> None:
        """When numpy is unavailable, surprise must degrade gracefully to 0.5."""
        from verity.cognitive import scoring as _scoring_mod

        original = _scoring_mod._HAS_NUMPY
        try:
            _scoring_mod._HAS_NUMPY = False
            scorer = ImportanceScorer()
            # Even if centroid was somehow set, _HAS_NUMPY=False forces 0.5
            result = scorer.surprise([1.0, 0.0])
            assert result == pytest.approx(0.5)
        finally:
            _scoring_mod._HAS_NUMPY = original


# ---------------------------------------------------------------------------
# recency_decay()
# ---------------------------------------------------------------------------


class TestRecencyDecay:
    def test_decreases_as_hours_increase(self) -> None:
        scorer = ImportanceScorer()
        recent = scorer.recency_decay(datetime.now(UTC) - timedelta(hours=1))
        older = scorer.recency_decay(datetime.now(UTC) - timedelta(hours=48))
        assert recent > older

    def test_near_one_for_just_accessed(self) -> None:
        scorer = ImportanceScorer()
        result = scorer.recency_decay(datetime.now(UTC))
        assert result > 0.99

    def test_below_half_at_200_hours(self) -> None:
        """0.995^200 ≈ 0.367 — well below the 0.5 half-point."""
        scorer = ImportanceScorer()
        far_past = datetime.now(UTC) - timedelta(hours=200)
        result = scorer.recency_decay(far_past)
        assert result < 0.5

    def test_clamped_above_zero(self) -> None:
        """Very old memories must not go negative."""
        scorer = ImportanceScorer()
        ancient = datetime.now(UTC) - timedelta(days=3650)
        result = scorer.recency_decay(ancient)
        assert result >= 0.0

    def test_clamped_below_one_for_future(self) -> None:
        """A future timestamp (clipped to 0 hours) must not exceed 1.0."""
        scorer = ImportanceScorer()
        future = datetime.now(UTC) + timedelta(hours=10)
        result = scorer.recency_decay(future)
        assert result <= 1.0


# ---------------------------------------------------------------------------
# reference_boost()
# ---------------------------------------------------------------------------


class TestReferenceBoost:
    def test_zero_returns_half(self) -> None:
        scorer = ImportanceScorer()
        assert scorer.reference_boost(0) == pytest.approx(0.5)

    def test_ten_returns_one(self) -> None:
        scorer = ImportanceScorer()
        assert scorer.reference_boost(10) == pytest.approx(1.0)

    def test_increases_with_access_count(self) -> None:
        scorer = ImportanceScorer()
        assert scorer.reference_boost(5) > scorer.reference_boost(0)

    def test_caps_at_one_for_high_count(self) -> None:
        """Counts beyond 10 must not produce a value > 1.0."""
        scorer = ImportanceScorer()
        assert scorer.reference_boost(100) == pytest.approx(1.0)

    def test_intermediate_count(self) -> None:
        scorer = ImportanceScorer()
        # access_count=5: min(2.0, 1.5) / 2.0 = 0.75
        assert scorer.reference_boost(5) == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore:
    def test_in_range_no_embedding(self) -> None:
        scorer = ImportanceScorer()
        entry = _make_entry(embedding=None)
        result = scorer.score(entry)
        assert 0.0 <= result <= 1.0

    def test_in_range_with_embedding(self) -> None:
        pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        scorer.update_centroid([1.0, 0.0, 0.0])
        entry = _make_entry(embedding=[0.5, 0.5, 0.0])
        result = scorer.score(entry)
        assert 0.0 <= result <= 1.0

    def test_in_range_with_query_embedding(self) -> None:
        pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        entry = _make_entry(embedding=[1.0, 0.0, 0.0])
        result = scorer.score(entry, query_embedding=[1.0, 0.0, 0.0])
        assert 0.0 <= result <= 1.0

    def test_query_embedding_biases_similar_entry_higher(self) -> None:
        """A query-similar entry must score above a dissimilar one."""
        pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        now = datetime.now(UTC)

        similar = _make_entry(embedding=[1.0, 0.0, 0.0], last_accessed=now)
        dissimilar = _make_entry(embedding=[0.0, 1.0, 0.0], last_accessed=now)

        query = [1.0, 0.0, 0.0]
        assert scorer.score(similar, query_embedding=query) > scorer.score(
            dissimilar, query_embedding=query
        )

    def test_custom_weights_respected(self) -> None:
        """Changing weights must change the score."""
        scorer_default = ImportanceScorer()
        scorer_recency = ImportanceScorer(
            weights=ImportanceWeights(
                surprise_weight=0.0,
                recency_weight=1.0,
                reference_weight=0.0,
                relevance_weight=0.0,
            )
        )
        old_entry = _make_entry(
            embedding=None,
            last_accessed=datetime.now(UTC) - timedelta(hours=500),
        )
        # Recency-only scorer penalises old entries more harshly
        assert scorer_recency.score(old_entry) < scorer_default.score(old_entry)

    def test_no_embedding_weights_sum_to_one(self) -> None:
        """Redistributed weights must still yield a normalised result."""
        scorer = ImportanceScorer()
        # access_count=10 → reference_boost=1.0, just-accessed → recency≈1.0
        entry = _make_entry(embedding=None, access_count=10)
        result = scorer.score(entry)
        # With perfect recency and reference the score should be close to 1.0
        assert result > 0.9


# ---------------------------------------------------------------------------
# update_centroid()
# ---------------------------------------------------------------------------


class TestUpdateCentroid:
    def test_initialises_on_first_call(self) -> None:
        pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        assert scorer._centroid is None
        scorer.update_centroid([3.0, 4.0, 0.0])
        assert scorer._centroid is not None

    def test_shifts_toward_new_embedding(self) -> None:
        pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        scorer.update_centroid([1.0, 0.0, 0.0])
        scorer.update_centroid([0.0, 1.0, 0.0])
        # After blending toward [0, 1, 0] the y-component must be > 0
        assert scorer._centroid[1] > 0.0

    def test_centroid_is_unit_length_after_init(self) -> None:
        np = pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        scorer.update_centroid([3.0, 4.0, 0.0])  # non-unit input
        norm = float(np.linalg.norm(scorer._centroid))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_centroid_is_unit_length_after_multiple_updates(self) -> None:
        np = pytest.importorskip("numpy")
        scorer = ImportanceScorer()
        for vec in [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]]:
            scorer.update_centroid(vec)
        norm = float(np.linalg.norm(scorer._centroid))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_no_numpy_is_noop(self) -> None:
        """When numpy is unavailable, update_centroid must not raise."""
        from verity.cognitive import scoring as _scoring_mod

        original = _scoring_mod._HAS_NUMPY
        try:
            _scoring_mod._HAS_NUMPY = False
            scorer = ImportanceScorer()
            scorer.update_centroid([1.0, 0.0])  # must not raise
            assert scorer._centroid is None
        finally:
            _scoring_mod._HAS_NUMPY = original


# ---------------------------------------------------------------------------
# Signal tracking
# ---------------------------------------------------------------------------


class TestSignalScore:
    def test_no_signals_returns_half(self) -> None:
        scorer = ImportanceScorer()
        assert scorer.signal_score("unknown-id") == pytest.approx(0.5)

    def test_correction_pulls_below_half(self) -> None:
        scorer = ImportanceScorer()
        scorer.record_signal("mem1", "correction", weight=1.0)
        assert scorer.signal_score("mem1") < 0.5

    def test_recall_pushes_above_half(self) -> None:
        scorer = ImportanceScorer()
        scorer.record_signal("mem1", "recall", weight=1.0)
        assert scorer.signal_score("mem1") > 0.5

    def test_reference_pushes_above_half(self) -> None:
        scorer = ImportanceScorer()
        scorer.record_signal("mem1", "reference", weight=1.0)
        assert scorer.signal_score("mem1") > 0.5

    def test_dwell_pushes_above_half(self) -> None:
        scorer = ImportanceScorer()
        scorer.record_signal("mem1", "dwell", weight=1.0)
        assert scorer.signal_score("mem1") > 0.5

    def test_multiple_corrections_deeply_negative(self) -> None:
        scorer = ImportanceScorer()
        for _ in range(5):
            scorer.record_signal("mem1", "correction")
        assert scorer.signal_score("mem1") < 0.1

    def test_mixed_signals_recall_wins(self) -> None:
        """Strong recall should outweigh a single correction."""
        scorer = ImportanceScorer()
        scorer.record_signal("mem1", "correction", weight=1.0)
        for _ in range(5):
            scorer.record_signal("mem1", "recall", weight=1.0)
        assert scorer.signal_score("mem1") > 0.5

    def test_signal_score_in_range(self) -> None:
        scorer = ImportanceScorer()
        for signal_type in ("recall", "correction", "dwell", "reference"):
            scorer.record_signal("mem1", signal_type, weight=10.0)
        result = scorer.signal_score("mem1")
        assert 0.0 <= result <= 1.0


class TestGetSignals:
    def test_empty_for_unknown_id(self) -> None:
        scorer = ImportanceScorer()
        assert scorer.get_signals("unknown") == []

    def test_round_trip(self) -> None:
        scorer = ImportanceScorer()
        scorer.record_signal("mem1", "recall", 1.0)
        scorer.record_signal("mem1", "dwell", 0.5)
        signals = scorer.get_signals("mem1")
        assert len(signals) == 2
        assert ("recall", 1.0) in signals
        assert ("dwell", 0.5) in signals

    def test_returns_copy(self) -> None:
        """Mutating the returned list must not affect internal state."""
        scorer = ImportanceScorer()
        scorer.record_signal("mem1", "recall", 1.0)
        signals = scorer.get_signals("mem1")
        signals.clear()
        assert len(scorer.get_signals("mem1")) == 1

    def test_separate_memory_ids_do_not_cross(self) -> None:
        scorer = ImportanceScorer()
        scorer.record_signal("mem1", "recall", 1.0)
        scorer.record_signal("mem2", "correction", 1.0)
        assert scorer.get_signals("mem1") == [("recall", 1.0)]
        assert scorer.get_signals("mem2") == [("correction", 1.0)]


# ---------------------------------------------------------------------------
# ImportanceWeights default values
# ---------------------------------------------------------------------------


class TestImportanceWeights:
    def test_default_weights_sum_to_one(self) -> None:
        w = ImportanceWeights()
        total = w.surprise_weight + w.recency_weight + w.reference_weight + w.relevance_weight
        assert total == pytest.approx(1.0)
