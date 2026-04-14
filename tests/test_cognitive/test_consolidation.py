"""
tests.test_cognitive.test_consolidation
========================================
Tests for ConsolidationCycle — sleep phases decay / prune / abstract.

Coverage:
- decay_pass() reduces all strengths by decay_factor (no exemptions)
- prune_pass() removes entries below threshold, preserves IMMUTABLE/PROTECTED
- abstract_pass() clusters similar fast-buffer memories into slow abstractions
- Full run() returns accurate SleepCycleResult counts
- Cycle is safe to run on empty store
"""

from __future__ import annotations

import pytest

from verity.cognitive.consolidation import ConsolidationCycle
from verity.cognitive.scoring import ImportanceScorer
from verity.cognitive.store import DualSpeedStore
from verity.cognitive.types import ConfidenceTier, SleepCycleResult

# ---------------------------------------------------------------------------
# numpy availability — used to skip embedding-dependent tests
# ---------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_store(**kwargs) -> DualSpeedStore:
    kwargs.setdefault("path", ":memory:")
    kwargs.setdefault("embedding_model", "none")
    return DualSpeedStore(**kwargs)


def _make_cycle(store: DualSpeedStore, **kwargs) -> ConsolidationCycle:
    scorer = ImportanceScorer()
    return ConsolidationCycle(store=store, scorer=scorer, **kwargs)


def _set_strength(store: DualSpeedStore, memory_id: str, strength: float) -> None:
    """Directly set strength in whichever table holds the entry."""
    for table in ("fast_memories", "slow_memories"):
        store._conn.execute(
            f"UPDATE {table} SET strength = ? WHERE memory_id = ?",
            (strength, memory_id),
        )
    store._conn.commit()


def _set_confidence_tier(
    store: DualSpeedStore, memory_id: str, tier: ConfidenceTier
) -> None:
    """Directly set confidence_tier in whichever table holds the entry."""
    for table in ("fast_memories", "slow_memories"):
        store._conn.execute(
            f"UPDATE {table} SET confidence_tier = ? WHERE memory_id = ?",
            (str(tier), memory_id),
        )
    store._conn.commit()


def _get_strength(store: DualSpeedStore, memory_id: str) -> float | None:
    """Fetch strength directly from DB."""
    for table in ("fast_memories", "slow_memories"):
        cur = store._conn.cursor()
        cur.execute(
            f"SELECT strength FROM {table} WHERE memory_id = ?", (memory_id,)
        )
        row = cur.fetchone()
        if row is not None:
            return float(row[0])
    return None


def _normalize(v: list[float]) -> list[float]:
    """Return unit-length version of v (requires numpy)."""
    arr = np.array(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist()


# ---------------------------------------------------------------------------
# TestDecayPass
# ---------------------------------------------------------------------------


class TestDecayPass:
    def test_reduces_strength_by_factor(self):
        store = _make_store()
        e1 = store.add("memory one")
        e2 = store.add("memory two")
        e3 = store.add("memory three")
        _set_strength(store, e1.memory_id, 1.0)
        _set_strength(store, e2.memory_id, 1.0)
        _set_strength(store, e3.memory_id, 1.0)

        cycle = _make_cycle(store, decay_factor=0.90)
        cycle.decay_pass()

        assert _get_strength(store, e1.memory_id) == pytest.approx(0.90, abs=0.001)
        assert _get_strength(store, e2.memory_id) == pytest.approx(0.90, abs=0.001)
        assert _get_strength(store, e3.memory_id) == pytest.approx(0.90, abs=0.001)

    def test_returns_correct_count(self):
        store = _make_store()
        store.add("one")
        store.add("two")
        store.add("three")
        cycle = _make_cycle(store)
        count = cycle.decay_pass()
        assert count == 3

    def test_immutable_entry_is_also_decayed(self):
        """decay_pass() has no exemptions — IMMUTABLE decays too."""
        store = _make_store()
        entry = store.add("immutable memory")
        _set_strength(store, entry.memory_id, 0.1)
        _set_confidence_tier(store, entry.memory_id, ConfidenceTier.IMMUTABLE)

        cycle = _make_cycle(store, decay_factor=0.90)
        cycle.decay_pass()

        assert _get_strength(store, entry.memory_id) == pytest.approx(0.09, abs=0.001)

    def test_decays_slow_store_entries_too(self):
        store = _make_store()
        entry = store.add("will be promoted")
        store.promote(entry.memory_id)
        _set_strength(store, entry.memory_id, 1.0)

        cycle = _make_cycle(store, decay_factor=0.90)
        count = cycle.decay_pass()

        assert count == 1
        assert _get_strength(store, entry.memory_id) == pytest.approx(0.90, abs=0.001)

    def test_decays_both_fast_and_slow(self):
        store = _make_store()
        store.add("fast entry")
        s = store.add("slow entry")
        store.promote(s.memory_id)

        cycle = _make_cycle(store)
        count = cycle.decay_pass()
        assert count == 2

    def test_empty_store_returns_zero(self):
        store = _make_store()
        cycle = _make_cycle(store)
        assert cycle.decay_pass() == 0


# ---------------------------------------------------------------------------
# TestPrunePass
# ---------------------------------------------------------------------------


class TestPrunePass:
    def test_entry_below_threshold_is_deleted(self):
        store = _make_store()
        entry = store.add("weak memory")
        _set_strength(store, entry.memory_id, 0.04)

        cycle = _make_cycle(store, prune_threshold=0.05)
        cycle.prune_pass()

        assert store.get(entry.memory_id) is None

    def test_entry_above_threshold_survives(self):
        store = _make_store()
        entry = store.add("strong memory")
        _set_strength(store, entry.memory_id, 0.06)

        cycle = _make_cycle(store, prune_threshold=0.05)
        cycle.prune_pass()

        assert store.get(entry.memory_id) is not None

    def test_entry_at_exact_threshold_survives(self):
        """Strict less-than: strength == prune_threshold is NOT pruned."""
        store = _make_store()
        entry = store.add("boundary memory")
        _set_strength(store, entry.memory_id, 0.05)

        cycle = _make_cycle(store, prune_threshold=0.05)
        cycle.prune_pass()

        assert store.get(entry.memory_id) is not None

    def test_immutable_not_deleted_even_below_threshold(self):
        store = _make_store()
        entry = store.add("immutable weak memory")
        _set_strength(store, entry.memory_id, 0.01)
        _set_confidence_tier(store, entry.memory_id, ConfidenceTier.IMMUTABLE)

        cycle = _make_cycle(store, prune_threshold=0.05)
        cycle.prune_pass()

        assert store.get(entry.memory_id) is not None

    def test_protected_not_deleted_even_below_threshold(self):
        store = _make_store()
        entry = store.add("protected weak memory")
        _set_strength(store, entry.memory_id, 0.01)
        _set_confidence_tier(store, entry.memory_id, ConfidenceTier.PROTECTED)

        cycle = _make_cycle(store, prune_threshold=0.05)
        cycle.prune_pass()

        assert store.get(entry.memory_id) is not None

    def test_labile_entry_below_threshold_is_deleted(self):
        store = _make_store()
        entry = store.add("labile weak memory")
        _set_strength(store, entry.memory_id, 0.01)
        _set_confidence_tier(store, entry.memory_id, ConfidenceTier.LABILE)

        cycle = _make_cycle(store, prune_threshold=0.05)
        cycle.prune_pass()

        assert store.get(entry.memory_id) is None

    def test_returns_correct_deleted_count(self):
        store = _make_store()
        e1 = store.add("weak 1")
        e2 = store.add("weak 2")
        e3 = store.add("strong")
        _set_strength(store, e1.memory_id, 0.01)
        _set_strength(store, e2.memory_id, 0.02)
        _set_strength(store, e3.memory_id, 0.99)

        cycle = _make_cycle(store, prune_threshold=0.05)
        count = cycle.prune_pass()
        assert count == 2

    def test_prunes_from_slow_store_too(self):
        store = _make_store()
        entry = store.add("weak in slow")
        store.promote(entry.memory_id)
        _set_strength(store, entry.memory_id, 0.01)

        cycle = _make_cycle(store, prune_threshold=0.05)
        count = cycle.prune_pass()

        assert count == 1
        assert store.get(entry.memory_id) is None

    def test_empty_store_returns_zero(self):
        store = _make_store()
        cycle = _make_cycle(store)
        assert cycle.prune_pass() == 0


# ---------------------------------------------------------------------------
# TestAbstractPass
# ---------------------------------------------------------------------------


class TestAbstractPass:
    def test_without_embeddings_returns_zero_no_error(self):
        """No embeddings → entries_with_emb is empty → returns 0 gracefully."""
        store = _make_store(embedding_model="none")
        store.add("memory 1")
        store.add("memory 2")
        store.add("memory 3")

        cycle = _make_cycle(store)
        result = cycle.abstract_pass()
        assert result == 0

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required for embedding tests")
    def test_cluster_of_four_creates_one_abstraction(self):
        """4 similar + 1 orthogonal → 1 abstraction, 1 fast entry remains."""
        store = _make_store(embedding_model="none")
        scorer = ImportanceScorer()
        cycle = ConsolidationCycle(
            store=store,
            scorer=scorer,
            cluster_min_size=3,
            similarity_threshold=0.85,
        )

        # Pre-normalised 4-D vectors: v1-v4 are mutually similar (cos >= 0.85)
        # v5 is orthogonal to all of them (cos = 0).
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = _normalize([0.99, 0.14, 0.0, 0.0])
        v3 = _normalize([0.98, 0.20, 0.0, 0.0])
        v4 = _normalize([0.96, 0.28, 0.0, 0.0])
        v5 = [0.0, 1.0, 0.0, 0.0]

        store.add("memory 1", importance=0.6, _embedding=v1)
        store.add("memory 2", importance=0.7, _embedding=v2)
        store.add("memory 3", importance=0.8, _embedding=v3)
        store.add("memory 4", importance=0.9, _embedding=v4)
        store.add("orthogonal memory", importance=0.3, _embedding=v5)

        result = cycle.abstract_pass()

        assert result == 1
        stats = store.stats()
        assert stats["slow_count"] == 1
        assert stats["fast_count"] == 1  # only orthogonal entry remains

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required for embedding tests")
    def test_abstraction_has_correct_metadata(self):
        """Abstraction entry has correct metadata fields."""
        store = _make_store(embedding_model="none")
        cycle = _make_cycle(store, cluster_min_size=3, similarity_threshold=0.85)

        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = _normalize([0.99, 0.14, 0.0, 0.0])
        v3 = _normalize([0.98, 0.20, 0.0, 0.0])
        v4 = _normalize([0.96, 0.28, 0.0, 0.0])

        e1 = store.add("memory 1", importance=0.6, _embedding=v1)
        e2 = store.add("memory 2", importance=0.7, _embedding=v2)
        e3 = store.add("memory 3", importance=0.8, _embedding=v3)
        e4 = store.add("memory 4", importance=0.9, _embedding=v4)

        cycle.abstract_pass()

        slow = store.all_slow()
        assert len(slow) == 1
        abstraction = slow[0]

        assert "abstracted_from" in abstraction.metadata
        assert abstraction.metadata["cluster_size"] == 4
        assert len(abstraction.metadata["abstracted_from"]) == 4

        # Verify all original IDs appear in the abstraction metadata
        original_ids = {e1.memory_id, e2.memory_id, e3.memory_id, e4.memory_id}
        assert set(abstraction.metadata["abstracted_from"]) == original_ids

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required for embedding tests")
    def test_abstraction_confidence_tier_is_modifiable(self):
        store = _make_store(embedding_model="none")
        cycle = _make_cycle(store, cluster_min_size=3, similarity_threshold=0.85)

        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = _normalize([0.99, 0.14, 0.0, 0.0])
        v3 = _normalize([0.98, 0.20, 0.0, 0.0])
        v4 = _normalize([0.96, 0.28, 0.0, 0.0])

        store.add("memory 1", importance=0.6, _embedding=v1)
        store.add("memory 2", importance=0.7, _embedding=v2)
        store.add("memory 3", importance=0.8, _embedding=v3)
        store.add("memory 4", importance=0.9, _embedding=v4)

        cycle.abstract_pass()

        abstraction = store.all_slow()[0]
        assert abstraction.confidence_tier == ConfidenceTier.MODIFIABLE
        assert abstraction.source_count == 4

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required for embedding tests")
    def test_cluster_below_min_size_no_abstraction(self):
        """Cluster of 2 entries (below cluster_min_size=3) → no abstraction."""
        store = _make_store(embedding_model="none")
        cycle = _make_cycle(store, cluster_min_size=3, similarity_threshold=0.85)

        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = _normalize([0.99, 0.14, 0.0, 0.0])

        store.add("memory 1", importance=0.6, _embedding=v1)
        store.add("memory 2", importance=0.7, _embedding=v2)

        result = cycle.abstract_pass()

        assert result == 0
        assert store.stats()["slow_count"] == 0
        assert store.stats()["fast_count"] == 2

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required for embedding tests")
    def test_orthogonal_entry_stays_in_fast(self):
        """The orthogonal entry should remain in fast store after abstraction."""
        store = _make_store(embedding_model="none")
        cycle = _make_cycle(store, cluster_min_size=3, similarity_threshold=0.85)

        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = _normalize([0.99, 0.14, 0.0, 0.0])
        v3 = _normalize([0.98, 0.20, 0.0, 0.0])
        v4 = _normalize([0.96, 0.28, 0.0, 0.0])
        v5 = [0.0, 1.0, 0.0, 0.0]

        store.add("memory 1", importance=0.6, _embedding=v1)
        store.add("memory 2", importance=0.7, _embedding=v2)
        store.add("memory 3", importance=0.8, _embedding=v3)
        store.add("memory 4", importance=0.9, _embedding=v4)
        e5 = store.add("orthogonal memory", importance=0.3, _embedding=v5)

        cycle.abstract_pass()

        fast = store.all_fast()
        assert len(fast) == 1
        assert fast[0].memory_id == e5.memory_id

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required for embedding tests")
    def test_merged_count_tracked_correctly(self):
        """_last_merged_count == total cluster members deleted."""
        store = _make_store(embedding_model="none")
        cycle = _make_cycle(store, cluster_min_size=3, similarity_threshold=0.85)

        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = _normalize([0.99, 0.14, 0.0, 0.0])
        v3 = _normalize([0.98, 0.20, 0.0, 0.0])
        v4 = _normalize([0.96, 0.28, 0.0, 0.0])

        store.add("m1", importance=0.6, _embedding=v1)
        store.add("m2", importance=0.7, _embedding=v2)
        store.add("m3", importance=0.8, _embedding=v3)
        store.add("m4", importance=0.9, _embedding=v4)

        cycle.abstract_pass()

        assert cycle._last_merged_count == 4

    @pytest.mark.skipif(not _HAS_NUMPY, reason="numpy required for embedding tests")
    def test_summarizer_called_with_cluster_contents(self):
        """When a summarizer is provided, it receives the cluster contents."""
        store = _make_store(embedding_model="none")
        received: list[list[str]] = []

        def mock_summarizer(texts: list[str]) -> str:
            received.append(texts)
            return "summarized content"

        cycle = ConsolidationCycle(
            store=store,
            scorer=ImportanceScorer(),
            cluster_min_size=3,
            similarity_threshold=0.85,
            summarizer=mock_summarizer,
        )

        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = _normalize([0.99, 0.14, 0.0, 0.0])
        v3 = _normalize([0.98, 0.20, 0.0, 0.0])

        store.add("text one", importance=0.6, _embedding=v1)
        store.add("text two", importance=0.7, _embedding=v2)
        store.add("text three", importance=0.8, _embedding=v3)

        cycle.abstract_pass()

        assert len(received) == 1
        assert set(received[0]) == {"text one", "text two", "text three"}

        slow = store.all_slow()
        assert slow[0].content == "summarized content"


# ---------------------------------------------------------------------------
# TestFullRun
# ---------------------------------------------------------------------------


class TestFullRun:
    def test_empty_store_returns_all_zeros(self):
        store = _make_store()
        cycle = _make_cycle(store)
        result = cycle.run()

        assert isinstance(result, SleepCycleResult)
        assert result.memories_decayed == 0
        assert result.memories_pruned == 0
        assert result.memories_merged == 0
        assert result.abstractions_created == 0

    def test_run_with_three_entries_reports_decayed(self):
        store = _make_store()
        store.add("one")
        store.add("two")
        store.add("three")

        cycle = _make_cycle(store)
        result = cycle.run()

        assert result.memories_decayed == 3

    def test_duration_is_positive(self):
        store = _make_store()
        cycle = _make_cycle(store)
        result = cycle.run()
        assert result.duration_seconds >= 0.0

    def test_cycle_timestamp_is_set(self):
        from datetime import UTC, datetime

        store = _make_store()
        cycle = _make_cycle(store)
        before = datetime.now(UTC)
        result = cycle.run()
        after = datetime.now(UTC)

        assert result.cycle_timestamp is not None
        assert before <= result.cycle_timestamp <= after

    def test_run_twice_does_not_crash(self):
        store = _make_store()
        store.add("memory one")
        store.add("memory two")
        cycle = _make_cycle(store)
        cycle.run()
        cycle.run()  # must not raise

    def test_run_pruned_count_matches_deleted_entries(self):
        store = _make_store()
        e1 = store.add("weak one")
        e2 = store.add("weak two")
        store.add("strong three")
        _set_strength(store, e1.memory_id, 0.01)
        _set_strength(store, e2.memory_id, 0.02)
        # strong three keeps default strength 1.0, which after decay is 0.9

        cycle = _make_cycle(store, prune_threshold=0.05)
        result = cycle.run()

        # After decay_pass: weak entries 0.01*0.9=0.009, 0.02*0.9=0.018 → both < 0.05
        assert result.memories_pruned == 2

    def test_sleepresult_type(self):
        store = _make_store()
        cycle = _make_cycle(store)
        result = cycle.run()
        assert isinstance(result, SleepCycleResult)

    def test_immutable_entry_survives_full_run(self):
        """IMMUTABLE entry: decays (strength decreases) but is NOT pruned."""
        store = _make_store()
        entry = store.add("immutable memory")
        # Set strength just below prune_threshold so it would be pruned if not exempt
        _set_strength(store, entry.memory_id, 0.01)
        _set_confidence_tier(store, entry.memory_id, ConfidenceTier.IMMUTABLE)

        cycle = _make_cycle(store, prune_threshold=0.05)
        result = cycle.run()

        # Entry is decayed but not pruned
        assert store.get(entry.memory_id) is not None
        assert result.memories_pruned == 0
        # decay_pass touched it
        assert result.memories_decayed == 1
        # strength decayed to 0.01 * 0.90 = 0.009
        assert _get_strength(store, entry.memory_id) == pytest.approx(0.009, abs=0.001)
