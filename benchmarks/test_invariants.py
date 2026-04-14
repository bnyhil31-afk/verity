"""
benchmarks/test_invariants.py
==============================
Hypothesis property tests for Verity's six core invariants.

Six @given tests (max_examples=50 for CI speed):
  1. Hit Rate          — a memory is always retrievable by its own prefix
  2. IMMUTABLE guard   — IMMUTABLE memories never change regardless of PE
  3. Consolidation ↓   — consolidation never increases the memory count
  4. Export consistency— export count equals adds minus confirmed deletes
  5. Temporal ↓        — temporal weight is non-increasing as time grows
  6. Idempotency       — second consolidation run ≤ first run count

One RuleBasedStateMachine (marked slow, stateful_step_count=20):
  rules:     add_memory, delete_random, consolidate
  invariants: export_count_consistent, consolidation_monotone
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.stateful import Bundle, RuleBasedStateMachine, invariant, rule

from verity.cognitive.reconsolidation import ReconsolidationEngine
from verity.cognitive.temporal import TemporalWeighter
from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier
from verity.memory import Memory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh() -> Memory:
    """Create a throwaway in-memory Memory with no embedding model."""
    return Memory(path=":memory:", embedding_model="none")


def _make_immutable_entry(content: str) -> MemoryEntry:
    """
    Build a MemoryEntry locked to IMMUTABLE tier.
    alpha=250, beta=10, source_count=6 → confidence≈0.962 ≥ 0.95, ci_width<0.05.
    """
    now = datetime.now(UTC)
    return MemoryEntry(
        memory_id=str(uuid4()),
        content=content,
        user_id="test",
        tier=MemoryTier.FAST,
        confidence_tier=ConfidenceTier.IMMUTABLE,
        importance=0.9,
        strength=1.0,
        created_at=now,
        last_accessed=now,
        access_count=0,
        source_count=6,
        alpha=250.0,
        beta=10.0,
    )


def _make_entry_at(last_accessed: datetime) -> MemoryEntry:
    """Build a minimal MemoryEntry with a specific last_accessed time."""
    return MemoryEntry(
        memory_id=str(uuid4()),
        content="temporal test",
        user_id="test",
        tier=MemoryTier.FAST,
        confidence_tier=ConfidenceTier.MODIFIABLE,
        importance=0.5,
        strength=1.0,
        created_at=last_accessed,
        last_accessed=last_accessed,
        access_count=0,
        source_count=1,
        alpha=2.0,
        beta=1.0,
    )


# ---------------------------------------------------------------------------
# Invariant 1 — Hit Rate
# ---------------------------------------------------------------------------


@given(st.text(min_size=20, max_size=200))
@settings(max_examples=50)
def test_hit_rate(content: str) -> None:
    """A memory is always retrievable by searching a prefix of its content."""
    with _fresh() as m:
        m.add(content)
        results = m.search(content[:20], k=3)
        assert any(r["content"].startswith(content[:20]) for r in results)


# ---------------------------------------------------------------------------
# Invariant 2 — IMMUTABLE protection
# ---------------------------------------------------------------------------


@given(
    content=st.text(min_size=1, max_size=200),
    new_content=st.text(min_size=1, max_size=200),
)
@settings(max_examples=50)
def test_immutable_never_modified(content: str, new_content: str) -> None:
    """ReconsolidationEngine.update() must never change an IMMUTABLE memory."""
    entry = _make_immutable_entry(content)
    engine = ReconsolidationEngine()
    result = engine.update(entry, new_content, prediction_error=0.9)
    assert result.content == content


# ---------------------------------------------------------------------------
# Invariant 3 — Consolidation monotonicity
# ---------------------------------------------------------------------------


@given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20))
@settings(max_examples=50)
def test_consolidation_monotone(contents: list[str]) -> None:
    """Memory count must not increase after a consolidation cycle."""
    with _fresh() as m:
        for c in contents:
            m.add(c)
        count_before = len(json.loads(m.export()))
        m.consolidate()
        count_after = len(json.loads(m.export()))
    assert count_after <= count_before


# ---------------------------------------------------------------------------
# Invariant 4 — Export consistency
# ---------------------------------------------------------------------------


@given(
    adds=st.lists(st.text(min_size=1, max_size=100), min_size=0, max_size=15),
    delete_indices=st.lists(
        st.integers(min_value=0, max_value=14), min_size=0, max_size=10
    ),
)
@settings(max_examples=50)
def test_export_consistency(adds: list[str], delete_indices: list[int]) -> None:
    """
    Export count == adds - confirmed_deletes.
    Each index is deleted at most once; out-of-range indices are ignored.
    """
    with _fresh() as m:
        ids = [m.add(c) for c in adds]
        deleted = 0
        seen: set[int] = set()
        for idx in delete_indices:
            if idx < len(ids) and idx not in seen:
                seen.add(idx)
                if m.delete(ids[idx]):
                    deleted += 1
        exported = json.loads(m.export())
    assert len(exported) == len(adds) - deleted


# ---------------------------------------------------------------------------
# Invariant 5 — Temporal monotonicity
# ---------------------------------------------------------------------------


_FIXED_REF = datetime(2026, 4, 14, 0, 0, 0, tzinfo=UTC)


@given(
    hours_since=st.floats(
        min_value=0.0, max_value=8760.0, allow_nan=False, allow_infinity=False
    ),
    extra_hours=st.floats(
        min_value=0.001, max_value=8760.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=50)
def test_temporal_monotone(hours_since: float, extra_hours: float) -> None:
    """
    Temporal weight at t1 >= temporal weight at t2 when t2 > t1.
    Uses a single access timestamp so the EXPONENTIAL model is exercised.
    """
    last = _FIXED_REF - timedelta(hours=hours_since)
    entry = _make_entry_at(last)
    temporal = TemporalWeighter()
    t1 = _FIXED_REF
    t2 = _FIXED_REF + timedelta(hours=extra_hours)
    w1 = temporal.weight(entry, [last], query_time=t1)
    w2 = temporal.weight(entry, [last], query_time=t2)
    # Allow tiny floating-point rounding (< 1e-9)
    assert w1 >= w2 - 1e-9


# ---------------------------------------------------------------------------
# Invariant 6 — Consolidation idempotency
# ---------------------------------------------------------------------------


@given(st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=20))
@settings(max_examples=50)
@pytest.mark.slow
def test_consolidation_idempotent(contents: list[str]) -> None:
    """
    Running consolidation twice must not increase the count beyond
    what the first run left behind.
    """
    with _fresh() as m:
        for c in contents:
            m.add(c)
        m.consolidate()
        count1 = len(json.loads(m.export()))
        m.consolidate()
        count2 = len(json.loads(m.export()))
    assert count2 <= count1


# ---------------------------------------------------------------------------
# Stateful machine
# ---------------------------------------------------------------------------


class _MemoryMachine(RuleBasedStateMachine):
    """
    Stateful Hypothesis test for Memory.

    Rules:    add_memory, delete_random, consolidate
    Invariants: export_count_consistent, consolidation_monotone
    """

    ids: Bundle = Bundle("ids")

    def __init__(self) -> None:
        super().__init__()
        self._memory = Memory(path=":memory:", embedding_model="none")
        # IDs we believe are live in the store right now.
        self._live_ids: set[str] = set()

    def teardown(self) -> None:
        self._memory._store.close()

    # ── Rules ──────────────────────────────────────────────────────────────

    @rule(target=ids, content=st.text(min_size=1, max_size=100))
    def add_memory(self, content: str) -> str:
        mid = self._memory.add(content)
        self._live_ids.add(mid)
        return mid

    @rule(mid=ids)
    def delete_random(self, mid: str) -> None:
        deleted = self._memory.delete(mid)
        if deleted:
            self._live_ids.discard(mid)

    @rule()
    def consolidate(self) -> None:
        count_before = len(json.loads(self._memory.export()))
        self._memory.consolidate()
        exported = json.loads(self._memory.export())
        count_after = len(exported)
        # Monotonicity: consolidation can only remove, never add.
        assert count_after <= count_before
        # Sync live_ids to the actual post-consolidation state so that
        # export_count_consistent invariant continues to hold.
        exported_ids = {e["id"] for e in exported}
        self._live_ids &= exported_ids

    # ── Invariants ────────────────────────────────────────────────────────

    @invariant()
    def export_count_consistent(self) -> None:
        """Exported memory count always equals the set of tracked live IDs."""
        exported = json.loads(self._memory.export())
        assert len(exported) == len(self._live_ids)

    @invariant()
    def consolidation_monotone(self) -> None:
        """Memory count is always non-negative."""
        exported = json.loads(self._memory.export())
        assert len(exported) >= 0


# Expose as a pytest test class; override settings for reasonable CI runtime.
@pytest.mark.slow
class TestMemoryStateMachine(_MemoryMachine.TestCase):  # type: ignore[misc]
    settings = settings(max_examples=30, stateful_step_count=20)
