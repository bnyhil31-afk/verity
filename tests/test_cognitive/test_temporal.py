"""
tests.test_cognitive.test_temporal
====================================
Tests for verity.cognitive.temporal.TemporalWeighter.

Covers:
- model_for() returns correct TemporalModelType for all boundary counts
- exponential_weight() numeric properties and monotonicity
- renewal_weight() with scipy: high weight when recent, low when overdue
- renewal_weight() graceful fallbacks (no scipy, zero variance)
- hawkes_weight() in-range and staleness
- weight() full fallback chain for all input sizes
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta

import pytest

from verity.cognitive.temporal import TemporalWeighter
from verity.cognitive.types import (
    ConfidenceTier,
    MemoryEntry,
    MemoryTier,
    TemporalModelType,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_timestamps(
    n: int,
    spacing_hours: float = 168.0,
    last_hours_ago: float = 24.0,
) -> list[datetime]:
    """Create n evenly-spaced timestamps ending last_hours_ago before now."""
    now = datetime.now(UTC)
    last = now - timedelta(hours=last_hours_ago)
    return [last - timedelta(hours=spacing_hours * (n - 1 - i)) for i in range(n)]


def _sample_entry(last_accessed_hours_ago: float = 24.0) -> MemoryEntry:
    """Return a minimal MemoryEntry for use in weight() tests."""
    now = datetime.now(UTC)
    return MemoryEntry(
        memory_id="test-id",
        content="test content",
        user_id="test-user",
        tier=MemoryTier.FAST,
        confidence_tier=ConfidenceTier.MODIFIABLE,
        importance=0.5,
        strength=0.8,
        created_at=now - timedelta(hours=last_accessed_hours_ago + 1),
        last_accessed=now - timedelta(hours=last_accessed_hours_ago),
        access_count=3,
        source_count=1,
    )


def _make_varied_ts(now: datetime, last_hours_ago: float = 24.0) -> list[datetime]:
    """
    10 timestamps with alternating 100h/200h gaps to ensure non-zero variance.

    Mean inter-event ≈ 144h (>> 24h), so gamma.sf(24, ...) ≈ 1.0.
    Mean inter-event ≈ 144h, so gamma.sf(290h, ...) << 0.05.
    """
    last = now - timedelta(hours=last_hours_ago)
    # Build 10 timestamps by inserting 9 gaps from newest backward.
    # gaps: i=0→100, i=1→200, ..., i=8→100 (palindromic alternating)
    ts: list[datetime] = [last]
    gaps = [100.0 if i % 2 == 0 else 200.0 for i in range(9)]
    for g in reversed(gaps):
        ts.insert(0, ts[0] - timedelta(hours=g))
    return ts


# ── TestModelFor ──────────────────────────────────────────────────────────────


class TestModelFor:
    def test_zero_events(self) -> None:
        assert TemporalWeighter().model_for(0) == TemporalModelType.EXPONENTIAL

    def test_four_events(self) -> None:
        assert TemporalWeighter().model_for(4) == TemporalModelType.EXPONENTIAL

    def test_five_events(self) -> None:
        assert TemporalWeighter().model_for(5) == TemporalModelType.RENEWAL

    def test_nineteen_events(self) -> None:
        assert TemporalWeighter().model_for(19) == TemporalModelType.RENEWAL

    def test_twenty_events(self) -> None:
        assert TemporalWeighter().model_for(20) == TemporalModelType.HAWKES

    def test_hundred_events(self) -> None:
        assert TemporalWeighter().model_for(100) == TemporalModelType.HAWKES


# ── TestExponentialWeight ─────────────────────────────────────────────────────


class TestExponentialWeight:
    def test_empty_timestamps_returns_float_in_range(self) -> None:
        w = TemporalWeighter()
        entry = _sample_entry(last_accessed_hours_ago=24.0)
        result = w.weight(entry, [], query_time=datetime.now(UTC))
        assert 0.0 <= result <= 1.0

    def test_one_hour_ago_high_weight(self) -> None:
        w = TemporalWeighter()
        now = datetime.now(UTC)
        result = w.exponential_weight(last=now - timedelta(hours=1), query_time=now)
        assert result > 0.99

    def test_one_week_ago_approx_point_432(self) -> None:
        # exp(-0.005 * 168) = exp(-0.84) ≈ 0.432
        w = TemporalWeighter()
        now = datetime.now(UTC)
        result = w.exponential_weight(last=now - timedelta(hours=168), query_time=now)
        assert abs(result - 0.432) < 0.01

    def test_one_month_ago_low_weight(self) -> None:
        # exp(-0.005 * 720) = exp(-3.6) ≈ 0.027
        w = TemporalWeighter()
        now = datetime.now(UTC)
        result = w.exponential_weight(last=now - timedelta(hours=720), query_time=now)
        assert result < 0.05

    def test_monotonically_decreasing(self) -> None:
        w = TemporalWeighter()
        now = datetime.now(UTC)
        w_1h = w.exponential_weight(last=now - timedelta(hours=1), query_time=now)
        w_24h = w.exponential_weight(last=now - timedelta(hours=24), query_time=now)
        w_168h = w.exponential_weight(last=now - timedelta(hours=168), query_time=now)
        w_720h = w.exponential_weight(last=now - timedelta(hours=720), query_time=now)
        assert w_1h > w_24h > w_168h > w_720h

    def test_beta_override_higher_decay(self) -> None:
        w = TemporalWeighter()
        now = datetime.now(UTC)
        last = now - timedelta(hours=24)
        slow = w.exponential_weight(last=last, query_time=now, beta=0.005)
        fast = w.exponential_weight(last=last, query_time=now, beta=0.01)
        assert fast < slow  # higher beta → faster decay → lower weight


# ── TestRenewalWeight ─────────────────────────────────────────────────────────


class TestRenewalWeight:
    # Skip entire class when scipy is not installed.
    scipy = pytest.importorskip("scipy")

    def test_recent_access_high_weight(self) -> None:
        """
        Varying-spacing access pattern queried 24h after last access.
        Mean inter-event ≈ 144h >> 24h → gamma SF near 1.0.
        """
        now = datetime.now(UTC)
        ts = _make_varied_ts(now, last_hours_ago=24.0)
        w = TemporalWeighter()
        result = w.renewal_weight(ts, query_time=now)
        assert result > 0.95

    def test_overdue_access_low_weight(self) -> None:
        """
        Same access pattern but queried ≈2× mean inter-event time after last.
        290h >> 144h mean → gamma SF ≈ 0.
        """
        now = datetime.now(UTC)
        ts = _make_varied_ts(now, last_hours_ago=290.0)
        w = TemporalWeighter()
        result = w.renewal_weight(ts, query_time=now)
        assert result < 0.05

    def test_fallback_when_scipy_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should fall back to exponential and return float in [0, 1]."""
        now = datetime.now(UTC)
        ts = _make_varied_ts(now, last_hours_ago=24.0)
        # Mask scipy.stats so the import inside renewal_weight raises ImportError.
        monkeypatch.setitem(sys.modules, "scipy.stats", None)
        w = TemporalWeighter()
        result = w.renewal_weight(ts, query_time=now)
        assert 0.0 <= result <= 1.0

    def test_zero_variance_no_raise(self) -> None:
        """
        5 timestamps all exactly 100h apart → var=0 → falls back cleanly.
        Must not raise ZeroDivisionError; must return float in [0, 1].
        """
        now = datetime.now(UTC)
        ts = [now - timedelta(hours=100 * i) for i in range(4, -1, -1)]
        w = TemporalWeighter()
        result = w.renewal_weight(ts, query_time=now)
        assert 0.0 <= result <= 1.0


# ── TestHawkesWeight ──────────────────────────────────────────────────────────


class TestHawkesWeight:
    def test_recent_access_in_open_interval(self) -> None:
        """25 timestamps at 3-day spacing, queried 24h after last → in (0, 1)."""
        now = datetime.now(UTC)
        ts = make_timestamps(25, spacing_hours=72, last_hours_ago=24.0)
        w = TemporalWeighter()
        result = w.hawkes_weight(ts, query_time=now)
        assert 0.0 < result < 1.0

    def test_very_stale_near_zero(self) -> None:
        """25 timestamps, last one 10000h ago → kernel contributions ≈ 0."""
        now = datetime.now(UTC)
        ts = make_timestamps(25, spacing_hours=72, last_hours_ago=10000.0)
        w = TemporalWeighter()
        result = w.hawkes_weight(ts, query_time=now)
        assert result < 0.001

    def test_result_in_unit_interval(self) -> None:
        now = datetime.now(UTC)
        ts = make_timestamps(25, spacing_hours=72, last_hours_ago=24.0)
        w = TemporalWeighter()
        result = w.hawkes_weight(ts, query_time=now)
        assert 0.0 <= result <= 1.0


# ── TestFallbackChain ─────────────────────────────────────────────────────────


class TestFallbackChain:
    """weight() must return [0, 1] for any input size, with any library config."""

    def test_empty_timestamps(self) -> None:
        now = datetime.now(UTC)
        entry = _sample_entry(last_accessed_hours_ago=24.0)
        result = TemporalWeighter().weight(entry, [], query_time=now)
        assert 0.0 <= result <= 1.0

    def test_single_timestamp(self) -> None:
        now = datetime.now(UTC)
        entry = _sample_entry(last_accessed_hours_ago=24.0)
        ts = [now - timedelta(hours=24)]
        result = TemporalWeighter().weight(entry, ts, query_time=now)
        assert 0.0 <= result <= 1.0

    def test_three_timestamps_exponential(self) -> None:
        now = datetime.now(UTC)
        entry = _sample_entry()
        result = TemporalWeighter().weight(entry, make_timestamps(3), query_time=now)
        assert 0.0 <= result <= 1.0

    def test_ten_timestamps_renewal(self) -> None:
        now = datetime.now(UTC)
        entry = _sample_entry()
        result = TemporalWeighter().weight(entry, make_timestamps(10), query_time=now)
        assert 0.0 <= result <= 1.0

    def test_twenty_five_timestamps_hawkes(self) -> None:
        now = datetime.now(UTC)
        entry = _sample_entry()
        result = TemporalWeighter().weight(entry, make_timestamps(25), query_time=now)
        assert 0.0 <= result <= 1.0

    def test_all_sizes_in_unit_interval(self) -> None:
        """weight() always returns [0.0, 1.0] regardless of timestamp count."""
        now = datetime.now(UTC)
        entry = _sample_entry()
        w = TemporalWeighter()
        for n in [0, 1, 3, 5, 10, 15, 20, 25]:
            ts = make_timestamps(n) if n > 0 else []
            result = w.weight(entry, ts, query_time=now)
            assert 0.0 <= result <= 1.0, f"Out of [0,1] for n={n}: {result}"
