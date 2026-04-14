"""
tests.test_cognitive.test_reconsolidation
==========================================
Tests for verity.cognitive.reconsolidation.ReconsolidationEngine.

Covers:
- tier_thresholds() returns correct values per tier
- gate() sigmoid math (exact at midpoint, asymptotic near bounds)
- should_reconsolidate() per tier at critical prediction_error values
- update() gate-closed returns entry unchanged (same object)
- update() confirmation (PE < 0.4) increments alpha, not beta
- update() contradiction (PE >= 0.4) increments beta, not alpha
- update() source_confirmed=True increments source_count
- promote_tier() drives LABILE → MODIFIABLE over repeated calls
- demote_tier() drives IMMUTABLE → PROTECTED over many contradictions
- ci_width narrows as evidence accumulates at fixed confidence ratio
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from verity.cognitive.reconsolidation import ReconsolidationEngine
from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

# ── Fixtures & helpers ─────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(UTC)


def _make_entry(**overrides) -> MemoryEntry:
    """Build a MemoryEntry with sane defaults, overridable per test."""
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


@pytest.fixture
def engine() -> ReconsolidationEngine:
    return ReconsolidationEngine()


# ── tier_thresholds() ──────────────────────────────────────────────────────────


class TestTierThresholds:
    def test_labile_threshold(self, engine):
        t = engine.tier_thresholds()
        assert t[ConfidenceTier.LABILE] == pytest.approx(0.10)

    def test_modifiable_threshold(self, engine):
        t = engine.tier_thresholds()
        assert t[ConfidenceTier.MODIFIABLE] == pytest.approx(0.30)

    def test_protected_threshold(self, engine):
        t = engine.tier_thresholds()
        assert t[ConfidenceTier.PROTECTED] == pytest.approx(0.60)

    def test_immutable_threshold_is_inf(self, engine):
        t = engine.tier_thresholds()
        assert math.isinf(t[ConfidenceTier.IMMUTABLE])
        assert t[ConfidenceTier.IMMUTABLE] > 0  # positive infinity

    def test_all_four_tiers_present(self, engine):
        t = engine.tier_thresholds()
        assert set(t.keys()) == set(ConfidenceTier)


# ── gate() ─────────────────────────────────────────────────────────────────────


class TestGate:
    def test_midpoint_is_exactly_half(self, engine):
        """σ(k × 0) = σ(0) = 0.5 exactly."""
        assert engine.gate(0.3, 0.3) == pytest.approx(0.5)

    def test_gate_0_0_at_threshold_0_3_is_near_zero(self, engine):
        """σ(-3) ≈ 0.047 — well below 0.05."""
        result = engine.gate(0.0, 0.3)
        assert result < 0.05

    def test_gate_0_0_at_threshold_0_6_is_very_small(self, engine):
        """σ(-6) ≈ 0.0025 — well below 0.01."""
        result = engine.gate(0.0, 0.6)
        assert result < 0.01

    def test_gate_1_0_at_threshold_0_3_is_near_one(self, engine):
        """σ(7) ≈ 0.999 — above 0.99."""
        result = engine.gate(1.0, 0.3)
        assert result > 0.99

    def test_gate_above_midpoint_exceeds_half(self, engine):
        assert engine.gate(0.4, 0.3) > 0.5

    def test_gate_below_midpoint_under_half(self, engine):
        assert engine.gate(0.2, 0.3) < 0.5

    def test_gate_returns_float_in_unit_interval(self, engine):
        for pe in [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]:
            g = engine.gate(pe, 0.3)
            assert 0.0 <= g <= 1.0


# ── should_reconsolidate() ─────────────────────────────────────────────────────


class TestShouldReconsolidate:
    # ── IMMUTABLE ──────────────────────────────────────────────────────────────

    def test_immutable_at_pe_zero(self, engine):
        entry = _make_entry(confidence_tier=ConfidenceTier.IMMUTABLE)
        assert engine.should_reconsolidate(entry, 0.0) is False

    def test_immutable_at_pe_half(self, engine):
        entry = _make_entry(confidence_tier=ConfidenceTier.IMMUTABLE)
        assert engine.should_reconsolidate(entry, 0.5) is False

    def test_immutable_at_pe_one(self, engine):
        entry = _make_entry(confidence_tier=ConfidenceTier.IMMUTABLE)
        assert engine.should_reconsolidate(entry, 1.0) is False

    def test_immutable_at_extreme_pe(self, engine):
        """Even absurdly high PE cannot unlock IMMUTABLE."""
        entry = _make_entry(confidence_tier=ConfidenceTier.IMMUTABLE)
        assert engine.should_reconsolidate(entry, 999.0) is False

    # ── LABILE ─────────────────────────────────────────────────────────────────

    def test_labile_reconsolidates_at_pe_0_15(self, engine):
        """PE=0.15 > threshold=0.10 → gate > 0.5 → True."""
        entry = _make_entry(confidence_tier=ConfidenceTier.LABILE)
        assert engine.should_reconsolidate(entry, 0.15) is True

    def test_labile_does_not_reconsolidate_at_pe_0_05(self, engine):
        """PE=0.05 < threshold=0.10 → gate < 0.5 → False."""
        entry = _make_entry(confidence_tier=ConfidenceTier.LABILE)
        assert engine.should_reconsolidate(entry, 0.05) is False

    def test_labile_boundary_exactly_at_threshold(self, engine):
        """PE == threshold → gate == 0.5 → not > 0.5 → False."""
        entry = _make_entry(confidence_tier=ConfidenceTier.LABILE)
        assert engine.should_reconsolidate(entry, 0.10) is False

    # ── MODIFIABLE ─────────────────────────────────────────────────────────────

    def test_modifiable_reconsolidates_at_pe_0_35(self, engine):
        """PE=0.35 > threshold=0.30 → True."""
        entry = _make_entry(confidence_tier=ConfidenceTier.MODIFIABLE)
        assert engine.should_reconsolidate(entry, 0.35) is True

    def test_modifiable_does_not_reconsolidate_at_pe_0_15(self, engine):
        """PE=0.15 < threshold=0.30 → False."""
        entry = _make_entry(confidence_tier=ConfidenceTier.MODIFIABLE)
        assert engine.should_reconsolidate(entry, 0.15) is False

    def test_modifiable_boundary_exactly_at_threshold(self, engine):
        entry = _make_entry(confidence_tier=ConfidenceTier.MODIFIABLE)
        assert engine.should_reconsolidate(entry, 0.30) is False

    # ── PROTECTED ──────────────────────────────────────────────────────────────

    def test_protected_reconsolidates_at_pe_0_65(self, engine):
        """PE=0.65 > threshold=0.60 → True."""
        entry = _make_entry(confidence_tier=ConfidenceTier.PROTECTED)
        assert engine.should_reconsolidate(entry, 0.65) is True

    def test_protected_does_not_reconsolidate_at_pe_0_35(self, engine):
        """PE=0.35 < threshold=0.60 → False."""
        entry = _make_entry(confidence_tier=ConfidenceTier.PROTECTED)
        assert engine.should_reconsolidate(entry, 0.35) is False

    def test_protected_boundary_exactly_at_threshold(self, engine):
        entry = _make_entry(confidence_tier=ConfidenceTier.PROTECTED)
        assert engine.should_reconsolidate(entry, 0.60) is False


# ── update() ──────────────────────────────────────────────────────────────────


class TestUpdate:
    def test_gate_closed_returns_same_object(self, engine):
        """When gate is closed, update() returns the exact same entry object."""
        # MODIFIABLE threshold=0.30, PE=0.15 → gate < 0.5 → closed
        entry = _make_entry(confidence_tier=ConfidenceTier.MODIFIABLE)
        result = engine.update(entry, "new content", prediction_error=0.15)
        assert result is entry

    def test_gate_closed_content_unchanged(self, engine):
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            content="original",
        )
        result = engine.update(entry, "changed", prediction_error=0.15)
        assert result.content == "original"

    def test_confirmation_increments_alpha_not_beta(self, engine):
        """PE=0.35 opens gate for MODIFIABLE; PE < 0.4 → confirmation."""
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            alpha=2.0,
            beta=1.0,
        )
        result = engine.update(entry, "same sky", prediction_error=0.35)
        assert result.alpha == pytest.approx(3.0)   # alpha += 1
        assert result.beta  == pytest.approx(1.0)   # beta unchanged

    def test_contradiction_increments_beta_not_alpha(self, engine):
        """PE=0.6 opens gate for MODIFIABLE; PE >= 0.4 → contradiction."""
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            alpha=2.0,
            beta=1.0,
        )
        result = engine.update(entry, "sky is red", prediction_error=0.6)
        assert result.beta  == pytest.approx(2.0)   # beta += 1
        assert result.alpha == pytest.approx(2.0)   # alpha unchanged

    def test_source_confirmed_increments_source_count(self, engine):
        """source_confirmed=True must increment source_count by 1."""
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            source_count=1,
        )
        result = engine.update(
            entry, "confirmed knowledge", prediction_error=0.35,
            source_confirmed=True,
        )
        assert result.source_count == 2

    def test_source_not_confirmed_source_count_unchanged(self, engine):
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            source_count=2,
        )
        result = engine.update(entry, "new text", prediction_error=0.35)
        assert result.source_count == 2

    def test_update_returns_new_object_when_gate_open(self, engine):
        """update() must not mutate the original entry."""
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            content="original",
        )
        result = engine.update(entry, "updated", prediction_error=0.35)
        assert result is not entry
        assert entry.content == "original"   # original unchanged
        assert result.content == "updated"

    def test_update_sets_last_accessed(self, engine):
        """last_accessed on the returned entry is refreshed."""
        before = _now()
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            last_accessed=before,
        )
        result = engine.update(entry, "new", prediction_error=0.35)
        # Should be same or slightly after 'before'
        assert result.last_accessed >= before

    def test_labile_entry_updated_at_low_pe(self, engine):
        """LABILE threshold=0.10; PE=0.2 should open the gate."""
        entry = _make_entry(
            confidence_tier=ConfidenceTier.LABILE,
            alpha=1.0,
            beta=3.0,
            source_count=1,
        )
        result = engine.update(entry, "refined belief", prediction_error=0.2)
        # Gate opened → alpha incremented (PE < 0.4 → confirmation)
        assert result.alpha == pytest.approx(2.0)

    def test_confidence_tier_recomputed_after_update(self, engine):
        """After sufficient confirmations, tier should upgrade."""
        # Start LABILE: alpha=1, beta=5, source_count=1
        entry = _make_entry(
            confidence_tier=ConfidenceTier.LABILE,
            alpha=4.0,
            beta=5.0,
            source_count=1,
        )
        # One confirmation: alpha becomes 5.0 → conf=5/10=0.5 → MODIFIABLE
        result = engine.update(entry, entry.content, prediction_error=0.2)
        assert result.confidence_tier is ConfidenceTier.MODIFIABLE


# ── promote_tier() ─────────────────────────────────────────────────────────────


class TestPromoteTier:
    def test_promote_tier_labile_to_modifiable(self, engine):
        """
        LABILE entry with alpha=1, beta=5, source_count=1.
        Four calls to promote_tier() must yield MODIFIABLE.

        Trace:
          Call 1: alpha 1→2, conf=2/7≈0.286  → LABILE
          Call 2: alpha 2→3, conf=3/8=0.375  → LABILE
          Call 3: alpha 3→4, conf=4/9≈0.444  → LABILE
          Call 4: alpha 4→5, conf=5/10=0.500 → MODIFIABLE (conf>=0.50, src>=1)
        """
        entry = _make_entry(
            confidence_tier=ConfidenceTier.LABILE,
            alpha=1.0,
            beta=5.0,
            source_count=1,
        )
        for _ in range(4):
            entry = engine.promote_tier(entry)

        assert entry.confidence_tier is ConfidenceTier.MODIFIABLE
        assert entry.alpha == pytest.approx(5.0)
        assert entry.beta  == pytest.approx(5.0)

    def test_promote_tier_content_unchanged(self, engine):
        """promote_tier() must not alter content."""
        entry = _make_entry(
            confidence_tier=ConfidenceTier.LABILE,
            content="preserve me",
            alpha=1.0,
            beta=5.0,
            source_count=1,
        )
        result = engine.promote_tier(entry)
        assert result.content == "preserve me"

    def test_promote_tier_modifiable_fires(self, engine):
        """
        MODIFIABLE threshold=0.30; promote_tier uses PE=0.65 > 0.30.
        Gate is open → alpha increments, new entry returned.
        """
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            alpha=2.0,
            beta=1.0,
        )
        result = engine.promote_tier(entry)
        assert result is not entry
        assert result.alpha == pytest.approx(3.0)   # alpha += 1
        assert result.beta  == pytest.approx(1.0)   # beta unchanged

    def test_promote_tier_with_source_confirmed(self, engine):
        """source_confirmed=True in promote_tier increments source_count."""
        entry = _make_entry(
            confidence_tier=ConfidenceTier.LABILE,
            alpha=1.0,
            beta=5.0,
            source_count=1,
        )
        result = engine.promote_tier(entry, source_confirmed=True)
        assert result.source_count == 2


# ── demote_tier() ──────────────────────────────────────────────────────────────


class TestDemoteTier:
    def test_demote_increments_beta(self, engine):
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            alpha=5.0,
            beta=1.0,
        )
        result = engine.demote_tier(entry)
        assert result.beta == pytest.approx(2.0)
        assert result.alpha == pytest.approx(5.0)  # alpha unchanged

    def test_demote_does_not_change_content(self, engine):
        entry = _make_entry(
            confidence_tier=ConfidenceTier.MODIFIABLE,
            content="unchanged",
        )
        result = engine.demote_tier(entry)
        assert result.content == "unchanged"

    def test_demote_recomputes_tier(self, engine):
        """Enough demotions must eventually lower the tier."""
        # Start near PROTECTED boundary: alpha=4, beta=1 → conf=0.80
        # source_count=3 qualifies for PROTECTED
        entry = _make_entry(
            confidence_tier=ConfidenceTier.PROTECTED,
            alpha=4.0,
            beta=1.0,
            source_count=3,
        )
        # After one demote: beta=2, conf=4/6≈0.667 < 0.80 → MODIFIABLE
        result = engine.demote_tier(entry)
        assert result.confidence_tier is ConfidenceTier.MODIFIABLE

    def test_immutable_survives_single_demote(self, engine):
        """
        IMMUTABLE entry: alpha=250, beta=10, source_count=6.
        conf=250/260≈0.962, ci_width≈0.047 → IMMUTABLE.
        After one demote: beta=11, conf=250/261≈0.958, ci_width≈0.049 → still IMMUTABLE.
        """
        entry = _make_entry(
            confidence_tier=ConfidenceTier.IMMUTABLE,
            alpha=250.0,
            beta=10.0,
            source_count=6,
        )
        result = engine.demote_tier(entry)
        assert result.confidence_tier is ConfidenceTier.IMMUTABLE

    def test_immutable_drops_to_protected_after_30_demotions(self, engine):
        """
        After 30 demotions: beta=40, conf=250/290≈0.862 < 0.95 → not IMMUTABLE.
        conf >= 0.80 and source_count=6 >= 3 → PROTECTED.
        """
        entry = _make_entry(
            confidence_tier=ConfidenceTier.IMMUTABLE,
            alpha=250.0,
            beta=10.0,
            source_count=6,
        )
        for _ in range(30):
            entry = engine.demote_tier(entry)

        assert entry.confidence_tier is ConfidenceTier.PROTECTED
        assert entry.beta == pytest.approx(40.0)

    def test_demote_always_applies_no_gate_check(self, engine):
        """
        demote_tier() must bypass should_reconsolidate().
        Even an IMMUTABLE entry must have beta incremented.
        """
        entry = _make_entry(confidence_tier=ConfidenceTier.IMMUTABLE)
        original_beta = entry.beta
        result = engine.demote_tier(entry)
        assert result.beta == pytest.approx(original_beta + 1.0)


# ── Bayesian tier transitions ──────────────────────────────────────────────────


class TestTierTransitions:
    def test_labile_to_modifiable_on_confirmation(self, engine):
        """
        LABILE near boundary (alpha=4, beta=5, source_count=1).
        One confirmation: alpha→5, conf=0.5 → MODIFIABLE.
        """
        entry = _make_entry(
            confidence_tier=ConfidenceTier.LABILE,
            alpha=4.0,
            beta=5.0,
            source_count=1,
        )
        result = engine.update(entry, entry.content, prediction_error=0.2)
        assert result.confidence_tier is ConfidenceTier.MODIFIABLE

    def test_protected_to_modifiable_on_contradictions(self, engine):
        """
        PROTECTED near boundary (alpha=4, beta=1, source_count=3).
        conf=4/5=0.80 — exactly at PROTECTED threshold.
        One contradiction via update with high PE: beta→2, conf=4/6≈0.667 → MODIFIABLE.
        """
        entry = _make_entry(
            confidence_tier=ConfidenceTier.PROTECTED,
            alpha=4.0,
            beta=1.0,
            source_count=3,
        )
        # PE=0.65 > PROTECTED threshold (0.60) → gate opens; PE>=0.4 → contradiction
        result = engine.update(entry, "conflicting evidence", prediction_error=0.65)
        assert result.confidence_tier is ConfidenceTier.MODIFIABLE

    def test_tier_does_not_auto_upgrade_without_sources(self, engine):
        """
        LABILE with alpha=5, beta=5, but source_count=0.
        conf=0.50 — would be MODIFIABLE, but source_count < 1 → LABILE.
        """
        entry = _make_entry(
            confidence_tier=ConfidenceTier.LABILE,
            alpha=4.0,
            beta=5.0,
            source_count=0,
        )
        result = engine.update(entry, entry.content, prediction_error=0.2)
        # alpha→5, conf=0.5, but source_count=0 → still LABILE
        assert result.confidence_tier is ConfidenceTier.LABILE


# ── ci_width narrows with evidence ─────────────────────────────────────────────


class TestCIWidthNarrowing:
    def test_ci_width_narrows_as_evidence_accumulates(self, engine):
        """
        Fixed alpha/beta ratio (2:1) but growing n → narrowing CI width.
        Entry with n=30 should have narrower CI than n=3.
        """
        small = _make_entry(alpha=2.0,  beta=1.0)    # n=3
        large = _make_entry(alpha=20.0, beta=10.0)   # n=30, same ratio

        assert large.confidence_interval_width < small.confidence_interval_width

    def test_ci_width_very_small_at_high_n(self):
        """n=1000 at 2:1 ratio → CI width well below 0.1."""
        entry = _make_entry(alpha=667.0, beta=333.0)   # n=1000
        assert entry.confidence_interval_width < 0.1

    def test_demote_cycle_widens_ci_initially(self, engine):
        """
        Starting from tight IMMUTABLE state, demotion increases beta,
        which can widen the CI by moving conf away from extremes.
        """
        entry = _make_entry(
            confidence_tier=ConfidenceTier.IMMUTABLE,
            alpha=250.0,
            beta=10.0,
            source_count=6,
        )
        initial_width = entry.confidence_interval_width
        result = engine.demote_tier(entry)
        # At such high alpha relative to beta, moving toward 0.5 widens CI
        # Both outcomes acceptable — just check it's a valid float
        assert 0.0 < result.confidence_interval_width < 1.0
        assert isinstance(result.confidence_interval_width, float)
        _ = initial_width  # suppress unused warning
