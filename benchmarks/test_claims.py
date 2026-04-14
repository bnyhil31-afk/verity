"""
benchmarks/test_claims.py
=========================
Core claim validation for Verity's three novel claims:

1. Reconsolidation stability — updating memories without runaway drift
2. Sleep consolidation quality — importance-stratified survival
3. Tiered temporal weighting — auto-graduating temporal models

All tests are LLM-free and use only synthetic data from data/generator.py.
"""

import importlib.util  # noqa: F401
import json  # noqa: F401
import math  # noqa: F401
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from benchmarks.data.generator import (
    consolidation_set,
    importance_set,
    temporal_set,
)

# ── CLAIM 1: RECONSOLIDATION STABILITY ──────────────────────────────────────


class TestReconsolidationStability:

    def test_telephone_game_100_cycles(self):
        """
        IMMUTABLE memory survives 100 neutral update calls unchanged.
        Neutral update = update(entry, same_content, PE=0.0).
        For IMMUTABLE (threshold=inf): gate(0.0, inf) = 0.0 → never fires.
        Content, confidence_tier, alpha, and beta must all be unchanged.
        """
        from verity.cognitive.reconsolidation import ReconsolidationEngine
        from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

        engine = ReconsolidationEngine()
        original_content = "The production database is PostgreSQL 15"

        entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content=original_content,
            user_id="test",
            tier=MemoryTier.SLOW,
            confidence_tier=ConfidenceTier.IMMUTABLE,
            importance=0.9,
            strength=1.0,
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
            access_count=50,
            source_count=6,
            alpha=250.0,
            beta=10.0,
        )

        original_alpha = entry.alpha
        original_beta  = entry.beta

        for _ in range(100):
            # PE=0.0 → gate(0.0, inf) = 0 → reconsolidation never fires
            entry = engine.update(entry, original_content, prediction_error=0.0)

        assert entry.content == original_content, (
            "IMMUTABLE memory content changed after 100 neutral update cycles"
        )
        assert entry.confidence_tier == ConfidenceTier.IMMUTABLE, (
            "IMMUTABLE tier should not change"
        )
        assert entry.alpha == original_alpha, (
            f"Alpha changed unexpectedly: {original_alpha} → {entry.alpha}. "
            "PE=0.0 should not trigger reconsolidation for any tier."
        )
        assert entry.beta == original_beta, (
            f"Beta changed unexpectedly: {original_beta} → {entry.beta}."
        )

    def test_immutable_resists_high_pe_contradiction(self):
        """
        IMMUTABLE memory ignores direct contradiction at PE=0.95.
        should_reconsolidate(IMMUTABLE, any_pe) must always return False.
        gate(any_pe, threshold=inf) = 0 for all finite PE values.
        """
        from verity.cognitive.reconsolidation import ReconsolidationEngine
        from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

        engine = ReconsolidationEngine()
        original = "The API uses JWT authentication"

        entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content=original,
            user_id="test",
            tier=MemoryTier.SLOW,
            confidence_tier=ConfidenceTier.IMMUTABLE,
            importance=0.95,
            strength=1.0,
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
            access_count=100,
            source_count=6,
            alpha=250.0,
            beta=10.0,
        )

        # Attack with maximum prediction error
        result = engine.update(entry, "The API uses OAuth2 now", 0.95)

        assert result.content == original, (
            f"IMMUTABLE memory was modified by contradiction. "
            f"Expected: '{original}'\nGot:      '{result.content}'"
        )
        # Gate must be zero for all PE values when tier is IMMUTABLE
        assert engine.should_reconsolidate(entry, 0.0)  is False
        assert engine.should_reconsolidate(entry, 0.5)  is False
        assert engine.should_reconsolidate(entry, 0.95) is False
        assert engine.should_reconsolidate(entry, 1.0)  is False

    def test_bayesian_updates_match_analytical_posterior(self):
        """
        promote_tier() fires for all tiers except IMMUTABLE (PE=0.65 opens
        LABILE threshold=0.1, MODIFIABLE threshold=0.3, PROTECTED threshold=0.6).

        Starting state: alpha=1.0, beta=3.0 (LABILE, conf=0.25)

        After 5 promote_tier() calls — all 5 fire:
          Calls 1–2: LABILE  → alpha 1→2→3, conf reaches 0.5 → MODIFIABLE
          Calls 3–5: MODIFIABLE → alpha 3→4→5→6
          alpha = 6.0, conf = 6 / (6+3) ≈ 0.667 → MODIFIABLE

        After 2 contradiction updates (PE=0.8, different content):
          Call 1: MODIFIABLE (threshold=0.3) → fires → beta=4, conf≈0.60 → MODIFIABLE
          Call 2: MODIFIABLE (threshold=0.3) → fires → beta=5
        """
        from verity.cognitive.reconsolidation import ReconsolidationEngine
        from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

        engine = ReconsolidationEngine()

        entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content="Standup is at 9am",
            user_id="test",
            tier=MemoryTier.FAST,
            confidence_tier=ConfidenceTier.LABILE,
            importance=0.5,
            strength=1.0,
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
            access_count=0,
            source_count=1,
            alpha=1.0,
            beta=3.0,
        )

        # Analytical: alpha starts 1.0, all 5 promote_tier() calls fire → 6.0
        for _ in range(5):
            entry = engine.promote_tier(entry)

        assert entry.alpha == pytest.approx(6.0, abs=0.01), (
            f"Expected alpha=6.0 after 5 promote_tier() calls "
            f"(all fire; LABILE: calls 1–2, MODIFIABLE: calls 3–5), "
            f"got {entry.alpha}"
        )
        assert entry.beta == pytest.approx(3.0, abs=0.01), (
            "Beta should not change from confirmations"
        )
        assert entry.bayesian_confidence == pytest.approx(6.0 / 9.0, abs=0.01)
        assert entry.confidence_tier == ConfidenceTier.MODIFIABLE, (
            f"Expected MODIFIABLE (conf≥0.50) after promotions, "
            f"got {entry.confidence_tier}"
        )

        # 2 contradictions: high PE + clearly different content
        for _ in range(2):
            entry = engine.update(
                entry,
                "Standup has been CANCELLED permanently",
                prediction_error=0.8,
            )

        assert entry.beta == pytest.approx(5.0, abs=0.01), (
            f"Expected beta=5.0 after 2 contradictions, got {entry.beta}"
        )

        # source_confirmed=True should increment source_count
        entry2 = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content="Database is PostgreSQL",
            user_id="test",
            tier=MemoryTier.FAST,
            confidence_tier=ConfidenceTier.LABILE,
            importance=0.5,
            strength=1.0,
            created_at=datetime.now(UTC),
            last_accessed=datetime.now(UTC),
            access_count=0,
            source_count=1,
            alpha=1.0,
            beta=3.0,
        )
        # Use high PE so gate fires for LABILE
        entry2 = engine.update(
            entry2, entry2.content, 0.5, source_confirmed=True
        )
        assert entry2.source_count == 2, (
            "source_confirmed=True should increment source_count from 1 to 2"
        )


# ── CLAIM 2: SLEEP CONSOLIDATION QUALITY ────────────────────────────────────


class TestSleepConsolidationQuality:

    def test_importance_stratification_survival(self):
        """
        IMMUTABLE memories survive 35 decay+prune cycles.
        LABILE memories are pruned (strength 0.9^35 ≈ 0.025 < 0.05).
        Fisher's exact test: p < 0.001.

        Insertion: add content to get a new entry, then update_entry()
        to overwrite alpha/beta/confidence_tier with the test values.
        IMMUTABLE entries → slow store (tier=MemoryTier.SLOW).
        LABILE entries    → fast store (default).
        """
        pytest.importorskip("scipy")
        from scipy.stats import fisher_exact

        from verity.cognitive.consolidation import ConsolidationCycle
        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.store import DualSpeedStore
        from verity.cognitive.types import MemoryTier

        data = importance_set()
        store = DualSpeedStore(path=":memory:", embedding_model="none")
        scorer = ImportanceScorer()
        cycle = ConsolidationCycle(
            store=store,
            scorer=scorer,
            decay_factor=0.90,
            prune_threshold=0.05,
        )

        high_ids = set()
        low_ids  = set()

        # Insert IMMUTABLE entries into slow store
        for template in data["high"]:
            result = store.add(
                template.content,
                tier=MemoryTier.SLOW,
            )
            # Overwrite Bayesian fields with IMMUTABLE values
            result.alpha            = template.alpha
            result.beta             = template.beta
            result.confidence_tier  = template.confidence_tier
            result.source_count     = template.source_count
            result.importance       = template.importance
            store.update_entry(result)
            high_ids.add(result.memory_id)

        # Insert LABILE entries into fast store (default)
        for template in data["low"]:
            result = store.add(template.content)
            result.alpha            = template.alpha
            result.beta             = template.beta
            result.confidence_tier  = template.confidence_tier
            result.source_count     = template.source_count
            result.importance       = template.importance
            store.update_entry(result)
            low_ids.add(result.memory_id)

        # 35 decay + prune cycles
        for _ in range(35):
            cycle.decay_pass()
            cycle.prune_pass()

        all_surviving = (
            {e.memory_id for e in store.all_fast()} |
            {e.memory_id for e in store.all_slow()}
        )

        high_survived = sum(1 for mid in high_ids if mid in all_surviving)
        high_pruned   = len(high_ids) - high_survived
        low_survived  = sum(1 for mid in low_ids  if mid in all_surviving)
        low_pruned    = len(low_ids)  - low_survived

        _, p_value = fisher_exact(
            [[high_survived, high_pruned],
             [low_survived,  low_pruned]],
            alternative="greater",
        )

        assert high_survived == 20, (
            f"IMMUTABLE memories should never be pruned, "
            f"but {high_pruned} were removed"
        )
        assert low_survived == 0, (
            f"LABILE memories should all be pruned after 35 cycles, "
            f"but {low_survived} survived"
        )
        assert p_value < 0.001, (
            f"Survival difference not statistically significant: p={p_value:.4f}"
        )

    def test_entity_preservation_in_abstraction(self):
        """
        After abstract_pass() on a group of similar memories, the
        abstraction content contains the known entities from that group.
        Target entity recall >= 0.6 per group.
        Requires numpy for cosine similarity clustering.
        """
        pytest.importorskip("numpy")
        from verity.cognitive.consolidation import ConsolidationCycle
        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.store import DualSpeedStore

        groups = consolidation_set()

        for group in groups:
            store = DualSpeedStore(path=":memory:", embedding_model="none")
            scorer = ImportanceScorer()
            cycle = ConsolidationCycle(
                store=store,
                scorer=scorer,
                similarity_threshold=0.85,
                cluster_min_size=3,
            )

            for mem in group["memories"]:
                store.add(
                    mem["content"],
                    metadata={"group": group["group_id"]},
                    _embedding=mem["embedding"],
                )

            abstractions_created = cycle.abstract_pass()

            assert abstractions_created >= 1, (
                f"Group '{group['group_id']}': expected >= 1 abstraction, "
                f"got {abstractions_created}"
            )

            slow_memories = store.all_slow()
            assert len(slow_memories) >= 1, (
                f"Group '{group['group_id']}': no slow-store abstraction found"
            )

            abstraction_lower = slow_memories[0].content.lower()
            preserved = [
                e for e in group["known_entities"]
                if e.lower() in abstraction_lower
            ]
            recall = len(preserved) / len(group["known_entities"])

            assert recall >= 0.6, (
                f"Group '{group['group_id']}': entity recall={recall:.2f} "
                f"(need >= 0.6). "
                f"Missing: {set(group['known_entities']) - set(preserved)}\n"
                f"Abstraction content: '{slow_memories[0].content}'"
            )

    def test_consolidation_reduces_fast_store_count(self):
        """
        After abstract_pass(), fast store count is lower.
        Clustered memories are absorbed into slow-store abstractions.
        """
        pytest.importorskip("numpy")
        from verity.cognitive.consolidation import ConsolidationCycle
        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.store import DualSpeedStore

        groups = consolidation_set()
        store = DualSpeedStore(path=":memory:", embedding_model="none")
        scorer = ImportanceScorer()
        cycle = ConsolidationCycle(
            store=store,
            scorer=scorer,
            similarity_threshold=0.85,
            cluster_min_size=3,
        )

        for group in groups:
            for mem in group["memories"]:
                store.add(mem["content"], _embedding=mem["embedding"])

        before = store.stats()["fast_count"]
        cycle.abstract_pass()
        after = store.stats()["fast_count"]

        assert after < before, (
            f"abstract_pass() should reduce fast_count. "
            f"Before={before}, After={after}"
        )
        assert store.stats()["slow_count"] > 0, (
            "abstract_pass() should have created slow-store abstractions"
        )


# ── CLAIM 3: TIERED TEMPORAL WEIGHTING ──────────────────────────────────────


class TestTieredTemporalWeighting:

    def test_tier_selection_by_event_count(self):
        """
        model_for() selects the correct model at every boundary.
        <5 → EXPONENTIAL, 5-19 → RENEWAL, >=20 → HAWKES.
        """
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.types import TemporalModelType

        tw = TemporalWeighter()

        assert tw.model_for(0)   == TemporalModelType.EXPONENTIAL
        assert tw.model_for(4)   == TemporalModelType.EXPONENTIAL
        assert tw.model_for(5)   == TemporalModelType.RENEWAL
        assert tw.model_for(19)  == TemporalModelType.RENEWAL
        assert tw.model_for(20)  == TemporalModelType.HAWKES
        assert tw.model_for(100) == TemporalModelType.HAWKES

    def test_recent_beats_stale_same_topic(self):
        """
        Memory accessed 1 hour ago ranks temporally higher than
        memory accessed 720 hours ago (same topic, same access pattern).
        """
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

        now = datetime.now(UTC)
        tw  = TemporalWeighter()

        def make_entry(last_accessed_hours_ago):
            la = now - timedelta(hours=last_accessed_hours_ago)
            return MemoryEntry(
                memory_id=str(uuid.uuid4()),
                content="standup meeting at 9am",
                user_id="test",
                tier=MemoryTier.FAST,
                confidence_tier=ConfidenceTier.MODIFIABLE,
                importance=0.5,
                strength=1.0,
                created_at=la,
                last_accessed=la,
                access_count=10,
                source_count=1,
                alpha=2.0,
                beta=1.0,
            )

        recent = make_entry(1)
        stale  = make_entry(720)

        # Use periodic timestamps from the generator
        data = temporal_set()
        periodic = next(e for e in data if e["access_pattern"] == "periodic")

        # Shift timestamps so the recent entry's last access was 1 hour ago
        base = periodic["access_timestamps"]
        shift = (now - timedelta(hours=1)) - base[-1]
        ts_recent = [t + shift for t in base]
        ts_stale  = [t - timedelta(hours=719) for t in ts_recent]

        weight_recent = tw.weight(recent, ts_recent, query_time=now)
        weight_stale  = tw.weight(stale,  ts_stale,  query_time=now)

        assert weight_recent > weight_stale, (
            f"Recent memory (weight={weight_recent:.4f}) should outrank "
            f"stale memory (weight={weight_stale:.4f})"
        )

    def test_temporal_score_monotonically_decreasing(self):
        """
        Temporal weight decreases monotonically as query time advances.
        Tested with RENEWAL-range timestamp history (n=10).
        """
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

        now = datetime.now(UTC)
        tw  = TemporalWeighter()

        entry = MemoryEntry(
            memory_id=str(uuid.uuid4()),
            content="test memory",
            user_id="test",
            tier=MemoryTier.FAST,
            confidence_tier=ConfidenceTier.MODIFIABLE,
            importance=0.5,
            strength=1.0,
            created_at=now - timedelta(hours=1000),
            last_accessed=now,
            access_count=10,
            source_count=1,
            alpha=2.0,
            beta=1.0,
        )

        # 10 weekly-spaced timestamps → RENEWAL model
        timestamps = [
            now - timedelta(hours=168 * i)
            for i in range(10, 0, -1)
        ]

        query_offsets_hours = [1, 24, 72, 168, 336, 720]
        weights = [
            tw.weight(entry, timestamps,
                      query_time=now + timedelta(hours=h))
            for h in query_offsets_hours
        ]

        for i in range(len(weights) - 1):
            assert weights[i] >= weights[i + 1], (
                f"Temporal weight not monotonically decreasing at step {i}: "
                f"w({query_offsets_hours[i]}h)={weights[i]:.4f} < "
                f"w({query_offsets_hours[i+1]}h)={weights[i+1]:.4f}"
            )

    def test_tier_transition_continuity(self):
        """
        No jump > 20% at tier transition boundaries (n=4→5, n=19→20).
        Detects ranking artifacts caused by abrupt model switching.
        """
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

        now = datetime.now(UTC)
        tw  = TemporalWeighter()

        def entry_and_timestamps(n):
            e = MemoryEntry(
                memory_id=str(uuid.uuid4()),
                content="test",
                user_id="test",
                tier=MemoryTier.FAST,
                confidence_tier=ConfidenceTier.MODIFIABLE,
                importance=0.5,
                strength=1.0,
                created_at=now - timedelta(days=100),
                last_accessed=now - timedelta(hours=24),
                access_count=n,
                source_count=1,
                alpha=2.0,
                beta=1.0,
            )
            # Chronological timestamps, evenly spaced at 168h, last = 24h ago
            ts = list(reversed([
                now - timedelta(hours=24 + 168 * i)
                for i in range(n)
            ]))
            return e, ts

        query_time = now

        # EXPONENTIAL → RENEWAL boundary
        e4, ts4 = entry_and_timestamps(4)
        e5, ts5 = entry_and_timestamps(5)
        w4 = tw.weight(e4, ts4, query_time)
        w5 = tw.weight(e5, ts5, query_time)
        jump_45 = abs(w5 - w4) / max(w4, 1e-9)

        assert jump_45 < 0.20, (
            f"Jump at EXPONENTIAL→RENEWAL boundary: "
            f"n=4→{w4:.4f}, n=5→{w5:.4f}, jump={jump_45:.1%} (max 20%)"
        )

        # RENEWAL → HAWKES boundary
        e19, ts19 = entry_and_timestamps(19)
        e20, ts20 = entry_and_timestamps(20)
        w19 = tw.weight(e19, ts19, query_time)
        w20 = tw.weight(e20, ts20, query_time)
        jump_1920 = abs(w20 - w19) / max(w19, 1e-9)

        assert jump_1920 < 0.20, (
            f"Jump at RENEWAL→HAWKES boundary: "
            f"n=19→{w19:.4f}, n=20→{w20:.4f}, jump={jump_1920:.1%} (max 20%)"
        )
