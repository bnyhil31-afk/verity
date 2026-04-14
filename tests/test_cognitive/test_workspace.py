"""
tests/test_cognitive/test_workspace.py
=======================================
Tests for GlobalWorkspace — capacity-limited competitive context selection.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from verity.cognitive.types import (
    ConfidenceTier,
    MemoryEntry,
    MemoryTier,
    RetrievalResult,
)
from verity.cognitive.workspace import GlobalWorkspace

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_result(
    score: float,
    importance: float = 0.5,
    embedding: list[float] | None = None,
) -> RetrievalResult:
    entry = MemoryEntry(
        memory_id=str(uuid.uuid4()),
        content=f"memory with score {score}",
        user_id="test",
        tier=MemoryTier.FAST,
        confidence_tier=ConfidenceTier.MODIFIABLE,
        importance=importance,
        strength=1.0,
        created_at=datetime.now(UTC),
        last_accessed=datetime.now(UTC),
        access_count=1,
        source_count=1,
        embedding=embedding,
    )
    return RetrievalResult(memory=entry, score=score, position=1)


def make_candidates(n: int) -> list[RetrievalResult]:
    """Scores descend: 1.0, 0.9, 0.8, … (best-first for easy testing)."""
    return [make_result(score=round(1.0 - i * 0.1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# TestCapacity
# ---------------------------------------------------------------------------


class TestCapacity:
    ws = GlobalWorkspace(capacity=5)

    def test_empty_returns_empty(self) -> None:
        assert self.ws.select([]) == []

    def test_fewer_than_capacity_returns_all(self) -> None:
        result = self.ws.select(make_candidates(3))
        assert len(result) == 3

    def test_exactly_capacity(self) -> None:
        result = self.ws.select(make_candidates(5))
        assert len(result) == 5

    def test_more_than_capacity_returns_capacity(self) -> None:
        result = self.ws.select(make_candidates(20))
        assert len(result) == 5

    def test_capacity_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            GlobalWorkspace(capacity=0)

    def test_capacity_21_raises(self) -> None:
        with pytest.raises(ValueError):
            GlobalWorkspace(capacity=21)


# ---------------------------------------------------------------------------
# TestPositionReorder
# ---------------------------------------------------------------------------


class TestPositionReorder:
    ws = GlobalWorkspace(capacity=5)

    def test_rank1_is_first(self) -> None:
        result = self.ws.select(make_candidates(5))
        assert result[0].score == pytest.approx(1.0)

    def test_rank2_is_last(self) -> None:
        result = self.ws.select(make_candidates(5))
        assert result[-1].score == pytest.approx(0.9)

    def test_rank5_in_middle(self) -> None:
        # K=5: [rank1, rank3, rank5, rank4, rank2]
        # rank5 has score 0.6 (5th best from candidates with scores 1.0,0.9,0.8,0.7,0.6)
        result = self.ws.select(make_candidates(5))
        assert result[2].score == pytest.approx(0.6)

    def test_capacity3_rank1_first(self) -> None:
        ws = GlobalWorkspace(capacity=3)
        result = ws.select(make_candidates(10))
        assert len(result) == 3
        scores = [r.score for r in result]
        assert scores[0] == max(scores)

    def test_capacity3_rank2_last(self) -> None:
        ws = GlobalWorkspace(capacity=3)
        result = ws.select(make_candidates(10))
        assert len(result) == 3
        # Sorted scores: second-highest should be last
        sorted_scores = sorted([r.score for r in result], reverse=True)
        assert result[-1].score == pytest.approx(sorted_scores[1])

    def test_capacity7_rank1_first(self) -> None:
        ws = GlobalWorkspace(capacity=7)
        result = ws.select(make_candidates(20))
        assert len(result) == 7
        scores = [r.score for r in result]
        assert result[0].score == max(scores)

    def test_capacity7_rank2_last(self) -> None:
        ws = GlobalWorkspace(capacity=7)
        result = ws.select(make_candidates(20))
        assert len(result) == 7
        sorted_scores = sorted([r.score for r in result], reverse=True)
        assert result[-1].score == pytest.approx(sorted_scores[1])

    def test_k1_returns_rank1_only(self) -> None:
        ws = GlobalWorkspace(capacity=1)
        result = ws.select(make_candidates(5))
        assert len(result) == 1
        assert result[0].score == pytest.approx(1.0)

    def test_k2_rank1_first_rank2_second(self) -> None:
        ws = GlobalWorkspace(capacity=2)
        result = ws.select(make_candidates(5))
        assert len(result) == 2
        assert result[0].score == pytest.approx(1.0)
        assert result[1].score == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# TestCompositeScoring
# ---------------------------------------------------------------------------


class TestCompositeScoring:
    ws = GlobalWorkspace(capacity=5)

    def test_high_importance_wins_over_high_search_score(self) -> None:
        # composite(low_score, high_imp) = 0.5 * 0.9 = 0.45
        # composite(high_score, low_imp) = 0.9 * 0.1 = 0.09
        low_score_high_imp = make_result(score=0.5, importance=0.9)
        high_score_low_imp = make_result(score=0.9, importance=0.1)
        result = self.ws.select([high_score_low_imp, low_score_high_imp])
        assert result[0].memory.importance == pytest.approx(0.9)

    def test_goal_embedding_biases_toward_aligned_memory(self) -> None:
        pytest.importorskip("numpy")

        on_topic = make_result(
            score=0.5, importance=0.5, embedding=[1.0, 0.0, 0.0, 0.0]
        )
        off_topic = make_result(
            score=0.5, importance=0.5, embedding=[0.0, 1.0, 0.0, 0.0]
        )
        # Equal scores and importance — only goal alignment differs.
        # on_topic  top_down ≈ 1.0
        # off_topic top_down = 0.2 (floor, orthogonal embeddings → cosine = 0)
        result = self.ws.select(
            [on_topic, off_topic], goal_embedding=[1.0, 0.0, 0.0, 0.0]
        )
        assert result[0].memory.embedding is not None
        assert result[0].memory.embedding[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# TestFallbacks
# ---------------------------------------------------------------------------


class TestFallbacks:
    def test_no_scorer_no_temporal(self) -> None:
        ws = GlobalWorkspace(capacity=5)
        result = ws.select(make_candidates(10))
        assert len(result) == 5

    def test_scorer_only(self) -> None:
        from verity.cognitive.scoring import ImportanceScorer

        ws = GlobalWorkspace(capacity=5, scorer=ImportanceScorer())
        result = ws.select(make_candidates(10))
        assert len(result) == 5

    def test_temporal_only(self) -> None:
        from verity.cognitive.temporal import TemporalWeighter

        ws = GlobalWorkspace(capacity=5, temporal=TemporalWeighter())
        result = ws.select(make_candidates(10))
        assert len(result) == 5

    def test_numpy_unavailable_does_not_raise(self) -> None:
        """
        When numpy is not importable, goal_embedding bias should be silently
        disabled and select() must still return results.
        """
        ws = GlobalWorkspace(capacity=5)
        candidates = [
            make_result(score=0.5, importance=0.5, embedding=[1.0, 0.0])
            for _ in range(10)
        ]
        goal_embedding = [1.0, 0.0]

        # Patch builtins.__import__ so that 'numpy' raises ImportError.
        import builtins

        original_import = builtins.__import__

        def fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
            if name == "numpy":
                raise ImportError("numpy mocked as unavailable")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fake_import):
            result = ws.select(candidates, goal_embedding=goal_embedding)

        assert len(result) == 5
