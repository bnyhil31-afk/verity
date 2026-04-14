"""
verity.cognitive.workspace
==========================
Global Workspace Theory as a capacity-limited broadcast buffer.

NOT a general retrieval system.  Given candidates from DualSpeedStore.search(),
selects the top-K most relevant using goal-directed composite scoring, then
applies position-aware reordering to mitigate the 'lost in the middle' effect.

K=5 default (Cowan 2001: 4±1 working memory capacity).  Configurable 1–20.

Composite score: salience × relevance × recency × top_down_weight
Output order:   [rank-1, rank-3, rank-5, ..., rank-4, rank-2]
                (best and second-best at boundaries, weakest in the middle)

numpy is optional — top-down goal alignment is silently disabled when absent.
"""

from __future__ import annotations

from verity.cognitive.scoring import ImportanceScorer
from verity.cognitive.temporal import TemporalWeighter
from verity.cognitive.types import RetrievalResult


class GlobalWorkspace:
    """
    Capacity-limited broadcast buffer for context-window selection.

    capacity: max items admitted to the buffer.  Range 1–20.
    scorer:   optional ImportanceScorer (provides recency_decay fallback).
    temporal: optional TemporalWeighter (preferred recency source).
    """

    def __init__(
        self,
        capacity: int = 5,
        scorer: ImportanceScorer | None = None,
        temporal: TemporalWeighter | None = None,
    ) -> None:
        if not 1 <= capacity <= 20:
            raise ValueError(f"capacity must be 1–20, got {capacity}")
        self._capacity = capacity
        self._scorer = scorer
        self._temporal = temporal

    @property
    def capacity(self) -> int:
        return self._capacity

    # ── Public API ────────────────────────────────────────────────────────────

    def select(
        self,
        candidates: list[RetrievalResult],
        goal: str | None = None,
        goal_embedding: list[float] | None = None,
    ) -> list[RetrievalResult]:
        """
        Competitive selection into the capacity-limited buffer.

        Steps:
          1. Score all candidates with _composite_score.
          2. Sort descending; keep top self.capacity items.
          3. Apply position-aware reordering (_position_reorder).

        Returns an empty list when candidates is empty.
        """
        if not candidates:
            return []

        ranked = sorted(
            candidates,
            key=lambda r: self._composite_score(r, goal_embedding),
            reverse=True,
        )
        selected = ranked[: self._capacity]
        return self._position_reorder(selected)

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _composite_score(
        self,
        result: RetrievalResult,
        goal_embedding: list[float] | None,
    ) -> float:
        """
        composite = salience × relevance × recency × top_down_weight

        salience   — memory.importance (Phase C composite, already in [0,1])
        relevance  — result.score (cosine similarity from DualSpeedStore.search)
        recency    — TemporalWeighter.weight  or  ImportanceScorer.recency_decay
                     or 1.0 if neither is available
        top_down   — cosine(goal_embedding, memory.embedding) if both present
                     and numpy available; 1.0 otherwise.  Floor at 0.2 so a
                     partially-matching memory is never fully suppressed.

        All components nominally in [0,1] → composite in [0,1].
        """
        memory = result.memory

        salience = memory.importance
        relevance = result.score

        # ── Recency ──────────────────────────────────────────────────────────
        if self._temporal is not None:
            recency = self._temporal.weight(
                memory,
                access_timestamps=[memory.last_accessed],
            )
        elif self._scorer is not None:
            recency = self._scorer.recency_decay(memory.last_accessed)
        else:
            recency = 1.0

        # ── Top-down goal alignment ───────────────────────────────────────────
        if goal_embedding is not None and memory.embedding is not None:
            try:
                import numpy as np  # noqa: PLC0415

                ge = np.array(goal_embedding, dtype=np.float32)
                me = np.array(memory.embedding, dtype=np.float32)
                norm = float(np.linalg.norm(ge)) * float(np.linalg.norm(me))
                if norm > 1e-8:
                    top_down = float(np.dot(ge, me) / norm)
                    top_down = max(0.2, top_down)  # floor: never fully suppress
                else:
                    top_down = 1.0
            except ImportError:
                top_down = 1.0
        else:
            top_down = 1.0

        return salience * relevance * recency * top_down

    # ── Position reordering ───────────────────────────────────────────────────

    def _position_reorder(
        self, ranked: list[RetrievalResult]
    ) -> list[RetrievalResult]:
        """
        Reorders for LLM consumption to mitigate 'lost in the middle'.

        Algorithm:
            odd_indexed  = ranked[0::2]   # ranks 1,3,5,… (0-indexed 0,2,4,…)
            even_indexed = ranked[1::2]   # ranks 2,4,6,… (0-indexed 1,3,5,…)
            result = odd_indexed + list(reversed(even_indexed))

        Verified outputs by K:
            K=1: [rank1]
            K=2: [rank1, rank2]
            K=3: [rank1, rank3, rank2]
            K=4: [rank1, rank3, rank4, rank2]
            K=5: [rank1, rank3, rank5, rank4, rank2]
            K=6: [rank1, rank3, rank5, rank6, rank4, rank2]
            K=7: [rank1, rank3, rank5, rank7, rank6, rank4, rank2]

        Invariant for K >= 2: first item is rank-1, last item is rank-2.
        """
        if len(ranked) <= 2:
            return ranked[:]
        odd_indexed = ranked[0::2]
        even_indexed = ranked[1::2]
        return odd_indexed + list(reversed(even_indexed))
