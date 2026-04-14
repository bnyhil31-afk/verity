"""
verity.cognitive.scoring
========================
Importance scoring for memory entries.

Uses prediction error (embedding distance from a running centroid) as a
dopamine/norepinephrine proxy.  High deviation from the system's "current
expectation" = high surprise = high importance.

Zero-dependency path: if numpy is unavailable the surprise and relevance
components are replaced by their neutral value (0.5) and the weights are
redistributed to recency and reference when the embedding field is None.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any

from verity.cognitive.types import ImportanceWeights, MemoryEntry

# ---------------------------------------------------------------------------
# Optional numpy — detected once at import time
# ---------------------------------------------------------------------------
try:
    import numpy as _np  # type: ignore[import-untyped]

    _HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _cosine(a: Any, b: Any) -> float:
    """Cosine similarity in [-1, 1].  Caller must ensure numpy is available."""
    av = _np.array(a, dtype=_np.float64)
    bv = _np.array(b, dtype=_np.float64)
    norm_a = float(_np.linalg.norm(av))
    norm_b = float(_np.linalg.norm(bv))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(_np.dot(av, bv) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# ImportanceScorer
# ---------------------------------------------------------------------------


class ImportanceScorer:
    """
    Computes composite importance scores using prediction error as a
    dopamine/norepinephrine proxy.

    Formula:
        importance = w_s × surprise
                   + w_r × recency
                   + w_ref × reference
                   + w_rel × relevance

    Where ``surprise = 1 - cosine_similarity(embedding, running_centroid)``.

    The running centroid (exponential moving average of all embeddings)
    represents the system's current expectation.  High deviation = high
    surprise = high importance.

    When ``entry.embedding`` is ``None``, surprise and relevance are skipped
    and their combined weight is redistributed proportionally to recency and
    reference.

    When numpy is unavailable, ``surprise()`` and ``update_centroid()`` are
    no-ops / return the neutral 0.5 value.
    """

    def __init__(self, weights: ImportanceWeights | None = None) -> None:
        self._weights: ImportanceWeights = weights if weights is not None else ImportanceWeights()
        self._centroid: Any = None  # numpy float64 array, or None until first update
        self._signals: dict[str, list[tuple[str, float]]] = {}

    # ------------------------------------------------------------------
    # Core scoring
    # ------------------------------------------------------------------

    def score(
        self,
        entry: MemoryEntry,
        query_embedding: list[float] | None = None,
    ) -> float:
        """Compute composite importance score clamped to [0.0, 1.0]."""
        w = self._weights
        recency = self.recency_decay(entry.last_accessed)
        reference = self.reference_boost(entry.access_count)

        if entry.embedding is None:
            # Redistribute surprise_weight + relevance_weight proportionally
            # to the two available components (recency and reference).
            avail = w.recency_weight + w.reference_weight
            if avail == 0.0:
                return 0.0
            missing = w.surprise_weight + w.relevance_weight
            rec_w = w.recency_weight + (w.recency_weight / avail) * missing
            ref_w = w.reference_weight + (w.reference_weight / avail) * missing
            result = rec_w * recency + ref_w * reference
        else:
            surprise = self.surprise(entry.embedding)

            if query_embedding is not None and _HAS_NUMPY:
                relevance = max(0.0, min(1.0, _cosine(query_embedding, entry.embedding)))
            else:
                relevance = 0.5

            result = (
                w.surprise_weight * surprise
                + w.recency_weight * recency
                + w.reference_weight * reference
                + w.relevance_weight * relevance
            )

        return max(0.0, min(1.0, result))

    def surprise(self, embedding: list[float]) -> float:
        """
        Return ``1 - cosine_similarity(embedding, running_centroid)``.

        Returns 0.5 if the centroid has not yet been initialized or if
        numpy is unavailable.
        """
        if not _HAS_NUMPY or self._centroid is None:
            return 0.5
        sim = _cosine(embedding, self._centroid)
        return max(0.0, min(1.0, 1.0 - sim))

    def recency_decay(self, last_accessed: datetime) -> float:
        """
        Exponential decay: ``0.995 ^ hours_since_last_accessed``.

        Result clamped to [0.0, 1.0].  The half-life is ~138 hours.
        """
        now = datetime.now(UTC)
        hours = max(0.0, (now - last_accessed).total_seconds() / 3600.0)
        return max(0.0, min(1.0, 0.995**hours))

    def reference_boost(self, access_count: int) -> float:
        """
        ``min(2.0, 1.0 + 0.1 × access_count)`` normalized to [0.0, 1.0].

        access_count=0  → 0.5  (1.0 / 2.0)
        access_count=10 → 1.0  (2.0 / 2.0, capped)
        """
        raw = min(2.0, 1.0 + 0.1 * access_count)
        return raw / 2.0

    # ------------------------------------------------------------------
    # Centroid management
    # ------------------------------------------------------------------

    def update_centroid(self, embedding: list[float]) -> None:
        """
        Exponential moving average update: ``μ = 0.99μ + 0.01 × embedding``.

        The centroid is normalized to unit length after every update.
        No-op when numpy is unavailable.
        """
        if not _HAS_NUMPY:
            return
        emb_arr = _np.array(embedding, dtype=_np.float64)
        if self._centroid is None:
            self._centroid = emb_arr.copy()
        else:
            self._centroid = 0.99 * self._centroid + 0.01 * emb_arr
        norm = float(_np.linalg.norm(self._centroid))
        if norm > 0.0:
            self._centroid = self._centroid / norm

    # ------------------------------------------------------------------
    # Signal tracking
    # ------------------------------------------------------------------

    def record_signal(
        self,
        memory_id: str,
        signal_type: str,
        weight: float = 1.0,
    ) -> None:
        """
        Record an implicit feedback signal for *memory_id*.

        Stored in-memory only (no SQLite writes).  Used by
        ``ConsolidationCycle`` to identify high-signal memories.

        ``signal_type`` should be one of:
        ``"recall"`` | ``"correction"`` | ``"dwell"`` | ``"reference"``
        """
        self._signals.setdefault(memory_id, []).append((signal_type, weight))

    def get_signals(self, memory_id: str) -> list[tuple[str, float]]:
        """Return all recorded signals for *memory_id*.  Empty list if none."""
        return list(self._signals.get(memory_id, []))

    def signal_score(self, memory_id: str) -> float:
        """
        Weighted sum of recorded signals, passed through a sigmoid.

        Base signal weights:
            recall=1.0, reference=0.8, dwell=0.5, correction=-1.0

        Each entry contributes ``base_weight × recorded_weight`` to the sum.
        The sigmoid maps the sum to (0, 1):
            sigmoid(0) = 0.5 → neutral (returned when no signals recorded)
            positive sum > 0.5 → memory is high-signal
            negative sum < 0.5 → memory is being contradicted

        Returns 0.5 for unknown ``memory_id``.
        """
        signals = self._signals.get(memory_id)
        if not signals:
            return 0.5

        _base: dict[str, float] = {
            "recall": 1.0,
            "reference": 0.8,
            "dwell": 0.5,
            "correction": -1.0,
        }
        total = sum(_base.get(st, 0.0) * w for st, w in signals)
        return 1.0 / (1.0 + math.exp(-total))
