"""
verity.cognitive.consolidation
==============================
Sleep consolidation cycle: decay, prune, abstract.

Implements the SO-spindle-ripple cascade as offline memory processing:
  Phase 1 (Decay):    Reduce all memory strengths globally.
  Phase 2 (Prune):    Remove weak, non-protected memories.
  Phase 3 (Abstract): Cluster similar fast-buffer memories into
                      slow-store abstractions.

No LLM required for Phases 1 and 2.
Phase 3 uses centroid-based summarization by default.
Optional LLM-powered summarization via ``summarizer=`` parameter.

Zero dependencies beyond stdlib for Phases 1 and 2.
Phase 3 requires numpy (silently returns 0 without it).
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from verity.cognitive.scoring import ImportanceScorer
from verity.cognitive.store import DualSpeedStore
from verity.cognitive.types import (
    ConfidenceTier,
    MemoryEntry,
    MemoryTier,
    SleepCycleResult,
)

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
    """Cosine similarity in [-1, 1]. Caller must ensure numpy is available."""
    dot = _np.dot(a, b)
    na = float(_np.linalg.norm(a))
    nb = float(_np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(dot / (na * nb))


# ---------------------------------------------------------------------------
# ConsolidationCycle
# ---------------------------------------------------------------------------


class ConsolidationCycle:
    """
    Implements sleep-like memory consolidation:
    Phase 1 (Decay): Reduce all memory strengths globally.
    Phase 2 (Prune): Remove memories below strength threshold.
    Phase 3 (Abstract): Cluster similar fast-buffer memories,
                        summarize into slow-store abstractions.

    This is the computational analog of the SO-spindle-ripple cascade.

    No LLM required for Phases 1 and 2.
    Phase 3 uses centroid-based summarization by default.
    LLM-powered summarization is opt-in via summarizer= parameter.
    """

    def __init__(
        self,
        store: DualSpeedStore,
        scorer: ImportanceScorer,
        decay_factor: float = 0.90,
        prune_threshold: float = 0.05,
        cluster_min_size: int = 3,
        similarity_threshold: float = 0.85,
        summarizer: Any = None,
    ) -> None:
        self._store = store
        self._scorer = scorer
        self._decay_factor = decay_factor
        self._prune_threshold = prune_threshold
        self._cluster_min_size = cluster_min_size
        self._similarity_threshold = similarity_threshold
        self._summarizer = summarizer
        # Tracks total fast entries deleted during the most recent abstract_pass.
        # Updated by abstract_pass(); read by run() to populate memories_merged.
        self._last_merged_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> SleepCycleResult:
        """
        Run a full consolidation cycle. Returns summary of what changed.
        Safe to call at any time — idempotent if called twice in quick succession.
        """
        start = time.perf_counter()
        cycle_timestamp = datetime.now(UTC)

        decayed = self.decay_pass()
        pruned = self.prune_pass()
        abstractions = self.abstract_pass()
        merged = self._last_merged_count

        duration = time.perf_counter() - start

        return SleepCycleResult(
            memories_decayed=decayed,
            memories_pruned=pruned,
            memories_merged=merged,
            abstractions_created=abstractions,
            duration_seconds=duration,
            cycle_timestamp=cycle_timestamp,
        )

    def decay_pass(self) -> int:
        """
        Multiply all memory strengths by decay_factor via direct SQL.
        No exemptions — ALL confidence tiers decay, including IMMUTABLE.
        (Strength tracks recency-of-use. Protection tiers prevent deletion,
        not decay.)
        Returns total count of entries updated.
        """
        conn = self._store._conn
        total = 0
        for table in ("fast_memories", "slow_memories"):
            cur = conn.execute(
                f"UPDATE {table} SET strength = strength * ?",
                (self._decay_factor,),
            )
            total += cur.rowcount
        conn.commit()
        return total

    def prune_pass(self) -> int:
        """
        Delete memories with strength < prune_threshold.
        IMMUTABLE and PROTECTED entries are exempt.
        Boundary is strict less-than: strength == prune_threshold survives.
        Returns count of entries deleted.
        """
        conn = self._store._conn
        deleted = 0
        for table in ("fast_memories", "slow_memories"):
            cur = conn.cursor()
            cur.execute(
                f"SELECT memory_id, confidence_tier FROM {table} WHERE strength < ?",
                (self._prune_threshold,),
            )
            rows = cur.fetchall()
            for memory_id, ct_str in rows:
                ct = ConfidenceTier(ct_str)
                if ct in (ConfidenceTier.IMMUTABLE, ConfidenceTier.PROTECTED):
                    continue
                conn.execute(
                    f"DELETE FROM {table} WHERE memory_id = ?",
                    (memory_id,),
                )
                deleted += 1
        conn.commit()
        return deleted

    def abstract_pass(self) -> int:
        """
        Cluster fast-buffer memories by embedding similarity.
        For clusters >= cluster_min_size:
          - Create a slow-store abstraction (centroid content or LLM summary).
          - Delete all cluster members from the fast buffer.
        Returns count of abstractions created.

        Updates self._last_merged_count with total fast entries deleted.
        Returns 0 immediately if numpy is unavailable.
        """
        if not _HAS_NUMPY:
            self._last_merged_count = 0
            return 0

        # Fetch fast entries that have stored embeddings
        fast_entries = self._store.all_fast()
        entries_with_emb = [e for e in fast_entries if e.embedding is not None]

        if len(entries_with_emb) < self._cluster_min_size:
            self._last_merged_count = 0
            return 0

        # Load embeddings as numpy float32 arrays
        emb_arrays = [
            _np.array(e.embedding, dtype=_np.float32) for e in entries_with_emb
        ]

        # Sort by importance descending so the highest-importance entry
        # anchors each cluster first.
        sorted_pairs = sorted(
            zip(entries_with_emb, emb_arrays),
            key=lambda p: p[0].importance,
            reverse=True,
        )
        entries_sorted = [p[0] for p in sorted_pairs]
        embs_sorted = [p[1] for p in sorted_pairs]

        claimed: set[str] = set()
        abstractions_created = 0
        total_merged = 0

        for i in range(len(entries_sorted)):
            anchor = entries_sorted[i]
            anchor_emb = embs_sorted[i]

            if anchor.memory_id in claimed:
                continue

            # Greedy cluster: all unclaimed entries (including anchor)
            # whose cosine similarity to the anchor >= similarity_threshold.
            cluster: list[MemoryEntry] = []
            for j in range(len(entries_sorted)):
                other = entries_sorted[j]
                if other.memory_id in claimed:
                    continue
                sim = _cosine(anchor_emb, embs_sorted[j])
                if sim >= self._similarity_threshold:
                    cluster.append(other)

            if len(cluster) < self._cluster_min_size:
                continue

            # Mark all cluster members as claimed
            for e in cluster:
                claimed.add(e.memory_id)

            # Highest-importance member provides the fallback content
            highest = max(cluster, key=lambda e: e.importance)

            # Determine abstraction content
            if self._summarizer is not None:
                content = self._summarizer([e.content for e in cluster])
            else:
                content = highest.content

            # Centroid embedding (mean of cluster, normalized to unit length)
            cluster_embs = [
                _np.array(e.embedding, dtype=_np.float32) for e in cluster
            ]
            centroid = _np.mean(cluster_embs, axis=0)
            norm = float(_np.linalg.norm(centroid))
            if norm > 0.0:
                centroid = centroid / norm

            mean_importance = sum(e.importance for e in cluster) / len(cluster)
            now = datetime.now(UTC)

            # Build abstraction with explicit Bayesian fields.
            # confidence_tier is set to MODIFIABLE directly — we do not run
            # _compute_tier() here because the cluster merging gives a
            # naturally higher source_count that would otherwise yield PROTECTED.
            abstraction = MemoryEntry(
                memory_id=str(uuid.uuid4()),
                content=content,
                user_id=self._store._user_id,
                tier=MemoryTier.SLOW,
                confidence_tier=ConfidenceTier.MODIFIABLE,
                importance=mean_importance,
                strength=1.0,
                created_at=now,
                last_accessed=now,
                access_count=0,
                source_count=len(cluster),
                alpha=float(len(cluster)) + 1.0,
                beta=1.0,
                metadata={
                    "abstracted_from": [e.memory_id for e in cluster],
                    "cluster_size": len(cluster),
                },
                embedding=centroid.tolist(),
            )
            # Insert directly into slow_memories, bypassing add() to preserve
            # the explicitly set Bayesian fields and confidence_tier.
            self._store._insert_entry("slow_memories", abstraction)

            # Delete all cluster members from the fast buffer
            for e in cluster:
                self._store.delete(e.memory_id)

            total_merged += len(cluster)
            abstractions_created += 1

        self._last_merged_count = total_merged
        return abstractions_created
