"""
verity.cognitive.reconsolidation
=================================
Memory reconsolidation with biological boundary conditions.

Four protection tiers prevent runaway drift:
- IMMUTABLE:  never modified (conf >= 0.95, sources >= 5, narrow CI)
- PROTECTED:  higher PE threshold (conf >= 0.80, sources >= 3)
- MODIFIABLE: standard rules (conf >= 0.50, sources >= 1)
- LABILE:     freely modifiable (conf < 0.50)

Modification gate: sigmoid(k × (PE - threshold)), k=10.

Zero dependencies — stdlib only. No numpy, scipy, or ML libraries.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import UTC, datetime

from verity.cognitive.types import ConfidenceTier, MemoryEntry


class ReconsolidationEngine:
    """
    Implements memory reconsolidation with biological boundary conditions.

    The key insight: retrieving a memory only triggers modification if
    prediction error exceeds a threshold. Strong, old, multiply-confirmed
    memories require much higher prediction error to destabilize.

    Four protection tiers (maps to MemoryEntry.confidence_tier):
    - IMMUTABLE:  confidence >= 0.95, sources >= 5 → never modified
    - PROTECTED:  confidence >= 0.80, sources >= 3 → high PE required
    - MODIFIABLE: confidence >= 0.50, sources >= 1 → standard rules
    - LABILE:     confidence < 0.50  → freely modifiable

    Modification gate: sigmoid(k × (prediction_error - threshold))
    where k=10 (steep), threshold varies by tier.
    """

    def tier_thresholds(self) -> dict[ConfidenceTier, float]:
        """
        Prediction error required to trigger reconsolidation per tier.
        LABILE: 0.1, MODIFIABLE: 0.3, PROTECTED: 0.6, IMMUTABLE: inf
        """
        return {
            ConfidenceTier.LABILE:     0.10,
            ConfidenceTier.MODIFIABLE: 0.30,
            ConfidenceTier.PROTECTED:  0.60,
            ConfidenceTier.IMMUTABLE:  math.inf,
        }

    def gate(self, prediction_error: float, threshold: float) -> float:
        """
        Sigmoid gate: σ(k × (PE - threshold))
        Returns value in [0.0, 1.0] — probability of reconsolidation.
        At PE == threshold the gate returns exactly 0.5.
        """
        k = 10.0
        return 1.0 / (1.0 + math.exp(-k * (prediction_error - threshold)))

    def should_reconsolidate(
        self,
        entry: MemoryEntry,
        prediction_error: float,
    ) -> bool:
        """
        Returns True only if prediction_error exceeds the tier threshold.
        IMMUTABLE memories always return False regardless of prediction_error.
        """
        if entry.confidence_tier is ConfidenceTier.IMMUTABLE:
            return False
        threshold = self.tier_thresholds()[entry.confidence_tier]
        # gate > 0.5 iff prediction_error > threshold (sigmoid property)
        return self.gate(prediction_error, threshold) > 0.5

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_tier(
        alpha: float,
        beta: float,
        source_count: int,
    ) -> ConfidenceTier:
        """
        Derive ConfidenceTier from Bayesian parameters.
        Checks tiers top-down: IMMUTABLE → PROTECTED → MODIFIABLE → LABILE.
        """
        n = alpha + beta
        conf = alpha / n
        ci_width = 2.0 * 1.96 * math.sqrt(conf * (1.0 - conf) / max(n, 1))

        if conf >= 0.95 and source_count >= 5 and ci_width < 0.05:
            return ConfidenceTier.IMMUTABLE
        if conf >= 0.80 and source_count >= 3:
            return ConfidenceTier.PROTECTED
        if conf >= 0.50 and source_count >= 1:
            return ConfidenceTier.MODIFIABLE
        return ConfidenceTier.LABILE

    # ── Public API ────────────────────────────────────────────────────────────

    def update(
        self,
        entry: MemoryEntry,
        new_content: str,
        prediction_error: float,
        source_confirmed: bool = False,
    ) -> MemoryEntry:
        """
        Conditionally update a memory based on the reconsolidation gate.

        If gate is closed: return entry unchanged (same object, no copy).
        If gate is open:
          - Classify as confirmation (PE < 0.4) or contradiction (PE >= 0.4).
          - Apply Bayesian update: confirmation → alpha += 1, contradiction → beta += 1.
          - If source_confirmed: source_count += 1.
          - Replace content and last_accessed.
          - Recompute confidence_tier from new alpha/beta/source_count.

        Pure logic — does not write to SQLite. The caller owns persistence.
        """
        # Step 1 — gate check
        if not self.should_reconsolidate(entry, prediction_error):
            return entry

        # Step 2 — classify update type
        is_confirmation = prediction_error < 0.4

        # Step 3 — Bayesian update
        new_alpha = entry.alpha + (1.0 if is_confirmation else 0.0)
        new_beta  = entry.beta  + (0.0 if is_confirmation else 1.0)
        new_source_count = entry.source_count + (1 if source_confirmed else 0)

        # Step 5 — recompute tier (before building the new entry)
        new_confidence_tier = self._compute_tier(new_alpha, new_beta, new_source_count)

        # Steps 4 & 6 — build and return updated entry (no mutation of original)
        return replace(
            entry,
            content=new_content,
            last_accessed=datetime.now(UTC),
            alpha=new_alpha,
            beta=new_beta,
            source_count=new_source_count,
            confidence_tier=new_confidence_tier,
        )

    def promote_tier(
        self,
        entry: MemoryEntry,
        source_confirmed: bool = False,
    ) -> MemoryEntry:
        """
        Record a confirmation without changing content.

        Uses prediction_error=0.65 to open the gate for all tiers except
        IMMUTABLE (LABILE: 0.10, MODIFIABLE: 0.30, PROTECTED: 0.60).
        Always increments alpha — recall is a confirmation signal regardless
        of the PE value used to pass the gate.

        Used when a memory is recalled and verified to be correct.
        """
        if not self.should_reconsolidate(entry, 0.65):
            return entry
        new_alpha = entry.alpha + 1.0
        new_source_count = entry.source_count + (1 if source_confirmed else 0)
        new_confidence_tier = self._compute_tier(
            new_alpha, entry.beta, new_source_count
        )
        return replace(
            entry,
            last_accessed=datetime.now(UTC),
            alpha=new_alpha,
            source_count=new_source_count,
            confidence_tier=new_confidence_tier,
        )

    def demote_tier(self, entry: MemoryEntry) -> MemoryEntry:
        """
        Record a contradiction without changing content.

        Increments beta by 1.0 and recomputes confidence_tier.
        Does NOT check should_reconsolidate() — demotion always applies.
        Used when external evidence contradicts a memory.
        """
        new_beta = entry.beta + 1.0
        new_confidence_tier = self._compute_tier(
            entry.alpha, new_beta, entry.source_count
        )
        return replace(entry, beta=new_beta, confidence_tier=new_confidence_tier)
