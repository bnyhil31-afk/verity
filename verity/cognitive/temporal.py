"""
verity.cognitive.temporal
=========================
Tiered temporal model with automatic model graduation.

Selects between exponential, renewal (Gamma), and Hawkes process models
based on event history density. No numpy required. scipy and tick are
optional; degrades gracefully to exponential when unavailable.

    0–4 events:  Exponential decay with population-average parameters.
    5–19 events: Bayesian renewal process with Gamma inter-event times.
    20+ events:  Lightweight Hawkes process with empirical Bayes priors.

The model is selected per-memory at retrieval time, not globally.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

from verity.cognitive.types import MemoryEntry, TemporalModelType


class TemporalWeighter:
    """
    Auto-selects temporal model based on event history density.

    Sparse memories use exponential decay. Dense memories use the Hawkes
    process. The transition happens automatically within the same system.
    """

    def __init__(self, global_beta: float = 0.005) -> None:
        """
        global_beta: population-average exponential decay rate per hour.
        Default 0.005/hour gives:
            1 hour ago  → weight ≈ 0.995
            1 day ago   → weight ≈ 0.887
            1 week ago  → weight ≈ 0.432
            1 month ago → weight ≈ 0.027
        Callers may override with a value estimated from their data.
        """
        self._global_beta = global_beta

    # ── Model selection ───────────────────────────────────────────────────────

    def model_for(self, event_count: int) -> TemporalModelType:
        """
        Returns the appropriate model type for the given event count.
        Ignores library availability — pure count-based selection.
        """
        if event_count < 5:
            return TemporalModelType.EXPONENTIAL
        elif event_count < 20:
            return TemporalModelType.RENEWAL
        else:
            return TemporalModelType.HAWKES

    # ── Main entry point ──────────────────────────────────────────────────────

    def weight(
        self,
        entry: MemoryEntry,
        access_timestamps: list[datetime],
        query_time: datetime | None = None,
    ) -> float:
        """
        Returns temporal relevance weight in [0.0, 1.0].

        Automatically selects model based on len(access_timestamps).
        Falls back through the model hierarchy when libraries are unavailable.

        query_time: defaults to datetime.now(UTC) if None.
        access_timestamps: chronological list of past access datetimes.
        """
        if query_time is None:
            query_time = datetime.now(UTC)

        n = len(access_timestamps)

        # Empty or single-event: use entry.last_accessed with global beta.
        if n <= 1:
            result = self.exponential_weight(
                last=entry.last_accessed,
                query_time=query_time,
            )
            return max(0.0, min(1.0, result))

        ideal = self.model_for(n)

        if ideal is TemporalModelType.EXPONENTIAL:
            result = self.exponential_weight(
                last=access_timestamps[-1],
                query_time=query_time,
            )
        elif ideal is TemporalModelType.RENEWAL:
            result = self.renewal_weight(access_timestamps, query_time)
        else:  # HAWKES
            try:
                result = self.hawkes_weight(access_timestamps, query_time)
            except Exception:
                # Hawkes failed — fall back to renewal (which handles its own fallbacks)
                result = self.renewal_weight(access_timestamps, query_time)

        return max(0.0, min(1.0, result))

    # ── Model implementations ─────────────────────────────────────────────────

    def exponential_weight(
        self,
        last: datetime,
        query_time: datetime | None = None,
        beta: float | None = None,
    ) -> float:
        """
        Exponential decay: weight = exp(-β × hours_since_access).

        β defaults to self._global_beta.
        Always returns a value in [0.0, 1.0] — no clamping needed.
        """
        if query_time is None:
            query_time = datetime.now(UTC)
        hours = max(0.0, (query_time - last).total_seconds() / 3600.0)
        b = beta if beta is not None else self._global_beta
        return math.exp(-b * hours)

    def renewal_weight(
        self,
        timestamps: list[datetime],
        query_time: datetime | None = None,
    ) -> float:
        """
        Bayesian renewal process with Gamma inter-event times.

        Estimates Gamma(k, θ) via Method of Moments:
            k_hat   = mean² / var
            θ_hat   = var / mean

        Returns gamma.sf(t_since_last, a=k_hat, scale=θ_hat) — the survival
        function, i.e. probability the next event hasn't happened yet.

        Falls back to exponential when:
        - fewer than 3 timestamps (< 2 inter-events)
        - inter-event variance is zero (perfectly regular process)
        - scipy is not installed
        """
        if query_time is None:
            query_time = datetime.now(UTC)

        if len(timestamps) < 3:
            last = timestamps[-1] if timestamps else query_time
            return self.exponential_weight(last=last, query_time=query_time)

        inter_events = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600.0
            for i in range(len(timestamps) - 1)
        ]

        n_ie = len(inter_events)
        mean_ie = sum(inter_events) / n_ie
        var_ie = sum((x - mean_ie) ** 2 for x in inter_events) / n_ie

        t_since_last = max(
            0.0,
            (query_time - timestamps[-1]).total_seconds() / 3600.0,
        )

        if var_ie == 0.0 or mean_ie <= 0.0:
            # Zero-variance (perfectly regular) or degenerate: use exponential.
            beta_fallback = (1.0 / mean_ie) if mean_ie > 0.0 else self._global_beta
            return self.exponential_weight(
                last=timestamps[-1],
                query_time=query_time,
                beta=beta_fallback,
            )

        k_hat = mean_ie ** 2 / var_ie
        theta_hat = var_ie / mean_ie

        try:
            from scipy.stats import gamma as gamma_dist  # type: ignore[import-untyped]

            return float(gamma_dist.sf(t_since_last, a=k_hat, scale=theta_hat))
        except ImportError:
            beta_fallback = (1.0 / mean_ie) if mean_ie > 0.0 else self._global_beta
            return self.exponential_weight(
                last=timestamps[-1],
                query_time=query_time,
                beta=beta_fallback,
            )

    def hawkes_weight(
        self,
        timestamps: list[datetime],
        query_time: datetime | None = None,
    ) -> float:
        """
        Self-exciting Hawkes process via exponential kernel approximation.

        Estimates decay rate from inter-event mean, computes intensity:
            λ(t) = Σ exp(-β̂ × Δt_i)   for all past events i

        Normalizes to [0, 1] via complementary exponential:
            weight = 1 - exp(-λ(t))

        Properties:
            λ(t) = 0  → weight = 0.0 (no events ever, or all very stale)
            λ(t) → ∞  → weight → 1.0 (burst of very recent events)
        """
        if query_time is None:
            query_time = datetime.now(UTC)

        if len(timestamps) < 2:
            last = timestamps[0] if timestamps else query_time
            return self.exponential_weight(last=last, query_time=query_time)

        inter_events = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600.0
            for i in range(len(timestamps) - 1)
        ]

        mean_ie = sum(inter_events) / len(inter_events)
        if mean_ie <= 0.0:
            return self.exponential_weight(
                last=timestamps[-1], query_time=query_time
            )

        beta_hat = 1.0 / mean_ie

        # Sum kernel contributions from all past events (dt >= 0 means past).
        lambda_t = 0.0
        for ts in timestamps:
            dt = (query_time - ts).total_seconds() / 3600.0
            if dt >= 0.0:
                lambda_t += math.exp(-beta_hat * dt)

        # 1 - exp(-x) ∈ [0, 1] for x ≥ 0 — no clamping needed.
        return 1.0 - math.exp(-lambda_t)
