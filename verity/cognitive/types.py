"""
verity.cognitive.types
======================
Data contracts for the cognitive memory layer.

No logic, no I/O, no imports from elsewhere in the package.
All downstream cognitive modules import from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class MemoryTier(StrEnum):
    FAST = "fast"   # Episodic buffer — recent, high detail, limited capacity
    SLOW = "slow"   # Semantic store — consolidated, abstracted, persistent


class ConfidenceTier(StrEnum):
    IMMUTABLE  = "immutable"   # conf >= 0.95, sources >= 5 — never modified
    PROTECTED  = "protected"   # conf >= 0.80, sources >= 3 — modification penalized
    MODIFIABLE = "modifiable"  # conf >= 0.50, sources >= 1 — standard rules apply
    LABILE     = "labile"      # conf < 0.50 — freely modifiable, eligible for pruning


class TemporalModelType(StrEnum):
    EXPONENTIAL = "exponential"  # 0-5 events — population-average parameters
    RENEWAL     = "renewal"      # 5-20 events — Bayesian Gamma inter-event times
    HAWKES      = "hawkes"       # 20+ events — Hawkes with empirical Bayes priors


@dataclass
class MemoryEntry:
    """A single memory in the dual-speed store."""
    memory_id: str                  # uuid4
    content: str                    # The raw text content
    user_id: str                    # Scope — memories are per-user
    tier: MemoryTier
    confidence_tier: ConfidenceTier
    importance: float               # [0.0, 1.0] — composite score
    strength: float                 # [0.0, 1.0] — decays over time
    created_at: datetime
    last_accessed: datetime
    access_count: int               # Drives reference_boost
    source_count: int               # Number of independent sources confirming this
    alpha: float = 2.0              # Beta-Bernoulli confirmations (Bayesian confidence)
    beta: float  = 1.0              # Beta-Bernoulli contradictions
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None  # None until computed

    @property
    def bayesian_confidence(self) -> float:
        """Expected value of Beta(alpha, beta) — calibrated confidence."""
        return self.alpha / (self.alpha + self.beta)

    @property
    def confidence_interval_width(self) -> float:
        """95% credible interval width — narrow = trustworthy."""
        n = self.alpha + self.beta
        p = self.bayesian_confidence
        return 2 * 1.96 * math.sqrt(p * (1 - p) / max(n, 1))


@dataclass
class ImportanceWeights:
    """Tunable weights for composite importance scoring."""
    surprise_weight: float   = 0.35  # Prediction error / embedding distance
    recency_weight: float    = 0.30  # How recent the memory is
    reference_weight: float  = 0.20  # How often recalled
    relevance_weight: float  = 0.15  # Query-time relevance


@dataclass
class SleepCycleResult:
    """Results from one full consolidation cycle."""
    memories_decayed: int
    memories_pruned: int
    memories_merged: int
    abstractions_created: int
    duration_seconds: float
    cycle_timestamp: datetime


@dataclass
class RetrievalResult:
    """A single result from semantic search."""
    memory: MemoryEntry
    score: float                    # Similarity + importance composite
    position: int                   # 1-indexed rank
