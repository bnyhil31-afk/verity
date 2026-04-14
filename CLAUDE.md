# Verity — Claude Code Briefing

Read this entire file before touching any code.
It describes both what currently exists AND what to build next.

---

## What Verity Is

Verity is a **cognitive memory system** — an embedded Python library that brings
genuine neuroscience-inspired memory architecture to AI agents and applications.

The unique contribution: **Mem0's adoptability + HippoRAG's cognitive architecture
+ zero-dependency default.** One-line install. Seven-method API. No GPU. No cloud.
No API key required. Runs on a Raspberry Pi or in a Kubernetes pod identically.

Verity solves three problems nobody else has solved:
1. **Reconsolidation stability** — updating memories on access without runaway drift
2. **Sleep consolidation** — offline decay, pruning, and abstraction between sessions
3. **Tiered temporal weighting** — auto-graduating temporal models as data grows

Repository: https://github.com/bnyhil31-afk/verity

---

## The Five Invariants — These Cannot Change

If you are about to change anything that touches these, stop and ask.

1. **Crisis barrier is absolute.** `crisis.py` runs first, before anything else,
   on every ingestion call. It cannot be disabled, configured away, or bypassed.
   The canary tests verify this at every boot. **Do not touch `crisis.py`.**

2. **Everything is recorded.** The Merkle-chained audit trail is append-only.
   `AuditEvent` records are never modified or deleted. The `verify_chain()` method
   detects any tampering. **Do not add UPDATE or DELETE paths to audit records.**

3. **Humans decide.** GOVERN checkpoints cannot be automated away. Veto is the
   default — timeout = veto, not approval. This is enforced in `engine.py`.

4. **Nothing without consent.** The consent gate runs before any graph traversal.
   `ConsentRequiredError` halts the operation.

5. **Uncertainty is mandatory.** `ContextBundle.uncertainty` is never Optional.
   `ContextBundle.reasoning_trace` is never empty. Enforced in `__post_init__`.

---

## What Already Exists — Do Not Rebuild

These are complete, tested, and CI-green. **Do not modify unless a phase
explicitly requires it.**

```
verity/
├── __init__.py                  Public API — all exports
├── cli.py                       verity init / verity status / verity connect
└── core/
    ├── __init__.py              Re-exports from types.py
    ├── types.py                 ALL data contracts — TypedFact, ContextBundle,
    │                            WeightedEdge, ThreeAxisWeight, DecayParameters,
    │                            ModuleManifest, ConsentRecord, AuditEvent, etc.
    │                            Enums are proper StrEnum (Python 3.11+).
    ├── exceptions.py            Full exception hierarchy
    ├── crisis.py                Absolute crisis barrier — DO NOT TOUCH
    ├── principles.py            Boot-time verifier + canary test runner
    ├── profiles.py              PERSONAL/DEVELOPER/PROFESSIONAL/ENTERPRISE profiles
    ├── engine.py                RELATE/NAVIGATE/GOVERN/REMEMBER engine
    │                            Profile-aware. Accepts str|dict|ConnectorRecord.
    │                            Session.ingest_from(connector, resource) works.
    │                            _PPR_THRESHOLD = 50_000 (correct value).
    └── graph_store/
        ├── __init__.py          GraphStore Protocol
        ├── rdflib_store.py      Personal tier — auto-detects pyoxigraph (37x faster)
        └── registry.py          Backend factory (VERITY_GRAPH_BACKEND env var)
    └── connectors/
        ├── __init__.py          Connector Protocol (read/write/describe)
        │                        ConnectorRecord dataclass
        ├── base.py              BaseConnector with lifecycle management
        ├── filesystem.py        Local files (.txt/.md/.json/.csv/.yaml)
        ├── mcp_client.py        Any MCP server as a Verity connector
        ├── dlt_connector.py     dlt wrapper (60+ data sources)
        ├── dlt_sources.py       Pre-configured factories (REST API, SQL, GitHub)
        └── registry.py          ConnectorRegistry with entry-point discovery

tests/
├── conftest.py                  Shared fixtures — fresh_store, started_engine,
│                                sample_consent
├── test_types.py                Contract invariants
├── test_crisis.py               Canary tests — MUST ALWAYS PASS
├── test_graph_store.py          rdflib backend + Protocol compliance
├── test_engine.py               RELATE/NAVIGATE/GOVERN/REMEMBER integration
├── test_connectors.py           Connector Protocol compliance
├── test_profiles.py             Profile system
└── test_cli.py                  CLI commands
```

---

## The New Layer to Build — Cognitive Architecture

This is what does NOT exist yet. Build it in the order listed.

The cognitive layer lives in `verity/cognitive/` and exposes a simple
bolt-on API at `verity/memory.py`.

**The Key Design Constraint:** Zero dependencies by default.
- Tier 0 (stdlib): `sqlite3` + `json` — text storage, no semantic search
- Tier 1 (+ `numpy`): Model2Vec static embeddings + brute-force cosine
- Tier 2 (+ `hnswlib`): Fast approximate search for >10k memories
- Tier 3 (+ `sentence-transformers`): Higher quality embeddings on CPU
- Tier 4 (+ LLM client): Cloud-enhanced summarization (opt-in)

Never make Tier 1-4 features hard dependencies. Always detect and degrade.

---

## Coding Plan — New Phases

Complete each phase fully before starting the next.
Run `pytest tests/ -v && ruff check verity/ tests/` after every phase.

---

### Phase A — Cognitive Types

**Goal:** Contracts before code. No logic.

**File:** `verity/cognitive/__init__.py` — empty package init

**File:** `verity/cognitive/types.py`

Define these frozen dataclasses and enums:

```python
from enum import StrEnum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

class MemoryTier(StrEnum):
    FAST   = "fast"   # Episodic buffer — recent, high detail, limited capacity
    SLOW   = "slow"   # Semantic store — consolidated, abstracted, persistent

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
        import math
        n = self.alpha + self.beta
        p = self.bayesian_confidence
        return 2 * 1.96 * math.sqrt(p * (1 - p) / max(n, 1))

@dataclass
class ImportanceWeights:
    """Tunable weights for composite importance scoring."""
    surprise_weight: float    = 0.35  # Prediction error / embedding distance
    recency_weight: float     = 0.30  # How recent the memory is
    reference_weight: float   = 0.20  # How often recalled
    relevance_weight: float   = 0.15  # Query-time relevance

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
```

**Tests:** `tests/test_cognitive/test_types.py`
- All dataclasses construct correctly
- `bayesian_confidence` returns alpha/(alpha+beta)
- `confidence_interval_width` narrows as alpha+beta grows
- ConfidenceTier enum serializes correctly as string

Commit: `cognitive: add types — MemoryEntry, ConfidenceTier, SleepCycleResult`

---

### Phase B — Dual-Speed Store

**Goal:** The CLS dual-memory architecture as SQLite + numpy. Zero-config.

**File:** `verity/cognitive/store.py`

```python
class DualSpeedStore:
    """
    Complementary Learning Systems theory as a Python class.

    Fast buffer (hippocampal): SQLite table, capacity-limited ring buffer.
    Slow store (neocortical): SQLite table, promoted consolidated memories.

    Both tables share one .db file. Default: ~/.verity/memory.db
    In-memory option for testing: path=":memory:"

    Embedding is optional. If numpy is available, embeddings are computed
    and stored as BLOB. If not, semantic search falls back to BM25-style
    keyword matching.
    """

    def __init__(
        self,
        path: str = "~/.verity/memory.db",
        fast_capacity: int = 500,         # ring buffer size
        embedding_model: str = "auto",    # "none"|"model2vec"|"sentence-transformers"
        user_id: str = "default",
    ): ...

    def add(self, content: str, metadata: dict = {}) -> MemoryEntry: ...
    def search(self, query: str, k: int = 5) -> list[RetrievalResult]: ...
    def get(self, memory_id: str) -> MemoryEntry | None: ...
    def update(self, memory_id: str, new_content: str) -> MemoryEntry: ...
    def delete(self, memory_id: str) -> bool: ...
    def all_fast(self) -> list[MemoryEntry]: ...    # for consolidation
    def all_slow(self) -> list[MemoryEntry]: ...    # for consolidation
    def promote(self, memory_id: str) -> MemoryEntry: ...  # fast → slow
    def stats(self) -> dict: ...
```

Implementation notes:
- SQLite schema: two tables (`fast_memories`, `slow_memories`) with identical columns
- Embeddings stored as BLOB (numpy array serialized via `numpy.frombuffer`)
- When fast_capacity is reached: evict by lowest importance score (not FIFO)
- Embedding model detection order: model2vec → sentence-transformers → None
- model2vec: `pip install model2vec` — 500x faster than MiniLM, numpy-only, no GPU
- Fallback search (no embeddings): simple trigram overlap on content field

**Tests:** `tests/test_cognitive/test_store.py`
- Store init creates tables
- add() returns MemoryEntry with memory_id
- search() returns <= k results
- get() returns None for unknown id
- update() changes content, increments access_count
- delete() removes memory, returns True
- Capacity limit: adding beyond fast_capacity evicts lowest-importance entry
- promote() moves entry from fast to slow table
- In-memory path (":memory:") works for testing
- Works with embedding_model="none" (no numpy required)

Commit: `cognitive: add DualSpeedStore — SQLite CLS implementation`

---

### Phase C — Importance Scoring

**Goal:** Prediction error as reward proxy. Composite importance = surprise × recency × reference.

**File:** `verity/cognitive/scoring.py`

```python
class ImportanceScorer:
    """
    Computes composite importance scores using prediction error as
    a dopamine/norepinephrine proxy.

    Formula: importance = w_s × surprise + w_r × recency + w_ref × reference
    Where surprise = 1 - cosine_similarity(embedding, running_centroid)

    The running centroid (exponential moving average of all embeddings)
    represents the system's "current expectation." High deviation from
    expectation = high surprise = high importance.

    For memories without embeddings: use normalized recency + reference count.
    """

    def __init__(self, weights: ImportanceWeights = ImportanceWeights()): ...

    def score(
        self,
        entry: MemoryEntry,
        query_embedding: list[float] | None = None,
    ) -> float:
        """Compute composite importance score [0.0, 1.0]."""
        ...

    def surprise(self, embedding: list[float]) -> float:
        """1 - cosine_similarity(embedding, running_centroid)."""
        ...

    def recency_decay(self, last_accessed: datetime) -> float:
        """Exponential decay: 0.995^hours_since_access"""
        ...

    def reference_boost(self, access_count: int) -> float:
        """1 + 0.1 × access_count (capped at 2.0)"""
        ...

    def update_centroid(self, embedding: list[float]) -> None:
        """Exponential moving average update: μ = 0.99μ + 0.01×embedding"""
        ...

    def record_signal(
        self,
        memory_id: str,
        signal_type: str,  # "recall" | "correction" | "dwell" | "reference"
        weight: float = 1.0,
    ) -> None:
        """Record implicit feedback signal. Stored in-memory, used to adjust scores."""
        ...
```

**Tests:** `tests/test_cognitive/test_scoring.py`
- surprise() returns 1.0 for completely orthogonal embedding
- surprise() returns ~0.0 for identical embedding
- recency_decay() decreases as hours increase
- reference_boost() increases with access_count, caps at 2.0
- score() returns value in [0.0, 1.0]
- update_centroid() shifts centroid toward new embedding

Commit: `cognitive: add ImportanceScorer — prediction error as reward proxy`

---

### Phase D — Reconsolidation Engine

**Goal:** The unsolved problem #1. Stable memory updating that cannot drift.

**File:** `verity/cognitive/reconsolidation.py`

This is the most novel component. Read the design carefully.

```python
class ReconsolidationEngine:
    """
    Implements memory reconsolidation with biological boundary conditions.

    The key insight: retrieving a memory only triggers modification if
    prediction error exceeds a threshold. Strong, old, multiply-confirmed
    memories require much higher prediction error to destabilize.

    Four protection tiers (maps to MemoryEntry.confidence_tier):
    - IMMUTABLE:  confidence >= 0.95, sources >= 5 → never modified
    - PROTECTED:  confidence >= 0.80, sources >= 3 → max_delta = 0.1
    - MODIFIABLE: confidence >= 0.50, sources >= 1 → standard rules
    - LABILE:     confidence < 0.50 → freely modifiable

    Modification gate: sigmoid(k × (prediction_error - threshold))
    where k=10 (steep), threshold varies by tier.
    """

    def should_reconsolidate(
        self,
        entry: MemoryEntry,
        prediction_error: float,  # [0.0, 1.0] — embedding distance from query
    ) -> bool:
        """
        Returns True only if prediction_error exceeds the tier threshold.
        IMMUTABLE memories always return False.
        """
        ...

    def update(
        self,
        entry: MemoryEntry,
        new_content: str,
        prediction_error: float,
        source_confirmed: bool = False,  # New independent source?
    ) -> MemoryEntry:
        """
        Conditionally update a memory based on reconsolidation gate.
        If gate is closed: return entry unchanged.
        If gate is open: update content, adjust Bayesian confidence,
        update confidence_tier based on new alpha/beta.

        Bayesian update rules:
        - New content confirms existing: alpha += 1
        - New content contradicts existing: beta += 1
        - New independent source: source_count += 1

        Tier promotion/demotion is automatic from alpha/beta values.
        """
        ...

    def tier_thresholds(self) -> dict[ConfidenceTier, float]:
        """
        Prediction error required to trigger reconsolidation per tier.
        LABILE: 0.1, MODIFIABLE: 0.3, PROTECTED: 0.6, IMMUTABLE: inf
        """
        ...

    def gate(self, prediction_error: float, threshold: float) -> float:
        """
        Sigmoid gate: σ(k × (PE - threshold))
        Returns value in [0, 1] — probability of reconsolidation.
        """
        import math
        k = 10.0
        return 1 / (1 + math.exp(-k * (prediction_error - threshold)))
```

**Tests:** `tests/test_cognitive/test_reconsolidation.py`
- IMMUTABLE entry never reconsolidates regardless of prediction_error
- LABILE entry reconsolidates at low prediction_error (0.15)
- MODIFIABLE entry requires higher prediction_error (0.35)
- Bayesian update: confirming evidence increases alpha
- Bayesian update: contradicting evidence increases beta
- Tier promotion: LABILE → MODIFIABLE when confidence crosses 0.50
- Tier demotion: PROTECTED → MODIFIABLE when contradictions pile up
- gate(0.0, 0.3) ≈ 0.05 (very unlikely)
- gate(0.9, 0.3) ≈ 0.99 (very likely)

Commit: `cognitive: add ReconsolidationEngine — 4-tier stable belief revision`

---

### Phase E — Sleep Consolidation

**Goal:** The unsolved problem #2. Offline processing between sessions.

**File:** `verity/cognitive/consolidation.py`

```python
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
        decay_factor: float = 0.90,       # Global strength multiplier per cycle
        prune_threshold: float = 0.05,    # Delete memories below this strength
        cluster_min_size: int = 3,        # Min cluster size to trigger abstraction
        similarity_threshold: float = 0.85,  # Cosine similarity to cluster
        summarizer = None,                # Optional: callable(list[str]) -> str
    ): ...

    def run(self) -> SleepCycleResult:
        """
        Run a full consolidation cycle. Returns summary of what changed.
        Safe to call at any time — idempotent if called twice in quick succession.
        """
        ...

    def decay_pass(self) -> int:
        """
        Multiply all memory strengths by decay_factor.
        Updates importance scores after decay.
        Returns count of memories decayed.
        """
        ...

    def prune_pass(self) -> int:
        """
        Delete memories with strength < prune_threshold.
        IMMUTABLE and PROTECTED memories are exempt from pruning.
        Returns count of memories pruned.
        """
        ...

    def abstract_pass(self) -> int:
        """
        Cluster fast-buffer memories by embedding similarity.
        For clusters >= cluster_min_size:
        - Create a slow-store abstraction (centroid content or LLM summary)
        - Promote the highest-importance member to slow store
        - Delete remaining cluster members from fast buffer
        Returns count of abstractions created.
        """
        ...
```

**Tests:** `tests/test_cognitive/test_consolidation.py`
- decay_pass() reduces all strengths by decay_factor
- prune_pass() removes entries below threshold
- prune_pass() preserves IMMUTABLE and PROTECTED entries
- abstract_pass() with cluster of 4 similar memories creates 1 abstraction
- Full run() returns accurate SleepCycleResult counts
- Cycle is safe to run on empty store
- IMMUTABLE memories survive all three passes

Commit: `cognitive: add ConsolidationCycle — sleep phases decay/prune/abstract`

---

### Phase F — Tiered Temporal Model

**Goal:** The unsolved problem #3. Temporal weighting that auto-graduates.

**File:** `verity/cognitive/temporal.py`

```python
class TemporalWeighter:
    """
    Auto-selects temporal model based on event history density.

    0-5 events:  Exponential decay with population-average parameters.
                 weight = exp(-β × hours_since_access)
                 β estimated from global statistics across all memories.

    5-20 events: Bayesian renewal process with Gamma inter-event times.
                 Uses scipy.stats.gamma if available, else falls back to
                 exponential with per-memory estimated rate.

    20+ events:  Lightweight Hawkes process with empirical Bayes priors.
                 Uses tick library if available, else Neural Hawkes
                 approximation via LSTM (if torch available), else
                 falls back to Bayesian renewal.

    The model is selected per-memory at retrieval time, not globally.
    This means sparse memories use exponential, dense memories use Hawkes,
    within the same system simultaneously.
    """

    def weight(
        self,
        entry: MemoryEntry,
        access_timestamps: list[datetime],
        query_time: datetime | None = None,
    ) -> float:
        """
        Returns temporal relevance weight [0.0, 1.0].
        Automatically selects model based on len(access_timestamps).
        """
        ...

    def model_for(self, event_count: int) -> TemporalModelType:
        """Returns the appropriate model type for given event count."""
        if event_count < 5:
            return TemporalModelType.EXPONENTIAL
        elif event_count < 20:
            return TemporalModelType.RENEWAL
        else:
            return TemporalModelType.HAWKES

    def exponential_weight(
        self,
        last_access: datetime,
        beta: float | None = None,  # None = use global estimate
    ) -> float: ...

    def renewal_weight(
        self,
        timestamps: list[datetime],
        query_time: datetime,
    ) -> float: ...

    def hawkes_weight(
        self,
        timestamps: list[datetime],
        query_time: datetime,
    ) -> float: ...
```

**Tests:** `tests/test_cognitive/test_temporal.py`
- model_for(3) returns EXPONENTIAL
- model_for(10) returns RENEWAL
- model_for(25) returns HAWKES
- exponential_weight() decreases as time since access increases
- weight() returns value in [0.0, 1.0] for all models
- Works with only stdlib (no scipy, no torch)
- Gracefully degrades when scipy unavailable

Commit: `cognitive: add TemporalWeighter — auto-graduating temporal models`

---

### Phase G — Global Workspace

**Goal:** K=5 competitive selection. Position-optimized output.

**File:** `verity/cognitive/workspace.py`

```python
class GlobalWorkspace:
    """
    Implements Global Workspace Theory for context selection.

    NOT a general retrieval system — a capacity-limited broadcast buffer.
    Given candidates from search, selects the top-K most relevant using
    goal-directed scoring. Applies position-aware reordering to mitigate
    the 'lost in the middle' effect.

    K=5 default (Cowan 2001: 4±1 working memory capacity).
    Configurable to 3-7.

    Scoring: salience × relevance × recency × top_down_weight
    Output order: [rank-1, rank-3, rank-5, rank-4, rank-2]
    (best and second-best at boundaries, rest in middle)
    """

    def __init__(
        self,
        capacity: int = 5,        # 3-7 recommended
        scorer: ImportanceScorer | None = None,
        temporal: TemporalWeighter | None = None,
    ): ...

    def select(
        self,
        candidates: list[RetrievalResult],
        goal: str | None = None,          # Current task/query for top-down bias
        goal_embedding: list[float] | None = None,
    ) -> list[RetrievalResult]:
        """
        Competitive selection into capacity-limited buffer.
        Returns <= self.capacity items in position-optimized order.
        """
        ...

    def _composite_score(
        self,
        result: RetrievalResult,
        goal_embedding: list[float] | None,
    ) -> float:
        """salience × relevance × recency × top_down_weight"""
        ...

    def _position_reorder(self, ranked: list[RetrievalResult]) -> list[RetrievalResult]:
        """
        Reorders for LLM consumption to mitigate lost-in-the-middle:
        [rank-1, rank-3, rank-5, ..., rank-4, rank-2]
        Best evidence at start and end. Middle slots for supporting context.
        """
        ...
```

**Tests:** `tests/test_cognitive/test_workspace.py`
- select() returns <= capacity items
- select() with 20 candidates returns exactly 5
- Position reordering places rank-1 first
- Position reordering places rank-2 last
- goal_embedding biases selection toward topically relevant memories
- Works with capacity=3 and capacity=7

Commit: `cognitive: add GlobalWorkspace — K=5 competitive context selection`

---

### Phase H — Simple API

**Goal:** Mem0 adoptability. Seven methods. Zero config.

**File:** `verity/memory.py`

This is the user-facing bolt-on API. It wires together the cognitive layer.

```python
class Memory:
    """
    Verity's bolt-on memory API.

    Zero config:
        m = Memory()                    # SQLite in ~/.verity/memory.db
        m.add("I prefer dark mode")     # Store
        m.search("preferences")         # Retrieve

    Full config:
        m = Memory(
            path="/data/memory.db",
            user_id="alice",
            capacity=500,
            embedding_model="sentence-transformers",
        )

    Works with or without numpy, sentence-transformers, scipy, torch.
    Automatically uses the best available backend.
    """

    def __init__(
        self,
        path: str = "~/.verity/memory.db",
        user_id: str = "default",
        capacity: int = 500,
        embedding_model: str = "auto",  # "none"|"model2vec"|"sentence-transformers"|"auto"
    ): ...

    # ── Core 7 methods ────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        metadata: dict = {},
        importance: float | None = None,  # None = auto-compute
    ) -> str:
        """Store a memory. Returns memory_id."""
        ...

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: str | None = None,
    ) -> list[dict]:
        """
        Semantic search. Returns list of dicts with keys:
        id, content, score, metadata, confidence, strength
        """
        ...

    def get(self, memory_id: str) -> dict | None:
        """Fetch one memory by ID. Returns dict or None."""
        ...

    def update(self, memory_id: str, content: str) -> dict:
        """Update memory content. Applies reconsolidation rules."""
        ...

    def delete(self, memory_id: str) -> bool:
        """Delete a memory. GDPR Article 17 compliant."""
        ...

    def consolidate(self) -> dict:
        """
        Run sleep consolidation cycle.
        Returns: {decayed, pruned, merged, abstractions, duration_seconds}
        """
        ...

    def export(self, format: str = "json") -> str | dict:
        """
        Export all memories. GDPR Article 20 compliant.
        format: "json" | "csv"
        """
        ...

    # ── Async variants ────────────────────────────────────────────────────
    # Each method above has an async equivalent: aadd, asearch, aget, etc.

    # ── Context manager ───────────────────────────────────────────────────
    def __enter__(self): ...
    def __exit__(self, *args): ...
    async def __aenter__(self): ...
    async def __aexit__(self, *args): ...
```

Export `Memory` from `verity/__init__.py` alongside the existing exports.

Update `pyproject.toml`:
- Add `[project.optional-dependencies]` section `cognitive`:
  ```
  model2vec>=0.3.0
  hnswlib>=0.8.0
  ```
- Add `full` extra combining `fast` + `connectors` + `cognitive`

**Tests:** `tests/test_memory.py`
- Zero-dependency path works (embedding_model="none")
- add() returns a string ID
- search() returns list of dicts with required keys
- get() returns None for unknown ID
- update() modifies content
- delete() returns True, subsequent get() returns None
- consolidate() returns dict with expected keys
- export() returns valid JSON string
- Async variants (aadd, asearch, etc.) work
- Context manager (with Memory() as m:) works

Commit: `cognitive: add Memory API — zero-config bolt-on with 7 methods`

---

## How to Run Tests

```bash
# Install in dev mode with all optional extras
pip install -e ".[dev,cognitive,connectors,fast]"

# All tests
pytest tests/ -v

# Just cognitive tests
pytest tests/test_cognitive/ -v

# Coverage
pytest tests/ -v --cov=verity --cov-report=term-missing

# Lint
ruff check verity/ tests/

# Type check (informational)
mypy verity/
```

CI runs on push to main via `.github/workflows/ci.yml`.
**Never push with failing tests.**

---

## Commit Message Format

```
scope: short description
```

Examples:
```
cognitive: add types — MemoryEntry, ConfidenceTier, SleepCycleResult
cognitive: add DualSpeedStore — SQLite CLS implementation
cognitive: add ReconsolidationEngine — 4-tier stable belief revision
cognitive: add Memory API — zero-config bolt-on with 7 methods
```

One line. Under 72 characters. Imperative mood.

---

## What You Must Not Do

- Do not modify `crisis.py` for any reason
- Do not add UPDATE or DELETE paths to audit records (`AuditEvent`)
- Do not make `uncertainty` Optional on `ContextBundle`
- Do not make `reasoning_trace` Optional or allow empty tuple
- Do not bypass `_validate_consent()` in the navigate path
- Do not change the three Named Graph URIs
- Do not add a fourth Named Graph
- Do not remove the power-law decay formula from `DecayParameters`
- Do not collapse `ThreeAxisWeight` to a single float
- Do not break existing passing tests — fix them if the change requires it
- Do not add non-optional dependencies to `[dependencies]` in pyproject.toml
- Do not require numpy, scipy, torch, or any ML library in the zero-dep path

---

## Internal Architectural Vocabulary

These patterns exist in the codebase. Know them so you don't conflict with them.

**VERIFY** — not a protocol, not a class. It is the immune system that runs
*inside* every core function. It checks principles integrity, consent validity,
and crisis flags before any graph operation proceeds. You will see it referenced
in design documents. It is not something you build — it is the name for the
interlocking checks already present in `crisis.py`, `principles.py`, and
`_validate_consent()`.

**SOMA** — the integration layer inside `navigate()` that assembles a
`ContextBundle` from raw traversal results. It is not a separate file or class —
it describes the assembly logic: scoring, ranking, uncertainty annotation, and
`agent_prompt` formatting that happens between graph traversal and returning
context to the caller. When you see "SOMA" in design documents, it means that
assembly step in `engine.py`'s `navigate()` method.

Neither VERIFY nor SOMA needs new files. They are patterns, not modules.

---

## Benchmark Plan — Active Work

The next phase is deterministic benchmarking of Verity's three core claims.
All four sessions are LLM-free and run in CI.

### Structure

```
benchmarks/
├── __init__.py
├── conftest.py
├── data/
│   ├── __init__.py
│   └── generator.py         # deterministic synthetic datasets (seeded)
├── test_invariants.py       # Hypothesis property + stateful tests
├── test_claims.py           # three core claim validation suites
├── test_retrieval.py        # golden set + 8-condition ablation study
├── test_performance.py      # pytest-benchmark latency + memory
└── README.md
```

### Key design decisions — do not change without asking

1. **Synthetic embeddings**: `generator.py` creates structured numpy unit
   vectors. Each semantic topic maps to a distinct direction with small
   gaussian noise (σ=0.05). Works everywhere, no GPU, no real model.
   Uses `DualSpeedStore.add(..., _embedding=vector)` to inject.
   64 dimensions. 5 topic centroids (orthogonal). Seeded: rng(42).

2. **Ablation uses lower-level APIs**: `DualSpeedStore` + individual
   cognitive components directly (not `Memory()`), since `Memory` hardwires
   all components. Enables toggling each component without touching the API.

3. **Importance stratification**: Uses Bayesian confidence tiers
   (IMMUTABLE/PROTECTED vs LABILE), not raw importance scores.
   `decay_pass()` is uniform — only protection tiers exempt from pruning.
   High-importance: alpha=250, beta=10. Low-importance: alpha=2, beta=5.

4. **Telephone game test**: Uses `ReconsolidationEngine` directly,
   not `Memory.update()`, to isolate reconsolidation logic from I/O.

5. **Calibration test**: Structural calibration only (CI width narrows
   with source_count, rank correlation between confidence and source_count).
   Full ECE deferred to community benchmarks (needs LLM ground truth).

6. **Retrieval invariant**: Hit Rate (memory appears in top-3 results
   when searching its own content prefix), not a score threshold.

### Session 1 — Foundation (current)

Creates `benchmarks/` structure, `data/generator.py`, `test_invariants.py`.
Adds `hypothesis>=6.100.0` to `[dev]` extras in `pyproject.toml`.

Five datasets from generator (all seeded, reproducible):
- `retrieval_set()` — 100 memories, 30 queries, known relevant IDs
- `interference_set()` — 20 (old, new) fact pairs, ground truth = new
- `temporal_set()` — 30 memories with 3 access patterns
- `consolidation_set()` — 5 groups × 4 similar memories, known entity lists
- `importance_set()` — 20 IMMUTABLE + 20 LABILE MemoryEntry objects

Six Hypothesis invariants:
1. Hit Rate (memory retrievable by own content prefix)
2. IMMUTABLE protection (no content change regardless of PE)
3. Consolidation monotonicity (count never increases)
4. Export consistency (export count = adds - deletes)
5. Temporal monotonicity (weights strictly decrease with time)
6. Consolidation idempotency (two runs = one run)

Commit: `bench: add generator and property invariant tests`

### Session 2 — Core Claim Validation

Creates `benchmarks/test_claims.py`.
Tests Verity's three novel claims end-to-end.
Commit: `bench: add core claim validation tests`

### Session 3 — Retrieval Quality + Ablation

Creates `benchmarks/test_retrieval.py`.
Metrics: P@1, P@5, R@5, MRR, NDCG@5, Hit@5.
8 ablation conditions with Wilcoxon + bootstrap CIs.
Commit: `bench: add retrieval quality and ablation study`

### Session 4 — Performance

Creates `benchmarks/test_performance.py`.
pytest-benchmark. Targets: P95 search < 300ms, add < 50ms.
Updates `.github/workflows/ci.yml` with benchmark job.
Adds `pytest-benchmark>=4.0` to `[dev]` extras.
Commit: `bench: add performance benchmarks and CI integration`

---

## The Three Tests (apply to every change)

**Machine Test:** Can the four functions (RELATE/NAVIGATE/GOVERN/REMEMBER)
be swapped without touching each other? If not, redesign.

**Brain Test:** Does the system preserve contextual flow and accumulated
awareness? Does it behave like a mind with memory, not a stateless query engine?

**Adoptability Test:** Can a developer with zero background in this codebase
install Verity, call `Memory().add()` and `Memory().search()`, and get a useful
result within five minutes? If not, the API is too complex.

---

## The Unique Contribution in One Sentence

Verity combines Mem0's one-line-install adoptability with HippoRAG's
cognitive architecture, solving three problems no other package has solved:
reconsolidation stability, sleep consolidation, and tiered temporal weighting —
with zero required dependencies beyond Python's stdlib.
