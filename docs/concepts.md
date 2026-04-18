# Concepts

Verity's cognitive layer maps each component to a specific neuroscience model. This page explains the science behind each one.

## Complementary Learning Systems — `DualSpeedStore`

The brain uses two systems for memory: the hippocampus (fast, episodic, short-term) and the neocortex (slow, semantic, long-term). Verity implements this as `DualSpeedStore` — two SQLite tables in a single file.

- **Fast table** (hippocampal) — new memories land here. Capacity-bounded. Rapid encoding.
- **Slow table** (neocortical) — promoted memories. Slower to update. Semantically organised.

The dual-speed architecture lets Verity store new information immediately while the slow semantic store accumulates stable long-term knowledge. All of this lives in one SQLite file with no server required.

## Predictive Processing — `ImportanceScorer`

The brain assigns importance to information based on *prediction error* — how surprising an input is relative to what was expected. High surprise → high dopamine/norepinephrine → stronger encoding.

`ImportanceScorer` approximates this with embedding cosine distance:

- A memory that is close to existing memories = low surprise = low importance
- A memory that diverges from the current centroid = high surprise = high importance

The scorer maintains a running centroid of stored embeddings. Each new memory's distance from the centroid is its prediction error proxy. Recency decay is applied as a secondary signal — recent accesses boost importance independently of surprise.

## Memory Reconsolidation — `ReconsolidationEngine`

When a memory is recalled in the brain, it becomes temporarily *labile* (unstable) and open to modification. This is called reconsolidation. Verity implements this as a 4-tier stability system.

**Stability tiers:**

| Tier | Confidence | Conditions |
|------|-----------|------------|
| LABILE | < 0.50 | New or rarely accessed |
| MODIFIABLE | ≥ 0.50 | Accessed multiple times |
| PROTECTED | ≥ 0.80 | 3+ independent sources |
| IMMUTABLE | ≥ 0.95 | 5+ independent sources |

Each tier uses a **Bayesian Beta-Bernoulli model** to track confidence. When a memory is accessed, `promote_tier()` runs with a prediction error threshold of `PE=0.65`. At this threshold:

- LABILE memories: nearly always labile on access (probability 0.996)
- MODIFIABLE memories: frequently labile (probability 0.971)
- PROTECTED memories: sometimes labile (probability 0.623)
- IMMUTABLE memories: never labile (threshold = ∞)

This mirrors real neuroscience: frequently-accessed, well-corroborated memories resist modification, while new and uncertain memories remain open to update.

!!! note "Why PE=0.65?"
    The original implementation used PE=0.2, which silently made reconsolidation a no-op above LABILE tier. This was discovered during benchmark validation and corrected. At 0.65, the gate fires correctly at all tiers except IMMUTABLE.

## Sleep Consolidation — `ConsolidationCycle`

Between sessions, the brain replays memories during sleep to decay weak traces, prune noise, and merge similar episodes into abstractions. `ConsolidationCycle` replicates this in three passes:

1. **Decay** — multiply each memory's strength by 0.90. Repeated non-access weakens memories.
2. **Prune** — remove memories below a strength threshold. IMMUTABLE and PROTECTED memories are exempt.
3. **Abstract** — cluster similar memories and replace the cluster with a single semantic abstraction.

Running `m.consolidate()` triggers this cycle manually. In production use, it can be scheduled between sessions to keep the memory store lean and semantically coherent.

## Temporal Point Processes — `TemporalWeighter`

Recency is not a single signal — it depends on the *history* of access events. Verity uses different temporal models depending on how much access history is available for a given memory:

| History | Model | Rationale |
|---------|-------|-----------|
| n < 5 events | Exponential decay (β = 0.005/hr) | Not enough data for statistical estimation |
| 5 ≤ n < 20 events | Bayesian renewal (Gamma, Method of Moments) | Sufficient for inter-arrival distribution |
| n ≥ 20 events | Hawkes process (self-exciting, empirical β̂) | Rich enough history for self-excitation |

`TemporalWeighter` selects the right model automatically, per memory, with no configuration. The model upgrades silently as access history grows.

## Global Workspace Theory — `GlobalWorkspace`

The brain's global workspace (Baars 1988, Dehaene 2001) integrates information from multiple specialised systems and selects what enters conscious awareness. Cowan (2001) established that working memory capacity is approximately 4±1 items.

`GlobalWorkspace` implements K=5 competitive selection:

1. Score all candidate memories using importance + temporal weight + relevance
2. Select the top K=5 by salience
3. **Position-aware reordering**: output as `[rank1, rank3, rank5, rank4, rank2]`

The position reordering mitigates the *lost-in-the-middle* problem in LLMs: models attend more strongly to the beginning and end of context. Placing the highest-ranked result first, the next-highest last, and lower-ranked results in the middle maximises the probability that the LLM uses the most important memories.

!!! note "NDCG ceiling"
    This reordering places rank-2 at output position 5, creating an NDCG@5 ceiling of ~0.947 by design. If you are computing retrieval metrics for standard benchmarks (LOCOMO, LongMemEval), compute NDCG on the pre-reorder ranked list, not the position-reordered output.
