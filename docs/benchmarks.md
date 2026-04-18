# Benchmarks

Verity's benchmark suite validates three categories of claims: mathematical invariants, novel cognitive claims, and performance baselines. All benchmarks run in under 2 minutes and are CI-green on Python 3.11, 3.12, and 3.13.

The full benchmark code lives in `benchmarks/`. Performance benchmarks (`benchmarks/test_performance.py`) are excluded from regular CI and run only on push to main.

---

## Claim validation

These benchmarks validate Verity's three novel claims deterministically — no LLM, no network, no randomness.

### Reconsolidation stability

IMMUTABLE memories (alpha=250, beta=10, confidence ≥ 0.95, 5+ independent sources) survive 100 consecutive neutral access cycles with zero content drift.

The reconsolidation gate (PE=0.65) correctly fires at all tiers except IMMUTABLE:

| Tier | Gate fires | Lability probability |
|------|-----------|---------------------|
| LABILE | Yes | 0.996 |
| MODIFIABLE | Yes | 0.971 |
| PROTECTED | Yes | 0.623 |
| IMMUTABLE | No | 0.000 |

### Sleep consolidation

After 35 sleep cycles:

- IMMUTABLE memories: 20/20 survive
- LABILE memories: 0/20 survive (pruned)
- Fisher's exact test: p < 0.001

IMMUTABLE and PROTECTED memories are exempt from pruning. Decay (strength × 0.90 per cycle) still applies, but they do not cross the prune threshold.

### Temporal weighting

The correct temporal model activates automatically based on access history:

| History | Model activated | Score jump at boundary |
|---------|----------------|------------------------|
| n < 5 | Exponential decay (β = 0.005/hr) | < 20% |
| 5 ≤ n < 20 | Bayesian renewal (Gamma MoM) | < 20% |
| n ≥ 20 | Hawkes process (empirical β̂) | < 20% |

Tier transitions are smooth — the < 20% score jump at model boundaries ensures no discontinuities in retrieval ranking.

---

## Retrieval quality

All retrieval benchmarks use `embedding_model='none'` (keyword search, zero dependencies).

### Cognitive layer vs raw search

The full cognitive pipeline (scoring + reconsolidation + temporal weighting + global workspace selection) outperforms raw keyword search:

- Wilcoxon signed-rank test: p < 0.01
- Improvement is consistent across multiple query types

### Temporal signal contribution

Tested on an adversarial dataset where importance scores favour decoy memories (wrong answers) over correct ones — forcing the retrieval system to rely on the temporal signal to succeed:

- NDCG@5 ≥ 0.93

This validates that temporal weighting is load-bearing, not decorative.

---

## Performance baselines

Measured on `embedding_model='none'` (stdlib-only tier). These are CI-enforced targets.

| Operation | Scale | CI target |
|-----------|-------|-----------|
| `add()` | 1K entries | mean < 50ms |
| `search()` | 1K entries | mean < 100ms |
| `consolidate()` | 1K entries | mean < 2s |
| File size | 1K entries | < 1MB |

Actual results on typical hardware are well below these targets (1–20ms for add/search, < 500ms for consolidate at 1K scale).

---

## Comparison context

Published baselines from competing libraries use embedding models:

| Library | P95 latency | Notes |
|---------|------------|-------|
| Graphiti | ~300ms | Uses embeddings |
| Mem0 | ~1.44s | Uses embeddings |
| Verity (keyword) | < 100ms | Zero dependencies |

For a fair comparison against Graphiti and Mem0, use `pip install "veritycog[cognitive]"` which enables model2vec semantic search. The zero-dependency baseline is not a fair comparison to embedding-backed systems — it is a different operating mode.

Community benchmark results (LOCOMO, LongMemEval) are planned for a future release.

---

## NDCG note

Verity's `GlobalWorkspace` applies position-aware reordering before returning results:

```
Output order: [rank1, rank3, rank5, rank4, rank2]
```

This places the most important result first and second-most important last, mitigating the LLM lost-in-the-middle problem. The tradeoff is an NDCG@5 ceiling of **~0.947** — rank-2 is at output position 5, not position 2.

For standard retrieval metric comparison (LOCOMO, LongMemEval, etc.), compute NDCG on the **pre-reorder ranked list**, not the position-reordered output. The `benchmarks/README.md` file in the repository documents this in detail.
