# Verity Benchmark Suite

Property-based and data-driven benchmarks for Verity's cognitive memory system.

## How to Run

```bash
# Install all required extras (numpy pulled in via cognitive)
pip install -e ".[dev,cognitive,connectors,fast]"

# All benchmark tests
pytest benchmarks/ -v

# Slow / deep property tests only
pytest benchmarks/ -v -m slow

# Skip slow tests (fast CI path)
pytest benchmarks/ -v -m "not slow"

# With coverage
pytest benchmarks/ -v --cov=verity
```

## Files

| File | What it tests |
|---|---|
| `data/generator.py` | Deterministic synthetic datasets (seeded, reproducible) |
| `conftest.py` | Shared `fresh_memory` fixture and `make_embedding` helper |
| `test_invariants.py` | Six Hypothesis property tests for core correctness invariants |

## Invariants Tested

1. **Hit Rate** — a stored memory is always retrievable by its own content prefix
2. **IMMUTABLE guard** — IMMUTABLE-tier memories never change regardless of prediction error
3. **Consolidation ↓** — one consolidation cycle never increases the total memory count
4. **Export consistency** — `len(export) == adds - confirmed_deletes`
5. **Temporal ↓** — temporal weight is non-increasing as query time advances
6. **Idempotency** — second consolidation run leaves count ≤ first run count

## Notes

- `@given` tests use `max_examples=50` by default for fast CI (< 30 s total)
- Tests marked `@pytest.mark.slow` use `max_examples=200` / `stateful_step_count=50`
  for deeper exploration in nightly or pre-release runs
- Requires numpy (installed via `.[cognitive]`); no GPU, no network
- Hypothesis database is stored in `.hypothesis/` — commit to reproduce failures

## Performance Benchmarks

Performance tests are excluded from regular CI. Run explicitly:

    pytest benchmarks/test_performance.py -v -s

### Targets (zero-dependency baseline, embedding_model='none')

| Operation        | Scale | Target      |
|-----------------|-------|-------------|
| Memory.add()    | 100   | mean < 50ms |
| Memory.add()    | 1K    | mean < 50ms |
| Memory.search() | 100   | mean < 100ms |
| Memory.search() | 1K    | mean < 100ms |
| consolidate()   | 100   | mean < 500ms |
| consolidate()   | 1K    | mean < 2s |
| File size       | 100   | < 200KB |
| File size       | 1K    | < 1MB |

### Comparison context

Published baselines use embedding models:
- Graphiti: P95 ~300ms (with embedding model)
- Mem0: p50 0.148s, p95 1.44s (with embedding model)

To compare fairly, run Verity with `embedding_model='sentence-transformers'`.

### Notes for future LOCOMO / LongMemEval benchmarks

Verity's GlobalWorkspace applies position reordering before returning results.
With K=5, output order is [rank1, rank3, rank5, rank4, rank2]. This optimises
for LLM context placement but creates an NDCG@5 ceiling of ~0.947 (not 1.0)
because rank-2 is placed at output position 5.

When running community benchmarks: compute NDCG on the PRE-reorder ranked
list, not the position-reordered output. Otherwise Verity's scores are
unfairly penalised relative to systems without position reordering.

### Saving a regression baseline

    pytest benchmarks/test_performance.py --benchmark-save=baseline

Future regression detection:

    pytest benchmarks/test_performance.py \
      --benchmark-compare=baseline \
      --benchmark-compare-fail=mean:25%
