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
