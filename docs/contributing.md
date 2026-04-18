# Contributing

See [CONTRIBUTING.md](https://github.com/bnyhil31-afk/verity/blob/main/CONTRIBUTING.md) in the repository for the full contribution guide. This page summarises the key points.

## Dev setup

```bash
git clone https://github.com/bnyhil31-afk/verity
cd verity
pip install -e ".[dev]"
```

## Running tests

```bash
# Unit and integration tests
pytest tests/ benchmarks/ --ignore=benchmarks/test_performance.py -v

# With coverage
pytest tests/ -v --cov=verity --cov-report=term-missing

# Performance benchmarks (excluded from regular CI)
pytest benchmarks/test_performance.py -v -s
```

All tests must pass before submitting a pull request. CI runs on Python 3.11, 3.12, and 3.13.

## Linting and type checking

```bash
ruff check verity/ tests/ benchmarks/
mypy verity/
```

## Commit format

```
scope: short description
```

One line. Under 72 characters. Imperative mood. Examples:

```
cognitive: add TemporalWeighter — Hawkes process tier
fix: promote_tier fires at all tiers except IMMUTABLE
docs: update README, add quickstart example
tests: add adversarial dataset for temporal signal validation
```

## The five invariants

These cannot change. Any contribution that weakens, removes, or works around any of these will not be merged. See [Concepts — Memory Reconsolidation](concepts.md#memory-reconsolidation) for background.

1. **Human wellbeing is above all else**
2. **Crisis content triggers an absolute barrier — always** (`crisis.py` runs first, cannot be disabled)
3. **Everything is recorded** — the Merkle-chained audit trail is append-only. No UPDATE or DELETE paths on `AuditEvent`.
4. **Humans decide** — GOVERN checkpoints cannot be automated. Timeout = veto.
5. **Nothing without consent** — the consent gate runs before any graph traversal.

The canary tests in `tests/test_crisis.py` verify invariants 2 and 3 on every boot. Do not modify them to make your code pass.

## Cognitive layer rules

The cognitive layer (`verity/cognitive/`) has additional constraints:

1. No file in `verity/cognitive/` may import from `verity/core/engine.py`
2. All components must work with `embedding_model="none"` (zero dependencies)
3. Optional dependencies (`numpy`, `hnswlib`, `model2vec`) must be detected at use-time with `try/import`, not at module load time
4. Tests for cognitive components live in `tests/`

## Pull request checklist

- [ ] All existing tests pass
- [ ] New behaviour has test coverage
- [ ] Linting passes (`ruff check`)
- [ ] The five invariants are intact
- [ ] Commit messages follow the format above
- [ ] Machine Test passes: can the four core functions be swapped independently?
- [ ] Brain Test passes: does the system behave like a mind with memory?
