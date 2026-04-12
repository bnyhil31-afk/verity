# Contributing to Verity

Thank you for your interest in contributing. This document covers
everything you need to know to contribute effectively.

---

## The Two Tests

Every architectural decision in Verity is governed by two tests.
Read these before writing any code.

**The Machine Test:** Can the four core functions (RELATE / NAVIGATE /
GOVERN / REMEMBER) be swapped for equivalent implementations without
touching each other? If no, the architecture has failed.

**The Brain Test:** Does the system preserve contextual flow and
accumulated awareness across sessions? Does it behave like a mind
with memory, not a stateless query engine?

If your contribution passes both tests, it belongs here.
If it fails either, it needs to be redesigned.

---

## Getting Started

```bash
# Clone the repo
git clone https://github.com/bnyhil31-afk/verity
cd verity

# Install in development mode with all dev dependencies
pip install -e ".[dev]"

# Run the test suite
pytest tests/ -v

# Run linting
ruff check verity/ tests/

# Run type checking
mypy verity/
```

All tests must pass before submitting a pull request.
CI runs automatically on every push and pull request.

---

## The Five Invariants

These cannot change. Any contribution that weakens, removes, or
works around any of these will not be merged. See `PRINCIPLES.md`
for the full explanation.

1. Human wellbeing is above all else
2. Crisis content triggers an absolute barrier — always
3. Everything is recorded. The record cannot be modified.
4. Humans decide. The system presents candidates.
5. Nothing happens without consent

The canary tests in `tests/test_crisis.py` verify invariants 1 and 2
directly. They run in CI. Do not modify them to make your code pass.

---

## Types of Contributions

### Bug fixes
Open an issue first if the bug is non-obvious. For clear bugs,
a pull request with a failing test that your fix resolves is ideal.

### Engine improvements
Changes to `verity/core/` must:
- Not break any existing tests
- Add tests covering the new behavior
- Pass the Machine Test and Brain Test
- Not touch the five invariants

### Domain modules
Domain modules live in **separate repositories**, not in this repo.

A domain module is an independent Python package that registers
itself via setuptools entry points:

```toml
[project.entry-points."verity.modules"]
fhir_r4 = "verity_fhir:manifest"
```

The entry point must return a `ModuleManifest` instance
(see `verity.core.types.ModuleManifest`).

Your module package should contain:
```
verity_yourmodule/
├── schema.yaml          # LinkML source — single source of truth
├── entities.py          # generated Python dataclasses
├── shapes.shacl.ttl     # generated SHACL shapes
├── ontology.owl.ttl     # generated OWL
├── recognizer.py        # entity recognizer for RELATE
├── prompts.py           # agent_prompt templates
└── tests/               # module-specific compliance tests
```

The engine discovers your module at runtime — no changes to the
core Verity package are required.

### New graph store backends
New backends implement the `GraphStore` Protocol
(`verity.core.graph_store.GraphStore`).

All protocol methods must be implemented. The backend must:
- Maintain the three Named Graphs (`urn:verity:knowledge`,
  `urn:verity:provenance`, `urn:verity:consent`)
- Preserve Merkle chain integrity across restarts
- Pass the full graph store test suite

Register your backend in `verity/core/graph_store/registry.py`.

---

## Commit Message Format

```
scope: short description — what and why

Optional longer explanation if needed.
```

Examples from this repo:
```
core: add types.py — all data contracts, no logic
core: add crisis.py — absolute barrier, runs before everything
tests: add full test suite — types, crisis, graph store, engine
```

Keep the subject line under 72 characters.
Use the imperative mood ("add", not "added").

---

## Pull Request Checklist

- [ ] All existing tests pass (`pytest tests/ -v`)
- [ ] New behavior has test coverage
- [ ] Linting passes (`ruff check verity/ tests/`)
- [ ] The five invariants are intact
- [ ] Machine Test passes
- [ ] Brain Test passes
- [ ] Commit messages follow the format above

---

## Questions

Open a GitHub Discussion for questions about architecture or direction.
Open an issue for bugs or concrete feature requests.
See `SECURITY.md` for reporting vulnerabilities.
