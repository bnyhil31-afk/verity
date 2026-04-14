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

## Current Status — All Phases Complete

All eight cognitive architecture phases are complete and CI-green
on Python 3.11, 3.12, and 3.13.

| Phase | File | Status |
|-------|------|--------|
| A | `verity/cognitive/types.py` | Complete |
| B | `verity/cognitive/store.py` | Complete |
| C | `verity/cognitive/scoring.py` | Complete |
| D | `verity/cognitive/reconsolidation.py` | Complete |
| E | `verity/cognitive/consolidation.py` | Complete |
| F | `verity/cognitive/temporal.py` | Complete |
| G | `verity/cognitive/workspace.py` | Complete |
| H | `verity/memory.py` | Complete |

## What to build next

- **Benchmarks** — run Memory against LongMemEval and LOCOMO to
  validate the neuroscience-grounded architecture actually
  outperforms standard RAG
- **MCP connector end-to-end** — prove MCPConnector works against
  a live MCP server (Google Calendar, Gmail)
- **Documentation site** — convert PRINCIPLES.md and CONTRIBUTING.md
  to a proper docs site (mkdocs-material recommended)
- **Package release** — pyproject.toml is ready; publish to PyPI

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
