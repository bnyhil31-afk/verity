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

## Cognitive Layer — Complete

All eight phases are complete and CI-green on Python 3.11, 3.12, and 3.13.

**Dependency tiers (enforced, do not break):**
- Tier 0 (stdlib): `sqlite3` + `json` — text storage, keyword search
- Tier 1 (+ `numpy`): Model2Vec embeddings + brute-force cosine
- Tier 2 (+ `hnswlib`): Fast approximate search for >10k memories
- Tier 3 (+ `sentence-transformers`): Higher quality embeddings on CPU
- Tier 4 (+ LLM client): Cloud-enhanced summarization (opt-in)

Never make Tier 1-4 features hard dependencies. Always detect and degrade.

### Phase Status

| Phase | File | Status |
|-------|------|--------|
| A | `verity/cognitive/types.py` | ✅ Complete |
| B | `verity/cognitive/store.py` | ✅ Complete |
| C | `verity/cognitive/scoring.py` | ✅ Complete |
| D | `verity/cognitive/reconsolidation.py` | ✅ Complete |
| E | `verity/cognitive/consolidation.py` | ✅ Complete |
| F | `verity/cognitive/temporal.py` | ✅ Complete |
| G | `verity/cognitive/workspace.py` | ✅ Complete |
| H | `verity/memory.py` | ✅ Complete |

### Cognitive Layer File Map

```
verity/
├── memory.py                    Public bolt-on API — 7 methods, zero config
│                                add(), search(), get(), update(), delete(),
│                                consolidate(), export()
│                                Async variants: aadd(), asearch(), etc.
│                                Context manager: with Memory() as m:
└── cognitive/
    ├── types.py                 MemoryEntry, ConfidenceTier, SleepCycleResult,
    │                            TemporalModelType, MemoryTier, RetrievalResult
    ├── store.py                 DualSpeedStore — SQLite dual-table CLS
    │                            add(tier=, _embedding=), search(), get(),
    │                            delete(), update_entry(), compute_embedding(),
    │                            all_fast(), all_slow(), close(), stats()
    ├── scoring.py               ImportanceScorer — prediction error proxy
    │                            update_centroid(), recency_decay()
    ├── reconsolidation.py       ReconsolidationEngine — 4-tier Bayesian
    │                            update(), promote_tier(), should_reconsolidate()
    ├── consolidation.py         ConsolidationCycle — decay/prune/abstract
    │                            run() → SleepCycleResult
    ├── temporal.py              TemporalWeighter — auto-graduating temporal
    │                            weight(), model_for() — n<5/5-19/≥20 tiers
    └── workspace.py             GlobalWorkspace — K=5 competitive selection
                                 select() → position-reordered results
```

### Critical Implementation Details — Do Not Break

Discovered during implementation and benchmarking. Change only with evidence:

**promote_tier() uses PE=0.65** — patched from 0.2. At 0.65, gate fires for
LABILE (0.996), MODIFIABLE (0.971), PROTECTED (0.623), blocked only by
IMMUTABLE (threshold=∞). Correct neuroscience: access reinforces all tiers
except the most stable.

**scorer=None only removes the recency fallback** — in GlobalWorkspace,
`salience = memory.importance` is always used regardless of scorer=.
scorer controls only: temporal.weight() → scorer.recency_decay() → 1.0.

**Position reordering creates NDCG@5 ceiling of 0.947** — output order
[rank1, rank3, rank5, rank4, rank2] places rank-2 at position 5.
Optimises LLM context (avoids lost-in-middle) but penalises traditional
NDCG metrics. For LOCOMO/LongMemEval: compute NDCG on the pre-reorder
ranked list, not the position-reordered output.

**gate(0.0, 0.3) ≈ 0.047** — not < 0.01. The gate is a smooth sigmoid,
not a hard threshold. Relevant for MODIFIABLE-tier test assertions.

**IMMUTABLE requires alpha+beta > 236** for CI width < 0.05 at conf=0.96.
Use alpha=250, beta=10 in tests requiring IMMUTABLE tier.

**DualSpeedStore.add() signature:**
`add(content, metadata={}, importance=None, tier=MemoryTier.FAST, _embedding=None)`

---

## Benchmark Suite — Complete

All four sessions are complete and CI-green. benchmarks/ runs in under 2
minutes. test_performance.py is excluded from regular CI (runs on push to
main only, results uploaded as artifact).

### Benchmark Session Status

| Session | File | What it proves | Status |
|---------|------|---------------|--------|
| 1 | `test_invariants.py` | 6 mathematical guarantees + stateful machine | ✅ |
| 2 | `test_claims.py` | 3 novel claims validated end-to-end | ✅ |
| 3 | `test_retrieval.py` | Cognitive layer improves over raw search (p<0.01) | ✅ |
| 4 | `test_performance.py` | Latency + footprint baselines established | ✅ |

### Key Benchmark Findings

These were discovered during benchmarking and are now documented:

1. **promote_tier() bug found and fixed** — was using PE=0.2, silently became
   no-op above LABILE. Fixed to PE=0.65 in `reconsolidation.py`.

2. **NDCG ceiling** — position reordering limits NDCG@5 to 0.947 (not 1.0).
   See the `benchmarks/README.md` for community benchmark guidance.

3. **Ablation finding** — scorer=None does not remove importance weighting
   from GlobalWorkspace. The original 8-condition ablation was redesigned to
   a cleaner 4-condition comparison as a result.

4. **Session 3 adversarial dataset corrections** — Claude Code made three
   smart fixes during Session 3: common last_accessed timestamp, importance
   inversion (decoys > correct) to make temporal signal load-bearing, and
   NDCG assertion corrected to ≥0.93 (not ==1.0) due to position reordering.

### Performance Baselines (embedding_model='none')

| Operation | Scale | Observed | Target |
|-----------|-------|----------|--------|
| add() | 100 entries | ~1-5ms | < 50ms |
| add() | 1K entries | ~1-10ms | < 50ms |
| search() | 100 entries | ~1-5ms | < 100ms |
| search() | 1K entries | ~5-20ms | < 100ms |
| consolidate() | 100 entries | < 100ms | < 500ms |
| consolidate() | 1K entries | < 500ms | < 2s |

Note: Graphiti P95 ~300ms and Mem0 p95 ~1.44s use embedding models.
These baselines use keyword search only. Targets are zero-dependency tier.

---

## MCP Connector — Next Phase

Prove that `MCPConnector` (already built in `verity/core/connectors/mcp_client.py`)
works end-to-end against a live MCP server. This is an integration test, not
new code — unless bugs are found that require fixes.

### What to verify

1. **MCPConnector can connect to a real MCP server** (Google Calendar or Gmail)
2. **read() returns ConnectorRecord objects** with valid content + metadata
3. **ingest_from(connector, resource) flows through the Engine correctly**
4. **The full RELATE→NAVIGATE→GOVERN→REMEMBER loop completes** with MCP data

### Implementation plan

Read `verity/core/connectors/mcp_client.py` first. Then:

**File:** `tests/test_mcp_integration.py`

Mark all tests with `@pytest.mark.integration` — these require live MCP servers
and are skipped in regular CI. Run manually with `pytest -m integration`.

```python
# Skip if MCP server not available
pytest.importorskip("mcp")

@pytest.mark.integration
class TestMCPConnectorLive:

    def test_connect_and_describe(self):
        # Connect to a local or mock MCP server
        # Verify describe() returns valid ConnectorInfo
        ...

    def test_read_returns_records(self):
        # Verify read() returns list[ConnectorRecord]
        # Each record has content (str) and metadata (dict)
        ...

    def test_ingest_from_mcp(self):
        # Full Engine loop with MCP data
        # Verify context is assembled correctly
        ...
```

**If MCPConnector needs fixes:** fix them in `mcp_client.py`, add tests
to `tests/test_connectors.py` (which already exists and tests the Protocol).

**If MCPConnector works as-is:** document the working configuration in
`examples/04_mcp_google_calendar.py` following the pattern of examples/02_rest_api.py.

### Commit message

`feat: verify MCPConnector end-to-end with live MCP server`

---

## After MCP — PyPI Release

pyproject.toml is structurally ready. Before publishing:

1. **Verify metadata** — description, keywords, classifiers, project URLs
2. **Add CHANGELOG.md** — document v0.1.0 release
3. **Build and verify** — `python -m build`, inspect the wheel contents
4. **TestPyPI first** — `twine upload --repository testpypi dist/*`
5. **Install from TestPyPI** — verify `pip install --index-url https://test.pypi.org/simple/ verity`
6. **PyPI** — `twine upload dist/*`

Tag: `git tag v0.1.0 && git push --tags`

---


## How to Run Tests

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run benchmarks (excluded from regular CI)
pytest benchmarks/ --ignore=benchmarks/test_performance.py -v

# Run performance benchmarks explicitly
pytest benchmarks/test_performance.py -v -s

# Run with coverage
pytest tests/ -v --cov=verity --cov-report=term-missing

# Lint
ruff check verity/ tests/ benchmarks/

# Type check
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
bench: add generator and property invariant tests
fix: promote_tier fires at all tiers except IMMUTABLE
docs: update README, add quickstart example
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
- Always quote version specifiers in shell commands: `pip install "pkg>=1.0"`
  not `pip install pkg>=1.0` — the unquoted `>` is output redirection in bash

---

## Internal Architectural Vocabulary

These patterns exist in the codebase. Know them so you don't conflict with them.

**VERIFY** — not a protocol, not a class. It is the immune system that runs
*inside* every core function. It checks principles integrity, consent validity,
and crisis flags before any graph operation proceeds. It is not something you
build — it is the name for the interlocking checks already present in
`crisis.py`, `principles.py`, and `_validate_consent()`.

**SOMA** — the integration layer inside `navigate()` that assembles a
`ContextBundle` from raw traversal results. It is not a separate file or class —
it describes the assembly logic: scoring, ranking, uncertainty annotation, and
`agent_prompt` formatting that happens between graph traversal and returning
context to the caller.

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
