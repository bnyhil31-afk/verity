# Changelog

All notable changes to Verity are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
Versioning: [Semantic Versioning](https://semver.org/)

---

## [0.1.0] — 2026-04-14

First public release.

### The three problems Verity solves

These were unsolved in production memory systems before this release:

1. **Reconsolidation stability** — memories can be updated on access
   without runaway drift. A 4-tier Bayesian protection system
   (LABILE → MODIFIABLE → PROTECTED → IMMUTABLE) controls when and
   how memories accept new information.

2. **Sleep consolidation** — an offline cycle (decay → prune → abstract)
   removes stale memories, strengthens important ones, and merges
   clusters of similar episodes into semantic abstractions.

3. **Tiered temporal weighting** — the recency signal automatically
   graduates from exponential decay (n<5 events) to Bayesian renewal
   (n<20) to Hawkes process (n≥20) as access history grows.

### Added

#### Engine layer (`verity/core/`)
- RELATE / NAVIGATE / GOVERN / REMEMBER traversal engine
- Four profiles: PERSONAL, DEVELOPER, PROFESSIONAL, ENTERPRISE
- rdflib graph backend with optional pyoxigraph (37x faster)
- Connector Protocol: FilesystemConnector, DltConnector, MCPConnector
- Merkle-chained append-only audit trail
- Consent gate with GDPR-aligned purpose binding
- Crisis barrier (absolute, cannot be disabled)
- CLI: `verity init`, `verity status`, `verity connect`

#### Cognitive layer (`verity/cognitive/`)
- `DualSpeedStore` — SQLite dual-table Complementary Learning Systems
- `ImportanceScorer` — prediction error as dopamine/surprise proxy
- `ReconsolidationEngine` — 4-tier Bayesian stability (Beta-Bernoulli)
- `ConsolidationCycle` — decay/prune/abstract sleep cycle
- `TemporalWeighter` — exponential → renewal → Hawkes auto-graduation
- `GlobalWorkspace` — K=5 competitive selection with position reordering

#### Simple API (`verity/memory.py`)
- `Memory` — zero-config bolt-on: `Memory().add()`, `Memory().search()`
- Seven methods: `add`, `search`, `get`, `update`, `delete`,
  `consolidate`, `export`
- Full async variants: `aadd`, `asearch`, `aget`, `aupdate`,
  `adelete`, `aconsolidate`, `aexport`
- Context manager support: `with Memory() as m:`
- GDPR Article 17 (erasure) and Article 20 (portability) compliant

### Install tiers

```
pip install verity                    # stdlib only
pip install "verity[cognitive]"       # + model2vec + hnswlib
pip install "verity[connectors]"      # + dlt
pip install "verity[mcp]"             # + fastmcp
pip install "verity[fast]"            # + pyoxigraph (graph backend)
pip install "verity[full]"            # everything
```

### Benchmark results (zero-dependency tier, embedding_model='none')

| Operation       | Scale | CI target  |
|----------------|-------|------------|
| `add()`         | 1K    | mean < 50ms |
| `search()`      | 1K    | mean < 100ms |
| `consolidate()` | 1K    | mean < 2s  |

These are CI-enforced targets on `embedding_model='none'` (keyword search,
no ML model). Actual results on typical hardware are faster.
For measured results, see `benchmarks/README.md` in the repository.
Results will vary significantly with embedding models enabled.

### Compatibility

- Python: 3.11, 3.12, 3.13
- OS: Linux, macOS, Windows
- SQLite: bundled with Python stdlib (no separate install)

---

[0.1.0]: https://github.com/bnyhil31-afk/verity/releases/tag/v0.1.0
