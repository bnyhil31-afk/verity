# Verity — Claude Code Briefing

Read this entire file before touching any code.
Then read it again if you're unsure about anything.

---

## What Verity is

Verity is a **universal traversal and comprehension engine** — an embedded Python
library that sits between raw data and understanding. It ingests data from any source,
builds a weighted graph of relationships, traverses that graph organically (the way a
mind follows connections), and assembles uncertainty-annotated context ready for
any consumer — human, AI agent, or application.

It is not a knowledge graph tool. It is not a RAG system. It is not a chatbot memory
layer. It is a comprehension layer that any of those things can sit on top of.

Target users: home users (zero config), developers (connector SDK), professionals
(team knowledge), enterprises (regulated industries). One codebase, four profiles.

Repository: https://github.com/bnyhil31-afk/verity

---

## The Five Invariants — These Cannot Change

If you are about to change anything that touches these, stop and ask.

1. **Crisis barrier is absolute.** `crisis.py` runs first, before anything else,
   on every ingestion call. It cannot be disabled, configured away, or bypassed.
   The canary tests verify this at every boot. Do not touch `crisis.py`.

2. **Everything is recorded.** The Merkle-chained audit trail is append-only.
   `AuditEvent` records are never modified or deleted. The `verify_chain()` method
   detects any tampering. Do not add UPDATE or DELETE paths to audit records.

3. **Humans decide.** GOVERN checkpoints cannot be automated away. Veto is the
   default — timeout = veto, not approval. This is enforced in `engine.py` and
   verified by canary tests.

4. **Nothing without consent.** The consent gate runs before any graph traversal.
   `ConsentRequiredError` halts the operation. Do not add paths that bypass
   `_validate_consent()`.

5. **Uncertainty is mandatory.** `ContextBundle.uncertainty` is never Optional.
   `ContextBundle.reasoning_trace` is never empty. These are enforced in
   `__post_init__`. Do not remove these validations.

---

## Architecture — What Each File Does

```
verity/                          Python package root
├── __init__.py                  Public API — imports everything the user needs
└── core/
    ├── __init__.py              Re-exports from types.py for clean imports
    ├── types.py                 ALL data contracts. No logic. Every other module
    │                            imports from here. Nothing here imports from Verity.
    ├── exceptions.py            All custom exceptions. Hierarchy is shallow.
    ├── crisis.py                Absolute crisis barrier. DO NOT MODIFY.
    ├── principles.py            Boot-time verifier — reads principles.yaml,
    │                            checks Ed25519 signature, runs canary tests.
    ├── engine.py                The engine. RELATE/NAVIGATE/GOVERN/REMEMBER.
    │                            The four functions. All logic lives here.
    └── graph_store/
        ├── __init__.py          GraphStore Protocol — the Machine Test boundary.
        │                        The engine calls this. Never knows what backend runs.
        ├── rdflib_store.py      Personal tier backend. rdflib ConjunctiveGraph
        │                        with three Named Graphs. Auto-detects pyoxigraph.
        └── registry.py          Factory. Reads VERITY_GRAPH_BACKEND env var.

tests/
├── conftest.py                  pytest-asyncio config + shared fixtures
├── test_types.py                Contract invariants — frozen dataclasses,
│                                validation in __post_init__, enum correctness
├── test_crisis.py               Crisis barrier tests. CANARY TESTS LIVE HERE.
│                                If these fail, something is seriously wrong.
├── test_graph_store.py          rdflib backend + Protocol compliance + registry
└── test_engine.py               Full integration — RELATE→NAVIGATE→GOVERN→REMEMBER

principles.yaml                  Genesis block. Signed at init. Boot-time verified.
pyproject.toml                   Dependencies, optional extras, tool config.
PRINCIPLES.md                    Human-readable invariants (references this file).
CONTRIBUTING.md                  How to contribute, module interface spec.
SECURITY.md                      Security policy, scope, known limitations.
```

---

## The Three Named Graphs

Every graph store backend maintains exactly these three:

```
urn:verity:knowledge    — typed facts and weighted edges (the working graph)
urn:verity:provenance   — append-only Merkle chain (the audit trail)
urn:verity:consent      — consent ledger (who authorized what for whom)
```

These URIs are canonical. Do not change them. Do not add a fourth.

---

## Data Model — The Key Types

```python
# The unit of knowledge
TypedFact(entity_id, entity_type, classification, trust_score,
          provenance_ref, created_at, source, domain_properties, ...)

# The unit of relationship
WeightedEdge(edge_id, source_id, target_id, relationship_type,
             base_weight: ThreeAxisWeight, effective_weight, ...)

# The three axes — never collapse to one
ThreeAxisWeight(distance, complexity, size)  # all [0.0, 1.0]

# Power-law decay — not exponential, this is intentional and scientifically grounded
# effective_weight = base × (1 + days)^(-exponent)  [Wixted 2004 / Jost's Law]
DecayParameters(exponent=0.5, sensitive_multiplier=1.4, spacing_cap=2.0,
                prune_threshold=0.05)

# The output — agent_prompt is the product
ContextBundle(facts, edges, uncertainty, completeness, excluded,
              reasoning_trace, consent_ref, purpose, assembled_at,
              audit_id, session_id, agent_prompt, agent_prompt_tokens,
              checkpoint_required, checkpoint_context)
```

---

## What Does NOT Exist Yet (Build This)

These are missing from the codebase. The coding plan below builds them in order.

1. `verity/core/connectors/` — The Connector Protocol and implementations
2. `verity/core/profiles.py` — Four deployment profiles as real code
3. `verity/cli.py` — `verity init` and `verity status` commands

---

## Known Bugs (Fix First — Phase 0)

### Bug 1: DataClassification is not a proper Enum

**Location:** `verity/core/types.py`

**Problem:** `DataClassification` inherits from `str` but defines class attributes
as plain strings. `DataClassification.PHI` is just the string `"phi"`, not an
instance of `DataClassification`. The `requires_consent` property, `audit_on_access`
property, and `escalate()` classmethod will not work correctly.

**Fix:** Make it a proper `StrEnum` (Python 3.11+):

```python
from enum import StrEnum

class DataClassification(StrEnum):
    PUBLIC       = "public"
    INTERNAL     = "internal"
    CONFIDENTIAL = "confidential"
    PHI          = "phi"
    PII          = "pii"
    FINANCIAL    = "financial"
    LEGAL        = "legal"

    @property
    def requires_consent(self) -> bool:
        return self in (
            DataClassification.PHI,
            DataClassification.PII,
            DataClassification.FINANCIAL,
            DataClassification.LEGAL,
        )

    @property
    def audit_on_access(self) -> bool:
        return self in (
            DataClassification.PHI,
            DataClassification.PII,
            DataClassification.FINANCIAL,
            DataClassification.LEGAL,
            DataClassification.CONFIDENTIAL,
        )

    @classmethod
    def escalate(cls, a: "DataClassification", b: "DataClassification") -> "DataClassification":
        order = [cls.PUBLIC, cls.INTERNAL, cls.CONFIDENTIAL,
                 cls.PII, cls.FINANCIAL, cls.LEGAL, cls.PHI]
        return a if order.index(a) >= order.index(b) else b
```

Apply the same StrEnum fix to: `Completeness`, `CheckpointDecision`,
`TrustSource`, `AuditEventType`. These all inherit from `str` but should
be `StrEnum`.

### Bug 2: BFS→PPR threshold is wrong

**Location:** `verity/core/engine.py`

**Problem:** `_PPR_THRESHOLD = 200`. PPR only adds value above ~50,000 nodes.
At 200 nodes BFS is faster and more accurate.

**Fix:** Change to `_PPR_THRESHOLD = 50_000`

### Bug 3: Completeness, CheckpointDecision, TrustSource, AuditEventType

Same issue as DataClassification — all need to be `StrEnum`.

---

## Coding Plan — Phases

Complete each phase fully before starting the next.
After each phase: run `pytest tests/ -v` and confirm green before committing.

---

### Phase 0 — Fix Bugs (surgical changes to existing files)

Files to change:
- `verity/core/types.py` — Fix all enums to StrEnum (see Bug 1 above)
- `verity/core/engine.py` — Fix PPR threshold (see Bug 2 above)

Commit message: `fix: correct StrEnum inheritance and BFS threshold`

After this phase: ALL existing tests must still pass. If any fail, fix them
before moving to Phase 1. The StrEnum change may affect string comparisons
in tests — `DataClassification.PHI == "phi"` is still True with StrEnum,
so most tests should pass unchanged.

---

### Phase 1 — Upgrade Graph Store (upgrade rdflib_store.py)

**Goal:** Auto-detect pyoxigraph and use it as the rdflib store backend.
Same API, 37x faster SPARQL, 3-5 MB memory footprint.

In `verity/core/graph_store/rdflib_store.py`, the `_select_store_backend()`
method already has the right pattern. Make it actually work:

```python
def _select_store_backend(self) -> str:
    try:
        import oxrdflib  # noqa: F401
        logger.info("oxrdflib detected — using Oxigraph store (37x faster SPARQL)")
        return "Oxigraph"
    except ImportError:
        logger.debug("oxrdflib not available — using rdflib default store")
        return "default"
```

In `pyproject.toml`, add to the `[fast]` optional extras:
```
"pyoxigraph>=0.3.22",
"oxrdflib>=0.5.0",
```

Commit message: `perf: auto-detect pyoxigraph for 37x SPARQL speedup`

---

### Phase 2 — Build the Connector Protocol (5 new files)

**Goal:** The most important missing piece. Three methods. Zero coupling.
Covers every data source.

Create `verity/core/connectors/` with these files:

#### `verity/core/connectors/__init__.py`

The core Protocol:

```python
from typing import Protocol, AsyncIterator, Any, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime, timezone

@dataclass
class ConnectorRecord:
    """Standard record shape every connector produces."""
    id: str                          # Unique within source
    content: str | dict | bytes      # The actual data
    source_id: str                   # Which connector produced this
    resource: str                    # Which resource within the source
    metadata: dict[str, Any] = field(default_factory=dict)
    classification: str = "internal" # DataClassification value
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trust_score: float = 0.5

@runtime_checkable
class Connector(Protocol):
    """
    Universal data source interface. Three methods. Zero coupling.

    Implement these three methods and any class is a valid Verity connector.
    No inheritance required. No imports from Verity required.

    resource: a string address for the data within the source.
    Examples: file path, table name, API endpoint, MQTT topic,
              calendar ID, email folder, FHIR resource type.
    """

    async def read(
        self,
        resource: str,
        query: dict | None = None,
        **opts: Any,
    ) -> AsyncIterator[ConnectorRecord]:
        """Yield records from the source. Streaming-native."""
        ...

    async def write(
        self,
        resource: str,
        data: AsyncIterator[ConnectorRecord] | list[ConnectorRecord],
        **opts: Any,
    ) -> dict[str, Any]:
        """Write records to the source. Returns stats."""
        ...

    async def describe(
        self,
        resource: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """
        Describe available resources and their schemas.
        resource=None: list all available resources.
        resource=name: describe that specific resource.
        """
        ...
```

#### `verity/core/connectors/base.py`

`BaseConnector` with lifecycle management (`async with`), logging,
credential storage, and default `write()` that raises `NotImplementedError`
(most connectors are read-only). Subclasses override only what they need.

#### `verity/core/connectors/filesystem.py`

`FilesystemConnector` — reads local files.
- `resource`: file path or glob pattern (`"~/notes/**/*.md"`)
- Handles: `.txt`, `.md`, `.json`, `.csv`, `.yaml`
- Yields one `ConnectorRecord` per file (or per row for CSV/JSON arrays)
- Zero external dependencies

#### `verity/core/connectors/mcp_client.py`

`MCPConnector` — wraps any MCP server as a Verity connector.
- Uses FastMCP's Python client (already a dependency)
- `resource`: MCP tool name or resource URI
- Gives access to 10,000+ existing MCP servers with ~50 lines of adapter code
- Handle connection pooling — one MCP connection per connector instance

#### `verity/core/connectors/registry.py`

`ConnectorRegistry` — register, discover, route by source ID.
- `register(connector_id, connector)` — add a connector
- `get(connector_id)` — retrieve by ID
- `list()` — all registered connectors with their `describe()` output
- `discover()` — find connectors via setuptools entry points
  (`verity.connectors` group)
- Registry is engine-scoped, not global

Commit message: `feat: add Connector Protocol and core connector implementations`

Add tests in `tests/test_connectors.py`:
- Protocol compliance (isinstance check)
- FilesystemConnector read round-trip
- ConnectorRecord shape validation
- Registry register/get/list

---

### Phase 3 — Widen RELATE (surgical change to engine.py)

**Goal:** RELATE accepts anything — text, dicts, streaming connector records.

Changes to `verity/core/engine.py`:

1. Add `Engine.ingest_from(connector, resource, **opts)` method:
   - Calls `connector.read(resource, **opts)`
   - For each `ConnectorRecord`, calls the existing `relate()` logic
   - Returns `RelateResult` with aggregate counts
   - Crisis barrier still runs on every record's content

2. Modify `Session.ingest()` signature:
   ```python
   async def ingest(
       self,
       data: str | dict | ConnectorRecord,
       source: str = "manual_entry",
       ...
   ) -> RelateResult:
   ```
   - `str` → existing YAKE path (unchanged)
   - `dict` → treat as pre-structured fact, skip YAKE
   - `ConnectorRecord` → use record's fields directly

3. Add `Session.ingest_from(connector, resource, **opts)` — convenience
   method that calls `Engine.ingest_from()` with session context.

Commit message: `feat: widen RELATE to accept structured data and connector records`

---

### Phase 4 — Build the Profile System (2 files)

**Goal:** `Engine.start(profile="personal")` works. Four real profiles.

#### `verity/core/profiles.py`

```python
from dataclasses import dataclass, field
from verity.core.types import DecayParameters, DEFAULT_DECAY_PARAMETERS

@dataclass(frozen=True)
class EngineProfile:
    """Deployment profile — activates the right blades for each user type."""
    name: str
    decay_parameters: DecayParameters
    bfs_max_depth: int
    checkpoint_timeout_seconds: int
    checkpoint_interactive: bool   # True = stdout/stdin, False = deferred
    auto_sign_principles: bool     # True = use package key, no ceremony
    graph_store_backend: str       # "rdflib" | "oxigraph" | "postgres" | "jena"
    description: str

PERSONAL = EngineProfile(
    name="personal",
    decay_parameters=DEFAULT_DECAY_PARAMETERS,
    bfs_max_depth=2,
    checkpoint_timeout_seconds=300,
    checkpoint_interactive=True,
    auto_sign_principles=True,      # pre-signed at install, no ceremony
    graph_store_backend="rdflib",
    description="Home user. Zero config. Local only. Pre-signed principles.",
)

DEVELOPER = EngineProfile(
    name="developer",
    decay_parameters=DEFAULT_DECAY_PARAMETERS,
    bfs_max_depth=3,
    checkpoint_timeout_seconds=600,
    checkpoint_interactive=True,
    auto_sign_principles=False,     # developer signs with their own key
    graph_store_backend="rdflib",
    description="Developer building AI agents. Full connector SDK. Local signing.",
)

PROFESSIONAL = EngineProfile(
    name="professional",
    decay_parameters=DEFAULT_DECAY_PARAMETERS,
    bfs_max_depth=3,
    checkpoint_timeout_seconds=3600,
    checkpoint_interactive=False,   # deferred — web UI or API response
    auto_sign_principles=False,
    graph_store_backend="postgres",
    description="Team/SMB. Multi-user. PostgreSQL backend.",
)

ENTERPRISE = EngineProfile(
    name="enterprise",
    decay_parameters=DecayParameters(exponent=0.4, sensitive_multiplier=1.6),
    bfs_max_depth=4,
    checkpoint_timeout_seconds=86400,  # 24 hours
    checkpoint_interactive=False,
    auto_sign_principles=False,        # M-of-N ceremony required
    graph_store_backend="jena",
    description="Regulated industries. Full audit trail. M-of-N principles ceremony.",
)

PROFILES = {p.name: p for p in [PERSONAL, DEVELOPER, PROFESSIONAL, ENTERPRISE]}

def get_profile(name: str) -> EngineProfile:
    if name not in PROFILES:
        raise ValueError(f"Unknown profile '{name}'. Valid: {list(PROFILES)}")
    return PROFILES[name]
```

#### `verity/core/engine.py` — add profile support

Add `profile` parameter to `Engine.start()`:
```python
@classmethod
async def start(
    cls,
    profile: str | EngineProfile = "personal",
    modules: list[str] | None = None,
    decay_parameters: DecayParameters | None = None,
) -> "Engine":
```

When `profile` is a string, call `get_profile(profile)`.
Profile sets defaults for decay_parameters, BFS depth,
checkpoint timeout, and checkpoint mode.
Explicit `decay_parameters` argument overrides profile.

Commit message: `feat: add profile system — personal/developer/professional/enterprise`

---

### Phase 5 — CLI (1 new file)

**Goal:** `verity init` and `verity status` work.

#### `verity/cli.py`

Use Python's `argparse` (no new dependencies — no Click, no Typer).

```
verity init                    # detect context, suggest profile, sign principles
verity init --profile personal # explicit profile selection
verity init --profile enterprise  # walk through M-of-N ceremony
verity status                  # engine health, graph stats, connected connectors
verity connect <source-id>     # add a connector to the active profile
```

Register in `pyproject.toml`:
```toml
[project.scripts]
verity = "verity.cli:main"
```

For `personal` profile: use the package's pre-signed key to sign
`principles.yaml` without user interaction.
For `developer` and above: generate an Ed25519 keypair, store
private key in `~/.verity/keys/` (never in the repo), sign principles.

Commit message: `feat: add CLI — verity init, verity status, verity connect`

---

### Phase 6 — Tests

Add after each phase above. Specific files:

- `tests/test_connectors.py` — after Phase 2
- `tests/test_profiles.py` — after Phase 4
- Expand `tests/test_engine.py` — after Phase 3

All new tests follow the existing pattern:
- Class per feature area
- `async def test_*` with `@pytest.mark.asyncio` (or asyncio_mode=auto)
- One happy path + one adversarial case minimum per function
- Use fixtures from `conftest.py` rather than setting up state inline

---

## How to Run Tests

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific file
pytest tests/test_crisis.py -v

# Run with coverage
pytest tests/ -v --cov=verity --cov-report=term-missing

# Lint
ruff check verity/ tests/

# Type check
mypy verity/
```

CI runs on push to main via `.github/workflows/ci.yml`.
**Never push with failing tests.**

---

## Commit Message Format

```
scope: short description — what and why
```

Examples:
```
fix: correct StrEnum inheritance and BFS threshold
feat: add Connector Protocol and core connector implementations
perf: auto-detect pyoxigraph for 37x SPARQL speedup
test: add connector protocol compliance tests
docs: update CONTRIBUTING with connector authoring guide
```

One line. Under 72 characters. Imperative mood.

---

## What You Must Not Do

- Do not modify `crisis.py` for any reason
- Do not add UPDATE or DELETE paths to audit records
- Do not make `uncertainty` Optional on `ContextBundle`
- Do not make `reasoning_trace` Optional or allow empty tuple
- Do not bypass `_validate_consent()` in the navigate path
- Do not change the three Named Graph URIs
- Do not add a fourth Named Graph
- Do not change the power-law decay formula
- Do not collapse ThreeAxisWeight to a single float
- Do not break existing passing tests — fix them if the change requires it

---

## The Two Tests (apply to every change)

**Machine Test:** Can the four functions (RELATE/NAVIGATE/GOVERN/REMEMBER)
be swapped for equivalent implementations without touching each other?
If a change requires editing multiple function implementations simultaneously,
it probably violates this.

**Brain Test:** Does the system preserve contextual flow and accumulated
awareness? Does it behave like a mind with memory, not a stateless query engine?

If your change fails either test, redesign it.

---

## Questions?

If you're unsure about the intent of any design decision, the architecture
documentation lives in `PRINCIPLES.md`, `CONTRIBUTING.md`, and `SECURITY.md`.
The principles themselves are in `principles.yaml`.

When in doubt: make the smallest change that passes the tests.
