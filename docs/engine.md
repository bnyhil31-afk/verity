# Engine API

The `Engine` is the lower-level API that `Memory` wraps. Use it when you need full control over graph traversal, consent management, profiles, and the audit trail.

## When to use Engine vs Memory

| Use case | Recommended |
|----------|-------------|
| Simple memory for an AI agent | `Memory` |
| Reading from files, APIs, or MCP servers | `Engine` with connectors |
| Multi-tenant with consent enforcement | `Engine` |
| Compliance audit trail required | `Engine` |
| Custom profiles (personal vs enterprise) | `Engine` |
| Research or inspection of graph internals | `Engine` |

## The five invariants

The Engine enforces five invariants that cannot be disabled:

1. **Crisis barrier** — `crisis.py` runs before every ingestion. It cannot be configured away or bypassed.
2. **Audit trail** — every operation produces an `AuditEvent` with a Merkle chain link. Records are append-only and cannot be modified or deleted.
3. **Human sovereignty** — GOVERN checkpoints cannot be automated. Timeout = veto, not approval.
4. **Consent gate** — `_validate_consent()` runs before any graph traversal. `ConsentRequiredError` halts the operation.
5. **Uncertainty annotation** — every `ContextBundle` carries a non-optional `uncertainty` field and a non-empty `reasoning_trace`.

## Profiles

Four profiles control the engine's behaviour:

| Profile | Use case |
|---------|----------|
| `PERSONAL` | Single-user, local storage, relaxed retention |
| `DEVELOPER` | Local development, verbose logging |
| `PROFESSIONAL` | Multi-user, consent enforcement, audit trail |
| `ENTERPRISE` | Full compliance, GOVERN checkpoints, external audit |

## Basic usage

```python
from verity import Engine
from verity.core.profiles import Profile

# Start the engine with a profile
engine = Engine.start(profile=Profile.PERSONAL)

# Open a session with a consent reference
with engine.session(consent_ref="user-consent-2024-01") as session:

    # Ingest from a connector
    from verity.core.connectors.filesystem import FilesystemConnector
    connector = FilesystemConnector(base_path="./notes")
    session.ingest_from(connector, "project-notes.md")

    # Retrieve context
    bundle = session.context(
        query="What did we decide about the API?",
        purpose="answer user question",
    )
    print(bundle.agent_prompt)
    print(f"Uncertainty: {bundle.uncertainty:.0%}")
```

## ContextBundle

`session.context()` returns a `ContextBundle`:

```python
@dataclass
class ContextBundle:
    agent_prompt: str          # formatted context ready for LLM injection
    uncertainty: float         # 0.0–1.0, never Optional
    reasoning_trace: tuple     # non-empty, explains how context was assembled
    audit_id: str              # links this retrieval to the audit trail
    sources: list[str]         # provenance URIs
```

The `agent_prompt` field is ready to inject directly into your LLM system prompt or user message. It includes the retrieved facts, source attributions, and uncertainty annotation.

## Connector pattern

```python
from verity import Engine
from verity.core.connectors.filesystem import FilesystemConnector
from verity.core.profiles import Profile

engine = Engine.start(profile=Profile.DEVELOPER)

with engine.session(consent_ref="dev-session") as session:
    # Load local files
    fs = FilesystemConnector(base_path="./docs")
    session.ingest_from(fs, "architecture.md")
    session.ingest_from(fs, "decisions.md")

    # Query
    bundle = session.context("What storage layer does this use?", purpose="dev query")
    print(bundle.agent_prompt)
```

## Personal notes example

The following pattern (from `examples/01_personal_notes.py`) shows the full RELATE → NAVIGATE → GOVERN → REMEMBER loop:

```python
from verity import Engine
from verity.core.profiles import Profile
from verity.core.connectors.filesystem import FilesystemConnector

engine = Engine.start(profile=Profile.PERSONAL)

with engine.session(consent_ref="personal-2024") as session:
    # RELATE — ingest and link new information
    connector = FilesystemConnector(base_path="~/notes")
    session.ingest_from(connector, "meeting-notes.md")

    # NAVIGATE — traverse the knowledge graph
    # GOVERN — human checkpoint (auto-approved in PERSONAL profile)
    # REMEMBER — assemble and return context
    bundle = session.context(
        query="What action items came out of last week's meetings?",
        purpose="personal review",
    )

    print(bundle.agent_prompt)
    print(f"Confidence: {1 - bundle.uncertainty:.0%}")
    print(f"Audit ID: {bundle.audit_id}")
```
