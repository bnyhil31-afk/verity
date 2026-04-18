# Connectors

Connectors let the Engine ingest data from external sources. All connectors implement the `Connector` Protocol: `read()`, `write()`, and `describe()`.

## FilesystemConnector

Reads local files in `.txt`, `.md`, `.json`, `.csv`, and `.yaml` formats.

**When to use:** local files, notes, documents, configuration files.

**Install:** no extras needed — included in the base package.

```python
from verity.core.connectors.filesystem import FilesystemConnector

connector = FilesystemConnector(base_path="./notes")

# Describe available resources
info = connector.describe()
print(info.available_resources)

# Read a file — returns list[ConnectorRecord]
records = connector.read("architecture.md")
for record in records:
    print(record.content)      # str
    print(record.metadata)     # dict with path, format, size, etc.
```

**With the Engine:**

```python
from verity import Engine
from verity.core.profiles import Profile
from verity.core.connectors.filesystem import FilesystemConnector

engine = Engine.start(profile=Profile.PERSONAL)

with engine.session(consent_ref="my-consent") as session:
    fs = FilesystemConnector(base_path="~/documents")
    session.ingest_from(fs, "notes.md")
    bundle = session.context("summary of notes", purpose="review")
```

---

## DltConnector

Wraps [dlt](https://dlthub.com/) to provide access to 60+ data sources including REST APIs, SQL databases, cloud storage, and SaaS tools.

**When to use:** databases, REST APIs, cloud services, Salesforce, Stripe, GitHub, PostgreSQL, BigQuery, etc.

**Install:**

```bash
pip install "veritycog[connectors]"
```

```python
from verity.core.connectors.dlt_connector import DltConnector

# REST API source
connector = DltConnector.from_rest_api(
    base_url="https://api.example.com",
    endpoints=["users", "projects"],
    headers={"Authorization": "Bearer TOKEN"},
)
records = connector.read("users")

# SQL database
connector = DltConnector.from_sql(
    connection_string="postgresql://user:pass@localhost/mydb",
    tables=["users", "events"],
)
records = connector.read("users")
```

See `examples/02_rest_api.py` and `examples/03_sql_database.py` for complete walkthroughs.

**With the Engine:**

```python
from verity import Engine
from verity.core.profiles import Profile
from verity.core.connectors.dlt_connector import DltConnector

engine = Engine.start(profile=Profile.PROFESSIONAL)

with engine.session(consent_ref="org-consent-2024") as session:
    connector = DltConnector.from_rest_api(
        base_url="https://api.github.com",
        endpoints=["repos/{owner}/{repo}/issues"],
        headers={"Authorization": "Bearer GITHUB_TOKEN"},
    )
    session.ingest_from(connector, "repos/myorg/myrepo/issues")
    bundle = session.context("open bugs", purpose="sprint planning")
```

---

## MCPConnector

Connects to any [MCP](https://modelcontextprotocol.io/) server as a Verity data source. Compatible with Google Calendar, Gmail, or any MCP-compatible service.

**When to use:** calendar events, email, or any service that exposes an MCP server.

**Install:**

```bash
pip install "veritycog[mcp]"
```

`fastmcp >= 2.0` auto-detects transport from the URL format:

- `http://...` or `https://...` → HTTP/SSE transport
- `stdio://...` → subprocess stdio transport

```python
from verity.core.connectors.mcp_client import MCPConnector

# Connect to a local MCP server
connector = MCPConnector(server_url="http://localhost:8080")
info = connector.describe()
records = connector.read("calendar/events/today")

# Connect via stdio
connector = MCPConnector(server_url="stdio://npx @google/mcp-server-calendar")
```

See `examples/04_mcp_google_calendar.py` for a complete Google Calendar integration example.

**With the Engine:**

```python
from verity import Engine
from verity.core.profiles import Profile
from verity.core.connectors.mcp_client import MCPConnector

engine = Engine.start(profile=Profile.PERSONAL)

with engine.session(consent_ref="calendar-consent") as session:
    connector = MCPConnector(server_url="http://localhost:8080")
    session.ingest_from(connector, "calendar/events/this-week")
    bundle = session.context("What meetings do I have Friday?", purpose="schedule query")
    print(bundle.agent_prompt)
```

---

## Building your own connector

All connectors implement the `Connector` Protocol. The simplest way to start is to subclass `BaseConnector` from `verity/core/connectors/base.py`.

```python
from verity.core.connectors.base import BaseConnector
from verity.core.connectors import ConnectorRecord

class MyConnector(BaseConnector):

    def describe(self):
        # Return ConnectorInfo describing what this connector provides
        ...

    def read(self, resource: str) -> list[ConnectorRecord]:
        # Fetch data and return as ConnectorRecord objects
        # Each record: ConnectorRecord(content=str, metadata=dict)
        ...

    def write(self, record: ConnectorRecord) -> None:
        # Optional: implement if the source is writable
        raise NotImplementedError
```

Register your connector via the `verity.connectors` setuptools entry point group so it is discoverable at runtime:

```toml
[project.entry-points."verity.connectors"]
my_source = "my_package.connector:MyConnector"
```
