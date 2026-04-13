# Verity Examples

Working examples that demonstrate Verity's core features.

## Prerequisites

```bash
# Minimum — covers examples/01_personal_notes.py
pip install verity

# Full — covers all examples
pip install "verity[connectors]"
```

## Running the examples

All examples are run from the **repo root**:

```bash
python examples/01_personal_notes.py
python examples/02_rest_api.py
python examples/03_sql_database.py
```

## What each example demonstrates

- **01_personal_notes.py** — FilesystemConnector reading local Markdown files;
  the full RELATE → NAVIGATE loop with uncertainty output and an audit ID
- **02_rest_api.py** — DltConnector wrapping a public REST API (no auth required);
  ingesting structured JSON records and assembling context from them
- **03_sql_database.py** — DltConnector over an in-memory SQLite database;
  ingesting rows as typed facts and traversing the resulting knowledge graph
