# Verity Examples

## Prerequisites

```bash
pip install verity                 # covers 00_quickstart.py
pip install "verity[connectors]"   # covers 01, 02, 03
```

## Running

All examples are run from the repo root:

```bash
python examples/00_quickstart.py   # start here
python examples/01_personal_notes.py
python examples/02_rest_api.py
python examples/03_sql_database.py
```

## What each example demonstrates

- **00_quickstart.py** — Memory API: add/search/get/update/consolidate/
  export/delete. No connectors. Zero config. Start here.
- **01_personal_notes.py** — Engine API with FilesystemConnector reading
  local Markdown files; full RELATE→NAVIGATE→GOVERN→REMEMBER loop
- **02_rest_api.py** — DltConnector over a public REST API (no auth);
  ingesting structured JSON and assembling context
- **03_sql_database.py** — DltConnector over an in-memory SQLite database;
  ingesting rows as typed facts
