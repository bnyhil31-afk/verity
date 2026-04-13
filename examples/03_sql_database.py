# Run from repo root: python examples/03_sql_database.py
# Requires: pip install "verity[connectors]"

"""
Example 03 — SQL Database via DltConnector
===========================================
Creates an in-memory SQLite database with sample project data, then ingests
rows using the DltConnector wrapping a dlt sql_database source.

What this demonstrates:
  - Setting up an in-memory SQLite database (no external DB needed)
  - DltConnector wrapping a dlt sql_database source
  - Ingesting database rows as typed Verity facts
  - Context assembly over relational data

dlt sql_database source overview:
  dlt's sql_database source uses SQLAlchemy to connect to any SQL database.
  It reflects the schema automatically and yields rows as Python dicts.
  DltConnector streams these rows into the Verity engine as ConnectorRecords.
"""

import asyncio
import os
import sqlite3
import tempfile

from verity import Engine
from verity.core.connectors.dlt_connector import DltConnector


def create_sample_database(db_path: str) -> None:
    """Create a small SQLite database with sample project data."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE projects (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            description TEXT,
            owner TEXT
        )
    """)

    cur.executemany(
        "INSERT INTO projects (name, status, description, owner) VALUES (?, ?, ?, ?)",
        [
            (
                "Project Alpha",
                "active",
                "Internal REST API for task tracking. FastAPI + SQLite.",
                "alex",
            ),
            (
                "Data Pipeline",
                "active",
                "ETL pipeline moving warehouse data to the analytics DB nightly.",
                "riley",
            ),
            (
                "Mobile App v2",
                "planning",
                "Redesign of the customer-facing mobile application.",
                "sam",
            ),
            (
                "Legacy Migration",
                "on-hold",
                "Migrating the old PHP monolith to microservices. Blocked on budget.",
                "jordan",
            ),
        ],
    )

    conn.commit()
    conn.close()


async def main() -> None:
    # 1. Create a temporary SQLite database file with sample data.
    #    We use a file-based SQLite (not :memory:) so that dlt's SQLAlchemy
    #    engine can connect to it via a standard connection string.
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        create_sample_database(db_path)

        # 2. Start the engine.
        engine = await Engine.start(profile="developer")

        # 3. Create a DltConnector for the SQLite database.
        #    source_type="sql_database" uses dlt's sql_database source.
        #    credentials is the SQLAlchemy connection string.
        connector = DltConnector(
            source_id="projects_db",
            source_type="sql_database",
            source_config={
                "credentials": f"sqlite:///{db_path}",
                "schema": None,  # SQLite has no schema namespacing
            },
        )

        async with engine.session(consent_ref="consent:demo") as s:

            # 4. Ingest from the "projects" table.
            #    Each row becomes a ConnectorRecord and is ingested as a fact.
            await s.ingest_from(connector, "projects")

            # 5. Assemble context from the ingested rows.
            context = await s.context(
                query="which projects are currently active",
                purpose="project_review",
            )

        print(context.agent_prompt)
        print(f"Uncertainty: {context.uncertainty:.0%}")
        print(f"Audit ID:    {context.audit_id}")

    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(main())
