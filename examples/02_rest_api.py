# Run from repo root: python examples/02_rest_api.py
# Requires: pip install "verity[connectors]"

"""
Example 02 — REST API via DltConnector
=======================================
Ingests todo items from a public REST API (jsonplaceholder.typicode.com)
using the DltConnector, which wraps a dlt REST API source.

What this demonstrates:
  - DltConnector wrapping a dlt rest_api source
  - Ingesting structured JSON records (no text parsing needed)
  - Context assembly over structured data
  - No authentication required — jsonplaceholder is a free public API

dlt rest_api source overview:
  dlt's rest_api source accepts a config dict describing the API base URL,
  authentication, and a list of resources (endpoints) to fetch. Each resource
  yields rows as Python dicts. DltConnector converts these to ConnectorRecords
  and streams them into the Verity engine.
"""

import asyncio

from verity import Engine
from verity.core.connectors.dlt_connector import DltConnector

# dlt REST API source configuration.
# See: https://dlthub.com/docs/dlt-ecosystem/verified-sources/rest_api
REST_API_CONFIG = {
    "client": {
        "base_url": "https://jsonplaceholder.typicode.com",
    },
    "resources": [
        {
            "name": "todos",
            "endpoint": {
                "path": "todos",
                "params": {
                    # Fetch only the first 10 items to keep the demo fast
                    "_limit": 10,
                },
            },
        },
    ],
}


async def main() -> None:
    # 1. Start the engine. The developer profile allows a deeper BFS traversal.
    engine = await Engine.start(profile="developer")

    # 2. Create a DltConnector for the REST API.
    #    source_type="rest_api" tells DltConnector to use dlt's rest_api source.
    #    source_config is passed directly to the dlt source constructor.
    connector = DltConnector(
        source_id="jsonplaceholder_todos",
        source_type="rest_api",
        source_config=REST_API_CONFIG,
    )

    async with engine.session(consent_ref="consent:demo") as s:

        # 3. Ingest from the connector.
        #    resource="todos" routes to the matching dlt resource defined above.
        #    Each todo item becomes a ConnectorRecord and is ingested as a fact.
        await s.ingest_from(connector, "todos")

        # 4. Assemble context from the ingested todo data.
        context = await s.context(
            query="what tasks are incomplete",
            purpose="task_review",
        )

    print(context.agent_prompt)
    print(f"Uncertainty: {context.uncertainty:.0%}")
    print(f"Audit ID:    {context.audit_id}")


if __name__ == "__main__":
    asyncio.run(main())
