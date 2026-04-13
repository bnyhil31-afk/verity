# Run from repo root: python examples/01_personal_notes.py
# Requires: pip install verity

"""
Example 01 — Personal Notes
============================
Reads a directory of Markdown files using FilesystemConnector, ingests them
into the Verity knowledge graph, and assembles context answering a natural-
language query.

What this demonstrates:
  - Engine.start() with the "personal" profile (zero config, local-only)
  - FilesystemConnector reading a directory of .md files
  - Session.ingest_from() streaming records from a connector
  - Session.context() traversing the graph and producing a ContextBundle
  - Reading uncertainty, audit_id, and agent_prompt from the bundle
"""

import asyncio

from verity import Engine
from verity.core.connectors.filesystem import FilesystemConnector


async def main() -> None:
    # 1. Start the engine with the personal profile.
    #    This uses a local rdflib graph store — no external database needed.
    engine = await Engine.start(profile="personal")

    # 2. Create a FilesystemConnector pointing at our sample notes directory.
    #    source_id is a label that appears in the provenance trail.
    connector = FilesystemConnector(source_id="my_notes")

    # 3. Open a session with a consent reference.
    #    The consent ref is recorded in the audit trail for every operation.
    async with engine.session(consent_ref="consent:me") as s:

        # 4. Ingest all files in the sample_notes directory.
        #    FilesystemConnector streams one ConnectorRecord per file.
        #    Each record passes through the crisis barrier before being ingested.
        await s.ingest_from(connector, "examples/sample_notes")

        # 5. Traverse the graph and assemble a ContextBundle.
        #    Verity picks the most relevant facts, applies power-law decay,
        #    and calculates an uncertainty score across the assembly.
        context = await s.context(
            query="what have I been working on",
            purpose="personal_review",
        )

    # 6. Print the outputs.
    #    agent_prompt is the ready-to-inject string for an LLM.
    #    uncertainty is a float [0, 1] — lower is more confident.
    #    audit_id is the Merkle-chained record of this exact assembly.
    print(context.agent_prompt)
    print(f"Uncertainty: {context.uncertainty:.0%}")
    print(f"Audit ID:    {context.audit_id}")


if __name__ == "__main__":
    asyncio.run(main())
