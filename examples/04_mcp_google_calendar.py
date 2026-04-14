# Run from repo root: python examples/04_mcp_google_calendar.py
# Requires: pip install "verity[mcp]"
# Requires: Google Calendar MCP server access

"""
Example 04 — Google Calendar via MCP
=====================================
Connects to the Google Calendar MCP server, reads upcoming events,
and ingests them into the Verity engine for context assembly.

What this demonstrates:
  - MCPConnector with an HTTP-based MCP server
  - describe() to discover available tools before ingesting
  - Keeping the connector open across describe() and ingest_from()
  - context() to query the resulting knowledge graph

MCP server: Google Calendar (https://gcal.mcp.claude.com/mcp)
Transport:  fastmcp >= 2.0 auto-detects streamable-http from the https:// URL.
"""

import asyncio

from verity import Engine
from verity.core.connectors.mcp_client import MCPConnector

GCAL_MCP_URL = "https://gcal.mcp.claude.com/mcp"


async def main() -> None:
    engine = await Engine.start(profile="developer")

    # MCPConnector must stay open for both describe() and ingest_from().
    # Use a single 'async with' block that covers all operations.
    async with MCPConnector(
        server_url=GCAL_MCP_URL,
        source_id="google_calendar",
    ) as conn:

        # 1. Discover available tools.
        info = await conn.describe()
        print("Connected to Google Calendar MCP server.")
        print("Available tools:")
        for tool in info.get("tools", []):
            print(f"  - {tool['name']}: {tool['description']}")

        # 2. Ingest calendar events through the Engine session.
        async with engine.session(consent_ref="consent:demo") as s:
            await s.ingest_from(conn, "list_events")

            # 3. Query for upcoming schedule.
            context = await s.context(
                query="what meetings do I have coming up",
                purpose="schedule_review",
            )

    print("\n── Context assembled ──")
    print(context.agent_prompt)
    print(f"Uncertainty: {context.uncertainty:.0%}")
    print(f"Audit ID:    {context.audit_id}")


if __name__ == "__main__":
    asyncio.run(main())
