"""
verity.core.connectors.mcp_client
==================================
MCPConnector — wraps any MCP server as a Verity connector.

Uses FastMCP's Python client (fastmcp>=2.0.0, already a core dependency).
Gives access to 10,000+ existing MCP servers with ~50 lines of adapter code.

Connection pooling: one FastMCP Client instance per MCPConnector instance,
initialized in __aenter__, closed in __aexit__. Always use as an async
context manager:

    async with MCPConnector("path/to/server.py") as conn:
        async for record in conn.read("tool_name", query={"arg": "value"}):
            process(record)

Graceful degradation: if fastmcp is not installed, raises ImportError with
an install hint at construction time (not at import time).
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from verity.core.connectors import ConnectorCapability, ConnectorRecord
from verity.core.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Import FastMCP client — fail gracefully at construction time, not import time
try:
    from fastmcp import Client as _FastMCPClient  # type: ignore[import]
    _FASTMCP_AVAILABLE = True
except ImportError:
    _FastMCPClient = None  # type: ignore[assignment,misc]
    _FASTMCP_AVAILABLE = False


class MCPConnector(BaseConnector):
    """
    Wraps any MCP server as a Verity connector.

    resource in read():
      - MCP tool name  → calls call_tool(resource, query or {})
      - URI with "://" → calls read_resource(resource)

    resource in describe():
      - None           → lists all available tools and resources
      - name/uri       → returns schema for that specific tool or resource
    """

    def __init__(
        self,
        server_url: str,
        source_id: str | None = None,
        transport: str = "stdio",
        **opts: Any,
    ) -> None:
        if not _FASTMCP_AVAILABLE:
            raise ImportError(
                "fastmcp is required for MCPConnector but is not installed. "
                "Install it with: pip install fastmcp"
            )
        super().__init__(
            source_id=source_id or f"mcp:{server_url}",
            credentials=opts.pop("credentials", None),
        )
        self._server_url = server_url
        self._transport = transport
        self._extra_opts = opts
        self._client: Any = None  # _FastMCPClient instance

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def _connect(self) -> None:
        """Initialize and enter the FastMCP client context."""
        self._client = _FastMCPClient(self._server_url)
        await self._client.__aenter__()
        logger.debug("MCPConnector: connected to '%s'.", self._server_url)

    async def _disconnect(self) -> None:
        """Exit the FastMCP client context."""
        if self._client is not None:
            await self._client.__aexit__(None, None, None)
            self._client = None
            logger.debug("MCPConnector: disconnected from '%s'.", self._server_url)

    # ── read ──────────────────────────────────────────────────────────────────

    async def read(
        self,
        resource: str,
        query: dict | None = None,
        **opts: Any,
    ) -> AsyncIterator[ConnectorRecord]:
        """
        Yield records from an MCP tool or resource.

        resource: MCP tool name, or a resource URI containing "://"
        query:    passed as tool arguments when calling a tool
        """
        if self._client is None:
            raise RuntimeError(
                "MCPConnector must be used as an async context manager. "
                "Example: async with MCPConnector(url) as conn: ..."
            )

        count = 0
        args = query or {}

        if "://" in resource:
            # Resource URI — use read_resource
            result = await self._client.read_resource(resource)
            items = result if isinstance(result, list) else [result]
            for i, item in enumerate(items):
                content = _extract_content(item)
                yield ConnectorRecord(
                    id=f"{self.source_id}:{resource}:{i}",
                    content=content,
                    source_id=self.source_id,
                    resource=resource,
                    metadata={"mcp_resource": resource, "index": i},
                    classification=_extract_classification(item),
                )
                count += 1
        else:
            # Tool name — use call_tool
            result = await self._client.call_tool(resource, args)
            items = result if isinstance(result, list) else [result]
            for i, item in enumerate(items):
                content = _extract_content(item)
                yield ConnectorRecord(
                    id=f"{self.source_id}:{resource}:{i}",
                    content=content,
                    source_id=self.source_id,
                    resource=resource,
                    metadata={"mcp_tool": resource, "args": args, "index": i},
                    classification=_extract_classification(item),
                )
                count += 1

        self._log_read(resource, count)

    # ── describe ──────────────────────────────────────────────────────────────

    async def describe(
        self,
        resource: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """
        List available tools and resources from the MCP server.

        resource=None: return all available tools and resources.
        resource=name: return schema for a specific tool or resource URI.
        """
        if self._client is None:
            return {
                "source_id": self.source_id,
                "connector_type": "MCPConnector",
                "server_url": self._server_url,
                "connected": False,
                "error": "Not connected. Use MCPConnector as an async context manager.",
            }

        tools = await self._client.list_tools() or []
        resources = await self._client.list_resources() or []

        if resource is not None:
            # Find specific tool or resource
            for t in tools:
                if getattr(t, "name", None) == resource:
                    return {
                        "source_id": self.source_id,
                        "name": t.name,
                        "description": getattr(t, "description", ""),
                        "input_schema": getattr(t, "inputSchema", {}),
                        "type": "tool",
                    }
            for r in resources:
                if str(getattr(r, "uri", "")) == resource:
                    return {
                        "source_id": self.source_id,
                        "uri": str(r.uri),
                        "description": getattr(r, "description", ""),
                        "type": "resource",
                    }
            return {
                "source_id": self.source_id,
                "error": f"Tool or resource '{resource}' not found on this server.",
            }

        return {
            "source_id": self.source_id,
            "connector_type": "MCPConnector",
            "server_url": self._server_url,
            "connected": True,
            "capabilities": [
                str(ConnectorCapability.READ),
                str(ConnectorCapability.STREAMING),
            ],
            "tools": [
                {
                    "name": getattr(t, "name", str(t)),
                    "description": getattr(t, "description", ""),
                    "type": "tool",
                }
                for t in tools
            ],
            "resources": [
                {
                    "uri": str(getattr(r, "uri", r)),
                    "description": getattr(r, "description", ""),
                    "type": "resource",
                }
                for r in resources
            ],
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract_content(item: Any) -> str | dict | bytes:
    """Extract usable content from an MCP result item."""
    if isinstance(item, (str, bytes)):
        return item
    if isinstance(item, dict):
        # MCP content items often carry a 'text' or 'data' field
        if "text" in item:
            return item["text"]  # type: ignore[return-value]
        if "data" in item:
            return item["data"]  # type: ignore[return-value]
        return item
    # MCP SDK objects (TextContent, etc.) often have .text or .data attributes
    for attr in ("text", "data", "content"):
        value = getattr(item, attr, None)
        if value is not None:
            return value  # type: ignore[return-value]
    return str(item)


def _extract_classification(item: Any) -> str:
    """Extract a DataClassification hint from MCP result metadata if present."""
    if isinstance(item, dict):
        return str(item.get("classification", "internal"))
    meta = getattr(item, "metadata", None)
    if isinstance(meta, dict):
        return str(meta.get("classification", "internal"))
    return "internal"
