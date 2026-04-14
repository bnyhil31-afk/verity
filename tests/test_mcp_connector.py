import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from verity.core.connectors.mcp_client import MCPConnector, _extract_content

# Convenience: patch both availability flag and the client class together.
_AVAIL = "verity.core.connectors.mcp_client._FASTMCP_AVAILABLE"
_CLIENT = "verity.core.connectors.mcp_client._FastMCPClient"


class TestExtractContent:

    def test_string_passthrough(self):
        assert _extract_content("hello") == "hello"

    def test_bytes_decoded(self):
        assert _extract_content(b"hello") == "hello"

    def test_bytes_invalid_utf8_does_not_raise(self):
        result = _extract_content(bytes([0xFF, 0xFE, 0x41]))
        assert isinstance(result, str)

    def test_dict_with_text_key(self):
        assert _extract_content({"text": "hello"}) == "hello"

    def test_dict_with_data_key(self):
        assert _extract_content({"data": "payload"}) == "payload"

    def test_dict_with_data_bytes(self):
        assert _extract_content({"data": b"bytes"}) == "bytes"

    def test_dict_without_known_keys_is_json(self):
        result = _extract_content({"key": "val", "num": 42})
        parsed = json.loads(result)
        assert parsed["key"] == "val"
        assert parsed["num"] == 42

    def test_object_with_text_attr(self):
        obj = MagicMock(spec=["text"])
        obj.text = "from attr"
        assert _extract_content(obj) == "from attr"

    def test_always_returns_str(self):
        """The core contract: every input must produce a str."""
        inputs = [
            "text", b"bytes",
            {"text": "val"}, {"data": b"raw"}, {"unknown": 1},
            42, None, object(),
        ]
        for inp in inputs:
            result = _extract_content(inp)
            assert isinstance(result, str), (
                f"Expected str for {type(inp).__name__}, got {type(result).__name__}"
            )


class TestMCPConnectorConstruction:

    def test_raises_if_fastmcp_not_installed(self, monkeypatch):
        """ImportError at construction time, not at import time."""
        import verity.core.connectors.mcp_client as mod
        monkeypatch.setattr(mod, "_FASTMCP_AVAILABLE", False)
        with pytest.raises(ImportError, match="fastmcp"):
            MCPConnector("https://example.com/mcp")

    def test_source_id_defaults_to_mcp_prefix(self, monkeypatch):
        """When no source_id given, defaults to 'mcp:{url}'."""
        import verity.core.connectors.mcp_client as mod
        monkeypatch.setattr(mod, "_FASTMCP_AVAILABLE", True)
        monkeypatch.setattr(mod, "_FastMCPClient", lambda url: None)
        conn = MCPConnector("https://example.com/mcp")
        assert conn.source_id == "mcp:https://example.com/mcp"

    def test_custom_source_id_is_used(self, monkeypatch):
        """Explicit source_id overrides the default."""
        import verity.core.connectors.mcp_client as mod
        monkeypatch.setattr(mod, "_FASTMCP_AVAILABLE", True)
        monkeypatch.setattr(mod, "_FastMCPClient", lambda url: None)
        conn = MCPConnector("https://example.com/mcp", source_id="my_cal")
        assert conn.source_id == "my_cal"


class TestMCPConnectorLifecycle:

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=False)
        client.list_tools = AsyncMock(return_value=[])
        client.list_resources = AsyncMock(return_value=[])
        return client

    @pytest.mark.asyncio
    async def test_connect_enters_fastmcp_context(self, mock_client):
        with patch(_AVAIL, True), patch(_CLIENT, return_value=mock_client):
            conn = MCPConnector("https://example.com/mcp")
            await conn._connect()
            mock_client.__aenter__.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_exits_fastmcp_context(self, mock_client):
        with patch(_AVAIL, True), patch(_CLIENT, return_value=mock_client):
            conn = MCPConnector("https://example.com/mcp")
            conn._client = mock_client
            await conn._disconnect()
            mock_client.__aexit__.assert_called_once()
            assert conn._client is None

    @pytest.mark.asyncio
    async def test_describe_when_not_connected_returns_error_dict(self):
        with patch(_AVAIL, True), patch(_CLIENT, return_value=AsyncMock()):
            conn = MCPConnector("https://example.com/mcp")
            # _client is None — not connected
            result = await conn.describe()
            assert result["connected"] is False
            assert "error" in result


class TestMCPConnectorRead:

    @pytest.fixture
    def conn(self):
        """MCPConnector with mocked _client already set."""
        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(
            return_value=[{"text": "Event: team standup at 9am"}]
        )
        mock_client.read_resource = AsyncMock(
            return_value=[{"text": "Resource content"}]
        )
        with patch(_AVAIL, True), patch(_CLIENT, return_value=mock_client):
            c = MCPConnector("https://example.com/mcp", source_id="test")
        c._client = mock_client
        c._log_read = MagicMock()  # avoid any BaseConnector dependency
        return c

    @pytest.mark.asyncio
    async def test_tool_call_yields_records(self, conn):
        records = [r async for r in conn.read("list_events")]
        assert len(records) == 1
        assert records[0].content == "Event: team standup at 9am"
        assert records[0].resource == "list_events"
        assert "mcp_tool" in records[0].metadata

    @pytest.mark.asyncio
    async def test_resource_uri_dispatches_to_read_resource(self, conn):
        records = [r async for r in conn.read("calendar://primary/events")]
        assert len(records) == 1
        assert records[0].content == "Resource content"
        assert "mcp_resource" in records[0].metadata

    @pytest.mark.asyncio
    async def test_read_without_context_manager_raises_runtime_error(self):
        with patch(_AVAIL, True), patch(_CLIENT, return_value=AsyncMock()):
            conn = MCPConnector("https://example.com/mcp")
            # _client is None — raises RuntimeError before first yield
            with pytest.raises(RuntimeError, match="async context manager"):
                async for _ in conn.read("list_events"):
                    pass

    @pytest.mark.asyncio
    async def test_query_dict_passed_as_tool_args(self, conn):
        async for _ in conn.read(
            "list_events", query={"calendar_id": "primary"}
        ):
            pass
        conn._client.call_tool.assert_called_once_with(
            "list_events", {"calendar_id": "primary"}
        )

    @pytest.mark.asyncio
    async def test_content_is_always_str(self, conn):
        """Core contract: ConnectorRecord.content is always str."""
        # Mock returns a raw dict with no known text/data key
        conn._client.call_tool = AsyncMock(
            return_value=[{"unknown_key": 42, "other": True}]
        )
        records = [r async for r in conn.read("some_tool")]
        for record in records:
            assert isinstance(record.content, str), (
                f"content must be str, got {type(record.content)}"
            )


class TestMCPConnectorDescribe:

    @pytest.fixture
    def conn(self):
        mock_tool = MagicMock()
        mock_tool.name = "list_events"
        mock_tool.description = "List calendar events"
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        mock_resource = MagicMock()
        mock_resource.uri = "calendar://primary"
        mock_resource.description = "Primary calendar"

        mock_client = AsyncMock()
        mock_client.list_tools = AsyncMock(return_value=[mock_tool])
        mock_client.list_resources = AsyncMock(return_value=[mock_resource])

        with patch(_AVAIL, True), patch(_CLIENT, return_value=mock_client):
            c = MCPConnector("https://example.com/mcp", source_id="gcal")
        c._client = mock_client
        return c

    @pytest.mark.asyncio
    async def test_lists_tools_and_resources(self, conn):
        result = await conn.describe()
        assert result["connected"] is True
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "list_events"
        assert len(result["resources"]) == 1
        assert result["resources"][0]["uri"] == "calendar://primary"

    @pytest.mark.asyncio
    async def test_specific_tool_schema(self, conn):
        result = await conn.describe("list_events")
        assert result["name"] == "list_events"
        assert result["type"] == "tool"
        assert "input_schema" in result

    @pytest.mark.asyncio
    async def test_unknown_resource_returns_error(self, conn):
        result = await conn.describe("nonexistent_tool")
        assert "error" in result
