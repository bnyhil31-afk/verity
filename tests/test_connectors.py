"""
tests/test_connectors.py
========================
Tests for verity.core.connectors — Protocol compliance, ConnectorRecord
shape, FilesystemConnector read/write/describe, and ConnectorRegistry.

All async tests use asyncio_mode = "auto" (configured in pyproject.toml),
so no @pytest.mark.asyncio decorator is needed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from verity.core.connectors import Connector, ConnectorCapability, ConnectorRecord
from verity.core.connectors.filesystem import FilesystemConnector
from verity.core.connectors.registry import ConnectorRegistry

# ── TestConnectorRecord ───────────────────────────────────────────────────────


class TestConnectorRecord:
    """ConnectorRecord dataclass shape and defaults."""

    def test_required_fields_accepted(self) -> None:
        rec = ConnectorRecord(
            id="test:1",
            content="hello world",
            source_id="test",
            resource="/path/to/file.txt",
        )
        assert rec.id == "test:1"
        assert rec.content == "hello world"
        assert rec.source_id == "test"
        assert rec.resource == "/path/to/file.txt"

    def test_classification_defaults_to_internal(self) -> None:
        rec = ConnectorRecord(id="1", content="x", source_id="s", resource="r")
        assert rec.classification == "internal"

    def test_trust_score_defaults_to_half(self) -> None:
        rec = ConnectorRecord(id="1", content="x", source_id="s", resource="r")
        assert rec.trust_score == 0.5

    def test_timestamp_is_utc(self) -> None:
        rec = ConnectorRecord(id="1", content="x", source_id="s", resource="r")
        assert rec.timestamp.tzinfo is not None
        assert rec.timestamp.utcoffset().total_seconds() == 0  # type: ignore[union-attr]

    def test_metadata_defaults_to_empty_dict(self) -> None:
        rec = ConnectorRecord(id="1", content="x", source_id="s", resource="r")
        assert rec.metadata == {}

    def test_dict_content_accepted(self) -> None:
        rec = ConnectorRecord(
            id="1", content={"key": "value"}, source_id="s", resource="r"
        )
        assert isinstance(rec.content, dict)
        assert rec.content["key"] == "value"

    def test_bytes_content_accepted(self) -> None:
        rec = ConnectorRecord(
            id="1", content=b"\x00\x01\x02", source_id="s", resource="r"
        )
        assert isinstance(rec.content, bytes)

    def test_custom_classification(self) -> None:
        rec = ConnectorRecord(
            id="1", content="x", source_id="s", resource="r",
            classification="phi",
        )
        assert rec.classification == "phi"

    def test_custom_trust_score(self) -> None:
        rec = ConnectorRecord(
            id="1", content="x", source_id="s", resource="r",
            trust_score=0.9,
        )
        assert rec.trust_score == 0.9


# ── TestConnectorCapability ───────────────────────────────────────────────────


class TestConnectorCapability:
    """ConnectorCapability StrEnum values."""

    def test_read_value(self) -> None:
        assert ConnectorCapability.READ == "read"

    def test_write_value(self) -> None:
        assert ConnectorCapability.WRITE == "write"

    def test_streaming_value(self) -> None:
        assert ConnectorCapability.STREAMING == "streaming"

    def test_batch_value(self) -> None:
        assert ConnectorCapability.BATCH == "batch"

    def test_search_value(self) -> None:
        assert ConnectorCapability.SEARCH == "search"

    def test_is_str_subtype(self) -> None:
        assert isinstance(ConnectorCapability.READ, str)


# ── TestConnectorProtocol ─────────────────────────────────────────────────────


class TestConnectorProtocol:
    """runtime_checkable isinstance checks for the Connector Protocol."""

    def test_filesystem_connector_satisfies_protocol(self) -> None:
        conn = FilesystemConnector()
        assert isinstance(conn, Connector)

    def test_minimal_three_method_class_satisfies_protocol(self) -> None:
        class MinimalConnector:
            async def read(
                self, resource: str, query: dict | None = None, **opts: Any
            ):  # type: ignore[override]
                yield  # async generator

            async def write(
                self,
                resource: str,
                data: Any,
                **opts: Any,
            ) -> dict[str, Any]:
                return {}

            async def describe(
                self, resource: str | None = None, **opts: Any
            ) -> dict[str, Any]:
                return {}

        assert isinstance(MinimalConnector(), Connector)

    def test_class_missing_write_fails_protocol(self) -> None:
        class NoWrite:
            async def read(self, resource: str, query: dict | None = None, **opts: Any):  # type: ignore[override]
                yield

            async def describe(
                self, resource: str | None = None, **opts: Any
            ) -> dict[str, Any]:
                return {}

        assert not isinstance(NoWrite(), Connector)

    def test_class_missing_describe_fails_protocol(self) -> None:
        class NoDescribe:
            async def read(self, resource: str, query: dict | None = None, **opts: Any):  # type: ignore[override]
                yield

            async def write(self, resource: str, data: Any, **opts: Any) -> dict[str, Any]:
                return {}

        assert not isinstance(NoDescribe(), Connector)

    def test_class_missing_read_fails_protocol(self) -> None:
        class NoRead:
            async def write(self, resource: str, data: Any, **opts: Any) -> dict[str, Any]:
                return {}

            async def describe(
                self, resource: str | None = None, **opts: Any
            ) -> dict[str, Any]:
                return {}

        assert not isinstance(NoRead(), Connector)

    def test_plain_object_fails_protocol(self) -> None:
        assert not isinstance(object(), Connector)


# ── TestFilesystemConnector ───────────────────────────────────────────────────


class TestFilesystemConnector:
    """FilesystemConnector read / write / describe round-trips."""

    async def test_read_txt_file(self, tmp_path: Path) -> None:
        f = tmp_path / "note.txt"
        f.write_text("Hello, Verity!", encoding="utf-8")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(f))]

        assert len(records) == 1
        assert records[0].content == "Hello, Verity!"
        assert records[0].source_id == "filesystem"
        assert records[0].classification == "internal"
        assert records[0].resource == str(f)

    async def test_read_md_file(self, tmp_path: Path) -> None:
        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nBody text.", encoding="utf-8")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(f))]

        assert len(records) == 1
        assert "# Title" in records[0].content  # type: ignore[operator]

    async def test_read_json_object_yields_one_record(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text(json.dumps({"name": "Alice", "score": 99}), encoding="utf-8")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(f))]

        assert len(records) == 1
        assert records[0].content == {"name": "Alice", "score": 99}

    async def test_read_json_array_yields_one_record_per_item(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "items.json"
        f.write_text(json.dumps([{"id": 1}, {"id": 2}, {"id": 3}]), encoding="utf-8")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(f))]

        assert len(records) == 3
        assert records[0].content == {"id": 1}
        assert records[2].content == {"id": 3}

    async def test_read_csv_yields_one_record_per_row(self, tmp_path: Path) -> None:
        f = tmp_path / "data.csv"
        f.write_text("name,score\nAlice,90\nBob,85\n", encoding="utf-8")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(f))]

        assert len(records) == 2
        assert records[0].content == {"name": "Alice", "score": "90"}
        assert records[1].content == {"name": "Bob", "score": "85"}

    async def test_read_yaml_file(self, tmp_path: Path) -> None:
        f = tmp_path / "config.yaml"
        f.write_text("host: localhost\nport: 5432\n", encoding="utf-8")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(f))]

        assert len(records) == 1
        assert records[0].content == {"host": "localhost", "port": 5432}

    async def test_read_glob_pattern_matches_multiple_files(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "a.txt").write_text("file a", encoding="utf-8")
        (tmp_path / "b.txt").write_text("file b", encoding="utf-8")
        (tmp_path / "c.md").write_text("file c", encoding="utf-8")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(tmp_path / "*.txt"))]

        assert len(records) == 2
        contents = {r.content for r in records}
        assert contents == {"file a", "file b"}

    async def test_read_nonexistent_path_yields_nothing(
        self, tmp_path: Path
    ) -> None:
        conn = FilesystemConnector()
        records = [
            r async for r in conn.read(str(tmp_path / "does_not_exist.txt"))
        ]
        assert records == []

    async def test_read_unsupported_format_is_skipped(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "binary.bin"
        f.write_bytes(b"\x00\x01\x02")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(f))]
        assert records == []

    async def test_read_metadata_includes_filename_and_size(
        self, tmp_path: Path
    ) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello", encoding="utf-8")

        conn = FilesystemConnector()
        records = [r async for r in conn.read(str(f))]

        assert records[0].metadata["filename"] == "hello.txt"
        assert records[0].metadata["size"] == 5
        assert records[0].metadata["format"] == "txt"

    async def test_describe_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "data.json"
        f.write_text("{}", encoding="utf-8")

        conn = FilesystemConnector()
        info = await conn.describe(str(f))

        assert info["exists"] is True
        assert info["format"] == "json"
        assert info["supported"] is True
        assert "size" in info

    async def test_describe_nonexistent_file(self, tmp_path: Path) -> None:
        conn = FilesystemConnector()
        info = await conn.describe(str(tmp_path / "ghost.txt"))
        assert info["exists"] is False

    async def test_describe_none_returns_base_info(self, tmp_path: Path) -> None:
        conn = FilesystemConnector(base_path=tmp_path)
        info = await conn.describe()
        assert "capabilities" in info
        assert "supported_formats" in info
        assert "base_path" in info

    async def test_write_creates_json_file(self, tmp_path: Path) -> None:
        output = tmp_path / "output.json"
        conn = FilesystemConnector()
        records = [
            ConnectorRecord(id="r1", content={"x": 1}, source_id="t", resource="s"),
            ConnectorRecord(id="r2", content={"x": 2}, source_id="t", resource="s"),
        ]
        stats = await conn.write(str(output), records)

        assert output.exists()
        assert stats["records_written"] == 2
        assert stats["path"] == str(output)
        loaded = json.loads(output.read_text())
        assert len(loaded) == 2
        assert loaded[0] == {"x": 1}

    async def test_write_creates_txt_file(self, tmp_path: Path) -> None:
        output = tmp_path / "output.txt"
        conn = FilesystemConnector()
        records = [
            ConnectorRecord(id="r1", content="line one", source_id="t", resource="s"),
            ConnectorRecord(id="r2", content="line two", source_id="t", resource="s"),
        ]
        stats = await conn.write(str(output), records)

        assert output.exists()
        assert stats["records_written"] == 2
        text = output.read_text()
        assert "line one" in text
        assert "line two" in text

    async def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        output = tmp_path / "nested" / "deep" / "output.json"
        conn = FilesystemConnector()
        await conn.write(str(output), [])
        assert output.exists()

    async def test_read_round_trip_via_write(self, tmp_path: Path) -> None:
        """Write records to a file, then read them back."""
        output = tmp_path / "round_trip.json"
        conn = FilesystemConnector()

        original = [
            ConnectorRecord(id="r1", content={"value": 42}, source_id="t", resource="s"),
        ]
        await conn.write(str(output), original)

        recovered = [r async for r in conn.read(str(output))]
        assert len(recovered) == 1
        assert recovered[0].content == {"value": 42}


# ── TestConnectorRegistry ─────────────────────────────────────────────────────


class TestConnectorRegistry:
    """ConnectorRegistry register / get / list / discover."""

    def test_register_and_get(self, tmp_path: Path) -> None:
        registry = ConnectorRegistry()
        conn = FilesystemConnector(source_id="fs_test", base_path=tmp_path)
        registry.register("fs_test", conn)
        assert registry.get("fs_test") is conn

    def test_get_raises_key_error_for_unknown_id(self) -> None:
        registry = ConnectorRegistry()
        with pytest.raises(KeyError, match="unknown_connector"):
            registry.get("unknown_connector")

    def test_key_error_message_lists_available_ids(self) -> None:
        registry = ConnectorRegistry()
        registry.register("a", FilesystemConnector(source_id="a"))
        with pytest.raises(KeyError, match="'a'"):
            registry.get("missing")

    async def test_list_returns_describe_output_for_all_connectors(
        self, tmp_path: Path
    ) -> None:
        registry = ConnectorRegistry()
        conn_a = FilesystemConnector(source_id="fs_a", base_path=tmp_path)
        conn_b = FilesystemConnector(source_id="fs_b", base_path=tmp_path)
        registry.register("fs_a", conn_a)
        registry.register("fs_b", conn_b)

        result = await registry.list()

        assert "fs_a" in result
        assert "fs_b" in result
        assert result["fs_a"]["source_id"] == "fs_a"
        assert result["fs_b"]["source_id"] == "fs_b"

    async def test_list_empty_registry_returns_empty_dict(self) -> None:
        registry = ConnectorRegistry()
        assert await registry.list() == {}

    def test_len_reflects_registered_count(self) -> None:
        registry = ConnectorRegistry()
        assert len(registry) == 0
        registry.register("a", FilesystemConnector(source_id="a"))
        assert len(registry) == 1
        registry.register("b", FilesystemConnector(source_id="b"))
        assert len(registry) == 2

    def test_contains_registered_id(self) -> None:
        registry = ConnectorRegistry()
        registry.register("fs", FilesystemConnector(source_id="fs"))
        assert "fs" in registry

    def test_not_contains_unregistered_id(self) -> None:
        registry = ConnectorRegistry()
        assert "fs" not in registry

    def test_register_rejects_non_connector(self) -> None:
        registry = ConnectorRegistry()
        with pytest.raises(TypeError, match="Connector Protocol"):
            registry.register("bad", object())  # type: ignore[arg-type]

    def test_discover_finds_no_connectors_when_none_registered(self) -> None:
        """discover() should not raise even when no entry points exist."""
        registry = ConnectorRegistry()
        registry.discover()  # Should complete without error
        assert len(registry) == 0

    def test_register_overwrites_existing_id(self) -> None:
        registry = ConnectorRegistry()
        conn_a = FilesystemConnector(source_id="fs")
        conn_b = FilesystemConnector(source_id="fs")
        registry.register("fs", conn_a)
        registry.register("fs", conn_b)
        assert registry.get("fs") is conn_b
        assert len(registry) == 1
