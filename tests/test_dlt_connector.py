"""
tests/test_dlt_connector.py
============================
Tests for verity.core.connectors.dlt_connector and dlt_sources.

All tests are skipped if dlt is not installed. This keeps CI green when the
connectors extra is not installed, and tests properly when it is.

    pip install 'verity[connectors]'  # to run these tests

Test design:
  - Uses dlt's @dlt.resource decorator to create in-memory test sources.
    No external API, database, or filesystem access required.
  - Mocks are used only for adversarial/error cases.
  - Follows the project pattern: class per feature area, async tests,
    asyncio_mode = "auto" (no @pytest.mark.asyncio needed).
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import patch

import pytest

# Skip the entire module if dlt is not installed.
dlt = pytest.importorskip("dlt")

from verity.core.connectors import _DLT_AVAILABLE, Connector, ConnectorRecord  # noqa: E402
from verity.core.connectors.dlt_connector import DltConnector  # noqa: E402
from verity.core.connectors.dlt_sources import (  # noqa: E402
    filesystem_dlt_source,
    github_source,
    rest_api_source,
    sql_source,
)

# ── Shared test fixtures ──────────────────────────────────────────────────────


@dlt.resource
def _three_record_source() -> Any:
    """In-memory dlt resource yielding three known dicts."""
    yield {"id": "r1", "value": "alpha", "score": 1}
    yield {"id": "r2", "value": "beta",  "score": 2}
    yield {"id": "r3", "value": "gamma", "score": 3}


@dlt.resource
def _no_id_source() -> Any:
    """dlt resource yielding dicts without an 'id' field."""
    yield {"name": "Alice", "age": 30}
    yield {"name": "Bob",   "age": 25}


@dlt.resource
def _empty_source() -> Any:
    """dlt resource that yields nothing."""
    return
    yield  # type: ignore[misc]  # unreachable — marks as async generator


# ── TestDltConnectorProtocol ──────────────────────────────────────────────────


class TestDltConnectorProtocol:
    """DltConnector satisfies the Connector Protocol."""

    def test_dlt_connector_isinstance_connector(self) -> None:
        conn = DltConnector(
            source_id="test",
            source_factory=_three_record_source,
        )
        assert isinstance(conn, Connector)

    def test_dlt_available_flag_is_true_when_dlt_installed(self) -> None:
        assert _DLT_AVAILABLE is True

    def test_dlt_connector_has_read_write_describe(self) -> None:
        conn = DltConnector(source_id="test", source_factory=_three_record_source)
        assert callable(conn.read)
        assert callable(conn.write)
        assert callable(conn.describe)


# ── TestDltConnectorRead ──────────────────────────────────────────────────────


class TestDltConnectorRead:
    """DltConnector.read() yields correct ConnectorRecords."""

    async def test_read_yields_three_records(self) -> None:
        conn = DltConnector(source_id="test_src", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        assert len(records) == 3

    async def test_record_ids_match_source_ids(self) -> None:
        conn = DltConnector(source_id="test_src", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        ids = [r.id for r in records]
        assert ids == ["r1", "r2", "r3"]

    async def test_content_is_full_source_dict(self) -> None:
        conn = DltConnector(source_id="test_src", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        assert records[0].content == {"id": "r1", "value": "alpha", "score": 1}

    async def test_source_id_set_correctly(self) -> None:
        conn = DltConnector(source_id="my_source", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        assert all(r.source_id == "my_source" for r in records)

    async def test_resource_field_set_correctly(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        records = [r async for r in conn.read("my_resource")]
        assert all(r.resource == "my_resource" for r in records)

    async def test_classification_applied_to_all_records(self) -> None:
        conn = DltConnector(
            source_id="s",
            source_factory=_three_record_source,
            classification="confidential",
        )
        records = [r async for r in conn.read("default")]
        assert all(r.classification == "confidential" for r in records)

    async def test_default_classification_is_internal(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        assert all(r.classification == "internal" for r in records)

    async def test_trust_score_applied_to_all_records(self) -> None:
        conn = DltConnector(
            source_id="s",
            source_factory=_three_record_source,
            trust_score=0.9,
        )
        records = [r async for r in conn.read("default")]
        assert all(r.trust_score == 0.9 for r in records)

    async def test_default_trust_score_is_0_75(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        assert all(r.trust_score == 0.75 for r in records)

    async def test_metadata_excludes_id_field(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        # 'id' is promoted to record.id — should not appear in metadata
        assert "id" not in records[0].metadata

    async def test_metadata_contains_other_fields(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        assert records[0].metadata["value"] == "alpha"
        assert records[0].metadata["score"] == 1

    async def test_empty_source_yields_no_records(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_empty_source)
        records = [r async for r in conn.read("default")]
        assert records == []

    async def test_records_are_connector_record_instances(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        records = [r async for r in conn.read("default")]
        assert all(isinstance(r, ConnectorRecord) for r in records)

    async def test_source_kwargs_forwarded_to_factory(self) -> None:
        """source_kwargs are passed through to source_factory."""
        received_kwargs: dict[str, Any] = {}

        def _capturing_factory(**kwargs: Any) -> Any:
            received_kwargs.update(kwargs)
            return _three_record_source()

        conn = DltConnector(
            source_id="s",
            source_factory=_capturing_factory,
            source_kwargs={"page_size": 50},
        )
        _ = [r async for r in conn.read("default")]
        assert received_kwargs.get("page_size") == 50


# ── TestDltConnectorNoIdField ─────────────────────────────────────────────────


class TestDltConnectorNoIdField:
    """Records without 'id' or '_id' get a generated UUID."""

    async def test_no_id_field_generates_uuid(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_no_id_source)
        records = [r async for r in conn.read("default")]
        assert len(records) == 2
        for r in records:
            # Should be a valid UUID string
            parsed = uuid.UUID(r.id)
            assert parsed.version == 4

    async def test_each_generated_id_is_unique(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_no_id_source)
        records = [r async for r in conn.read("default")]
        ids = [r.id for r in records]
        assert len(set(ids)) == len(ids), "Generated IDs must be unique"

    async def test_underscore_id_field_used_as_id(self) -> None:
        @dlt.resource
        def _underscore_id_source() -> Any:
            yield {"_id": "uid-1", "data": "x"}

        conn = DltConnector(source_id="s", source_factory=_underscore_id_source)
        records = [r async for r in conn.read("default")]
        assert records[0].id == "uid-1"
        assert "_id" not in records[0].metadata


# ── TestDltConnectorErrorHandling ─────────────────────────────────────────────


class TestDltConnectorErrorHandling:
    """Errors during extraction are handled gracefully — no crash, log only."""

    async def test_factory_exception_yields_nothing(self) -> None:
        def _exploding_factory(**_kw: Any) -> Any:
            raise RuntimeError("Source exploded during initialisation")

        conn = DltConnector(source_id="boom", source_factory=_exploding_factory)
        records = [r async for r in conn.read("default")]
        assert records == []

    async def test_factory_exception_does_not_raise(self) -> None:
        def _exploding_factory(**_kw: Any) -> Any:
            raise ConnectionError("Network unreachable")

        conn = DltConnector(source_id="boom", source_factory=_exploding_factory)
        # Must not raise — graceful degradation
        try:
            _ = [r async for r in conn.read("default")]
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"read() raised unexpectedly: {exc}")

    async def test_mid_stream_error_stops_iteration_gracefully(self) -> None:
        """Iterator that raises mid-stream should stop without crashing."""

        def _partial_factory(**_kw: Any) -> Any:
            def _gen() -> Any:
                yield {"id": "ok1", "val": 1}
                yield {"id": "ok2", "val": 2}
                raise ValueError("Oops, mid-stream error")

            return _gen()

        conn = DltConnector(source_id="partial", source_factory=_partial_factory)
        records = [r async for r in conn.read("default")]
        # Should yield the two records that came before the error
        assert len(records) == 2
        assert records[0].id == "ok1"
        assert records[1].id == "ok2"

    async def test_import_error_raised_without_dlt(self) -> None:
        """DltConnector raises ImportError at construction when dlt unavailable."""
        with patch(
            "verity.core.connectors.dlt_connector._DLT_AVAILABLE", False
        ):
            with pytest.raises(ImportError, match="dlt"):
                DltConnector(source_id="s", source_factory=_three_record_source)


# ── TestDltConnectorDescribe ──────────────────────────────────────────────────


class TestDltConnectorDescribe:
    """DltConnector.describe() returns expected metadata."""

    async def test_describe_returns_dict(self) -> None:
        conn = DltConnector(source_id="src", source_factory=_three_record_source)
        info = await conn.describe()
        assert isinstance(info, dict)

    async def test_describe_includes_source_id(self) -> None:
        conn = DltConnector(source_id="my_src", source_factory=_three_record_source)
        info = await conn.describe()
        assert info["source_id"] == "my_src"

    async def test_describe_includes_connector_type(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        info = await conn.describe()
        assert info["connector_type"] == "DltConnector"

    async def test_describe_includes_source_name(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        info = await conn.describe()
        assert "_three_record_source" in info["source_name"]

    async def test_describe_with_resource_includes_requested_resource(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        info = await conn.describe(resource="users")
        assert info["requested_resource"] == "users"

    async def test_describe_none_does_not_include_requested_resource(self) -> None:
        conn = DltConnector(source_id="s", source_factory=_three_record_source)
        info = await conn.describe()
        assert "requested_resource" not in info


# ── TestDltSources ────────────────────────────────────────────────────────────


class TestDltSources:
    """Factory functions in dlt_sources return DltConnector with correct config."""

    def test_rest_api_source_returns_dlt_connector(self) -> None:
        conn = rest_api_source(
            base_url="https://api.example.com/v1",
            endpoints=[{"path": "users", "primary_key": "id"}],
        )
        assert isinstance(conn, DltConnector)
        assert isinstance(conn, Connector)

    def test_rest_api_source_id_defaults_to_hostname(self) -> None:
        conn = rest_api_source(
            base_url="https://api.example.com/v1",
            endpoints=[],
        )
        assert conn.source_id == "api.example.com"

    def test_rest_api_source_custom_source_id(self) -> None:
        conn = rest_api_source(
            base_url="https://api.example.com/v1",
            endpoints=[],
            source_id="my_api",
        )
        assert conn.source_id == "my_api"

    def test_rest_api_source_trust_score(self) -> None:
        conn = rest_api_source(
            base_url="https://api.example.com/v1",
            endpoints=[],
        )
        assert conn.trust_score == 0.75

    def test_sql_source_returns_dlt_connector(self) -> None:
        conn = sql_source(
            connection_string="postgresql://user:pass@localhost:5432/mydb",
        )
        assert isinstance(conn, DltConnector)
        assert isinstance(conn, Connector)

    def test_sql_source_id_defaults_to_db_name(self) -> None:
        conn = sql_source(
            connection_string="postgresql://user:pass@localhost:5432/mydb",
        )
        assert conn.source_id == "mydb"

    def test_sql_source_sqlite_path_id(self) -> None:
        conn = sql_source(connection_string="sqlite:///path/to/my_data.db")
        assert conn.source_id == "path/to/my_data.db"

    def test_sql_source_custom_source_id(self) -> None:
        conn = sql_source(
            connection_string="postgresql://user:pass@localhost/db",
            source_id="prod_db",
        )
        assert conn.source_id == "prod_db"

    def test_sql_source_high_trust_score(self) -> None:
        conn = sql_source(connection_string="sqlite:///db.sqlite")
        assert conn.trust_score == 0.85

    def test_github_source_returns_dlt_connector(self) -> None:
        conn = github_source(repo="owner/repo")
        assert isinstance(conn, DltConnector)
        assert isinstance(conn, Connector)

    def test_github_source_id_format(self) -> None:
        conn = github_source(repo="bnyhil31-afk/verity")
        assert conn.source_id == "github:bnyhil31-afk/verity"

    def test_github_source_custom_id(self) -> None:
        conn = github_source(repo="owner/repo", source_id="my_repo")
        assert conn.source_id == "my_repo"

    def test_filesystem_dlt_source_returns_dlt_connector(self) -> None:
        conn = filesystem_dlt_source(path="s3://my-bucket/data/")
        assert isinstance(conn, DltConnector)
        assert isinstance(conn, Connector)

    def test_filesystem_dlt_source_id_from_bucket(self) -> None:
        conn = filesystem_dlt_source(path="s3://my-bucket/data/")
        assert conn.source_id == "my-bucket"

    def test_filesystem_dlt_source_custom_id(self) -> None:
        conn = filesystem_dlt_source(path="gs://bucket/prefix/", source_id="gcs_data")
        assert conn.source_id == "gcs_data"
