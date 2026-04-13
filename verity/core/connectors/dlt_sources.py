"""
verity.core.connectors.dlt_sources
====================================
Pre-configured DltConnector factories for the most common data sources.

Each factory returns a ready-to-use DltConnector instance. Users can also
build their own DltConnector by passing any dlt source factory directly.

All factories require dlt to be installed:
    pip install 'verity[connectors]'   # or: pip install dlt

Available factories
-------------------
rest_api_source   — any REST API via dlt's rest_api verified source
sql_source        — any SQL database via dlt's sql_database verified source
filesystem_dlt_source — cloud storage (S3, GCS, Azure Blob) via dlt's filesystem source
github_source     — GitHub issues, PRs, and commits via dlt's github source

For LOCAL files, prefer FilesystemConnector (no dlt required).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from verity.core.connectors.dlt_connector import DltConnector

logger = logging.getLogger(__name__)

# Import dlt lazily inside each factory so import errors surface with context.
try:
    import dlt as _dlt  # type: ignore[import]
    _DLT_AVAILABLE = True
except ImportError:
    _dlt = None  # type: ignore[assignment]
    _DLT_AVAILABLE = False


def _require_dlt(factory_name: str) -> None:
    """Raise a helpful ImportError if dlt is not installed."""
    if not _DLT_AVAILABLE:
        raise ImportError(
            f"{factory_name} requires dlt but it is not installed. "
            "Install it with: pip install 'verity[connectors]'  "
            "or: pip install dlt"
        )


# ── rest_api_source ───────────────────────────────────────────────────────────


def rest_api_source(
    base_url: str,
    endpoints: list[dict[str, Any]],
    auth: dict[str, Any] | None = None,
    source_id: str | None = None,
    **kwargs: Any,
) -> DltConnector:
    """
    Connect to any REST API using dlt's rest_api verified source.

    Parameters
    ----------
    base_url:  Base URL of the API (e.g. "https://api.example.com/v1").
    endpoints: List of endpoint dicts following dlt's rest_api schema.
               Minimal example:
                   [{"path": "users", "primary_key": "id"}]
               Full example:
                   [{
                       "path": "posts",
                       "primary_key": "id",
                       "params": {"per_page": 100},
                       "data_selector": "items",
                   }]
    auth:      Optional auth config dict. Examples:
                   {"type": "bearer", "token": "my_token"}
                   {"type": "api_key", "name": "X-API-Key", "api_key": "key"}
    source_id: Connector identifier. Defaults to the API hostname.
    **kwargs:  Additional kwargs forwarded to the dlt rest_api source.

    Credentials
    -----------
    API keys / tokens are passed via the auth dict. For OAuth flows, use
    dlt's secrets.toml or environment variables (DLT_SECRETS_* prefix).

    Example
    -------
    from verity.core.connectors.dlt_sources import rest_api_source

    conn = rest_api_source(
        base_url="https://jsonplaceholder.typicode.com",
        endpoints=[{"path": "posts", "primary_key": "id"}],
        source_id="jsonplaceholder",
    )
    async for record in conn.read("posts"):
        print(record.content["title"])
    """
    _require_dlt("rest_api_source")
    from verity.core.connectors.dlt_connector import DltConnector

    derived_id = source_id or urlparse(base_url).netloc or "rest_api"

    config: dict[str, Any] = {
        "client": {"base_url": base_url},
        "resources": endpoints,
    }
    if auth is not None:
        config["client"]["auth"] = auth

    def _factory(**_kw: Any) -> Any:
        from dlt.sources.rest_api import rest_api as _rest_api  # type: ignore[import]
        merged = {**config, **_kw}
        return _rest_api(merged)

    _factory.__name__ = f"rest_api({derived_id})"

    return DltConnector(
        source_id=derived_id,
        source_factory=_factory,
        source_kwargs=kwargs,
        trust_score=0.75,
    )


# ── sql_source ────────────────────────────────────────────────────────────────


def sql_source(
    connection_string: str,
    tables: list[str] | None = None,
    source_id: str | None = None,
    **kwargs: Any,
) -> DltConnector:
    """
    Connect to any SQL database using dlt's sql_database verified source.

    Supports PostgreSQL, MySQL, SQLite, MS SQL Server, BigQuery, and any
    SQLAlchemy-compatible database.

    Parameters
    ----------
    connection_string: SQLAlchemy connection URL.
                       Examples:
                           "postgresql://user:pass@host:5432/mydb"
                           "mysql+pymysql://user:pass@host/mydb"
                           "sqlite:///path/to/db.sqlite"
    tables:   List of table names to include. None = all tables (may be slow
              on large schemas; prefer explicit table lists in production).
    source_id: Connector identifier. Defaults to the database name extracted
               from the connection string.
    **kwargs: Additional kwargs forwarded to the dlt sql_database source
              (e.g. chunk_size=1000, reflect_columns=True).

    Credentials
    -----------
    Embed credentials in the connection_string or use dlt's secrets.toml /
    environment variables. Never hard-code credentials in source code.

    Example
    -------
    from verity.core.connectors.dlt_sources import sql_source

    conn = sql_source(
        connection_string="sqlite:///my_data.db",
        tables=["users", "orders"],
        source_id="my_sqlite_db",
    )
    async for record in conn.read("users"):
        print(record.content["email"])
    """
    _require_dlt("sql_source")
    from verity.core.connectors.dlt_connector import DltConnector

    # Derive source_id from the database name in the connection string.
    parsed = urlparse(connection_string)
    db_name = (parsed.path or "").lstrip("/").split("?")[0] or "sql_database"
    derived_id = source_id or db_name

    def _factory(**_kw: Any) -> Any:
        from dlt.sources.sql_database import sql_database as _sql_database  # type: ignore[import]
        call_kwargs: dict[str, Any] = {"credentials": connection_string, **_kw}
        if tables is not None:
            call_kwargs["table_names"] = tables
        return _sql_database(**call_kwargs)

    _factory.__name__ = f"sql_database({derived_id})"

    return DltConnector(
        source_id=derived_id,
        source_factory=_factory,
        source_kwargs=kwargs,
        trust_score=0.85,  # SQL databases are well-structured, high trust
    )


# ── filesystem_dlt_source ─────────────────────────────────────────────────────


def filesystem_dlt_source(
    path: str,
    file_glob: str = "**/*",
    source_id: str | None = None,
    **kwargs: Any,
) -> DltConnector:
    """
    Connect to cloud storage (S3, GCS, Azure Blob) via dlt's filesystem source.

    For LOCAL files, prefer FilesystemConnector (no dlt required).
    This factory is for cloud storage paths.

    Parameters
    ----------
    path:      Storage path. Supported protocols:
                   "s3://bucket/prefix/"            — AWS S3
                   "gs://bucket/prefix/"            — Google Cloud Storage
                   "az://container/prefix/"         — Azure Blob Storage
                   "/local/path/"                   — Local filesystem (via dlt;
                                                       prefer FilesystemConnector)
    file_glob: Glob pattern for filtering files within the path.
               Default "**/*" matches all files recursively.
               Example: "**/*.parquet" for only Parquet files.
    source_id: Connector identifier. Defaults to the bucket/container name.
    **kwargs:  Additional kwargs forwarded to the dlt filesystem source.

    Credentials
    -----------
    S3:     Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION
            or configure in dlt's secrets.toml under [sources.filesystem.credentials].
    GCS:    Set GOOGLE_APPLICATION_CREDENTIALS to a service account JSON file.
    Azure:  Set AZURE_STORAGE_ACCOUNT_NAME and AZURE_STORAGE_ACCOUNT_KEY.

    Example
    -------
    from verity.core.connectors.dlt_sources import filesystem_dlt_source

    conn = filesystem_dlt_source(
        path="s3://my-bucket/data/",
        file_glob="**/*.json",
        source_id="s3_data_lake",
    )
    async for record in conn.read("default"):
        print(record.metadata["file_url"])
    """
    _require_dlt("filesystem_dlt_source")
    from verity.core.connectors.dlt_connector import DltConnector

    # Derive source_id from bucket/container name in the path.
    parsed = urlparse(path)
    bucket_name = parsed.netloc or parsed.path.lstrip("/").split("/")[0] or "filesystem"
    derived_id = source_id or bucket_name

    def _factory(**_kw: Any) -> Any:
        from dlt.sources.filesystem import filesystem as _filesystem  # type: ignore[import]
        return _filesystem(bucket_url=path, file_glob=file_glob, **_kw)

    _factory.__name__ = f"filesystem({derived_id})"

    return DltConnector(
        source_id=derived_id,
        source_factory=_factory,
        source_kwargs=kwargs,
        trust_score=0.70,
    )


# ── github_source ─────────────────────────────────────────────────────────────


def github_source(
    repo: str,
    access_token: str | None = None,
    source_id: str | None = None,
    **kwargs: Any,
) -> DltConnector:
    """
    Connect to a GitHub repository via dlt's github verified source.

    Yields issues, pull requests, and commits as ConnectorRecords.

    Parameters
    ----------
    repo:         Repository in "owner/repo" format (e.g. "bnyhil31-afk/verity").
    access_token: GitHub Personal Access Token (PAT) or fine-grained token.
                  Optional for public repos; required for private repos and to
                  avoid rate limiting. Defaults to GITHUB_TOKEN env var if set.
    source_id:    Connector identifier. Defaults to "github:{repo}".
    **kwargs:     Additional kwargs forwarded to the dlt github source.

    Credentials
    -----------
    Set GITHUB_TOKEN environment variable or pass access_token directly.
    Fine-grained PATs scoped to "Contents: Read" and "Issues: Read" are sufficient
    for most read operations.

    Yields
    ------
    ConnectorRecords representing:
      - GitHub issues   (resource="issues")
      - Pull requests   (resource="pull_requests")
      - Stargazers      (resource="stargazers")

    Example
    -------
    from verity.core.connectors.dlt_sources import github_source

    conn = github_source(
        repo="bnyhil31-afk/verity",
        source_id="verity_github",
    )
    async for record in conn.read("issues"):
        print(record.content["title"], record.content["state"])
    """
    _require_dlt("github_source")
    from verity.core.connectors.dlt_connector import DltConnector

    owner, _, name = repo.partition("/")
    derived_id = source_id or f"github:{repo}"

    def _factory(**_kw: Any) -> Any:
        from dlt.sources.github import github_reactions as _github  # type: ignore[import]
        call_kwargs: dict[str, Any] = {"owner": owner, "name": name, **_kw}
        if access_token is not None:
            call_kwargs["access_token"] = access_token
        return _github(**call_kwargs)

    _factory.__name__ = f"github({repo})"

    return DltConnector(
        source_id=derived_id,
        source_factory=_factory,
        source_kwargs=kwargs,
        classification="internal",
        trust_score=0.80,  # GitHub is a structured, verifiable source
    )
