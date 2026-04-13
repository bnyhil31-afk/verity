"""
verity.core.connectors.filesystem
==================================
FilesystemConnector — reads local files. Zero external dependencies.

resource: file path or glob pattern (e.g. "~/notes/**/*.md")

Supported formats:
  .txt / .md   — yield one ConnectorRecord per file (content: str)
  .json        — yield one record for objects, one per item for arrays
  .csv         — yield one ConnectorRecord per row (content: dict)
  .yaml / .yml — yield one ConnectorRecord per file (content: dict)

Handles ~ expansion and missing paths gracefully (log warning, yield nothing).
"""

from __future__ import annotations

import csv
import glob as _glob
import json
import logging
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from verity.core.connectors import ConnectorCapability, ConnectorRecord
from verity.core.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

_SUPPORTED_FORMATS: frozenset[str] = frozenset(
    {".txt", ".md", ".json", ".csv", ".yaml", ".yml"}
)

_CAPABILITIES = [
    ConnectorCapability.READ,
    ConnectorCapability.WRITE,
    ConnectorCapability.STREAMING,
]


class FilesystemConnector(BaseConnector):
    """
    Reads (and writes) local files. Zero external dependencies beyond PyYAML.

    resource: file path or glob pattern (e.g. "~/notes/**/*.md")
    Supported: .txt, .md, .json, .csv, .yaml, .yml
    """

    def __init__(
        self,
        source_id: str = "filesystem",
        credentials: dict | None = None,
        base_path: str | Path | None = None,
    ) -> None:
        super().__init__(source_id=source_id, credentials=credentials)
        self._base_path: Path | None = (
            Path(base_path).expanduser() if base_path else None
        )

    # ── read ──────────────────────────────────────────────────────────────────

    async def read(
        self,
        resource: str,
        query: dict | None = None,
        **opts: Any,
    ) -> AsyncIterator[ConnectorRecord]:
        """
        Yield records from files matching resource (path or glob pattern).

        Unsupported file types are skipped with a debug log.
        Missing paths produce a warning and zero records — no exception raised.
        """
        resource_expanded = str(Path(resource).expanduser())

        # Resolve relative paths against base_path when set
        if (
            self._base_path is not None
            and not Path(resource).is_absolute()
            and not resource.startswith("~")
        ):
            resource_expanded = str(self._base_path / resource)

        # Resolve to concrete paths
        is_glob = any(c in resource_expanded for c in ("*", "?", "["))
        if is_glob:
            paths = sorted(
                Path(p) for p in _glob.glob(resource_expanded, recursive=True)
            )
        else:
            candidate = Path(resource_expanded)
            if not candidate.exists():
                logger.warning(
                    "FilesystemConnector: path not found: '%s'", resource_expanded
                )
                return
            paths = [candidate]

        count = 0
        for path in paths:
            if not path.is_file():
                continue
            if path.suffix.lower() not in _SUPPORTED_FORMATS:
                logger.debug(
                    "FilesystemConnector: skipping unsupported format '%s'", path
                )
                continue
            try:
                async for record in self._read_file(path):
                    count += 1
                    yield record
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "FilesystemConnector: error reading '%s': %s", path, exc
                )

        self._log_read(resource, count)

    async def _read_file(
        self, path: Path
    ) -> AsyncIterator[ConnectorRecord]:
        """Yield one or more ConnectorRecords from a single file."""
        suffix = path.suffix.lower()
        stat = path.stat()
        base_meta: dict[str, Any] = {
            "filename": path.name,
            "path": str(path),
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=UTC
            ).isoformat(),
            "format": suffix.lstrip("."),
        }

        if suffix in (".txt", ".md"):
            yield ConnectorRecord(
                id=f"{self.source_id}:{path}",
                content=path.read_text(encoding="utf-8"),
                source_id=self.source_id,
                resource=str(path),
                metadata=base_meta,
            )

        elif suffix == ".json":
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                for i, item in enumerate(data):
                    yield ConnectorRecord(
                        id=f"{self.source_id}:{path}:{i}",
                        content=item,
                        source_id=self.source_id,
                        resource=str(path),
                        metadata={**base_meta, "index": i},
                    )
            else:
                yield ConnectorRecord(
                    id=f"{self.source_id}:{path}",
                    content=data,
                    source_id=self.source_id,
                    resource=str(path),
                    metadata=base_meta,
                )

        elif suffix == ".csv":
            text = path.read_text(encoding="utf-8")
            reader = csv.DictReader(text.splitlines())
            for i, row in enumerate(reader):
                yield ConnectorRecord(
                    id=f"{self.source_id}:{path}:{i}",
                    content=dict(row),
                    source_id=self.source_id,
                    resource=str(path),
                    metadata={**base_meta, "index": i},
                )

        elif suffix in (".yaml", ".yml"):
            raw = path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw)
            if not isinstance(data, dict):
                data = {"data": data}
            yield ConnectorRecord(
                id=f"{self.source_id}:{path}",
                content=data,
                source_id=self.source_id,
                resource=str(path),
                metadata=base_meta,
            )

    # ── write ─────────────────────────────────────────────────────────────────

    async def write(
        self,
        resource: str,
        data: AsyncIterator[ConnectorRecord] | list[ConnectorRecord],
        **opts: Any,
    ) -> dict[str, Any]:
        """
        Write ConnectorRecords to a file.

        Supported output formats: .json (default) and .txt.
        The output directory is created if it does not exist.
        """
        output_path = Path(resource).expanduser()
        suffix = output_path.suffix.lower()

        records: list[ConnectorRecord]
        if isinstance(data, list):
            records = data
        else:
            records = [r async for r in data]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if suffix == ".json":
            payload = [
                r.content
                if isinstance(r.content, (dict, list))
                else {"content": r.content}
                for r in records
            ]
            output_path.write_text(
                json.dumps(payload, indent=2, default=str), encoding="utf-8"
            )
        else:
            # Plain text — one record per line
            lines = [
                r.content if isinstance(r.content, str) else json.dumps(r.content, default=str)
                for r in records
            ]
            output_path.write_text("\n".join(lines), encoding="utf-8")

        logger.info(
            "FilesystemConnector: wrote %d record(s) to '%s'.",
            len(records),
            output_path,
        )
        return {"records_written": len(records), "path": str(output_path)}

    # ── describe ──────────────────────────────────────────────────────────────

    async def describe(
        self,
        resource: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """
        Return file metadata and inferred schema.

        resource=None: list supported formats and base path info.
        resource=path: return size, modification time, and format for that file.
        """
        base = await super().describe(resource, **opts)
        caps = [str(c) for c in _CAPABILITIES]

        if resource is None:
            base_dir = self._base_path or Path.cwd()
            available: list[str] = []
            if base_dir.exists():
                available = [
                    str(p)
                    for p in sorted(base_dir.rglob("*"))
                    if p.is_file() and p.suffix.lower() in _SUPPORTED_FORMATS
                ][:50]  # cap listing at 50 files
            return {
                **base,
                "capabilities": caps,
                "supported_formats": sorted(_SUPPORTED_FORMATS),
                "base_path": str(base_dir),
                "available_files": available,
            }

        candidate = Path(resource).expanduser()
        if self._base_path and not candidate.is_absolute() and not resource.startswith("~"):
            candidate = self._base_path / resource

        if not candidate.exists():
            return {**base, "capabilities": caps, "exists": False}

        stat = candidate.stat()
        return {
            **base,
            "capabilities": caps,
            "exists": True,
            "size": stat.st_size,
            "modified_at": datetime.fromtimestamp(
                stat.st_mtime, tz=UTC
            ).isoformat(),
            "format": candidate.suffix.lower().lstrip("."),
            "supported": candidate.suffix.lower() in _SUPPORTED_FORMATS,
        }
