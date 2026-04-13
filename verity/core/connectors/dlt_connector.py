"""
verity.core.connectors.dlt_connector
=====================================
DltConnector — wraps any dlt (data load tool) source as a Verity connector.

dlt sources are Python generators that yield dicts — exactly what
ConnectorRecord expects. This adapter gives access to 60+ data sources
with ~50 lines of adapter code.

dlt is an optional dependency. If not installed, an ImportError with an
install hint is raised at construction time, not at import time.

Usage:
    import dlt
    from verity.core.connectors.dlt_connector import DltConnector

    @dlt.resource
    def my_source():
        yield {"id": "1", "content": "hello"}
        yield {"id": "2", "content": "world"}

    connector = DltConnector(
        source_id="my_source",
        source_factory=my_source,
    )
    async for record in connector.read("default"):
        print(record.content)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

from verity.core.connectors import ConnectorRecord
from verity.core.connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Import dlt at module level to detect availability for _DLT_AVAILABLE flag,
# but only raise at construction time (consistent with MCPConnector pattern).
try:
    import dlt as _dlt  # type: ignore[import]
    _DLT_AVAILABLE = True
except ImportError:
    _dlt = None  # type: ignore[assignment]
    _DLT_AVAILABLE = False


class DltConnector(BaseConnector):
    """
    Wraps any dlt source as a Verity connector.

    dlt sources are Python generators (or dlt resource/source objects) that
    yield dicts. This connector iterates them and converts each dict to a
    ConnectorRecord with the configured classification and trust_score.

    source_factory: any callable that returns a dlt source / resource, or
                    any iterable of dicts (for testing without dlt installed).
    source_kwargs: keyword arguments forwarded to source_factory on each read().
    classification: DataClassification value applied to all records.
    trust_score: defaults to 0.75 (TrustSource.INSTITUTIONAL) — dlt sources
                 are typically structured, verified data feeds.

    Lifecycle: DltConnector is usable directly (no async context manager
    required) since dlt sources manage their own connections internally.
    The async context manager from BaseConnector is still supported.
    """

    def __init__(
        self,
        source_id: str,
        source_factory: Callable[..., Any],
        source_kwargs: dict | None = None,
        classification: str = "internal",
        trust_score: float = 0.75,
    ) -> None:
        if not _DLT_AVAILABLE:
            raise ImportError(
                "dlt is required for DltConnector but is not installed. "
                "Install it with: pip install 'verity[connectors]'  "
                "or: pip install dlt"
            )
        super().__init__(source_id=source_id)
        self._source_factory = source_factory
        self._source_kwargs = source_kwargs or {}
        self.classification = classification
        self.trust_score = trust_score

    # ── read ──────────────────────────────────────────────────────────────────

    async def read(
        self,
        resource: str,
        query: dict | None = None,
        **opts: Any,
    ) -> AsyncIterator[ConnectorRecord]:
        """
        Yield ConnectorRecords from a dlt source.

        Calls source_factory(**source_kwargs) to get a dlt source, optionally
        filters to a specific sub-resource by name, then iterates records.

        resource: name of a specific dlt sub-resource within the source, or
                  "default" to iterate the source directly.
        query:    currently unused; reserved for future filter support.
        opts:     merged into source_kwargs for this call only.
        """
        effective_kwargs = {**self._source_kwargs, **opts}
        count = 0

        try:
            source = self._source_factory(**effective_kwargs)
        except Exception:
            logger.exception(
                "DltConnector '%s': failed to initialise source_factory '%s'. "
                "Yielding nothing.",
                self.source_id,
                getattr(self._source_factory, "__name__", repr(self._source_factory)),
            )
            return

        # Resolve the iterable to use: a specific named sub-resource or the
        # source itself. dlt source objects expose `.resources` as a dict.
        iterable = _resolve_resource(source, resource)

        try:
            for raw in iterable:
                if not isinstance(raw, dict):
                    # dlt can yield non-dict items from some sources; skip them
                    logger.debug(
                        "DltConnector '%s': skipping non-dict record: %r",
                        self.source_id,
                        type(raw),
                    )
                    continue

                record_id = str(
                    raw.get("id") or raw.get("_id") or uuid.uuid4()
                )
                # metadata: all fields except id/_id (already promoted to record.id)
                metadata = {
                    k: v for k, v in raw.items() if k not in ("id", "_id")
                }

                yield ConnectorRecord(
                    id=record_id,
                    content=raw,
                    source_id=self.source_id,
                    resource=resource or "default",
                    metadata=metadata,
                    classification=self.classification,
                    trust_score=self.trust_score,
                )
                count += 1

        except Exception:
            logger.exception(
                "DltConnector '%s': error during extraction from resource '%s'. "
                "Yielded %d record(s) before failure.",
                self.source_id,
                resource,
                count,
            )
            # Graceful degradation — partial results already yielded; stop here.
            return

        self._log_read(resource or "default", count)

    # ── describe ──────────────────────────────────────────────────────────────

    async def describe(
        self,
        resource: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """
        Return metadata about this dlt source.

        resource=None: lists source name and all available sub-resources.
        resource=name: describes that specific sub-resource (schema if available).
        """
        source_name = getattr(
            self._source_factory, "__name__", repr(self._source_factory)
        )

        # Attempt to introspect the dlt source object for resource names.
        available_resources: list[str] = []
        schema_info: dict[str, Any] = {}
        try:
            source_obj = self._source_factory(**self._source_kwargs)
            # dlt DltSource objects have a .resources attribute (dict[str, DltResource])
            if hasattr(source_obj, "resources"):
                available_resources = list(source_obj.resources.keys())
            # dlt DltSource objects may expose a .schema attribute
            if hasattr(source_obj, "schema") and source_obj.schema is not None:
                schema_info = {
                    "name": getattr(source_obj.schema, "name", None),
                    "version": getattr(source_obj.schema, "version", None),
                }
        except Exception:
            logger.debug(
                "DltConnector '%s': could not introspect source for describe().",
                self.source_id,
            )

        base = {
            "source_id": self.source_id,
            "connector_type": "DltConnector",
            "source_name": source_name,
            "classification": self.classification,
            "trust_score": self.trust_score,
            "available_resources": available_resources,
            "schema": schema_info,
        }

        if resource is not None:
            base["requested_resource"] = resource
            base["resource_found"] = resource in available_resources

        return base


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_resource(source: Any, resource: str) -> Any:
    """
    Given a dlt source/resource object and a resource name, return the
    iterable to use for extraction.

    - If source has .resources[resource], use that sub-resource.
    - If resource == "default" or no matching sub-resource, use source directly.
    - Falls back to using source directly if introspection fails.
    """
    if resource and resource != "default":
        resources = getattr(source, "resources", None)
        if isinstance(resources, dict) and resource in resources:
            return resources[resource]
    return source
