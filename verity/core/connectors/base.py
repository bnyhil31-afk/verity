"""
verity.core.connectors.base
===========================
BaseConnector — lifecycle management, logging, credential storage.

Subclasses override only what they need:
  - _connect() / _disconnect() for connection lifecycle
  - read()     — MUST override (raises NotImplementedError by default)
  - write()    — optional (raises NotImplementedError; most connectors are read-only)
  - describe() — optional (returns basic metadata by default)

Usage:
    class MyConnector(BaseConnector):
        async def read(self, resource, query=None, **opts):
            async for item in _fetch(resource):
                yield ConnectorRecord(
                    id=item["id"],
                    content=item["data"],
                    source_id=self.source_id,
                    resource=resource,
                )

    async with MyConnector(source_id="my_source") as conn:
        async for record in conn.read("my_resource"):
            process(record)
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from verity.core.connectors import ConnectorRecord

logger = logging.getLogger(__name__)


class BaseConnector:
    """
    Base class for Verity connectors.

    Provides:
      - Async context manager lifecycle (__aenter__ / __aexit__)
      - Credential storage (self._credentials)
      - Consistent logging via _log_read()
      - Default write() that raises NotImplementedError (override when needed)
      - Default describe() returning basic source metadata (override for richer info)

    Subclasses must implement read() as an async generator.
    """

    def __init__(
        self,
        source_id: str,
        credentials: dict | None = None,
    ) -> None:
        self.source_id = source_id
        self._credentials: dict[str, Any] = credentials or {}
        self._connected: bool = False
        self._logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> BaseConnector:
        await self._connect()
        self._connected = True
        self._logger.debug("Connector '%s' connected.", self.source_id)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self._disconnect()
        self._connected = False
        self._logger.debug("Connector '%s' disconnected.", self.source_id)

    # ── Lifecycle hooks — override in subclasses ──────────────────────────────

    async def _connect(self) -> None:
        """Called on __aenter__. Override to establish connections."""

    async def _disconnect(self) -> None:
        """Called on __aexit__. Override to release connections."""

    # ── Protocol methods ──────────────────────────────────────────────────────

    async def read(  # type: ignore[return]
        self,
        resource: str,
        query: dict | None = None,
        **opts: Any,
    ) -> AsyncIterator[ConnectorRecord]:
        """
        Override in subclasses as an async generator.

        Raises NotImplementedError on first iteration if not overridden.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.read() must be overridden as an async generator."
        )
        yield  # type: ignore[misc]  # unreachable — marks this as an async generator

    async def write(
        self,
        resource: str,
        data: AsyncIterator[ConnectorRecord] | list[ConnectorRecord],
        **opts: Any,
    ) -> dict[str, Any]:
        """
        Read-only by default. Override to enable write operations.

        Raises NotImplementedError immediately when called.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} is read-only. Override write() to enable writes."
        )

    async def describe(
        self,
        resource: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """
        Returns basic source metadata.
        Override to provide richer schema information.
        """
        return {
            "source_id": self.source_id,
            "connector_type": self.__class__.__name__,
            "connected": self._connected,
            "resource": resource,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_read(self, resource: str, count: int) -> None:
        """
        Consistent log line after a read() operation completes.
        Call at the end of read() with the total count of yielded records.
        """
        self._logger.info(
            "Connector '%s' read %d record(s) from '%s'.",
            self.source_id,
            count,
            resource,
        )
