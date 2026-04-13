"""
verity.core.connectors
======================
The Connector Protocol — universal data source interface.

Three methods. Zero coupling. Covers every data source.

Any class that implements read(), write(), and describe() with matching
signatures is a valid Verity connector. No inheritance required.
No imports from Verity required in connector implementations.

resource: a string address for the data within the source.
Examples: file path, table name, API endpoint, MQTT topic,
          calendar ID, email folder, FHIR resource type.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class ConnectorCapability(StrEnum):
    """
    Capabilities a connector may advertise via describe().

    Connectors declare their capabilities so callers know what to expect.
    READ is effectively required. WRITE is optional — most connectors are
    read-only. The rest are hints for callers to optimize access patterns.
    """

    READ      = "read"       # Can yield records via read()
    WRITE     = "write"      # Can accept records via write()
    STREAMING = "streaming"  # read() yields records lazily, not all at once
    BATCH     = "batch"      # Supports bulk operations efficiently
    SEARCH    = "search"     # Supports query dict for filtered retrieval


@dataclass
class ConnectorRecord:
    """
    Standard record shape every connector produces.

    This is the universal data container — the lug width that every
    connector outputs and every consumer accepts. It carries enough
    metadata that downstream code can classify, route, and audit
    each record without knowing the source.
    """

    id: str                           # Unique within source
    content: str | dict | bytes       # The actual data
    source_id: str                    # Which connector produced this
    resource: str                     # Which resource within the source
    metadata: dict[str, Any] = field(default_factory=dict)
    classification: str = "internal"  # DataClassification value
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )
    trust_score: float = 0.5


@runtime_checkable
class Connector(Protocol):
    """
    Universal data source interface. Three methods. Zero coupling.

    Implement these three methods and any class is a valid Verity connector.
    No inheritance required. No imports from Verity required.

    Protocol is runtime_checkable — isinstance(obj, Connector) works for
    testing and runtime validation. The check verifies that read, write,
    and describe are all present as attributes.

    resource: a string address for the data within the source.
    Examples: file path, table name, API endpoint, MQTT topic,
              calendar ID, email folder, FHIR resource type.
    """

    async def read(
        self,
        resource: str,
        query: dict | None = None,
        **opts: Any,
    ) -> AsyncIterator[ConnectorRecord]:
        """
        Yield records from the source. Streaming-native.

        Implementations should be async generators (yield each record as
        it arrives). Callers iterate with: async for record in conn.read(r)
        No await needed — the generator is returned directly.
        """
        ...

    async def write(
        self,
        resource: str,
        data: AsyncIterator[ConnectorRecord] | list[ConnectorRecord],
        **opts: Any,
    ) -> dict[str, Any]:
        """Write records to the source. Returns stats dict."""
        ...

    async def describe(
        self,
        resource: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """
        Describe available resources and their schemas.
        resource=None: list all available resources.
        resource=name: describe that specific resource.
        """
        ...


__all__ = [
    "ConnectorCapability",
    "ConnectorRecord",
    "Connector",
]
