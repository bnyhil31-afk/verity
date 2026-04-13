"""
verity.core.connectors.registry
================================
ConnectorRegistry — register, discover, and route connectors by source_id.

Engine-scoped: one registry per engine instance, never a global singleton.
The engine passes the registry into sessions; sessions route ingest_from()
calls through it.

Discovery via setuptools entry points:

    [project.entry-points."verity.connectors"]
    my_source = "mypackage:MyConnector"

The registered class is instantiated with no arguments, then registered
under its source_id attribute (or the entry-point name as a fallback).
"""

from __future__ import annotations

import logging
from typing import Any

from verity.core.connectors import Connector

logger = logging.getLogger(__name__)


class ConnectorRegistry:
    """
    Engine-scoped connector registry.

    register()  — add a connector under a source_id
    get()       — retrieve a connector by source_id (KeyError if missing)
    list()      — async: returns describe() output for every registered connector
    discover()  — load connectors from "verity.connectors" entry points
    __len__     — number of registered connectors
    __contains__ — membership test by source_id
    """

    def __init__(self) -> None:
        self._connectors: dict[str, Connector] = {}

    # ── Core operations ───────────────────────────────────────────────────────

    def register(self, connector_id: str, connector: Connector) -> None:
        """
        Register connector under connector_id.

        Raises TypeError if connector does not satisfy the Connector Protocol
        (i.e. is missing read, write, or describe methods).
        """
        if not isinstance(connector, Connector):
            raise TypeError(
                f"Cannot register '{connector_id}': object of type "
                f"'{type(connector).__name__}' does not satisfy the Connector "
                "Protocol. Implement read(), write(), and describe()."
            )
        self._connectors[connector_id] = connector
        logger.debug(
            "ConnectorRegistry: registered '%s' (%s).",
            connector_id,
            type(connector).__name__,
        )

    def get(self, connector_id: str) -> Connector:
        """
        Return the connector for connector_id.

        Raises KeyError with a helpful message listing available IDs if not found.
        """
        try:
            return self._connectors[connector_id]
        except KeyError:
            available = list(self._connectors)
            raise KeyError(
                f"Connector '{connector_id}' not registered. "
                f"Registered connectors: {available}"
            ) from None

    async def list(self) -> dict[str, dict[str, Any]]:
        """
        Return describe() output for every registered connector.

        Keys are the connector_ids passed to register().
        Values are the dicts returned by each connector's describe() method.
        If a connector's describe() raises, the error is captured in the result.
        """
        result: dict[str, dict[str, Any]] = {}
        for cid, connector in self._connectors.items():
            try:
                info = await connector.describe()
                result[cid] = info
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ConnectorRegistry.list(): describe() failed for '%s': %s",
                    cid,
                    exc,
                )
                result[cid] = {"connector_id": cid, "error": str(exc)}
        return result

    # ── Discovery ─────────────────────────────────────────────────────────────

    def discover(self) -> None:
        """
        Auto-discover and register connectors via setuptools entry points.

        Searches the "verity.connectors" entry-point group. Each entry point
        should point to a class (or callable) that returns a Connector instance
        when called with no arguments.

        Logs what was found (info) or not found (debug). Failures for individual
        entry points are logged as warnings and do not abort discovery.
        """
        try:
            from importlib.metadata import entry_points
        except ImportError:
            logger.warning(
                "ConnectorRegistry.discover(): importlib.metadata not available. "
                "Skipping entry-point discovery."
            )
            return

        eps = entry_points(group="verity.connectors")
        found = 0
        for ep in eps:
            try:
                connector_class = ep.load()
                connector = connector_class()
                connector_id = getattr(connector, "source_id", ep.name)
                self.register(connector_id, connector)
                logger.info(
                    "ConnectorRegistry.discover(): loaded '%s' from entry point '%s'.",
                    connector_id,
                    ep.name,
                )
                found += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ConnectorRegistry.discover(): failed to load '%s': %s",
                    ep.name,
                    exc,
                )

        if found == 0:
            logger.debug(
                "ConnectorRegistry.discover(): no connectors found in "
                "'verity.connectors' entry-point group."
            )

    # ── Convenience ───────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._connectors)

    def __contains__(self, connector_id: object) -> bool:
        return connector_id in self._connectors

    def __repr__(self) -> str:
        ids = list(self._connectors)
        return f"ConnectorRegistry({ids})"
