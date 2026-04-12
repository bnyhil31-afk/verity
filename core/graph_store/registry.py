"""
verity.core.graph_store.registry
=================================
Backend factory. Returns the configured GraphStore implementation.

Backend is selected via the VERITY_GRAPH_BACKEND environment variable.
Default: rdflib (personal tier, pure Python, no external services).

    VERITY_GRAPH_BACKEND=rdflib    — Personal tier (default)
    VERITY_GRAPH_BACKEND=pgvector  — Team tier (requires verity[team])
    VERITY_GRAPH_BACKEND=jena      — Enterprise tier (requires verity[enterprise])

Storage path for the rdflib backend:
    VERITY_GRAPH_PATH=/path/to/store  — On-disk persistence
    (unset)                           — In-memory (ephemeral)

The engine calls get_graph_store() once at startup. Nothing else in the
engine knows or cares which backend is running beneath the Protocol.
That is the Machine Test boundary.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path

from verity.core.exceptions import BackendNotAvailableError
from verity.core.graph_store import GraphStore
from verity.core.types import DEFAULT_DECAY_PARAMETERS, DecayParameters

logger = logging.getLogger(__name__)

# Environment variable names
_ENV_BACKEND = "VERITY_GRAPH_BACKEND"
_ENV_PATH    = "VERITY_GRAPH_PATH"

# Valid backend identifiers
_BACKEND_RDFLIB   = "rdflib"
_BACKEND_PGVECTOR = "pgvector"
_BACKEND_JENA     = "jena"
_VALID_BACKENDS   = {_BACKEND_RDFLIB, _BACKEND_PGVECTOR, _BACKEND_JENA}


def get_graph_store(
    decay_parameters: DecayParameters = DEFAULT_DECAY_PARAMETERS,
) -> GraphStore:
    """
    Return the configured GraphStore backend.

    Reads VERITY_GRAPH_BACKEND (default: rdflib).
    Falls back to rdflib with a warning if an unknown backend is specified.
    Raises BackendNotAvailableError if a valid but uninstalled backend is requested.

    Args:
        decay_parameters: Decay constants to pass to the backend.
                          Uses engine defaults if not provided.

    Returns:
        An uninitialized GraphStore. Call await store.initialize() before use.
        Engine.start() handles this automatically.
    """
    backend = os.getenv(_ENV_BACKEND, _BACKEND_RDFLIB).lower().strip()
    path    = os.getenv(_ENV_PATH)

    if backend not in _VALID_BACKENDS:
        warnings.warn(
            f"Unknown VERITY_GRAPH_BACKEND='{backend}'. "
            f"Valid options: {sorted(_VALID_BACKENDS)}. "
            f"Falling back to rdflib.",
            stacklevel=2,
        )
        backend = _BACKEND_RDFLIB

    if backend == _BACKEND_RDFLIB:
        return _get_rdflib(path, decay_parameters)

    if backend == _BACKEND_PGVECTOR:
        return _get_pgvector(decay_parameters)

    if backend == _BACKEND_JENA:
        return _get_jena(decay_parameters)

    # Should never reach here — caught above
    return _get_rdflib(path, decay_parameters)


def _get_rdflib(
    path: str | None,
    decay_parameters: DecayParameters,
) -> GraphStore:
    """Return an RDFLibStore instance."""
    from verity.core.graph_store.rdflib_store import RDFLibStore

    store_path = Path(path) if path else None

    logger.info(
        f"Graph backend: rdflib | "
        f"path={store_path or 'in-memory'}"
    )
    return RDFLibStore(path=store_path, decay_parameters=decay_parameters)


def _get_pgvector(decay_parameters: DecayParameters) -> GraphStore:
    """
    Return a pgvector-backed store (team tier).
    Not yet implemented — raises BackendNotAvailableError with install hint.
    """
    try:
        from verity.backends.pgvector_store import PgVectorStore  # noqa: F401
    except ImportError:
        raise BackendNotAvailableError(
            backend="pgvector",
            install_hint=(
                "Install the team tier extras: pip install verity[team]\n"
                "Also requires a running PostgreSQL instance with pgvector extension."
            ),
        )
    # Unreachable until PgVectorStore is implemented
    raise BackendNotAvailableError("pgvector", "Team tier backend not yet implemented.")


def _get_jena(decay_parameters: DecayParameters) -> GraphStore:
    """
    Return a Jena Fuseki-backed store (enterprise tier).
    Not yet implemented — raises BackendNotAvailableError with install hint.
    """
    try:
        from verity.backends.jena_store import JenaStore  # noqa: F401
    except ImportError:
        raise BackendNotAvailableError(
            backend="jena",
            install_hint=(
                "Install the enterprise tier extras: pip install verity[enterprise]\n"
                "Also requires a running Apache Jena Fuseki instance.\n"
                "Set VERITY_JENA_ENDPOINT=http://localhost:3030/verity"
            ),
        )
    # Unreachable until JenaStore is implemented
    raise BackendNotAvailableError("jena", "Enterprise tier backend not yet implemented.")
