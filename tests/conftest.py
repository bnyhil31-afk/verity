"""
tests/conftest.py
=================
Pytest configuration for the Verity test suite.

Configures pytest-asyncio so all async test functions run correctly.
Provides shared fixtures available to all test modules.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio

# ── pytest-asyncio configuration ──────────────────────────────────────────────
# asyncio_mode = "auto" is set in pyproject.toml.
# This file exists to provide shared fixtures and any future hooks.


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def now() -> datetime:
    """Current UTC datetime. Use in tests that need a consistent timestamp."""
    return datetime.now(UTC)


@pytest_asyncio.fixture
async def fresh_store():
    """
    A freshly initialized in-memory RDFLibStore.
    Isolated per test — no state leaks between tests.
    """
    from verity.core.graph_store.rdflib_store import RDFLibStore
    from verity.core.types import DEFAULT_DECAY_PARAMETERS

    store = RDFLibStore(path=None, decay_parameters=DEFAULT_DECAY_PARAMETERS)
    await store.initialize()
    yield store
    await store.close()


@pytest_asyncio.fixture
async def started_engine():
    """
    A started Engine with an in-memory store and mock principles.
    Bypasses principles verification for unit testing.
    Isolated per test.
    """
    from verity.core.engine import Engine
    from verity.core.graph_store.rdflib_store import RDFLibStore
    from verity.core.principles import LoadedPrinciples
    from verity.core.types import (
        DEFAULT_DECAY_PARAMETERS,
        AuditEvent,
        AuditEventType,
    )

    store = RDFLibStore(path=None, decay_parameters=DEFAULT_DECAY_PARAMETERS)
    await store.initialize()

    principles = LoadedPrinciples(
        version=1,
        sequence="0000000001",
        timestamp="2025-01-01T00:00:00Z",
        immutable=(),
        regulated=(),
        operational=(),
        canary_tests=(),
        content_hash="test_hash",
    )

    engine = Engine(
        store=store,
        principles=principles,
        decay_parameters=DEFAULT_DECAY_PARAMETERS,
    )
    engine._started = True

    # Seed audit trail
    await store.append_audit(AuditEvent(
        sequence=0,
        event_type=AuditEventType.PRINCIPLES_VERIFIED,
        timestamp=datetime.now(UTC),
        actor="test_fixture",
        session_id=None,
        consent_ref=None,
        payload={"fixture": True},
        content_hash="",
        previous_hash=None,
        chain_valid=True,
    ))

    yield engine
    await engine.stop()


@pytest.fixture
def sample_consent():
    """A valid, active ConsentRecord for use in tests."""
    from verity.core.types import ConsentRecord, DataClassification

    return ConsentRecord(
        consent_ref="consent:test_fixture",
        subject_id="patient:test_subject",
        granted_by="test_clinician",
        granted_at=datetime.now(UTC),
        purpose="test_purpose",
        classifications=(
            DataClassification.PUBLIC,
            DataClassification.INTERNAL,
        ),
        audit_id=1,
    )
