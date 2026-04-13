"""
tests/test_profiles.py
======================
Tests for the profile system — EngineProfile, get_profile, and Engine integration.

Structure:
  TestEngineProfile       — dataclass invariants and field validation
  TestGetProfile          — factory function happy paths and error cases
  TestProfileAwareEngine  — Engine stores and exposes profile correctly
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from verity.core.engine import Engine
from verity.core.graph_store.rdflib_store import RDFLibStore
from verity.core.principles import LoadedPrinciples
from verity.core.profiles import (
    DEVELOPER,
    ENTERPRISE,
    PERSONAL,
    PROFESSIONAL,
    EngineProfile,
    get_profile,
)
from verity.core.types import (
    DEFAULT_DECAY_PARAMETERS,
    AuditEvent,
    AuditEventType,
    DecayParameters,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

_ALL_PROFILES = [PERSONAL, DEVELOPER, PROFESSIONAL, ENTERPRISE]


def _mock_principles() -> LoadedPrinciples:
    return LoadedPrinciples(
        version=1,
        sequence="0000000001",
        timestamp="2025-01-01T00:00:00Z",
        immutable=(),
        regulated=(),
        operational=(),
        canary_tests=(),
        content_hash="test_hash",
    )


async def _make_engine(profile: EngineProfile | None = None) -> Engine:
    """
    Build a started Engine with an in-memory store and a given profile.
    Bypasses principles verification — mirrors the started_engine fixture
    pattern from conftest.py, with an additional profile parameter.
    """
    store = RDFLibStore(path=None, decay_parameters=DEFAULT_DECAY_PARAMETERS)
    await store.initialize()

    engine = Engine(
        store=store,
        principles=_mock_principles(),
        decay_parameters=DEFAULT_DECAY_PARAMETERS,
        profile=profile,
    )
    engine._started = True

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

    return engine


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestEngineProfile:
    """Invariants on the EngineProfile dataclass and the four constants."""

    def test_all_profiles_have_required_fields(self):
        required = {
            "name", "decay_parameters", "bfs_max_depth",
            "checkpoint_timeout_seconds", "checkpoint_interactive",
            "auto_sign_principles", "graph_store_backend", "description",
        }
        for profile in _ALL_PROFILES:
            for field in required:
                assert hasattr(profile, field), (
                    f"Profile '{profile.name}' missing field '{field}'"
                )

    def test_personal_has_auto_sign_principles_true(self):
        assert PERSONAL.auto_sign_principles is True

    def test_enterprise_has_auto_sign_principles_false(self):
        assert ENTERPRISE.auto_sign_principles is False

    def test_developer_has_auto_sign_principles_false(self):
        assert DEVELOPER.auto_sign_principles is False

    def test_professional_has_auto_sign_principles_false(self):
        assert PROFESSIONAL.auto_sign_principles is False

    def test_decay_parameters_valid_on_all_profiles(self):
        for profile in _ALL_PROFILES:
            dp = profile.decay_parameters
            assert isinstance(dp, DecayParameters), (
                f"Profile '{profile.name}' decay_parameters is not a DecayParameters"
            )
            assert 0.0 < dp.exponent <= 5.0
            assert dp.sensitive_multiplier >= 1.0
            assert dp.spacing_cap >= 1.0
            assert 0.0 < dp.prune_threshold < 1.0

    def test_enterprise_has_custom_decay_parameters(self):
        # ENTERPRISE uses a stricter exponent than the default
        assert ENTERPRISE.decay_parameters.exponent == 0.4
        assert ENTERPRISE.decay_parameters.sensitive_multiplier == 1.6

    def test_personal_developer_professional_use_default_decay(self):
        for profile in [PERSONAL, DEVELOPER, PROFESSIONAL]:
            assert profile.decay_parameters == DEFAULT_DECAY_PARAMETERS, (
                f"Profile '{profile.name}' should use DEFAULT_DECAY_PARAMETERS"
            )

    def test_profiles_are_frozen_cannot_mutate_name(self):
        for profile in _ALL_PROFILES:
            with pytest.raises((AttributeError, TypeError)):
                profile.name = "tampered"  # type: ignore[misc]

    def test_profiles_are_frozen_cannot_mutate_backend(self):
        for profile in _ALL_PROFILES:
            with pytest.raises((AttributeError, TypeError)):
                profile.graph_store_backend = "sqlite"  # type: ignore[misc]

    def test_enterprise_bfs_max_depth_is_deepest(self):
        assert ENTERPRISE.bfs_max_depth == 4
        assert PERSONAL.bfs_max_depth == 2

    def test_enterprise_checkpoint_timeout_is_longest(self):
        assert ENTERPRISE.checkpoint_timeout_seconds == 86400  # 24 hours
        assert PERSONAL.checkpoint_timeout_seconds == 300

    def test_professional_enterprise_checkpoint_not_interactive(self):
        assert PROFESSIONAL.checkpoint_interactive is False
        assert ENTERPRISE.checkpoint_interactive is False

    def test_personal_developer_checkpoint_interactive(self):
        assert PERSONAL.checkpoint_interactive is True
        assert DEVELOPER.checkpoint_interactive is True


class TestGetProfile:
    """Factory function happy paths and error handling."""

    def test_get_profile_personal_returns_personal(self):
        assert get_profile("personal") is PERSONAL

    def test_get_profile_developer_returns_developer(self):
        assert get_profile("developer") is DEVELOPER

    def test_get_profile_professional_returns_professional(self):
        assert get_profile("professional") is PROFESSIONAL

    def test_get_profile_enterprise_returns_enterprise(self):
        assert get_profile("enterprise") is ENTERPRISE

    def test_get_profile_unknown_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            get_profile("nonexistent")

    def test_get_profile_unknown_error_lists_valid_names(self):
        with pytest.raises(ValueError) as exc_info:
            get_profile("unicorn")
        msg = str(exc_info.value)
        for name in ["personal", "developer", "professional", "enterprise"]:
            assert name in msg

    def test_get_profile_returns_same_instance(self):
        # Profiles are singletons — same object every call
        assert get_profile("personal") is get_profile("personal")
        assert get_profile("enterprise") is get_profile("enterprise")


class TestProfileAwareEngine:
    """
    Engine stores and exposes EngineProfile correctly.

    Most tests build the engine directly (bypassing principles verification)
    to isolate profile behavior from startup concerns.
    The Engine.start() tests mock verify_principles and get_graph_store.
    """

    async def test_engine_profile_is_none_by_default(self):
        engine = await _make_engine(profile=None)
        assert engine._profile is None
        await engine.stop()

    async def test_engine_profile_is_set_when_passed_personal(self):
        engine = await _make_engine(profile=PERSONAL)
        assert engine._profile is PERSONAL
        await engine.stop()

    async def test_engine_profile_is_set_when_passed_developer(self):
        engine = await _make_engine(profile=DEVELOPER)
        assert engine._profile is DEVELOPER
        await engine.stop()

    async def test_engine_profile_is_set_when_passed_enterprise(self):
        engine = await _make_engine(profile=ENTERPRISE)
        assert engine._profile is ENTERPRISE
        await engine.stop()

    async def test_stats_includes_profile_name_personal(self):
        engine = await _make_engine(profile=PERSONAL)
        stats = await engine.stats()
        assert stats["profile"] == "personal"
        await engine.stop()

    async def test_stats_includes_profile_name_enterprise(self):
        engine = await _make_engine(profile=ENTERPRISE)
        stats = await engine.stats()
        assert stats["profile"] == "enterprise"
        await engine.stop()

    async def test_stats_profile_is_none_when_no_profile(self):
        engine = await _make_engine(profile=None)
        stats = await engine.stats()
        assert stats["profile"] is None
        await engine.stop()

    async def test_engine_start_string_profile_resolves_to_personal(self, monkeypatch):
        """Engine.start(profile="personal") sets engine._profile to PERSONAL."""
        monkeypatch.delenv("VERITY_GRAPH_BACKEND", raising=False)
        fresh_store = RDFLibStore(path=None, decay_parameters=DEFAULT_DECAY_PARAMETERS)

        with patch("verity.core.engine.verify_principles", return_value=_mock_principles()), \
             patch("verity.core.engine.get_graph_store", return_value=fresh_store), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERITY_GRAPH_BACKEND", None)
            engine = await Engine.start(profile="personal")

        assert engine._profile is PERSONAL
        await engine.stop()

    async def test_engine_start_engineprofile_instance_accepted(self, monkeypatch):
        """Engine.start(profile=DEVELOPER) accepts an EngineProfile directly."""
        monkeypatch.delenv("VERITY_GRAPH_BACKEND", raising=False)
        fresh_store = RDFLibStore(path=None, decay_parameters=DEFAULT_DECAY_PARAMETERS)

        with patch("verity.core.engine.verify_principles", return_value=_mock_principles()), \
             patch("verity.core.engine.get_graph_store", return_value=fresh_store), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERITY_GRAPH_BACKEND", None)
            engine = await Engine.start(profile=DEVELOPER)

        assert engine._profile is DEVELOPER
        await engine.stop()

    async def test_explicit_decay_parameters_overrides_profile(self, monkeypatch):
        """Explicit decay_parameters argument takes precedence over profile's defaults."""
        monkeypatch.delenv("VERITY_GRAPH_BACKEND", raising=False)
        custom_decay = DecayParameters(exponent=0.3, sensitive_multiplier=1.2)
        fresh_store = RDFLibStore(path=None, decay_parameters=custom_decay)

        with patch("verity.core.engine.verify_principles", return_value=_mock_principles()), \
             patch("verity.core.engine.get_graph_store", return_value=fresh_store), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERITY_GRAPH_BACKEND", None)
            engine = await Engine.start(
                profile="enterprise",
                decay_parameters=custom_decay,
            )

        # Profile is still ENTERPRISE, but decay is the explicit override
        assert engine._profile is ENTERPRISE
        assert engine._decay == custom_decay
        assert engine._decay != ENTERPRISE.decay_parameters
        await engine.stop()

    async def test_engine_start_stats_includes_profile_name(self, monkeypatch):
        """engine.stats() includes profile name after Engine.start()."""
        monkeypatch.delenv("VERITY_GRAPH_BACKEND", raising=False)
        fresh_store = RDFLibStore(path=None, decay_parameters=DEFAULT_DECAY_PARAMETERS)

        with patch("verity.core.engine.verify_principles", return_value=_mock_principles()), \
             patch("verity.core.engine.get_graph_store", return_value=fresh_store), \
             patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VERITY_GRAPH_BACKEND", None)
            engine = await Engine.start(profile="developer")

        stats = await engine.stats()
        assert stats["profile"] == "developer"
        await engine.stop()

    async def test_engine_start_unknown_profile_raises_value_error(self, monkeypatch):
        """Engine.start(profile='bogus') raises ValueError before doing anything."""
        with pytest.raises(ValueError, match="Unknown profile"):
            await Engine.start(profile="bogus")
