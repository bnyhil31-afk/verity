"""
tests/test_crisis.py
====================
Tests for verity.core.crisis — the absolute barrier.

These are the most important behavioral tests in the suite.
They are also run as canary tests at every engine boot.
A failure here means the system does not behave as declared.

Design rule: false positives are acceptable.
             False negatives are not.
"""

from __future__ import annotations

import pytest

from verity.core.crisis import (
    DEFAULT_CRISIS_RESOURCES,
    CrisisResource,
    check_and_raise,
    get_crisis_resources,
    is_crisis_input,
    register_pattern,
)
from verity.core.exceptions import CrisisBarrierError

# ── is_crisis_input ───────────────────────────────────────────────────────────

class TestIsCrisisInput:
    """
    Verify the crisis detection patterns fire on known crisis inputs
    and do not fire on normal inputs.
    """

    # Direct crisis inputs — must all return True
    @pytest.mark.parametrize("text", [
        "I want to end my life",
        "I want to take my life",
        "I want to kill myself",
        "I'm thinking about suicide",
        "I feel suicidal",
        "I want to end it all",
        "I want to end it now",
        "I have no reason to live",
        "I'm better off dead",
        "I can't go on",
        "I can't keep going",
        "I wish I was never born",
        "I don't want to wake up",
        "I don't want to be alive",
        "I've been hurting myself",
        "I've been self-harming",
        "I've been cutting myself",
        "I've been thinking about not being here anymore",
    ])
    def test_crisis_detected(self, text: str):
        assert is_crisis_input(text) is True, (
            f"Crisis barrier failed to fire for: '{text}'"
        )

    # Normal inputs — must all return False
    @pytest.mark.parametrize("text", [
        "The patient reported fatigue and low mood",
        "Schedule a follow-up appointment",
        "Review the quarterly financial report",
        "Update the patient medication record",
        "The project deadline is next Friday",
        "I need to end this meeting early",
        "Let's end the call",
        "Can you help me finish this report?",
        "I want to live closer to work",
        "My life has been busy lately",
    ])
    def test_normal_input_not_flagged(self, text: str):
        assert is_crisis_input(text) is False, (
            f"Crisis barrier incorrectly fired for: '{text}'"
        )

    def test_empty_string(self):
        assert is_crisis_input("") is False

    def test_whitespace_only(self):
        assert is_crisis_input("   ") is False

    def test_none_like_empty(self):
        assert is_crisis_input("") is False

    def test_case_insensitive(self):
        assert is_crisis_input("I WANT TO KILL MYSELF") is True
        assert is_crisis_input("i want to kill myself") is True
        assert is_crisis_input("I Want To Kill Myself") is True

    def test_embedded_in_longer_text(self):
        """Crisis signal embedded in a longer message still fires."""
        text = (
            "I've been feeling really down lately and honestly "
            "I want to end my life. I don't know what to do."
        )
        assert is_crisis_input(text) is True


# ── check_and_raise ───────────────────────────────────────────────────────────

class TestCheckAndRaise:

    def test_raises_on_crisis_input(self):
        with pytest.raises(CrisisBarrierError):
            check_and_raise(
                text="I want to kill myself",
                actor="test",
                session_id="session:test123",
            )

    def test_does_not_raise_on_normal_input(self):
        """Should return normally — no exception."""
        check_and_raise(
            text="Schedule a follow-up appointment",
            actor="test",
            session_id=None,
        )

    def test_crisis_barrier_error_message(self):
        """CrisisBarrierError must contain routing instruction."""
        with pytest.raises(CrisisBarrierError) as exc_info:
            check_and_raise(text="I want to end my life", actor="test")
        assert "crisis" in str(exc_info.value).lower()
        assert "blocked" in str(exc_info.value).lower()

    def test_session_id_none_accepted(self):
        """session_id is optional."""
        check_and_raise(
            text="Normal clinical note",
            actor="clinician",
            session_id=None,
        )


# ── register_pattern ──────────────────────────────────────────────────────────

class TestRegisterPattern:

    def test_custom_pattern_fires(self):
        """Domain modules can register additional patterns."""
        register_pattern(r"\btest_crisis_signal_xyz\b")
        assert is_crisis_input("this contains test_crisis_signal_xyz phrase") is True

    def test_invalid_pattern_raises(self):
        """Invalid regex must raise immediately, not at match time."""
        with pytest.raises(Exception):
            register_pattern("[invalid regex (")

    def test_custom_pattern_case_insensitive(self):
        register_pattern(r"\bcustom_crisis_abc\b")
        assert is_crisis_input("CUSTOM_CRISIS_ABC") is True


# ── Crisis resources ──────────────────────────────────────────────────────────

class TestCrisisResources:

    def test_default_resources_present(self):
        assert len(DEFAULT_CRISIS_RESOURCES) >= 2

    def test_988_lifeline_included(self):
        names = [r.name for r in DEFAULT_CRISIS_RESOURCES]
        assert any("988" in name or "Lifeline" in name for name in names)

    def test_get_crisis_resources_returns_defaults(self):
        resources = get_crisis_resources()
        assert len(resources) >= len(DEFAULT_CRISIS_RESOURCES)

    def test_additional_resources_prepended(self):
        """Domain-specific resources appear first."""
        domain_resource = CrisisResource(
            name="Clinical Crisis Line",
            contact="1-800-000-0000",
            available="24/7",
        )
        resources = get_crisis_resources(additional=(domain_resource,))
        assert resources[0].name == "Clinical Crisis Line"

    def test_all_resources_have_contact(self):
        for resource in DEFAULT_CRISIS_RESOURCES:
            assert resource.contact, f"Resource '{resource.name}' has no contact"

    def test_all_resources_have_availability(self):
        for resource in DEFAULT_CRISIS_RESOURCES:
            assert resource.available, f"Resource '{resource.name}' has no availability"


# ── Canary verification ───────────────────────────────────────────────────────

class TestCanaryBehaviors:
    """
    These tests mirror the canary tests in principles.yaml exactly.
    If these fail, the engine boot will also fail.
    Run these first when debugging canary failures.
    """

    def test_canary_crisis_barrier_fires(self):
        """principles.yaml: canary crisis_barrier_fires"""
        assert is_crisis_input("I want to end my life") is True

    def test_canary_crisis_barrier_indirect(self):
        """principles.yaml: canary crisis_barrier_fires_indirect"""
        assert is_crisis_input("I've been thinking about not being here anymore") is True

    def test_canary_crisis_does_not_write_graph(self):
        """
        principles.yaml: crisis must_not graph_write
        Verified structurally: check_and_raise raises before any graph call.
        The engine's relate() only calls write_fact() after check_and_raise()
        returns normally. If it raises, write_fact() is never called.
        """
        raised = False
        try:
            check_and_raise(text="I want to end my life", actor="test")
        except CrisisBarrierError:
            raised = True
        assert raised, "CrisisBarrierError must be raised for crisis input"
