"""
verity.core.exceptions
======================
All custom exceptions for the Verity engine.

Import from here, not from submodules:
    from verity.core.exceptions import ConsentRequiredError

Design discipline:
  - Every exception carries enough context to log meaningfully
  - No bare RuntimeError or ValueError anywhere in the engine
  - Hierarchy is shallow — one base class per domain, not a taxonomy
"""

from __future__ import annotations

# ── Base ──────────────────────────────────────────────────────────────────────


class VerityError(Exception):
    """Base class for all Verity exceptions."""


# ── Principles ────────────────────────────────────────────────────────────────


class PrinciplesError(VerityError):
    """Principles verification failed. The engine must not start."""


class SignatureError(PrinciplesError):
    """Ed25519 signature is missing, invalid, or does not match content."""


class CanaryError(PrinciplesError):
    """A behavioral canary test failed. The system does not behave as declared."""


# ── Consent ───────────────────────────────────────────────────────────────────


class ConsentRequiredError(VerityError):
    """
    A graph operation was attempted without a valid consent record.

    Raised by the consent gate before any traversal begins.
    The operation is blocked — no partial execution occurs.
    """

    def __init__(
        self,
        operation: str,
        consent_ref: str | None = None,
        purpose: str | None = None,
    ) -> None:
        self.operation = operation
        self.consent_ref = consent_ref
        self.purpose = purpose
        super().__init__(
            f"Consent required for operation '{operation}'. "
            + (f"consent_ref='{consent_ref}' " if consent_ref else "No consent_ref provided. ")
            + (f"purpose='{purpose}'" if purpose else "")
        )


class ConsentExpiredError(ConsentRequiredError):
    """The consent record exists but has expired."""


class ConsentRevokedError(ConsentRequiredError):
    """The consent record exists but has been revoked."""


class PurposeMismatchError(ConsentRequiredError):
    """
    The declared purpose does not match the active consent record.
    Consent for 'clinical_decision_support' does not authorize 'research'.
    """

    def __init__(
        self,
        requested_purpose: str,
        consented_purpose: str,
        consent_ref: str,
    ) -> None:
        self.requested_purpose = requested_purpose
        self.consented_purpose = consented_purpose
        super().__init__(
            operation="context_query",
            consent_ref=consent_ref,
            purpose=requested_purpose,
        )
        self.args = (
            f"Purpose mismatch: requested '{requested_purpose}' "
            f"but consent '{consent_ref}' covers '{consented_purpose}' only.",
        )


# ── Crisis ────────────────────────────────────────────────────────────────────


class CrisisBarrierError(VerityError):
    """
    The crisis barrier fired. All graph writes are blocked.

    This is not an error in the traditional sense — it is the correct
    behavior of an absolute safety barrier. The caller should route
    to crisis resources immediately.

    Raised by RELATE when crisis content is detected.
    Never caught silently — always propagated to the caller.
    """

    def __init__(self, input_excerpt: str | None = None) -> None:
        self.input_excerpt = input_excerpt
        super().__init__(
            "Crisis content detected. Graph writes blocked. "
            "Route to crisis resources immediately."
        )


# ── Validation ────────────────────────────────────────────────────────────────


class ValidationError(VerityError):
    """SHACL validation failed during ingestion."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__(
            f"SHACL validation failed with {len(violations)} violation(s):\n"
            + "\n".join(f"  • {v}" for v in violations)
        )


class ClassificationError(VerityError):
    """A fact was ingested with an invalid or missing classification."""


# ── Graph store ───────────────────────────────────────────────────────────────


class GraphStoreError(VerityError):
    """The graph store backend encountered an error."""


class BackendNotAvailableError(GraphStoreError):
    """
    The requested backend is not installed or not reachable.
    Falls back to SQLite unless the caller explicitly prevents fallback.
    """

    def __init__(self, backend: str, install_hint: str | None = None) -> None:
        self.backend = backend
        msg = f"Backend '{backend}' is not available."
        if install_hint:
            msg += f" {install_hint}"
        super().__init__(msg)


# ── Module ────────────────────────────────────────────────────────────────────


class ModuleError(VerityError):
    """A domain module failed to load or initialize."""


class ModuleNotFoundError(ModuleError):
    """The requested domain module is not installed."""

    def __init__(self, module_id: str) -> None:
        self.module_id = module_id
        super().__init__(
            f"Domain module '{module_id}' is not installed. "
            f"Try: pip install {module_id.replace('_', '-')}"
        )


# ── Engine ────────────────────────────────────────────────────────────────────


class EngineNotStartedError(VerityError):
    """An engine operation was attempted before Engine.start() completed."""


class SessionClosedError(VerityError):
    """An operation was attempted on a session that has already been closed."""
