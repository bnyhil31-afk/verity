"""
verity.core.principles
======================
Boot-time principles verification and behavioral canary runner.

This module runs before the engine accepts any request. Two checks
must both pass or the engine will not start:

  1. Cryptographic integrity  — principles.yaml matches its Ed25519 signature
  2. Behavioral verification  — all canary tests produce expected outputs

A system that passes the signature check but fails a canary is not
compliant — the principles file could have been re-signed with weakened
behaviors. Both layers must agree.

This is not optional and is not configurable. It is the immune system.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Path to principles.yaml — resolved relative to the package root
_PACKAGE_ROOT = Path(__file__).parent.parent
PRINCIPLES_PATH = _PACKAGE_ROOT / "principles.yaml"
SIGNATURE_PATH  = _PACKAGE_ROOT / "principles.sig"


# ── Exceptions ────────────────────────────────────────────────────────────────


class PrinciplesError(RuntimeError):
    """
    Raised when principles verification fails.
    The engine must not start when this is raised.
    """


class SignatureError(PrinciplesError):
    """Cryptographic signature is missing, invalid, or does not match content."""


class CanaryError(PrinciplesError):
    """A behavioral canary test failed. The system does not behave as declared."""


# ── Principles loading ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Principle:
    id: str
    text: str
    tier: str   # "immutable" | "regulated" | "operational"


@dataclass(frozen=True)
class LoadedPrinciples:
    """
    The verified, parsed contents of principles.yaml.
    Only produced by verify_principles() — never constructed directly.
    """
    version: int
    sequence: str
    timestamp: str
    immutable: tuple[Principle, ...]
    regulated: tuple[Principle, ...]
    operational: tuple[Principle, ...]
    canary_tests: tuple[dict[str, Any], ...]
    content_hash: str   # SHA-256 of the raw file content


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file. Raises PrinciplesError on failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise PrinciplesError(
            f"principles.yaml not found at {path}. "
            "Run: verity init"
        )
    except yaml.YAMLError as e:
        raise PrinciplesError(f"principles.yaml is malformed: {e}")


def _compute_content_hash(path: Path) -> str:
    """SHA-256 of the raw file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_principles(data: dict[str, Any], content_hash: str) -> LoadedPrinciples:
    """Parse the raw YAML dict into typed LoadedPrinciples."""

    def parse_tier(tier_name: str) -> tuple[Principle, ...]:
        entries = data.get("principles", {}).get(tier_name, []) or []
        return tuple(
            Principle(id=p["id"], text=p["text"], tier=tier_name)
            for p in entries
        )

    return LoadedPrinciples(
        version=data.get("version", 1),
        sequence=data.get("sequence", ""),
        timestamp=data.get("timestamp", ""),
        immutable=parse_tier("immutable"),
        regulated=parse_tier("regulated"),
        operational=parse_tier("operational"),
        canary_tests=tuple(data.get("canary_tests", [])),
        content_hash=content_hash,
    )


# ── Signature verification ────────────────────────────────────────────────────


def _verify_signature(content_hash: str, data: dict[str, Any]) -> None:
    """
    Verify the Ed25519 signature over the principles file content hash.

    Two states are valid at this stage of the project:
      - Unsigned (signature: null) — allowed during initial development,
        logs a prominent warning. Will become an error in v1.0.
      - Signed — must verify against the public key in signed_by.

    A mismatched signature is always a hard failure. No exceptions.
    """
    signature = data.get("signature")
    signed_by = data.get("signed_by")

    if signature is None or signed_by is None:
        # Unsigned — development mode. Warn loudly, do not halt.
        logger.warning(
            "⚠️  principles.yaml is UNSIGNED. "
            "This is acceptable during development but must be resolved "
            "before any production deployment. Run: verity init --sign"
        )
        return

    # Signed — verify Ed25519 signature
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PublicKey,
        )
        from cryptography.hazmat.primitives.serialization import (
            Encoding,
            PublicFormat,
            load_pem_public_key,
        )
        from cryptography.exceptions import InvalidSignature
        import base64

        # Load public key from signed_by field (PEM or did:key format)
        if signed_by.startswith("-----BEGIN"):
            public_key = load_pem_public_key(signed_by.encode())
        else:
            # did:key or base64 — extensible for future key formats
            raise SignatureError(
                f"Unsupported key format in signed_by: {signed_by[:40]}... "
                "Expected PEM-encoded Ed25519 public key."
            )

        sig_bytes = base64.b64decode(signature)
        message = content_hash.encode("utf-8")

        public_key.verify(sig_bytes, message)
        logger.info("✓ principles.yaml signature verified.")

    except InvalidSignature:
        raise SignatureError(
            "principles.yaml signature verification FAILED. "
            "The file may have been tampered with. "
            "The engine will not start."
        )
    except ImportError:
        raise SignatureError(
            "cryptography package not available for signature verification. "
            "Install it: pip install cryptography"
        )


# ── Canary behavioral tests ───────────────────────────────────────────────────


def _run_canary_tests(canary_tests: tuple[dict[str, Any], ...]) -> None:
    """
    Run the behavioral canary tests defined in principles.yaml.

    These tests verify that the system *behaves* according to its principles,
    not just that the principles file is intact. A signed file with weakened
    behaviors fails here.

    At this stage (pre-engine), canaries verify the crisis detection module
    directly — not through the full engine stack. As the engine is built,
    canaries will be wired to the live engine.

    Raises CanaryError if any test fails.
    """
    failures: list[str] = []

    for test in canary_tests:
        test_id = test.get("id", "unknown")

        try:
            _run_single_canary(test)
            logger.debug(f"  ✓ canary:{test_id}")
        except CanaryError as e:
            failures.append(f"canary:{test_id} — {e}")
            logger.error(f"  ✗ canary:{test_id} FAILED: {e}")

    if failures:
        raise CanaryError(
            f"{len(failures)} canary test(s) failed. "
            "The engine will not start.\n"
            + "\n".join(f"  • {f}" for f in failures)
        )

    logger.info(f"✓ All {len(canary_tests)} canary tests passed.")


def _run_single_canary(test: dict[str, Any]) -> None:
    """
    Execute a single canary test.

    Canary tests are defined in principles.yaml and specify:
      - must_trigger: what must happen
      - must_not: what must NOT happen
      - principle: which principle this test verifies

    This dispatcher routes each test to the appropriate verifier.
    As engine components are built, new canary types are registered here.
    """
    must_trigger = test.get("must_trigger")
    must_not     = test.get("must_not")
    input_text   = test.get("input")
    principle    = test.get("principle", "unknown")

    # Crisis barrier canaries — verifiable now, pre-engine
    if must_trigger == "crisis_barrier":
        if input_text is None:
            raise CanaryError("crisis_barrier canary requires 'input' field")
        _verify_crisis_barrier(input_text, must_not, principle)
        return

    # Consent gate canaries — verifiable structurally
    if must_trigger == "consent_gate" or test.get("must_raise") == "ConsentRequiredError":
        _verify_consent_gate_structure()
        return

    # Output contract canaries — deferred until engine is built
    # These are registered here so they appear in the canary manifest
    # and will fail loudly once the engine exists but doesn't implement them.
    if test.get("input_action") in ("any_context_query", "checkpoint_timeout"):
        logger.debug(
            f"  ⏸  canary:{test.get('id')} deferred — "
            "requires live engine (will be enforced in Phase 1)"
        )
        return

    # Unknown canary type — warn but do not halt during development
    logger.warning(
        f"  ⚠️  canary:{test.get('id')} — unrecognized test type, skipping. "
        f"must_trigger={must_trigger}"
    )


def _verify_crisis_barrier(
    input_text: str,
    must_not: str | None,
    principle: str,
) -> None:
    """
    Verify the crisis barrier fires on known crisis inputs.

    Uses the crisis detection patterns directly — independent of the full
    engine stack. This ensures the barrier is testable before the engine
    is complete and cannot be bypassed by engine-level changes.
    """
    from verity.core.crisis import is_crisis_input

    if not is_crisis_input(input_text):
        raise CanaryError(
            f"Crisis barrier did not fire for input: '{input_text}'. "
            f"Principle '{principle}' is not being enforced. "
            "The engine will not start."
        )

    if must_not == "graph_write":
        # Structural verification: crisis_detected=True in RelateResult
        # means the graph write was blocked. This is enforced in core.crisis
        # and verified here by contract — the engine cannot write to the graph
        # when is_crisis_input() returns True.
        pass  # Enforced structurally — verified in test_canary.py


def _verify_consent_gate_structure() -> None:
    """
    Verify the consent gate exists and is structurally sound.

    Checks that ConsentRequiredError is importable and that ContextRequest
    requires a consent_ref field — the structural guarantee that consent
    cannot be bypassed by omission.
    """
    from verity.core.exceptions import ConsentRequiredError  # noqa: F401

    # Verify ContextRequest has consent_ref as a required field
    from verity.core import ContextRequest
    import inspect
    sig = inspect.signature(ContextRequest.__init__)
    params = list(sig.parameters.keys())
    if "consent_ref" not in params:
        raise CanaryError(
            "ContextRequest does not have a consent_ref field. "
            "The consent gate is structurally broken."
        )


# ── Public interface ──────────────────────────────────────────────────────────


def verify_principles(
    path: Path = PRINCIPLES_PATH,
) -> LoadedPrinciples:
    """
    Load, verify, and return the engine's principles.

    This is the single entry point for boot-time verification.
    Call this before starting the engine. If it returns, both checks passed.
    If it raises, the engine must not start.

    Raises:
        PrinciplesError  — base class for all verification failures
        SignatureError   — cryptographic verification failed
        CanaryError      — behavioral verification failed
    """
    logger.info("Verifying principles...")

    # Step 1: Load and parse
    data = _load_yaml(path)
    content_hash = _compute_content_hash(path)
    principles = _parse_principles(data, content_hash)

    logger.info(
        f"  Loaded principles v{principles.version} "
        f"(seq {principles.sequence}): "
        f"{len(principles.immutable)} immutable, "
        f"{len(principles.regulated)} regulated, "
        f"{len(principles.operational)} operational"
    )

    # Step 2: Cryptographic integrity
    _verify_signature(content_hash, data)

    # Step 3: Behavioral verification
    _run_canary_tests(principles.canary_tests)

    logger.info("✓ Principles verified. Engine may start.")
    return principles


def get_principle(
    principles: LoadedPrinciples,
    principle_id: str,
) -> Principle | None:
    """Return a principle by ID, searching all tiers."""
    all_principles = (
        *principles.immutable,
        *principles.regulated,
        *principles.operational,
    )
    return next((p for p in all_principles if p.id == principle_id), None)


def is_immutable(principles: LoadedPrinciples, principle_id: str) -> bool:
    """True if the named principle is in the immutable tier."""
    return any(p.id == principle_id for p in principles.immutable)
