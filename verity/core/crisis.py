"""
verity.core.crisis
==================
The absolute crisis barrier.

This module runs FIRST — before entity recognition, before graph writes,
before anything. It is the unconditional safety layer that cannot be
configured away, disabled, or bypassed.

If is_crisis_input() returns True:
  - All graph writes are blocked
  - CrisisBarrierError is raised
  - An audit event is recorded (CRISIS_DETECTED)
  - Crisis resources are returned to the caller

This behavior is verified by canary tests at every engine boot.
A canary failure halts the engine. There are no exceptions to this.

Design discipline:
  - No external dependencies — pattern matching only
  - Deterministic — same input always produces same output
  - Fast — runs on every single ingestion call
  - Conservative — false positives are acceptable, false negatives are not
  - Extensible — domain modules may register additional patterns
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime

from verity.core.exceptions import CrisisBarrierError

logger = logging.getLogger(__name__)


# ── Crisis resources ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CrisisResource:
    """A single crisis resource returned when the barrier fires."""
    name: str
    contact: str
    available: str
    notes: str | None = None


# Default crisis resources — always present, always returned
# Domain modules may prepend domain-specific resources (e.g. clinical hotlines)
DEFAULT_CRISIS_RESOURCES: tuple[CrisisResource, ...] = (
    CrisisResource(
        name="988 Suicide and Crisis Lifeline",
        contact="Call or text 988",
        available="24/7",
        notes="Free, confidential support for people in distress.",
    ),
    CrisisResource(
        name="Crisis Text Line",
        contact="Text HOME to 741741",
        available="24/7",
        notes="Free crisis counseling via text message.",
    ),
    CrisisResource(
        name="International Association for Suicide Prevention",
        contact="https://www.iasp.info/resources/Crisis_Centres/",
        available="24/7",
        notes="Directory of crisis centers worldwide.",
    ),
)


# ── Crisis patterns ───────────────────────────────────────────────────────────
#
# Conservative by design. False positives are acceptable — a legitimate
# message routed to crisis resources causes inconvenience. A crisis message
# that passes through causes harm. The asymmetry is not symmetric.
#
# Patterns are compiled once at module load. No runtime compilation.

_CRISIS_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        # Direct statements of suicidal ideation
        r"\bwant\s+to\s+(end|take)\s+(my|this)\s+life\b",
        r"\bkill\s+myself\b",
        r"\bsuicid(e|al)\b",
        r"\bend\s+it\s+(all|now)\b",
        r"\bnot\s+(want|wanting)\s+to\s+(be\s+here|live|exist)\b",
        r"\bthinking\s+about\s+(not\s+being\s+here|ending\s+(it|things|my\s+life))\b",

        # Indirect but high-signal expressions
        r"\bno\s+reason\s+to\s+(live|go\s+on|keep\s+going)\b",
        r"\bbetter\s+off\s+(dead|without\s+me)\b",
        r"\bcan'?t\s+(go\s+on|keep\s+going|do\s+this\s+anymore)\b",
        r"\bwish\s+I\s+(was|were|wasn'?t|hadn'?t)\s+(\w+\s+)?(born|alive|here)\b",
        r"\bdon'?t\s+want\s+to\s+(wake\s+up|be\s+alive|exist)\b",

        # Self-harm
        r"\bhurt(ing)?\s+(myself|my\s+self)\b",
        r"\bself[\s-]?harm(ing)?\b",
        r"\bcut(ting)?\s+(myself|my\s+(arms?|wrists?|legs?|body))\b",

        # Crisis states
        r"\boverdos(e|ing)\b",
        r"\bactive\s+(crisis|emergency)\b",
    ]
)

# Additional patterns registered by domain modules at runtime
_REGISTERED_PATTERNS: list[re.Pattern[str]] = []


# ── Detection ─────────────────────────────────────────────────────────────────


def is_crisis_input(text: str) -> bool:
    """
    Return True if the text contains crisis signals.

    Runs all built-in patterns plus any domain-registered patterns.
    Conservative — errs toward True. Fast — compiled regex only.

    This is the function called by the canary tests at every boot.
    It must be deterministic and have no side effects.
    """
    if not text or not text.strip():
        return False

    normalized = text.strip()

    for pattern in _CRISIS_PATTERNS:
        if pattern.search(normalized):
            return True

    for pattern in _REGISTERED_PATTERNS:
        if pattern.search(normalized):
            return True

    return False


def register_pattern(pattern: str) -> None:
    """
    Register an additional crisis detection pattern from a domain module.

    Domain modules call this during initialization to add domain-specific
    crisis signals (e.g. clinical codes, domain terminology).

    The pattern is compiled immediately — invalid regex raises ValueError.
    Registered patterns supplement built-in patterns, never replace them.
    """
    compiled = re.compile(pattern, re.IGNORECASE)
    _REGISTERED_PATTERNS.append(compiled)
    logger.debug(f"Crisis pattern registered: {pattern!r}")


# ── Barrier ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CrisisBarrierResult:
    """
    Returned when the crisis barrier fires.

    The caller must surface crisis_resources to the user immediately.
    The audit_event must be passed to REMEMBER for immutable recording.
    No graph writes occur when this is returned.
    """
    detected_at: datetime
    input_excerpt: str              # First 100 chars only — never full input
    crisis_resources: tuple[CrisisResource, ...]
    audit_payload: dict             # Ready for AuditEvent construction


def check_and_raise(
    text: str,
    actor: str,
    session_id: str | None = None,
    additional_resources: tuple[CrisisResource, ...] = (),
) -> None:
    """
    Run the crisis barrier. Raise CrisisBarrierError if crisis detected.

    This is the function called by RELATE before any other processing.
    If it returns normally, the input is safe to proceed.
    If it raises, all processing must stop immediately.

    The caller is responsible for passing the audit payload to REMEMBER.

    Args:
        text:                 The raw input text to check
        actor:                Who submitted this input (for audit trail)
        session_id:           Active session, if any
        additional_resources: Domain-specific resources to prepend
    """
    if not is_crisis_input(text):
        return

    excerpt = text[:100].strip()

    logger.critical(
        f"CRISIS BARRIER FIRED | actor={actor} | "
        f"session={session_id} | excerpt='{excerpt[:40]}...'"
    )

    raise CrisisBarrierError(input_excerpt=excerpt)


def get_crisis_resources(
    additional: tuple[CrisisResource, ...] = (),
) -> tuple[CrisisResource, ...]:
    """
    Return the full set of crisis resources.

    Domain modules pass their resources as `additional` — they are
    prepended so domain-specific help appears first.
    """
    return (*additional, *DEFAULT_CRISIS_RESOURCES)
