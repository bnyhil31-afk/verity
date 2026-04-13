"""
verity.core.profiles
====================
Deployment profiles for the Verity engine.

Four profiles cover every target user type:
  personal     — Home user. Zero config. Local only. Pre-signed principles.
  developer    — Developer building AI agents. Full connector SDK. Local signing.
  professional — Team/SMB. Multi-user. PostgreSQL backend.
  enterprise   — Regulated industries. Full audit trail. M-of-N principles ceremony.

Usage:
    from verity.core.profiles import get_profile, PERSONAL

    # In Engine.start():
    engine = await Engine.start(profile="personal")
    engine = await Engine.start(profile=DEVELOPER)

Profile sets defaults for:
  - decay_parameters    (scientific constants; override only with citation)
  - bfs_max_depth       (graph traversal depth)
  - checkpoint_timeout  (GOVERN checkpoint timeout — veto fires at expiry)
  - checkpoint_mode     (interactive stdout/stdin vs deferred web/API)
  - auto_sign_principles (True = use package key; False = operator signs)
  - graph_store_backend (which backend to initialize by default)
"""

from __future__ import annotations

from dataclasses import dataclass

from verity.core.types import DEFAULT_DECAY_PARAMETERS, DecayParameters


@dataclass(frozen=True)
class EngineProfile:
    """
    Deployment profile — activates the right blades for each user type.

    Frozen: profiles are constants, not configuration objects.
    Pass to Engine.start(profile=...) as a string name or instance.
    """
    name: str
    decay_parameters: DecayParameters
    bfs_max_depth: int
    checkpoint_timeout_seconds: int
    checkpoint_interactive: bool   # True = stdout/stdin, False = deferred
    auto_sign_principles: bool     # True = use package key, no ceremony
    graph_store_backend: str       # "rdflib" | "oxigraph" | "postgres" | "jena"
    description: str


PERSONAL = EngineProfile(
    name="personal",
    decay_parameters=DEFAULT_DECAY_PARAMETERS,
    bfs_max_depth=2,
    checkpoint_timeout_seconds=300,
    checkpoint_interactive=True,
    auto_sign_principles=True,      # pre-signed at install, no ceremony
    graph_store_backend="rdflib",
    description="Home user. Zero config. Local only. Pre-signed principles.",
)

DEVELOPER = EngineProfile(
    name="developer",
    decay_parameters=DEFAULT_DECAY_PARAMETERS,
    bfs_max_depth=3,
    checkpoint_timeout_seconds=600,
    checkpoint_interactive=True,
    auto_sign_principles=False,     # developer signs with their own key
    graph_store_backend="rdflib",
    description="Developer building AI agents. Full connector SDK. Local signing.",
)

PROFESSIONAL = EngineProfile(
    name="professional",
    decay_parameters=DEFAULT_DECAY_PARAMETERS,
    bfs_max_depth=3,
    checkpoint_timeout_seconds=3600,
    checkpoint_interactive=False,   # deferred — web UI or API response
    auto_sign_principles=False,
    graph_store_backend="postgres",
    description="Team/SMB. Multi-user. PostgreSQL backend.",
)

ENTERPRISE = EngineProfile(
    name="enterprise",
    decay_parameters=DecayParameters(exponent=0.4, sensitive_multiplier=1.6),
    bfs_max_depth=4,
    checkpoint_timeout_seconds=86400,  # 24 hours
    checkpoint_interactive=False,
    auto_sign_principles=False,        # M-of-N ceremony required
    graph_store_backend="jena",
    description="Regulated industries. Full audit trail. M-of-N principles ceremony.",
)

PROFILES: dict[str, EngineProfile] = {
    p.name: p for p in [PERSONAL, DEVELOPER, PROFESSIONAL, ENTERPRISE]
}


def get_profile(name: str) -> EngineProfile:
    """
    Return a named EngineProfile.

    Raises:
        ValueError — if name is not a recognised profile.
    """
    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Valid: {list(PROFILES)}"
        )
    return PROFILES[name]
