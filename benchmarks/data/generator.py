"""
benchmarks/data/generator.py
=============================
Deterministic synthetic benchmark datasets for Verity.

All five functions are seeded and return identical output on every call.

Usage:
    from benchmarks.data.generator import (
        retrieval_set, interference_set, temporal_set,
        consolidation_set, importance_set,
    )
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

# ---------------------------------------------------------------------------
# Fixed reference time — avoids any dependency on the actual clock.
# ---------------------------------------------------------------------------

_REF_TIME = datetime(2026, 4, 14, 0, 0, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Topic catalogue
# ---------------------------------------------------------------------------

_TOPIC_NAMES = ["preferences", "schedule", "technical", "contacts", "project"]

_TOPIC_MEMORIES: dict[int, list[str]] = {
    0: [  # preferences
        "preferences: I prefer dark mode in all applications",
        "preferences: I use vim keybindings in my IDE",
        "preferences: My preferred programming language is Python",
        "preferences: I prefer tabs over spaces for indentation",
        "preferences: I like to use a standing desk setup",
        "preferences: My favorite terminal is iTerm2",
        "preferences: I prefer async programming patterns",
        "preferences: I like type hints in all Python code",
        "preferences: I prefer functional programming style when possible",
        "preferences: I use a 60% keyboard for coding",
        "preferences: I prefer minimal UI with no distractions",
        "preferences: I like keyboard shortcuts over mouse navigation",
        "preferences: I prefer using virtual environments for Python projects",
        "preferences: I like code reviews to be detailed and thorough",
        "preferences: I prefer test-driven development",
        "preferences: I like to use Git for all version control",
        "preferences: I prefer clear documentation over comments",
        "preferences: I like modular architecture for better maintainability",
        "preferences: I prefer REST APIs over GraphQL",
        "preferences: I like to use linters and formatters in my workflow",
    ],
    1: [  # schedule
        "schedule: Team standup every weekday at 9am",
        "schedule: Weekly sprint review on Fridays at 2pm",
        "schedule: 1-on-1 with manager every other Tuesday at 3pm",
        "schedule: Architecture review meeting last Thursday of each month",
        "schedule: All-hands company meeting first Monday of the month",
        "schedule: Daily meditation practice at 7am",
        "schedule: Gym session every Monday, Wednesday, Friday at 6am",
        "schedule: Book club meets every other Sunday at 4pm",
        "schedule: Quarterly performance review in March, June, September, December",
        "schedule: Weekly team retrospective on Fridays at 4pm",
        "schedule: Lunch break is typically 12pm to 1pm",
        "schedule: Deep work blocks from 10am to 12pm daily",
        "schedule: Code review time blocked on Tuesday afternoons",
        "schedule: Monthly one-on-one with skip-level manager",
        "schedule: Annual conference attendance in October",
        "schedule: Bi-weekly syncs with product team on Wednesdays",
        "schedule: Morning email check at 8am, afternoon check at 4pm",
        "schedule: On-call rotation every 4th week",
        "schedule: Team lunch every other Friday at noon",
        "schedule: Project planning sessions at the start of each sprint",
    ],
    2: [  # technical
        "technical: The REST API uses JWT authentication",
        "technical: PostgreSQL is the production database",
        "technical: Docker is used for all service containerization",
        "technical: CI/CD pipeline runs on GitHub Actions",
        "technical: The frontend is built with React and TypeScript",
        "technical: Redis is used for session caching",
        "technical: Kubernetes orchestrates production containers",
        "technical: The API follows OpenAPI 3.0 specification",
        "technical: All services communicate via gRPC internally",
        "technical: Python 3.11 is the minimum supported version",
        "technical: Terraform manages all infrastructure as code",
        "technical: Datadog is used for monitoring and alerting",
        "technical: The codebase enforces 80% test coverage minimum",
        "technical: Secrets are managed via HashiCorp Vault",
        "technical: The message queue uses RabbitMQ",
        "technical: Nginx serves as the reverse proxy",
        "technical: All code must pass mypy strict mode",
        "technical: S3 is used for object storage",
        "technical: Database migrations use Alembic",
        "technical: Load balancing uses round-robin with health checks",
    ],
    3: [  # contacts
        "contacts: Alice Chen is the frontend lead",
        "contacts: Bob Kumar handles infrastructure",
        "contacts: Carol Johnson is the product manager",
        "contacts: David Lee is the backend lead",
        "contacts: Emma Wilson manages QA and testing",
        "contacts: Frank Martinez is the CTO",
        "contacts: Grace Kim handles customer support",
        "contacts: Henry Brown is the security lead",
        "contacts: Isabel Davis manages the data platform",
        "contacts: James Thompson is the engineering manager",
        "contacts: Karen White handles developer relations",
        "contacts: Larry Harris is the database administrator",
        "contacts: Maria Garcia is the DevOps lead",
        "contacts: Nathan Clark handles legal compliance",
        "contacts: Olivia Scott is the UX designer",
        "contacts: Paul Anderson manages partnerships",
        "contacts: Quinn Taylor is the mobile developer",
        "contacts: Rachel Thomas handles documentation",
        "contacts: Sam Moore is the site reliability engineer",
        "contacts: Tina Jackson manages the API team",
    ],
    4: [  # project
        "project: Project Alpha targets Q3 delivery",
        "project: The API migration is the current blocker",
        "project: Database schema v2 is in review",
        "project: Mobile app beta launches next month",
        "project: Microservices migration is 40% complete",
        "project: Security audit is scheduled for next quarter",
        "project: Performance optimization reduced latency by 30%",
        "project: New onboarding flow reduced drop-off by 25%",
        "project: The monitoring dashboard was completed last sprint",
        "project: Authentication service refactor is in progress",
        "project: API rate limiting feature is in testing",
        "project: Data pipeline migration to AWS is planned for Q4",
        "project: The legacy codebase cleanup is ongoing",
        "project: Test coverage increased from 62% to 81% this quarter",
        "project: The new search feature uses vector embeddings",
        "project: GDPR compliance review is complete",
        "project: The billing system integration is scheduled for Q2",
        "project: CI/CD pipeline rebuild reduced build times by 50%",
        "project: The team is transitioning to trunk-based development",
        "project: Documentation update is due by end of sprint",
    ],
}

# (query_text, [memory_indices_in_topic]) — 6 queries per topic, 3–5 matches each
_TOPIC_QUERIES: dict[int, list[tuple[str, list[int]]]] = {
    0: [
        ("dark mode UI settings", [0, 10, 11]),
        ("keyboard and coding tools", [1, 5, 9, 11]),
        ("Python development preferences", [2, 7, 12]),
        ("code quality practices", [13, 14, 15, 19]),
        ("development workflow", [6, 15, 16, 17]),
        ("programming style and language preferences", [3, 7, 8, 18]),
    ],
    1: [
        ("morning standup daily meeting", [0, 5, 6]),
        ("weekly review and retrospective", [1, 9, 18]),
        ("manager one-on-one reviews", [2, 3, 13]),
        ("all-hands company wide meetings", [4, 14, 15]),
        ("deep work and focus blocks", [11, 12, 16]),
        ("social events and team lunch", [7, 18, 19]),
    ],
    2: [
        ("API authentication and security tokens", [0, 7, 13]),
        ("database storage and migrations", [1, 5, 17, 18]),
        ("container orchestration deployment", [2, 6, 15]),
        ("CI/CD pipeline and build automation", [3, 10, 19]),
        ("code quality testing coverage", [12, 16, 9]),
        ("monitoring infrastructure and alerting", [10, 11, 14]),
    ],
    3: [
        ("engineering technical leads", [0, 3, 9]),
        ("infrastructure DevOps operations", [1, 12, 18]),
        ("product management and leadership", [2, 5, 9]),
        ("security and compliance team", [7, 13, 1]),
        ("design and documentation roles", [14, 17, 10]),
        ("backend API development team", [3, 9, 19]),
    ],
    4: [
        ("project delivery timeline and deadlines", [0, 1, 5]),
        ("database infrastructure changes", [1, 2, 11]),
        ("performance and onboarding improvements", [6, 7, 13]),
        ("API development and features", [1, 10, 14]),
        ("team process and development changes", [18, 19, 3]),
        ("security compliance and legal", [5, 15, 10]),
    ],
}

# Interference pairs: (old_content, new_content) per topic
_INTERFERENCE_PAIRS: dict[str, list[tuple[str, str]]] = {
    "technical": [
        (
            "technical: API uses JWT for authentication",
            "technical: API switched to OAuth2 in v3",
        ),
        (
            "technical: PostgreSQL version 13 in production",
            "technical: Upgraded to PostgreSQL 16 in production",
        ),
        (
            "technical: Tests run on Travis CI",
            "technical: Tests migrated to GitHub Actions",
        ),
        (
            "technical: Redis 6 for session caching",
            "technical: Upgraded to Redis 7 with cluster mode",
        ),
    ],
    "schedule": [
        (
            "schedule: Standup is at 9am",
            "schedule: Standup moved to 10am starting Monday",
        ),
        (
            "schedule: Sprint review on Thursdays at 3pm",
            "schedule: Sprint review moved to Fridays at 2pm",
        ),
        (
            "schedule: All-hands every two weeks",
            "schedule: All-hands moved to monthly cadence",
        ),
        (
            "schedule: 1-on-1 on Mondays at 2pm",
            "schedule: 1-on-1 rescheduled to Wednesdays at 11am",
        ),
    ],
    "project": [
        (
            "project: Target launch is Q3",
            "project: Launch delayed to Q4 due to scope change",
        ),
        (
            "project: API migration is 20% complete",
            "project: API migration is now 80% complete",
        ),
        (
            "project: Using monorepo structure",
            "project: Migrated to separate repos per service",
        ),
        (
            "project: Deploy to AWS us-east-1 only",
            "project: Expanded deployment to us-east-1 and eu-west-1",
        ),
    ],
    "contacts": [
        (
            "contacts: Alice Chen is the frontend developer",
            "contacts: Alice Chen promoted to frontend lead",
        ),
        (
            "contacts: Bob Kumar is a junior DevOps engineer",
            "contacts: Bob Kumar now handles all infrastructure",
        ),
        (
            "contacts: Carol Johnson is the project manager",
            "contacts: Carol Johnson promoted to product manager",
        ),
        (
            "contacts: David Lee is the Python developer",
            "contacts: David Lee is the backend lead",
        ),
    ],
    "preferences": [
        (
            "preferences: I use Sublime Text as my editor",
            "preferences: Switched to VSCode with vim extension",
        ),
        (
            "preferences: I prefer 2-space indentation",
            "preferences: Updated preference to 4-space indentation",
        ),
        (
            "preferences: I use pip for Python packages",
            "preferences: Now using uv for Python package management",
        ),
        (
            "preferences: I prefer monolithic architecture",
            "preferences: Now prefer microservices for better scaling",
        ),
    ],
}

# Consolidation groups: (group_label, [4 memory contents], known_entities)
_CONSOLIDATION_GROUPS: list[tuple[str, list[str], list[str]]] = [
    (
        "grp-00",
        [
            "project: Project Alpha is targeting Q3 delivery with Alice Chen leading",
            "project: Alice Chen confirmed Project Alpha is on track for Q3",
            "project: Q3 deadline for Project Alpha was reviewed in sprint",
            "project: Alice Chen gave Project Alpha status update for Q3 milestone",
        ],
        ["Project Alpha", "Alice Chen", "Q3"],
    ),
    (
        "grp-01",
        [
            "technical: PostgreSQL database migration to staging is complete",
            "technical: Staging environment validated the PostgreSQL migration",
            "technical: Database migration for PostgreSQL passed staging tests",
            "technical: PostgreSQL migration to staging finished without errors",
        ],
        ["PostgreSQL", "database migration", "staging"],
    ),
    (
        "grp-02",
        [
            "technical: JWT authentication tokens are used for API security",
            "technical: API security relies on JWT tokens for auth",
            "technical: JWT-based authentication secures all API endpoints",
            "technical: API uses JWT for authentication and access control",
        ],
        ["JWT", "API", "authentication"],
    ),
    (
        "grp-03",
        [
            "schedule: Daily standup meeting is held every day at 9am",
            "schedule: The 9am standup covers daily progress and blockers",
            "schedule: Team meets for standup at 9am each weekday morning",
            "schedule: Daily 9am standup is the primary team sync meeting",
        ],
        ["standup", "9am", "daily"],
    ),
    (
        "grp-04",
        [
            "preferences: Dark mode is enabled in all UI preferences",
            "preferences: UI settings are configured for dark mode",
            "preferences: Editor and UI both use dark mode as preference",
            "preferences: Dark mode preference is set for all applications and UI",
        ],
        ["dark mode", "UI", "preferences"],
    ),
]


# ---------------------------------------------------------------------------
# Internal embedding helpers
# ---------------------------------------------------------------------------


def _topic_embedding(topic_idx: int, variant: int, noise: float = 0.05) -> list[float]:
    """
    Deterministic unit vector for (topic_idx, variant).
    Topic direction: 1.0 at position topic_idx*12.
    Seeds are spread to avoid collisions across topic×variant pairs.
    """
    rng = np.random.default_rng(seed=42 + topic_idx * 1000 + variant)
    vec = np.zeros(64, dtype=np.float32)
    vec[topic_idx * 12] = 1.0
    vec += rng.normal(0.0, noise, size=64).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec.tolist()


def _group_embedding(group_idx: int, member_idx: int, noise: float = 0.02) -> list[float]:
    """
    Very-similar unit vectors within one consolidation group.
    σ=0.02 gives intra-group cosine similarity ≈ 0.975 >> 0.85 threshold.
    """
    rng = np.random.default_rng(seed=42 + group_idx * 100 + member_idx + 7000)
    vec = np.zeros(64, dtype=np.float32)
    vec[group_idx * 12] = 1.0
    vec += rng.normal(0.0, noise, size=64).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec.tolist()


# ---------------------------------------------------------------------------
# Timestamp generators
# ---------------------------------------------------------------------------


def _periodic_timestamps(ref: datetime, count: int = 10) -> list[datetime]:
    """10 timestamps spaced 168 h apart; last access 24 h before ref."""
    last = ref - timedelta(hours=24)
    return [last - timedelta(hours=168 * (count - 1 - i)) for i in range(count)]


def _bursty_timestamps(ref: datetime) -> list[datetime]:
    """
    8 timestamps in a ~21 h window ending 720 h before ref (30 days ago),
    plus 2 isolated older timestamps.  Last access = ref - 720 h.
    """
    burst_end = ref - timedelta(hours=720)
    # 8 timestamps: every 3 h, from burst_end-21h to burst_end
    burst = [burst_end - timedelta(hours=21 - i * 3) for i in range(8)]
    isolated = [
        ref - timedelta(hours=1440),  # 60 days ago
        ref - timedelta(hours=900),   # 37.5 days ago
    ]
    return sorted(isolated + burst)


def _declining_timestamps(ref: datetime) -> list[datetime]:
    """
    10 timestamps at 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 h before ref.
    Exponential spacing — densely accessed recently, sparse further back.
    Returned in chronological order (oldest first).
    """
    hours = [2**i for i in range(10)]  # 1, 2, 4, ..., 512
    return [ref - timedelta(hours=h) for h in reversed(hours)]


# ---------------------------------------------------------------------------
# Public dataset generators
# ---------------------------------------------------------------------------


def retrieval_set() -> dict[str, Any]:
    """
    100 memories (20 per topic) + 30 queries (6 per topic, 3–5 matches each).

    Returns:
        {
            "memories": list of {id, content, topic, embedding},
            "queries":  list of {query, relevant_ids},
        }

    Seeded with numpy.random.default_rng(42) for ordering shuffle.
    """
    rng = np.random.default_rng(42)

    memories: list[dict[str, Any]] = []
    for topic_idx in range(5):
        for mem_idx, content in enumerate(_TOPIC_MEMORIES[topic_idx]):
            memories.append(
                {
                    "id": f"ret-{topic_idx}-{mem_idx:02d}",
                    "content": content,
                    "topic": topic_idx,
                    "embedding": _topic_embedding(topic_idx, mem_idx),
                }
            )

    # Reproducible shuffle
    order = np.arange(len(memories))
    rng.shuffle(order)
    memories = [memories[int(i)] for i in order]

    queries: list[dict[str, Any]] = []
    for topic_idx, topic_queries in _TOPIC_QUERIES.items():
        for _q_idx, (query_text, mem_indices) in enumerate(topic_queries):
            relevant_ids = [f"ret-{topic_idx}-{m:02d}" for m in mem_indices]
            queries.append({"query": query_text, "relevant_ids": relevant_ids})

    return {"memories": memories, "queries": queries}


def interference_set() -> list[dict[str, Any]]:
    """
    20 (old, new) fact pairs where new_content supersedes old_content.
    4 pairs per topic.  Old and new embeddings share the topic direction
    but use different noise seeds, so they are similar but not identical.

    Returns list of:
        {id, old_content, new_content, topic, old_embedding, new_embedding}
    """
    result: list[dict[str, Any]] = []
    for topic_name, pairs in _INTERFERENCE_PAIRS.items():
        topic_idx = _TOPIC_NAMES.index(topic_name)
        for pair_idx, (old_content, new_content) in enumerate(pairs):
            result.append(
                {
                    "id": f"int-{topic_name}-{pair_idx:02d}",
                    "old_content": old_content,
                    "new_content": new_content,
                    "topic": topic_name,
                    "old_embedding": _topic_embedding(topic_idx, 500 + pair_idx),
                    "new_embedding": _topic_embedding(topic_idx, 600 + pair_idx),
                }
            )
    return result


def temporal_set() -> list[dict[str, Any]]:
    """
    30 memories with structured access patterns (10 per pattern).

    Patterns:
      periodic  — 10 accesses weekly (168 h apart), last 24 h ago
      bursty    — burst of 8 in ~21 h window + 2 isolated; last 720 h ago
      declining — exponential spacing 512→1 h before now

    Returns list of:
        {id, content, topic, embedding, access_pattern, access_timestamps}
    """
    patterns: dict[str, list[datetime]] = {
        "periodic": _periodic_timestamps(_REF_TIME),
        "bursty": _bursty_timestamps(_REF_TIME),
        "declining": _declining_timestamps(_REF_TIME),
    }

    result: list[dict[str, Any]] = []
    for pattern_name, timestamps in patterns.items():
        # Map each pattern to a distinct topic for embedding variety
        topic_map = {"periodic": 1, "bursty": 2, "declining": 3}
        topic_idx = topic_map[pattern_name]
        for i in range(10):
            result.append(
                {
                    "id": f"temporal-{pattern_name}-{i:02d}",
                    "content": (
                        f"{_TOPIC_NAMES[topic_idx]}: "
                        f"temporal memory {pattern_name} {i}"
                    ),
                    "topic": topic_idx,
                    "embedding": _topic_embedding(topic_idx, 200 + i),
                    "access_pattern": pattern_name,
                    "access_timestamps": list(timestamps),  # same pattern, 10 memories
                }
            )
    return result


def consolidation_set() -> list[dict[str, Any]]:
    """
    5 groups × 4 memories.  Within each group all embeddings have
    cosine similarity ≈ 0.975 (well above the 0.85 clustering threshold).

    Returns list of:
        {group_id, memories: [{id, content, embedding}], known_entities}
    """
    result: list[dict[str, Any]] = []
    for group_idx, (group_id, contents, entities) in enumerate(_CONSOLIDATION_GROUPS):
        memories: list[dict[str, Any]] = [
            {
                "id": f"{group_id}-mem-{member_idx:02d}",
                "content": content,
                "embedding": _group_embedding(group_idx, member_idx),
            }
            for member_idx, content in enumerate(contents)
        ]
        result.append(
            {
                "group_id": group_id,
                "memories": memories,
                "known_entities": list(entities),
            }
        )
    return result


def importance_set() -> dict[str, list[MemoryEntry]]:
    """
    20 IMMUTABLE entries (high confidence) + 20 LABILE entries (low confidence).

    High-tier: alpha=250, beta=10, source_count=6, confidence≈0.962 → IMMUTABLE
    Low-tier:  alpha=2,   beta=5,  source_count=1, confidence≈0.286 → LABILE

    Returns:
        {"high": list[MemoryEntry], "low": list[MemoryEntry]}
    """
    high_contents = [
        f"preferences: well-established preference #{i:02d}" for i in range(20)
    ]
    low_contents = [
        f"preferences: uncertain preference #{i:02d}" for i in range(20)
    ]

    def _make_entry(
        entry_id: str,
        content: str,
        alpha: float,
        beta: float,
        source_count: int,
        confidence_tier: ConfidenceTier,
        importance: float,
    ) -> MemoryEntry:
        return MemoryEntry(
            memory_id=str(uuid.uuid5(
                uuid.UUID("deadbeef-dead-beef-dead-beefdeadbeef"),
                entry_id,
            )),
            content=content,
            user_id="benchmark",
            tier=MemoryTier.FAST,
            confidence_tier=confidence_tier,
            importance=importance,
            strength=1.0,
            created_at=_REF_TIME,
            last_accessed=_REF_TIME,
            access_count=0,
            source_count=source_count,
            alpha=alpha,
            beta=beta,
        )

    high = [
        _make_entry(
            f"imp-high-{i:02d}",
            content,
            alpha=250.0,
            beta=10.0,
            source_count=6,
            confidence_tier=ConfidenceTier.IMMUTABLE,
            importance=0.9,
        )
        for i, content in enumerate(high_contents)
    ]

    low = [
        _make_entry(
            f"imp-low-{i:02d}",
            content,
            alpha=2.0,
            beta=5.0,
            source_count=1,
            confidence_tier=ConfidenceTier.LABILE,
            importance=0.3,
        )
        for i, content in enumerate(low_contents)
    ]

    return {"high": high, "low": low}
