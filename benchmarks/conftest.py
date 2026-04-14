"""
benchmarks/conftest.py
======================
Shared fixtures and helpers for all benchmark tests.

Requires: numpy (via `pip install -e ".[cognitive]"`)
"""

from __future__ import annotations

import numpy as np
import pytest

from verity.memory import Memory


def pytest_configure(config: pytest.Config) -> None:
    """Register custom marks so pytest doesn't warn about unknowns."""
    config.addinivalue_line("markers", "slow: marks tests as slow (use -m slow)")


@pytest.fixture
def fresh_memory() -> Memory:
    """
    A clean in-memory Memory instance with no embedding model.
    Uses SQLite ':memory:' so each test gets an isolated, empty store.
    """
    m = Memory(path=":memory:", embedding_model="none")
    yield m  # type: ignore[misc]
    m._store.close()


def make_embedding(topic_idx: int, noise: float = 0.05) -> list[float]:
    """
    Return a 64-dim float32 unit vector for the given topic (0–4).

    Topic i has a 1.0 at position i*12 and zeros elsewhere.
    Gaussian noise (std=noise) is added, then the vector is L2-normalised.
    Uses numpy.random.default_rng(seed=42+topic_idx) for reproducibility.

    Calling this function twice with the same arguments always returns
    the same vector.
    """
    rng = np.random.default_rng(seed=42 + topic_idx)
    vec = np.zeros(64, dtype=np.float32)
    vec[topic_idx * 12] = 1.0
    vec += rng.normal(0.0, noise, size=64).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec.tolist()
