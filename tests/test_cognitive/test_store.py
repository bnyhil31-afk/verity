"""
tests.test_cognitive.test_store
================================
Tests for DualSpeedStore — the SQLite-backed CLS dual-memory implementation.

Coverage:
- Store init creates both SQLite tables
- add() returns MemoryEntry with a valid memory_id
- search() returns <= k results
- get() returns None for unknown id
- get() increments access_count on each call
- update() changes content and increments access_count
- delete() removes memory and returns True; subsequent get() returns None
- delete() returns False for unknown id
- Capacity limit: adding beyond fast_capacity evicts the lowest-importance entry
- promote() moves entry from fast to slow table
- all_fast() / all_slow() return correct entries
- stats() returns expected keys and values
- In-memory path (":memory:") works
- Works with embedding_model="none" (no numpy required)
- File-based path creates the .db file
"""

from __future__ import annotations

import pytest

from verity.cognitive.store import DualSpeedStore
from verity.cognitive.types import ConfidenceTier, MemoryEntry, MemoryTier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(**kwargs) -> DualSpeedStore:
    """Return a fresh in-memory store with embedding disabled."""
    kwargs.setdefault("path", ":memory:")
    kwargs.setdefault("embedding_model", "none")
    return DualSpeedStore(**kwargs)


def _table_count(store: DualSpeedStore, table: str) -> int:
    cur = store._conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    return cur.fetchone()[0]


def _set_importance(store: DualSpeedStore, memory_id: str, importance: float) -> None:
    """Directly update importance in the DB for eviction tests."""
    store._conn.execute(
        "UPDATE fast_memories SET importance = ? WHERE memory_id = ?",
        (importance, memory_id),
    )
    store._conn.commit()


# ---------------------------------------------------------------------------
# Init / schema
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_fast_table(self):
        store = _make_store()
        assert _table_count(store, "fast_memories") == 0

    def test_creates_slow_table(self):
        store = _make_store()
        assert _table_count(store, "slow_memories") == 0

    def test_in_memory_db_path(self):
        store = _make_store(path=":memory:")
        assert store._db_path == ":memory:"

    def test_file_db_created(self, tmp_path):
        db_file = tmp_path / "test_memory.db"
        DualSpeedStore(
            path=str(db_file),
            embedding_model="none",
        )
        assert db_file.exists()

    def test_tilde_expansion(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        store = DualSpeedStore(
            path="~/.verity/memory.db",
            embedding_model="none",
        )
        assert ":memory:" not in store._db_path
        assert "~" not in store._db_path

    def test_embedding_model_none(self):
        store = _make_store(embedding_model="none")
        assert store._embed_model is None

    def test_default_user_id(self):
        store = _make_store()
        assert store._user_id == "default"

    def test_custom_user_id(self):
        store = _make_store(user_id="alice")
        assert store._user_id == "alice"


# ---------------------------------------------------------------------------
# add()
# ---------------------------------------------------------------------------

class TestAdd:
    def test_returns_memory_entry(self):
        store = _make_store()
        entry = store.add("Hello, world!")
        assert isinstance(entry, MemoryEntry)

    def test_returns_valid_memory_id(self):
        store = _make_store()
        entry = store.add("Something memorable.")
        assert isinstance(entry.memory_id, str)
        assert len(entry.memory_id) > 0

    def test_memory_id_is_uuid(self):
        import re
        store = _make_store()
        entry = store.add("UUID check.")
        pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
        assert re.match(pattern, entry.memory_id, re.I), f"Not a UUID4: {entry.memory_id}"

    def test_stores_content(self):
        store = _make_store()
        entry = store.add("The answer is 42.")
        assert entry.content == "The answer is 42."

    def test_tier_is_fast(self):
        store = _make_store()
        entry = store.add("Fast memory.")
        assert entry.tier == MemoryTier.FAST

    def test_confidence_tier_is_labile(self):
        store = _make_store()
        entry = store.add("New memory.")
        assert entry.confidence_tier == ConfidenceTier.LABILE

    def test_increments_fast_count(self):
        store = _make_store()
        store.add("First.")
        store.add("Second.")
        assert _table_count(store, "fast_memories") == 2

    def test_metadata_stored(self):
        store = _make_store()
        entry = store.add("With metadata.", metadata={"source": "test"})
        assert entry.metadata == {"source": "test"}

    def test_default_metadata_empty(self):
        store = _make_store()
        entry = store.add("No metadata.")
        assert entry.metadata == {}

    def test_access_count_starts_at_zero(self):
        store = _make_store()
        entry = store.add("Fresh memory.")
        assert entry.access_count == 0

    def test_strength_starts_at_one(self):
        store = _make_store()
        entry = store.add("Strong memory.")
        assert entry.strength == pytest.approx(1.0)

    def test_unique_ids_per_add(self):
        store = _make_store()
        ids = {store.add(f"Memory {i}").memory_id for i in range(10)}
        assert len(ids) == 10

    def test_user_id_scoped(self):
        store_a = _make_store(user_id="alice")
        store_b = _make_store(user_id="bob")
        # Both use different connections, but if they share a DB they're isolated
        store_a.add("Alice's memory.")
        # store_b sees 0 for its own user
        assert len(store_b.all_fast()) == 0

    def test_without_numpy_no_embedding(self):
        """Works with embedding_model='none' even if numpy isn't installed."""
        store = _make_store(embedding_model="none")
        entry = store.add("Zero-dep path.")
        assert entry.embedding is None


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestSearch:
    def test_empty_store_returns_empty(self):
        store = _make_store()
        results = store.search("anything")
        assert results == []

    def test_returns_list(self):
        store = _make_store()
        store.add("Python is great.")
        results = store.search("Python")
        assert isinstance(results, list)

    def test_returns_retrieval_results(self):
        from verity.cognitive.types import RetrievalResult
        store = _make_store()
        store.add("Hello world.")
        results = store.search("hello")
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_returns_at_most_k(self):
        store = _make_store()
        for i in range(10):
            store.add(f"Memory number {i}.")
        results = store.search("memory", k=3)
        assert len(results) <= 3

    def test_returns_at_most_k_default(self):
        store = _make_store()
        for i in range(10):
            store.add(f"Entry {i}")
        results = store.search("entry")
        assert len(results) <= 5

    def test_positions_are_one_indexed(self):
        store = _make_store()
        for i in range(3):
            store.add(f"Item {i}")
        results = store.search("item", k=3)
        positions = [r.position for r in results]
        assert positions == list(range(1, len(results) + 1))

    def test_sorted_by_score_descending(self):
        store = _make_store()
        for i in range(5):
            store.add(f"Entry {i}")
        results = store.search("entry", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_in_zero_one_range(self):
        store = _make_store()
        store.add("cats and dogs")
        store.add("birds and fish")
        results = store.search("cats", k=5)
        for r in results:
            assert 0.0 <= r.score <= 1.0 + 1e-9

    def test_keyword_match_higher_than_unrelated(self):
        store = _make_store()
        store.add("cats are wonderful pets")
        store.add("the stock market crashed yesterday")
        results = store.search("cats pets", k=2)
        # The cat memory should score higher
        assert results[0].memory.content == "cats are wonderful pets"

    def test_searches_both_fast_and_slow(self):
        store = _make_store()
        entry = store.add("Slow store content.")
        store.promote(entry.memory_id)
        store.add("Fast store content.")
        results = store.search("content", k=5)
        contents = {r.memory.content for r in results}
        assert "Slow store content." in contents
        assert "Fast store content." in contents

    def test_no_results_above_k(self):
        store = _make_store()
        for i in range(20):
            store.add(f"Unique content about topic {i}")
        results = store.search("topic", k=7)
        assert len(results) <= 7


# ---------------------------------------------------------------------------
# get()
# ---------------------------------------------------------------------------

class TestGet:
    def test_returns_none_for_unknown_id(self):
        store = _make_store()
        result = store.get("nonexistent-id")
        assert result is None

    def test_returns_entry_for_known_id(self):
        store = _make_store()
        entry = store.add("Findable memory.")
        fetched = store.get(entry.memory_id)
        assert fetched is not None
        assert fetched.content == "Findable memory."

    def test_increments_access_count(self):
        store = _make_store()
        entry = store.add("Access counting.")
        assert entry.access_count == 0
        store.get(entry.memory_id)
        fetched = store.get(entry.memory_id)
        assert fetched is not None
        assert fetched.access_count == 2

    def test_updates_last_accessed(self):
        store = _make_store()
        entry = store.add("Time tracking.")
        original_ts = entry.last_accessed
        fetched = store.get(entry.memory_id)
        assert fetched is not None
        assert fetched.last_accessed >= original_ts

    def test_finds_in_slow_store(self):
        store = _make_store()
        entry = store.add("Promoted memory.")
        store.promote(entry.memory_id)
        fetched = store.get(entry.memory_id)
        assert fetched is not None
        assert fetched.content == "Promoted memory."

    def test_returns_correct_memory_id(self):
        store = _make_store()
        entry = store.add("ID check.")
        fetched = store.get(entry.memory_id)
        assert fetched is not None
        assert fetched.memory_id == entry.memory_id


# ---------------------------------------------------------------------------
# update()
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_changes_content(self):
        store = _make_store()
        entry = store.add("Original content.")
        updated = store.update(entry.memory_id, "Updated content.")
        assert updated.content == "Updated content."

    def test_increments_access_count(self):
        store = _make_store()
        entry = store.add("Access count test.")
        assert entry.access_count == 0
        updated = store.update(entry.memory_id, "New content.")
        assert updated.access_count == 1

    def test_updates_last_accessed(self):
        store = _make_store()
        entry = store.add("Timestamp test.")
        original_ts = entry.last_accessed
        updated = store.update(entry.memory_id, "Changed.")
        assert updated.last_accessed >= original_ts

    def test_does_not_change_confidence_tier(self):
        store = _make_store()
        entry = store.add("Immutable confidence.")
        original_tier = entry.confidence_tier
        updated = store.update(entry.memory_id, "New text.")
        assert updated.confidence_tier == original_tier

    def test_does_not_change_alpha_beta(self):
        store = _make_store()
        entry = store.add("Bayesian params.")
        updated = store.update(entry.memory_id, "Changed text.")
        assert updated.alpha == pytest.approx(entry.alpha)
        assert updated.beta == pytest.approx(entry.beta)

    def test_raises_for_unknown_id(self):
        store = _make_store()
        with pytest.raises(KeyError):
            store.update("no-such-id", "content")

    def test_persists_after_get(self):
        store = _make_store()
        entry = store.add("Before update.")
        store.update(entry.memory_id, "After update.")
        fetched = store.get(entry.memory_id)
        assert fetched is not None
        assert fetched.content == "After update."

    def test_works_on_slow_store(self):
        store = _make_store()
        entry = store.add("Will be promoted.")
        store.promote(entry.memory_id)
        updated = store.update(entry.memory_id, "Updated in slow.")
        assert updated.content == "Updated in slow."


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestDelete:
    def test_returns_true_on_success(self):
        store = _make_store()
        entry = store.add("Delete me.")
        assert store.delete(entry.memory_id) is True

    def test_returns_false_for_unknown_id(self):
        store = _make_store()
        assert store.delete("no-such-id") is False

    def test_entry_not_found_after_delete(self):
        store = _make_store()
        entry = store.add("Gone soon.")
        store.delete(entry.memory_id)
        assert store.get(entry.memory_id) is None

    def test_decrements_fast_count(self):
        store = _make_store()
        entry = store.add("To delete.")
        store.add("To keep.")
        store.delete(entry.memory_id)
        assert _table_count(store, "fast_memories") == 1

    def test_deletes_from_slow_store(self):
        store = _make_store()
        entry = store.add("Promote then delete.")
        store.promote(entry.memory_id)
        assert store.delete(entry.memory_id) is True
        assert _table_count(store, "slow_memories") == 0

    def test_double_delete_returns_false(self):
        store = _make_store()
        entry = store.add("One-time delete.")
        store.delete(entry.memory_id)
        assert store.delete(entry.memory_id) is False


# ---------------------------------------------------------------------------
# Capacity eviction
# ---------------------------------------------------------------------------

class TestCapacityEviction:
    def test_fast_count_does_not_exceed_capacity(self):
        store = _make_store(fast_capacity=3)
        for i in range(5):
            store.add(f"Memory {i}")
        assert _table_count(store, "fast_memories") == 3

    def test_evicts_lowest_importance_entry(self):
        store = _make_store(fast_capacity=3)
        e1 = store.add("Entry one")
        e2 = store.add("Entry two")
        e3 = store.add("Entry three")
        # Set e1 to very low importance — it should be evicted next
        _set_importance(store, e1.memory_id, 0.01)
        _set_importance(store, e2.memory_id, 0.8)
        _set_importance(store, e3.memory_id, 0.9)
        # Adding a 4th entry triggers eviction of e1 (lowest importance)
        e4 = store.add("Entry four")
        remaining_ids = {e.memory_id for e in store.all_fast()}
        assert e1.memory_id not in remaining_ids
        assert e4.memory_id in remaining_ids

    def test_high_importance_entries_survive(self):
        store = _make_store(fast_capacity=2)
        e1 = store.add("High importance")
        e2 = store.add("Medium importance")
        _set_importance(store, e1.memory_id, 0.99)
        _set_importance(store, e2.memory_id, 0.5)
        # Adding a third triggers eviction of e2 (lower importance)
        store.add("New entry")
        remaining_ids = {e.memory_id for e in store.all_fast()}
        assert e1.memory_id in remaining_ids

    def test_capacity_one(self):
        store = _make_store(fast_capacity=1)
        store.add("First")
        store.add("Second")
        assert _table_count(store, "fast_memories") == 1

    def test_slow_store_unaffected_by_fast_eviction(self):
        store = _make_store(fast_capacity=2)
        e1 = store.add("Fast one")
        store.promote(e1.memory_id)
        store.add("Fast two")
        store.add("Fast three")
        # Fast is now at capacity (2) before adding a 3rd; promote removed e1
        # slow_memories should still have e1
        slow = store.all_slow()
        assert any(e.memory_id == e1.memory_id for e in slow)


# ---------------------------------------------------------------------------
# promote()
# ---------------------------------------------------------------------------

class TestPromote:
    def test_moves_to_slow_table(self):
        store = _make_store()
        entry = store.add("Promote me.")
        store.promote(entry.memory_id)
        assert _table_count(store, "slow_memories") == 1
        assert _table_count(store, "fast_memories") == 0

    def test_promoted_entry_has_slow_tier(self):
        store = _make_store()
        entry = store.add("Tier check.")
        promoted = store.promote(entry.memory_id)
        assert promoted.tier == MemoryTier.SLOW

    def test_fast_entry_removed_after_promote(self):
        store = _make_store()
        entry = store.add("Will be promoted.")
        store.promote(entry.memory_id)
        fast = store.all_fast()
        assert not any(e.memory_id == entry.memory_id for e in fast)

    def test_content_preserved_after_promote(self):
        store = _make_store()
        entry = store.add("Preserve my content.")
        promoted = store.promote(entry.memory_id)
        assert promoted.content == "Preserve my content."

    def test_raises_for_unknown_id(self):
        store = _make_store()
        with pytest.raises(KeyError):
            store.promote("no-such-id")

    def test_raises_if_already_in_slow(self):
        store = _make_store()
        entry = store.add("Already promoted.")
        store.promote(entry.memory_id)
        with pytest.raises(KeyError):
            store.promote(entry.memory_id)

    def test_promoted_entry_retrievable_via_get(self):
        store = _make_store()
        entry = store.add("Slow store get.")
        store.promote(entry.memory_id)
        fetched = store.get(entry.memory_id)
        assert fetched is not None
        assert fetched.memory_id == entry.memory_id


# ---------------------------------------------------------------------------
# all_fast() / all_slow()
# ---------------------------------------------------------------------------

class TestAllFastAllSlow:
    def test_all_fast_empty(self):
        store = _make_store()
        assert store.all_fast() == []

    def test_all_slow_empty(self):
        store = _make_store()
        assert store.all_slow() == []

    def test_all_fast_returns_correct_count(self):
        store = _make_store()
        for i in range(4):
            store.add(f"Fast {i}")
        assert len(store.all_fast()) == 4

    def test_all_slow_returns_correct_count(self):
        store = _make_store()
        for i in range(3):
            entry = store.add(f"Promote {i}")
            store.promote(entry.memory_id)
        assert len(store.all_slow()) == 3

    def test_all_fast_returns_memory_entries(self):
        store = _make_store()
        store.add("Entry A.")
        entries = store.all_fast()
        assert all(isinstance(e, MemoryEntry) for e in entries)

    def test_all_fast_user_scoped(self):
        store_a = _make_store(user_id="alice")
        store_b = _make_store(user_id="bob")
        store_a.add("Alice's memory.")
        assert store_b.all_fast() == []


# ---------------------------------------------------------------------------
# stats()
# ---------------------------------------------------------------------------

class TestStats:
    def test_required_keys_present(self):
        store = _make_store()
        s = store.stats()
        for key in ("fast_count", "slow_count", "fast_capacity",
                    "embedding_model", "db_path", "total_count"):
            assert key in s, f"Missing key: {key}"

    def test_fast_count_accurate(self):
        store = _make_store()
        store.add("A")
        store.add("B")
        assert store.stats()["fast_count"] == 2

    def test_slow_count_accurate(self):
        store = _make_store()
        entry = store.add("Promote me.")
        store.promote(entry.memory_id)
        s = store.stats()
        assert s["slow_count"] == 1
        assert s["fast_count"] == 0

    def test_total_count(self):
        store = _make_store()
        e1 = store.add("One")
        store.add("Two")
        store.promote(e1.memory_id)
        s = store.stats()
        assert s["total_count"] == s["fast_count"] + s["slow_count"]
        assert s["total_count"] == 2

    def test_fast_capacity_correct(self):
        store = _make_store(fast_capacity=42)
        assert store.stats()["fast_capacity"] == 42

    def test_embedding_model_none(self):
        store = _make_store(embedding_model="none")
        assert store.stats()["embedding_model"] == "none"

    def test_db_path_in_memory(self):
        store = _make_store(path=":memory:")
        assert store.stats()["db_path"] == ":memory:"

    def test_db_path_file(self, tmp_path):
        db_file = tmp_path / "stats_test.db"
        store = DualSpeedStore(path=str(db_file), embedding_model="none")
        assert store.stats()["db_path"] == str(db_file)


# ---------------------------------------------------------------------------
# In-memory path
# ---------------------------------------------------------------------------

class TestInMemory:
    def test_add_search_roundtrip(self):
        store = DualSpeedStore(path=":memory:", embedding_model="none")
        entry = store.add("In-memory test.")
        results = store.search("memory", k=5)
        assert any(r.memory.memory_id == entry.memory_id for r in results)

    def test_independent_connections(self):
        a = DualSpeedStore(path=":memory:", embedding_model="none")
        b = DualSpeedStore(path=":memory:", embedding_model="none")
        a.add("Only in A.")
        assert b.all_fast() == []

    def test_full_lifecycle(self):
        store = DualSpeedStore(path=":memory:", embedding_model="none")
        # add
        entry = store.add("Lifecycle test.", metadata={"tag": "test"})
        mid = entry.memory_id
        # get
        fetched = store.get(mid)
        assert fetched is not None
        # update
        updated = store.update(mid, "Updated lifecycle.")
        assert updated.content == "Updated lifecycle."
        # promote
        promoted = store.promote(mid)
        assert promoted.tier == MemoryTier.SLOW
        # delete
        assert store.delete(mid) is True
        assert store.get(mid) is None


# ---------------------------------------------------------------------------
# Zero-dependency (embedding_model="none")
# ---------------------------------------------------------------------------

class TestZeroDependency:
    def test_add_works_without_embeddings(self):
        store = _make_store(embedding_model="none")
        entry = store.add("No numpy needed.")
        assert entry.memory_id is not None
        assert entry.embedding is None

    def test_search_works_without_embeddings(self):
        store = _make_store(embedding_model="none")
        store.add("trigram search test")
        results = store.search("trigram", k=5)
        assert len(results) > 0

    def test_search_score_without_embeddings(self):
        store = _make_store(embedding_model="none")
        store.add("cats and dogs")
        results = store.search("cats", k=1)
        assert results[0].score >= 0.0

    def test_all_operations_without_embeddings(self):
        store = _make_store(embedding_model="none")
        entry = store.add("All ops.")
        store.get(entry.memory_id)
        store.update(entry.memory_id, "Updated.")
        entry2 = store.add("Second.")
        store.promote(entry2.memory_id)
        store.search("ops", k=5)
        store.all_fast()
        store.all_slow()
        store.stats()
        store.delete(entry.memory_id)
