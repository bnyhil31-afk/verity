"""
tests/test_memory.py
====================
Tests for verity.memory.Memory — Phase H.

All tests use Memory(path=":memory:", embedding_model="none") so no files
are written to disk and no ML libraries are required.
"""

from __future__ import annotations

import json

import pytest

from verity.memory import Memory

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def mem() -> Memory:
    """Fresh in-memory store, no embeddings, default user."""
    return Memory(path=":memory:", embedding_model="none")


# ---------------------------------------------------------------------------
# TestMemoryAdd
# ---------------------------------------------------------------------------

class TestMemoryAdd:
    def test_returns_nonempty_string(self, mem: Memory) -> None:
        mid = mem.add("some content")
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_two_adds_different_ids(self, mem: Memory) -> None:
        id1 = mem.add("same content")
        id2 = mem.add("same content")
        assert id1 != id2

    def test_metadata_stored(self, mem: Memory) -> None:
        mid = mem.add("content with meta", metadata={"key": "value"})
        result = mem.get(mid)
        assert result is not None
        assert result["metadata"] == {"key": "value"}

    def test_explicit_importance_stored(self, mem: Memory) -> None:
        mid = mem.add("important content", importance=0.8)
        result = mem.get(mid)
        assert result is not None
        assert abs(result["importance"] - 0.8) < 1e-4


# ---------------------------------------------------------------------------
# TestMemorySearch
# ---------------------------------------------------------------------------

class TestMemorySearch:
    def test_empty_store_returns_empty_list(self, mem: Memory) -> None:
        assert mem.search("anything") == []

    def test_search_returns_list_of_dicts(self, mem: Memory) -> None:
        mem.add("a memory about cats")
        results = mem.search("cats")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, dict)

    def test_every_dict_has_required_keys(self, mem: Memory) -> None:
        mem.add("a memory about dogs")
        results = mem.search("dogs")
        required_keys = {"id", "content", "score", "metadata", "confidence", "strength"}
        for r in results:
            assert set(r.keys()) == required_keys

    def test_k_limits_results(self, mem: Memory) -> None:
        for i in range(10):
            mem.add(f"memory number {i}")
        results = mem.search("memory", k=2)
        assert len(results) <= 2


# ---------------------------------------------------------------------------
# TestMemoryGet
# ---------------------------------------------------------------------------

class TestMemoryGet:
    def test_unknown_id_returns_none(self, mem: Memory) -> None:
        assert mem.get("unknown-id") is None

    def test_returns_dict_after_add(self, mem: Memory) -> None:
        mid = mem.add("hello world")
        result = mem.get(mid)
        assert result is not None
        assert isinstance(result, dict)

    def test_returned_dict_has_all_keys(self, mem: Memory) -> None:
        mid = mem.add("test memory")
        result = mem.get(mid)
        assert result is not None
        expected_keys = {
            "id", "content", "metadata", "confidence", "confidence_tier",
            "strength", "importance", "access_count", "source_count",
            "created_at", "last_accessed", "tier",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_get_twice_increments_access_count(self, mem: Memory) -> None:
        mid = mem.add("access count test")
        r1 = mem.get(mid)
        assert r1 is not None
        count1 = r1["access_count"]
        r2 = mem.get(mid)
        assert r2 is not None
        assert r2["access_count"] == count1 + 1


# ---------------------------------------------------------------------------
# TestMemoryUpdate
# ---------------------------------------------------------------------------

class TestMemoryUpdate:
    def test_update_changes_content(self, mem: Memory) -> None:
        mid = mem.add("original content")
        mem.update(mid, "updated content")
        result = mem.get(mid)
        assert result is not None
        assert result["content"] == "updated content"

    def test_update_unknown_id_raises_key_error(self, mem: Memory) -> None:
        with pytest.raises(KeyError):
            mem.update("unknown-id", "new content")

    def test_update_returns_dict_with_updated_content(self, mem: Memory) -> None:
        mid = mem.add("before")
        result = mem.update(mid, "after")
        assert isinstance(result, dict)
        assert result["content"] == "after"

    def test_returned_dict_has_all_expected_keys(self, mem: Memory) -> None:
        mid = mem.add("base content")
        result = mem.update(mid, "new content")
        expected_keys = {
            "id", "content", "metadata", "confidence", "confidence_tier",
            "strength", "importance", "access_count", "source_count",
            "created_at", "last_accessed", "tier",
        }
        assert expected_keys.issubset(set(result.keys()))


# ---------------------------------------------------------------------------
# TestMemoryDelete
# ---------------------------------------------------------------------------

class TestMemoryDelete:
    def test_delete_existing_returns_true(self, mem: Memory) -> None:
        mid = mem.add("to be deleted")
        assert mem.delete(mid) is True

    def test_delete_unknown_returns_false(self, mem: Memory) -> None:
        assert mem.delete("unknown-id") is False

    def test_get_returns_none_after_delete(self, mem: Memory) -> None:
        mid = mem.add("will be deleted")
        mem.delete(mid)
        assert mem.get(mid) is None

    def test_search_does_not_return_deleted_memory(self, mem: Memory) -> None:
        mid = mem.add("special unique phrase xyzzy")
        mem.delete(mid)
        results = mem.search("special unique phrase xyzzy")
        ids = [r["id"] for r in results]
        assert mid not in ids


# ---------------------------------------------------------------------------
# TestMemoryConsolidate
# ---------------------------------------------------------------------------

class TestMemoryConsolidate:
    def test_consolidate_returns_dict(self, mem: Memory) -> None:
        result = mem.consolidate()
        assert isinstance(result, dict)

    def test_consolidate_has_required_keys(self, mem: Memory) -> None:
        result = mem.consolidate()
        expected_keys = {"decayed", "pruned", "merged", "abstractions", "duration_seconds"}
        assert set(result.keys()) == expected_keys

    def test_consolidate_empty_store_zeros(self, mem: Memory) -> None:
        result = mem.consolidate()
        assert result["decayed"] == 0
        assert result["pruned"] == 0
        assert result["merged"] == 0
        assert result["abstractions"] == 0
        assert result["duration_seconds"] > 0.0

    def test_consolidate_values_nonnegative(self, mem: Memory) -> None:
        for i in range(3):
            mem.add(f"memory {i}")
        result = mem.consolidate()
        for key, val in result.items():
            assert val >= 0, f"{key} should be non-negative, got {val}"


# ---------------------------------------------------------------------------
# TestMemoryExport
# ---------------------------------------------------------------------------

class TestMemoryExport:
    def test_export_empty_returns_empty_json_array(self, mem: Memory) -> None:
        output = mem.export()
        assert output.strip() == "[]"

    def test_export_returns_valid_json(self, mem: Memory) -> None:
        for i in range(3):
            mem.add(f"memory {i}")
        output = mem.export()
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    def test_export_json_dicts_have_id_and_content(self, mem: Memory) -> None:
        mem.add("alpha")
        mem.add("beta")
        mem.add("gamma")
        parsed = json.loads(mem.export())
        for item in parsed:
            assert "id" in item
            assert "content" in item

    def test_export_csv_has_header(self, mem: Memory) -> None:
        output = mem.export(format="csv")
        first_line = output.splitlines()[0]
        assert "id" in first_line
        assert "content" in first_line

    def test_export_unknown_format_raises_value_error(self, mem: Memory) -> None:
        with pytest.raises(ValueError):
            mem.export(format="unknown")


# ---------------------------------------------------------------------------
# TestMemoryAsyncVariants
# ---------------------------------------------------------------------------

class TestMemoryAsyncVariants:
    @pytest.mark.asyncio
    async def test_aadd_returns_nonempty_string(self) -> None:
        async with Memory(path=":memory:", embedding_model="none") as m:
            mid = await m.aadd("async content")
            assert isinstance(mid, str)
            assert len(mid) > 0

    @pytest.mark.asyncio
    async def test_asearch_returns_list(self) -> None:
        async with Memory(path=":memory:", embedding_model="none") as m:
            await m.aadd("async search test")
            results = await m.asearch("search")
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_aget_unknown_returns_none(self) -> None:
        async with Memory(path=":memory:", embedding_model="none") as m:
            result = await m.aget("unknown-id")
            assert result is None

    @pytest.mark.asyncio
    async def test_adelete_unknown_returns_false(self) -> None:
        async with Memory(path=":memory:", embedding_model="none") as m:
            result = await m.adelete("unknown-id")
            assert result is False

    @pytest.mark.asyncio
    async def test_aconsolidate_returns_dict_with_decayed(self) -> None:
        async with Memory(path=":memory:", embedding_model="none") as m:
            result = await m.aconsolidate()
            assert isinstance(result, dict)
            assert "decayed" in result

    @pytest.mark.asyncio
    async def test_aexport_returns_string_starting_with_bracket(self) -> None:
        async with Memory(path=":memory:", embedding_model="none") as m:
            output = await m.aexport()
            assert isinstance(output, str)
            assert output.strip().startswith("[")


# ---------------------------------------------------------------------------
# TestMemoryContextManager
# ---------------------------------------------------------------------------

class TestMemoryContextManager:
    def test_sync_context_manager_works(self) -> None:
        with Memory(path=":memory:", embedding_model="none") as m:
            mid = m.add("inside context manager")
            assert isinstance(mid, str)
            assert len(mid) > 0

    @pytest.mark.asyncio
    async def test_async_context_manager_works(self) -> None:
        async with Memory(path=":memory:", embedding_model="none") as m:
            mid = await m.aadd("inside async context manager")
            assert isinstance(mid, str)
            assert len(mid) > 0
