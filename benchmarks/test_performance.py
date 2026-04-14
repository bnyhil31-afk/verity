import os
import tracemalloc
from itertools import cycle

from verity.memory import Memory


def _populate_memory(m: Memory, n: int) -> None:
    """Add n entries to a Memory instance with varied content."""
    topics = ["preferences", "schedule", "technical", "contacts", "project"]
    for i in range(n):
        topic = topics[i % len(topics)]
        m.add(f"{topic}: synthetic benchmark entry number {i}")


class TestAddLatency:
    """
    Benchmark Memory.add() at two store sizes using benchmark.pedantic().

    IMPORTANT: Each round creates a FRESH database via a counter-based unique
    path. Without this, rounds 2+ open the existing db (same path), growing
    the store across rounds and making later rounds artificially slower.

    With unique paths, every round measures add() on a store of exactly N
    entries, giving a clean, unbiased mean.

    Targets (embedding_model='none'):
      Mean < 50ms at 100 entries.
      Mean < 50ms at 1K entries.
    """

    def test_add_at_100_entries(self, benchmark, tmp_path):
        counter = [0]

        def setup():
            counter[0] += 1
            db = str(tmp_path / f"add_100_{counter[0]}.db")
            m = Memory(path=db, embedding_model="none")
            _populate_memory(m, 100)
            return (m,), {}

        benchmark.pedantic(
            lambda m: m.add("benchmark: new entry for timing"),
            setup=setup,
            rounds=20,
            warmup_rounds=2,
        )
        assert benchmark.stats["mean"] < 0.050, (
            f"add() mean={benchmark.stats['mean']*1000:.1f}ms "
            f"at 100 entries (target < 50ms)"
        )

    def test_add_at_1k_entries(self, benchmark, tmp_path):
        counter = [0]

        def setup():
            counter[0] += 1
            db = str(tmp_path / f"add_1k_{counter[0]}.db")
            m = Memory(path=db, embedding_model="none")
            _populate_memory(m, 1_000)
            return (m,), {}

        benchmark.pedantic(
            lambda m: m.add("benchmark: new entry for timing"),
            setup=setup,
            rounds=10,
            warmup_rounds=2,
        )
        assert benchmark.stats["mean"] < 0.050, (
            f"add() mean={benchmark.stats['mean']*1000:.1f}ms "
            f"at 1K entries (target < 50ms)"
        )


class TestSearchLatency:
    """
    Benchmark Memory.search() at two store sizes.
    Store is created ONCE and reused across all calls — correct here because
    search() does not modify the store between calls.

    Rotates through 5 queries using itertools.cycle so the benchmark
    doesn't cache a single query result.

    Context: Graphiti P95 ~300ms (uses embedding model).
    Target: mean < 100ms at 1K entries (keyword search, no embeddings).
    """

    QUERIES = [
        "preferences", "schedule", "technical", "contacts", "project",
    ]

    def test_search_at_100_entries(self, benchmark, tmp_path):
        db = str(tmp_path / "search_100.db")
        m = Memory(path=db, embedding_model="none")
        _populate_memory(m, 100)
        query_cycle = cycle(self.QUERIES)

        benchmark(lambda: m.search(next(query_cycle), k=5))

        assert benchmark.stats["mean"] < 0.100, (
            f"search() mean={benchmark.stats['mean']*1000:.1f}ms "
            f"at 100 entries (target < 100ms)"
        )

    def test_search_at_1k_entries(self, benchmark, tmp_path):
        db = str(tmp_path / "search_1k.db")
        m = Memory(path=db, embedding_model="none")
        _populate_memory(m, 1_000)
        query_cycle = cycle(self.QUERIES)

        benchmark(lambda: m.search(next(query_cycle), k=5))

        assert benchmark.stats["mean"] < 0.100, (
            f"search() mean={benchmark.stats['mean']*1000:.1f}ms "
            f"at 1K entries (target < 100ms)"
        )

    def test_search_returns_results(self, tmp_path):
        """Sanity check: search returns results with expected keys."""
        db = str(tmp_path / "search_sanity.db")
        m = Memory(path=db, embedding_model="none")
        _populate_memory(m, 50)
        results = m.search("technical", k=5)
        assert len(results) > 0, "search() returned no results on a populated store"
        assert all("content" in r for r in results), (
            "search() result dicts missing 'content' key"
        )


class TestConsolidateLatency:
    """
    Benchmark Memory.consolidate() at two store sizes.

    CRITICAL: Each round MUST use a fresh unique database path.
    consolidate() mutates the store — decay reduces entry strength, prune
    removes entries that fall below the threshold. Without unique paths,
    rounds 2+ benchmark an increasingly empty store, making later rounds
    artificially fast and the mean meaningless.

    abstract_pass() is skipped when embedding_model='none' (no embeddings
    to cluster). This benchmark measures decay_pass() + prune_pass() only.

    Targets: mean < 500ms at 100 entries, mean < 2s at 1K entries.
    """

    def test_consolidate_at_100_entries(self, benchmark, tmp_path):
        counter = [0]

        def setup():
            counter[0] += 1
            db = str(tmp_path / f"cons_100_{counter[0]}.db")
            m = Memory(path=db, embedding_model="none")
            _populate_memory(m, 100)
            return (m,), {}

        benchmark.pedantic(
            lambda m: m.consolidate(),
            setup=setup,
            rounds=5,
            warmup_rounds=1,
        )
        assert benchmark.stats["mean"] < 0.500, (
            f"consolidate() mean={benchmark.stats['mean']*1000:.1f}ms "
            f"at 100 entries (target < 500ms)"
        )

    def test_consolidate_at_1k_entries(self, benchmark, tmp_path):
        counter = [0]

        def setup():
            counter[0] += 1
            db = str(tmp_path / f"cons_1k_{counter[0]}.db")
            m = Memory(path=db, embedding_model="none")
            _populate_memory(m, 1_000)
            return (m,), {}

        benchmark.pedantic(
            lambda m: m.consolidate(),
            setup=setup,
            rounds=3,
            warmup_rounds=1,
        )
        assert benchmark.stats["mean"] < 2.000, (
            f"consolidate() mean={benchmark.stats['mean']*1000:.1f}ms "
            f"at 1K entries (target < 2s)"
        )


class TestMemoryFootprint:
    """
    Measures SQLite file size and peak in-process memory.
    Not a benchmark fixture test.

    File size: use context manager (with Memory(...) as m:) so that
    __exit__ calls _store.close(), which flushes SQLite to disk before
    os.path.getsize() is called.

    tracemalloc: always call stop() + clear_traces() before start() to
    prevent stale measurements from earlier tests or pytest-benchmark
    internals that may already be tracing.
    """

    def test_file_size_at_100_entries(self, tmp_path):
        db = str(tmp_path / "footprint_100.db")
        with Memory(path=db, embedding_model="none") as m:
            _populate_memory(m, 100)
        # __exit__ has called _store.close() — SQLite flushed to disk
        size_kb = os.path.getsize(db) / 1024
        assert size_kb < 200, (
            f"File size at 100 entries: {size_kb:.1f}KB (expected < 200KB)"
        )
        print(f"\n  File size at 100 entries: {size_kb:.1f}KB")

    def test_file_size_at_1k_entries(self, tmp_path):
        db = str(tmp_path / "footprint_1k.db")
        with Memory(path=db, embedding_model="none") as m:
            _populate_memory(m, 1_000)
        size_kb = os.path.getsize(db) / 1024
        assert size_kb < 1024, (
            f"File size at 1K entries: {size_kb:.1f}KB (expected < 1MB)"
        )
        print(f"\n  File size at 1K entries: {size_kb:.1f}KB")

    def test_storage_growth_is_linear(self, tmp_path):
        """
        Storage grows approximately linearly. The 1K:100 size ratio
        should be < 20x (not 100x, which would indicate quadratic growth).
        """
        db_100 = str(tmp_path / "growth_100.db")
        with Memory(path=db_100, embedding_model="none") as m:
            _populate_memory(m, 100)

        db_1k = str(tmp_path / "growth_1k.db")
        with Memory(path=db_1k, embedding_model="none") as m:
            _populate_memory(m, 1_000)

        size_100 = os.path.getsize(db_100)
        size_1k  = os.path.getsize(db_1k)
        ratio = size_1k / max(size_100, 1)

        assert ratio < 20, (
            f"Storage growth ratio {ratio:.1f}x exceeds 20x for 10x entries. "
            f"100→{size_100/1024:.1f}KB, 1K→{size_1k/1024:.1f}KB"
        )
        print(f"\n  Storage: {size_100/1024:.1f}KB → {size_1k/1024:.1f}KB "
              f"({ratio:.1f}x for 10x entries)")

    def test_peak_memory_add_100(self, tmp_path):
        """
        Peak in-process memory during 100 add() calls. Target: < 10MB.
        """
        db = str(tmp_path / "peak_mem.db")
        m = Memory(path=db, embedding_model="none")

        # Always reset tracemalloc before measuring
        tracemalloc.stop()
        tracemalloc.clear_traces()
        tracemalloc.start()

        _populate_memory(m, 100)

        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        m._store.close()

        peak_mb = peak / (1024 * 1024)
        assert peak_mb < 10, (
            f"Peak memory for 100 add() calls: {peak_mb:.2f}MB (target < 10MB)"
        )
        print(f"\n  Peak memory (100 adds): {peak_mb:.2f}MB")
