"""
verity.memory
=============
Bolt-on cognitive memory API for Verity.

Zero config:
    m = Memory()
    m.add("I prefer dark mode")
    results = m.search("preferences")

Custom config:
    m = Memory(
        path="/data/memory.db",
        user_id="alice",
        embedding_model="sentence-transformers",
    )

Context manager:
    with Memory() as m:
        m.add("...")
        results = m.search("...")

Async:
    async with Memory() as m:
        await m.aadd("...")
        results = await m.asearch("...")

Works with or without numpy, sentence-transformers, scipy, torch.
Automatically uses the best available backend.
"""

from __future__ import annotations

import json

from verity.cognitive.consolidation import ConsolidationCycle
from verity.cognitive.reconsolidation import ReconsolidationEngine
from verity.cognitive.scoring import ImportanceScorer
from verity.cognitive.store import DualSpeedStore
from verity.cognitive.temporal import TemporalWeighter
from verity.cognitive.types import MemoryEntry, RetrievalResult
from verity.cognitive.workspace import GlobalWorkspace

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _entry_to_dict(result: RetrievalResult) -> dict:
    """Convert RetrievalResult to the public search result shape."""
    e = result.memory
    return {
        "id":         e.memory_id,
        "content":    e.content,
        "score":      round(result.score, 4),
        "metadata":   e.metadata,
        "confidence": round(e.bayesian_confidence, 4),
        "strength":   round(e.strength, 4),
    }


def _entry_to_dict_full(entry: MemoryEntry) -> dict:
    """Convert MemoryEntry to the public get/update/export shape."""
    return {
        "id":               entry.memory_id,
        "content":          entry.content,
        "metadata":         entry.metadata,
        "confidence":       round(entry.bayesian_confidence, 4),
        "confidence_tier":  entry.confidence_tier,
        "strength":         round(entry.strength, 4),
        "importance":       round(entry.importance, 4),
        "access_count":     entry.access_count,
        "source_count":     entry.source_count,
        "created_at":       entry.created_at.isoformat(),
        "last_accessed":    entry.last_accessed.isoformat(),
        "tier":             entry.tier,
    }


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

class Memory:
    """
    Verity's bolt-on cognitive memory API.

    Zero config:
        m = Memory()                    # SQLite in ~/.verity/memory.db
        m.add("I prefer dark mode")     # Store
        m.search("preferences")         # Retrieve

    Full config:
        m = Memory(
            path="/data/memory.db",
            user_id="alice",
            capacity=500,
            embedding_model="sentence-transformers",
        )

    Works with or without numpy, sentence-transformers, scipy, torch.
    Automatically uses the best available backend.
    """

    def __init__(
        self,
        path: str = "~/.verity/memory.db",
        user_id: str = "default",
        capacity: int = 5,
        embedding_model: str = "auto",
    ) -> None:
        self._store    = DualSpeedStore(
            path=path,
            user_id=user_id,
            embedding_model=embedding_model,
        )
        self._scorer   = ImportanceScorer()
        self._recon    = ReconsolidationEngine()
        self._cycle    = ConsolidationCycle(self._store, self._scorer)
        self._temporal = TemporalWeighter()
        self._capacity = capacity

    # ── Core 7 methods ────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        metadata: dict | None = None,
        importance: float | None = None,
    ) -> str:
        """Store a memory. Returns memory_id (string uuid)."""
        if metadata is None:
            metadata = {}
        entry = self._store.add(content, metadata)
        if importance is not None:
            entry.importance = importance
            self._store.update_entry(entry)
        if entry.embedding is not None:
            self._scorer.update_centroid(entry.embedding)
        return entry.memory_id

    def search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict]:
        """
        Semantic search with cognitive workspace selection.
        Retrieves k*4 candidates, then selects best k via GlobalWorkspace.
        Returns list of dicts with keys:
            id, content, score, metadata, confidence, strength
        """
        raw = self._store.search(query, k=k * 4)
        if not raw:
            return []

        goal_embedding = self._store.compute_embedding(query)

        ws = GlobalWorkspace(
            capacity=k,
            scorer=self._scorer,
            temporal=self._temporal,
        )
        selected = ws.select(raw, goal=query, goal_embedding=goal_embedding)
        return [_entry_to_dict(r) for r in selected]

    def get(self, memory_id: str) -> dict | None:
        """Fetch one memory by ID. Returns dict or None."""
        entry = self._store.get(memory_id)
        if entry is None:
            return None
        return _entry_to_dict_full(entry)

    def update(self, memory_id: str, content: str) -> dict:
        """
        Update memory content with reconsolidation rules applied.
        Raises KeyError if memory_id not found.
        """
        current = self._store.get(memory_id)
        if current is None:
            raise KeyError(f"Memory {memory_id!r} not found")

        prediction_error = 0.3
        new_embedding = self._store.compute_embedding(content)
        if new_embedding is not None and current.embedding is not None:
            try:
                import numpy as np  # noqa: PLC0415
                old = np.array(current.embedding, dtype=np.float32)
                new = np.array(new_embedding, dtype=np.float32)
                norm = np.linalg.norm(old) * np.linalg.norm(new)
                if norm > 1e-8:
                    cos_sim = float(np.dot(old, new) / norm)
                    prediction_error = 1.0 - max(0.0, cos_sim)
            except ImportError:
                pass

        updated = self._recon.update(current, content, prediction_error)

        if new_embedding is not None:
            updated.embedding = new_embedding

        self._store.update_entry(updated)
        return _entry_to_dict_full(updated)

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory. Returns True if deleted, False if not found.
        GDPR Article 17 compliant.
        """
        return self._store.delete(memory_id)

    def consolidate(self) -> dict:
        """
        Run one sleep consolidation cycle (decay → prune → abstract).
        Returns summary dict with keys:
            decayed, pruned, merged, abstractions, duration_seconds
        """
        result = self._cycle.run()
        return {
            "decayed":          result.memories_decayed,
            "pruned":           result.memories_pruned,
            "merged":           result.memories_merged,
            "abstractions":     result.abstractions_created,
            "duration_seconds": result.duration_seconds,
        }

    def export(self, format: str = "json") -> str:
        """
        Export all memories. GDPR Article 20 compliant.
        format: "json" | "csv"
        Returns a string.
        """
        all_entries = self._store.all_fast() + self._store.all_slow()

        if format == "json":
            data = [_entry_to_dict_full(e) for e in all_entries]
            return json.dumps(data, indent=2, default=str)

        elif format == "csv":
            import csv
            import io
            import json as _json  # noqa: PLC0415
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=[
                "id", "content", "created_at", "last_accessed",
                "confidence", "strength", "metadata",
            ])
            writer.writeheader()
            for e in all_entries:
                writer.writerow({
                    "id":            e.memory_id,
                    "content":       e.content,
                    "created_at":    e.created_at.isoformat(),
                    "last_accessed": e.last_accessed.isoformat(),
                    "confidence":    round(e.bayesian_confidence, 4),
                    "strength":      round(e.strength, 4),
                    "metadata":      _json.dumps(e.metadata),
                })
            return buf.getvalue()

        else:
            raise ValueError(
                f"Unknown format {format!r}. Valid values: 'json', 'csv'."
            )

    # ── Async variants ────────────────────────────────────────────────────

    async def aadd(
        self,
        content: str,
        metadata: dict | None = None,
        importance: float | None = None,
    ) -> str:
        import asyncio
        return await asyncio.to_thread(self.add, content, metadata, importance)

    async def asearch(self, query: str, k: int = 5) -> list[dict]:
        import asyncio
        return await asyncio.to_thread(self.search, query, k)

    async def aget(self, memory_id: str) -> dict | None:
        import asyncio
        return await asyncio.to_thread(self.get, memory_id)

    async def aupdate(self, memory_id: str, content: str) -> dict:
        import asyncio
        return await asyncio.to_thread(self.update, memory_id, content)

    async def adelete(self, memory_id: str) -> bool:
        import asyncio
        return await asyncio.to_thread(self.delete, memory_id)

    async def aconsolidate(self) -> dict:
        import asyncio
        return await asyncio.to_thread(self.consolidate)

    async def aexport(self, format: str = "json") -> str:
        import asyncio
        return await asyncio.to_thread(self.export, format)

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *args: object) -> None:
        self._store.close()

    async def __aenter__(self) -> Memory:
        return self

    async def __aexit__(self, *args: object) -> None:
        self._store.close()
