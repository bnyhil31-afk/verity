"""
verity.cognitive.store
======================
Dual-Speed Store: SQLite-backed Complementary Learning Systems architecture.

Fast buffer (hippocampal): capacity-limited ring buffer of recent memories.
Slow store (neocortical): promoted, consolidated, persistent memories.

Zero-dependency path: stdlib sqlite3 + json only.
Optional: numpy (embeddings + cosine search), model2vec, sentence-transformers.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from verity.cognitive.types import (
    ConfidenceTier,
    MemoryEntry,
    MemoryTier,
    RetrievalResult,
)

# ---------------------------------------------------------------------------
# Optional numpy — detected once at import time
# ---------------------------------------------------------------------------
try:
    import numpy as _np  # type: ignore[import-untyped]
    _HAS_NUMPY = True
except ImportError:
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------
_COLUMNS = (
    "memory_id",
    "content",
    "user_id",
    "tier",
    "confidence_tier",
    "importance",
    "strength",
    "created_at",
    "last_accessed",
    "access_count",
    "source_count",
    "alpha",
    "beta",
    "metadata",
    "embedding",
)

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS {table} (
    memory_id       TEXT PRIMARY KEY,
    content         TEXT    NOT NULL,
    user_id         TEXT    NOT NULL,
    tier            TEXT    NOT NULL,
    confidence_tier TEXT    NOT NULL DEFAULT 'labile',
    importance      REAL    NOT NULL DEFAULT 0.5,
    strength        REAL    NOT NULL DEFAULT 1.0,
    created_at      TEXT    NOT NULL,
    last_accessed   TEXT    NOT NULL,
    access_count    INTEGER NOT NULL DEFAULT 0,
    source_count    INTEGER NOT NULL DEFAULT 1,
    alpha           REAL    NOT NULL DEFAULT 2.0,
    beta            REAL    NOT NULL DEFAULT 1.0,
    metadata        TEXT    NOT NULL DEFAULT '{{}}',
    embedding       BLOB
)"""

_SELECT_COLS = ", ".join(_COLUMNS)

# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(UTC)


def _to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)


# ---------------------------------------------------------------------------
# Trigram fallback (no embeddings)
# ---------------------------------------------------------------------------

def _trigrams(text: str) -> set[str]:
    t = text.lower()
    if len(t) < 3:
        return {t} if t else set()
    return {t[i : i + 3] for i in range(len(t) - 2)}


def _trigram_score(query: str, content: str) -> float:
    q_grams = _trigrams(query)
    if not q_grams:
        return 0.0
    c_grams = _trigrams(content)
    return len(q_grams & c_grams) / len(q_grams)


# ---------------------------------------------------------------------------
# Embedding / cosine helpers
# ---------------------------------------------------------------------------

def _cosine(a: Any, b: Any) -> float:
    if not _HAS_NUMPY:
        return 0.0
    dot = _np.dot(a, b)
    norm = float(_np.linalg.norm(a)) * float(_np.linalg.norm(b))
    if norm == 0.0:
        return 0.0
    return float(dot / norm)


def _embed_to_blob(arr: Any) -> bytes:
    return arr.astype(_np.float32).tobytes()


def _blob_to_embed(blob: bytes | None) -> Any:
    if not blob or not _HAS_NUMPY:
        return None
    return _np.frombuffer(blob, dtype=_np.float32)


# ---------------------------------------------------------------------------
# Row → MemoryEntry
# ---------------------------------------------------------------------------

def _row_to_entry(row: tuple[Any, ...]) -> MemoryEntry:
    (
        memory_id, content, user_id, tier, confidence_tier,
        importance, strength, created_at, last_accessed,
        access_count, source_count, alpha, beta, metadata_str, emb_blob,
    ) = row

    embedding: list[float] | None = None
    if emb_blob and _HAS_NUMPY:
        arr = _blob_to_embed(emb_blob)
        if arr is not None:
            embedding = arr.tolist()

    return MemoryEntry(
        memory_id=memory_id,
        content=content,
        user_id=user_id,
        tier=MemoryTier(tier),
        confidence_tier=ConfidenceTier(confidence_tier),
        importance=importance,
        strength=strength,
        created_at=_from_iso(created_at),
        last_accessed=_from_iso(last_accessed),
        access_count=access_count,
        source_count=source_count,
        alpha=alpha,
        beta=beta,
        metadata=json.loads(metadata_str),
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# DualSpeedStore
# ---------------------------------------------------------------------------

class DualSpeedStore:
    """
    Complementary Learning Systems theory as a Python class.

    Fast buffer (hippocampal): SQLite table, capacity-limited ring buffer.
    Slow store (neocortical): SQLite table, promoted consolidated memories.

    Both tables share one .db file. Default: ~/.verity/memory.db
    In-memory option for testing: path=":memory:"

    Embedding is optional. If numpy is available, embeddings are computed
    and stored as BLOB. If not, semantic search falls back to trigram-based
    keyword matching scored by recency.
    """

    def __init__(
        self,
        path: str = "~/.verity/memory.db",
        fast_capacity: int = 500,
        embedding_model: str = "auto",
        user_id: str = "default",
    ) -> None:
        self._user_id = user_id
        self._fast_capacity = fast_capacity
        self._embedding_model_name = embedding_model

        if path == ":memory:":
            self._db_path = ":memory:"
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            resolved = Path(path).expanduser().resolve()
            resolved.parent.mkdir(parents=True, exist_ok=True)
            self._db_path = str(resolved)
            self._conn = sqlite3.connect(str(resolved), check_same_thread=False)

        self._create_tables()
        self._embed_model = self._detect_embedding_model(embedding_model)

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _create_tables(self) -> None:
        cur = self._conn.cursor()
        cur.execute(_CREATE_TABLE_SQL.format(table="fast_memories"))
        cur.execute(_CREATE_TABLE_SQL.format(table="slow_memories"))
        self._conn.commit()

    def _detect_embedding_model(self, model_name: str) -> Any:
        if model_name == "none" or not _HAS_NUMPY:
            return None
        if model_name in ("auto", "model2vec"):
            try:
                from model2vec import StaticModel  # type: ignore[import-untyped]
                return StaticModel.from_pretrained("minishlab/potion-base-8M")
            except Exception:
                pass
        if model_name in ("auto", "sentence-transformers"):
            try:
                from sentence_transformers import (
                    SentenceTransformer,  # type: ignore[import-untyped]
                )
                return SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                pass
        return None

    def _compute_embedding(self, text: str) -> Any:
        if self._embed_model is None or not _HAS_NUMPY:
            return None
        try:
            result = self._embed_model.encode(text)
            return _np.array(result, dtype=_np.float32)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal DB helpers
    # ------------------------------------------------------------------

    def _insert_entry(self, table: str, entry: MemoryEntry) -> None:
        emb_blob: bytes | None = None
        if entry.embedding is not None and _HAS_NUMPY:
            emb_blob = _embed_to_blob(_np.array(entry.embedding, dtype=_np.float32))

        self._conn.execute(
            f"""INSERT OR REPLACE INTO {table}
            ({_SELECT_COLS})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry.memory_id,
                entry.content,
                entry.user_id,
                str(entry.tier),
                str(entry.confidence_tier),
                entry.importance,
                entry.strength,
                _to_iso(entry.created_at),
                _to_iso(entry.last_accessed),
                entry.access_count,
                entry.source_count,
                entry.alpha,
                entry.beta,
                json.dumps(entry.metadata),
                emb_blob,
            ),
        )
        self._conn.commit()

    def _fetch_row(self, table: str, memory_id: str) -> tuple[Any, ...] | None:
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT {_SELECT_COLS} FROM {table} WHERE memory_id = ?",
            (memory_id,),
        )
        return cur.fetchone()

    def _find_table_and_row(
        self, memory_id: str
    ) -> tuple[str, tuple[Any, ...]] | None:
        for table in ("fast_memories", "slow_memories"):
            row = self._fetch_row(table, memory_id)
            if row is not None:
                return (table, row)
        return None

    def _evict_if_needed(self) -> None:
        """Remove the lowest-importance fast entry when over capacity."""
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM fast_memories WHERE user_id = ?",
            (self._user_id,),
        )
        count: int = cur.fetchone()[0]
        while count > self._fast_capacity:
            cur.execute(
                """SELECT memory_id FROM fast_memories
                   WHERE user_id = ?
                   ORDER BY importance ASC LIMIT 1""",
                (self._user_id,),
            )
            victim = cur.fetchone()
            if victim is None:
                break
            cur.execute(
                "DELETE FROM fast_memories WHERE memory_id = ?",
                (victim[0],),
            )
            self._conn.commit()
            count -= 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        importance: float | None = None,
        tier: MemoryTier = MemoryTier.FAST,
        _embedding: list[float] | None = None,
    ) -> MemoryEntry:
        """
        Store a memory. Returns the new MemoryEntry.

        tier=MemoryTier.FAST (default): insert into fast buffer, apply capacity check.
        tier=MemoryTier.SLOW: insert directly into slow store, skip capacity check.

        _embedding: if provided, store this pre-computed embedding instead of
            running the embedding model. Used by consolidation and tests.
        importance: if provided, use this value instead of the default 0.5.
        """
        if metadata is None:
            metadata = {}
        now = _now()
        memory_id = str(uuid.uuid4())

        # Resolve embedding: prefer pre-computed, else compute, else None
        if _embedding is not None:
            embedding: list[float] | None = _embedding
        else:
            emb_arr = self._compute_embedding(content)
            embedding = emb_arr.tolist() if emb_arr is not None else None

        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            user_id=self._user_id,
            tier=tier,
            confidence_tier=ConfidenceTier.LABILE,
            importance=importance if importance is not None else 0.5,
            strength=1.0,
            created_at=now,
            last_accessed=now,
            access_count=0,
            source_count=1,
            alpha=2.0,
            beta=1.0,
            metadata=metadata,
            embedding=embedding,
        )

        table = "slow_memories" if tier == MemoryTier.SLOW else "fast_memories"
        self._insert_entry(table, entry)
        if tier == MemoryTier.FAST:
            self._evict_if_needed()
        return entry

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        """
        Semantic search across both fast and slow stores.
        Returns up to k results sorted by score descending.
        Uses cosine similarity when embeddings are available,
        else trigram overlap scored by recency.
        """
        cur = self._conn.cursor()
        all_rows: list[tuple[Any, ...]] = []
        for table in ("fast_memories", "slow_memories"):
            cur.execute(
                f"SELECT {_SELECT_COLS} FROM {table} WHERE user_id = ?",
                (self._user_id,),
            )
            all_rows.extend(cur.fetchall())

        if not all_rows:
            return []

        query_emb = (
            self._compute_embedding(query)
            if self._embed_model is not None
            else None
        )

        scored: list[tuple[float, MemoryEntry]] = []
        now = _now()

        for row in all_rows:
            entry = _row_to_entry(row)
            if query_emb is not None and entry.embedding is not None:
                emb_arr = _np.array(entry.embedding, dtype=_np.float32)
                score = _cosine(query_emb, emb_arr)
            else:
                trig = _trigram_score(query, entry.content)
                hours_old = (now - entry.last_accessed).total_seconds() / 3600.0
                recency = max(0.0, 1.0 - hours_old / (24.0 * 365.0))
                score = 0.7 * trig + 0.3 * recency
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [
            RetrievalResult(memory=entry, score=score, position=pos)
            for pos, (score, entry) in enumerate(scored[:k], start=1)
        ]

    def get(self, memory_id: str) -> MemoryEntry | None:
        """
        Fetch one memory by ID.
        Increments access_count and updates last_accessed on every call.
        Returns None if not found.
        """
        result = self._find_table_and_row(memory_id)
        if result is None:
            return None
        table, row = result
        entry = _row_to_entry(row)
        now = _now()
        self._conn.execute(
            f"""UPDATE {table}
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE memory_id = ?""",
            (_to_iso(now), memory_id),
        )
        self._conn.commit()
        entry.access_count += 1
        entry.last_accessed = now
        return entry

    def update(self, memory_id: str, new_content: str) -> MemoryEntry:
        """
        Update memory content and last_accessed. Increments access_count.
        Does NOT change confidence_tier or alpha/beta — that is
        ReconsolidationEngine's responsibility.
        Raises KeyError if memory_id is not found.
        """
        result = self._find_table_and_row(memory_id)
        if result is None:
            raise KeyError(f"Memory not found: {memory_id!r}")
        table, _ = result
        now = _now()

        emb_arr = self._compute_embedding(new_content)
        if emb_arr is not None:
            emb_blob = _embed_to_blob(emb_arr)
            self._conn.execute(
                f"""UPDATE {table}
                    SET content = ?,
                        last_accessed = ?,
                        access_count = access_count + 1,
                        embedding = ?
                    WHERE memory_id = ?""",
                (new_content, _to_iso(now), emb_blob, memory_id),
            )
        else:
            self._conn.execute(
                f"""UPDATE {table}
                    SET content = ?,
                        last_accessed = ?,
                        access_count = access_count + 1
                    WHERE memory_id = ?""",
                (new_content, _to_iso(now), memory_id),
            )
        self._conn.commit()

        row = self._fetch_row(table, memory_id)
        assert row is not None
        return _row_to_entry(row)

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory. Returns True if found and deleted, False otherwise.
        """
        result = self._find_table_and_row(memory_id)
        if result is None:
            return False
        table, _ = result
        self._conn.execute(
            f"DELETE FROM {table} WHERE memory_id = ?",
            (memory_id,),
        )
        self._conn.commit()
        return True

    def all_fast(self) -> list[MemoryEntry]:
        """Return all entries from the fast buffer."""
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT {_SELECT_COLS} FROM fast_memories WHERE user_id = ?",
            (self._user_id,),
        )
        return [_row_to_entry(row) for row in cur.fetchall()]

    def all_slow(self) -> list[MemoryEntry]:
        """Return all entries from the slow store."""
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT {_SELECT_COLS} FROM slow_memories WHERE user_id = ?",
            (self._user_id,),
        )
        return [_row_to_entry(row) for row in cur.fetchall()]

    def promote(self, memory_id: str) -> MemoryEntry:
        """
        Move a memory from the fast buffer to the slow store.
        Raises KeyError if not found in fast buffer.
        """
        row = self._fetch_row("fast_memories", memory_id)
        if row is None:
            raise KeyError(f"Memory not found in fast store: {memory_id!r}")
        entry = _row_to_entry(row)
        entry.tier = MemoryTier.SLOW
        self._insert_entry("slow_memories", entry)
        self._conn.execute(
            "DELETE FROM fast_memories WHERE memory_id = ?",
            (memory_id,),
        )
        self._conn.commit()
        return entry

    def compute_embedding(self, content: str) -> list[float] | None:
        """
        Compute and return an embedding for content using the store's
        configured embedding model.

        Returns None if embedding_model="none" or no model is available.
        Does NOT store anything — pure computation only.
        """
        arr = self._compute_embedding(content)
        if arr is None:
            return None
        return arr.tolist()

    def close(self) -> None:
        """
        Close the SQLite connection cleanly.
        Safe to call multiple times (no-op if already closed).
        """
        try:
            self._conn.close()
        except Exception:
            pass

    def update_entry(self, entry: MemoryEntry) -> None:
        """
        Overwrite ALL persisted fields for an existing memory_id in
        whichever table (fast or slow) contains it.

        Fields updated: content, confidence_tier, importance, strength,
        last_accessed, access_count, source_count, alpha, beta,
        metadata, embedding.

        Raises KeyError if memory_id not found in either table.
        """
        result = self._find_table_and_row(entry.memory_id)
        if result is None:
            raise KeyError(f"Memory not found: {entry.memory_id!r}")
        table, _ = result

        emb_blob: bytes | None = None
        if entry.embedding is not None and _HAS_NUMPY:
            emb_blob = _embed_to_blob(_np.array(entry.embedding, dtype=_np.float32))

        self._conn.execute(
            f"""UPDATE {table}
                SET content          = ?,
                    confidence_tier  = ?,
                    importance       = ?,
                    strength         = ?,
                    last_accessed    = ?,
                    access_count     = ?,
                    source_count     = ?,
                    alpha            = ?,
                    beta             = ?,
                    metadata         = ?,
                    embedding        = ?
                WHERE memory_id = ?""",
            (
                entry.content,
                str(entry.confidence_tier),
                entry.importance,
                entry.strength,
                _to_iso(entry.last_accessed),
                entry.access_count,
                entry.source_count,
                entry.alpha,
                entry.beta,
                json.dumps(entry.metadata),
                emb_blob,
                entry.memory_id,
            ),
        )
        self._conn.commit()

    def stats(self) -> dict[str, Any]:
        """
        Return store statistics.
        Keys: fast_count, slow_count, fast_capacity,
              embedding_model, db_path, total_count
        """
        cur = self._conn.cursor()
        cur.execute(
            "SELECT COUNT(*) FROM fast_memories WHERE user_id = ?",
            (self._user_id,),
        )
        fast_count: int = cur.fetchone()[0]
        cur.execute(
            "SELECT COUNT(*) FROM slow_memories WHERE user_id = ?",
            (self._user_id,),
        )
        slow_count: int = cur.fetchone()[0]

        embed_name = (
            type(self._embed_model).__name__
            if self._embed_model is not None
            else "none"
        )

        return {
            "fast_count": fast_count,
            "slow_count": slow_count,
            "fast_capacity": self._fast_capacity,
            "embedding_model": embed_name,
            "db_path": self._db_path,
            "total_count": fast_count + slow_count,
        }
