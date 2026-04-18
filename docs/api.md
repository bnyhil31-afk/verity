# Memory API

`Memory` is the primary public interface. Zero configuration required — it works out of the box with Python's stdlib.

## Constructor

```python
Memory(
    path: str = ":memory:",
    user_id: str = "default",
    capacity: int = 1000,
    embedding_model: str = "auto",
)
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | `":memory:"` | SQLite path. Use a file path for persistence across sessions. |
| `user_id` | `"default"` | Namespace for multi-user scenarios. |
| `capacity` | `1000` | Maximum memories in the fast episodic buffer. |
| `embedding_model` | `"auto"` | Controls semantic search quality. See [Install tiers](#install-tiers). |

```python
from verity import Memory

# In-memory (lost on exit)
m = Memory()

# Persistent
m = Memory(path="~/.verity/memories.db")

# Multi-user
m = Memory(path="app.db", user_id="user-123")
```

## Context manager

Use `with Memory() as m:` to ensure the store is cleanly closed:

```python
with Memory(path="memories.db") as m:
    m.add("Important fact")
    results = m.search("fact")
# Store is closed here
```

---

## Methods

### `add`

```python
add(content: str, metadata: dict = {}, importance: float | None = None) -> str
```

Store a memory. Returns the memory ID.

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `content` | The text to store. |
| `metadata` | Arbitrary key-value pairs attached to the memory. |
| `importance` | Override the auto-computed importance score (0.0–1.0). |

```python
memory_id = m.add("Team standup every weekday at 9am")

# With metadata
memory_id = m.add(
    "The API uses JWT authentication",
    metadata={"source": "onboarding-doc", "project": "backend"},
)

# With explicit importance
memory_id = m.add("Critical: never delete production data", importance=0.95)
```

---

### `search`

```python
search(query: str, k: int = 5) -> list[dict]
```

Retrieve the most relevant memories for a query. Returns up to `k` results, ordered by the Global Workspace competitive selection (importance + temporal weight + relevance).

**Return shape (each item):**

```python
{
    "id": str,           # memory ID
    "content": str,      # stored text
    "confidence": float, # 0.0–1.0, maps to ConfidenceTier
    "importance": float, # 0.0–1.0, prediction-error-based score
    "created_at": str,   # ISO 8601
    "last_accessed": str, # ISO 8601
    "metadata": dict,
}
```

```python
results = m.search("daily schedule")
for r in results:
    print(r["content"], f"  ({r['confidence']:.0%} confidence)")

# Retrieve more results
results = m.search("authentication", k=10)
```

---

### `get`

```python
get(memory_id: str) -> dict | None
```

Retrieve a single memory by ID. Returns `None` if not found. The return dict has the same shape as each item in `search()` results.

```python
entry = m.get(memory_id)
if entry:
    print(entry["content"])
```

---

### `update`

```python
update(memory_id: str, content: str) -> dict
```

Update the content of an existing memory. The reconsolidation engine applies tier-appropriate stability gates before the update proceeds. Returns the updated memory dict.

```python
updated = m.update(memory_id, "Standup moved to 10am on Fridays")
print(updated["content"])
```

---

### `delete`

```python
delete(memory_id: str) -> bool
```

Permanently delete a memory (GDPR Article 17 erasure). Returns `True` if deleted, `False` if not found.

```python
deleted = m.delete(memory_id)
```

---

### `consolidate`

```python
consolidate() -> dict
```

Run the sleep consolidation cycle: decay → prune → abstract. Returns a summary of what happened.

**Return shape:**

```python
{
    "decayed": int,    # memories whose strength was reduced
    "pruned": int,     # memories removed (below threshold)
    "abstracted": int, # clusters merged into abstractions
    "duration_ms": float,
}
```

```python
result = m.consolidate()
print(f"Pruned {result['pruned']} stale memories")
```

---

### `export`

```python
export(format: str = "json") -> str
```

Export all memories as a portable data package (GDPR Article 20 portability). Returns a JSON string.

```python
data = m.export()
with open("my_memories.json", "w") as f:
    f.write(data)
```

---

## Async variants

Every method has an async equivalent prefixed with `a`:

```python
import asyncio
from verity import Memory

async def main():
    m = Memory()
    memory_id = await m.aadd("Async memory")
    results = await m.asearch("memory")
    entry = await m.aget(memory_id)
    updated = await m.aupdate(memory_id, "Updated content")
    deleted = await m.adelete(memory_id)
    result = await m.aconsolidate()
    data = await m.aexport()

asyncio.run(main())
```

---

## Install tiers

The `embedding_model` parameter controls which search backend is used:

| Value | Requires | Search quality | Speed |
|-------|----------|---------------|-------|
| `"none"` | stdlib only | Keyword (BM25-style) | Fastest |
| `"auto"` | `model2vec` if available, else keyword | Semantic if installed | Auto |
| `"model2vec"` | `pip install "veritycog[cognitive]"` | Semantic, fast | Fast |
| `"sentence-transformers"` | `pip install sentence-transformers` | Semantic, higher quality | Slower |

```bash
# Zero dependencies — keyword search only
pip install veritycog

# Recommended — semantic search, fast
pip install "veritycog[cognitive]"
```

With `embedding_model="auto"` (the default), Verity automatically uses the best available backend. No code changes are needed when upgrading tiers.
