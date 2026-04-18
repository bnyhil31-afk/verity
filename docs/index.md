# Verity

A cognitive memory system for AI agents — neuroscience-inspired, zero dependencies by default.

## The three problems Verity solves

These were unsolved in production memory systems before Verity:

1. **Reconsolidation stability** — memories update on access without runaway drift. A 4-tier Bayesian protection system controls when and how memories accept new information.

2. **Sleep consolidation** — an offline cycle (decay → prune → abstract) removes stale memories and merges similar episodes into abstractions.

3. **Tiered temporal weighting** — recency signal auto-graduates from exponential decay to Bayesian renewal to Hawkes process as access history grows.

## Install

```bash
pip install veritycog                    # stdlib only
pip install "veritycog[cognitive]"       # + embeddings (recommended)
pip install "veritycog[full]"            # everything
```

## Quickstart

```python
from verity import Memory

m = Memory()                                          # zero config
m.add("Team standup every weekday at 9am")
m.add("The API uses JWT authentication")
m.add("PostgreSQL is the production database")

results = m.search("daily schedule")
for r in results:
    print(r["content"], f"  ({r['confidence']:.0%} confidence)")

m.consolidate()   # sleep cycle — decay, prune, abstract
m.export()        # GDPR Article 20 portability
```

!!! note "Import name"
    The pip package is `veritycog` but the import is `from verity import Memory`.
    The internal module name does not change.

## For AI agents

```python
from verity import Memory

memory = Memory()

def agent_response(user_input: str) -> str:
    context = memory.search(user_input, k=5)
    context_str = "\n".join(r["content"] for r in context)
    # ... call your LLM with context_str ...
    memory.add(f"User asked: {user_input}")
    return response
```

## What's next

- [Concepts](concepts.md) — the neuroscience behind each component
- [Memory API](api.md) — all seven methods with examples
- [Engine API](engine.md) — graph traversal for advanced use cases
- [Benchmarks](benchmarks.md) — measured performance and claim validation
