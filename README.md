# Verity

[![PyPI version](https://badge.fury.io/py/veritycog.svg)](https://badge.fury.io/py/veritycog)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://pypi.org/project/veritycog/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/bnyhil31-afk/verity/actions/workflows/ci.yml/badge.svg)](https://github.com/bnyhil31-afk/verity/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-veritycog-blue)](https://bnyhil31-afk.github.io/verity/)

A cognitive memory system for AI agents and applications — inspired by how the brain actually stores, recalls, and forgets.

## What it does

AI memory systems treat memory as a database: store a string, retrieve a string. Verity treats memory as cognition. It uses a dual-speed store (fast episodic buffer + slow semantic store), runs a sleep consolidation cycle between sessions to decay, prune, and abstract memories, and applies reconsolidation rules that let memories update without drifting. The result is a memory system that behaves more like a mind than a key-value store.

Three things Verity does that no other package does: reconsolidation stability (memories update on access but cannot drift — a four-tier Bayesian system gates every modification), sleep consolidation (an offline decay → prune → abstract cycle that runs between sessions, mirrors the SO-spindle-ripple cascade), and tiered temporal weighting (the system auto-graduates from exponential to Bayesian renewal to Hawkes processes as event history grows, per-memory, without any configuration).

The API is seven methods. Zero configuration. No GPU. No cloud. No API key required. Runs on a Raspberry Pi and in a Kubernetes pod identically. Works with nothing but Python's stdlib, and gets progressively smarter as you add optional packages.

## Install

```bash
pip install veritycog                    # stdlib only — text search
pip install "veritycog[cognitive]"       # + model2vec + hnswlib (recommended)
pip install "veritycog[connectors]"      # + dlt (60+ data sources)
pip install "veritycog[full]"            # everything
```

## Quickstart

```python
from verity import Memory

m = Memory()                                      # SQLite, zero config
m.add("I prefer dark mode")                       # store
m.add("Team standup every day at 9am")
m.add("The API uses JWT authentication")

results = m.search("daily schedule")              # retrieve
for r in results:
    print(r["content"], f"({r['confidence']:.0%} confidence)")

m.update(results[0]["id"], "standup moved to 10am")  # update
m.consolidate()                                   # sleep cycle
m.export()                                        # GDPR portability
m.delete(results[0]["id"])                        # GDPR erasure
```

## The cognitive layer

| Component | Neuroscience model | What it does | Maps to |
|---|---|---|---|
| DualSpeedStore | Complementary Learning Systems | Fast episodic buffer + slow semantic store | SQLite + numpy |
| ImportanceScorer | Predictive Processing | Prediction error as surprise signal | Embedding cosine distance |
| ReconsolidationEngine | Memory Reconsolidation | 4-tier stability prevents drift | Bayesian Beta-Bernoulli |
| ConsolidationCycle | Sleep Consolidation | Decay/prune/abstract between sessions | Scheduled background pass |
| TemporalWeighter | Temporal Point Processes | Auto-selects exponential/renewal/Hawkes | Tiered by event density |
| GlobalWorkspace | Global Workspace Theory | K=5 competitive selection + position-aware output | Mitigates lost-in-middle |

## Advanced: Engine API

`Memory` wraps the lower-level Engine API, which provides the full RELATE/NAVIGATE/GOVERN/REMEMBER loop with connectors, profiles, consent management, and a Merkle-chained audit trail. See [examples/01_personal_notes.py](examples/01_personal_notes.py) for a complete walkthrough.

## For AI agents

```python
from verity import Memory

memory = Memory()

def agent_response(user_input: str) -> str:
    # Recall relevant context
    context = memory.search(user_input, k=5)
    context_str = "\n".join(r["content"] for r in context)

    # ... call your LLM with context_str ...

    # Remember the interaction
    memory.add(f"User asked: {user_input}")
    return response
```

## License

Apache-2.0
