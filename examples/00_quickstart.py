# Run from repo root: python examples/00_quickstart.py
# Requires: pip install verity

"""
Quickstart — Memory API
=======================
The simplest demonstration of Verity's cognitive memory system.
No connectors, no profiles, no async — just the seven-method API.

What this demonstrates:
  - Zero-config Memory instantiation (SQLite, no embedding model)
  - add(), search(), get(), update(), consolidate(), export(), delete()
  - Confidence scores on every result
  - GDPR-compliant export and erasure
"""

from verity import Memory


def main():
    m = Memory()  # zero config — SQLite in ~/.verity/memory.db

    print("── Adding memories ──────────────────────────────")
    id1 = m.add("I prefer dark mode in my IDE and terminal")
    id2 = m.add("Team standup is every weekday at 9am")
    m.add("The REST API uses JWT for authentication")
    m.add("Postgres is the production database")
    id5 = m.add("Deploy to staging every Friday afternoon")
    print(f"Stored 5 memories. Example ID: {id1[:8]}...")

    print("\n── Searching ────────────────────────────────────")
    results = m.search("daily schedule")
    print(f"Found {len(results)} results for 'daily schedule':")
    for r in results:
        print(f"  [{r['confidence']:.0%}] {r['content']}")

    print("\n── Getting one memory ───────────────────────────")
    detail = m.get(id2)
    print(f"Memory {id2[:8]}...: {detail['content']}")
    print(f"  confidence_tier: {detail['confidence_tier']}")
    print(f"  strength:        {detail['strength']:.2f}")

    print("\n── Updating ─────────────────────────────────────")
    m.update(id2, "Team standup moved to 10am starting next Monday")
    updated = m.get(id2)
    print(f"Updated: {updated['content']}")

    print("\n── Consolidating (sleep cycle) ──────────────────")
    result = m.consolidate()
    print(f"Decayed: {result['decayed']}, Pruned: {result['pruned']}, "
          f"Duration: {result['duration_seconds']:.3f}s")

    print("\n── Exporting (GDPR Article 20) ──────────────────")
    export_data = m.export(format="json")
    import json
    memories = json.loads(export_data)
    print(f"Exported {len(memories)} memories as JSON")

    print("\n── Deleting (GDPR Article 17) ───────────────────")
    deleted = m.delete(id5)
    print(f"Deleted deploy memory: {deleted}")
    print(f"Verify gone: {m.get(id5)}")

    print("\n── Done ─────────────────────────────────────────")


if __name__ == "__main__":
    main()
