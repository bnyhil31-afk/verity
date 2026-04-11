# Verity

**Trustworthy context for AI agents.**  
Embedded semantic reasoning for regulated industries.

---

Verity is an open-source Python library that assembles verified, uncertainty-annotated,
consent-gated context for large language models — with a complete, immutable audit trail.

It is designed for teams building AI agents in healthcare, finance, legal, and other
regulated industries where "just stuff the data into the prompt" is not an option.

## The 10-line demo

```python
import asyncio
from verity import Engine

async def main():
    engine = await Engine.start(modules=["fhir_r4"])

    async with engine.session(consent_ref="consent:abc123") as s:
        await s.ingest(patient_fhir_bundle, source="fhir_r4")

        context = await s.context(
            query="recent encounters and active diagnoses",
            purpose="clinical_decision_support",
        )

        print(context.agent_prompt)           # ready for LLM injection
        print(f"Uncertainty: {context.uncertainty:.0%}")
        print(f"Audit ID: {context.audit_id}")  # immutable, Merkle-chained

asyncio.run(main())
```

FHIR R4-validated. PHI-classified. Consent-gated. Merkle-chained. Uncertainty-annotated.
No configuration. No Java. No external database.
Runs identically on a Raspberry Pi 4 and in a Kubernetes pod.

## What it does

- **RELATE** — Ingests typed entities into a knowledge graph (RDF Named Graphs). PHI
  is classified at ingest. Crisis content triggers an absolute barrier before anything
  else happens.

- **NAVIGATE** — Traverses the graph using three-axis weighted edges (Distance /
  Complexity / Size) with power-law decay. Assembles a `ContextBundle` ranked by
  relevance, recency, and trust.

- **GOVERN** — Every consequential action passes through a human checkpoint. Veto is
  the default. The decision — and the context it was made in — is recorded immutably.

- **REMEMBER** — Append-only Merkle-chained audit trail. Every context assembly,
  every checkpoint, every consent event has a permanent, tamper-evident record.

## The output

```
[CONTEXT — clinical_decision_support]
Based on 7 verified observations (uncertainty: 23%):

Patient: John Smith (PHI-classified)
Active conditions: Type 2 Diabetes Mellitus (ICD-11: 5A11)
  — established: 847 days ago, confidence: 0.91
  — reinforced: 3 encounters across 2 care settings
Recent encounter: Primary Care, 2025-11-14
  — HbA1c: 7.2% (LOINC: 4548-4), trend: improving

[EXCLUDED: 2 observations below consent threshold | 1 below weight threshold]
[UNCERTAINTY NOTE: This context is bounded. Missing data ≠ absence of fact.]
```

That's what the LLM receives. Not raw graph data. Structured, domain-annotated,
uncertainty-explicit context.

## Status

**Pre-alpha.** The foundation is being laid. Not ready for production use.

See [PRINCIPLES.md](./PRINCIPLES.md) for the invariants that will never change.

## License

Apache-2.0. See [LICENSE](./LICENSE).
