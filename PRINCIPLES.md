# Verity Principles

These are the invariants. They do not change.

A Verity deployment that does not enforce all five of these is not Verity.
The behavioral canary tests verify enforcement at every engine boot —
not as documentation, but as running code that halts startup on failure.

---

## The Five Invariants

### 1. Human wellbeing is above all else
No optimization target, performance metric, or business requirement overrides
this. The system exists to serve people — not the other way around.

### 2. Crisis content triggers an absolute barrier
Always. No exceptions. No configuration disables this.

The barrier runs before entity recognition, before graph writes, before
anything else. A crisis signal routes immediately to crisis resources and
halts all other processing. This cannot be turned off. It is verified
by canary tests at every boot.

### 3. Everything is recorded. The record cannot be modified.
The audit trail is append-only and Merkle-chained. Every context assembly,
every checkpoint decision, every consent grant and revocation has a permanent,
tamper-evident record. If you can modify an audit record, the system is broken.

### 4. Humans decide. The system presents candidates.
GOVERN checkpoints cannot be bypassed or automated away. Veto is the default
— if no human response is received within the timeout, the action is vetoed,
not approved. The automation bias warning is displayed at every checkpoint.
This is not a UX choice. It is an architectural invariant.

### 5. Nothing happens without consent
No graph traversal, no context assembly, no data ingestion proceeds without
a valid consent reference. Consent is per-purpose — authorization for one
purpose does not authorize another. The consent gate runs before any
graph operation begins.

---

## The Scientific Foundation

Two additional invariants govern the engine's scientific claims:

**Power-law decay, not exponential.** Edge weights decay according to
Wixted (2004) / Jost's Law. Most systems use exponential decay — this
is the difference. The formula is:

```
effective_weight = base × (1 + days)^(-exponent)
```

Sensitive edges use `exponent × 1.4` (Nolen-Hoeksema 1991).
Spacing reinforcement applies `min(2.0, 1.0 + days/30.0)` (Cepeda et al. 2006).

**Three axes, always.** Edge relevance is measured on three orthogonal axes:
Distance (information retrieval), Complexity (graph theory), and Size
(cognitive science). These cannot be collapsed to one without losing
the orthogonal information each carries.

---

## How Enforcement Works

Enforcement is three-layered:

1. **Cryptographic** — `principles.yaml` is Ed25519-signed. A modified file
   without a valid signature will not pass boot verification.

2. **Behavioral** — canary tests run at every boot and verify the system
   *behaves* as declared, not just that the principles file is intact.
   A signed file with weakened behaviors fails the canary tests.
   The engine does not start.

3. **Structural** — the type system enforces invariants at construction.
   A `ContextBundle` without `uncertainty` cannot be built.
   A `ContextBundle` with an empty `reasoning_trace` cannot be built.
   These are not runtime checks — they are compile-time constraints.

---

## What Cannot Be Changed

The `immutable` tier in `principles.yaml` contains principles that cannot
be modified by any process — not by re-sealing, not by enterprise
customization, not by any ceremony. These are:

- `life_first` — human wellbeing above all else
- `crisis_barrier` — absolute barrier, no exceptions
- `audit_trail` — append-only, tamper-evident, permanent
- `human_sovereignty` — humans decide, veto is the default
- `consent_gate` — nothing without consent

Any deployment that removes or weakens these is not Verity.

---

*The three axes are right. The power-law decay is right. The human checkpoint
is right. The uncertainty-first output is right. The Merkle chain is right.
Everything else is implementation. Those five things cannot change.*
