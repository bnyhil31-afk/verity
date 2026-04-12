# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Verity, please report it
responsibly. **Do not open a public GitHub issue for security vulnerabilities.**

Report privately via GitHub's built-in security advisory feature:
https://github.com/bnyhil31-afk/verity/security/advisories/new

Include:
- A description of the vulnerability
- Steps to reproduce
- The potential impact
- Any suggested mitigations

You will receive a response within 72 hours. We take security reports
seriously and will work with you to understand and address the issue.

---

## Scope

### In scope
- Crisis barrier bypass or weakening
- Consent gate bypass
- Audit trail tampering or suppression
- Principles file substitution without detection
- PHI/PII exposure through classification errors
- Authentication or authorization bypass in the API layer
- Merkle chain integrity violations

### Out of scope
- Vulnerabilities in dependencies (report to the dependency maintainer)
- Issues requiring physical access to the host system
- Social engineering attacks
- Issues in domain modules (report to the module maintainer)

---

## Security Architecture

### Crisis barrier
The crisis detection barrier is an absolute safety control. It runs before
any other processing on every ingestion call. It has no configuration
surface and cannot be disabled. Bypassing it is a critical severity finding.

### Principles integrity
`principles.yaml` is cryptographically signed with Ed25519. The signature
is verified at every engine boot. Behavioral canary tests additionally
verify that the system behaves as the principles declare — a signed file
with weakened behaviors still fails verification.

### Audit trail
The audit trail is append-only and Merkle-chained. Every record's hash
includes the previous record's hash. Any modification to any record
breaks all subsequent hashes. `verify_chain()` detects tampering.

### Consent gate
The consent gate runs before any graph traversal. It validates that:
- A consent record exists for the given `consent_ref`
- The record is active (not revoked, not expired)
- The declared purpose matches the consented purpose

A missing, revoked, or mismatched consent record raises
`ConsentRequiredError` and halts the operation.

### Data classification
PHI, PII, FINANCIAL, and LEGAL classifications are applied at ingestion.
Classifications can only be escalated — never downgraded. Facts ingested
without explicit classification default to INTERNAL.

---

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x (pre-alpha) | Security fixes only |

Verity is currently pre-alpha. The security model is being actively
developed alongside the codebase.

---

## Known Limitations (Pre-Alpha)

- `principles.yaml` is currently unsigned during development
  (`signature: null`). This is expected and logged as a warning.
  Signing will be enforced in v1.0.

- The GOVERN checkpoint currently uses stdout/stdin. A production
  deployment should replace `_await_checkpoint_response()` with a
  secure UI adapter before handling real regulated data.

- The rdflib backend stores data in memory by default (no encryption
  at rest). Set `VERITY_GRAPH_PATH` to enable persistent storage.
  Disk encryption is the operator's responsibility in pre-alpha.
