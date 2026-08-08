# SCI-PTC-001 coordinator decision brief — 2026-08-08

Record ID: `SCI-PTC-001-COORDINATOR-DECISION-BRIEF-2026-08-08`

Status: audit integrated; six scientific owner decisions ready; no decision,
repair, validation execution, re-audit, downstream launch, or production
change authorized.

## Verified audit authority

- Application: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`
- Independent core: `66e8d6f98c3e22da74de4eea84e568a0b4cc6310`
  (parent application SHA; tree `b4775f0f18277b28497fbbc0b24a7b583b9c0980`)
- Final audit: `01ee247461d6c19bc4db81ccac4fec21af162c88`
  (parent core commit; tree `e6685c920ff37f1d4e51d27ecf23b73ac16087b5`)
- Scope: exactly two commits and eight documentation additions from the
  application base; no application/test/configuration path.

The package is accepted into coordination with `contract_status: proposed`,
`implementation_status: nonconformant`, `validation_status: in_progress`,
`production_status: existing_use_only`, and verdict `amend`. Four P0
implementation findings, two additional P0 dependency findings, seven P1
contract/policy/dependency/evidence findings, and all associated closure gates
remain open.

## Owner choices ready for review

### `SCI-PTC-001-D001` — disabled-clean semantics

Choose whether all-disabled PTC is exact identity or source-mask mean-centering
is a mandatory independent stage. Coordinator recommendation: exact identity
when all PTC cleaning is disabled; if centering is required, name and serialize
it separately in requested, effective, observation-resolved, and realized
state.

### `SCI-PTC-001-D002` — direct cause and transitive influence

Approve only this scientific invariant: no scientifically eligible output may
retain influence from an input that is later rejected. The durable
representation, compact bookkeeping, and choice between recomputation and
explicit descendant invalidation are engineering decisions, provided the
invariant is exact, fail-closed, and falsifiable for signal, kernel,
coefficients, and every consumer.

### `SCI-PTC-001-D003` — response product families

Choose which response classes PTC promises: fixed-state conditioned operator,
realized local Jacobian, global/extended response, response uncertainty, and
unavailable. Coordinator recommendation: publish only classes actually
computed and bound to exact realized and upstream state; mark every stronger
class explicitly unavailable. Do not infer global or beam response from the
current partial conditioned kernel.

### `SCI-PTC-001-D004` — coefficient, precision, and covariance families

Choose the factor identity, units, normalization scope, lifecycle,
marginal-precision conditions, and retained covariance required for each
coefficient family. Coordinator recommendation: type approximate, full,
hybrid, validated, constant, correlation-penalized, and busy-row values as
nonprecision coefficients unless complete precision conditions are proved.
Covariance may be declared unavailable; constructing or retaining full
covariance is not mandatory for the bounded repair. When unavailable, every
stronger covariance, precision, significance, or independent-noise claim must
fail closed. Current map denominators and the universal file unit establish
none of those stronger claims.

### `SCI-PTC-001-D005` — immutable product and state bundle

Approve the bounded repair minimum: correct full, mini, diagnostic, simulated,
and processed product identity, rate, and extents; scan-specific detector
binding; explicit parent/provenance links; and a completion/atomicity rule.
Exhaustive serialization or replay of every selector, mode, random state,
response state, and covariance state remains P1 work unless a declared
consumer requires it; it is not a prerequisite for the bounded minimum.

### `SCI-PTC-001-D006` — missing-data, fallback, and null policy

Approve eligible-only arithmetic, coupled surrogate signal/validity shifts,
and fail-closed behavior when support is insufficient. Persist enough seed,
algorithm, and input identity for deterministic replay. Storing every realized
random shift and computing selection uncertainty may be explicitly unavailable
or deferred; neither is a prerequisite for repairing F001/F002.

## Dependencies and outgoing routing

- RTC remains audited `nonconformant` and `existing_use_only`; its handoff is
  acknowledged but neither accepted nor closed.
- CAL repair results are not admitted. AST remains behind ALIGN.
- ALIGN commit `5c6309125fef15f7c98e70a62b591c663944b130` is acknowledged only
  as corroborating post-core evidence for F004. It is not integrated or
  treated as Unity-validated.
- `SCI-VAL-001-XAUD-008`, `SCI-MAP-001-XAUD-004`, and
  `SCI-NOI-001-XAUD-001` are registered as submitted and pending recipient
  review. They do not launch their targets. BEAM receives no separate launch
  or handoff; later BEAM-relevant facts remain routed through VAL and this
  brief.

## Required order after owner review

1. Record approved or superseding D001--D006 authority.
2. Select one exact integrated application line with accepted RTC and relevant
   CAL/AST/ALIGN successors.
3. Separately authorize a bounded PTC repair.
4. Run only an owner-authorized exact-successor evidence protocol.
5. Perform a fresh independent PTC re-audit before consumer or production
   expansion.

## Exact immutable audit artifacts

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/packages/SCI-PTC-001_INDEPENDENT_CORE.tex` | `82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2` |
| `doc/audits/packages/SCI-PTC-001_SCIENTIFIC_CONTRACT_AUDIT.tex` | `c46a15c142d0938baf9576d84a19332e0d46b34852b4d59c0029ba00ac62d7e6` |
| `doc/audits/evidence/SCI-PTC-001_LOCAL_EVIDENCE_2026-08-08.yaml` | `091059abd088b8bca58ca5a885e12620972c1f75f75574e33bfff8b0eb90b195` |
| `doc/audits/proposals/SCI-PTC-001_LEDGER_PROPOSAL_2026-08-08.yaml` | `8daabce7d0d585e82d233dadb3f535bb993a28c62acb351c4474272c525eee63` |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-008.yaml` | `fdeaa3d18909a35b3caff85257f70e7f51ae6115ec07d172cbe96fd1b5007a32` |
| `doc/audits/handoffs/SCI-MAP-001/SCI-MAP-001-XAUD-004.yaml` | `5c5221366d9fd66cffc3881cb8fad2f9b1fee990bfd581b4583cf0d1b72c53d2` |
| `doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001-XAUD-001.yaml` | `eb9c8588e0c09d8a9882ed076ee0b8cc33ccad9c49319dea0b56e4419eee3c8c` |
| `doc/audits/packages/SCI-PTC-001_OWNER_DECISION_BRIEF_2026-08-08.md` | `eaaf7bc06988dcb4bc1ae2a7da235aab4ac5ebfd96850fb580d850d5c82a2752` |

The three handoff digests above identify immutable auditor-submitted bytes at
audit commit `01ee2474...`. Coordinator-owned source-commit and pending-review
fields change the canonical registry bytes; their current canonical digests
are recorded separately in the ledger.

## Non-authorizations

This record does not authorize application, test, or configuration changes;
repair; Unity or external evidence; local reductions; broad/costly execution;
re-audit; VAL/MAP/NOI/BEAM launch; production change; or promotion of the
unintegrated ALIGN repair.
