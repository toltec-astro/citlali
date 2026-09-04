# MAP-SPACE-SHARED-CONVENTIONS-REPAIR-001 Work Order

Status: owner-authorized repair candidate; independent exact-SHA review required

Date: 2026-09-04

## Purpose and authority

Repair the four owner-resolved repository-documentation conflicts reported by
`MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001`.  Scientific owner
disposition `MSP-OD-001` selects the frozen SCI-MAP v0.1/r0.7.1 and SCI-JINC
v0.1/r0.3 meanings.  This work records those already-selected meanings in the
shared conventions; it creates no new scientific choice.

## Preflight

| Field | Exact value |
| --- | --- |
| Owner | Citlali scientific owner, Grant Wilson |
| Risk tier | Tier 2: shared scientific-convention repair |
| Effective governance read | `doc/governance/ENGINEERING_GOVERNANCE.md`; `doc/governance/REVIEW_AND_CONFORMANCE.md`; accepted digests recorded in `doc/INTEGRATION_LEDGER.md` |
| Current sequencing authority | `doc/REFACTOR_STATUS.md` at the exact base |
| Scientific authority | `MSP-OD-001`; frozen SCI-MAP v0.1/r0.7.1; frozen SCI-JINC v0.1/r0.3 |
| Exact canonical base | commit `5f0fc20042b88fb6cd883c92d1b59b7f22832901`, tree `97a4d908061e51418f93afc1d97d27433af441b8` |
| Preserved audit commit | `34a29a1ea` on this branch; seven files byte-identical to the verifier-passing detached audit packet |
| Branch | `codex/map-space-shared-conventions-repair-2026-09-04` |
| Worktree | `/private/tmp/citlali-map-space-shared-conventions-repair-2026-09-04` |
| Initial dirt | none before the byte-identical audit packet was copied; the packet was preserved as its own commit |
| WIP slot | scientific-contract documentation repair; no application-implementation slot |

## Included scope

1. Make the ordinary SCI-MAP quantity explicitly nonpolarimetric
   total-intensity-equivalent, with the inherited top-of-atmosphere,
   point-source-equivalent `mJy/beam` convention and exact fixed-nominal-beam
   lineage.  A legacy component index or label `I` does not establish formal
   Stokes I.
2. State SCI-MAP coaddition at the observation-output-row domain with
   dimensionless `u_op = 1`, without flattening separately typed sample,
   pixel, numerator, denominator, validity, coverage, response, or covariance
   information and without extending the rule to JINC.
3. State MAP original-footprint exposure as unique-original geometric
   accounting at each original occurrence's own AST ALIGN-grid coordinate,
   independent of descendant signal membership, filtering, interpolation,
   operator/response support, or statistical weight.
4. Limit the base SCI-JINC v0.1 numerical bundle to its exact five roles and
   preserve its compact generative record as information state rather than a
   sixth product.  Do not infer any additional weight, response, covariance,
   exposure, standalone-support, diagnostic, generalized-provenance, or coadd
   product.
5. Adjust only direct cross-references that would otherwise continue to assign
   one of those four superseded meanings.  Such an adjustment must restate the
   same owner disposition and must not introduce another scientific choice.
6. Add a deterministic repair verifier and candidate report.

## Excluded scope

- No change to any frozen scientific-contract package, application code,
  validation product, executable registry, algorithm, default, numerical
  route, response/covariance availability, implementation, FRUIT, or ALIGN.
- No resolution of the audit's `MSP-F-005` or `MSP-F-006` minor findings.
- No claim of implementation conformity, validation, performance, readiness,
  production, activation, deployment, or Unity behavior.
- No integration, canonical-ref movement, push, rebase, cleanup, or deletion.

## Expected changed paths

- `doc/SCIENTIFIC_CONVENTIONS.md`
- this repair directory only

## Gates and stop conditions

- Exact-base and ancestry check.
- Frozen-package byte nonmutation check against the base.
- Deterministic checks for removal of the four conflicting meanings and
  presence of their owner-selected replacements.
- Markdown reference and repository-whitespace checks for the changed paths.
- Diff review proving no application, validation, package, FRUIT, or ALIGN
  path changed.
- Fresh-context independent review of one exact full candidate SHA before any
  integration decision.

Stop rather than choose if the repair reveals a new scientific ambiguity,
requires a frozen-package change, or cannot preserve an unrelated shared
convention.

## Authorization boundary

The owner's 2026-09-04 direction authorizes this bounded branch, worktree,
repair, verification, and coherent local commits.  It does not authorize a
push, integration, activation, cleanup, or production claim.
