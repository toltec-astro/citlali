# SCI-MAP-002 re-audit coordinator disposition — 2026-08-08

Record ID: `SCI-MAP-002-REAUDIT-COORDINATOR-DISPOSITION-2026-08-08`

Status: corrected re-audit integrated; repair candidate rejected; bounded
successor-repair definition ready but not authorized or launched

## Exact integrated authority

- Repair candidate: `854a04b124e083e64706fd043e105182fee568af`,
  sole parent `46ad23888a40f5102cdfd50c06e49a549bdf8a20`, tree
  `188ae8535b61a3f560fa992b3bac0a5196436e5b`.
- Initial re-audit: `4fa876d066a8bf7e6b971147a92f0b8b7ffd5c77`,
  parent `854a04b124e083e64706fd043e105182fee568af`.
- Documentation-identity correction successor:
  `756e448361587d492d5bb72adfc7775bfad67851`, parent
  `4fa876d066a8bf7e6b971147a92f0b8b7ffd5c77`, tree
  `38ba3bbb6f9eb2e09ea163ab6a8ad5bebb2bf1ab`.
- Frozen coordination authority:
  `dd5894679bf12bf4a5fb551e871b3c6010ef9b9b`, tree
  `e87b507a6dc5246da0f65e563d96b94824e61ba1`.

The corrected immutable artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/packages/SCI-MAP-002_REAUDIT_2026-08-08.md` | `da2f04936c4fb191b8f54274623f779854222a2d97defba05617ff3abc5301f0` |
| `doc/audits/proposals/SCI-MAP-002_REAUDIT_LEDGER_PROPOSAL_2026-08-08.yaml` | `48f1c264eefd55bf4044e4661aa283592938bf31c8338df83ee2d67076ab53f3` |
| `756e448...:doc/REFACTOR_STATUS.md` | `48ea8a731c4e4d27968498a606a078c4a158e593a4fa71b0e058c6e7769fd4ed` |

The source status bytes are identity evidence only and are not copied over the
current coordination line's living status. All 14 authority-artifact digests
in the corrected proposal were independently recomputed from
`dd589467...` and match.

## Coordinator disposition

The owner-approved SCI-MAP-002 contract remains approved. Candidate
`854a04b...` is rejected as nonconformant. Canonical status is
`implementation_status: nonconformant`, `validation_status: in_progress` with
the explicit assessment `incomplete`, `production_status: existing_use_only`,
and verdict `amend`. No Unity evidence exists for the candidate, and no repair,
re-audit, BEAM audit, downstream audit, or production expansion is authorized.

- `SCI-MAP-002-RA-001` is open P0: iterative Beammap clears fresh JINC
  N/C/Q-side buffers without clearing `denominator_sum_abs` and
  `contributor_count`, so later finalization uses stale conditioning state;
  active-subset realized summaries are incoherent.
- `SCI-MAP-002-RA-002` is open P1: kernel-template, processing-chain, and
  coverage-rate provenance remain placeholders or incomplete.
- `SCI-MAP-002-RA-003` is open P1: sequential/concurrent agreement is not
  exercised through the production JINC population paths.
- `SCI-MAP-002-RA-004` is open P1: strict positivity admission broadens to
  inactive JINC settings for naive mapmaking rather than the selected JINC
  boundary.
- `SCI-MAP-002-RA-005` is open P1: production-boundary response, admission,
  failure suppression, identity, writer join, and concurrency evidence remains
  incomplete.

## Bounded successor-repair definition

This section defines readiness only. It is not a dispatch, authorization, or
launch. Any later repair must name an exact branch, base, owner, and stop rule.

1. Reset `denominator_sum_abs` and `contributor_count` atomically with the
   corresponding JINC N/C/Q iteration state, and publish a coherent realized
   summary whose observation/iteration and active-map scope are explicit.
2. Replace generic kernel-template and processing-realization placeholders
   with compact exact identities for the actual upstream template, enabled
   processing/filter state, and coverage sample-frequency linkage. Do not add
   per-sample, per-detector, or per-pixel provenance payloads.
3. Restrict positive JINC parameter admission to selected JINC execution and
   products; preserve the prior inactive-JINC behavior of naive mapmaking.
4. Add deterministic production-seam tests for iterative/active-subset reset,
   below/equal/above response boundaries, non-finite failure before
   publication, formal support after failed admission, coverage-rate linkage,
   actual template/processing identity, writer joins/failure suppression, and
   sequential/concurrent population agreement under the declared policy.
5. Preserve the approved signed-lobe N/C/Q estimator, square cached support,
   phase-point subpixel response, dimensionless conditioning rule, formal-mask
   and coefficient-squared coverage meanings, K/C response, and compact
   four-stage/product provenance. Do not redesign RTC, PTC, temporal filters,
   notches, PCA/common-mode cleaning, noise, Wiener, convolve, covariance, or
   GLS behavior.

A completed successor repair must stop with exact commit/parent/tree,
documentation, deterministic local evidence, and a proposed fresh independent
re-audit packet for coordinator/owner review. Unity evidence and re-audit
remain separate later authorizations and cannot cure source nonconformance.

## Non-authorizations

This record does not modify application, test, build, configuration, or
validation code; accept `854a04b...`; select or launch a successor repair;
request or access Unity; launch re-audit, BEAM, or another downstream audit;
change production status; alter any RTC/PTC authority; merge; push; or contact
an external party.
