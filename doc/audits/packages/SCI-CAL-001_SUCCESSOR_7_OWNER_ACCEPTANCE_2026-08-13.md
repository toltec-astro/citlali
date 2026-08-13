# SCI-CAL-001 successor-7 owner acceptance

Date: 2026-08-13

Record ID: `SCI-CAL-001-SUCCESSOR-7-OWNER-ACCEPTANCE-001`

Status: owner accepted return for bounded successor-8 repair; repair not launched

## Exact accepted identities

The owner accepts the independent successor-7 re-audit at exact commit
`0461d1e6ce484d52bf654afcfc563703943fb847` on branch
`codex/reaudit-sci-cal-001-successor-7-20260813`:

- parent: exact application candidate
  `9037314fd84241fa535c486d4ffb28966bb0394d`;
- tree: `902bfc1a67a41ac16eda8b60a8be6690cb915ad0`;
- documentation patch SHA-256:
  `8ccdcdf0575ff0fc0aaf47ab18a9c3843544930dc4ac851f8f9e59193e75b911`;
- immutable [report](../SCI-CAL-001_SUCCESSOR_7_REAUDIT_2026-08-13.md),
  SHA-256
  `93e57a60aba11ed3f2996b08f0b35b639d7d2abc86466adedff9428e74ee33bc`;
- immutable [local evidence](../evidence/SCI-CAL-001_SUCCESSOR_7_LOCAL_EVIDENCE_2026-08-13.yaml),
  SHA-256
  `974bc36e9a28647eda5b4932a1f0b85e79f8a4c9d3f97fbf6dcc92b6c2544029`.

The accepted application candidate is exact pushed commit
`9037314fd84241fa535c486d4ffb28966bb0394d` on
`origin/codex/repair-sci-cal-001-successor-7`:

- parent: `211e2f16f6354609de3ce6c6ee526d8aa4c6c59c`;
- tree: `9d2095159ac208a1096519a6fa710172275d3b73`; and
- parent-to-candidate binary-patch SHA-256:
  `e761c4c25070ea7b925f8e75c05c5dbec05c1849132312f279112482dcded4e0`.

## Accepted disposition

The verdict remains `amend`. The controlled axes remain:

- scientific contract: `approved`;
- implementation: `nonconformant`;
- validation/readiness: `in_progress`; and
- production: `fail_closed`.

F002 through F006 retain their accepted bounded closures. F007 retains the
accepted source-conformant response-identity repair without a new claim that
its independent runtime gate passed. F001 and F010 remain open and
conditioned. F008 and the local implementation portion of F009 remain open
only for F008-A, F008-B, F009-A, and F009-B and are returned for the bounded
successor-8 technical repair defined by the
[repair handoff](SCI-CAL-001_SUCCESSOR_8_BOUNDED_REPAIR_HANDOFF_2026-08-13.md)
and [finding ledger](../proposals/SCI-CAL-001_SUCCESSOR_8_REPAIR_FINDING_LEDGER_2026-08-13.yaml).

## Bounded successor-8 authority

F008-A must atomically publish every required Pointing RawObs, FilteredObs,
RawCoadd, and FilteredCoadd data and noise FITS artifact before its owner is
cleared. A failed replacement preserves any existing valid final.

F008-B must atomically publish every required Science Wiener-filtered
observation and coadd data and noise FITS artifact and remove the false
unconditional assumption that those products were already written. This does
not authorize a global transaction or unrelated writer redesign.

F009-A must compare exact YAML scalar type and value when validating the
requested-config preimage; boolean `true` is not integer `1`. F009-B must
require exact package-local selected-APT membership coverage and reject unused
or extra rows, consistent with the existing production admission boundary.

The repair must first add focused executable counterexamples for all four
defects and then run the complete fresh broad deterministic matrix. It may not
adopt prior candidate results as fresh successor-8 evidence.

## Preserved authority

The historical `phase5_readiness --verify-fixtures` gate remains
`failed_owner_waived_never_passed`, solely for six point and twelve science
immutable historical `sig2noise_pixel_I` errors. Historical bytes, verifier,
declared outcomes, contracts, and accepted runs remain unchanged.
`sig2noise_pixel_I` remains prohibited for every new/current product. The v4
current-production candidate epoch, contracts, and profiles remain
`preparing`; no production promotion is authorized.

## Non-authorization

This acceptance does not launch successor-8, create its branch or worktree,
close SCI-CAL-001, promote production, or rewrite accepted evidence. It
authorizes no arithmetic or scientific redesign, global transaction,
unrelated writer refactor, validation weakening, accepted-run rewrite, Unity
access or request, reduction, external contact, downstream work, re-audit,
merge, or push.
