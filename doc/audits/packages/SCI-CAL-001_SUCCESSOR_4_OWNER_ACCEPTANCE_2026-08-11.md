# SCI-CAL-001 successor-4 owner acceptance

Date: 2026-08-11

Record ID: `SCI-CAL-001-SUCCESSOR-4-OWNER-ACCEPTANCE-001`

Status: owner accepted `AMEND` / return for bounded successor-5 repair;
successor-5 authority prepared; repair not launched

## Exact identities

- Coordination authority:
  `aaa578f12a1bacc9476d8067d9f1554029b67f89` (parent
  `7d586ad06f136264fcfa7f5cffe3f2041d7438fb`, tree
  `7ff67a3ad8ca65387a7606f6b25246bd70b33022`).
- Audited application candidate:
  `693f1b107855e3ae9b36617323ca14aac868f304` (parent
  `3af6faf996fa002b2647adca8f33991002d49ff1`, tree
  `fb317a7862ff474c118d229bb45320adb560b3bc`) on
  `codex/repair-sci-cal-001-successor-4`.
- Candidate binary-patch SHA-256:
  `59ba71493377630be5bff5164d779aa43d14ee1e788c002707f1f4fe62d5902d`.
- Independent re-audit:
  `7163dfa45b269b04e9525c224ec1cb5c4525ea61` (parent exact candidate,
  tree `396b12c403b2142e2f149b65473f5246bf116450`) on
  `codex/reaudit-sci-cal-001-successor-4-20260811`.
- Audit documentation patch SHA-256:
  `b63f50f9afc79679efb1d50753d2e378d680e5443384cc36d3bb48daa914716d`.

The exact imported audit artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| [Successor-4 re-audit](../SCI-CAL-001_SUCCESSOR_4_REAUDIT_2026-08-11.md) | `47a7bc888a2c5b1981e96287b46c20392146f073f1cdb5cc092e842ddeeb9d9c` |
| [Local evidence](../evidence/SCI-CAL-001_SUCCESSOR_4_LOCAL_EVIDENCE_2026-08-11.yaml) | `597f8fe2c9106e287cc1eda9e341ff8c50d0d90d151add8cf703904cbfae44a0` |

## Accepted disposition

The owner accepts verdict `amend` and return for a bounded successor-5
technical repair. Successor-4 is not complete CAL closure and is not accepted
for integration or production.

The controlled axes remain:

- contract: `approved`;
- implementation: `nonconformant`;
- validation: `in_progress`;
- production: `fail_closed`; and
- verdict: `amend`.

F002, F003, F004, and F006 retain their previously accepted bounded closures.
F001 and F010 remain open and conditioned. F005, F007, F008, and the local
implementation portion of F009 remain open and are returned for the exact
bounded successor-5 repair below.

## Owner-approved successor-5 authority

1. F005 keeps the source APT and source sensitivity immutable. The
   per-observation flxscale correction `a` is explicit applied calibration
   state. It applies exactly once to calibrated samples and exactly once as
   `W' = W/a^2` in the approximate baseline inherited by approximate, hybrid,
   and validated. Full-weight behavior is preserved. Applied-factor and
   recipient provenance must be truthful, and the real production correction
   route must be tested through nonzero map realizations and
   `noise_variance_I`.
2. F007 records fixed, shared, and detector notch applications at the actual
   RTC/PTC application point, including RTC-versus-PTC phase, scan, PTC
   iteration, model-subtraction state, scope, detector/ordinal, geometry, and
   phase convention. Immutable requested state remains distinct from actual
   effective and realized state; only executed stages are realized.
3. TOD-only operation carries finalized canonical joins. Coadds use one CALID
   only when every contributing observation has that CALID; differing CALIDs
   fail closed. Successor-5 does not design heterogeneous coadd membership.
4. F008 applied-response history is observation-owned and resets at
   observation boundaries. Repeat finalization rejects or idempotently
   preserves the immutable consumed snapshot and CALID/PKGID. Interrupted,
   unavailable, reused-scan-number, and multiscan lifecycles are covered.
5. The canonical calibration package is published and validated before
   dependent linked products. Publication remains atomic per artifact. An
   orphan canonical package after a later dependent-output failure is
   acceptable; unresolved linked products are not. No global cross-output
   transaction or rollback architecture is authorized.
6. Local F009 publishes, contracts, and validates
   `{obs}/selected_calibration_apt.ecsv` once per calibrated observation.
   Lineage and member requirements are conditional on effective calibration;
   supported uncalibrated v4 remains valid without that member.
7. Local F009 validation is path-aware, hashes the actual sibling member,
   requires canonical lineage schema and complete components, verifies
   source/component/package digest joins, and recomputes package identity.
   v4, product contracts, profiles, baselines, and production-shaped fixtures
   are synchronized for all approved success/failure cases.

These decisions resolve the factor-placement, homogeneous-coadd, lifecycle,
publication-order, and per-observation-layout choices returned by the audit.
They authorize no new CAL science, covariance, uncertainty product, empirical
response-fidelity claim, heterogeneous coadd schema, or global transaction
architecture.

## Successor-5 handoff and non-authorization

The frozen successor-5 scope is defined by
[`SCI-CAL-001_SUCCESSOR_5_BOUNDED_REPAIR_HANDOFF_2026-08-11.md`](SCI-CAL-001_SUCCESSOR_5_BOUNDED_REPAIR_HANDOFF_2026-08-11.md)
and its machine-readable finding ledger. The proposed branch is
`codex/repair-sci-cal-001-successor-5` from exact pushed base
`693f1b107855e3ae9b36617323ca14aac868f304`.

This record does not create a branch or worktree and does not launch repair or
re-audit. It does not modify application, configuration, test, build, or
validation-product code, access Unity, run reductions, change production,
merge, push, contact external parties, or launch RTC, PTC, MAP, BEAM, TEL,
ALIGN, FLT, or downstream work.
