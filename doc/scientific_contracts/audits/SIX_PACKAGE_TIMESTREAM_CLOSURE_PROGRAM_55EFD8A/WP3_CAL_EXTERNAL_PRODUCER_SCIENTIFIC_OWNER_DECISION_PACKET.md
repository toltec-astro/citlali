# WP-3 CAL And External-Producer Scientific-Owner Decision Packet

Date opened: `2026-08-23`

Status: in progress; `WP3-OWNER-D001` approved and realized; external
producer-interface decisions remain to be reviewed one at a time

Scope: `WP-3_CAL_EXTERNAL_PRODUCERS`, the CAL facet of `F-016`, the
timestream-route facets of `F-017`, and the applicable CAL/input contribution
to `TS-CLAR-001`.

This packet records scientific-owner decisions only. It does not inspect
Citlali, assert implementation conformity, execute validation, or authorize
MAP work.

## WP3-OWNER-D001 — CAL Authority Freeze Versus Achieved Performance

Question:

> Should SCI-CAL v0.1—scientific rationale r0.5 and engineering conformance
> r0.4—be frozen now as the exact scientific authority, while observational
> validation and achieved-performance acceptance remain pending?

Recommendation presented to the owner:

> Yes. Q01--Q09 are scientifically decided, the science and engineering views
> are consistent, and the package is mechanically complete. Freeze the exact
> contract authority now without claiming implementation conformity,
> observational validation, achieved accuracy, total uncertainty, science
> qualification, production readiness, or achieved-performance acceptance.
> Treat the 1%, 5%, and 5--10% figures as reporting benchmarks. Later evidence
> attaches under its own identity and changes the frozen authority only if the
> science itself must change. Replace the ambiguous phrase “final scientific
> acceptance” with “achieved-performance acceptance.”

Owner response:

> I approve.

Disposition: **approved**.

Consequences:

1. SCI-CAL v0.1 science-rationale r0.5 and engineering-conformance r0.4 are
   frozen as the active scientific authority.
2. Contract-authority approval and achieved-performance acceptance are
   distinct states.
3. The Q09 validation workflow is authoritative, but no executed evidence or
   achieved claim is inferred.
4. The exact freeze and verification records live in
   `packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` and
   `packages/SCI-CAL/v0.1/FREEZE_VERIFICATION_R0.5.md`.
5. This supplies the CAL freeze artifact required by WP-3; `F-016` remains
   open pending VAL freeze, final binding, and WP-7 clean-room re-audit.

## Remaining WP-3 Owner Work

The next decisions must identify the smallest exact producer-interface set
required by the selected ordinary processed-timestream route. Each interface
must separate static/operator authority from observation-instance realization.
Optional roles may be explicitly not requested or unavailable; they may not
be supplied by inference. MAP-only roles remain deferred.
