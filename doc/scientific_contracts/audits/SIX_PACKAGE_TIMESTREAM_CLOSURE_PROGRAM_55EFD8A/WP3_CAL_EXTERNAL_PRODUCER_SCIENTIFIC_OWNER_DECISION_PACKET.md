# WP-3 CAL And External-Producer Scientific-Owner Decision Packet

Date opened: `2026-08-23`

Status: in progress; `WP3-OWNER-D001` and upstream clarification
`WP2-FOLLOWUP-D011` are approved; `WP3-OWNER-D002` awaits owner review

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

## WP2-FOLLOWUP-D011 — Tune/Readout Producer-Interface Closure

The first formulation of `WP3-OWNER-D002` incorrectly placed native
Tune/readout and \(x/r\) acquisition semantics at the RTC-to-CAL boundary.
Owner review established that frozen SCI-ALIGN v0.1/r0.3 and SCI-RTC
v0.1/r0.12 already assign those meanings upstream of RTC:

\[
(I,Q)^{\rm acq}
\longrightarrow \text{Tune/readout}
\longrightarrow (x,r)^{\rm acq}
\longrightarrow \text{SCI-ALIGN}
\longrightarrow (x,r)^A
\longrightarrow \text{SCI-RTC}.
\]

The remaining omission is an exact external producer-interface binding, not
a missing CAL acquisition interpretation.

Question:

> Should we add a bounded, package-neutral Tune/readout producer-interface
> record upstream of ALIGN, without reopening ALIGN or RTC?

Recommendation presented to the owner:

> Yes. Bind the versioned producer/interface meaning and observation-instance
> association required by frozen ALIGN and RTC. Preserve the producer-owned
> units, sign, reference, normalization, transform, Tune/mapping revision,
> paired occurrence identity, applicability, validity, uncertainty state, and
> provenance. Do not invent a new sign convention, make CAL interpret Tune
> data, duplicate ALIGN/RTC mathematics, or freeze observation payloads in the
> repository.

Owner response:

> approved

Disposition: **approved**.

Consequences:

1. The issue is routed to `WP-2A_NATIVE_READOUT_INTERFACE`, the bounded
   upstream facet of `F-017/XOD-015`.
2. The candidate interface is
   `producer_interfaces/v0.1/TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md`.
3. Frozen SCI-ALIGN and SCI-RTC remain unchanged.
4. Exact candidate-artifact approval remains pending.
5. The original broad CAL-related formulation of D002 is withdrawn.

## WP3-OWNER-D002 — Conditioned RTC `x` Handoff To CAL

Status: pending scientific-owner review

Question:

> Should the RTC-to-CAL signal boundary admit only the exact conditioned
> \(x^{\rm RTC}_{dn}\) member of a complete realized RTC atomic bundle, with
> its inherited upstream coordinate convention and RTC-owned grid, plan,
> support/response state, validity, uncertainty state, lineage, and
> provenance, while treating \(r\) as traceability rather than a CAL numerical
> input and treating calibration-factor and target-atmosphere records as
> separately owned CAL-time joins?

Recommendation:

> Yes. RTC shall guarantee that the handed signal is the exact conditioned
> \(x\) output for the named detector and stable RTC sample on the named
> realized plan and grid. Its raw detector unit or scale, sign, reference, and
> normalization are inherited unchanged from the admitted upstream mapping;
> this boundary creates none of them. The handoff shall preserve the exact
> representative ALIGN parent, complete RTC-local action/influence support,
> realized response or typed-unavailable response state, validity and causes,
> uncertainty/covariance availability, original-occurrence/exposure lineage,
> lifecycle, and provenance. The complete paired RTC bundle remains the
> immutable traceability parent, so raw or requested conditioned \(r\) and
> \(r\)-derived RTC evidence remain reachable, but SCI-CAL neither consumes
> them numerically nor calibrates them. Selected `flxscale`/child-APT and
> target-atmosphere/passband inputs remain separate producer-owned inputs
> joined by SCI-CAL; an RTC handoff may carry their exact references as
> required by frozen RTC authority but does not own, derive, repair, validate,
> or apply them. Missing or ambiguous required \(x\) identity, plan/grid,
> support, validity, or provenance fails the affected CAL admission. A typed
> unavailable response or uncertainty state never becomes zero and blocks
> each dependent claim or operation at the scope required by the frozen CAL
> and RTC contracts.

## Remaining WP-3 Owner Work

After D002, the remaining decisions must identify the smallest exact
producer-interface set required by the selected ordinary processed-timestream
route. Each interface must separate static/operator authority from
observation-instance realization. Optional roles may be explicitly not
requested or unavailable; they may not be supplied by inference. MAP-only
roles remain deferred.
