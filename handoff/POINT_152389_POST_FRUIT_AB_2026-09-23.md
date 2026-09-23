# POINT 152389 post-FRUIT A+B work order / preflight

Owner: Grant Wilson; authorized 2026-09-23 to implement, validate, execute and
assess the archived OWNER_DIRECTIVE.md in
validation/point_152389_post_fruit_ab_2026-09-23. Tier 2 scientific-module probe.
Exact starting candidate 94297bf1fdbc60a91478b18cb4479cf99f19bfa3;
branch codex/map-successor-first-forward-001, worktree
/private/tmp/citlali-point-additive-20260923, initially clean.

Read ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE and
REVIEW_AND_CONFORMANCE. Effective governance is accepted
06a3ade51c1b3f38887295433d913811bf25cd14, incorporated at
77507836325eff9f469062d5884481ea37599594; digests in INTEGRATION_LEDGER.
Sequencing: completed ten-cycle handoff and REFACTOR_STATUS. Scientific
authority: current A+B owner directive and recovered full-observation additive
directive, preserved saved-descriptive-v1 estimator, original frozen template
and apertures. MAP/POINT/NOI/FRUIT contracts are not amended. Architecture and
SCIENTIFIC_CONVENTIONS were read; compact companion utilities own inference,
simulation and reporting. Saved maps/measurements are immutable producers;
this separate evidence bundle is the sole consumer. No Engine/core change.

Prespecified choices, before noise comparisons: common finite science support
across M0/T1/T3/T5/T10; exclusion union of circles of four times each saved
major FWHM around its saved centroid. This enlarges the old three-FWHM mask
to conservatively exclude wings/bowls. Larger sensitivity uses five times;
neither mask depends on inferred noise. Four variants: mask5, north, south,
lag48 (base lag32); same original M0 centroid divides regions. All models use
a fixed100×100 generator grid, leaving the41×41 measurement stamp unchanged
and permitting identical random inputs for all64-trial comparisons. No mixture
amplitude alternative is fitted: report overlapping coverage and mixture
confounding without forcing separation. Base256, matched64 per variant.
One ordinary training mean removed; no other mode removal or clipping.
Seeds bind array/T10/trial plus a declared shared model-comparison stream.

Changed areas: additive inference's optional explicit mask, companion drivers,
diagnostics, focused tests, documentation and evidence. Existing defaults and
upstream products remain unchanged. New outputs bind all consumed inputs and
manifest ancestry before and after. Expected memory: full-map compact arrays,
source stamps and one small dense pair check; four process workers, one library
thread each. Profile parent/child memory and phase runtimes separately.

Gates: exact saved measurements, masks, covariance/generation/padding, coupled
background/aperture propagation, deterministic scheduling, failure handling;
existing independent synthetic192 outer ×16 inner ×6 cases with unchanged
tolerances; successor and baseline suites and full required config preflight.
Native CTest evidence inherited from ten-cycle source (no native source change
or new native science execution); local Python environment is supplemental,
not Spack V2 reproduction. Independent fresh-context exact-SHA read-only review
required before closure. New numerical/schema/failure/provenance findings
trigger recorded reassessment; repair only within the companion boundary.

Excluded: native RTC/PTC/MAP/FRUIT runs, response injections, C/D, cycle11,
application of m10, changes to fit/weights/admission/calibration or source
feedback model, new dependencies/Unity, integration/push/production/cleanup.
Missing scientific inputs make the affected component unavailable; they do not
authorize a replacement estimator or expanded science work.

## Initial checks and bounded reassessment

All15 selected saved source measurements reproduce with zero absolute difference
for peak/template/all apertures and the seven fit parameters (tolerance1e-10).
Focused14 tests passed; Apple BLAS emitted spurious floating-point flags on a
finite dense matrix product, replaced by explicit pair summation; focused7 new
tests then passed without warnings. Successor246 tests/111 subtests and full
required config preflight passed. Existing independent192×16 six-case campaign
passes ordinary checks with unchanged tolerances; challenge diagnostics retained.

The baseline gate initially reports11 failures (including subtests),203 passes/
130 subtests because the expected build/bin/citlali locator is absent. This is
an environment prerequisite, not a companion numerical failure. Refresh the
existing local native build at the exact candidate and run the protocol gate;
no observation execution or dependency/build-policy change. Native CTest may
then be repeated as supplemental exact-build evidence. The independently
reviewed Spack-backed V2 campaign remains a separate inherited evidence axis.
