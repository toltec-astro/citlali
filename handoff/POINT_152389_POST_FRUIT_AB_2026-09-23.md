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

Independent review of9d95916ca found bounded failure-path repairs: optional
larger-mask insufficiency must not suppress base work; scientific measurement
support failures need per-array/cycle availability, and a failed base64 prefix
must not divide a successful variant by unavailable scatter. These are repaired
with focused tests. Review also corrected signed east-axis plot pixel edges.
Local operator smoke exposed zero-retained segments with no MAP interval;
verification now skips only explicitly zero-retained segments and tests it.

## Completed base execution and stochastic-check reassessment

Source6b1e59c9e passed independent review in all three categories with scientific
limitations. All three256 refit ensembles completed, no failures; finite
constrained1/3/3 retained. Exact-source independent192×16 validation passed,
matching CLI baseline207/137,28 selected CTests and config passed. All selected
per-pixel network contribution maps are unchanged from M0.

One of24 fixed-background propagation sampling checks missed its prespecified
99.9% interval: a1100 double-domain largest aperture variance ratio0.711317
versus[0.734003,1.317320], two-sided Gaussian MC tail0.000307. The corresponding
analytic/dense/double-grid variance identities remain exact. Preserve this
failure, do not retune tolerances or rerun source ensembles. Before inspecting
new samples, prescribe a single4096-draw linear-only check of both existing
representations with independent seed9234096, same99.9% tolerance, no learning,
no fits and no changed scientific covariance. This bounded diagnostic resolves
an unexpected numerical result under the authorized generation-validation scope;
its outcome and the original anomaly must both be reported. It is not additional
observational calibration evidence or an opportunity to tune the model.

## Completion / conformance

Candidate evidence complete; owner decisions for integration/push/activation
remain absent. Primary runtime6b1e59c9e7032dcfe8181b85fed264ff805cbca4/tree
2894cd1fb52d6afc458b06fb8ba29d02ea94e51b; linear follow-up
5f5853bff67ff2f330400702461cb1afcd01fdb5/tree
c06eed4a5dc88c778e68ec0cfe965235cfd51c92. Exact base remains94297bf1f.
Changed paths: compact Python companion, tests and documentation/evidence only.
No core/Engine, upstream, source-measurement or feedback mutation.

A+B companion52.164s; source-bound synthetic192×16×6 validation30.496s,
ordinary cases pass unchanged tolerances. Source peak scatters7.905909/5.680591/
6.628750 mJy/nominal beam; zero failures, finite constraints1/3/3 retained.
All measurements include coupled trial backgrounds. Provisional conditional
estimates for a1100/a1400; assumption-dependent a2000. Source-support depth
interpolation is complete but spatial transfer is not demonstrated. See owner
assessment for every aperture and all five requested scientific answers.

All15 saved measurements reproduce with zero difference;212 companion files
verify. All12 selected per-pixel network contribution comparisons match M0.
23/24 initial fixed-background sampling checks pass; one independent4096-draw
linear-only follow-up passes8/8 while retaining original failed0.711317 flag.
No noise/source-model tuning or new source ensemble followed the anomaly.

Focused10 tests independently reproduced; complete successor249/111,
baseline207/137,28 selected native fixture/CLI CTests and full required config
passed. Native build matched primary source6b1e59c9e; local AppleClang/cached
dependencies are supplemental, not Spack V2. Baseline's initial missing-binary
failures are retained with repaired gate log. No unexpected error-level output
in successful scientific companion execution; inherited PTC condition unchanged.
The independent reviewer checked source, all artifacts, exact regeneration and
all9 figures, passing scientific/behavioral with limitations, architecture and
repository/evidence hygiene. Final document review is recorded in validation.

All prior input hashes/manifests unchanged before and after. Published source
values, fixed admission scale and m10 state unchanged. Source response/adaptive
uncertainty remain excluded. No new Unity/dependency/build-policy work, automatic
integration/push/production or cleanup. Source/worktree retained for review.
