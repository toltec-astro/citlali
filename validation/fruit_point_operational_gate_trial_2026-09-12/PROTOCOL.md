# Bounded POINT operational-gate trial

2026-09-12 · r0.1 · Prospective definition, frozen before numerical execution.

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md) and the
[reviewed method recovery](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md).
Adopt the original-parent, residual-relearning, reconstructed-total replacement
recurrence and unchanged PTC/MAP ownership. Cite the
[central-domain screen](../fruit_point_starlet_central_domain_2026-09-12/SCIENTIFIC_REPORT.md)
as prior evidence, preserving its rejection. This is an isolated development
experiment, outside scientific authorship and production.

## Owner authority and exact scope

Decision: `SCI-FRUIT-POINT-OPERATIONAL-GATE-TRIAL-2026-09-12`.
The owner responded **“Let's give this test run a try”** to the delivered
[use-specific proposal](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/POINT_USE_CASE_GATE_DESIGN_R0.1.md)
at commit `d76eb6135ffe123a211e280b4a91d3a648fae7fa`. This authorizes one
bounded execution of that design and the necessary prospective bindings here.
The owner separately answered **“Use 1 arcsecond for this test”** to the
operational per-observation centroid-error question. Peak response remains
the primary before/after OOF gain measure.

The investigator selects one concrete candidate within this trial: the same
60-arcsec central starlet estimator, with a predeclared larger numerical solve
budget. This is an intentional successor budget, not a routine-defect repair
or retrospective acceptance of old unfinished iterates. The earlier 300-step
results and all other rejected candidates remain closed under their old gates.
No second revision, sweep, 129081 comparison, Unity, production or general
qualification is included. Any change after freeze to method, gates, case
population, inputs or resource bounds stops for a separate owner decision.

## Candidate and controls

Candidate C inherits the [exact central estimator](../fruit_point_starlet_central_domain_2026-09-12/candidate.py):
same four-level transform, full computational mask, outer-plane subtraction,
band/coverage-stratum calibration from null20260911 bootstrap, absolute
five-scale coefficient selection, positivity, and weighted coefficient-error
objective. Both evidence eligibility and source-image support remain D intersect
radius ≤60 arcsec at the commanded origin. Q split and calibration use original
D; surrounding map and background regions are unchanged. There is no new radius,
cross-array prior, shape family, threshold, penalty or smoothing parameter.

The sole estimator change is L-BFGS-B `maxiter: 300 -> 3000`,
`maxfun: 3000 -> 30000`. Preserve `ftol=1e-12`, `gtol=1e-8`,
finite results, solver success **and** relative projected-gradient ≤1e-4.
Preserve the fixed-null normalization fallback only for zero current outer MAD.
Every call starts at zero. Empty coefficient support is a valid zero model;
a failed solve is unavailable, its trial retained and never applied.

P is the exact prior pixelwise baseline: positive total-map values exceeding
three current outer-annulus MAD scales, unfiltered flux on original D. Its
different feedback domain is part of this named policy comparison, not a pure
representation-only attribution. Both arms use the same immutable parent,
input flags, geometry, uniform occurrence weights, inherited 2-arcsec containing
pixel map, rank 5 per inherited network/chunk, and seven passes including
bootstrap. Relearn on each arm's current residual every pass. Retain each total,
applied/next model, selected coefficients, failed trial and learned state.

The [identified OG reduction](../fruit_point_coherent_feedback_2026-09-11/OG_BENCHMARK.json)
remains the operational benchmark. Its published offsets and seven-pass
205.38-second full-run time have different upstream/JINC/weighting/runtime
boundaries from this four-thread harness. Neither byte equality to an unavailable
historical executable nor an end-to-end speed ratio is claimed.

## Inputs and small known-signal population

Use only the existing hash-bound 123424 pre-PTC export in
[INPUT_PREFLIGHT.json](../fruit_point_coherent_feedback_2026-09-11/INPUT_PREFLIGHT.json),
the prior geometry/detector scales, null calibration map, and fixed bright-coma
truth. The exact files and all imported code are bound in `FREEZE.json` before
execution. Verify the regenerated geometry against the retained geometry.
Actual amplitudes retain the export's inherited mJy/beam convention; this does
not independently qualify its calibration or transfer function.

[CASES.json](CASES.json) records the eight states and two PCG64 nuisance seeds,
20260911 and 20260912. Nuisance generation is the inherited detector-scale AR
plus network-correlated construction. The first seed is also the known calibration
case; neither seed is an untouched validation sample. Each altered parent is
formed before learning and independently processed by each arm.

| State | Exact controlled construction |
| --- | --- |
| N | Nuisance only. |
| B | Nuisance plus the inherited plane `13+7*x/90-4*y/90`, with zero astronomical truth. |
| H | Gaussian apparent peak 100, centroid (17,−11) arcsec, FWHM 12×8 arcsec, angle 25 degrees; separate identical truths in each array. |
| H-shift | H displaced to (20,−9); known displacement (3,2) arcsec. |
| D | H at peak 90 with both widths multiplied by `1/sqrt(0.9)`, preserving the continuous Gaussian integral. This is a controlled 10% peak-loss/broadening probe, not a physical OOF simulation. |
| C | Previously frozen 4× diffraction/coma truth at (17,−11), with the new parent learning rerun for both seeds/arms. Startup source/degradation challenge; full-wing image error is diagnostic. |
| T | The entire a1400 parent stream of H multiplied by 0.8, including its nuisance. Other arrays unchanged. This is a 20% response-loss observability proxy, not a detector/atmosphere physical model. The perturbation is not supplied as an automatic quality flag to the evaluator. |
| E | H displaced to (58,−11), near the fixed central feedback/evaluation boundary; full original total-map support and truth are retained. |

Run these 16 synthetic cases plus real123424 for both arms: 34 trajectories,
at most 238 new cleaning calls. Alternate arm order across synthetic cases.
No oracle source information enters feedback, source detection or the measurement
fit. Truth supplies only the external scores and paired expected ratios.

## Common evaluation rules, fixed before results

`evaluation.py` measures total maps equally for both arms. It separately fits
the outer nuisance plane and records positive coefficients above five empirical
band/coverage scales in bands 2–4 or coarse, centered inside the fixed 60-arcsec
domain. At least one such coefficient is the finite-screen source-presence
rule. It is not a calibrated probability or independent confirmation of an
already rejoined model.

Fit the inherited free Gaussian plus plane to the central domain with the same
three data-only numerical starts and existing bounds. Its signed peak and
shape remain diagnostics even when inadequate. A usable compact measurement
requires source presence, peak >5 current outer MAD, no parameter-boundary
failure, at least ten core pixels, centroid radius <52 arcsec, and ≥95% of
the ellipse-enclosing circular aperture (radius 1.5 fitted major FWHM) inside
the supported central domain. These support tests do not change total-map D/S.

Measure relative residual norm on pixels where the fitted Gaussian is ≥10%
of its positive peak. A value >0.25 or major FWHM >24 arcsec gives a shape
warning, withholding precision photometry/pointing. These are trial adequacy
and gross-degradation rules, not fitted priors or an adopted OOF trigger.
The same rules apply blind to the state name and injected truth. Source
presence with such a warning can satisfy startup assessment. No source or
unavailable information cannot masquerade as a healthy measurement.

For each pass retain signed maps/residuals, fit parameters, core adequacy,
aperture brightness, exterior RMS, full-image error, support and failure status.
Use-specific finite-case gates are:

- U1: C must establish source presence in each required array and expose a
  shape/support warning; a correct warning may succeed without precise wings.
- U2: H/D peak gain must be available and within 5% of 1/0.9 for each seed and
  array. H(seed2)/H(seed1) must be within 5% of unity. Pointing is separately
  tested under U3.
- U3: H, H-shift and D must have usable centroids with error ≤1 arcsec in
  each seed/array. Report the H-shift minus H displacement error separately.
  Also report the two-realization mean error vector; do not call it a qualified
  ensemble-bias result or convert the old 0.05-FWHM bias target into a per-case
  requirement. Peak/FWHM 5% recovery remains diagnostic unless that additional
  absolute-recovery claim is made.
- U4: D/H must be available, within 5% of 0.9 and below 0.95; the unchanged
  H pair must remain within [0.95,1.05]. Record recovered width changes. This
  separates the two finite test states; 0.95 is not a telescope OOF action rule.
- U5: T/H must expose a1400 response loss (ratio <0.9, within 5% of 0.8) or
  explicitly unavailable affected-array measurement. The two unaffected-array
  ratios must remain available within 5% of unity. Missing a required ratio
  cannot establish response accuracy, even where the warning preserves health
  information. No cloud/tune cause is inferred.
- G1: zero C feedback admissions and zero source reports on N/B at all retained
  passes. Count baseline violations too; an inferior baseline does not excuse C.
- G2/G3: E must expose its support limitation and withhold a reliable correction.
  Preserve all degraded/failed cases. C has no feedback outside its declared
  domain; both total products retain original support and contributors.
- Leakage: candidate terminal exterior RMS ≤1.10 P on the same fixed O, for
  each completed paired case/array. Keep interior signed structure/full-image
  errors visible without a universal whole-coma fidelity limit.

Report finite errors and between-seed differences. Covariance, calibrated
confidence intervals, false-alarm rates, real-data truth and operational
uncertainty remain unqualified; no precision claim for real123424 derives
from a fit residual or agreement with OG. These limitations do not require a
comprehensive noise project before the finite controlled comparison.

## Completion, cost and decision

The primary result is pass 6 (seventh map) with all seven required decisions
available. Stop an individual trajectory on an invalid solve; save the current
map and trial, not a substituted zero-feedback continuation or an earlier
successful terminal. Continue other registered cases to characterize scope;
no adaptive changes follow failures. Record exact started/completed call counts.

Retain wall time to each map, full development wall time, source-evaluation
time, cleaning/mapping/inference components, and peak process RSS. The direct
sum of algorithm components is also reported, labeled separately from wall
time and output cost. Plot operational errors against cumulative wall time.
No new stopping policy is selected using truth-scored intermediate maps.

Resource ceilings: one hour overall, ten minutes per trajectory checked
between passes, 8 GiB peak RSS, 4 GiB retained output. The global alarm remains
a hard stop; per-call kernels may cross a between-pass check before returning.
No automatic extension or solver-budget revision. The old 0.5-second inference
screen is retained historical evidence, not this trial's operational ceiling.

Keep for a demonstrated subset only if its required operational and common
gates pass and a useful benefit is shown: crossing a 1-arcsec/5%-gain accuracy
gate that P fails, or reaching equivalent usable accuracy with lower measured
cumulative time. Report any additional cost; >2× matched full trajectory wall
time blocks a keep recommendation in this trial even if accuracy improves.
An unresolved timing difference is not a speed claim. Otherwise reject or
identify one specific revision opportunity without executing it. Replication
on a separately authorized independent pointing still precedes any numerical
policy recommendation. No 129081 values are opened here.
