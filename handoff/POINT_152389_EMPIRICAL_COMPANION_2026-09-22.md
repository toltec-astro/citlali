# POINT 152389 empirical uncertainty companion

## Work order / preflight

Owner request: implement a reproducible successor companion for background,
coverage-matched blank measurements, map-domain source refits and split-data
maps for a1100/a1400/a2000. Reuse immutable maps/PTC products. Do not change
upstream processing, calibration, six-plane primary MAP, weights or FRUIT.

Base: e33020bbd0bc826f01059b6b665e3098ce7dbb60, existing branch
`codex/map-successor-first-forward-001`, worktree
`/private/tmp/citlali-map-successor-first-forward-001`. Initial tracked tree
clean; existing build symlink untracked. Shared dirty ALIGN checkout untouched.
Single bounded successor application slot continues on the existing branch.

Read: ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE and
REVIEW_AND_CONFORMANCE; their accepted ledger binding remains the one used by
the reviewed parent. Tier 2, exact-SHA independent review required. Local Python
venv only; Agg plotting. No Unity, push, integration, activation or cleanup.

Authority recovery: current SCI-POINT v0.1/r0.4 is presentation republication of
accepted r0.3 science, not a numerical-method approval. Its separately owned
compatibility, formal-error and full-map-RMS methods remain unavailable in the
recovered package. Checked package source, owner-method response and closure,
package worktree and local refs/history for later approval. The saved successor
152389 description is a distinct seven-parameter elliptical Gaussian plus
constant in J2000 tangent coordinates, unweighted, 40 arcsec fit radius and
20 arcsec search/centroid bounds. Do not relabel it as the six-parameter
zero-background SCI-POINT compatibility method. Asked owner for any later
approved numerical-method record; independent background/split work can proceed.

Numerical owner: this explicitly requested empirical diagnostic companion.
It consumes exact read-only MAP bundles and retained PTC publications through
the existing bundle verifier; no Engine or pipeline numerical change. Request,
measured background/mask/location facts, and output products remain distinct.
This is not an NOI randomization product, a POINT formal covariance, a detection
significance, or a calibrated full-chain source-response measurement.

Planned interface: successor Python command with exact map/measurement parent
paths, deterministic seed and explicit analysis request. Separate reusable
measurement functions, input/replay boundary, reporting and synthetic validation.
Products retain definitions, units, masks, selected blank coordinates, coverage
mismatch, trial outcomes/failures and exact source identities. Missing support
produces an explicit unavailable diagnostic, never a fabricated error bar.

Blank locations are selected by geometry/coverage, not signal amplitude.
Background statistics include median, sample standard deviation and
1.4826022185*MAD with mask-radius sensitivity. No sqrt(hits) variance law is
assumed. Spatially overlapping blank patches are identified as dependent; their
count is not an independent-sample count or a significance calibration.
Source-template, fitted-amplitude and aperture quantities stay distinct.

Splits partition existing accepted PTC contributions by processing scan parity
(0,2,4 versus 1,3,5), retaining exact parent WCS and coefficients. They do not
rerun upstream learning. Report A, B, A-B, coverage balance and fixed-template
source residual; a single split is not an independent uncertainty ensemble.

Gates: white/correlated/background/incomplete-coverage synthetic maps; independent
realization scatter compared to within-map estimates; non-Gaussian wings and
model-mismatch separation; identity/mutation/missing-support/determinism tests;
existing successor Python/config/baseline gates as applicable; exact original
map preservation, split accounting, reproducible real products and visual QA;
separately measured companion runtime; independent read-only review.

## Recovered detailed proposal and resolved implementation scope

The user work order's “described above” refers to the detailed proposal in the
accessible task “Summarize Sky Model Admission” (2026-09-22). Recovered its
measurement definitions: source mask three measured major-axis FWHM and
four-FWHM sensitivity; raw fixed unit-peak-template projection without an
intercept; apertures at one/1.5/two measured major FWHM without annulus subtraction;
complete measurement footprints and explicit dependence/failure reporting.
The implementation follows those empirical definitions. The proposal's generic
zero-background POINT statement does not override the actual saved descriptive
fit or supply the separately unavailable numerical compatibility method.
No later approved numerical-method record was recovered. Reuse of the exact
saved description is within the current owner's explicit reuse instruction;
it is explicitly named and is not represented as generic SCI-POINT conformance.

SCI-CAL r0.5 Q03 binds mJy/nominal-beam peak calibration. FWHM alone is not an
admitted solid-angle conversion for integrated photometry. Apertures therefore
retain mJy/nominal-beam arcsec^2, with definitions and blank scatter in the same
units. They are enclosed map brightness, not an invented total-flux estimator.

Source fit objective/domain/bounds are preserved. Existing implicit SciPy
choices (TRF, exact dense solver, linear loss, two-point Jacobian) are now named
explicitly; actual saved fit amplitudes/centroids/widths must reproduce before
publication. No fit constant is newly added or background subtracted from MAP.

Synthetic reassessment: the preserved optimizer occasionally returns a finite
angle-bound solution. The first validation harness incorrectly demanded zero
such outcomes. The requested empirical procedure must retain and count these
outcomes rather than silently drop them or change the fitter. The corrected
harness compares the inclusive output distribution, records bound frequency,
and tests accounting explicitly. Any nonconvergent refit still withholds the
aggregate scatter. No optimizer constraints or numerical solution changed.

Initial real support audit finds 61/25/6 compact-template placements and
13/1/0 full 40-arcsec refit placements for a1100/a1400/a2000. Thus the latter
two fitted-amplitude scatter products will be unavailable at this exact fit
footprint. Do not relax coverage or shrink the fit to manufacture error bars.

A direct-pixel adversarial test caught an offset-template support bug before
publication: footprint eligibility used convolution instead of correlation for
the non-centered source template. Corrected the orientation and added a direct
check of every selected footprint against the actual source/background mask.
This preserves the intended measurement and source-exclusion definitions;
initial exploratory placement counts are superseded by the final report.

## Candidate gates before observational execution

202 successor Python tests (111 subtests), 207 baseline Python tests (137
subtests), required config preflight and diff whitespace checks pass. Forty-eight
independent realizations for each of five map cases (240 total) pass the
empirical-versus-actual scatter comparisons and known-kernel linear variance
checks; all trial records include constrained/failure states. Native numerical
source and build configuration are untouched, and no native executable is used
by this command. The parent native CTest gate remains preserved; current
companion validation is Python/map-level evidence, not a new native run.
Actual products and separate companion runtime will bind the committed source.

## Final runtime and evidence candidate

Implementation candidate 8b4b667c05a32d58184d31a4d9b4c2f3d2f812b4 was
independently reviewed. The bounded review repair now rejects refit scatter
when the source's 40-arcsec fit disk is incomplete, even if compact measurements
remain usable; tests cover absent support and nonfinite data separately.
Runtime source: 3e9b5512e42e47e67dbdfe06b4c4183d0c707f61, tree
 da4a3e437be5227ef5cb788e1547e250c096ce4b. All actual source disks have
1257/1257 finite pixels; repair leaves their numerical measurements unchanged.

Durable root:
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-empirical-companion-20260922`.
Final products: `companion-r2`, `synthetic-validation-r2`; `RESULTS.md` explains
results and limitations, `final-source.json` binds verification. Initial
`companion`/`synthetic-validation` directories preserve the earlier source run.
Root manifest SHA256:
`671d6d8e66f478c48929dad13cf8779acb35c3b606bed44983407f2cc03621ba`.

The 38 final companion manifest entries and all 42 FITS HDU checksums verify.
All three exact input map hashes remain unchanged. Each bundle passes existing
operator/parent/PTC replay checks. Counts of retained PTC contributions partition
exactly into the two splits; numerator recombination differs only by reported
floating-point reassociation. Current/native parent artifacts remain read-only.
Nine real diagnostic figures and the synthetic validation figure inspected;
shortened split titles and explicit pixel axes are verified in final renders.

Final measurements, mJy/nominal beam:

| Array | Fixed amplitude | Blank scatter | Fitted amplitude | Refit scatter / bias | Compact/refit locations |
|---|---:|---:|---:|---|---:|
| a1100 | 805.0 | 12.92 | 816.31 | 10.80 / +2.72 | 62 / 13 |
| a1400 | 551.0 | 8.35 | 554.15 | unavailable | 28 / 1 |
| a2000 | 583.5 | 4.79 | 593.40 | unavailable | 5 / 0 |

Compact disjoint subset counts are 10/6/2 (descriptive scatter 18.34/10.72/5.84).
Overlapping counts are not independent counts; the five-location a2000 result
is weakly sampled. No fit-size/coverage relaxation creates missing refit errors.
Apertures preserve raw enclosed brightness times pixel area; full values and
blank scatter are in the ECSV and report. No unsupported beam-area conversion.

Coverage-binned background SD spans 12.56–18.17 / 10.50–14.10 / 15.49–25.22.
Changing the exclusion from three to four measured major FWHM raises overall
SD by 0.33% / 0.57% / 1.12%. The companion scale has explicit support and
coverage-bin validity; it is not a calibrated variance/covariance image.

Split fitted amplitudes A/B are 780.96/866.53, 496.98/630.48 and 589.09/623.82.
The A-B fixed-template source residual is -63.18/-66.63/unavailable. a2000 B
has 1253 fit pixels versus 1257 in A, and missing common pixels in the compact
fixed-template footprint; its source-shaped residual remains visible in the
image, while that exact complete-footprint projection is unavailable. Split
coverage 10/50/90 percentile A fractions are .443/.513/.590, .392/.493/.601,
and .378/.500/.623. No independent/equal-depth scaling or significance claim.
Existing nw3 scan 5 failure remains unavailable, unchanged from the parent.

240 independent synthetic map realizations (48 in each of five cases) pass.
Empirical/actual scatter ratios range .812–1.200; actual-SD Monte Carlo relative
precision is about 10.3%. White noise, spatial correlation, nonzero background,
incomplete coverage and non-Gaussian wings are covered. Wing-case actual versus
empirical refit scatter is 8.483/8.484, while noiseless fitted peak differs from
physical peak by -14.566: model mismatch is explicitly separate. Finite bound
outcomes are retained/counted; no failed synthetic refits. This validates the
map-domain companion only, not upstream PTC source recovery.

Final run: 8.346 s wall, 6.178 s CPU, 460.3 MiB peak RSS, including parent
verification, saved-PTC split replay, trials and plots. Previously accessed
inputs; the initial source candidate took 40.763 s wall. Synthetic runtime
separate. Python 3.13.2 in the prescribed venv; one math thread and Agg backend.
No native executable or full RTC/PTC rerun used. Native source unchanged.

Gates: 13 focused tests; 204 successor tests plus 111 subtests; earlier unchanged
207 baseline tests plus 137 subtests; full required config preflight; compile
and whitespace checks. No unexpected error output in successful final gates.
An initial focused-test invocation named a nonexistent test file and collected
nothing; corrected invocation passes. This was a command typo, not a test skip.

Scientific/behavioral disposition: candidate pass with the named limitations.
Architectural disposition: separate read-only companion; no Engine, upstream,
MAP weighting, CAL or FRUIT changes. Repository disposition: existing isolated
branch, only intended paths changed, unrelated shared ALIGN dirt untouched,
original untracked build symlink retained. No push/integration/Unity/activation
or cleanup. Final exact-SHA independent evidence review pending; closure will
record its three separate dispositions. No owner FRUIT decision is made here.

Independent final evidence review reproduced blank amplitudes/scatter, checked
every blank/refit footprint, all runtime code/parent hashes, exact split counts,
FITS checksums, synthetic records and representative plots. One prose-only
rounding correction: a2000 trial coverage ratios span 0.727677–0.815917,
rounded 0.73–0.82 (the initial RESULTS text said 0.75–0.81). Corrected the
external report and rebound the root manifest above. Numerical products and
runtime source are unchanged. No additional numerical rerun is warranted.

## Independent exact-SHA review and closure

Fresh-context reviewer `/root/point_empirical_review` approves evidence candidate
`5b8a8ff763b325dc42372b5a54f2bc65bb9a22f8`, tree
`1b0b6624f71a82122aeedef3c36a8fc03505fc4c`, parent
`616b44b72ea697ff70a48424091aa070bd1c3de4`; runtime remains
`3e9b5512e42e47e67dbdfe06b4c4183d0c707f61`.

- Scientific and behavioral conformance: pass with recorded limitations.
  Independently reran 13 focused tests; reproduced blank measurements, all
  selected footprints, complete source fit domains, split count recovery,
  difference projections and explicit a2000 unavailability. Checked all 240
  saved synthetic records and representative figures. Conditional scatter,
  dependence/sparse placements, missing refit errors, unequal split domains
  and excluded upstream/calibration/model-mismatch uncertainties stay explicit.
- Architectural conformity and ownership: pass. Separate read-only companion
  preserves MAP, upstream processing, CAL, weights, FRUIT and Engine boundaries.
- Repository, branch and evidence hygiene: pass. Exact code/measurement hashes,
  three unchanged original MAP hashes, 38 companion entries, all FITS checksums
  and 96 root artifact entries independently verified. Coverage rounding repair
  resolved; final root digest matches above. Tracked tree clean, only original
  build symlink untracked. No unresolved actionable findings.

This closure changes documentation only and preserves the reviewed candidate
and runtime identities. No integration/push/Unity/activation/production-use or
cleanup authorization is inferred. Bounded empirical task complete; no FRUIT
admission decision or upstream source-response qualification supplied.
