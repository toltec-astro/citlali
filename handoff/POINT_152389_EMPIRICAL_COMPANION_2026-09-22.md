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
