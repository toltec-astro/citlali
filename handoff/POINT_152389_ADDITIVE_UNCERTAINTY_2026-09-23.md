# POINT 152389 conditional additive uncertainty: work order

Owner directive: `/Users/gwilson/Downloads/CODEX_POINT_152389_FULL_OBSERVATION_ADDITIVE_UNCERTAINTY.md`.
Tier 2 bounded continuation of the completed MAP+ diagnostic module probe.
Read ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE and
REVIEW_AND_CONFORMANCE; digests exactly match the accepted integration ledger.
Read AGENTS, living status, architecture, scientific conventions, saved empirical
method, balanced-pairing closure and current owner work order. The present owner
work order authorizes routine numerical design and validation; it does not amend
SCI-NOI, a pending formal POINT error method, or the source estimator.

Base `045eb9173952b94733d12343eb0e7a3f3686d944`, existing branch
`codex/map-successor-first-forward-001`. The previous temporary checkout vanished;
restored that exact branch in `/private/tmp/citlali-point-additive-20260923`.
No branch/ref was pruned, moved, integrated or renamed. Initial tracked/untracked
tree clean. The unrelated dirty main checkout remains untouched. One module slot,
no second implementation branch. Current task adds a separate compact map-domain
statistical diagnostic, reading immutable saved MAP/POINT/companion products.
Expected paths: tools/timestream_successor/point_additive*.py, reusable command,
focused tests, extension of synthetic validation using the existing fixture and
fitter, method documentation and this handoff/status. No Engine/core/native code.

Preserved authority: exact saved-descriptive-v1 seven-parameter unweighted
elliptical Gaussian plus constant; 40 arcsec RADIUS about AST center; 20 arcsec
peak search and original parameter bounds/solver/initialization/failure behavior.
Fixed template uses sum(t M)/sum(t²), original shape/centroid and complete radius
3 measured major FWHM, without background subtraction. J2000, 2 arcsec pixels,
nominal calibration beams 5/7.5/10 arcsec and mJy/nominal-beam peak units persist.
`load_parent` reproduces and verifies the exact original measurements before use.
Accepted MAP admission/coefficients and PTC/VAL state remain upstream-owned.
This companion owns empirical covariance learning, a frozen covariance product,
simulation and measurements, all under the present conditional additive target.

## Numerical design declared before final results

Subtract one ordinary mean of the selected source-masked background; retain
mean/robust/tail diagnostics. No clipping, gradient removal or Gaussian-source
subtraction. Five equal-count coverage bins estimate ordinary second moments
about that common mean, with explicit bin ranges and at least100 pixels each.
Piecewise-constant amplitude applies only within the observed coverage range;
endpoint continuation is separately marked assumption-dependent if required.
Masks use3/4/5 saved measured major FWHM. Covariance of normalized residuals is
computed by FFT pair-product sums divided by actual mask pair counts, retaining
both spatial axes, signs and pair counts. Fit a compact PSD spectrum by tapering
supported lags, projecting negative spectral power to zero, and renormalizing
rho(0)=1. Publish projection size and residuals so repair cannot conceal mismatch.
Base maximum lag32 pixels (64 arcsec), cosine lag taper; compare16/48 pixels and
taper strengths0.7/1.4. Generate on a padded grid at least twice the measurement
stamp dimension, crop, then modulate by amplitude. Double-domain comparison and
explicit qTCq propagation check normalization and wraparound. No fit weights change.

Sensitivity set: base; masks4/5; lags16/48; taper0.7/1.4; north/south training.
Disjoint quadrants and low/high coverage comparisons retain directional covariance,
second moments and projected fixed-template variance. These diagnose the shared
correlation assumption; they do not define a source-admission threshold.
Longer-scale modes beyond measured lags remain unknown; use estimator response
to constant/linear/broad sinusoidal perturbations to expose relevant modes.

Start256 trials per array, deterministic array/trial seeds, Gaussian correlated
noise on clean fitted source+constant. All nuisance parameters refit with exact
saved initialization. Primary refit SD is centered with ddof1, never SD/sqrt(B).
Any failed trial withholds aggregate; finite constrained fits remain counted.
Sensitivity runs use64 paired-seed trials. Independent synthetic outer budget
selected after a pilot; target48 outer maps per case, modest8 inner fits per map,
not256 per outer. Cases: white, anisotropic signed-kernel, variable amplitude with
missing coverage, long-range structure, spatially changing correlations, and
non-Gaussian wings. Generate via independent moving-average kernels/fields, not
via the inferred PSD. Compare mean estimated VARIANCE to actual variance.

Predeclared sampling checks: fixed-template simulation variance vs analytic
variance uses a central99% chi-square interval (exact under Gaussian generation).
Synthetic estimated/empirical variance comparison uses log-ratio uncertainty
sqrt(2/(outer-1) + sampling variance of mean estimated variance / mean²), 3 sigma,
with a declared20% variance-model approximation allowance. Report disagreements;
misspecified cases are diagnostic challenges, never tuned to pass. Monte Carlo
precision on SD ~1/sqrt(2(B-1)) is separate from model instability. Model adequacy
requires source-depth support and assessment of region/mask/lag sensitivity;
implementation gates alone confer no real-data applicability.

Gates: meaningful covariance/simulation/failure/missing-support tests; full
successor/baseline Python and required config preflight; full saved-map hashes;
independent synthetic inference validation, real blank checks, all-array256 trials,
separate wall/CPU/memory/threading profiles, visual inspection and fresh-context
exact-SHA independent review. Native/Spack/Unity not triggered by Python-only
saved-map analysis. Optional frozen-MAP cross-check only if existing boundary
permits a small bounded test. No upstream ensembles, calibration/weights/source
admission/feedback changes, FRUIT changes, integration, push, activation or cleanup.
Stop after numerical/model assessment and report; preserve unsupported candidates
as assumption-dependent diagnostics rather than observed error bars.

## Development reassessment

Five focused tests pass without warnings after replacing a diagnostic NumPy
pseudoinverse multiply with the equivalent SciPy least-squares solve; the local
NumPy/Accelerate matmul emitted invalid-status warnings despite finite correct
results. Explicit finite checks and independent direct sums remain. An initial
real-data pilot exposed FITS COMMENT cards being flattened into a multiline dict
value; preserve the original Header through fitting/serialization instead. No
measurement method or input changes.

Pilot: four outer/four inner per six synthetic cases, 1.795 s wall with two
workers. Real a1100 four/four pilot 39.098 s, predominantly mandatory parent
verification (23.349 s) and spawning small pools; covariance inference0.029 s.
First full development synthetic run48 outer/eight inner per six cases6s wall.
The anisotropic case predicted/actual fit variance0.609, despite broad declared
sampling gates passing. This material discrepancy justifies more independent
observations, not looser tolerances or fitting to the test. Preserve that run and
increase final budget to192 outer/16 inner per case (18,432 inner refits plus
1,152 independent observed fits). Keep generators/model/tolerances unchanged.
This improves precision of actual SD from10.3% to5.1%; mean estimated variance
still includes finite-inner variability. Misspecified/long-range cases remain
challenges even if scalar comparisons happen to agree.

The existing frozen-MAP Python boundary is a read-only full-parent verifier/replay,
not a small callable synthetic pointing/mask producer. The optional detector-noise
projection test is not performed; do not build a timestream simulator for it.

Development gates:220 successor tests/111 subtests pass; required config preflight
passes. Baseline suite has203 Python tests/130 subtests passing, but four CLI
protocol tests (plus seven subcases) cannot run because the prior temporary native
build disappeared with the old worktree. The saved Sep22 executable path is absent;
remaining local binaries predate changed APT protocol sources, so they are not
silently substituted. No new baseline/native pass is claimed. These unchanged
native tests remain unavailable environment evidence; focused additive behavior
and the rest of the Python surface are exercised. Independent review must assess
this limitation separately from scientific validity.

The192/16 development validation completes in30.185 s. Predicted/actual variance
ratios fixed/refit: white1.047/0.826, anisotropic1.089/0.932,
varying-incomplete0.906/0.881, long-range0.814/1.064,
spatially-changing1.147/0.948, wings0.909/1.059. No failures; finite constrained
fits retained. The revised anisotropic result resolves the small-ensemble concern.
Long-range fixed variance remains underestimated19%; misspecified regional
correlation differences persist despite a close scalar refit variance. These
are limitations, not reasons to tune the generator/model to this sample.

## Exact-source review and bounded repairs

Candidate0aaa11a67d8d6ce158a1d46feb9566d25f8d5d2b is the initial committed
real run. Independent review verified its complete104-file manifest, exact source
amplitudes, parent hashes, nonnegative/unit-normalized spectra and direct dense
pixel-pair qTCq matching the spectral calculation. All three256-trial ensembles
have no failures. Source-depth interpolation is supported in the base models.

Review found an all-failed-ensemble reporting defect: min/max over empty valid
sensitivity scatters and formatting a null mean could abort before retaining
trial failures. Repair preserves base trials immediately, uses explicit null
sensitivity ranges and null-safe report rows. Model/footprint unavailability is
per array after fatal parent verification. Added all-failed and actual trial
failure regressions. Also fixed representative plot seed for nondefault requests.

Review correctly noted that comparing256-trial base SD with64-trial sensitivity
SD confounds model changes with finite Monte Carlo variation. Now compare matched
prefixes of the same64 seeds, retaining the full256 base result separately. Each
variant reports a paired-bootstrap finite-Monte-Carlo ratio interval, explicitly
not a model or observational confidence interval. Unsupported sensitivity depth
endpoints remain flagged; they are not empirical sensitivity bounds. No source
model, input, fitting method, covariance inference or generated trial changed.

Native baseline gate remains required separately from Python scope. Attempt a
supplemental exact-candidate CLI rebuild using the already documented frozen local
FetchContent sources and Homebrew/AppleClang realization. No dependency download,
build-system adaptation, upstream reduction or Spack-equivalence claim is allowed.
If that bounded restoration fails, retain the explicit incomplete-gate evidence;
do not weaken tests or substitute an older binary. This is environment recovery,
not a native scientific change or pipeline experiment.
