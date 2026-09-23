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

## Completed numerical run and assessment

Runtime source `d70233e1c22b96065f700d43ed65f952da3cf32a`, tree
`41e199fc4769c25fff9326d499a9797cb5b2ed6a`. Durable bundle:
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-additive-uncertainty-20260923`.
Use `RESULTS.md` for the owner assessment, `companion-r2` for final numerical
products and `validation` for exact-source independent synthetic evidence.
Initial `companion` and development48/192 validation results remain immutable.

All three256-trial ensembles succeed;1/1/2 finite bound-limited fits remain
included. Unchanged source peaks816.311/554.151/593.401 mJy/nominal beam;
centered additive refit SD8.633/6.024/6.596, MC mean shifts+0.618/+0.196/+0.298.
Fixed-template analytic/simulated SD12.083/12.678,7.910/7.851,5.935/5.730.
Each analytic/simulation check passes its finite-MC interval. Base SD precision
is approximately4.4%, conditional on the fitted model only. No bias correction.
All base source depths are supported; no base negative PSD projection or missing
lag cells; doubling generation domain leaves fixed variance unchanged.

Scientific assessment is separate from those implementation checks. a1100 and
a1400 are defensible **provisional conditional additive estimates**, with regional
normalized second moments0.611–1.204 and0.834–1.182, limited independent blanks,
Gaussian generation and finite synthetic calibration accuracy retained as limits.
a2000's6.596 is **assumption-dependent only**, not an established observational
error bar: regional second moments0.381–1.747 expose spatial amplitude variation
remaining after coverage normalization. Network11 source/north/south fractions
49.7/24.0/78.1% show different contributing populations without establishing a
causal network attribution. North/south matched64-trial SD ratios1.188/0.749
are material; only five overlapping fixed-template blanks and no complete refit
blanks exist. More Monte Carlo draws cannot repair representativeness.

Depth-supported sensitivity ranges using64 matched trials are8.258–8.943,
5.913–6.236,7.271–8.640. These are not confidence bounds, nor proof of regional
representativeness. The automatic full ranges/plots also show explicitly flagged
unsupported variants: a1400 south(1 fit pixel), a2000 mask5(10), south(2).
The owner report distinguishes them; do not use their endpoints as empirical
uncertainty bounds. Paired-bootstrap intervals quantify MC comparison precision
only. Lag48 changes the generation domain and weakens paired-noise coupling,
leaving its comparison particularly sampling-limited. Base256 remains separate.

Existing fixed-template blank observed/predicted RMS SD at their own locations:
12.924/13.340 (62 overlapping placements),8.346/8.239 (28),4.786/6.552 (5).
These are descriptive comparisons of different statistics with overlap and local
means, not independent-placement significance. Same-estimator source-plus-noise
refits cover four a1100 and one a1400 prior locations with32 trials apiece.
Long-mode physical amplitudes remain unknown despite small/zero response to some
unit perturbations. Covariance range projections do not establish a zero omitted
long-scale variance. All original source/map/cleaning/calibration/weight/FRUIT
state remains unchanged.

Final synthetic192/16 run reproduces the development results above, now bound to
exact runtime source. It includes1,152 independent observed fits and18,432 inner
refits, with about5.1% actual-SD sampling precision. Long-range fixed variance
underprediction19% and deliberate spatial misspecification remain limitations.
Winged-source noiseless fitted amplitude235.434 versus physical peak250 isolates
−14.566 model mismatch from approximately+0.106 noise-related mean shift.
These tests do not validate PTC recovery; optional frozen-MAP detector-noise test
was not performed, as explained above.

Companion52.328 s wall, synthetic validation32.164 s, separately timed before
native build/verification. Four workers, one math-library thread each. Recorded
loading/inference/generation/fitting/plotting timings; peak parent472.3MB, peak
individual child135.7MB (not simultaneous aggregate). Original six-plane MAP
hashes and source-measurement hash preserved. Verified107 final companion files,
3 validation files,30 companion FITS HDUs and18 unchanged parent HDUs. Regenerated
trials0/127/255 per array reproduce all parameters and fixed amplitudes exactly.
All768 base trial parameters agree exactly with the initial committed run.
All seven scientific figures inspected visually.

## Repository gates and supplemental native recovery

Final successor suite222 tests/111 subtests passes. Focused additive suite7 tests
passes, independently reproduced. Failure coverage tests aggregation/report helpers
and the actual one_trial handler, not a fully mocked command end-to-end. Required
config preflight passes without drift. Initial baseline failures were missing
native executable environment failures, preserved in evidence.

Restored exact runtime-source CLI in
`/private/tmp/citlali-point-additive-build-20260923` from the preexisting4c31
FetchContent source cache and documented AppleClang/Homebrew realization, with
all dependency install switches off and fully disconnected FetchContent. First
configure omitted cached FETCH switches, allowing a default Conan helper download
attempt and failing before configuration; copied those existing switches and
reran successfully. No dependency installation or repository build-system change.
Cached dependency dirtiness predates this task and is recorded. Build completes
with two preexisting glog deprecated-syscall warnings and duplicate-library linker
warning, no compile errors. CLI version embedsd70233e1c. A temporary worktree
build symlink supplies the existing protocol test path, then is removed.
**All207 baseline tests/137 subtests now pass.** Binary, hash, configuration,
compiler/dependency identities, logs and failed first attempt are archived under
`evidence`. This is supplemental local protocol-gate evidence, not representative
Spack/Unity qualification or a new native reduction. Full native CTest remains
unchanged prior evidence, not newly rerun for this Python-only diagnostic.

No native scientific code, Engine ownership, upstream product, source model,
fit/MAP weights or FRUIT state changed. No integration, push or activation.

## Independent numerical review

Fresh-context reviewer independently reviewed initial0aaa and repaired runtime
d70233e1c, verified all107/3 manifest entries and18 POINT source hashes, unchanged
parents, reproduced trial scatters and direct dense pixel-pair qTCq, and reproduced
matched-budget sensitivity ratios and7 focused tests. No remaining implementation
finding. Scientific/behavioral pass with the per-array limitations above;
architecture/ownership pass; numerical-artifact evidence pass. Broader closure
requires review of this completion-document candidate and restored baseline gate.
Final companion manifest SHA256
`d650deddab4886d9ad386183a6425e5309643cab8effb82fd645106dde3f38c8`;
validation manifest SHA256
`b93cfbac6724a8cbbe97b2752d9782efc9fb1aa8d574b81b7fa7d5107b1b597c`.


## Final independent disposition and stop

Evidence candidate `3a544f3c93c1f5109765e0180e725992f3370154`, tree
`37a91cc66e3b2ef4f3ab84a884e4070b1931bf08`, received independent fresh-context
review: **scientific/behavioral pass with recorded limitations; architecture and
ownership pass; repository/evidence pass; no blocking or open findings.** Only
completion handoff/status documents changed after reviewed runtime d70233e1c.
The reviewer checked the owner report against artifacts, source snapshot digests,
archived CLI hash/version, passing207/137 baseline and222/111 successor logs, and
independently reran all four native protocol tests successfully. The failed initial
configuration, preexisting cached dependency modifications, supplemental build
scope and native CTest not rerun are explicitly disclosed. Owner report SHA256:
`a6e8fd9c942d10abce851211e3f4cb794f7ed923f515a7ec0ce250f64849bf30`.
This closure records review only; it changes no executable behavior or numerical
product. Scope stops at companion plus assessment. No integration,push,activation,
production qualification, FRUIT rule or response experiment is authorized by
this review. Unrelated main-checkout work remains untouched.
