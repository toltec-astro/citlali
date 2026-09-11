# One coherent-source feedback experiment

2026-09-11. Identity SCI-FRUIT-POINT-COHERENT-SCREEN@r0.1.

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md),
[reviewed method recovery](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md),
and [new owner direction](OWNER_DIRECTION.md). Adopt the shared conditional core,
POINT intent and provisional recovery goals, original-parent replacement,
residual relearning and separate feedback/background roles. Cite the historical
implementation for producer identity and practical controls only. Supersede the
paper-only next step and historical-environment prerequisite for this screen.
Keep weighting closed. This is an isolated empirical development experiment;
it does not adopt production source/profile successors or claim conformity.

## Hypothesis and matched arms

A single elliptical Gaussian source, refitted from the complete reconstructed
map on every pass, can retain coherent low-amplitude source structure that a
pixelwise threshold discards, improving recovered amplitude or reaching useful
recovery sooner. The Gaussian is a minimal compact-source hypothesis, not a
claim that every POINT source is Gaussian.

Both arms start without feedback. At each pass, form rho=Y−Pi(m) from the same
immutable pre-PTC parent Y, relearn centering and rank-5 PTC state on rho, clean,
rejoin the same model representative, then ordinary-map the complete total.
Infer a replacement model from that total, allowing entry and revocation.
Never accumulate increments or clean a previous cleaned result as the parent.
Retain seven passes including bootstrap. Evaluate every pass against cumulative
measured wall time; seven is a resource bound, not a stability or optimum claim.
Earliest truth-qualified pass is an evaluation statistic, not a deployed stop rule.

Reference P: keep each total-map pixel in fixed D where T>3R, and put an explicit
zero correction elsewhere. Candidate G: fit T with a signed-amplitude elliptical
Gaussian plus a plane; use only the Gaussian as feedback when its fitted positive
peak exceeds the same 3R and the fit is admissible. Otherwise reject the entire
source correction with recorded cause. Background is measured and retained for
diagnostics/evaluation only. No amplitude, centroid or width prior enters the fit.

R is 1.4826 times the median absolute deviation about the median of current
supported pixels at radii 90–150 arcsec. It is a declared empirical map-scatter
score scale, not inverse variance, a probability, a NOI-qualified uncertainty or
an OG MEDRMS alias. Both arms use the same rule and threshold; no threshold search.
Require at least 100 pixels and finite R>0; otherwise fail that trajectory.

The candidate is fitted unweighted on D inside radius 90 arcsec about the map
coordinate origin. Centroid is free in [−80,80] arcsec on each axis; each FWHM is
free in [4,60] arcsec; angle is free modulo pi; amplitude and plane coefficients
are unrestricted signed real values. These broad search/resolution limits are
not nominal source parameters. Three deterministic numerical initial widths
6,18,42 arcsec start at the largest positive plane-subtracted pixel; the converged
least-squares solution with smallest SSE wins. Initial widths are starts for the
same fit, not three scientific models. A centroid/width limit hit is recorded as
compact-model rejection, with zero feedback; solver failure is unavailable and
fails the trajectory. Negative/low-amplitude fitted sources are resolved rejections.
No failed solve is relabeled a zero or carried previous model.

## Common numerical and input definitions

Use the exact 123424 RTC export in INPUT_PREFLIGHT.json. Producer source
785d18ec writes the calibrated pre-PTC matrix before model subtraction/cleaning.
It is a legacy float32 export with mJy/beam labeling and extinction disabled;
this experiment tests recovery in that representation, not qualified absolute
TOA flux or optical calibration. Hold that one upstream realization immutable.
The saved scalar configuration, detector/row identities, geometry and masks are
bound. Recover the writer's chunk-index offset defect using cumulative recorded
endpoints, independently corroborated by all twelve chunk-summary lengths.

Use the fixed input finite/flag-zero/APT-flag-zero occurrences. Per segment and
network, center each participating detector by its valid-sample arithmetic mean.
Build the centered pair-overlap covariance (cross-product divided by joint count
minus one); pairs without two occurrences have zero fit influence. Learn its five
largest positive eigenmodes once. At each available group-time solve the binary
masked 5-by-5 normal system in the fixed basis; require smallest eigenvalue
>1e-10 times largest. No lower-rank, learned-mask, inverse-noise scaling or support
fallback. Discard centering; retain it and the learned basis as state. Empty
input group-times remain outside the fixed population. Insufficient rank or a
required solve failure fails the trajectory. This harness is explicitly bound;
no equality to the historical executable's projection is claimed.

Network grouping is the common operational starting control; the earlier
array-wide proposal was not approved. Rank remains 5 everywhere. Ordinary MAP
uses 2 arcsec half-open containing pixels, gamma=1, coverage_cut=0.1 and the
frozen sorted-index normalization/science-support formulas. Build the full grid
from valid input coordinates; no outcome-driven crop. D is its bootstrap science
support intersected with radius 90 arcsec, fixed across passes/arms. The scale
annulus uses the same fixed science support. Source models are zero outside D;
measured samples outside D follow residual-only processing. Pi and B match.
No detector reweighting, recentering of the WCS or dynamic support selection.

## Small known-signal/null set

Use two predeclared PCG64 seeds, 20260911 and 20260912, on 123424's exact geometry
and fixed masks. Synthetic nuisance has three network-local AR processes
(coefficients 0.98,0.90,0.70), random detector loadings, and detector-local AR(0.3)
noise. Detector scales are the robust first-difference scale of the immutable
RTC parent; this sets realistic units/heterogeneity without asserting a noise
model qualification. Common-process loading amplitude is five times each scale.
No field of these simulations is an inference input except the synthetic measured
matrix, its fixed masks/coordinates and the same runtime policy.

For each seed run null and a known Gaussian of peak 100 mJy/beam, centroid
(17,−11) arcsec, FWHM (12,8) arcsec and angle 25 degrees. On seed 20260911 also
run a mismatch source: that Gaussian at peak 80 plus a peak-20 companion with
the same widths, offset (+10,+6) arcsec. These are five synthetic parents total,
plus the real parent: six cases, two arms, seven passes = at most 84 PTC passes.
They are a small conditional stress test, not an ensemble bias qualification.

Known signal is defined on the ordinary grid and projected into the immutable
pre-PTC parent before any learning. Each arm/case/pass relearns independently.
For each injected case, subtract the corresponding same-seed null output at the
same pass and compare that signed response with the exact injected grid signal.
This tests response including learning/model changes. Also retain absolute
outputs, null feedback promotion and mismatch residuals, so paired subtraction
cannot hide false source formation. Truth never enters fitting or feedback.

## Measurements and decision

For real 123424 report per-array centroid, signed amplitude, two widths, angle,
Gaussian integral, direct signed aperture sum, fitted plane, residual RMS and
exterior structure; support, model admission, fit/solve failures, state/model
changes, cumulative wall time, component timings and peak process memory.
Measure total reconstructed maps, not just their Gaussian feedback components.
Use the same free fit as an evaluator on both arms; separately retain direct
truth errors/aperture recovery so a good Gaussian fit cannot hide mismatch.

For injected responses report peak-fit bias, centroid error, each width bias,
source/aperture and full-D error, and exterior leakage. For mismatch use the
exact composite truth and its separately evaluated best-fit Gaussian, together
with direct truth-map residuals; a Gaussian-summary fit alone cannot pass it.
Targets retain the provisional 5% amplitude/width and 0.05 minor-FWHM centroid
scales on adequate Gaussian cases. Compare errors with the matched reference;
no universal confidence interval or probability is claimed from two seeds.

Keep for this development lane only if G provides at least one useful benefit:
at least two percentage points less absolute amplitude bias where the reference
misses 5%, or at least one fewer pass and 10% less measured method map-ready wall time to
joint amplitude/centroid/width targets. Require the benefit on both Gaussian
seeds in at least one array, no >2-percentage-point amplitude degradation in
another required array, no >10% direct error/leakage increase in the mismatch
case, and no new false coherent-source promotion in either null. Real-data
centroid changes >0.05 minor FWHM or width changes >5% versus reference/OG require
explicit interpretation, not an automatic benefit claim from a larger peak.

Revise once only if one specific identifiable candidate weakness explains a
failure and suggests a bounded repair to this source model/admission. Record
and freeze the change before testing; rerun only the affected comparison set.
Otherwise reject this candidate for further development. No rank/threshold/
grouping sweep. A keep decision is provisional pending frozen 129081 replication;
freeze before opening its scientific values and never tune on replication.

## OG and resource boundary

Benchmark against the identified 123424 alpha=1 historical-recurrence control
in the existing EL-F2 run, with all seven retained products, ordered configuration,
version and OS timing receipt. Its extra diagnostics, JINC/weighting/learning,
upstream work and execution environment differ from this harness. Report actual
OG pointing/shape/amplitude and full latency separately. Matched-arm core timing
can establish relative benefit; it cannot alone establish an end-to-end speedup
against OG. No unavailable historical environment reconstruction is required.

Use one local process with four BLAS threads; retain process/thread/library
identities. Bound the initial campaign to two hours, 8 GiB RSS and 4 GiB retained
output. Preserve all maps, models, scales, fits, learned centering/bases, masks,
input identities, timings and failed attempts; transformed streams are exactly
reconstructible from the immutable parent/model and stored Apply state. No
production code or reduction product is overwritten; no Unity access occurs.

Timing records distinguish inclusive development wall time from method map-ready
wall time. The latter subtracts the reference's evaluation-only source fits,
charges every preceding candidate source fit as inference, and stops at the
current complete map before its next-model inference. Current unused terminal
inference is retained/timed but cannot inflate time-to-current-map. Both curves
and their component accounting are reported; common input setup remains separate.
