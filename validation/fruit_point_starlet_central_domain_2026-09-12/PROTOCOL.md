# One matched central-domain starlet test

## Program adherence and prior-work recovery

The owner requested on 2026-09-12: **“Please review this chat and make the
suggested test.”** The linked discussion, [Reject starlet candidate](chatgpt-conversation://6aa55ebe-e144-83e9-86b1-f2768831a16d),
is retained in [REVIEW_DISCUSSION.json](REVIEW_DISCUSSION.json). This authorizes
its spatial audit and one matched central-domain estimator test. It reopens
that bounded experiment after the prior closure; it does not authorize full
FRUIT trajectories, another PTC pass, production, Unity, qualification, OOF or
new 129081 comparisons.

Under the [program charter](../../doc/scientific_contracts/README.md), adopt the
[ordinary-MAP prior-work record](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md),
the existing independent PSF and compact truth definitions, and the frozen
[starlet screen](../fruit_point_starlet_preliminary_2026-09-11/PROTOCOL.md).
Cite its negative result as full-domain evidence. Do not reopen or edit that
frozen result. This is implementation-informed empirical work, excluded from
independent contract authorship. No fresh literature claim is needed for this
retained-product comparison; the chat's inaccessible citation tokens are not
scientific authority.

## Review disposition and exact question

The chat correctly separates edge failures from central-target failures and
noise weighting from source-location information. The prior estimator already
weighted each selected coefficient by its assigned band/coverage scatter.
Low weight alone cannot constrain a freely fitted amplitude; selected-coefficient
fidelity also does not constrain unselected structure. These remain possible
joint causes. The proposed audit will measure location, coverage-normalized
selection rates and central contamination without assuming the answer.

Its stronger phrase “recoverable information” is narrowed here: the brighter
coma passes a descriptive oracle detectability marker, which does not establish
recoverable unknown morphology, unbiased brightness, or deployable detection.
Inter-array alignment is not an absolute pointing-position bound and is unused.

One experimental central radius is fixed **before this spatial audit and before
new estimator results**: 60 arcsec around the existing commanded-origin map
coordinate (0,0). It reuses the existing 60-arcsec source-evaluation length scale,
centered now on the commanded origin rather than known truth. It is a bounded
trial domain, not a newly qualified POINT pointing-error range. No radius scan
or outcome-dependent region choice is allowed. The original sources have
centers (17,-11) and (19,-10); an additional stress case at (35,-11) tests a
larger absolute offset, without setting the model's fitted centroid.

## Matched definitions

F is the original domain D (supported radius <=90 arcsec), with the explicit
zero-MAD repair below. C uses D intersect radius <=60 arcsec for both eligible
coefficient centers and nonnegative reconstructed source pixels. The rest of
the full map stays available for convolution and the original outer-annulus
plane. There is no multiplicative spatial taper, recentering on a peak or truth,
new source detection gate, penalty, cross-array prior, or post-fit clipping.
The model is inferred directly in its declared domain.

The transform, original S and complete-support masks, signed four detail bands
plus coarse band, threshold 5, L-BFGS-B objective, zero start, maxiter300,
maxfun3000, ftol1e-12, gtol1e-8 and success/relative-gradient <=1e-4 gate are
unchanged. The solver limit will not be raised if the smaller domain fails.

Noise calibration is performed on the **original D** using only saved null
20260911 map1, exactly as before. Retain the original Q median, Q-stratum labels,
all 30 scatters and eligibility counts in both arms. Setting C must not silently
recalibrate on a quieter subset. Background uses the original O in both arms.
A central coefficient can depend on noisy outer pixels; only complete S support
makes it eligible. No additional measurement-quality mask is introduced, so the
experiment isolates the location-domain hypothesis.

The separately disclosed normalization repair is shared by F and C: use the
current positive finite outer-residual MAD as before. If that MAD is exactly
zero, use the fixed positive outer-residual MAD from the calibration null for
that array. A nonfinite or negative value remains unavailable. This is a change
of solver coordinates; scale both Y and sigma so the original weighted
least-squares objective is unchanged. It introduces no expected-flux constraint,
new threshold or stochastic perturbation. Old noiseless failures remain retained.
For nonzero-MAD maps, F is checked against the saved original outputs to verify
that the repair changed no numerical behavior there.

## Audit, cases and measurements

Before running C, audit the two retained nulls and background-only first maps.
Use fixed radial rings [0,30], (30,60], (60,90] arcsec and the original two Q
strata. For each array, scale, ring and stratum, report eligible and selected
counts, selected fraction, absolute standardized-coefficient quantiles and
empirical scatter. Report every selected coefficient's position, radius, Q,
assigned scatter, raw value and standardized value. These are descriptive
spatial measurements, not new false-alarm calibration or independently measured
per-pixel uncertainties. Report false model peak locations and brightness
inside/outside 60 arcsec, including any inward spread from selected edge
coefficients. Do not infer central safety from peak locations alone.

Then evaluate F and C on all 25 saved maps from the preliminary screen: ten
actual processed maps (including real123424 and the brighter-coma bootstrap),
eight noiseless phase/shape cases, six estimator-only truth-plus-null cases,
and a pure plane. No map is re-cleaned. Add exactly four **map-level** offset
stress inputs at (35,-11): noiseless compact and coma4, plus each truth added
to the same saved null20260911 map. Sample the existing truths on the original
D; do not truncate the reference truth to C. These do not rerun signal-dependent
PTC learning and cannot qualify the offset case for full FRUIT use.

Total: 29 maps x 2 domains x 3 arrays =174 estimator calls, zero cleaning
passes, one process/four requested numerical threads. Cap execution at ten
minutes, 4 GiB RSS and 1 GiB of new output. Preserve failures and failed iterates.
A fixed F-then-C ordering per map is recorded; timings are descriptive and may
reflect that order. No automatic scientific revision is allowed.

Keep the previous external Gaussian fit for compact amplitude, centroid and
width; it never enters inference. Coma brightness is integrated over the
original D; image error and centroid use the original truth-centered r<=60
aperture; wings use 15<r<=60. All denominators use full original truth, including
parts outside C. Report the truth brightness and wing fractions excluded by C
before interpreting recovery. Do not renormalize a clipped source or shrink the
evaluation aperture to make C pass. Report positivity/support, model leakage,
plane, selected counts, final objective, iterations, projected gradient, timing
and every unavailable measurement. The ordinary total map remains unchanged.

## Decision rules

The spatial hypothesis has its own descriptive result: quantify selection-rate
enrichment and false brightness by ring/Q, and whether the central variant
reduces null contamination. Do not turn that narrower result into method
acceptance.

A candidate must still pass all applicable prior preliminary gates: zero models
for both nulls, background and pure plane; admission in all six actual compact
and all three actual coma4 maps; noiseless compact amplitude/width <=5% and
centroid <=0.4 arcsec; noiseless coma4 brightness <=10%, L2<=15%, centroid <=0.4
arcsec, wings <=20%; map-add compact amplitude/width <=15%, centroid <=1 arcsec,
L2<=25%; map-add coma4 brightness <=20%, L2<=25%, centroid <=1 arcsec, wings<=30%.
Use the same limits for the new offset stress cases. Retain old coma1/half and
mismatch cases as challenges without manufacturing new success thresholds.
No unavailable solve counts as a successful zero/null or recovery result.
Phase changes in noiseless brightness/image error must remain <=2 percentage
points where evaluable. Record boundary limitations independently of solver
failure. Retain <=0.5 s median array inference per map and <=2 s maximum;
unavailable calls do not establish useful completed-method latency.

A better central null result with failed recovery is a partial finding and a
failed full candidate. Passing the central test would support separate owner
review, never an automatic trajectory. The current request does not select a
production POINT domain or generalize to other observing intents.
