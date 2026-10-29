# Experimental temporal-change method

Status: synthetic study only. The optical-response default remains 0.10 and
configurable in the existing application. This prototype does not recompute
optical eligibility, modify any native membership, or decide retention. Its
result explicitly has `admission: not_decided` and `harm_threshold: null`.

## What is measured

For each fixed peer reference x and source-free synthetic target y, the null is

    y(t) = offset[science block] + g*x(t) + error(t).

The alternatives add one column x(t)*h(t), where h is each of the five block
boundary step functions or a full-observation linear drift from zero to one.
The exact same six block offsets are present in null and alternatives. This
separates block offset jumps from response changes, but deliberately discards
between-block mean information about response. It needs within-block excitation.
It does not remove within-block disturbances; those remain an independent concern.

Each contrast is the coefficient of its added column after projecting out the
null columns (Frisch–Waugh–Lovell). Estimates are in dimensionless response units,
normal response one. When the injected mean is exactly a single step/drift, its
matching coefficient is that physical change. Other contrast coefficients are
projections, not independent physical gains. Simultaneous intervals cover all
these projections, not a reconstructed gain curve or a map-impact upper bound.

Two panels are analyzed together. Each bootstrap uses the same wild signs for
their shared target, and the statistic is the maximum absolute standardized
contrast across all twelve panel/shape combinations. This includes the search
penalty and panel sensitivity; two panels do not supply independent votes.
Reference disagreements can cause flags, so the result is evidence against
constant linear coupling to the supplied references, not a causal diagnosis of
detector instability. No reference is selected because it gives a preferred result.

## Conditional resampling

1. Fit the null by OLS. Divide residuals by sqrt(1-leverage), the HC2 correction.
2. Estimate one AR(1) coefficient from adjacent retained rows within blocks.
   Science-block boundaries and absent original slots break dependence in this
   prototype. No links bridge gaps. An estimated absolute coefficient >=1 or
   unavailable variance returns unavailable; there is no silent clipping.
3. Keep measured references, masks, times and noise innovation magnitudes fixed.
   Multiply innovations by independent random signs, then reconstruct the AR(1)
   residual sequence. Refit contrasts through the exact fixed-design linear
   projection. Starts have observed residual magnitude with a fresh sign.
4. Standardize each contrast by its bootstrap standard deviation; calibrate the
   maximum against the full bootstrap maximum distribution. The plus-one
   Monte Carlo p value has resolution 1/(999+1). p<=0.01 is only the declared
   evaluation point, never a detector cut or claimed array-wide error guarantee.
5. For intervals, fit an encompassing model containing all six change columns,
   estimate residual dependence again, and repeat the same shared-sign procedure.
   The 99th percentile of the maximum standardized contrast error supplies a
   simultaneous, symmetric conditional interval for the twelve projections.

This is an estimated AR(1) sieve-wild prototype, not a general dependence-robust
procedure. References and their normalization are treated as fixed; OLS is not
the existing frozen Huber fit. Correlation beyond AR(1), boundary dependence,
skewed innovations, uncertain reference errors, outliers, nonlinear coupling,
weak excitation and finite-sample estimation can invalidate it. The literature
motivates resampling construction but does not establish this implementation's
coverage. Main study outcomes must determine whether nominal calibration works.
No sample count, correlation or duration cutoff is invented to hide failures.

If any searched contrast lacks numerical rank, the complete search is explicitly
unavailable. This is conservative and can discard usable partial evidence; a
future supported-subset search would require its own semantics. Non-identifiability
is never a low-response finding. Lack of detection is never proof of stability.

## Synthetic calibration, power and stress tests

Six 40-bin science blocks; source-free reference with between-block levels and
sinusoidal within-block variation. Normalized gain one, ordinary innovation
sigma 0.1, AR coefficient 0.6 unless specified. Constant-response cases include
iid/quiet noise, AR(1), heteroskedastic AR(1), block offsets, low excitation in
the last block, gaps, and t(3) innovations. Each has 1,000 independently seeded
realizations and 999 resamples. Different cases have separate deterministic
random streams. Point estimates have Wilson 95% Monte Carlo intervals; these
describe simulation uncertainty only, not real-detector coverage.

Stress cases (250 each): AR coefficient 0.95, drifting second reference, noisy
second reference, stationary quadratic response, one large impulse, no excitation,
and a constant last block. Reference drift/nonlinearity are not physical time-varying
target-gain alternatives. Report their flags as attribution limitations, not
successful detection. Interval truth is the projection of the known synthetic
mean onto each fitted contrast; this does not validate a gain interpretation
under model mismatch.

Alternatives (250 each): midpoint steps 0.05/0.10/0.30/0.80; full-observation
drifts 0.10/0.30; one-block excursion 0.30; late response 0.03 or -1. Summaries
show detection/missed counts and simultaneous coverage. Earlier stationary
counterexample seeds 7/15 are replayed unchanged from the prior synthetic fixture.

## Map tradeoffs, without an acceptance threshold

A separate deterministic toy maps a unit-FWHM Gaussian (peak20 arbitrary units)
on a 41x41 grid spanning +-4 FWHM, with four cross-linked raster passes. A smooth
atmosphere has RMS30. A fixed offset-plus-atmosphere fit, learned outside 1.5FWHM
radius, removes a constant coupling; no contaminated cleaning basis is relearned.
The unperturbed source is assumed perfectly preserved. Project the residual
contaminant with identical four-hit support and fixed cohort weight share.

For atmospheric-only coupling, inject delta_g(t)*A(t). For all-signal gain,
also inject delta_g(t)*S(t). Apply the same source-masked linear cleaner, then
map. These are deliberately different physical hypotheses. Fixed shares model
the aggregate coherent affected weight, not detector count. They do not recompute
inverse-noise coefficients; actual weighting can change them.

Sweep five changes, four weight shares, four temporal patterns and two coupling
types (160 cases). Measure off-source background RMS/max, source amplitude,
centroid displacement and width bias using a bounded Gaussian-plus-offset fit.
Fits at a bound or unsuccessful fits are explicitly flagged. The toy's noise
scale comes from 256 independent white-noise realizations passed through the
same cleaner and map, with uncleaned per-pixel sigma1. No native map normalization
is treated as a noise estimate. This one beam-unit illustration is not three
array-specific simulations or an estimate for observations 152389/152418.

Two explicit conditional tradeoffs accompany each row:

- If the same map artifact repeats over16 independent-noise observations, its
  ratio to noise increases by4. Independent or changing artifacts behave
  differently; this is analytic scaling, not a simulated coadd.
- Dropping an affected inverse-variance share f increases independent-noise RMS
  by 1/sqrt(1-f). This assumes unchanged remaining noise/coverage and correct
  inverse-variance weights; the empirical production coefficients need not meet
  those assumptions. It is a sensitivity-cost illustration, not a real forecast.

Neither fixed cleaning nor toy projection establishes full ALS/FRUIT behavior,
source feedback safety, bias on extended emission, or real cohort performance.
The temporal and map experiments are separate controlled studies: temporal
intervals have not yet been propagated to map-impact uncertainty. Harm budgets
and later owner-run validation remain open.
