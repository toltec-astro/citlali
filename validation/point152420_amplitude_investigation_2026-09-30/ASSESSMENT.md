# POINT152420 amplitude investigation — Stage A / first causal batch

**The strongest quantified lead is calibration.** The historical and successor
APTs assign very different flxscale values to identical native detectors. In
networks1–4 the median successor/historical ratios among common flag-zero rows
are0.421,0.446,0.447,0.400. The actual successor atmosphere multiplier is only
about1.0333. The corresponding total-conversion medians among common CAL-available
detectors are0.439,0.453,0.465,0.411. These are detector distributions, not a
predicted map ratio. No absolute-calibration winner or map-level attribution
is established yet. Batch1 measures the conditional effect before PTC projection.

The fitter defect does **not** explain the endpoint factor of two. Corrected
common fits give B T5=291.810314 and redu09=574.587713 mJy/nominal beam: difference
282.777399, ratio1.969045. Gaussian area ratio1.611490 and angular-integral
ratio3.173097. The latter has brightness×arcsec² units, not absolute flux.
B remains a comparator, Q parked, photometric/source-profile acceptance open.

## Recovered provenance

Path abbreviations (all local):
- H: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/pointings/reduced/redu09`
- B: `/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930/B-REF`
- HA: `/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/apts/hero/apt_152420_matched.ecsv`
- SA: `/Users/gwilson/work_toltec/local_data/citlali-validation/v2/science/apts/v2/apt_152420_matched.apt-v2`
Exact present-day SHA256 bindings, resolved config, recovered log lines and prior
sampling-check evidence are in results/provenance-audit.json. Factor distributions,
20 deterministic native conversions and realized populations are in
results/calibration-audit.json. No Unity recovery was needed.

| Property | Historical redu09 | Successor B | Evidence / remaining limit |
| --- | --- | --- | --- |
| Target/input acquisition | 152420/0/2, Feb19 2026 09:33:45 raw files; one of eight observations in this reduction job | Same target; four network inputs; assigned times1771493634.85377–1771493697.85025 | H/citlali_o152389_0_2_c8.yaml; B/inputs/observation.json and per-network inputs. Native tone/frequency identities exactly match all5518 APT rows; original KIDS solver histories are not asserted identical. |
| a1100 network population | 0,1,2,3,4; actual positive-weight/some-unflagged detector counts241,315,339,364,355 | Networks1–4 only; CAL-available counts261,304,212,376 before segment-specific PTC support | H/152420/raw/*stats.nc, B/upstream/products/networkN/donor-continuity/cal/receipt.yaml. Historical counts establish contribution, not exact source-crossing acceptance. |
| Calibration parent | matched calibration152307,3c273, a1100 flux3896.367905; hero geometry/flags overlay explicitly preserves flxscale | canonical matched-v2 baseline148670, a1100 flux3127.216073; Aug28 baseline producer | HA metadata; SA baseline manifest/table; v2/beammaps/3c273/reduced/redu00/citlali_o148670_0_2_c1.yaml. Source-flux ratio alone is insufficient to explain detector-factor distribution; response estimation also differs. |
| Conversion | x×flxscale once; mJy/beam unit factor1; extinction disabled | x×selected child flxscale once×sample atmosphere; mJy/nominal beam, 272GHz, alpha0, Xref0 | Exact historical source ffc6b907 calib.cpp/calibrate.h; resolved config and chunk logs. Successor CAL receipt; all1153 available detectors reproduce saved CAL as filtered-x×probe multiplier exactly. No duplicate parent rescale. Atmosphere physics itself not independently certified by this arithmetic check. |
| APT realization | HA nw,array,flxscale exactly equal historical stats snapshot | Recorded selected row and flxscale equal SA | results/calibration-audit.json; native key is(nw,kids_tone), exact array/tone_freq/kids_f_out verification. |
| Coordinates | recomputed telescope file; native AltAz coadd; hero offsets | original telescope; bounded FK5/J2000 detector coordinates using selected APT and recorded pointing | H config; B/inputs/ast-source-binding.json and CAL AST bindings. Offsets differ, with outliers; no time-dependent coadd registration is assumed. |
| RTC/sample schedule | low-pass20Hz, factor2 decimation, source-protected despiking;12 final chunks | retained B RTC, native factor1/phase0;6 processing segments per network | H config/chunk logs/rtcdiag; B receipts. These are different realized schedules; frozen controls retain B's complete sample state. |
| PTC | standard PCA5, array grouping | observed-entry ALS10 by network/processing segment, warm full resolving in FRUIT | H config/FITS; B PTC receipts. Rank difference remains a hypothesis, not attribution. |
| MAP | JINC1arcsec, R1.5,a1.1,b1.67,c2, validated detector weights, cut0.1 | containing-pixel2arcsec, uniform coefficient, cut0.5 | H config/FITS and historical map.cpp; B M0 plan/operator and forward_map.cpp. Both are normalized brightness, not numerator-only products. Same BUNIT does not reconcile calibration/beam response. |
| Feedback | initial iteration0 plus four applications1–4; previous-map reads within same redu09; learned weights finalized after1 | initial M0 plus five applications; T5 consumes m4, m5 measured only | Bound historical log lines; B receipts/history. Historical MEDRMS absent: requested SNR100 gate disabled in execution. |
| Historical trajectory retention | save_all_iters=no; writes reuse H/152420/raw; initial/iter1–3 maps not recovered | all M0–T5 maps/measurements retained | Other redu00–08 jobs measured only as context, never substituted as redu09 parents. Earlier V2 is not its no-feedback parent. |

The calibration factors are the earliest **quantified brightness-scale** divergence
located so far; this does not exclude earlier RTC/pointing differences. Network0,
source-crossing selection, offsets, adaptive cleaning, weighting and historical
feedback remain possible contributors, particularly to the width/background.

## Measurement repair and implications

SciPy1.15.2 failed macOS loading of its PROPACK binary despite matching installed
RECORD hashes. An isolated official scipy1.17.1 wheel passed optimizer, convolution
and PROPACK checks, then replaced SciPy only in the designated TolTEC venv, leaving
NumPy2.2.3 and other packages unchanged. The wheel/version/smoke record and prior
pip freeze are saved. No machine-wide packages were changed.

The shared fitter now optimizes a free periodic angle and normalizes it only for
reporting. Equivalent starting ellipses are canonicalized; the analytic Jacobian
already used for POINT-EXPECTED measurement is shared with the solver. This avoids
spurious finite-difference angle sensitivity at a circle. Gaussian+constant,
unweighted objective, physical bounds and convergence tolerances stay fixed.
Tests compare images/objectives for periodic and axis-swapped representations,
old-boundary crossing and near/exact circles, plus independent derivative checks.
No claim of global optimality follows just from an interior successful result.

| Saved stage, remeasured only | Peak | Major×minor FWHM arcsec | Constant |
| --- | ---: | ---: | ---: |
| B M0 |223.27795|10.8139×4.8415|−2.0350|
| B T1 |273.55849|10.8423×5.0370|−1.2370|
| B T2 |287.03714|10.8621×5.1486|−0.9826|
| B T3 |290.74317|10.8637×5.1755|−0.9159|
| B T4 |291.44817|10.9624×5.1896|−0.8794|
| B T5 |291.81031|10.9802×5.1927|−0.8685|
| redu09 |574.58771|12.5668×7.3116|+32.5924|

Old bound-angle T1/T2/T4 peaks265.4188/278.7874/283.1223 are superseded as saved-map
measurements. The old algorithm under new SciPy reproduces all six executed B
parameters within5e-7, separating the environment change from the fit repair.
The Q-core-minus endpoint also changes261.6365→269.5305; earlier peak-derived
endpoint claims involving that constrained fit are not repaired by retaining
its old uncertainty/response values. This does not reopen Q adoption.

On the six original B maps with their fixed original noise context, candidate
membership is unchanged18/20/20/21/21/21; every guard remains accepted. The largest
candidate change is0.02358mJy/beam, from the fitted constant at T4 (under0.009% of
T5 peak), versus the282.78mJy discrepancy. Thus no consequential admission or
source-scale feedback effect is demonstrated. This is not a corrected executed
trajectory, nor a bound on nonlinear adaptive propagation. The first batch uses
M0 and zero feedback, so it does not inherit these candidate changes. A corrected
full reference remains available if later adaptive controls require it.

Native fitted centroids are B T5(+4.729,+7.409)arcsec in its FK5 tangent axes and
redu09(+7.353,−4.943)arcsec in AltAz axes. Compare invariant widths/radial profiles,
not these vectors by subtraction. Independent45–60arcsec median backgrounds are
−0.471 B T5 and+36.092 redu09, consistent with a historical pedestal. Historical
shoulders and30–40arcsec negative bowl remain after this subtraction. Profiles
recenter every map on its own fitted centroid, use fixed angular bins, and
preserve unavailable support (one outer B pixel missing; its full60arcsec aperture
is unavailable). The prior bound2arcsec averaging result559.89 is verified and
reused; it does not remove the discrepancy. A constant offset alone cannot
explain amplitudes already defined above a fitted constant.

## Accounting and next discriminator

B native receipts bind each actual selected sky model and apply one subtraction
and one restoration in the same contribution domain. All five T−R−A closures
are recorded; T5 maximum absolute error6.71e-12mJy/beam. This excludes an omitted
or duplicate add-back in that exposed accounting, not all possible sky-model
biases. Historical code/config/logs distinguish its upper/local-background/adaptive
support and evolving weights; overwritten intermediate map bytes prevent a
common-fit historical peak trajectory. Uncleaned atmospheric signal is not used
as source truth.

[Batch1](BATCH1.md) uses the complete saved M0 input/basis/validity/MAP checkpoint
for replay, common-population current conversion, and historical conversion on
that common population. These three frozen controls are charged to the six-start
planned allowance. They can establish a causal effect conditional on B's operator;
fully relearned consequences and the residual historical profile contrast remain
open. No numerical batch has started at this assessment snapshot. Source/tests,
resource admission and independent review must be complete before the owner
foreground command is issued. No source tuning toward574.59, cleanup, production
change, push, or remote work.
