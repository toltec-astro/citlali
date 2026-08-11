# SCI-ALIGN-001 direct Lissajous timestream timing diagnostic

## Historical run outcome

The exact PTC-timestream estimator, synthetic gates, real anchor, objective
profiles, complete-scan blocked model comparison, sensitivity checks, and
paired timestream/map bootstrap were implemented and locally validated. The
frozen 22-pointing run then stopped after 9 observations because ObsNum
136280 remained multimodal at the pre-specified 1,500-realization bootstrap
ceiling.

This is a successful protocol stop, not a complete corpus measurement. No
summary over all 22 pointings, frozen high-S/N pointings, or 11 beammap
brackets is reported. The 13 unopened pointings remain unexamined.

A later read-only audit, documented in
`REAL_BOOTSTRAP_OPTIMIZER_FAILURE_02.md`, found that the stopping distribution
was numerically contaminated: 361 fits at exactly the point estimate had
L-BFGS-B iteration count zero, `success=false`, and message `ABNORMAL` but
were retained because their objectives were finite. The table below is
preserved as authenticated run history. The ObsNum 136280 interval and all
paired statistics that use those timestream draws are not valid scientific
uncertainties.

## Estimator and safeguards

The primary model evaluates each detector coordinate from the complete
scan-local telescope state at `t + tau`. Angular states are unwrapped within
each scan before interpolation; interpolation never crosses scan boundaries.
One fixed support trims 50 ms from both scan edges for every tested tau and
model. The exact whole-scan bootstrap unit is one complete `scan_indices` row,
including all retained samples, eligible a1100 detectors, PTC flags and
weights, profiled detector amplitudes, and its nuisance baseline.

The source center is free. Primary detector-scan baselines are constant and
fit only outside 45 arcsec; source scoring is inside 35 arcsec, so the source
crossing cannot train the baseline. Fixed/free beam and constant/linear
baseline sensitivities are reported without choosing the version closest to a
prior answer. Constant, lag, direction-sign hysteresis, and joint models are
ranked by complete-scan held-out prediction rather than sample-count BIC.

The map comparison uses the same deterministic whole-scan draws wherever
paired. For each draw it forms
`delta_tau = tau_timestream - tau_map_coordinate_shift` directly and reports
the paired interval, covariance, and correlation. The authenticated map point
estimate is never treated as uncertainty-free.

## Anchor discipline

ObsNum 150818 was the sole real anchor. Changes prompted by it were made only
for documented numerical or coordinate-convention failures: seconds-scaled
optimization, explicit finite-difference scale, and the algebraic map sign
conversion. Each repair has a dated failure note, full synthetic rerun, and
successor implementation freeze. No change was triggered by proximity to a
particular tau value.

The anchor result was:

- exact tau `+5.162391 ms`;
- scan-bootstrap median `+5.415302 ms`, 95% interval
  `[+1.041215, +9.238817] ms` from 500 successful resamples;
- sign-aligned map point estimate `+5.549282 ms`;
- paired difference median `-0.535190 ms`, 95% interval
  `[-3.801745, +3.275917] ms` from 250 identical scan draws;
- covariance `4.367083 ms^2`, correlation `0.741679`.

## Opened-observation record

The sign convention here is the exact coordinate convention: positive tau
evaluates the telescope state later than the recorded PTC sample. `Delta` is
timestream minus sign-aligned map coordinate-shift tau. Intervals are
whole-scan bootstrap intervals.

| Pointing | Gate | tau point (ms) | tau 95% (ms) | map (ms) | Delta point (ms) | paired Delta 95% (ms) |
|---:|:---|---:|:---|---:|---:|:---|
| 131920 | pass | -28.302 | [-30.297, -26.172] | -32.836 | +4.534 | [+0.809, +8.431] |
| 131926 | pass | +8.678 | [+4.045, +12.457] | +12.040 | -3.362 | [-5.689, -0.612] |
| 133542 | pass | +10.213 | [+6.890, +13.253] | +12.417 | -2.204 | [-6.270, +0.876] |
| 133544 | pass | +8.375 | [+5.595, +10.564] | +11.257 | -2.881 | [-7.216, +0.820] |
| 135396 | pass | +12.897 | [+8.960, +17.086] | +15.602 | -2.705 | [-7.147, +1.516] |
| 135398 | pass | +9.746 | [+3.579, +23.884] | +17.952 | -8.206 | [-13.791, +4.273] |
| 136278 | pass | +12.886 | [+6.052, +15.981] | +12.843 | +0.043 | [-5.930, +5.013] |
| 136280 | **stop** | +11.728 | [-9.290, +27.957] | +7.790 | +3.938 | [-27.614, +25.915] |
| 150818 | pass (anchor) | +5.162 | [+1.041, +9.239] | +5.549 | -0.387 | [-3.802, +3.276] |

These nine rows are diagnostic history, not a post hoc cohort. In particular,
their signs and median must not be generalized to the selected 22.

## Stop evidence, audit, and interpretation

At the 1,500 maximum, ObsNum 136280 had two qualifying KDE peaks and 361
replicates exactly at the original optimizer start. The successor optimizer
probe proved that representative pile-up fits were abnormal zero-iteration
stops. Frozen multistart retries converged to different tau values with lower
objectives. The paired covariance `118.547553 ms^2` and paired 95% difference
interval `[-27.6,+25.9] ms` are therefore contaminated historical values, not
an uncertainty statement.

The scan audit also found genuine resample sensitivity beyond the numerical
spike. Pile-up and moved draws have the same median unique-scan count and
effective scan size, so deficient scan diversity is disfavored. Nearly
opposite scan rows 6 and 7 pull tau in opposite directions, and tau correlates
with resampled residual MSE. Together with the owner's inspection of the
modest-S/N structured map, this means a repaired bootstrap may remain broad
or model-sensitive for scientifically real source/background reasons.

This result prevents the planned corpus comparison. It does not falsify the
map-space effect, prove that a physical delay is absent, locate an upstream
cause, or authorize a timing correction. The analysis only tests relative
registration of delivered PTC signal and delivered telescope-coordinate
trajectory and remains dependent on upstream PTC processing. No repaired
real-data result has yet been computed.

## Reproducibility and repository scope

The diagnostic started from commit
`6ec08656fd5c12607e806f55389cc094aa4b6a2d`. The current frozen protocol SHA256
is `8bbfe6016d40b7e966ee8d4ee8ef3127162fe365c39133cac4586214487c26ce`;
the frozen selection SHA256 is
`b6e517112988cfe2cea8846cd474cca3649beb7a5355f19c8a6e18074870020a`.
`partial_input_identities.json` binds every opened PTC, PPT, authenticated map
result, observation result, and observation checksum manifest.

`PREEXISTING_WORKTREE_STATE.txt` lists all 32 owner-owned untracked bundle
files present before branch creation. They were not modified. A clean scoped
diagnostic diff is claimed; a globally clean worktree is not.

The point fits and initial 500-realization bootstraps bind protocol revision 4,
SHA256 `5366dd8cfe963e29bf273a7c764637f9b85586f211963920acdc95b2610f9ad1`.
Revision 5 changes only the convergence gate to require unimodality and binds
the unchanged estimator implementation; its SHA256 is
`da4cbc9385fcd02630592ae77a8bb3e1ecbbe4591edc9010d82e74e202060d48`.
The historical 136280 successor result records both protocol identities and
preserves the initial compact result byte-for-byte. Revision 6 is the
pre-real-rerun optimizer-control freeze; its SHA256 is
`8bbfe6016d40b7e966ee8d4ee8ef3127162fe365c39133cac4586214487c26ce`.
