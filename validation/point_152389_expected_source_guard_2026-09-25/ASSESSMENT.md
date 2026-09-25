# POINT 152389: expected-source guard and recalibrated admission

All six prescribed statistical screens pass with complete outcomes and retained bright controls. These are unapplied saved-map diagnostics; no feedback trajectory was launched. Validation replays previously examined fields and is not a new untouched holdout.

## Selected measurement

POINT-EXPECTED-v1 uses one existing unweighted seven-parameter elliptical Gaussian plus constant fit on the original 40-arcsec disk (1257 pixels). Peak initialization is restricted to the original 20-arcsec AST disk; the central coordinate box remains ±20 arcsec. There is one deterministic optimizer call, maximum 3000 function evaluations, and inherited stopping tolerances. Acceptance requires finite convergence, positive peak above the constant, radial centroid ≤20 arcsec, both sigmas strictly >1.5 arcsec with no active lower width bound, and A/σA >10. The width floor is 3.53223 arcsec FWHM on 2-arcsec pixels. No goodness-of-fit, circularity, exact-beam or source-depth cut is added.

The expected origin is tangent (0,0), FK5 J2000 RA 176.7429079166667°, Dec 39.97619555555556°. Reported tangent centroids are descriptive measurements; no operational pointing correction is emitted. Nominal beams remain 5/7.5/10 arcsec FWHM. A and σA use mJy/nominal beam. σA²=qAᵀC0qA: amplitude row of the SVD pseudoinverse of the full seven-column model Jacobian (W=I), contracted through the existing FFT machinery with the full original M0 covariance. All nuisance columns, including background and any active-bound directions, remain. SVD rank threshold is eps × max(J.shape) × largest singular value; amplitude identifiability uses the null projection ≤100 × eps × max(J.shape). Circular orientation null directions are allowed when amplitude remains identifiable.

Ordinary quality rejection, finite nonconvergence or nonidentifiable amplitude gives a failed POINT message and complete empty accepted products (zero candidate inside valid support, unavailable outside). Measured maps and raw fit diagnostics remain. Unexpected/nonfinite/missing-input failures remain computationally unavailable. There is no covariance inverse, refit ensemble inside a trial, uncertainty rescaling or exact ten-sigma false-alarm claim. BASE-POINT/CORR/DIAG share this guard and fitted brightness operand; DIAG changes only the later supplemental-neighborhood variance. Original C0, K0, means, seeds, grid, masks, connectivity, gain and threshold-3 baseline remain frozen.

## Complete outcomes

Each row below uses 1024 calibration fields, 1024 replayed-validation fields, or 256 reused stress fields; each field carries paired blank and core scenes. Rejection columns are quality rejection / finite nonconvergence / computational error.

| Array | Phase | Old blank unknown | New blank accepted | Blank Q / N / E | Core accepted / count | Core Q / N / E | Blank nfev median / max |
|---|---|---:|---:|---|---:|---|---|
| a1100 | calibration | 125 | 0 | 1023/1/0 | 1024/1024 | 0/0/0 | 40/3000 |
| a1100 | validation | 126 | 0 | 1024/0/0 | 1024/1024 | 0/0/0 | 39/1221 |
| a1100 | stress1.25 | 38 | 0 | 256/0/0 | 256/256 | 0/0/0 | 40/1088 |
| a1400 | calibration | 165 | 0 | 1024/0/0 | 1024/1024 | 0/0/0 | 47/2232 |
| a1400 | validation | 173 | 0 | 1024/0/0 | 1024/1024 | 0/0/0 | 47/2344 |
| a1400 | stress1.25 | 50 | 0 | 256/0/0 | 256/256 | 0/0/0 | 46.5/1948 |
| a2000 | calibration | 233 | 0 | 1007/17/0 | 1024/1024 | 0/0/0 | 71/3000 |
| a2000 | validation | 226 | 0 | 992/32/0 | 1024/1024 | 0/0/0 | 74/3000 |
| a2000 | stress1.25 | 53 | 0 | 242/14/0 | 256/256 | 0/0/0 | 80.5/3000 |

All draws and reason counts are retained in SUMMARY.json and per-field trial records. No failed draw is dropped or replaced. nfev is the solver counter, not the number of residual calls including numerical Jacobian evaluation.

## Calibration, validation and stress

Threshold is max(3, calibration order statistic 1015/1024); strict comparisons. Joint counts are preconnectivity maxima over paired scenes. The final-event column is any connected supplemental addition. U is the original exact one-sided 95% binomial upper bound for validation; no-supplement noiseless checks also pass.

| Array | Policy | τ | Cal >τ /1024 | Val >τ /1024 | U95 | Val final events | Stress >τ /256 | Stress final events | Screen |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| a1100 | CORR | 3.83548932 | 9 | 9 | 0.015287 | 0 | 33 | 0 | PASS |
| a1100 | DIAG | 6.91256798 | 9 | 9 | 0.015287 | 0 | 33 | 0 | PASS |
| a1400 | CORR | 3.70923381 | 9 | 6 | 0.011532 | 0 | 23 | 1 | PASS |
| a1400 | DIAG | 7.16588146 | 9 | 6 | 0.011532 | 0 | 23 | 0 | PASS |
| a2000 | CORR | 3.68998296 | 9 | 6 | 0.011532 | 0 | 17 | 2 | PASS |
| a2000 | DIAG | 4.68607630 | 9 | 6 | 0.011532 | 0 | 17 | 0 | PASS |

Stress uses 1.25 times the same noise, with the original frozen C0 and thresholds; it is a sensitivity diagnostic, not another calibrated screen. SUMMARY.json includes scene-wise baseline false support and supplemental pixel/brightness-area averages (brightness-area units mJy/nominal beam × arcsec²). Zero final supplemental events do not imply zero baseline false support.

## Amplitude uncertainty check, before S/N selection

Empirical values are sample standard deviations (ddof=1) of every finite converged core refit. Prediction is RMS of available σA; no accepted-only filtering or division by sqrt(N). Bias is mean amplitude minus the new fit to the original noiseless core. Missing predictions/fits are explicitly counted.

| Array | Phase | Converged / with σA | Empirical scatter | RMS σA | Ratio | Bias |
|---|---|---:|---:|---:|---:|---:|
| a1100 | calibration | 1024/1024 | 8.2207 | 8.6525 | 0.9501 | -0.2012 |
| a1100 | validation | 1024/1024 | 8.4973 | 8.6526 | 0.9821 | 0.1593 |
| a1100 | stress1.25 | 256/256 | 10.8467 | 8.6537 | 1.2534 | 0.0088 |
| a1400 | calibration | 1024/1024 | 6.0408 | 6.2366 | 0.9686 | -0.1585 |
| a1400 | validation | 1024/1024 | 6.1802 | 6.2412 | 0.9902 | 0.4364 |
| a1400 | stress1.25 | 256/256 | 7.5265 | 6.2431 | 1.2056 | -0.0105 |
| a2000 | calibration | 1024/1024 | 6.8946 | 7.1108 | 0.9696 | 0.2326 |
| a2000 | validation | 1024/1024 | 6.8017 | 7.1093 | 0.9567 | 0.1069 |
| a2000 | stress1.25 | 256/256 | 8.3330 | 7.1112 | 1.1718 | 0.2896 |

This conditional local-linear check is not a replacement for the existing empirical uncertainty companions. Stress deliberately leaves σA based on C0 unchanged. Source-depth/regional-model qualifications of C0 persist, including a2000 background/source-transfer limitations and the a1100 network-3 scan-5 PTC qualification. No source recovery through PTC is validated here.

## Recorded cases and historical comparison

All 18 historical diagnosis outcomes and all 15 historical real-map measurements/candidates replay exactly on the unchanged historical route. All nine old budget-exhausted blank cases now converge and fail the width criterion; all 15 diagnosed blanks produce complete empty results, and three noisy core controls are accepted. The three noiseless blanks are rejected and three noiseless cores accepted. Raw old/new optimizer records and geometry/error/support are in DIAGNOSIS_COMPARISON.json and checks/RESULTS.json. An uncertainty omitted after a position/width rejection is not a zero error.

| Array | Nine failed-case new nfev (three per array) | Bright controls | Max abs ΔA | Max abs centroid-component Δ arcsec | Max abs sorted-sigma Δ arcsec | Max abs ΔB | Added / removed baseline pixels |
|---|---|---:|---:|---:|---:|---:|---|
| a1100 | [32, 564, 140] | 5/5 | 2.43893e-06 | 9.81493e-09 | 1.41937e-08 | 1.57921e-07 | 0 / 0 |
| a1400 | [43, 144, 53] | 5/5 | 1.36803e-05 | 1.13993e-08 | 8.13792e-08 | 2.84062e-07 | 0 / 0 |
| a2000 | [33, 36, 618] | 5/5 | 3.62197e-06 | 1.00546e-08 | 3.559e-08 | 3.80996e-07 | 0 / 0 |

Brightness-operand and candidate-value differences are explicitly retained in checks/RESULTS.json; each maximum is the fitted constant change, below 4×10⁻⁷ mJy/nominal beam. New fit parameters are not replaced by historical values. The guard leaves candidate membership unchanged on these controls; this does not require future trajectories to be identical. The earlier authenticated historical ten-cycle BASE replays are reused; none is rerun here.

## Unapplied saved-map candidates

Each matched figure contains measured brightness (map minus fitted constant), BASE-POINT, CORR and DIAG at M0/T1/T3/T5/T10. Color limits are shared within an array and clipped at ±6% of its original T10 peak to expose low brightness. Black outlines show BASE-POINT; green on measured maps shows frozen K0; purple marks proposed and teal final connected extras. Gray is unavailable and white is valid zero. Source acceptance, thresholds and counts are in SAVED_SCREEN_COUNTS.json. Each FITS also retains score, variance and validity planes.

| Array | Stage | BASE-POINT pixels | CORR proposed / added | DIAG proposed / added |
|---|---:|---:|---|---|
| a1100 | 0 | 28 | 0 / 0 | 0 / 0 |
| a1100 | 1 | 32 | 0 / 0 | 0 / 0 |
| a1100 | 3 | 35 | 0 / 0 | 0 / 0 |
| a1100 | 5 | 35 | 0 / 0 | 0 / 0 |
| a1100 | 10 | 35 | 0 / 0 | 0 / 0 |
| a1400 | 0 | 38 | 0 / 0 | 0 / 0 |
| a1400 | 1 | 44 | 1 / 1 | 0 / 0 |
| a1400 | 3 | 49 | 1 / 1 | 0 / 0 |
| a1400 | 5 | 49 | 1 / 1 | 0 / 0 |
| a1400 | 10 | 49 | 1 / 1 | 0 / 0 |
| a2000 | 0 | 57 | 0 / 0 | 0 / 0 |
| a2000 | 1 | 59 | 0 / 0 | 0 / 0 |
| a2000 | 3 | 67 | 14 / 14 | 8 / 8 |
| a2000 | 5 | 68 | 19 / 19 | 10 / 10 |
| a2000 | 10 | 68 | 19 / 19 | 11 / 11 |

## Scope and runtime

Implementation/checks and the fixed-field diagnostic screen are complete. Original-C0 amplitude errors and same-map candidates are qualified diagnostics. New BASE-POINT/CORR/DIAG feedback trajectories, endpoint response/noise/uncertainty, and source-transfer improvements are **unrun**. No new production/default, map, covariance, weight, calibration, cleaning or FRUIT state is adopted. There are 6144 reused fields, 768 stress reuses, zero independent new fields and zero new feedback applications.

Recommended next action only: a separately authorized bounded trajectory comparison of BASE-POINT/CORR/DIAG using these frozen thresholds and methods, with source response and background diagnostics. Passing this selection screen does not demonstrate improved maps.

| Stage | Wall seconds | Worker CPU seconds | Max worker RSS MiB |
|---|---:|---:|---:|
| Checks | 9.787 | not captured | not captured |
| calibration | 26.648 | 235.973 | 164.75 |
| validation | 36.186 | 339.909 | 163.95 |

12 independent workers; one numerical thread requested. Apple Accelerate internal thread count is not observable. Calibration/validation source is 0441f293e3daa5bab853efd43e278cb445fdec3c. Reporting source is 110fab8d2325346eec20fe0e9b2a000ac2ffd27a. Reporting time is in REPORT.json. Saved-map publication time was not captured separately. This reporting stage performs no fits or field generation. Figure axes are tangent east/north arcseconds.
