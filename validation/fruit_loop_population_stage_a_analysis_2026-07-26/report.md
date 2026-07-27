# Fruit-loop population Stage A analysis

- Stage B gate: `PASS`
- Completed jobs: `16/16`
- Error-level messages: `0`
- Warning messages: `40`
- Iteration metrics: `480/480 valid`
- Combined-diagnostic transitions: `375/432 interpretable`
- Source-mismatch iteration fits: `10`
- Upper-bound-censored FWHM fits: `57`
- Production policy changed: `false`

## Gate checks

- frozen_binary_checksum_pass: `PASS`
- sixteen_jobs_complete: `PASS`
- zero_error_level_messages: `PASS`
- all_480_iteration_metrics_finite_or_classified: `PASS`
- all_432_transitions_finite_or_classified: `PASS`
- at_least_two_interpretable_observations_per_stratum: `PASS`
- all_four_tolerances_assessed: `PASS`
- no_quality_dependent_setup_or_measurement_failure: `PASS`

Warning-only audit: 10x gaps found in obnsum 130921 data file timing!; 10x gaps found in obnsum 133410 data file timing!; 10x gaps found in obnsum 133542 data file timing!; 10x gaps found in obnsum 134546 data file timing!

## Separate calibration-reference verdicts

| Use | Stage A verdict | Evidence boundary |
|---|---|---|
| Astrometric pointing offset | **Qualified per trajectory; not universal** | `47/48` trajectories retain the same cross-array source association; `45/47` of those pass the final two centroid steps below 0.1 arcsec. Cumulative movement reaches 1.123 arcsec, so the endpoint gate cannot be replaced by a seed-only comparison. |
| Effective processed PSF | **Qualified where the width fit is uncensored; not universal** | `40/48` trajectories avoid the fitter's upper FWHM bound. Of those, `33/40` pass at 1% and `40/40` pass at 5% in transitions 8;9. |
| Photometric amplitude / transfer | **No** | Real-source amplitude has no injected or external truth. Even stability is tolerance-dependent: `28/47` pass at 1%, versus `47/47` at 5%. |
| Associated science-processing response | **Not determined** | Stage A contains pointing reductions only; no matched pointing-versus-science injection was run. |

## Diagnostic endpoint yield

| Tolerance | Amplitude | FWHM | Whole map | Stepwise S/N | Combined |
|---:|---:|---:|---:|---:|---:|
| 1% | 28/47 | 33/40 | 8/48 | 33/47 | 7/48 |
| 2% | 38/47 | 39/40 | 22/48 | 44/47 | 21/48 |
| 5% | 47/47 | 40/40 | 41/48 | 47/47 | 36/48 |
| 10% | 47/47 | 40/40 | 48/48 | 47/47 | 40/48 |

## Combined convergence yield

| Tolerance | Stratum | Arrays ever passing | Arrays passing endpoint 8;9 | Observations all arrays ever passing |
|---:|---|---:|---:|---:|
| 1% | normal | 2/24 | 2/24 | 0/8 |
| 1% | marginal | 4/15 | 4/15 | 0/5 |
| 1% | stress | 1/9 | 1/9 | 0/3 |
| 2% | normal | 11/24 | 11/24 | 2/8 |
| 2% | marginal | 8/15 | 8/15 | 1/5 |
| 2% | stress | 2/9 | 2/9 | 0/3 |
| 5% | normal | 22/24 | 22/24 | 6/8 |
| 5% | marginal | 11/15 | 11/15 | 3/5 |
| 5% | stress | 3/9 | 3/9 | 0/3 |
| 10% | normal | 24/24 | 24/24 | 8/8 |
| 10% | marginal | 14/15 | 13/15 | 4/5 |
| 10% | stress | 3/9 | 3/9 | 0/3 |

## Interpretation

Stage A passes the predeclared Stage B gate because every job and metric is complete and every quality stratum has at least two fully source-associated observations. Censored PSF fits and a cross-array source mismatch are retained as classified failures, not converted into false convergence. This does not select a stopping tolerance. The strict combined diagnostic includes whole-map change and stepwise S/N, so it is intentionally more demanding than fitted source stability alone.

Cumulative S/N is a separate guard: `23/47` source-associated trajectories lose more than 10% from seed to iteration 9, including `7` that lose more than 20%.

At 1%, `16/16` observations do not have all three arrays satisfy the combined two-transition diagnostic by iteration 9. They remain part of yield accounting and are candidates for checkpoint-v2 continuation after the full population run.

Real-source trajectories constrain astrometric and effective-PSF stability but do not establish photometric truth. Pointing-to-science response remains unmeasured.

## Operational exception

2 config copies arrived with modes differing from their setup configs. In this run the two affected redu01 copies arrived mode 0200 on Unity; the owner restored owner-read permission, and both local copies are byte-identical to the checksummed setup configs. This was not quality-dependent and did not affect scientific products.
