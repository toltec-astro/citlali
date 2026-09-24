# POINT 152389 correlation-aware admission: failed-gate result

The numerical covariance and selection implementation pass their checks, but the prescribed complete null procedure cannot calibrate a threshold. Blank-scene descriptive fits fail; their search maxima are unknown. No CORR or DIAG feedback trajectory was launched.

| Array | Calibration blank failures /1024 | Validation blank failures /1024 | Stress blank failures /256 |
|---|---:|---:|---:|
| a1100 | 125 | 126 | 38 |
| a1400 | 165 | 173 | 50 |
| a2000 | 233 | 226 | 53 |

All core-scene outcomes are numerically valid, including finite constrained fits whose inherited candidate is absent. Legitimate no-positive-central-peak outcomes remain separate from solver nonconvergence. The prescribed field counts are complete; stress reuses existing fields.

All six calibrated thresholds, binomial upper bounds, supplemental false-admission rates, extra-brightness biases and real supplemental masks are **unavailable**, not zero. The original intermediate thresholds remain preserved as invalid diagnostics. No failed draw was replaced and no solver or science selection was retuned.

## What is and is not learned

The fixed K0/current BASE/anchor exclusions, original measured brightness publication, full footprints, covariance variances and deterministic generation pass numerical tests. Same-map figures expose the nominal evidence without implying admission. Since the common source-measurement procedure blocks calibration, this experiment cannot compare CORR with DIAG or measure new-policy compact/shoulder response, seed persistence, endpoint background, terminal drift or measurement precision. A larger nominal score does not establish a useful admitted region.

The three BASE M0–T10 controls replay the original maps, measurements, candidate values/masks, applied operands and same-stage PTC state under the unchanged recipe. Their response and history remain the original reference; they are not new-policy results.

## Partial BASE null diagnostics

Only numerically valid outcomes enter these averages. Blank rows are partial populations; failed placeholders are excluded. Units of brightness-area are mJy/nominal beam × arcsec². These describe inherited BASE false support, not calibrated supplemental false admission.

| Array | Phase | Scene | Valid / prescribed | BASE false events | Mean false pixels | Mean false brightness-area |
|---|---|---|---:|---:|---:|---:|
| a1100 | calibration | blank | 899/1024 | 48 | 0.05784 | 10.161 |
| a1100 | calibration | core | 1024/1024 | 29 | 0.03418 | 5.8571 |
| a1100 | validation | blank | 898/1024 | 37 | 0.04232 | 7.2094 |
| a1100 | validation | core | 1024/1024 | 32 | 0.03223 | 5.4226 |
| a1100 | stress1.25 | blank | 218/256 | 26 | 0.1376 | 25.273 |
| a1100 | stress1.25 | core | 256/256 | 35 | 0.2227 | 38.196 |
| a1400 | calibration | blank | 859/1024 | 30 | 0.03609 | 5.1302 |
| a1400 | calibration | core | 1024/1024 | 42 | 0.04297 | 5.9514 |
| a1400 | validation | blank | 851/1024 | 46 | 0.0564 | 8.0471 |
| a1400 | validation | core | 1024/1024 | 43 | 0.04492 | 6.345 |
| a1400 | stress1.25 | blank | 206/256 | 23 | 0.1408 | 22.538 |
| a1400 | stress1.25 | core | 256/256 | 42 | 0.207 | 30.361 |
| a2000 | calibration | blank | 791/1024 | 46 | 0.06195 | 13.191 |
| a2000 | calibration | core | 1024/1024 | 68 | 0.06934 | 14.41 |
| a2000 | validation | blank | 798/1024 | 57 | 0.07268 | 15.698 |
| a2000 | validation | core | 1024/1024 | 75 | 0.07715 | 16.334 |
| a2000 | stress1.25 | blank | 203/256 | 27 | 0.1576 | 37.765 |
| a2000 | stress1.25 | core | 256/256 | 87 | 0.4258 | 93.196 |

## Cost and execution scope

Preparation: 236.04 s wall, including 190.63 s initial authentication and 37.04 s repeated original verification.

| Phase | Wall seconds | Summed trial CPU seconds | Maximum worker high-water MiB |
|---|---:|---:|---:|
| CALIBRATION | 85.10 | 846.89 | 164.41 |
| VALIDATION | 105.49 | 1071.19 | 164.97 |

BASE replay: 279.35 s wall, 3 concurrent native jobs, sampled aggregate RSS 3.550 GiB.
Twelve independent simulation workers; one library thread requested. Apple Accelerate internal thread counts are not observable. High-water and aggregate RSS are distinct measurements; CPU totals are not elapsed time. All execution was local in the existing environment, not Unity/Spack replication.

## One next action, not executed

Recommend a bounded investigation of the inherited POINT solver’s blank-scene failures and a separately reviewed source-absence/candidate-outcome rule before repeating admission calibration. This is not permission to relabel current nonconvergence as absence, change the solver here, or choose a threshold from survivors.

The original a1100 network-3/scan-5 and a2000 background/source-transfer qualifications remain. C0 is the original A+B M0 model and remains frozen; conditional simulations do not validate observed source recovery, astronomical significance, or a ten-cycle false-feedback rate. MAP coefficients, source-fit weights, calibration, feedback gain and production defaults were not changed.

## Diagnostic figures

![null-outcomes](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/null-outcomes.png)

![nominal-score-distributions](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/nominal-score-distributions.png)

![a1100-saved-evidence](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/a1100-saved-evidence.png)

![a1400-saved-evidence](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/a1400-saved-evidence.png)

![a2000-saved-evidence](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/a2000-saved-evidence.png)

![covariance-and-support](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/covariance-and-support.png)

![BASE-replay](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/BASE-replay.png)

[Every trajectory and unavailable product](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/MATRIX.json)
[Detailed null outcomes](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/NULL_SUMMARY.json)
[Actual covariance and seed-regeneration checks](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924/report/CHECKS.json)
