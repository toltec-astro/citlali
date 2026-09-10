# Complete screen tables

These tables summarize immutable attempt 03. All nine synthetic cases and all six discovery array cases are retained. No omitted support or unavailable metric becomes a pass.

## Synthetic results

| Case | RMS U / N4U | Conditional disposition | Demonstrated failures | Inconclusive criteria |
| --- | ---: | --- | --- | --- |
| W00 | 0.996581 | pass | none | none |
| W01a | 1.251024 | failure | noise, excursion_difference | none |
| W01b | 2.103693 | failure | noise, excursion_difference | none |
| W02 | 1.184986 | failure | noise | excursion_difference |
| W03a | 1.006281 | pass | none | none |
| W03b | 1.820693 | failure | noise, excursion_difference | none |
| W04a | 1.363716 | failure | noise, excursion_difference | none |
| W04b | 1.286312 | failure | noise, U/amplitude_bias, excursion_difference | none |
| W05 | 1.246368 | failure | noise, excursion_difference | none |

Decisions use the registered 242 scalar tails at 0.05/256 each, with approximate Student-t continuous bounds and exact binomial bounds. The ratio column is descriptive. Four independent-Gaussian U analytic checks passed. Each full per-case JSON retains every numerical bound and margin.

## Full and common discovery regions

| Observation / array | Full O RMS N4U / U | Common O RMS N4U / U | Native template amplitude U → N4U (mJy/beam) | Missing D pixels U / N4U |
| --- | ---: | ---: | --- | ---: |
| 123424 a1100 | 0.544295 | 0.544295 | 830.375 → 562.657 | 0 / 0 |
| 123424 a1400 | unavailable | 0.722690 | unavailable → unavailable | 2 / 9 |
| 123424 a2000 | 0.781157 | 0.781157 | -28.7539 → -30.6104 | 0 / 0 |
| 152389 a1100 | 0.553433 | 0.553433 | 75.303 → 60.0146 | 0 / 0 |
| 152389 a1400 | unavailable | 0.505676 | unavailable → unavailable | 3 / 3 |
| 152389 a2000 | 0.454892 | 0.454892 | 187.039 → 164.72 | 0 / 0 |

O is the fixed R–3R annulus; D is the full radius-3R comparison disk. The two a1400 common-only values cannot satisfy full-region requirements. No value in this table is a confidence interval, physical flux recovery or detector-noise estimate.

## All full-map alerts

| Observation / array | Triggered alerts | Unavailable alerts |
| --- | --- | --- |
| 123424 a1100 | O_uniform_power_loss, O1_uniform_power_loss, O2_uniform_power_loss, amplitude_change, centroid_x_change, centroid_y_change, width_y_change | width_x_change |
| 123424 a1400 | O1_uniform_power_loss, source_unresolved | O_uniform_power_loss, O_U_over_N_power, O2_uniform_power_loss, O2_U_over_N_power, amplitude_change, centroid_x_change, centroid_y_change, width_x_change, width_y_change |
| 123424 a2000 | O_uniform_power_loss, O1_uniform_power_loss, O2_uniform_power_loss, source_unresolved | amplitude_change, centroid_x_change, centroid_y_change, width_x_change, width_y_change |
| 152389 a1100 | O_uniform_power_loss, O1_uniform_power_loss, O2_uniform_power_loss, amplitude_change, centroid_x_change, centroid_y_change | width_x_change, width_y_change |
| 152389 a1400 | O1_uniform_power_loss, source_unresolved | O_uniform_power_loss, O_U_over_N_power, O2_uniform_power_loss, O2_U_over_N_power, amplitude_change, centroid_x_change, centroid_y_change, width_x_change, width_y_change |
| 152389 a2000 | O_uniform_power_loss, O1_uniform_power_loss, O2_uniform_power_loss, amplitude_change, centroid_x_change, centroid_y_change | width_x_change, width_y_change |

## Window support and native source moments

| Observation / array | Window | Missing D pixels U / N4U | U amplitude; x/y widths | N4U amplitude; x/y widths |
| --- | ---: | ---: | --- | --- |
| 123424 a1100 | 0 | 0 / 0 | 896.398; unavailable / 5.76046 | 625.282; unavailable / 7.5008 |
| 123424 a1100 | 1 | 0 / 0 | 967.593; unavailable / 1.78182 | 798.282; unavailable / unavailable |
| 123424 a1100 | 2 | 0 / 0 | 684.706; 1.07631 / 2.95675 | 382.134; unavailable / unavailable |
| 123424 a1400 | 0 | 311 / 415 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 123424 a1400 | 1 | 33 / 98 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 123424 a1400 | 2 | 8 / 32 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 123424 a2000 | 0 | 26 / 100 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 123424 a2000 | 1 | 12 / 60 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 123424 a2000 | 2 | 0 / 9 | -47.7521; unavailable / unavailable | unavailable; unavailable / unavailable |
| 152389 a1100 | 0 | 0 / 0 | 103.725; unavailable / unavailable | 103.307; unavailable / unavailable |
| 152389 a1100 | 1 | 0 / 0 | 55.6861; unavailable / unavailable | 30.6631; unavailable / unavailable |
| 152389 a1100 | 2 | 0 / 1 | 68.8721; unavailable / unavailable | unavailable; unavailable / unavailable |
| 152389 a1400 | 0 | 93 / 175 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 152389 a1400 | 1 | 115 / 155 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 152389 a1400 | 2 | 20 / 243 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 152389 a2000 | 0 | 1 / 34 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 152389 a2000 | 1 | 22 / 123 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |
| 152389 a2000 | 2 | 41 / 253 | unavailable; unavailable / unavailable | unavailable; unavailable / unavailable |

All window centroids, ring means/powers/correlations, native/common supports, temporal differences and causes are retained in each case summary and map products. Widths use signed moments with no clipping; unavailable widths must not be replaced by fitted or absolute-valued widths.

## Fallback and cost

| Observation / array | Fallback groups | Full-array fallback coefficient fraction | Central fallback coefficient fraction | U mapping seconds | N4U mapping seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 123424 a1100 | 0 | 0 | 0 | 0.482517 | 0.481770 |
| 123424 a1400 | 0 | 0 | 0 | 0.136233 | 0.137739 |
| 123424 a2000 | 0 | 0 | 0 | 0.146914 | 0.147358 |
| 152389 a1100 | 1 | 3.3262341e-05 | 0.00022564249 | 0.357365 | 0.355776 |
| 152389 a1400 | 1 | 8.9990121e-05 | 0.0013088291 | 0.135735 | 0.135602 |
| 152389 a2000 | 0 | 0 | 0 | 0.123499 | 0.124169 |

Coefficient estimation is separately timed in each generation lock. Mapping times include required accumulation/influence work and signed checks, excluding serialization; timing order was fixed U then N4U, so these are descriptive cost measurements, not a speed ranking. T1 weight/mapping costs are recorded in its case JSON.
