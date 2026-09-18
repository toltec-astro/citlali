# Connected PTC result and bounded cost probe — 2026-09-18

Implementation: `6ee33211bfd449954a2fab0330160215275f9683`; tree `0799cc18d2a1bb0e417a83c5a61a6526020da857`. Base canonical `35b05ba91c761646dd08340906c7e2e05e8d2587`. Independent exact-SHA review: PASS with recorded limitations; all source findings resolved.

Ordinary `citlali development.yaml` now defaults to RTC → CAL → iterative PTC → PTC/VAL. Explicit `ptc.method: pairwise-covariance` selects the noniterative comparator. Grouping/rank come from the bound existing effective configuration (network/10 here), with explicit positive `ptc.rank` override. `terminal: cal` and `rtc-only` remain deliberate early stops. MAP and FRUIT are not implemented by this increment.

The twelve-detector 152390/0/2 network12 Science/Lissajous fixture retains **833,599 of 882,766 eligible entries** (94.43%) across119 successful fits of 124processing segments. Four fits reach the 100-iteration cap and one lacks pairwise initialization overlap; they stay unavailable and the invocation returns2. No method/rank/member fallback. All 882,766 admitted CAL inputs were checked bit-for-bit against the preserved CAL files; all 152 upstream RTC/CAL numerical and support artifacts match the accepted CAL run exactly.

The414-detector attempt retained a fixed cohort selected before PTC from producer validity and existing unique supported APT factors. It ran RTC for 499.05 s, then failed the unchanged terminal guard:102 of 828 coordinate treatment outcomes were unavailable (51 detectors; insufficient qualifying spectral windows). No CAL or PTC result was produced. This is an upstream connected-domain limitation, not real-network PTC timing. Bypassing the guard or iteratively trimming the cohort would change the fixed upstream policy, so no such repair is made. Synthetic scaling below is provisional.

## Matched solver costs

AppleClang 21/Homebrew/C++20, recorded Eigen thread setting 14; effective worker use was not separately measured. Three independent processes per method/rank; median times, maximum process RSS across repeats. Warm filesystem, cold preparation and numerical fits; no solver warm starts or shared cross-rank eigendecomposition. Memory scope is the standalone PTC input/preparation/fit/apply/response process, not the RTC/CAL process. Each method consumes identical preserved input and binary masks. Synthetic cases use610 samples at the post-F2 cadence (about 10 s),400 independent generated detector streams,100 detectors with60-sample flagged intervals; overlap and staggered geometry are separate. The real twelve-detector case is processing segment 13,610 times,7,287 eligible entries (complete-time fraction 94.59%).

| Case | Method | Rank | Shape (time × detector) | Converged / iterations | Fit ms | Apply ms | RSS MB | Retained |
|---|---|---:|---|---|---:|---:|---:|---:|
| synthetic-overlap | ALS | 5 | 610 × 400 | yes / 3 | 31.20 | 0.75 | 21.7 | 238,000 |
| synthetic-overlap | Pairwise | 5 | 610 × 400 | yes / 0 | 27.92 | 0.76 | 21.7 | 238,000 |
| synthetic-overlap | ALS | 10 | 610 × 400 | yes / 4 | 35.22 | 0.84 | 21.8 | 238,000 |
| synthetic-overlap | Pairwise | 10 | 610 × 400 | yes / 0 | 26.32 | 0.88 | 27.6 | 238,000 |
| synthetic-overlap | ALS | 15 | 610 × 400 | yes / 4 | 35.68 | 1.02 | 26.0 | 238,000 |
| synthetic-overlap | Pairwise | 15 | 610 × 400 | yes / 0 | 26.12 | 1.05 | 21.5 | 238,000 |
| synthetic-staggered | ALS | 5 | 610 × 400 | yes / 3 | 31.02 | 1.24 | 25.4 | 238,000 |
| synthetic-staggered | Pairwise | 5 | 610 × 400 | yes / 0 | 26.40 | 1.30 | 23.5 | 238,000 |
| synthetic-staggered | ALS | 10 | 610 × 400 | yes / 5 | 48.60 | 2.34 | 29.5 | 238,000 |
| synthetic-staggered | Pairwise | 10 | 610 × 400 | yes / 0 | 29.70 | 2.48 | 29.4 | 238,000 |
| synthetic-staggered | ALS | 15 | 610 × 400 | yes / 5 | 73.61 | 3.74 | 27.9 | 238,000 |
| synthetic-staggered | Pairwise | 15 | 610 × 400 | yes / 0 | 31.02 | 3.71 | 29.3 | 238,000 |
| real12 | ALS | 5 | 610 × 12 | yes / 5 | 1.38 | 0.07 | 14.7 | 7,287 |
| real12 | Pairwise | 5 | 610 × 12 | yes / 0 | 1.11 | 0.07 | 14.5 | 7,287 |
| real12 | ALS | 10 | 610 × 12 | yes / 11 | 2.06 | 0.06 | 14.6 | 7,287 |
| real12 | Pairwise | 10 | 610 × 12 | yes / 0 | 1.15 | 0.06 | 14.5 | 7,287 |

At rank 10, ALS costs 1.34×pairwise for overlapping masks and 1.64×for staggered masks (1.79×in the small real fixture). Summing the measured independent ranks 5/10/15 fits and applications gives 0.1047 s/overlap or 0.1605 s/staggered per synthetic segment, compared with 0.0361 s or 0.0509 s at rank 10: 2.90×and 3.15×respectively. Those are sums of measured trials, not a timed automatic optimizer. More folds/scan/network coverage would multiply the parts that must be refitted. Covariance/count preparation and the pairwise full eigendecomposition could be shared in a future rank-search implementation; independently fitted ALS ranks still need their own updates and convergence. This probe deliberately does not reuse that eigendecomposition. Prepared CAL input avoids all repeated RTC/CAL work.

At rank 10, covariance/eigendecomposition initialization occupies about 83% of overlap ALS fitting and 53% of staggered ALS fitting. Staggered masks increase coefficient/basis update and application-factorization costs. Equal-mask factorizations are already reused while the basis is fixed, with counts in every result; no old factorization survives a basis update. The optional holdout probe was omitted; no astronomical rank-selection score is selected.

## Stopping sensitivity and practical costs

Tightening relative objective tolerance from 1e-5 to 1e-7 changes the cleaned result by 0.321% of cleaned norm for overlapping synthetic masks and 0.413%for staggered masks. On the real twelve-detector segment it changes cleaned norm by 1.054% (difference RMS 0.1843 mJy/nominal-beam, maximum 1.0816); iterations rise 11→24 and fit time 2.06→3.45 ms. Gauge-invariant projector Frobenius differences are 0.00924,0.01169,0.01373 respectively. These are measured numerical sensitivity, not a source-transfer qualification or proof of a scientifically optimal stopping rule. Four other real segments remain nonconverged at the provisional cap; neither their cap nor a zero-filled unavailable output is reported as convergence.

The complete twelve-detector PTC phase takes 6.85 s: preparation 0.0775 s, fitting 0.1292 s, application 0.00717 s, and artifact publication/checksums 6.56 s. Output publication, not fitting, dominates this connection run. The cost-only400-detector cases isolate numerical scaling; upstream RTC/CAL time and memory are excluded from their process figures. No full-network PTC extrapolation is presented as measured.

## Verification and next responsibility

Exact-source CTest: 1253/1253 runnable passed of 1254 registered; the established MapFitterLifecycle.ExactProductSequence remains disabled. Focused 8 numerical + 32 CAL/PTC-related tests, 4 source-graph tests, 514 Python tests + 178 subtests and full config preflight pass. Public headers compile without PCH; both CMake source graphs contain the two new units. The existing kids 04088da-dirty local dependency banner is retained and disclosed; this is no new Unity/Spack qualification. Earlier Unity evidence remains bound to its earlier RTC source.

This completes the initial connected fixed-rank PTC operation and PTC/VAL publication. The next downstream implementation owner is MAP, consuming transformed calibrated x and its actual support/classification/response limits. Before treating performance as representative of a real network, the immediate bounded prerequisite is resolving why an RTC cohort with unavailable coordinate outcomes cannot publish a usable terminal under the existing contracts. This is not permission to change transient, filter, source, rank or rejection policy. Complete-chain response/covariance, generic ingress, array grouping, rank optimization and FRUIT remain outside this increment.

Machine-readable evidence: `CONNECTED_VERIFICATION.json`, `FINAL_COST_RESULTS.json`, `NETWORK_LIMITATION.json`, exact build/check logs and independent-review records under `/private/tmp/citlali-successor-ptc-001-20260918`.
