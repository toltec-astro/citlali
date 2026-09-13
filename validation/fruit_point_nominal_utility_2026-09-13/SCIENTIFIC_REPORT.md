# Nominal starlet POINT utility comparison — park for POINT

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

This report follows the [charter](../../doc/scientific_contracts/README.md),
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md)
and the [frozen protocol](PROTOCOL.md). The latest [owner direction](OWNER_DIRECTION.txt)
replaced the proposed fine-scale constraint with this POINT utility experiment.
The earlier [image-stability qualification remains failed](../fruit_point_bounded_repair_2026-09-12/SCIENTIFIC_REPORT.md).
This result neither reverses that failure nor rejects wavelets generally.

**Disposition: park this fixed candidate for POINT.** It produces better raw
peak ratios in the imposed-degradation cases, preserves tested pointing, and
finishes every trajectory. It does not deliver a sufficiently consistent usable
peak-response improvement across the registered cases. Six trajectories also
miss the twofold cost limit. No additional constraint, tolerance, evaluator
change or follow-on experiment is executed or automatically recommended.

## The comparison we actually completed

P is the matched pixelwise reference; C is the central positive starlet model
with the already tested nominal relative-gradient stop. Both use rank 5,
uniform weights, the same immutable pre-PTC parents and seven passes. Every pass
relearns PTC from the current model-subtracted parent and infers replacement
feedback from reconstructed total sky. Objective, support, coefficients,
background and normalization were unchanged. No tighter solution was computed.

All 34 trajectories completed with all seven required decisions available:
238 cleaning calls in **402.62 seconds**, peak RSS **2.07 GiB**, and approximately
**0.52 GiB** of run products. The 357 C inference decisions comprise 273
nonempty nominal solves and 84 valid empty-support zero models. All nonempty
solves finish within the existing caps, with at most 746 iterations; maximum
reported relative projected gradient is 9.99769e-5. Numerical completion is
established for these calls, not image convergence or scientific accuracy.

## Required outputs, including unavailable measurements

The terminal is the predeclared seventh map (pass 6). A ratio counts as useful
only when both peaks are usable under the same repaired data-only evaluator.
Truth scores accuracy; it cannot make a measurement usable.

| Required finite test | Reference P | Candidate C |
| --- | ---: | ---: |
| Usable pointing within 1 arcsec, H / shifted H / D | 18/18 | 18/18 |
| Usable H/D gain within 5% | 2/6 | 1/6 |
| Usable D/H degradation within 5%, loss preserved | 2/6 | 1/6 |
| Usable unchanged-H ratio within 5% | 0/3 | 1/3 |
| Usable T/H response-loss and unaffected-array ratios | 3/6 | 5/6 |
| Gross-coma source evidence plus degradation/support warning | 6/6 | 6/6 |
| Boundary challenge warned; reliable correction withheld | 6/6 | 6/6 |

H/D and D/H are reciprocal views of the same six pairs, not twelve independent
successes. The T/H count includes two affected-array tests and four unchanged
array tests. C measures the affected a1400 loss usefully in one of two seeds,
versus zero for P; the other C affected-array pair remains unavailable. Neither
method therefore establishes repeatable affected-array response monitoring.

Required-pointing errors range from **0.0336–0.5888 arcsec for P** and
**0.0267–0.7157 arcsec for C**. Both preserve the 1-arcsec terminal requirement;
neither has a qualified ensemble uncertainty. Across all 24 compact source
array cases (H, shifted H, D, T), both yield 24 usable centroids; peak yield is
15/24 for P and 16/24 for C. That one-measurement gain does not compensate for
losing a usable primary gain/degradation pair. Shape warnings remain on 9/24
P and 7/24 C compact results.

## A narrower scientific improvement is present

All six **raw** C H/D errors are within **0.81%**; five of six P errors are
within 5%, with the first a1400 seed at +5.96%. Raw C D/H values also preserve
the imposed 10% degradation within 0.80%. This is useful evidence about the
combined algorithm's finite ratio fidelity. It is not a production result:
five of six C gain pairs are withheld by the unchanged usability rule.
All three first-seed degraded-source peaks are withheld. We do not use their
known injected accuracy to override that rule.

The unchanged a2000 source pair still shows an apparent peak loss of **6.58%**
for C, versus **11.67%** for P. Both pairs are usable and both fail the 5%
requirement. C improves this error but does not meet the requirement. The only
passing unchanged-H C pair is a1100 (-2.82%); the a1400 C ratio is withheld and
its raw change is -5.54%.

Larger recovered peaks are not proof of better absolute response. Across the
24 compact source cases, only 10/24 raw fitted peaks in each arm are within 5%
of absolute truth. P errors span -15.53% to +10.11%; C spans -6.48% to +19.33%.
Thus the favorable ratio result does not establish generally accurate peaks.
[The terminal ratio figure](TERMINAL_PEAK_RATIOS.png) distinguishes usable and
withheld measurements explicitly.

## Failures, support and retained development sequence

Both arms issue zero source reports and zero usable point/peak reports on
all 84 null/background array-pass cases per arm. C also has **zero false
feedback admissions**, versus 84/84 for P. This cleaner feedback behavior does
not by itself improve the POINT source report, which was already safe for P.

Every boundary array-pass case carries its support warning and withholds a
centroid: 42/42 per arm. Gross-coma source and warning results are preserved,
without a precision coma, width or focus claim. Full support, coefficients,
model changes, fitted widths, residuals and learned states are retained.
Exterior RMS C/P ranges from 0.9990 to 1.0234. It is diagnostic in this protocol
and would also satisfy the old 1.10 comparison; it does not explain rejection.

The sequence is not uniformly stable merely because its last map is usable.
At pass 4, first-seed C a1400 H and shifted-H raw fits jump to errors of 60.70
and 65.04 arcsec. The data-only evaluator withholds both centroids and peaks
with support/stability warnings. Later passes recover. These failures remain
in the [time plots](POINT_RECOVERY_VS_TIME.png); no earlier favorable map was
selected, and no early-stopping or general convergence claim follows.

## Measured cost

The following full trajectory times include cleaning, mapping, inference,
common evaluation and output. All inference/admission work is charged. Separate
component timings and cumulative times to every map remain in the run receipts.
These are single local paired timings, not a qualified latency distribution.

| Case | P seconds | C seconds | C/P | Within 2x |
| --- | ---: | ---: | ---: | :---: |
| real123424 | 8.37 | 13.07 | 1.561 | yes |
| N_20260911 | 7.55 | 7.52 | 0.996 | yes |
| B_20260911 | 8.10 | 7.76 | 0.958 | yes |
| H_20260911 | 9.01 | 19.05 | 2.114 | no |
| H-shift_20260911 | 9.00 | 18.22 | 2.025 | no |
| D_20260911 | 9.26 | 17.95 | 1.938 | yes |
| C_20260911 | 6.17 | 15.20 | 2.464 | no |
| T_20260911 | 9.45 | 16.80 | 1.778 | yes |
| E_20260911 | 9.08 | 23.10 | 2.543 | no |
| N_20260912 | 8.32 | 7.81 | 0.939 | yes |
| B_20260912 | 8.26 | 7.87 | 0.953 | yes |
| H_20260912 | 9.86 | 15.66 | 1.588 | yes |
| H-shift_20260912 | 9.46 | 15.16 | 1.602 | yes |
| D_20260912 | 9.24 | 14.91 | 1.613 | yes |
| C_20260912 | 7.09 | 15.49 | 2.184 | no |
| T_20260912 | 9.85 | 15.56 | 1.580 | yes |
| E_20260912 | 9.12 | 20.14 | 2.208 | no |

C is 1.58–2.54 times P across the synthetic source cases, and 1.56 times P
on discovery 123424. The six cost misses include first-seed H and shifted H,
and both gross-coma and boundary seeds. The first H miss is 2.114x; the shifted
case is near the boundary at 2.025x. We do not round either into acceptance.
Null/background costs are roughly equal, where C performs no nonempty solve.
No reduction in expensive cleaning passes is claimed: both retained seven.

The identified **POINT-123424-OG-RECURRENCE-EL-F2-ALPHA1-CONTROL-2026-09-02**
benchmark remains seven passes in 205.38 seconds with one thread and different
upstream, JINC, weighting and diagnostic boundaries. Its full time is contextual;
it is not a matched speed denominator for this four-thread pre-PTC harness.
[The bound OG record](../fruit_point_coherent_feedback_2026-09-11/OG_BENCHMARK.json)
and native pointing sequence remain in the evidence.

## Discovery observation and limits of the recommendation

On real123424, both methods return usable a1100/a1400 centroids and peaks;
a2000 remains support/shape limited and unavailable. Raw real-data fitted
peaks and centers are not injection-truth scores:

| Array | P peak | C peak | P center (arcsec) | C center (arcsec) | Usability |
| --- | ---: | ---: | --- | --- | --- |
| a1100 | 5729.89 | 8899.97 | (10.396, -5.276) | (13.261, -5.598) | both usable |
| a1400 | 4949.08 | 4959.95 | (11.654, -5.329) | (11.806, -5.330) | both usable |
| a2000 | 113.10 | 169.57 | (-20.938, -21.676) | (-13.356, 27.274) | both unavailable |

The a1100 fitted peak and width change substantially: P FWHM 19.87x7.99
arcsec versus C 8.36x6.13 arcsec. The center shifts by about 2.88 arcsec.
C's a1100 center is closer to the identified OG terminal center
(12.963,-5.575), but OG is not truth and uses different measurement/reduction
choices. Neither a larger fitted peak nor closer OG agreement establishes
better real-data recovery. C's a1400 retains its shape warning. The signed
[total-map comparison](DISCOVERY_TOTAL_MAPS.png) remains a diagnostic display.

The evidence therefore supports **park for POINT**, with explicit retention
of the raw ratio-fidelity result. Selecting only the successful second-seed
a1100 gain pair would discard the first seed required for that use. Similarly,
one improved affected-array health pair is not repeatable health performance.
The aggregate availability gains do not establish a passing primary use across
the registered population at the declared cost. Parking does not promote P as
a fully adequate gain method: P also fails important peak-response requirements.

A future modification would require a specific demonstrated POINT failure and
a credible proposed remedy under a separate owner decision. This run does not
recommend another smoothing parameter, gate relaxation or numerical repair.
The evidence is preserved for possible OOF work without claiming OOF suitability.
Reserved 129081, independent-pointing replication, numerical-method qualification
and production remain outside this completed experiment.

## Verification and provenance

The protocol and source freeze were committed before execution at
`9a2c441948298febc57b38c98db14040377da5cb`, descending from recovered clean HEAD
`9073b479617f72f7a1ce9e2e4fb335fe7e09f56e`. The owner attachment is preserved
byte for byte; its original trailing whitespace is excluded only from the
Git whitespace check. Analysis/reporting files were added after execution;
they do not change the frozen estimator, inputs or gates.

All 17 parent pairs match their saved controls. All repeated P map/model arrays
and learned-state arrays match exactly. All 51 C bootstrap array models match
the previously saved nominal solutions exactly. Checks cover 238 saved model
recurrences, 357 fixed candidate problems, 672 external truth scores and 864
sampled terminal covariance/eigen identities (maximum residual 1.99e-15).
Verification makes no new cleaning or optimization calls. Three focused tests
cover nonfinite/constraint handling, unavailable-solve application and data-only
usability. No routine runtime defect or replacement replay was needed.

All 202 prior packet payloads, 1007 prior external products and 57 frozen
ordinary-map authority payloads pass preservation checks. The two protected
review archives remain opaque and unchanged in Git status; they were not
opened or hashed. This experiment retains 797 external payloads under
`/private/tmp/sci-fruit-point-nominal-utility-20260913-r0.1`, inventoried by
[the run manifest](RUN_PRODUCT_MANIFEST.json). Run products remain local, outside
Git. The repository stores the numerical evidence, plots and reproducible scripts.
The preservation and source hashes bind this result; a `/private/tmp` path is
local storage, not a long-term off-machine archive.

Detailed [decision evidence](DECISION_EVIDENCE.json),
[terminal measurements](TERMINAL_MEASUREMENTS.md),
[verification](VERIFICATION.json) and [result manifest](RESULT_MANIFEST.json)
retain the denominators, warnings, raw errors and timing needed to review it.
