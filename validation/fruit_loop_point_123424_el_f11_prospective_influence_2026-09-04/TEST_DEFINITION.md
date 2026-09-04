# SCI-FRUIT EL-F11 — Registered prospective-influence persistence test

Status: **authorized development test definition; no result yet**

Decision identity:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

## Fixed question and target

Replay the exact copied EL-F5 injected iteration-3 checkpoint once, advancing
absolute FRUIT iteration 3 to 4. The diagnostic target is observation 123424,
a1400, UID 4460, and zero-based scan 5. The ordinary calculation, learned
state, and action policy remain unchanged.

Use the exact JINC accounting at the end of iteration 4 to construct the
whole-target deletion response `D_4`. Compare it with retained EL-F10
iteration-5 response `D_5`. The target was chosen with knowledge of iteration
5, so the result is an oracle-targeted temporal-persistence feasibility test,
not a deployable selection method or an unbiased performance estimate.

## Gate order

1. Validate every registered file's exact size and SHA-256.
2. Require all nine a1100/a1400/a2000 `signal_I`, `kernel_I`, and `weight_I`
   planes and all three `weight_formal_I` planes to reproduce the existing
   EL-F5 injected iteration-4 maps bitwise.
3. Require `learning_iter_4.csv` to reproduce byte-for-byte. Require the new
   and reference map-diagnostic NetCDF structures, attributes, masks, and
   values to be identical. Whole-file NetCDF bytes are not compared because
   equivalent historical and diagnostic executables are already known to
   serialize different HDF5 metadata.
4. Require checkpoint structure and scientific values to match, permitting
   only observed `creator_version` and `learning_policy_yaml` differences.
   The latter must become exactly equal after adding the historically omitted
   `map_pixel_outlier_detector_exclusion_application: pre_cleaning` default;
   no other policy or checkpoint difference is allowed.
5. Require the captured total `N`, `C`, and `Q` to re-finalize the new a1400
   signal and formal coefficient bitwise under the recorded threshold and
   support rules. Captured formal and empirical coefficients and normalization
   support must likewise match their science-map values exactly.
6. Require one target-ledger row per unique final-PTC sample index, exact
   target UID and zero-based scan identity in every row, and only registered
   admission reasons. Noise-only-pass rows are forbidden. Counts are reported
   rather than assumed from iteration 5.
7. Require the exact leverage-times-contrast deletion identity for `D_4` to
   close under the registered binary64 treatment. Require identical a1400
   map shape, units, WCS axes, and finite conditioned common support before
   comparing `D_4` and `D_5`.

A failure at any step stops scientific comparison. No tolerance or support
rule may be changed after the iteration-4 accounting values are opened.

## Existing JINC finalization and numerical treatment

Use the unchanged EL-F10 finalization and accounting definitions:

- denominator is usable only when finite and `abs(C) > 1e-8`;
- `Q` is usable only when finite and positive;
- raw formal coefficient is `C*C/max(Q, 1e-30)` on usable support;
- the positive-order statistic and `coverage_cut/10` factor define
  normalization support;
- signal is `N/C` only on that support; and
- `D_k = M_{k,-t}-M_k` on conditioned target support.

Binary64 unit roundoff is exactly `2^-53`, the finalization safety factor is
16, and the accumulator subtraction, division, and support-edge bounds remain
those frozen for EL-F10. The deletion-identity residual is reported in full
precision; it may not be used to retune those constants.

## Fixed persistence support

The primary persistence population is the intersection of the iteration-4 and
iteration-5 conditioned target supports, additionally requiring finite
`D_4`, `D_5`, signed leverage, and processed-signal contrast. Map identity is
checked before forming this intersection. Results also report the individual
conditioned-support sizes, intersection, union, iteration-4-only, and
iteration-5-only counts.

The four spatial regions are:

- the complete primary persistence population;
- the 20-arcsec aperture about fitted Neptune at map-world
  `(12.53903, -5.334553)` arcsec;
- the 20-arcsec aperture about the injected source at map-world
  `(0, -60)` arcsec; and
- the 40--120 arcsec injection-centered annulus excluding the 25-arcsec
  Neptune neighborhood.

## Fixed descriptive summaries

On the primary population and each nonempty registered region report:

- normalized inner product `dot(D_4,D_5)/(||D_4|| ||D_5||)`;
- signed Pearson correlation and Spearman rank correlation, without p-values;
- `beta = dot(D_4,D_5)/dot(D_4,D_4)` and
  `||D_5-beta D_4||/||D_5||`;
- sign agreement where both responses are exactly nonzero;
- RMS, maximum absolute value, signed sum, squared norms, and
  `2*dot(D_4,D_5)` so the difference-map energy remains auditable; and
- `D_5-D_4` RMS and maximum absolute value.

For fractions 1%, 5%, and 10%, select exactly
`ceil(fraction * population_size)` pixels by descending absolute response with
stable row-major tie breaking, separately for `D_4` and `D_5`. Report set
sizes, intersection, overlap fraction relative to each exact-size set,
Jaccard fraction, selection thresholds, and the fraction of total `D_5^2`
captured by the iteration-4-selected pixels.

For both iterations, separately report distributions of signed leverage,
processed-signal contrast, absolute coefficient-mass share, quadratic-support
share, total and target cancellation, and total unique-detector count on the
primary population and registered regions. Report the iteration-4 ledger
counts and carry forward the registered iteration-5 occurrence-count
summaries without pretending the aggregate values are pixel-paired maps.

Pearson and rank correlations are descriptive only because adjacent map
pixels are spatially correlated. No p-value, predictive cutoff, pass/fail
threshold, safeguard parameter, or winner is registered.

## Retained products

Retain a machine-readable result, complete `D_4`, `D_5`, and `D_5-D_4` maps,
common and iteration-specific conditioned-support masks, mechanism maps used
in the comparison, a fixed-metric CSV, a three-panel response figure, analysis
provenance, an execution record, a scientific interpretation, and a result
manifest. Every derived product is diagnostic-only and not an independently
calibrated sky product.

## Resource, repair, and stop boundary

Use one configured thread and `--grppiex seq`, at most one hour, 64 GiB peak
memory, and 8 GiB retained output. Preserve all inputs and prior products.
The owner's standing routine-defect direction permits analysis reruns for
narrow tooling defects and at most one isolated replacement Citlali replay
for a diagnostic-only instrumentation defect; record both attempts. It does
not permit a scientific, compatibility, method, input, gate, bound, or scope
change.

Stop after the descriptive persistence result. Do not compare interventions,
choose a predictor or action threshold, change a penalty, judge the detector,
recommend a safeguard, claim generic behavior, qualify FRUIT/JINC, open Gate
D or Stage B, change production, or perform Unity activity.
