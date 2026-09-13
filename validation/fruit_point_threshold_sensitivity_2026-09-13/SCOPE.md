# Reporting-only threshold sensitivity

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md),
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md),
and the [saved readout comparison](../fruit_point_profiled_readout_2026-09-13/SCIENTIFIC_REPORT.md).
The owner authorizes reporting signed errors and success counts at 5%, 7.5%,
10% and 15% from saved JSON only. This is not a new experiment, threshold
selection, fitter change or revision of any registered result.

Use FIT_COMPARISON.json, RATIOS.json and MEASUREMENTS.json from the profiled
readout packet. Keep 60 arcsec primary and 52 arcsec as a sensitivity check.
Report H/D absolute peaks separately from matched and crossed H/D ratios,
grouped by array and exact noise realization. Preserve every original,
fixed-original-geometry, repaired and truth-template record in the detailed
output, but keep coefficient-only diagnostics distinct from free-profile peaks.
Retain shifted, null and boundary safeguards separately; do not mix them into
the primary H/D denominators. Null percentage error is undefined, not zero.

Error is 100*(measured/expected - 1); ratio expected value is H/D=100/90.
Threshold comparisons use unrounded saved errors and inclusive absolute bounds.
Raw successes include every finite numeric result; usable successes additionally
require the unchanged saved 60-arcsec peak-use judgment (both members for a
ratio). Both use the complete population denominator, not only usable cases.
No usable status exists for the two diagnostic coefficient readouts. The 52-
arcsec tables may show the same 60-arcsec-usable subset for context only; this
is not a new inner-domain availability rule or an alternate terminal product.

Preserve full judgments and repaired numerical flags for each contributing
fit, including its inner probe. Numerical completion is reported separately
and does not silently filter counts. The original fit's independent numerical
completion was not assessed under the repaired criterion; keep that unavailable
rather than copying the repaired flag onto it.

Report every signed error, sorted magnitudes with case identities, median
signed error, median absolute error and worst case. These small dependent
populations do not establish uncertainty or a winning cutoff. Starlet remains
parked; all historical judgments and runtime/qualification results remain intact.
No map reads, optimization, fitting, PTC, feedback, new data or reserved-pointing
analysis is needed. Verify input hashes and arithmetic; commit this separate
reporting packet without modifying the previous packet.
