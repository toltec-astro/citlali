# FRUIT EL-F10 targeted JINC accounting result r0.6

Test ID: `SCI-FRUIT-EL-F10-R4-NOISE-PASS-LEDGER-REPAIR-R0.1`

Result: **all registered gates pass; the direct map response is exactly
accounted for by local signed JINC leverage times processed-signal contrast**

Status: **descriptive development evidence from one observation, UID, scan,
and iteration; not a safeguard selection or qualification result**

## Gate result

The R0.6 registration validated all 21 bound files before opening the JINC
accounting values. The repaired analysis then passed every frozen gate:

- all nine ordinary a1100/a1400/a2000 signal, kernel, and realized-weight
  planes are bitwise identical to EL-F6 N5;
- all three unscaled formal-weight planes are bitwise identical;
- checkpoint scientific values are identical after the one registered
  historical-default normalization;
- all six total-accumulator re-finalization checks are exact;
- the corrected target ledger contains 305 unique proposed samples: 271
  admitted and 34 already final-flagged; and
- subtracting UID 4460 scan 5's registered `N_t`, `C_t`, and `Q_t`
  reconstructs the existing EL-F8 A5-map result within the pre-registered
  binary64 bounds, with no gained, lost, or unexplained support pixels.

Across 94,635 common conditioned pixels, the maximum signal difference from
the registered counterfactual is `6.82121e-13 mJy/beam`, compared with a
maximum allowed bound of `2.96313e-7 mJy/beam`. The maximum formal-coefficient
difference is `1.45717e-16`, compared with a maximum allowed bound of
`1.01298e-11`.

## Exact accounting result

On the 4,229 conditioned pixels in the target scan footprint, the deletion
identity closes with maximum absolute residual
`2.44360e-13 mJy/beam`. The response ranges from `-98.9712` to
`+17.7835 mJy/beam`, with RMS `6.40472 mJy/beam`. The signed target share
`C_t/C` has median `1.68166e-6`, 95th percentile `0.0170061`, and maximum
`0.191538`.

The largest absolute response occurs at map row 139, column 76, approximately
AZ `+102 arcsec`, EL `-38 arcsec`. There, `C_t/C=0.0487068`; the target-only
and without-target signals are approximately `2090.44` and
`58.4578 mJy/beam`, respectively. Their `-2031.98 mJy/beam` contrast times
the 4.87% signed share gives the observed `-98.9712 mJy/beam` deletion
response. There are 164 unique detectors contributing at that pixel.

The highest signed leverage is 19.15% at approximately AZ `-8 arcsec`, EL
`+8 arcsec`, where the response is `+14.3844 mJy/beam`. The largest positive
response is `+17.7835 mJy/beam` at approximately AZ `-6 arcsec`, EL
`+8 arcsec`, with 14.82% signed leverage.

The four map pixels that originally triggered the scan-local penalty contain
no UID 4460 scan-5 occurrence and have zero deletion response. The 20-arcsec
off-source injected-source aperture likewise contains no target occurrence.
The trigger location, response arcs, and injected compact source are therefore
distinct in this registered case.

## Routine repair lineage

The approved R2 retry encountered three narrow diagnostic-path defects, all
handled under the owner's standing routine-defect direction without changing
the science algorithm, target, gates, bounds, or claim limits:

1. the NetCDF scalar helper now accepts native Python strings, bytes, and
   NumPy scalar values;
2. receipt planes are transformed once from internal Eigen orientation to the
   existing FITS orientation; and
3. target-ledger rows are recorded only during the observation-map JINC pass,
   excluding the duplicate noise-only pass.

The isolated replacement replay took 33.17 seconds, peaked at 871,006,208
bytes resident memory, and emitted no error- or critical-level log records.
Its accounting receipt and checkpoint are byte-for-byte identical to those
from the retained defective-ledger replay; only the diagnostic ledger changed.

Local verification after the last code repair passed 632 enabled CTest tests
with one intentionally disabled test, 252 baseline/FRUIT Python tests, the
full configuration preflight, Ruff, Python compilation, and `git diff
--check`.

## Evidence and limits

The machine-readable result, component maps, binned summaries, trigger table,
and diagnostic figure accompany this record. The component products are
explicitly diagnostic and are not calibrated standalone sky maps or restart
state.

This result applies only to observation 123424, a1400, UID 4460, zero-based
scan 5, and the registered iteration-4-to-5 trajectory. It does not judge the
detector, establish a generic mechanism, select a safeguard, qualify FRUIT,
authorize production use, open Gate D, start Stage B, or authorize Unity
activity.
