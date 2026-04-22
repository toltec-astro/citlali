# Non-`mJy/beam` Units Audit Notes (2026-04-22)

## Scope
This note records unit-conversion issues found during the April 2026 signal-calibration audit that do **not** affect the default `mJy/beam` path.

Current working assumption:

- ongoing flux-loss investigation is focused on `mJy/beam`
- the issues below should be fixed later, in a separate unit-conversion cleanup

## What Appears Safe For `mJy/beam`
The default `mJy/beam` calibration path sets:

- `flux_conversion_factor(i) = 1`

Code path:

- `src/citlali/core/engine/calib.cpp`

So the specific non-`mJy/beam` problems below should not perturb current `mJy/beam` map amplitudes.

## Issue 1: TOD Calibration Indexes Flux Conversion By Array Id Instead Of Detector Id

### Problem
`calc_flux_calibration()` builds `flux_conversion_factor` as a detector-length vector:

- `src/citlali/core/engine/calib.cpp`

But TOD calibration later uses:

- `Eigen::Index array_index = calib.apt["array"](i);`
- `in.fcf.data(i) = calib.flux_conversion_factor(array_index);`

in:

- `include/citlali/core/timestream/rtc/calibrate.h`

That is not a detector-row index. It is an array id such as `0/1/2`.

### Why This Is Wrong
The calibration vector is stored in detector-row order, while `array_index` is just the array label.

So for non-`mJy/beam` outputs, detectors in arrays 1 and 2 can end up reading the conversion factor from detector rows `1` and `2`, rather than from their own detector row or even their own array block.

### Affected Units

- `MJy/sr`
- `Jy/pixel`
- `uK`

### Why `mJy/beam` Is Safe
For `mJy/beam`, the conversion vector is all ones, so the bad indexing is numerically harmless there.

## Issue 2: `mJy_beam_to_uK()` Drops The Beam Area

### Problem
The helper:

- `include/citlali/core/utils/utils.h`

contains a commented-out beam-area term:

- `//auto beam_area_rad = ...`
- `// ... K_to_mJy_beam = ... //*beam_area_rad;`

so the current `uK` conversion is effectively beam-size independent.

### Why This Is Wrong
A conversion from `mJy/beam` to temperature must depend on beam solid angle. If the beam term is omitted, the result no longer behaves like a true `mJy/beam -> uK` conversion.

### Affected Units

- `uK`

## Issue 3: `uK` Metadata/Header Conversion Uses A Different FWHM Unit Convention

### Problem
The runtime `uK` calibration path currently calls:

- `engine_utils::mJy_beam_to_uK(1, freq_Hz, det_fwhm);`

in:

- `src/citlali/core/engine/calib.cpp`

where `det_fwhm` is in arcsec.

But the output/header metadata paths call:

- `engine_utils::mJy_beam_to_uK(1, freq, fwhm * ASEC_TO_RAD);`

in:

- `include/citlali/core/engine/engine.h`

so those paths pass radians.

### Why This Is Wrong
Even after fixing the helper physics, the code still needs a single explicit unit contract for the `fwhm` argument. Right now runtime calibration and file metadata do not agree.

### Affected Units

- `uK`

## Likely Fix Order

1. Fix `calibrate_tod()` to index flux conversion by detector row, not array id.
2. Define the contract for `engine_utils::mJy_beam_to_uK()` explicitly:
   - either `fwhm` in arcsec
   - or `fwhm` in radians
3. Restore the beam-area dependence in the `uK` conversion helper.
4. Update all `uK` metadata/header writers to use the same helper contract as the runtime calibration path.
5. Recheck any derived conversion keywords written to TOD/map products.

## Minimum Regression Tests To Add Later

1. A unit test where detector rows have intentionally different conversion factors and `calibrate_tod()` must preserve detector-row mapping.
2. A `uK` conversion test showing that changing beam FWHM changes the output as expected.
3. A metadata test confirming runtime `uK` calibration and written FITS/TOD conversion keywords agree numerically.
4. A smoke test that `mJy/beam` outputs remain unchanged after the non-`mJy/beam` cleanup.

## Recommended Boundaries For The Future Fix
Keep this cleanup separate from the current `mJy/beam` flux-loss investigation.

Reason:

- Issue 1 is real, but it is inactive for `mJy/beam`.
- Issues 2 and 3 are specific to `uK`.
- Folding these changes into the current `mJy/beam` debugging loop would add noise without helping isolate the present science issue.
