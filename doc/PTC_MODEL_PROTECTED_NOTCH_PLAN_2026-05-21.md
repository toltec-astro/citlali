# PTC Model-Protected Notch Refactor Notes

Date: 2026-05-21
Branch: `gw_dev`
Starting commit: `b83fd7e3 Add fixed RTC line-audit notch stage`

## Motivation

The recent beammap tests show scan-direction kernel elongation when aggressive notch
filtering is applied while the compact source signal is still present in the TOD.
Because scan coordinate is effectively time coordinate, notching source-bearing TOD
can convolve the beam with the notch impulse response and broaden fitted beams.

The desired next test is to keep line suppression available for future science runs,
but apply aggressive line removal to model-subtracted residual TOD whenever a source
or fruitloops map model is available.

## Pre-Refactor Behavior

- Static `raw_time_chunk.filter.notch` runs inside the RTC filter stage.
- `raw_time_chunk.line_audit.fixed_notch_enabled` applies fixed census notches before
  the normal RTC filter chain when `line_audit.pre_filter_enabled` is true.
- `raw_time_chunk.line_audit.apply_shared_notches` applies dynamic shared notches in
  the RTC branch before normal filtering.
- `raw_time_chunk.line_audit.post_filter_apply_shared_notches` and
  `post_filter_apply_detector_notches` apply after RTC filtering/downsampling, but
  still on source-bearing TOD.
- For beammap Gaussian iterations, the fitted Gaussian feedback is applied only to
  `scans.data`; for fruitloops/map feedback, both `scans.data` and `kernel.data` are
  projected through `map_to_tod`.

## Intended Refactor

- Add an opt-in PTC/model-protected line-audit stage.
- Default all new controls to off to preserve existing behavior.
- Run the new stage after the current sky/source model has been subtracted and before
  PTC cleaning.
- Reuse the existing RTC line-audit notch implementation to avoid duplicating PSD,
  clustering, fixed-notch, shared-notch, detector-local notch, and edge-guard logic.
- Require a model-subtracted TOD by default so iteration 0 and non-fruitloops science
  passes do not receive aggressive source-bearing notches by accident.

## Kernel Handling

- Fruitloops `map_to_tod` subtracts/adds both `scans.data` and `kernel.data`, so
  the PTC notch stage acts on residual signal and residual kernel. After add-back,
  the model component is not broadened by the residual notch pass.
- Gaussian feedback only subtracts/adds `scans.data`; therefore it is not a safe
  hook for this stage when kernel shape is part of the diagnostic.

## Threading Note

The PTC hook reuses RTC line-audit helpers that cache diagnostic summaries in
maps keyed by scan id. The beammap scan loop can run in parallel, so calls to the
model-protected PTC line-audit helper are serialized there. PTC cleaning and
mapmaking remain parallel.

## Hook Points

- Science (`Lali`): after fruitloops `NegativeMap` subtraction and before
  `ptcproc.run(...)`.
- Pointing: same as science.
- Beammap: after fruitloops map subtraction in iterations `current_iter > 0` and
  before `ptcproc.run(...)`. The Gaussian feedback path is intentionally not
  treated as model-protected because it does not protect `kernel.data`.

## Rollback Plan

The new feature is config-gated. To back out behavior without reverting code, set:

```yaml
timestream:
  raw_time_chunk:
    line_audit:
      ptc_model_protected_enabled: false
```

To remove the code entirely, revert the commit that introduces this note and the
associated changes in:

- `include/citlali/core/timestream/rtc/rtcproc.h`
- `include/citlali/core/engine/engine.h`
- `include/citlali/core/engine/lali.h`
- `include/citlali/core/engine/pointing.h`
- `include/citlali/core/engine/beammap.h`
- `data/config.yaml`

## Acceptance Checks

- Existing configs with the new option omitted should produce unchanged behavior.
- A model-protected test should show line suppression in residual TOD without
  increasing kernel scan-direction elongation.
- Beammap tests should compare kernel FWHM ratio and signal FWHM ratio against a
  no-notch or conservative-notch reference.
