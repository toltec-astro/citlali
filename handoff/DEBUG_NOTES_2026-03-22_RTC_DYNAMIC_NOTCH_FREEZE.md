# RTC Dynamic Shared-Notch Freeze

This note records the accepted blank-sky RTC line-audit and dynamic shared-line
notch state as of `2026-03-22`.

## What "Freeze" Means

For this branch, "freeze" means:

- stop exploratory tuning of RTC dynamic shared-notch application
- keep the current live line-audit thresholds and apply policy as the accepted
  blank-sky defaults
- keep detector-local line candidates diagnostic-only for now
- only reopen this branch for a real regression, bug, or a clearly separate
  use case

It does **not** mean the line-audit code can never change again. It means the
current blank-sky policy is stable enough to serve as the baseline.

## Accepted Runtime Configuration

The accepted working config is represented by
[70_reduce.yaml](/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/70_reduce.yaml)
for the full 13-observation GOODS-N blank-sky reduction.

### Static vs Dynamic RTC Notches

- static config-set RTC notch filter remains available in Citlali
- static notch is **disabled** in the accepted blank-sky config:
  - `raw_time_chunk.filter.notch.enabled: false`
- dynamic RTC shared-line notching is the accepted active path:
  - `raw_time_chunk.line_audit.enabled: true`
  - `raw_time_chunk.line_audit.apply_shared_notches: true`

### Accepted RTC Line-Audit Detection Settings

- `line_min_hz: 1.0`
- `line_max_hz: 60.0`
- `segment_sec: 4.0`
- `min_segment_sec: 2.0`
- `overlap_frac: 0.5`
- `continuum_radius_bins: 8`
- `prominence_thresh: 12.0`
- `cm_prominence_thresh: 10.0`
- `min_good_frac: 0.8`
- `min_windows: 2`
- `max_peaks_per_detector: 3`
- `max_det: 128`
- `min_det_for_network: 16`
- `cluster_tol_hz: 0.15`
- `notch_min_detector_frac: 0.90`
- `notch_min_detectors: 64`
- `notch_min_cm_prominence: 100.0`
- `detector_min_prominence: 300.0`
- `detector_min_line_power_frac: 0.25`
- `bad_detector_max_cluster_frac: 0.03`

### Accepted RTC Shared-Notch Apply Policy

- `apply_shared_notches: true`
- `apply_min_support_networks: 3`
- `apply_min_detector_frac: 0.90`
- `apply_min_common_mode_prominence: 1.0e9`
- `apply_width_scale: 1.5`
- `apply_min_width_hz: 0.25`
- `apply_max_width_hz: 1.50`
- `apply_max_notches: 3`
- `apply_cluster_tol_hz: 0.25`

Operational meaning:

- only multi-network shared families are auto-notched
- the earlier single-network common-mode override is effectively disabled
- detector-local periodic offenders remain diagnostic-only and are **not**
  auto-flagged or auto-notched in the accepted baseline

## Accepted Observed Behavior

At the accepted `redu09` state:

- the dominant applied dynamic families are:
  - `29.767 Hz`
  - `11.006 Hz`
- the previously over-applied `47.777 Hz` family is no longer in the applied
  set
- all applied support counts are `>= 3`

This is the key policy outcome of the freeze.

## Validation Runs

### `redu04`

Purpose:

- verify that RTC line-audit provenance is fully written to RTC TOD outputs

Result:

- `CONFIG.RTC.LINE_AUDIT.*` variables are present in RTC TOD
- runtime audit path and saved diagnostic fields are confirmed live

### `redu05`

Purpose:

- first debug-style validation with dynamic shared-notch application enabled in
  the normal RTC pipeline

Result:

- dynamic actuation and metadata writing both worked
- but this run could not cleanly isolate efficacy for high-frequency families
  because the standard RTC FIR low-pass was also active

### `redu06`

Purpose:

- isolated single-observation validation with FIR/downsampling disabled so
  saved RTC TOD retains the high-frequency families after dynamic notching

Result:

- direct suppression of the dynamic-notch targets was confirmed
- example reductions:
  - `47.777 Hz`: about `127x`
  - `11.006 Hz`: about `22x`

This established that the runtime notch path is materially removing lines, not
just logging recommendations.

### `redu07`

Purpose:

- first full science-style run with dynamic shared notches enabled

Result:

- dynamic actuation was too aggressive
- the `47.777 Hz` family dominated the applied set
- many applications were weakly supported and map-space differences versus
  `redu00` were too large

Conclusion:

- the implementation was working, but the apply policy was too permissive

### `redu08`

Purpose:

- first config-only rollback with the single-network common-mode promotion
  effectively disabled

Result:

- actuation dropped substantially
- map impact improved
- but the `2`-network applications still moved the maps more than desired

Conclusion:

- better, but not yet accepted

### `redu09`

Purpose:

- second config-only rollback, raising `apply_min_support_networks` from `2`
  to `3`

Result:

- applied row count dropped again
- applied support counts are all `>= 3`
- filtered coadds moved back close to baseline:
  - `a1100` corr vs `redu00`: `0.9903`
  - `a1400`: `0.9905`
  - `a2000`: `0.9951`
- visual comparisons of `a1100` and `a1400` coadds show mild redistribution,
  not an obvious new artifact pattern

Conclusion:

- `redu09` is accepted as the blank-sky dynamic shared-notch baseline

## Practical Conclusion

The RTC dynamic shared-notch branch is frozen at the `redu09` policy state.

Blank-sky filtering work should now proceed with:

- RTC line audit enabled
- dynamic shared notches enabled
- static RTC notch list disabled
- only multi-network shared families eligible for automatic notch application

Future work should proceed as separate efforts:

- PCA/common-mode optimization, especially in `a2000`
- detector-level periodic offender policy, if still needed
- pointing-safe or bright-source-safe filtering behavior

This branch should only be reopened for:

- a real regression in blank-sky science products
- a bug in the RTC line-audit or dynamic-notch implementation
- or a clearly new operational requirement
