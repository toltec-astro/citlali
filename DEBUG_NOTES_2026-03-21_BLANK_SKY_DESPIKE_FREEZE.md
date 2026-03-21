# Blank-Sky Despike Freeze

This note records the accepted blank-sky despiking state as of
`2026-03-21`.

## What "Freeze" Means

For this branch, "freeze" means:

- stop exploratory tuning of the RTC impulsive mask
- stop exploratory tuning of the PTC second-pass residual flagger
- keep the current thresholds and windows as the accepted blank-sky defaults
- only reopen this branch for a real bug, regression, or a clearly separate
  new use case such as pointing-safe handling

It does **not** mean the code can never change again. It means this line of
development is complete enough to serve as the baseline.

## Accepted Runtime Configuration

The current working config is represented by
[70_reduce.yaml](/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/70_reduce.yaml)
for the full 13-observation GOODS-N blank-sky reduction.

### RTC / Pre-PCA

- `raw_time_chunk.despike.enabled: true`
- `raw_time_chunk.despike.min_spike_sigma: 8.0`
- `raw_time_chunk.despike.local_residual.enabled: true`
- `raw_time_chunk.flagging.network_step_mask.enabled: true`

Accepted impulsive capture settings:

- `impulsive_capture.enabled: true`
- `impulsive_capture.min_good_frac: 0.8`
- `impulsive_capture.min_event_z: 6.0`
- `impulsive_capture.near_event_z: 4.0`
- `impulsive_capture.max_events_per_network: 3`
- `impulsive_capture.snippet_pre_window_sec: 0.25`
- `impulsive_capture.snippet_post_window_sec: 0.75`

Accepted impulsive coincidence settings:

- `impulsive_coincidence.enabled: true`
- `impulsive_coincidence.min_good_frac: 0.8`
- `impulsive_coincidence.event_score_thresh: 6.0`
- `impulsive_coincidence.min_det_used: 32`
- `impulsive_coincidence.min_impulsive_det_frac: 0.05`
- `impulsive_coincidence.min_alignment_frac: 0.5`
- `impulsive_coincidence.min_networks_aligned: 5`
- `impulsive_coincidence.high_score_override_thresh: 180.0`
- `impulsive_coincidence.high_score_min_networks_aligned: 3`
- `impulsive_coincidence.cluster_tol_sec: 0.03`
- `impulsive_coincidence.mask_pre_window_sec: 0.03`
- `impulsive_coincidence.mask_post_window_sec: 0.10`
- `impulsive_coincidence.max_flagged_fraction: 0.10`

### PTC / Post-PCA

Accepted second-pass residual settings:

- `processed_time_chunk.flagging.second_pass_local.enabled: true`
- `min_spike_sigma: 8.0`
- `min_good_frac: 0.5`
- `baseline_window_sec: 0.25`
- `sigma_scale: 0.75`
- `delta_sigma_scale: 0.75`
- `raw_candidate_rel_sigma_scale: 1.0`
- `raw_window_sec: 0.18`
- `raw_half_peak_frac: 0.5`
- `raw_max_width_sec: 0.18`
- `delta_window_sec: 0.12`
- `delta_half_peak_frac: 0.5`
- `delta_max_width_sec: 0.10`
- `max_step_shift_z: 3.0`
- `merge_within_detector_sec: 0.08`
- `cluster_events_sec: 0.08`
- `min_cluster_detectors: 3`
- `high_score_cluster_override: 9.0`
- `max_auto_flag_clusters_per_network: 3`

Operational intent:

- isolated post-PCA survivors may be auto-flagged
- broad messy residual storms remain diagnostic-first and are vetoed from
  automatic flagging

### Reduction Logging

Accepted logging state:

- each reduction writes `reduNN/citlali.log.gz`
- both `console` and `citlali_logger` share the same gzip sink
- the gzip stream now passes integrity checks

## Validation Runs

### `redu62`

Purpose:

- full 13-observation blank-sky run validating the asymmetric RTC impulsive
  capture and mask windows

Result:

- impulsive masked row count stayed effectively unchanged relative to the
  earlier full-set baseline
- flagged fraction increased in the expected "same rows, slightly longer
  windows" pattern
- no map-level regression strong enough to reopen the RTC tuning work

### `redu65`

Purpose:

- 4-observation debug run validating the C++ port of the PTC second-pass
  residual flagger

Result:

- isolated PTC survivors were caught with tiny added-flag fractions
- busy residual-storm cases were vetoed from auto-flagging
- behavior matched the Python prototype closely enough to accept the port

### `redu66`

Purpose:

- full production-style 13-observation run with the RTC asymmetric windows and
  PTC second pass both enabled

Result:

- filtered coadds stayed very close to `redu60`
- no obvious science-map regression
- despiking behavior looked stable enough to treat as the new blank-sky
  baseline

### `redu67`

Purpose:

- single-observation confirmation of the reduction-local compressed log fix

Result:

- `citlali.log.gz` passes `gzip -t`
- beginning and end of the log are readable
- this closes the last remaining known issue on the branch

## Practical Conclusion

The blank-sky despiking branch is frozen at this state.

Future work should proceed as separate efforts:

- pointing-safe glitch handling
- PCA/common-mode optimization
- richer engineering diagnostics and UI

This branch should only be reopened for:

- a real regression in blank-sky map products
- a bug in the accepted RTC/PTC logic
- or a concrete need to expand behavior beyond the validated blank-sky scope
