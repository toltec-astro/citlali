# corr_nw First-Pass Implementation Notes (2026-02-20)

## Goal
Implement a first-pass correlation-matrix-driven detector grouping mode for PTC cleaning, so cleaning can operate on correlation-defined subgroups within each network.

## Summary of What Was Added

### New cleaning grouping mode
- `clean.grouping` now supports `corr_nw`.
- Behavior:
  - For each network block, compute detector-detector correlation on masked/flagged-aware, z-scored timestreams.
  - Build an undirected graph where edges pass a correlation threshold.
  - Use connected components as raw groups.
  - Keep groups with `size >= min_group_size`.
  - Optionally collect all smaller components into a residual group.
  - Run existing PCA cleaning independently on each final group.

### New config block
- `timestream.processed_time_chunk.clean.corr_grouping.*`:
  - `enabled` (bool)
  - `metric` (`abs` or `signed`)
  - `corr_min` (default `0.6`)
  - `min_overlap` (default `300`)
  - `min_good_frac` (default `0.8`)
  - `min_group_size` (default `10`)
  - `max_samples` (default `20000`)
  - `clean_residual` (default `true`)

### Null-model integration
- `clean.null_model.grouping` now accepts `corr_nw`.

## Diagnostics Added to PTC netCDF Output
When `clean.corr_grouping.enabled=true` and `corr_nw` appears in `clean.grouping`, the PTC TOD output now includes:

- `corr_nw_group_id[n_scans, n_dets]`
  - Per-scan detector group index assigned by `corr_nw`.
  - Fill value `-2147483647` means unassigned.

- `corr_nw_network_ids[n_nws_corr]`
  - Network IDs matching the network-axis in the summary vars below.

- Summary vars, each shaped `[n_scans, n_nws_corr]`:
  - `corr_nw_n_groups`
  - `corr_nw_n_groups_raw`
  - `corr_nw_n_det_input`
  - `corr_nw_n_det_candidates`
  - `corr_nw_n_det_used`
  - `corr_nw_n_det_grouped`
  - `corr_nw_n_det_ungrouped`
  - `corr_nw_sample_step`

These are written per scan and intended for direct post-run QA.

## Files Modified
- `include/citlali/core/timestream/ptc/clean.h`
  - `CorrGroupingOptions`, `CorrGroupingResult`
  - `get_corr_groups(...)`
  - disjoint-set helper for connected components
- `include/citlali/core/timestream/ptc/ptcproc.h`
  - config parsing for `clean.corr_grouping.*`
  - `corr_nw` run path
  - per-scan corr diagnostics staging + write-out in `append_to_netcdf`
- `include/citlali/core/engine/engine.h`
  - PTC TOD netCDF schema additions for corr diagnostics variables

## Suggested Initial Config for First Unity Test
```yaml
timestream:
  processed_time_chunk:
    clean:
      enabled: true
      grouping: ["corr_nw"]
      corr_grouping:
        enabled: true
        metric: "abs"
        corr_min: 0.6
        min_overlap: 300
        min_good_frac: 0.8
        min_group_size: 10
        max_samples: 20000
        clean_residual: true
```

## Known Limitations of This First Pass
- Grouping method is graph connected-components, not hierarchical or spectral clustering.
- At low thresholds this can produce one giant group per network.
- At high thresholds it can over-fragment sparse networks.
- `corr_min=0.6` was chosen as a practical starting point from the two `redu24` single-scan tests.

## What to inspect after first run
1. `corr_nw_n_groups` and `corr_nw_n_groups_raw` by network/scan.
2. `corr_nw_n_det_grouped` vs `corr_nw_n_det_ungrouped`.
3. `corr_nw_group_id` maps vs residual striping behavior in maps.
4. Whether problematic networks (e.g., 8/11/12) split in a stable, useful way.

## Build/Test note in this local environment
- Full local compile was not completed here due environment dependency/toolchain mismatches (Conan CLI compatibility and missing `fmt` package in the non-Conan path).
- Changes are prepared for Unity build/run validation.

## redu27 Follow-Up (2026-02-20)

### What happened
- `processed_time_chunk.output.enabled: true` with no `processed_time_chunk.output.indices` wrote all 70 chunks, producing a very large PTC TOD file.
- Log size blew up from two high-volume sources:
  - despike path emitted large vector/matrix dumps at `info` level (`logger->info("error {}", error)` and related lines).
  - netCDF append path used `fo.getVars()` in `append_base_to_netcdf`, which triggered repeated HDF5 quantize-attribute probe diagnostics in this environment.

### Changes made
- `include/citlali/core/timestream/rtc/despike.h`
  - demoted despike instrumentation logs from `info` to `trace` to avoid routine production log flooding.
- `include/citlali/core/timestream/timestream.h`
  - removed `fo.getVars()` usage in `append_base_to_netcdf`.
  - switched to direct `fo.getVar("...")` for `SourceRa`, `SourceDec`, and `scan_indices`.

### Operational note for next run
- Keep chunk selection explicit when requesting PTC TOD output, e.g.:
```yaml
timestream:
  processed_time_chunk:
    output:
      enabled: true
      indices: [2]
```
