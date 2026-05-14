# Citlali Performance Bottleneck Audit

Date: 2026-05-14
Branch audited: `gw_dev`
Scope: read-only speed and memory bottleneck audit after cleanup work.

No code changes were made as part of this audit.

## Executive Summary

The dominant performance risks are in mapmaking, noise-map handling, and unconditional post-map diagnostics. The standard science path defaults to naive array-grouped mapmaking, and that path still allocates large temporary sparse/triplet structures before serially merging into dense maps. Noise products are potentially much more expensive than the science map itself because full jackknife cubes are kept in memory and repeatedly traversed by normalization, PSD, histogram, RMS, and empirical-noise calculations.

Overall confidence is medium-high for identifying likely bottlenecks from static inspection. Priority ordering should still be verified with representative TolTEC observations and wall-clock/peak-RSS profiling.

Recommended first measurements:

1. Standard science reduction with naive array grouping, noise disabled.
2. Same dataset with noise enabled.
3. Same dataset with PSD/hist/noise products gated off, if possible.
4. Jinc mapmaking with `subpixel_n=1` and a representative higher value.
5. Beammap detector grouping with several iterations, noise disabled/enabled.

## Findings

| ID | Priority | Area | Summary | Evidence | Suggested Next Step |
| --- | --- | --- | --- | --- | --- |
| P-001 | P1 | Standard mapmaking | Default science reductions use naive array grouping, where each scan accumulates per-sample `Triplet` vectors and then serially converts/merges them into dense maps under a global mutex. This is likely allocation-heavy and limits scan-level scaling. | `data/config.yaml:97`, `include/citlali/core/engine/todproc.h:1092`, `include/citlali/core/engine/lali.h:392`, `include/citlali/core/mapmaking/naive_mm.h:100`, `include/citlali/core/mapmaking/naive_mm.h:240`, `include/citlali/core/mapmaking/naive_mm.h:384` | Benchmark `populate_maps_naive()` separately. Compare current triplet path against a direct dense/tiled accumulator for one representative scan. |
| P-002 | P1 | Noise maps | Noise maps allocate full `n_rows * n_cols * n_maps * n_noise` cubes. Naive mapmaking copies/zeros full noise cubes per call and later merges them. Post-processing repeatedly traverses every realization. | `include/citlali/core/engine/todproc.h:1499`, `include/citlali/core/mapmaking/naive_mm.h:144`, `include/citlali/core/mapmaking/naive_mm.h:421`, `src/citlali/core/mapmaking/map.cpp:232`, `src/citlali/core/mapmaking/map.cpp:442`, `src/citlali/core/mapmaking/map.cpp:653` | Add peak-RSS benchmarks with realistic map sizes and `n_noise`. Consider memory-budget checks and streaming empirical products. |
| P-003 | P1/P2 | Post-map diagnostics | PSD, histogram, median error, and median RMS are called unconditionally after mapmaking. PSDs are especially expensive because each map/noise realization allocates FFTW buffers/plans and performs radial binning over the full map. | `include/citlali/core/engine/lali.h:120`, `include/citlali/core/engine/pointing.h:237`, `src/citlali/core/mapmaking/map.cpp:378`, `include/citlali/core/utils/utils.h:705`, `include/citlali/core/utils/utils.h:829` | Add profiling timers around normalization, PSD, histogram, median, and noise-product stages. Gate expensive diagnostics when not required by outputs. |
| P-004 | P2 | Jinc mapmaking | Jinc mapmaking does per-sample block updates over kernel footprints. Subpixel mode multiplies precomputed kernels by `subpixel_n^2`; detector-parallel mode also runs hot-loop validation on each sample/block. | `include/citlali/core/mapmaking/jinc_mm.h:203`, `include/citlali/core/mapmaking/jinc_mm.h:491`, `include/citlali/core/mapmaking/jinc_mm.h:730`, `include/citlali/core/mapmaking/jinc_mm.h:931`, `include/citlali/core/mapmaking/jinc_mm.h:1091` | Benchmark jinc with realistic footprints and subpixel settings. Move repeated validation into debug/preflight paths if profiling confirms overhead. |
| P-005 | P2 | Pointing/index computation | Detector pointing is computed once for map sizing and again inside mapmaking per detector/chunk. The config has a `precompute_pointing` key, but it is marked ignored. | `include/citlali/core/engine/todproc.h:1235`, `include/citlali/core/mapmaking/naive_mm.h:215`, `include/citlali/core/mapmaking/jinc_mm.h:515`, `data/config.yaml:133` | Profile `calc_det_pointing()` cost and consider a bounded per-scan cache of pointing and pixel indices. |
| P-006 | P2 | NetCDF IO | KIDs reads and TOD/diagnostic writes are protected by a single global `netcdf_io_mutex()`. This is probably intentional for HDF5/netCDF safety, but it can flatten scan-farm parallelism when IO dominates. | `include/citlali/core/engine/io.h:55`, `include/citlali/core/engine/kidsproc.h:237`, `include/citlali/core/timestream/rtc/rtcproc.h:4656`, `include/citlali/core/timestream/ptc/ptcproc.h:3053` | Profile with TOD output and diagnostics enabled/disabled. If IO-bound, consider a dedicated ordered writer or persistent file handles. |
| P-007 | P2 | PCA cleaning | Standard cleaning builds dense covariance matrices and uses either Spectra over dense covariance or Eigen's full self-adjoint eigensolver. Adaptive/null-model selectors add repeated covariance/eigensolver work when enabled. Defaults keep adaptive modes disabled. | `include/citlali/core/timestream/ptc/clean.h:1912`, `include/citlali/core/timestream/ptc/clean.h:2009`, `include/citlali/core/timestream/ptc/clean.h:2043`, `include/citlali/core/timestream/ptc/clean.h:1093`, `include/citlali/core/timestream/ptc/clean.h:1453` | Keep adaptive options guarded. Add benchmark cases for array/nw/all cleaning groups at representative detector counts. |
| P-008 | P2 | Beammap iterative mode | Beammap stores all processed chunks, copies `ptcs0`/`calib_scans0` each iteration, and regenerates noise signs for all chunks inside the per-map reset loop. | `include/citlali/core/engine/beammap.h:755`, `include/citlali/core/engine/beammap.h:2574`, `include/citlali/core/engine/beammap.h:2750`, `include/citlali/core/engine/beammap.h:2768` | Move noise randomization outside the per-map reset loop and profile memory/time from full `ptcs` copies. |
| P-009 | P2/P3 | Map filtering | The non-OMP Wiener/convolve paths allocate FFTW buffers and plans repeatedly. The OMP implementation has more plan/scratch reuse but depends on build configuration. | `include/citlali/core/mapmaking/wiener_filter.h:561`, `include/citlali/core/mapmaking/wiener_filter.h:998`, `include/citlali/core/mapmaking/wiener_filter_omp.h:770`, `include/citlali/core/engine/engine.h:6451` | Verify production builds use the intended filtering implementation. Benchmark filtered maps with and without noise filtering. |
| P-010 | P3 | Benchmarks | The test binary initializes Google Benchmark but there are no benchmark registrations. Current tests do not cover performance regressions. | `tests/main.cpp:1`, `tests/main.cpp:11`, `tests/test_utils.cpp:1` | Add microbenchmarks for `populate_maps_naive`, `calc_2D_psd`, `calc_noise_products`, and jinc detector/sample loops. |

## Profiling Plan

### Highest-Value Timers

Add or collect timing around these stages first:

1. Raw KIDs read and solver output assembly.
2. RTC processing.
3. PTC cleaning.
4. Weight calculation and reset.
5. Mapmaking only.
6. Map normalization.
7. PSD and histogram generation.
8. Noise products.
9. FITS and netCDF output.

### Representative Cases

Use one standard science observation with array grouping and naive mapmaking as the baseline. Record:

- `n_pts` per scan.
- `n_dets`.
- `n_maps`.
- map rows and columns.
- `n_noise`.
- number of scans.
- `n_threads`.
- `parallel_policy`.
- wall time per stage.
- peak RSS.

Then repeat with:

- noise disabled.
- noise enabled.
- coadd disabled/enabled if both are common.
- jinc mapmaking with `subpixel_n=1`.
- jinc mapmaking with a representative higher `subpixel_n`.
- TOD output disabled/enabled.
- diagnostic sidecars disabled/enabled.

### Microbenchmarks To Add

1. `populate_maps_naive()` on synthetic PTC chunks with realistic shape, flags, and map indices.
2. `populate_maps_jinc()` and `populate_maps_jinc_parallel()` with realistic kernel footprints.
3. `MapBuffer::calc_map_psd()` on representative map sizes, with and without noise maps.
4. `MapBuffer::calc_noise_products()` for realistic `n_noise`.
5. `Cleaner::get_eigen_values()` over representative detector group sizes.
6. Beammap iteration reset/mapmaking path with synthetic `ptcs` vectors.

## Notes And Uncertainties

- This was a static audit. The report identifies code paths likely to dominate, but the final priority should be set from representative profiling.
- Some bottlenecks are scientifically intentional tradeoffs. Noise maps and PSD products are expensive because they preserve useful uncertainty diagnostics. The question is whether the pipeline should compute them eagerly, stream them, or make them more selectively configurable.
- The global netCDF lock may be necessary for library safety. Treat it as a scaling constraint to measure, not automatically as a bug.
- Adaptive PCA/null-model features are off by default. They should stay opt-in unless benchmarked on standard reductions.

