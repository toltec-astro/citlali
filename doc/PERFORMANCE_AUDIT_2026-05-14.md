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

## Validation Update: 2026-05-15

The first `P-001` mitigation was implemented on
`codex/perf-map-accumulation-noise-lifecycle` by replacing the naive
mapmaker's largest triplet/sparse staging path with a tiled direct accumulator
for signal, weight, kernel, and coverage maps. Noise-map accumulation and
post-map diagnostics remain on the original paths.

Representative Unity comparisons used a single-observation naive reduction in
`~/work_toltec/local_data/2025-C1-COM-21/gw_sandbox`, with noise maps disabled,
OpenMP enabled, and `n_threads: 15`.

Validated runs:

- `redu07`: current `gw_dev` at `68093fc4`, Citlali
  `v4.0.0-350-g68093fc4`, wall time `3m22.039004006s`.
- `redu08`: performance branch at `5f639e82`, Citlali
  `v4.0.0-353-g5f639e82`, rebuilt with native release flags, wall time
  `3m46.542854437s`.

The configs were byte-identical for the comparison. The checked FITS products
were obs and coadd maps for `a1100`, `a1400`, and `a2000`. For `signal_I`,
`weight_I`, `coverage_I`, and `coverage_bool_I`, all checked numerical
differences were exactly zero and all coverage-bool mismatch counts were zero.

An earlier performance-branch run, `redu06`, was slower because the Unity build
was missing `-march=native` in release flags. Build metadata copied from Unity
showed `gw_dev` had `-march=native` while the old performance-branch build did
not. After rebuilding the performance branch with native release flags, the
map products matched `gw_dev` exactly and wall time returned close to the
baseline.

The small-noise `P-002` validation was then run with identical generated
Citlali configs:

- `redu09`: performance branch at `5f639e82`, Citlali
  `v4.0.0-353-g5f639e82`, noise maps enabled with `n_noise_maps: 5`,
  wall time `3m39.877138032s`.
- `redu10`: `gw_dev` at `68093fc4`, Citlali `v4.0.0-350-g68093fc4`,
  noise maps enabled with `n_noise_maps: 5`, wall time
  `3m28.932295901s`.

For obs and coadd FITS products across `a1100`, `a1400`, and `a2000`,
coverage masks were identical. Science and noise-product image differences
were floating-point roundoff only; the largest signal-map max absolute
difference was `3.55e-15`, and the largest checked FITS max absolute
difference was `2.84e-14`. Histogram netCDF products were exactly equal.
Mapdiag and PSD netCDF products differed only at roundoff. Coadd empirical
weight scales were identical: `0.488468900326375`, `0.509016124292622`, and
`0.542209831503308`.

This validates the tiled naive accumulator for the tested standard
single-observation naive case with five noise realizations. It does not show a
whole-pipeline wall-clock speedup in this run. Targeted timers were added after
this validation so the next Unity run can separate naive accumulation time from
the rest of the reduction:

- `populate_maps_naive total`
- `populate_maps_naive accumulate`
- `populate_maps_naive merge`

The next structural work should address the `P-002`/`P-003` noise-map and
noise-product lifecycle directly. Repeat the comparison with the standard
production noise count before treating that path as fully validated.

A first `P-002` implementation prototype added a `TiledNoiseAccumulator` so
the naive mapmaker could accumulate noise realizations into touched tiles
instead of copying, zeroing, and merging a full noise cube for every
`populate_maps_naive()` call. The final resident `MapBuffer::noise` cube and
output format were unchanged. The intent was to reduce per-call temporary
memory traffic and merge work when each time chunk touched a sparse subset of
the output map, while still keeping the full realization cube resident for
downstream empirical products, PSDs, filtering, and optional realization
writes.

The first tiled-noise run was then validated against the existing `gw_dev`
control:

- `redu10`: `gw_dev` at `68093fc4`, Citlali `v4.0.0-350-g68093fc4`,
  noise maps enabled with `n_noise_maps: 5`, wall time
  `3m28.932295901s`.
- `redu11`: performance branch at `8ee78c30`, Citlali
  `v4.0.0-355-g8ee78c30`, same generated Citlali config, wall time
  `3m38.200127022s`.

For obs and coadd FITS products across `a1100`, `a1400`, and `a2000`,
coverage masks were identical. Science maps, weights, coverage, formal
weights, noise variance, and S/N products differed only at roundoff. The
largest signal-map max absolute difference was `5.33e-15`, and the largest
checked FITS max absolute difference was `2.13e-14`. Histogram netCDF products
were exactly equal. Mapdiag and PSD differences were roundoff-scale, with one
summed scalar `map_coverage_sum` differing by `4.66e-10`. Coadd empirical
weight scales were identical: `0.488468900326375`, `0.509016124292622`, and
`0.542209831503308`.

The new timers showed the bottleneck is not the merge step for this small
five-noise-map case:

- `populate_maps_naive total`: 124 calls, sum `423.871s`, mean `3.418s`.
- `populate_maps_naive accumulate`: 124 calls, sum `422.394s`, mean `3.406s`.
- `populate_maps_naive merge`: 124 calls, sum `1.431s`, mean `0.0115s`.

Interpretation: the tiled noise accumulator is scientifically clean for the
tested five-noise-map case, but it does not produce a whole-pipeline speedup in
this small-noise run. The next performance target should be the
detector/sample accumulation loop itself, while the standard `n_noise_maps: 25`
comparison should still be run to expose memory-pressure effects hidden by the
small case.

The standard 25-noise-map comparison was then run:

- `redu13`: performance branch at `8ee78c30`, Citlali
  `v4.0.0-355-g8ee78c30`, noise maps enabled with `n_noise_maps: 25`, wall
  time `4m28.969160386s`.
- `redu14`: `gw_dev` at `68093fc4`, Citlali `v4.0.0-350-g68093fc4`,
  same generated Citlali config, wall time `3m30.785061426s`.

Coverage masks were identical for all checked obs and coadd FITS products
across `a1100`, `a1400`, and `a2000`. Map products again differed only at
roundoff. The largest signal-map max absolute difference was `5.33e-15`, the
largest checked FITS max absolute difference was `4.26e-14`, netCDF mapdiag,
histogram, and PSD differences were roundoff-scale, and coadd empirical weight
scales matched exactly: `0.350383961423778`, `0.370406096095144`, and
`0.393315916421293`.

The targeted timers showed that the tiled-noise prototype moved time in the
wrong direction for the standard noise count:

- `populate_maps_naive total`: 124 calls, sum `1160.820s`, mean `9.361s`.
- `populate_maps_naive accumulate`: 124 calls, sum `1155.568s`, mean `9.319s`.
- `populate_maps_naive merge`: 124 calls, sum `4.851s`, mean `0.0391s`.

Decision: the tiled-noise prototype is not a good default path for this
standard case. It preserves numerical results, but it slows the reduction by
about 58 seconds, or roughly 28 percent, while merge time remains negligible
relative to the detector/sample accumulation loop. The branch therefore removed
`TiledNoiseAccumulator` and restored the prior full scratch-buffer noise
accumulation behavior.

The cleanup was validated with another standard 25-noise-map run:

- `redu15`: performance branch at `75c44812`, Citlali
  `v4.0.0-356-g75c44812`, same generated Citlali config as `redu14`, wall
  time `3m41.028757068s`.

`redu15` matched the `gw_dev` control at roundoff: coverage masks were
identical, the largest signal-map max absolute difference was `3.55e-15`, the
largest checked FITS max absolute difference was `2.84e-14`, histogram and
stats netCDF products were exact, PSD/mapdiag differences were roundoff-scale,
and coadd empirical weight scales matched exactly. Removing tiled-noise
accumulation recovered most of the slowdown (`redu13` was `4m28.969s`), but the
branch was still about 10 seconds slower than `gw_dev` (`redu14` was
`3m30.785s`).

Final decision for the scalar tiled map accumulator: remove it from the naive
path as well. It is numerically safe, but the tested reductions do not show a
wall-clock benefit, and carrying a known small slowdown forward would obscure
future structural measurements. The branch restores naive mapmaking to the
prior sparse-triplet accumulation/merge behavior while keeping
`populate_maps_naive total`, `populate_maps_naive accumulate`, and
`populate_maps_naive merge` timers.

Jinc mapmaking is a separate case. It already accumulates dense jinc footprint
blocks into scratch maps, so the scalar `TiledMapAccumulator` is not the right
primitive. A future jinc optimization should use a block-aware tile accumulator
or a precomputed sample-footprint plan if timing supports it. The branch now
adds coarse jinc timers:

- `populate_maps_jinc total`
- `populate_maps_jinc setup`
- `populate_maps_jinc accumulate`
- `populate_maps_jinc merge`
- `populate_maps_jinc_parallel total`
- `populate_maps_jinc_parallel accumulate`

The first Unity jinc run with these timers was `redu17`, using Citlali
`v4.0.0-357-gfe862127`, `method: jinc`, `parallel_policy: omp`,
`n_threads: 15`, and `n_noise_maps: 25`. The full process time was
`10m8.453s`. The timer split was decisive:

- `populate_maps_jinc total`: 124 calls, sum `5812.034s`, mean `46.871s`
- `populate_maps_jinc accumulate`: 124 calls, sum `5802.822s`, mean `46.797s`
- `populate_maps_jinc setup`: 124 calls, sum `5.346s`, mean `0.043s`
- `populate_maps_jinc merge`: 124 calls, sum `3.847s`, mean `0.031s`

This means the jinc bottleneck is the detector/sample accumulation body, not
scratch setup or merge. The next structural change therefore targets standard
noise-map accumulation directly: for the usual `nmb == omb` path, each
detector's jinc-gridded signal contribution is accumulated once into a
detector-local template, then scaled into each noise realization. This is
mathematically equivalent because `in.noise.data(nn,i)` is constant across
samples for a detector/noise realization. A simple work estimate keeps the
existing per-sample path when the template's map-sized write would not be
cheaper than per-sample footprint splatting.

That serial jinc template path was validated by `redu18`, using Citlali
`v4.0.0-358-gb0ba0c58` and the same generated config as `redu17`. Full process
time dropped from `10m8.453s` to `5m40.969s`. The timer split moved as intended:

- `populate_maps_jinc total`: 124 calls, sum `2056.529s`, mean `16.585s`
- `populate_maps_jinc accumulate`: 124 calls, sum `2046.610s`, mean `16.505s`
- `populate_maps_jinc setup`: 124 calls, sum `6.080s`, mean `0.049s`
- `populate_maps_jinc merge`: 124 calls, sum `3.821s`, mean `0.031s`

Product comparison against `redu17` over shared `coverage_bool_I` regions found
zero coverage-mask mismatches in the checked FITS maps. Signal maps and noise
products differed only at roundoff; the largest signal-map max absolute
difference was `2.66e-15`, the largest `noise_variance_I` max absolute
difference was `3.0e-15`, and the largest checked FITS max absolute difference
was `3.41e-13` in `coverage_I`.

The detector-template optimization has now also been ported to
`populate_maps_jinc_parallel` for detector-grouped OMB noise maps, which is the
standard beammap jinc path. This new parallel path is intentionally narrower
than the serial path: coadd noise and non-detector grouping continue to use the
existing per-sample noise accumulation. It is not yet Unity-validated. The next
beammap jinc run should compare against the previous jinc control and check
both products and `populate_maps_jinc_parallel accumulate` timing.

On 2026-05-16, lightweight timers were also added around centralized
`MapBuffer` post-map product methods on the performance branch. The goal is to
quantify P-003/P-002 before changing behavior:

- `MapBuffer::normalize_maps`
- `MapBuffer::normalize_polarized_maps`
- `MapBuffer::calc_map_psd`
- `MapBuffer::calc_map_hist`
- `MapBuffer::calc_median_err`
- `MapBuffer::calc_median_rms`
- `MapBuffer::calc_noise_products total`
- `MapBuffer::calc_noise_products map`

These timers should be checked on the next representative Unity reduction with
noise maps enabled. They do not gate or skip any products; they only expose the
relative cost of normalization, PSD/histogram generation, median diagnostics,
and empirical noise-product calculations.

The first run with those post-map timers was `redu19`, using Citlali
`v4.0.0-360-gcb252bdf`, the same generated config as `redu14` and `redu15`,
`method: naive`, and `n_noise_maps: 25`. Product comparison against the
`gw_dev` control `redu14` was roundoff-clean: coverage masks were identical,
the largest signal-map max absolute difference was `3.55e-15`, and the largest
checked FITS max absolute difference was `2.84e-14` in coadd
`noise_variance_I`.

`redu19` showed the post-map products are not a meaningful bottleneck for this
small map:

- `MapBuffer::normalize_maps`: total `0.035s`
- `MapBuffer::calc_map_psd`: total `0.891s`
- `MapBuffer::calc_map_hist`: total `0.281s`
- `MapBuffer::calc_noise_products total`: total `0.211s`

The remaining naive-mapmaking merge cost was more interesting:

- `populate_maps_naive merge`: 124 calls, sum `47.664s`, mean `0.384s`
- merge timer wall union: `46.693s`

This means the timer is mostly serialized work, not overlapping wait. The
dominant suspect is the full noise-cube merge: for this run the coadd noise
cube is about `0.115 GB`, and the old path added that whole cube into the
shared map buffer once per scan.

The first attempted fix, `c9847b4c` / `redu20`, tracked individual touched
pixels and merged only those pixels. It was numerically clean against `redu19`
and `redu14`: coverage masks matched, and the largest checked FITS difference
remained at roundoff scale (`2.84e-14`). It was not a speed win. The full
process time stayed flat at `218.485s`, `populate_maps_naive merge` increased
slightly to `48.473s`, and `populate_maps_naive accumulate` increased to
`306.260s`.

The second attempt, `e5494b12` / `redu21`, replaced the per-pixel hash
bookkeeping with monotonic touched rectangles per noise map. Scratch tensors
lazily zero newly exposed rectangular bands before accumulation, and the merge
adds only the final touched rectangle for each map/noise plane as contiguous
Eigen blocks. The generated config was identical to `redu20`. Product
comparison was again roundoff-clean: zero coverage-mask mismatches against
`redu20`, `redu19`, and `redu14`, and largest checked FITS differences at
`2.84e-14`. The targeted merge timer improved, but the whole process did not:

- `populate_maps_naive merge`: `42.596s` sum, down from `47.664s` in
  `redu19`.
- `populate_maps_naive accumulate`: `267.652s` sum, up from `255.678s` in
  `redu19`, but down from `306.260s` in `redu20`.
- `populate_maps_naive total`: `311.418s` sum, down from `318.740s` in
  `redu19`.
- `Citlali Process`: `3m42.633s`, slower than the `redu19`/`redu20` whole-run
  times near `3m38.5s`.

Interpretation: the bounded block merge is numerically safe and improves the
specific merge path, but this compact map is not the large-empty-map case where
it should dominate end-to-end runtime.

The next structural change, `ad6591f4` / `redu22`, targeted the
detector/sample noise accumulation body in the same standard naive/noise run.
Instead of writing each valid sample into every noise realization,
`populate_maps_naive` builds a sparse nearest-pixel weighted-signal template
per detector and scales that detector template into each noise realization
afterward. This preserves the final resident noise cube and output format while
removing the `n_noise_maps` loop from the per-sample hot path.

`redu22` used Citlali `v4.0.0-363-gad6591f4`; its generated config was
identical to `redu21`. Product comparisons against `redu21`, `redu19`, and
`redu14` were roundoff-clean: zero `coverage_bool_I` mismatches in all checked
FITS maps, largest checked FITS difference `3.55e-14` in coadd
`noise_variance_I`, and map sidecar netCDFs (`mapdiag`, `hist`, `psd`,
`stats`) differed only at roundoff. A broad netCDF comparison also shows
`rtcdiag` differences, but those diagnostics are outside this mapmaker change
and were not used as validation evidence.

The timing moved in the intended direction:

- `Citlali Process`: `3m26.752s`, down from `3m38.546s` in `redu19` and
  `3m42.633s` in `redu21`.
- `populate_maps_naive accumulate`: `206.491s` sum, down from `255.678s` in
  `redu19` and `267.652s` in `redu21`.
- `populate_maps_naive merge`: `39.523s` sum, down from `47.664s` in `redu19`
  and `42.596s` in `redu21`.

This is the first tested standard-naive/noise optimization on this branch that
shows both roundoff-clean products and a clear end-to-end win on the `148481`
benchmark.

The current unvalidated worktree change targets the remaining
`populate_maps_naive merge` cost. Sparse triplet compression now happens before
the global merge mutex is acquired, keeping only the final shared dense-map
additions, pointing-matrix additions, and noise-cube additions under lock. The
timer split was also expanded:

- `populate_maps_naive merge prepare`: sparse triplet compression outside the
  lock.
- `populate_maps_naive merge locked`: lock wait plus all shared-map updates.
- `populate_maps_naive merge maps`: signal, weight, kernel, coverage, and
  pointing additions while locked.
- `populate_maps_naive merge noise`: noise-cube additions while locked.

The next Unity run should use the same `148481` config, compare against
`redu22` over `coverage_bool_I`, and check whether total process time improves
or whether the newly exposed prepare/locked split only moves cost between
timer labels.
