# Debug Notes (2026-02-12) - Raw Ingest Speedups (Phase 1)

## Goal
Reduce raw ingest overhead in RTC construction while keeping behavior unchanged.

This phase intentionally avoids deep algorithm changes and focuses on:
1. Removing avoidable per-scan object churn.
2. Avoiding repeated metadata reads for each sliced data read.
3. Keeping rollback simple if subtle runtime issues appear.

## Summary of changes

### A) Added fused read+solve RTC path for non-gap mode

Instead of:
- `load_rawobs(...)` -> vector of `RawTimeStream` slices
- `populate_rtc(...)` -> solver pass over loaded vector

we now support:
- `populate_rtc_from_rawobs(...)` -> read slice + solve + copy directly into final RTC matrix in one loop.

Files:
- `include/citlali/core/engine/kidsproc.h`

Why:
- Removes intermediate container allocations/copies for normal (non-gap) ingest.
- Reduces memory pressure and per-scan bookkeeping overhead.

### B) Switched non-gap ingest call sites to fused path

Updated generators in:
- `include/citlali/core/engine/lali.h`
- `include/citlali/core/engine/pointing.h`
- `include/citlali/core/engine/beammap.h`

Behavior:
- Non-gap mode now uses `populate_rtc_from_rawobs(...)`.
- Gap mode keeps existing `load_rawobs_gaps(...) + populate_rtc_gaps(...)` path.

### C) Removed unused `scan_rawobs` payload forwarding in pointing/beammap

Pointing and beammap generators previously returned tuples containing `scan_rawobs`,
but farm stages did not consume that data.

Files:
- `include/citlali/core/engine/pointing.h`
- `include/citlali/core/engine/beammap.h`

Why:
- Eliminates unnecessary tuple wrapping and movement of potentially large vectors.

### D) Added data-kind metadata cache in `load_data_item`

`load_data_item(...)` used to call `kidsdata::get_meta(...)` on every sliced read.
Now it caches `KidsDataKind` by source filepath.

File:
- `include/citlali/core/engine/kidsproc.h`

Why:
- Prevents repeated metadata probing of the same files across many scans.

## Rollback plan (fast path)

If this phase causes any subtle regression, use one of these rollback levels:

### Rollback Level 1 (disable fused non-gap path; keep metadata cache)
Revert call-site usage in:
- `include/citlali/core/engine/lali.h`
- `include/citlali/core/engine/pointing.h`
- `include/citlali/core/engine/beammap.h`

Restore old flow:
- call `load_rawobs(...)`
- then call `populate_rtc(...)`

This preserves all prior control flow while keeping metadata cache improvement.

### Rollback Level 2 (restore old payload types in pointing/beammap)
In:
- `include/citlali/core/engine/pointing.h`
- `include/citlali/core/engine/beammap.h`

Restore tuple payload:
- `std::tuple<RTCData, std::vector<RawTimeStream>>`

and restore farm unpacking of tuple.

### Rollback Level 3 (remove metadata cache)
In `include/citlali/core/engine/kidsproc.h`, restore `load_data_item(...)` to:
- call `kidsdata::get_meta(...)` each time
- remove `m_data_item_kind_cache` member and `<unordered_map>` include.

## Validation to run after this phase

1. Functional parity:
- Compare map outputs/summary stats for one obs with non-gap mode pre/post change.

2. Runtime:
- Compare walltime of ingest-heavy stage (`raw time chunk processing`) pre/post.

3. Memory:
- Watch RSS during first ~10 scans to confirm lower transient usage.

4. Gap-mode safety:
- Run one obs with `interp_over_gaps=true` to verify unchanged behavior on gap path.

---

# Debug Notes (2026-02-12) - Jinc Mapmaking High-ROI Fixes

## Goal
Reduce jinc mapmaking hot-loop cost and remove detector-loop issues with minimal behavior risk.

## Summary of changes

### A) Precomputed squared jinc kernels

Added cached squared kernel matrices so weight/coverage/noise paths avoid repeated
`mat_block.array().square()` in inner loops.

File:
- `include/citlali/core/mapmaking/jinc_mm.h`

New members:
- `jinc_weights_sq_mat`
- `jinc_weights_sq_mat_subpix`

Also clear/rebuild caches in `allocate_jinc_matrix(...)`.

### B) Detector gating fix (`run_det`) in both jinc paths

`run_det` was computed but not applied. It now short-circuits detector processing when:
- `run_polarization=true` and detector has `fg==-1`.

File:
- `include/citlali/core/mapmaking/jinc_mm.h`

### C) Beammap detector-parallel index sizing fix

In `populate_maps_jinc_parallel(...)`, detector work vector now sizes to `n_dets`
instead of `omb.signal.size()`.

File:
- `include/citlali/core/mapmaking/jinc_mm.h`

### D) Reused scratch buffers in sequential jinc path

Replaced per-call scratch allocation (`omb_copy`, `cmb_copy`) with `thread_local`
scratch buffers and explicit zero/reset helpers.

File:
- `include/citlali/core/mapmaking/jinc_mm.h`

### E) Merge only touched regions

Sequential jinc merge now tracks per-map touched bounding boxes and merges only
those blocks for:
- signal
- weight
- coverage
- kernel
- noise tensors

File:
- `include/citlali/core/mapmaking/jinc_mm.h`

## Fast rollback plan

1. Rollback all jinc high-ROI fixes:
- Revert `include/citlali/core/mapmaking/jinc_mm.h` to previous commit.

2. Partial rollback options:
- Keep detector fixes but remove performance changes:
  - Remove squared kernel caches.
  - Restore old full-map merge behavior.
  - Restore per-call local scratch allocation.

3. Minimal safety rollback:
- Keep only:
  - `run_det` gating fix.
  - `n_dets` sizing fix in parallel detector map.

## Validation checklist

1. Functional parity:
- Compare jinc map outputs pre/post on one obs (signal + weight + kernel + coverage).

2. Noise map parity:
- Run with `noise_maps.enabled=true` and compare summary stats / RMS.

3. Performance:
- Measure walltime around mapmaking stage on same obs/chunking settings.

4. Beammap detector mode:
- Verify detector-group beammap run does not regress after index sizing change.

---

# Debug Notes (2026-02-12) - Wiener Filter Speedups (OMP Path)

## Goal
Reduce Wiener filter walltime in Unity-style builds (`CITLALI_USE_WIENER_FILTER_OMP=ON`) by:
1. Parallelizing noise-map filtering.
2. Reusing FFTW plans/buffers instead of recreating per call.
3. Removing per-iteration matrix allocations in denominator loop.

## Build context checked

From `~/foo/CMakeCache.txt`:
- `CITLALI_USE_WIENER_FILTER_OMP:BOOL=ON`
- `CMAKE_BUILD_TYPE:STRING=Release`
- OpenMP flags enabled (`-fopenmp`)

## Summary of code changes

### A) Parallel noise filtering in engine

File:
- `include/citlali/core/engine/engine.h`

Change:
- In `Engine::run_wiener_filter(...)`, when `CITLALI_USE_WIENER_FILTER_OMP` is set,
  noise map filtering now runs with `#pragma omp parallel for schedule(dynamic)`.
- Uses new Wiener method `filter_noise_threadsafe(...)`.
- Non-OMP path keeps previous sequential progress-bar loop.

### B) FFTW context reuse in OMP Wiener filter

File:
- `include/citlali/core/mapmaking/wiener_filter_omp.h`

Change:
- Added `WienerFilter::FFTWContext` with cached `a/b` buffers and forward/inverse plans.
- Added `get_thread_fft_context(rows, cols)` returning thread-local FFTW context.
- Planning is guarded with `#pragma omp critical (wfFFTWPlan)` to avoid FFTW planner races.

### C) Reused numerator/convolution compute kernels

File:
- `include/citlali/core/mapmaking/wiener_filter_omp.h`

Added helpers:
- `calc_numerator_from_input(...)`
- `run_convolve_on_input(...)`
- `divide_by_denom(...)`

Change:
- `calc_numerator()` now calls `calc_numerator_from_input(filtered_map)`.
- `run_convolve()` now calls `run_convolve_on_input(filtered_map, normalize)`.

### D) Denominator loop allocation reduction

File:
- `include/citlali/core/mapmaking/wiener_filter_omp.h`

Change in `calc_denominator()`:
- Removed repeated creation/destruction of per-thread FFTW plans on each call.
- Replaced with per-thread cached contexts via `get_thread_fft_context(...)`.
- Moved temporary matrices (`in_local`, `out_local`, `ffdq`, `in_prod`, `shift_indices`) outside
  the inner iteration loop so they are reused across iterations.
- Removed per-iteration `Eigen::MatrixXd updater` allocation; uses scalar-array update directly.

## Rollback plan

1. Full rollback:
- Revert:
  - `include/citlali/core/mapmaking/wiener_filter_omp.h`
  - `include/citlali/core/engine/engine.h`

2. Partial rollback (keep safer changes):
- Keep FFTW context reuse and denominator reuse.
- Restore sequential noise loop in `engine.h` if thread-safety concerns appear.

3. Minimal rollback:
- Keep only thread-safe parallel noise loop removed; return fully to prior behavior.

## Validation checklist

1. Performance:
- Compare filter walltime before/after with same obs + config.
- Track map loop total and per-map filtering times.

2. Numerical parity:
- Compare filtered signal/weight maps.
- Compare filtered noise maps for same random seed/config.

3. Stability:
- Run with `noise_maps.enabled=true` and `n_noise_maps > 1`.
- Run with and without `wiener_filter.lowpass_only`.

---

# Debug Notes (2026-02-12) - Wiener Filter Runtime Checkpoints

## Goal
Improve runtime visibility during Wiener filtering and filtered-map file output by adding
`info` logs at key checkpoints where long pauses were previously silent.

## Summary of changes

File:
- `include/citlali/core/engine/engine.h`

Function:
- `Engine::run_wiener_filter(...)`

Added `info` checkpoints for:
1. FITS header preparation start.
2. Per-map start (`map i/n`, array name).
3. Map filtering complete.
4. Noise filtering start/complete (including `n_noise` on OMP path).
5. Error renormalization start (with map context).
6. Per-map write start.
7. FITS handle close start/complete when `pfits->destroy()` is called.
8. Per-map complete.
9. Final vector clear/finalize start/complete.

## Rollback plan

If logs are too chatty:
1. Revert only this file:
- `include/citlali/core/engine/engine.h`
2. Specifically remove the `logger->info(...)` additions in
`Engine::run_wiener_filter(...)` and keep all data-path logic unchanged.

---

# Debug Notes (2026-02-12) - LTO ODR Warning Fix (PTC Sensitivity)

## Symptom
With `-march=native` + LTO enabled, link-time ODR warnings appeared for:
- `internal::Window`
- `internal::hann(...)`

between:
- `include/citlali/core/timestream/ptc/sensitivity.h`
- `build/_deps/kidscpp-src/include/kids/timestream/solver_psd.h`

## Root cause
Both headers defined entities in a global `namespace internal` with overlapping names,
which is an ODR hazard and can trigger misoptimization under LTO.

## Fix
File changed:
- `include/citlali/core/timestream/ptc/sensitivity.h`

Change:
- Renamed local namespace from `internal` to `citlali_ptc_internal`.
- Updated all internal references in that header accordingly.

## Rollback
If needed, revert:
- `include/citlali/core/timestream/ptc/sensitivity.h`

(Warning may return when LTO is enabled.)

Update:
- Also updated `src/citlali/core/timestream/ptc/sensitivity.cpp` namespace to
  `citlali_ptc_internal` so `stat(...)` and `freq(...)` definitions match declarations.

---

# Debug Notes (2026-02-12) - Wiener OMP Stability Hotfix (double free)

## Symptom
Runtime abort during filtered coadd Wiener step:
- `double free or corruption (out)`
- seen just after `starting filtered coadded maps map ...`.

## Hotfix applied
File changed:
- `include/citlali/core/mapmaking/wiener_filter_omp.h`

Changes:
1. `calc_numerator_from_input(...)` now uses local FFTW alloc/plan/free per call.
2. `run_convolve_on_input(...)` now uses local FFTW alloc/plan/free per call.
3. `calc_denominator()` switched to the known-stable local FFTW lifecycle and sequential update logic matching the non-OMP implementation.

## Rationale
The crash signature is consistent with heap corruption/double free in the cached FFTW context path.
This hotfix removes that path from active execution to prioritize runtime stability.

## Performance note
This may reduce the speedup from cached-plan optimization, but keeps the thread-parallel
noise filtering path in `engine.h` intact.

## Rollback/next-step options
1. If stability is confirmed and speed is acceptable, keep this state.
2. If speed regression is too large, reintroduce cache optimization incrementally with ASAN/UBSAN testing.
