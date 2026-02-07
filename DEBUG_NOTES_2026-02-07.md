# Debug Notes (2026-02-07) - Beammap TOD Pointing + High-Severity Fixes

## Why these changes
We clarified that beammap TOD pointing should be referenced to the *actual source position* and include the solved
per-detector offsets (`x_t`, `y_t`). Beammap products are used to build APTs for combined-detector mapmaking, so
TOD pointing should reflect the final per-detector offsets (post reference-subtraction and derotation when enabled).

Separately, several high-severity bugs identified in the beammap audit were fixed (see below).

## Behavior change (potentially controversial)
Beammap TOD outputs (`det_lat`, `det_lon`, and derived `det_ra`, `det_dec`) now **include per-detector offsets**
when `map_grouping == "detector"`.

Previously, `calc_det_pointing` zeroed offsets whenever `map_grouping == "detector"`, so TOD pointing ignored
`x_t/y_t` even though beammapping had solved them. This meant TOD pointing was in a per-detector local frame,
not an absolute frame relative to the source.

### What changed
- Added an optional `apply_det_offsets` parameter to `engine_utils::calc_det_pointing`.
- Plumbed that parameter through the TOD netCDF appenders.
- Enabled `apply_det_offsets=true` specifically for beammap TOD output paths.

### Where it changed
- `include/citlali/core/utils/pointing.h`
  - Added `apply_det_offsets` parameter (default `false`).
- `include/citlali/core/timestream/timestream.h`
  - `append_base_to_netcdf(..., bool apply_det_offsets=false)`
- `include/citlali/core/timestream/rtc/rtcproc.h`
  - `append_to_netcdf(..., bool apply_det_offsets=false)`
- `include/citlali/core/timestream/ptc/ptcproc.h`
  - `append_to_netcdf(..., bool apply_det_offsets=false)`
- `include/citlali/core/engine/beammap.h`
  - RTC TOD write now passes `apply_det_offsets=true`.
  - PTC TOD write now passes `apply_det_offsets=true`.
  - Final TOD `det_lat/det_lon` recalculation uses `apply_det_offsets=true`.

### Rollback / revert guidance
If we decide this behavior is wrong, the minimal rollback is:
1. In `include/citlali/core/engine/beammap.h`, pass `false` instead of `true` in the three call sites:
   - RTC TOD write (`rtcproc.append_to_netcdf`)
   - PTC TOD write (`ptcproc.append_to_netcdf`)
   - Final TOD `calc_det_pointing` in the rewrite loop
2. Alternatively, delete the `apply_det_offsets` parameter and restore the old logic:
   - Remove parameter plumbing in `pointing.h`, `timestream.h`, `rtcproc.h`, `ptcproc.h`.
   - Restore the old `calc_det_pointing` signature and call sites.

## High-severity bug fixes

### 1) Beammap TOD output iteration off-by-one
**Fix:** Set `beammap_tod_output_iter` to `beammap_iter_max - 1` when `iter_tolerance <= 0`.

- **File:** `include/citlali/core/engine/engine.h`
- **Reason:** Iterations are 0-based; previously no iteration matched when `iter_max=1`.
- **Rollback:** Restore previous line `beammap_tod_output_iter = beammap_iter_max;`.

### 2) Uninitialized `lower_col` in Gaussian fit
**Fix:** Initialize `lower_col` to `0` when `bounding_box_pix <= 0`.

- **File:** `include/citlali/core/utils/fitting.h`
- **Reason:** `lower_col` was uninitialized and could lead to undefined ROI bounds / OOB access.
- **Rollback:** Remove the `lower_col = 0;` line.

### 3) Fruit-loops convergence math
**Fix:** Correct relative-difference formula for fruit loops convergence test.

- **File:** `include/citlali/core/engine/beammap.h`
- **Reason:** Old code used `A - B/A` instead of `(A - B)/A`.
- **Notes:** Added a safe denominator branch for zero-valued pixels.
- **Rollback:** Restore the original single-line `diff = abs(...)` expression.

### 4) Sensitivity median computation used uninitialized values
**Fix:** Iterate over `nw_sens.size()` rather than `sens.size()` when collecting unflagged detector values.

- **File:** `include/citlali/core/engine/beammap.h`
- **Reason:** The previous loop could leave `sens` partially uninitialized, corrupting the median.
- **Rollback:** Revert the loop bound to `sens.size()` (not recommended).

### 5) Header reference detector index for auto-selected reference
**Fix:** After auto-selecting the reference detector, set `beammap_reference_det` to the resolved index.

- **File:** `include/citlali/core/engine/beammap.h`
- **Reason:** Downstream header writers used `beammap_reference_det` and could index `-99` even when a
  valid reference was found.
- **Rollback:** Remove the assignment `beammap_reference_det = beammap_reference_det_found;`.

## Suggested validation (manual)
- Run a beammap with `subtract_reference_det: true` and `reference_det: -99`.
- Confirm TOD outputs now show detector pointing offsets relative to the source (e.g., `det_lat/det_lon` differ
  by detector and match APT offsets).
- Confirm BEAMMAP.REF_* header keys use the resolved detector index rather than `-99`.
- Confirm TOD output exists for `iter_max=1` and `iter_tolerance=0` (previously missing).


## 2026-02-07 (late) - crash during map fitting

Observed crash at "fitting maps". Likely cause: `fit_to_gaussian` inner-radius search could leave
`ir/ic` uninitialized if no pixel exceeded `init_flux`, leading to invalid indexing.

### Fix
- Initialize `ir/ic` to map center and track `found_peak`.
- If inner-radius search finds nothing, fall back to global max.

**File:** `include/citlali/core/utils/fitting.h`

**Rollback:** Remove `found_peak` logic and restore previous `ir/ic` handling.

## 2026-02-07 (later) - reference detector selection

Observed APTs with array centers offset from (0,0) even though `x_t/y_t` were plotted. The code
was selecting the reference detector as the one closest to (0,0), while the config comment
says it should be closest to the median of the first array. This mismatch can shift the
array centroid if the detector closest to (0,0) is not near the array center.

### Fix
- When `reference_det < 0`, choose the detector closest to the median (x_t, y_t) of the
  first array (unflagged if available, else all detectors).

**File:** `include/citlali/core/engine/beammap.h`

**Rollback:** Restore distance-from-(0,0) selection logic.

## 2026-02-07 (latest) - reference median from selected networks

User expectation is array center at (0,0). Using the detector closest to (0,0) or
closest to the array median still allows large offsets if that detector is far
from the array centroid (e.g., missing network 6). We now define the reference
location as the **median x_t/y_t** of selected networks:

- Primary: `nw=3`
- Fallback: `nw=2,3,4`
- If none unflagged, fall back to array 0 median (previous behavior)

We subtract this **median location** directly. For metadata, we also record the
nearest detector index to that median in `reference_det`.

**Files:**
- `include/citlali/core/engine/beammap.h`
- `data/config.yaml` (comment updated)

**Rollback:** Restore prior reference selection logic (array median or (0,0) based).

### Header consistency
BEAMMAP.REF_X_T/Y_T in FITS/netCDF now use `reference_x_t/y_t` from APT meta when present, so
headers reflect the reference **location** (median) rather than the nearest detector’s offset.
