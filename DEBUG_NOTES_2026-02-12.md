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

