# GOODS-N RTCDiag Survey Handoff

Date: 2026-03-17

This note captures the current state after:

1. the RTC-side despike / step-mask / impulsive-event instrumentation work
2. the new lightweight `rtcdiag` sidecar product
3. the first full 13-obsnum GOODS-N survey run using `rtcdiag` without RTC/PTC
   timestream outputs

It is intended as the next handoff point after:

- `DEBUG_NOTES_2026-03-13_GOODSN_MP_HANDOFF.md`
- `doc/RTC_FLAGGING_AUDIT_2026-03-16.md`

## Executive Summary

- The new lightweight `rtcdiag` sidecar product is now viable as the normal
  analysis interface for RTC contamination diagnostics.
- The full 13-obsnum GOODS-N survey completed successfully in `redu40` with
  `rtcdiag` enabled and both RTC/PTC timestream outputs disabled.
- A concurrency bug was likely exposed by the new sidecar write path; adding a
  shared netCDF I/O mutex in Citlali fixed the observed `151928` crash.
- The shared `TransientEvent` refactor has now passed:
  - a two-obsnum RTC+PTC smoke test
  - a full 13-obsnum RTC survey
  - a sidecar-only validation run
- `151930` and especially `152524` are not representative of the broader
  GOODS-N set. The harder survey obsnums are now clear from `redu40`.
- The next step should not be more runtime tuning. The right next move is to
  use `redu40` to turn diagnostics into an operations/science policy, then only
  resume code work where the map-level evidence justifies it.

## What Was Validated

### 1. RTC diagnostic infrastructure

The following changes are now in place and validated:

- shared `TransientEvent` model for raw-like, delta-like, and step-like events
- explicit unit-safe RTC despike counters
- `network_step_mask` triggered before `altaz_destripe`
- lightweight `rtcdiag` sidecar output
- reporting tools that read `rtcdiag` directly

Relevant repo pieces:

- `include/citlali/core/timestream/rtc/despike.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`
- `include/citlali/core/engine/engine.h`
- `tools/blank_sky/rtc_impulsive_slot_report.py`
- `tools/blank_sky/rtcdiag_survey_report.py`

### 2. Sidecar-only product path

The sidecar path was validated with:

- `redu38`: `rtcdiag` written alongside RTC/PTC outputs for direct comparison
- `redu39`: `rtcdiag` written with both RTC/PTC timestream outputs disabled

The key result is that `rtcdiag` can now support the diagnostic/reporting path
without requiring full RTC TOD outputs.

### 3. Full 13-obsnum survey

The current full lightweight survey baseline is:

- reduction: `/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu40`
- survey report:
  `/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu40/rtcdiag_survey_report/RTCDIAG_SURVEY_REPORT.md`

This run used:

- all 13 GOODS-N science obsnums
- `rtcdiag` enabled
- RTC timestream output disabled
- PTC timestream output disabled

All 13 obsnums completed successfully.

## Crash / Stability Note

The first `redu40` attempt crashed during `151928` with:

- `std::runtime_error: no tones found`

The most credible working explanation is concurrent netCDF/HDF5 read/write:

- scan loading reads raw TolTEC netCDF through `KidsDataProc`
- worker threads were simultaneously writing `rtcdiag`
- the Unity netCDF/HDF5 stack likely was not safe enough for that overlap

The Citlali-side fix was to serialize raw netCDF reads against netCDF writes
using a shared mutex:

- `include/citlali/core/engine/io.h`
- `include/citlali/core/engine/kidsproc.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`
- `include/citlali/core/timestream/ptc/ptcproc.h`

After that change, the rerun completed successfully and `151928` is present in
`redu40`:

- `/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu40/151928/raw/toltec_commissioning_science_151928_rtcdiag.nc`

## Important Caveat About Survey Comparisons

`redu40` should not be compared numerically to the older `redu37` survey as if
they were identical products.

Why:

- `redu37` was summarized from sparse RTC TOD output
- `redu40` is summarized from `rtcdiag` over all scans

The more meaningful comparison is:

- `redu38` vs `redu39` for sidecar-path validation
- `redu39` vs `redu40` on common obsnums for stability sanity checks

That sanity check looked good for `152286`: step fraction, masking, and overall
severity remained effectively stable.

## Current Survey Readout

From `redu40`, the worst obsnums in `a1100` (`nw0-5`) are now:

1. `151103`
2. `150784`
3. `152526`
4. `151096`
5. `152524`

`151928` lands in the middle of the pack, not as a pathological outlier.

Headline values from the survey report:

- `151103`: strongest mixed/coherent case
- `150784`: strong step/coherent case
- `152526`: strongest impulsive-slot case
- `152524`: still useful as an impulsive-heavy case, but not the best general
  stress test

The top scan-network rows and top impulsive slots are listed in:

- `/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu40/rtcdiag_survey_report/rtcdiag_survey_top_scan_network_rows.csv`
- `/Users/wilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/reduced/redu40/rtcdiag_survey_report/rtcdiag_survey_top_impulsive_slots.csv`

## Interpretation

The main contamination families still look like:

- step-like / coherent subgroup structure
- impulsive / heavy-tail detector-local contamination

The step-family work is now in much better shape:

- the survey can see it cheaply through `rtcdiag`
- `network_step_mask` is applied in the correct place in the RTC order
- the diagnostics are good enough to support masking policy discussions

The impulsive family is still the open frontier:

- the diagnostic path is now much better
- the runtime handling is improved relative to where we started
- but we do not yet have a final generalized action policy for all impulsive
  survivors

## Recommended Next Step

Do not change runtime behavior again immediately.

Instead, use `redu40` to turn diagnostics into policy.

### Phase 1: Obsnum triage

Use these four obsnums as the next map/science check set:

- `151103`: worst mixed/coherent case
- `150784`: strong step/coherent case
- `152526`: strongest impulsive-slot case
- `151930`: familiar moderate/control case

Reason:

- this set spans the key failure families much better than continuing to focus
  only on `151930` and `152524`

### Phase 2: Small map-level check

For those obsnums, compare:

- suspicious `S/N > 4` candidate density
- AzTEC agreement
- SCUBA agreement
- map noise / weight behavior

The main question is:

- do the bad `rtcdiag` metrics actually predict bad map behavior?

### Phase 3: Freeze an operations policy

If the map check supports it, adopt the following working policy:

- step-dominated contamination:
  - use network-window masking
- impulsive-dominated contamination:
  - use detector/event-level handling
  - do not default to whole-scan rejection
- whole-scan rejection:
  - use only when events are both rare and map-catastrophic

### Phase 4: Only then resume code changes

If further runtime work is justified after the map check, the best next code
targets are:

1. add `edge_guard` to transient nomination
2. add a detector-local `slow_jump` detector on the shared `TransientEvent`
   model
3. decide event action more explicitly:
   - interpolate
   - subtract jump height
   - mask

The main point is to stop doing blind threshold sweeps and instead change the
runtime only where the survey+map evidence says it matters.

## Product / Tooling Follow-up

One lower-priority but worthwhile improvement is to reduce the size of
`rtcdiag`.

Current state:

- `rtcdiag` files are written as dense netCDF variables
- they are not currently compressed
- they do not appear to use chunking/compression the way some full TOD products
  do

That is why `rtcdiag` is now one of the larger downloaded products even though
it is much lighter than full RTC TODs.

I would treat compression/chunking of `rtcdiag` as a product-improvement task,
not as the immediate science/debugging priority.

## Practical Handoff

If resuming from this note, the highest-value starting point is:

1. read the `redu40` survey report
2. extract a keep/mask/review table by obsnum/network
3. perform the small map-level check on:
   - `151103`
   - `150784`
   - `152526`
   - `151930`
4. only then decide whether more runtime changes are warranted

At this point, the diagnostic/reporting infrastructure is strong enough that the
next progress should come from policy and science validation, not another round
of low-level tuning first.
