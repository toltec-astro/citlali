# Jinc Memo Alignment Notes (2026-03-09)

## Scope and Intent
This note records the cleanup that redefined `mapmaking.method: jinc` to match the memo-style OTF gridding interpretation throughout the main scalar mapmaking path.

The goal of this change was to remove the old mixed semantics where the same jinc kernel family was being used with a matched-amplitude style normalization in mapmaking while fruit-loops projection was using a kernel-weighted interpolation.

This note is also the handoff point before any further parameter tuning of:
- `r_max`
- `a`
- `b`
- `c`

## Branch / Code State
- Branch: `gw_dev`
- Main code change commit:
  - `de0f0a23 unified jinc weighting`
- Local compile status:
  - compile confirmed after landing this cleanup

## Contract Going Forward
`mapmaking.method: jinc` now means:
- memo-style jinc gridding in the forward mapmaker
- memo-style kernel-weighted jinc interpolation in the map-to-TOD projection path

It does **not** imply:
- Wiener filtering
- matched-filter detection products
- any automatic source-detection strategy

Those remain optional downstream choices under user control.

## Why The Change Was Needed
Before this cleanup, the jinc mapmaker was not using the memo estimator directly.

The practical difference was:
- signal was accumulated with `K`
- the normalization map was accumulated with `K^2`
- final map normalization used that same `K^2`-weighted map

That made the map closer to a kernel-amplitude estimate than to a direct kernel-weighted sky average.

At the same time, the jinc map sampler used by fruit loops already behaved like:
- `sum(K * map) / sum(K)`

So the forward and backward operators did not represent the same interpretation of `jinc`.

## What Changed In The Code

### 1) Separate memo gridding denominator from inverse-variance bookkeeping
File:
- `include/citlali/core/mapmaking/map.h`

Change:
- added `grid_weight` to `MapBuffer`

Purpose:
- store the memo-style gridding denominator `D = sum(wK)` separately from the variance accumulator

### 2) Normalize jinc maps with the memo denominator
File:
- `src/citlali/core/mapmaking/map.cpp`

Change:
- `normalize_maps()` now detects when `grid_weight` is present
- signal, kernel, and noise maps are divided by `grid_weight`
- the stored jinc `weight` map is then converted to final inverse variance using:
  - `weight_final = D^2 / Q`
  - where `D = sum(wK)`
  - and `Q = sum(wK^2)`

Purpose:
- keep the science map estimator memo-style
- still preserve quadratic variance propagation for the final weight map

### 3) Rework jinc accumulation semantics
File:
- `include/citlali/core/mapmaking/jinc_mm.h`

Change:
- signal accumulation stays proportional to `K`
- new `grid_weight` accumulation uses `wK`
- `weight` accumulation uses `wK^2`
- noise-map accumulation now uses `K`, not `K^2`

Purpose:
- make the deposited map follow the memo’s kernel-weighted average interpretation
- reserve `K^2` for variance propagation and coverage-like bookkeeping

### 4) Allocate and clear the new denominator map where needed
Files:
- `include/citlali/core/engine/todproc.h`
- `include/citlali/core/engine/beammap.h`
- `include/citlali/core/engine/engine.h`

Changes:
- allocate `grid_weight` only when `map_method == "jinc"`
- clear it when map buffers are reset
- include it in rough memory reporting

Purpose:
- prevent stale denominators between passes
- keep the additional storage explicit and limited to jinc mode

### 5) Keep fruit-loops jinc projection aligned with the memo interpretation
File:
- `include/citlali/core/timestream/timestream.h`

Change:
- documented the jinc sampler as a memo-style kernel-weighted average

Purpose:
- make the intended forward/backward operator pairing explicit

## Practical Result
For the scalar jinc path, the pipeline now separates three different roles that were previously conflated:

- map estimate:
  - memo-style kernel-weighted gridding
- final inverse-variance weight:
  - derived from quadratic propagation
- optional detection filtering:
  - left to Wiener or other post-processing choices

This is the cleaner model for TolTEC map products because it keeps the base jinc map interpretable as the gridded sky estimate, rather than a matched-amplitude product.

## What This Change Does Not Yet Do
- It does not choose optimized values for `r_max`, `a`, `b`, or `c`.
- It does not redefine Wiener filtering; that remains a separate post-processing choice.
- It does not attempt a broader redesign of the polarization solver beyond making the shared map-buffer bookkeeping consistent.

## Recommended Next Step
Now that `jinc` has a single intended meaning again, the next work should be parameter optimization with TolTEC-specific simulations, rather than further semantic cleanup.

The first tuning pass should focus on:
- point-source recovery bias and S/N
- beam broadening
- extended-emission transfer
- edge and coverage-gradient behavior
- band-dependent choices for `r_max`, `a`, `b`, and `c`
