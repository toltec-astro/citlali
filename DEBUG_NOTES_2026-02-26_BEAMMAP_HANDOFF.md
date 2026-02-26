# Beammap Debug Handoff Notes (2026-02-26)

## Scope and Intent
These notes capture the current beammap debugging state for the ongoing 3c273 work, with emphasis on:
- beammap fit instability / bound hitting,
- APT quality regressions,
- crash/debug instrumentation that was added,
- quantitative results from recent `redu03` runs,
- concrete next experiments.

This handoff is written so a separate Codex session (office environment) can continue from here without re-discovering context.

## Environment and Workflow Constraints
- Local laptop compile is intentionally not part of the workflow for this effort.
- Citlali build and execution for validation are done on Unity/dev environment.
- Local analysis here used output products copied under:
  - `/Users/wilson/work_toltec/local_data/beammaps/3c273/redu03`

## Code Changes in Current Tree (already integrated)

### 1) Beammap fit diagnostics and fit-QC expansion
Files:
- `include/citlali/core/utils/fitting.h`
- `include/citlali/core/engine/beammap.h`

Key additions:
- Per-fit diagnostics structure (`FitDiagnostics`) with init params, active limits, bound-hit booleans, bbox indices, sigma stats.
- Bound-hit detection for each fitted parameter (amp, x, y, a, b, angle).
- Beammap iteration log summary:
  - `beammap fit bound summary (iter N): ...`
- Final bound summary:
  - `beammap final bound-hit summary: ...`
- Extended `*_fit_qc.ecsv` with:
  - `fit_bound_*` columns,
  - init values and active FWHM bounds,
  - metadata legend for bound-code bitmask.

### 2) Beammap uncertainty path stabilization
File:
- `include/citlali/core/utils/fitting.h`

Change:
- Ceres covariance is used for pointing mode.
- Beammap mode uses a linearized `J^T J` uncertainty fallback to avoid covariance path instability and allocator issues observed during crash debugging.

### 3) Beammap fit loop safety/diagnostic behavior
File:
- `include/citlali/core/engine/beammap.h`

Change:
- Beammap fitting currently runs sequentially with checkpoint logging around `fit_to_gaussian` calls.
- This was used to localize prior crash points and produce deterministic debug traces.

### 4) CLI/build compatibility fixes
File:
- `src/citlali/cli/main.cpp`

Changes:
- Variant assignment uses `emplace<...>()` for todproc mode selection (`science`, `pointing`, `beammap`) to avoid template-assignment build errors.
- `CCfits::FitsError` catch no longer calls non-existent `what()`.

## Fruit Loops in Beammap: Actual Behavior
Relevant files:
- `src/citlali/cli/main.cpp`
- `include/citlali/core/engine/beammap.h`

Important details:
- In beammap mode, `fruit_loops_iters` is forced to `1` in CLI path.
- In beammap iterative loop (`current_iter > 0`):
  - `run_fruit_loops=false` => subtract/add fitted Gaussian in TOD.
  - `run_fruit_loops=true` => subtract/add map in TOD (`map_to_tod`).
- Convergence metric changes when fruit loops is enabled:
  - no fruit loops: parameter relative change,
  - fruit loops: map-relative change.
- For beammap type, external fruit-map loading branch used by other reduction types is compiled out by the `if constexpr (!Beammap)` branch.

Implication:
- In beammap, fruit loops acts mainly as a subtraction model toggle (Gaussian vs map) and convergence behavior toggle.

## Run Artifacts Analyzed
- Run log:
  - `/Users/wilson/work_toltec/local_data/beammaps/3c273/redu03/gw-52837160-3c273.out`
  - `/Users/wilson/work_toltec/local_data/beammaps/3c273/redu03/gw-52839784-3c273.out`
- Products:
  - `/Users/wilson/work_toltec/local_data/beammaps/3c273/redu03/151600/raw/apt_commissioning_beammap_151600_citlali.ecsv`
  - `/Users/wilson/work_toltec/local_data/beammaps/3c273/redu03/151600/raw/apt_commissioning_beammap_151600_citlali_fit_qc.ecsv`

## Configuration Progression and Outcomes (Obsnum 151600)

### Run A (fruit loops enabled): `amp=[0.1, 2.0]`, `fwhm=[0.2, 3.0]`
Source:
- `gw-52837160-3c273.out`

Final global bound summary:
- `any_hit=2706/5355`
- `amp(lo/hi)=343/1947`
- `x(lo/hi)=7/13`
- `y(lo/hi)=39/30`
- `a(lo/hi)=49/513`
- `b(lo/hi)=208/97`

Array-level status (selected):
- a1100 good (`flag==0`): `2325/3163`
- a2000 good: `843/959`

### Run B (fruit loops enabled): `amp=[0.1, 4.0]`, `fwhm=[0.2, 3.0]`
Source:
- `gw-52839784-3c273.out`

Config confirmed in log dump:
- `amp_limit_factors: [0.1, 4.0]`
- `fwhm_limit_factors: [0.2, 3.0]`

Final global bound summary:
- `any_hit=1795/5355` (down strongly)
- `amp(lo/hi)=335/647` (high-amp bound hits reduced strongly)
- `x(lo/hi)=15/17`
- `y(lo/hi)=103/115` (position bound hits increased)
- `a(lo/hi)=56/513`
- `b(lo/hi)=253/95`

Array-level status (selected):
- a1100 good: `2324/3163` (roughly unchanged)
- a2000 good: `805/959` (worse than Run A)

Interpretation:
- Increasing amp upper limit removed a major amp-ceiling artifact (`amp_hi` fell a lot).
- But this exposed/shifted pressure to position (`y`) bounds, especially in a2000.

## Current Focus Decision
- a1400 is acknowledged as a persistent "problem child" and deprioritized for immediate tuning.
- Focus is currently on improving a1100 + a2000 behavior.

## Why APT Plot Changed Less Than Expected
- `amp_limit_factors` mostly changes amplitude fit parameter freedom, not detector geometry directly.
- Top-row APT scatter plots are dominated by `x_t/y_t` distribution and flag outcomes.
- Since geometry/position issues remain, visual APT morphology can look similar despite major internal fit-bound improvements.

## a1100 Flagging Deep Dive (latest run)
From `apt_commissioning_beammap_151600_citlali.ecsv`:
- a1100 total: `3163`
- a1100 flagged: `839` (26.5%)

Important correction:
- "~1700 flagged in a1100" is not correct for this run.
- `~1655` is total flagged across all arrays.

### Dominant flag causes in a1100
Bit counts (overlapping):
- `AzFWHM`: `569`
- `ElFWHM`: `678`
- `Sig2Noise`: `122`
- `Position`: `122`
- `BadFit`: `15`

FWHM criteria dominate:
- a1100 FWHM threshold is `[3, 10]` arcsec (from beammap flagging config).
- `809/3163` violate at least one FWHM threshold.

### Network correlation (a1100)
Bad fractions by nw:
- nw0: `21.9%`
- nw1: `14.3%`
- nw2: `21.4%`
- nw3: `24.3%`
- nw4: `37.6%`
- nw5: `40.3%`

Clear concentration in nw4/nw5.

### Spatial correlation (a1100)
Flagged detectors concentrate at negative x / low y.
Examples:
- x-bin `[-150,-120)`: `99/99` flagged.
- x-bin `[-90,-60)`: `294/638` flagged.
- x-bin `[-30,60)`: ~`6-8%` flagged.
- y-bin `[-120,-60)`: ~`49-50%` flagged.
- y-bin `[0,120)`: ~`4-8%` flagged.

### Tone-frequency correlation (a1100)
There is frequency structure, especially in nw4/nw5:
- NW4:
  - ~`733.5-791.3 MHz`: `48/73` flagged (`65.8%`).
  - ~`618.0-675.8 MHz`: `37/74` flagged (`50.0%`).
- NW5:
  - ~`501.6-560.1 MHz`: `52/98` flagged (`53.1%`).
  - ~`677.1-735.5 MHz`: `21/43` flagged (`48.8%`).

This tone dependence is not independent of geometry: high-fail tone bins are still dominated by detectors in negative x / low y region.

## Lessons Learned / Gotchas
1. Always verify applied config from run log dump
- A previous run accidentally used `amp_limit_factors: [0,0]` / `fwhm_limit_factors: [0,0]` in generated config, silently reverting to defaults.
- Confirm actual values in log before interpreting fit outcomes.

2. Bound-hit diagnostics are essential
- `fit_bound_*` columns and per-iter summaries were necessary to separate amplitude ceilings from position/FWHM limits.

3. One-parameter tuning can move failure mode
- Relaxing amp upper bound helped a lot, but moved pressure toward x/y bounds for some arrays.

4. Fruit loops in beammap is not a no-op
- It changes subtraction model and convergence metric; keep this in mind when attributing effects.

## Recommended Next Experiments (priority order)

1. Increase source-fitting bbox for position freedom
- Change:
  - `post_processing.source_fitting.bounding_box_arcsec: 12` (if needed 14)
- Goal:
  - reduce x/y bound hits (especially a2000 y-bound concentration).

2. Keep current amplitude and FWHM limits while testing bbox
- Keep:
  - `amp_limit_factors: [0.1, 4.0]`
  - `fwhm_limit_factors: [0.2, 3.0]`
- Rationale:
  - isolate the position-bound effect next.

3. If a1100 still has high `amp_lo`, lower amplitude floor factor
- Candidate:
  - `amp_limit_factors: [0.03, 4.0]`
- Goal:
  - reduce lower-bound saturation in weak/biased fits without reintroducing high-amp clipping.

4. Optional controlled comparison
- Repeat one run with `fruit_loops.enabled: false` (same fitter params) to isolate fruit-loop subtraction effect on bound behavior.

## Quick Metrics to Compare Between Runs
Use these as go/no-go checks after each run:
- From log:
  - `beammap final bound-hit summary`
- From fit_qc (a1100 and a2000 only):
  - count(`fit_bound_amp > 0`)
  - count(`fit_bound_x != 0` or `fit_bound_y != 0`)
  - count(`fit_bound_a != 0` or `fit_bound_b != 0`)
  - good detector count (`flag==0`)

## Status at Handoff
- Crash-debug instrumentation and bound diagnostics are in place and working.
- Amp upper-bound saturation issue was substantially reduced by `[0.1,4.0]`.
- Next bottleneck appears to be position bounding (and known a1400 pathology), plus strong a1100 nw4/nw5 + spatial/tone correlations.
