# Fruitloops Loop-Gain Isolation (Next Session)

## What Was Added

Temporary debug controls were added under `timestream.fruit_loops`:

- `interp_mode_override`: `auto | nearest | bilinear | jinc | trunc`
- `legacy_center`: `true | false`
- `recompute_weights_after_addback`: `true | false`

Behavior:

- `interp_mode_override: auto` keeps the current default:
  - `jinc` when `mapmaking.method: jinc`
  - `bilinear` otherwise
- `legacy_center: true` uses the old map-center convention (`n/2`) for map->TOD projection.
- `legacy_center: false` uses the current convention (`(n-1)/2`).
- `recompute_weights_after_addback: true` restores pre-`b7c1a5ef` weight handling by recalculating
  weights after map add-back (instead of keeping source-subtracted weights).

## Recommended Obsnum For Isolation

Use **obsnum `152523`** first.

Reason: strongest monotonic fractional rise across saved FL iterations (`redu00..redu04`), while still looking physically usable for controlled tests.

Observed amplitude growth for obsnum `152523`:

- `a1100 (array 0)`: `2475 -> 4802 -> 7194 -> 10407 -> 15555` (ratio `6.285`)
- `a1400 (array 1)`: `1636 -> 3394 -> 5252 -> 7529 -> 10953` (ratio `6.695`)
- `a2000 (array 2)`: `2104 -> ... -> 9062` (ratio `4.308`)

## Test Matrix (Run Tomorrow)

Keep data/config fixed and run with `fruit_loops.max_iters: 3` for quick comparisons.

1. `interp_mode_override: auto`, `legacy_center: false` (current baseline)
2. `interp_mode_override: nearest`, `legacy_center: false` (interpolation effect)
3. `interp_mode_override: auto`, `legacy_center: true` (centering effect)
4. `interp_mode_override: nearest`, `legacy_center: true` (legacy proxy)

## What To Compare

For each run, track per-array gain:

- `g1 = A(iter1) / A(iter0)`
- `g2 = A(iter2) / A(iter1)`

Interpretation:

- If switching to `nearest` significantly reduces `g1/g2`, interpolation change is a major contributor.
- If `legacy_center: true` significantly reduces `g1/g2`, center-convention change is a major contributor.
- If both are needed to recover prior behavior, the effect is coupled.

## YAML Snippet

```yaml
timestream:
  fruit_loops:
    interp_mode_override: auto   # auto|nearest|bilinear|jinc|trunc
    legacy_center: false         # true => n/2 center, false => (n-1)/2
    recompute_weights_after_addback: true  # true => pre-b7c1a5ef behavior
```

## Direct `tolteca reduce` Commands (70 / 72)

`tolteca` CLI behavior used here:

- Extra CLI args after `reduce` are parsed as dotted config keys.
- Those keys are applied as overrides under `reduce.*`.
- Source: `tolteca/cli/reduce.py`, `tolteca/cli/utils.py`, `tolteca/utils/__init__.py::dict_from_cli_args`.

Assume runtime context dir is:

```bash
cd ~/work_toltec/local_data/2025-C1-COM-04/pointings
```

The commands below force:

- focus obsnum: `152523`
- `max_iters: 3`
- `save_all_iters: true`
- one of the four `(interp_mode_override, legacy_center)` combinations

### 70_reduce.yaml matrix

```bash
tolteca -d . -c 70_reduce.yaml reduce \
  --jobkey reduced_flgain_70_auto_newcenter \
  --inputs.0.select 'scannum==2 & (obsnum==152523)' \
  --steps.0.config.low_level.timestream.fruit_loops.max_iters 3 \
  --steps.0.config.low_level.timestream.fruit_loops.save_all_iters true \
  --steps.0.config.low_level.timestream.fruit_loops.interp_mode_override auto \
  --steps.0.config.low_level.timestream.fruit_loops.legacy_center false

tolteca -d . -c 70_reduce.yaml reduce \
  --jobkey reduced_flgain_70_nearest_newcenter \
  --inputs.0.select 'scannum==2 & (obsnum==152523)' \
  --steps.0.config.low_level.timestream.fruit_loops.max_iters 3 \
  --steps.0.config.low_level.timestream.fruit_loops.save_all_iters true \
  --steps.0.config.low_level.timestream.fruit_loops.interp_mode_override nearest \
  --steps.0.config.low_level.timestream.fruit_loops.legacy_center false

tolteca -d . -c 70_reduce.yaml reduce \
  --jobkey reduced_flgain_70_auto_legacycenter \
  --inputs.0.select 'scannum==2 & (obsnum==152523)' \
  --steps.0.config.low_level.timestream.fruit_loops.max_iters 3 \
  --steps.0.config.low_level.timestream.fruit_loops.save_all_iters true \
  --steps.0.config.low_level.timestream.fruit_loops.interp_mode_override auto \
  --steps.0.config.low_level.timestream.fruit_loops.legacy_center true

tolteca -d . -c 70_reduce.yaml reduce \
  --jobkey reduced_flgain_70_nearest_legacycenter \
  --inputs.0.select 'scannum==2 & (obsnum==152523)' \
  --steps.0.config.low_level.timestream.fruit_loops.max_iters 3 \
  --steps.0.config.low_level.timestream.fruit_loops.save_all_iters true \
  --steps.0.config.low_level.timestream.fruit_loops.interp_mode_override nearest \
  --steps.0.config.low_level.timestream.fruit_loops.legacy_center true
```

### 72_reduce.yaml matrix

```bash
tolteca -d . -c 72_reduce.yaml reduce \
  --jobkey reduced_flgain_72_auto_newcenter \
  --inputs.0.select 'scannum==2 & (obsnum==152523)' \
  --steps.0.config.low_level.timestream.fruit_loops.max_iters 3 \
  --steps.0.config.low_level.timestream.fruit_loops.save_all_iters true \
  --steps.0.config.low_level.timestream.fruit_loops.interp_mode_override auto \
  --steps.0.config.low_level.timestream.fruit_loops.legacy_center false

tolteca -d . -c 72_reduce.yaml reduce \
  --jobkey reduced_flgain_72_nearest_newcenter \
  --inputs.0.select 'scannum==2 & (obsnum==152523)' \
  --steps.0.config.low_level.timestream.fruit_loops.max_iters 3 \
  --steps.0.config.low_level.timestream.fruit_loops.save_all_iters true \
  --steps.0.config.low_level.timestream.fruit_loops.interp_mode_override nearest \
  --steps.0.config.low_level.timestream.fruit_loops.legacy_center false

tolteca -d . -c 72_reduce.yaml reduce \
  --jobkey reduced_flgain_72_auto_legacycenter \
  --inputs.0.select 'scannum==2 & (obsnum==152523)' \
  --steps.0.config.low_level.timestream.fruit_loops.max_iters 3 \
  --steps.0.config.low_level.timestream.fruit_loops.save_all_iters true \
  --steps.0.config.low_level.timestream.fruit_loops.interp_mode_override auto \
  --steps.0.config.low_level.timestream.fruit_loops.legacy_center true

tolteca -d . -c 72_reduce.yaml reduce \
  --jobkey reduced_flgain_72_nearest_legacycenter \
  --inputs.0.select 'scannum==2 & (obsnum==152523)' \
  --steps.0.config.low_level.timestream.fruit_loops.max_iters 3 \
  --steps.0.config.low_level.timestream.fruit_loops.save_all_iters true \
  --steps.0.config.low_level.timestream.fruit_loops.interp_mode_override nearest \
  --steps.0.config.low_level.timestream.fruit_loops.legacy_center true
```

## v4.x vs gw_dev Detailed Fruitloops Comparison (Code-Level)

Scope compared:

- `v4.x` branch tip vs `gw_dev` branch tip.
- Fruitloops data path: map load -> map->TOD projection -> weight handling around subtract/addback.

### High-impact behavior changes

1) Projection kernel changed (largest likely gain driver)

- `v4.x` map->TOD projection used direct integer pixel lookup:
  - `ir = irows(j)`, `ic = icols(j)` (implicit truncation toward zero)
  - center offset fixed at `n/2`.
- `gw_dev` supports `jinc | bilinear | nearest` projection:
  - default is tied to mapmaker method (`jinc` mapmaker => `jinc` feedback)
  - center default is `(n-1)/2` unless `legacy_center: true`
  - `nearest` uses `llround` (not old truncation).

Implication:

- Even with `nearest + legacy_center`, current behavior is still not identical to old truncating sampler.
- This can change effective loop gain enough to flip from convergence to divergence.

2) Post-addback weight behavior changed

- `v4.x`: recompute weights after add-back.
- `gw_dev`: default changed to keep source-subtracted weights.
- New knob restores old behavior:
  - `recompute_weights_after_addback: true`.

3) S/N gate dependency is explicit and currently inactive in these tests

- With `noise_maps.enabled: false`, fruitloops cannot load `MEDRMS` noise maps.
- Then `sig2noise_limit` is effectively ignored and gating becomes flux-only.

Implication:

- `sig2noise_limit: 100` is not constraining anything in this mode.
- Convergence/disconvergence is dominated by flux clipping and projection semantics.

4) Additional hardening changes (lower probability as primary cause)

- stricter map/WCS validation in `load_mb`
- map-index parsing by grouping
- sample-level flag check in map->TOD loop
- optional center keep mask for coverage-cut masking
- source-location-aware masking in full-weight calculation when `mask_radius_arcsec > 0`.

### Why array_flux_limit changes did not help much

Observed in current 152523 tests:

- Raising `[50,50,20] -> [200,200,80]` changed FL trajectories only weakly (sub-percent to ~1% level by iter4).
- This indicates clip threshold is not the dominant instability control in current gw_dev behavior.

### Most likely remaining mismatch to v4.x behavior

Projection operator mismatch:

- old: truncating pixel pick at `n/2` center
- new tested: nearest/rounded or interpolated samplers

This is the strongest remaining un-matched semantic difference after enabling:

- `legacy_center: true`
- `recompute_weights_after_addback: true`

`trunc` mode has now been added in gw_dev to restore old cast-to-index sampling.

### Practical next comparison target

Add an explicit old-style projection mode in gw_dev:

- `fruit_loops.interp_mode_override: trunc` (or `legacy_nearest`)
- implementation: use old cast/trunc indexing path exactly.

Then rerun 152523 with old production fruitloops parameters:

- `mode: upper`
- `array_flux_limit: [12,18,10]`
- `sig2noise_limit: 100`
- `max_iters: 5`

If this converges, root cause is projector semantics; if not, remaining differences are in modern cleaning/weighting internals.
