# Fruitloops Loop-Gain Isolation (Next Session)

## What Was Added

Temporary debug controls were added under `timestream.fruit_loops`:

- `interp_mode_override`: `auto | nearest | bilinear | jinc`
- `legacy_center`: `true | false`

Behavior:

- `interp_mode_override: auto` keeps the current default:
  - `jinc` when `mapmaking.method: jinc`
  - `bilinear` otherwise
- `legacy_center: true` uses the old map-center convention (`n/2`) for map->TOD projection.
- `legacy_center: false` uses the current convention (`(n-1)/2`).

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
    interp_mode_override: auto   # auto|nearest|bilinear|jinc
    legacy_center: false         # true => n/2 center, false => (n-1)/2
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
