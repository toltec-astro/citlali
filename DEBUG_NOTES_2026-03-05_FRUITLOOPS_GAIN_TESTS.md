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
