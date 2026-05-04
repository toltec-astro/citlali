# Fruit Loops And Weighting Audit Notes (2026-04-22)

## Scope
This note records concerns from the April 2026 audit about the interaction between:

- fruit-loops map subtraction / add-back
- timestream cleaning
- detector weighting
- optional source masking

This note is aimed at the current `mJy/beam` flux-loss investigation.

It is intentionally separate from:

- non-`mJy/beam` unit-conversion issues
- filtered-map edge-guard behavior

Those are real but orthogonal.

## Executive Summary
I do **not** currently see a simple raw-map calibration bug in the `mJy/beam` path.

The strongest concern is instead a **transfer-function / weighting semantic issue**:

1. fruit-loops subtracts a source model from the TOD
2. cleaning runs on that source-subtracted TOD
3. weights are computed on the source-subtracted residual
4. by default, those weights are then **kept** after the map is added back

Code path:

- `include/citlali/core/engine/lali.h`
- `data/config.yaml`

That means the final map can be built from:

- signal timestream with source added back
- weights derived from a different signal state

This is not automatically wrong, but it is a major semantic choice and a strong candidate for gain / apparent flux-loss differences across fruit-loops configurations.

The repo already contains evidence pointing in the same direction:

- `DEBUG_NOTES_2026-03-05_FRUITLOOPS_GAIN_TESTS.md`

which identifies:

- projection semantics
- post-addback weight behavior

as the dominant observed fruit-loops gain drivers.

## Current Fruit-Loops Weight Path

### Data flow in the science reduction path
Current logic in:

- `include/citlali/core/engine/lali.h`

is:

1. if fruit-loops map exists, subtract map from TOD
2. run cleaning on source-subtracted TOD
3. compute weights on source-subtracted TOD
4. optionally populate noise maps from that source-subtracted state
5. add source map back to TOD
6. either:
   - keep the source-subtracted weights, or
   - recompute weights after add-back

Relevant lines:

- subtraction: `lali.h:285-290`
- noise-only weight pass: `lali.h:298-305`
- add-back: `lali.h:320-324`
- keep weights vs recompute: `lali.h:330-345`

The default config is:

- `timestream.fruit_loops.recompute_weights_after_addback: false`

from:

- `data/config.yaml:377`

So current default behavior is:

- **keep source-subtracted weights**

## Primary Concerns

### 1) Default weight state is tied to the source-subtracted TOD, not the final TOD
This is the main concern.

If the source subtraction is imperfect, then the residual source power in the noise-only pass depends on:

- map amplitude
- map morphology
- map centering
- map-to-TOD interpolation mode
- coverage masking of the loaded fruit-loops map

Then those weights are reused for the final source-added-back timestream unless recomputation is explicitly enabled.

Practical implication:

- the final map transfer function is no longer determined only by the cleaned signal path
- it also depends on how much source power was removed before the weight pass

This can look like:

- flux loss
- flux gain drift across fruit-loops iterations
- different normalization between otherwise similar runs

depending on whether the subtraction residuals suppress or inflate the detector variance estimate.

### 2) Projection semantics feed directly into weighting through subtraction residuals
The source-subtracted weight pass is only as good as the map-to-TOD projection operator used to subtract the source.

The current projection path can use:

- `jinc`
- `bilinear`
- `nearest`
- `trunc`

with center convention controlled by:

- `legacy_center`

from:

- `data/config.yaml:375-376`

The March fruit-loops note already identifies this as the largest likely gain driver:

- `DEBUG_NOTES_2026-03-05_FRUITLOOPS_GAIN_TESTS.md:167-226`

The important coupling is:

- projection mismatch -> subtraction residuals -> different detector variances -> different weights

So even if the mapmaker is unchanged, changing projection semantics can move the weights and therefore move flux in the final map.

### 3) Full weighting is source-aware only when `mask_radius_arcsec > 0`
There are two separate places where source masking matters:

- mean subtraction before cleaning
- full-weight variance estimation

Mean subtraction:

- `include/citlali/core/timestream/ptc/ptcproc.h:1093-1100`

If `mask_radius_arcsec == 0`, per-detector means are computed with the source included.

Full-weight source masking:

- `include/citlali/core/timestream/ptc/ptcproc.h:2223-2274`

Again, this only activates when:

- `mask_radius_arcsec > 0`
- a fruit-loops map is loaded
- source-location metadata exists

Default config is:

- `mask_radius_arcsec: 0.0`

from:

- `data/config.yaml:301`

So default behavior is:

- source participates in mean subtraction
- source participates in full-weight variance estimation

unless fruit-loops happens to remove it well enough first.

That makes the weight transfer function sensitive to subtraction quality.

### 4) The source mask used by full weighting is anchored to one positive peak per map
When a fruit-loops map is loaded, source coordinates for the full-weight mask are derived from the maximum positive pixel in each loaded map:

- `include/citlali/core/timestream/timestream.h:1044-1070`

This creates one `(lat, lon)` per map, then `calc_weights()` masks a circular region around that location:

- `include/citlali/core/timestream/ptc/ptcproc.h:2246-2266`

Concerns:

- single-source assumption
- positive-peak assumption
- not robust for extended sources
- not robust for multi-source fields
- not robust for strong negative bowls / negative modes
- if the peak location drifts between fruit-loops iterations, the weight mask also moves

So even when source masking is enabled, it is only masking one peak-centered circular region, not the full source morphology.

### 5) `sig2noise_limit` can silently become inactive when noise maps are absent
Fruit-loops gating in `map_to_tod()` tests:

- S/N gate from `median_rms`
- flux gate from `array_flux_limit`

Code:

- `include/citlali/core/timestream/timestream.h:1465-1527`

Important detail:

- if `run_noise` is false, or no usable `median_rms` is available, `have_rms` stays false
- then `run_pix_s2n` stays false
- gating becomes flux-only

This matches the existing March note:

- `DEBUG_NOTES_2026-03-05_FRUITLOOPS_GAIN_TESTS.md:189-197`

Implication:

- users may think `sig2noise_limit` is constraining subtraction
- but in some modes it is not constraining anything

That can produce unexpectedly aggressive subtraction and therefore unexpectedly aggressive weight changes.

### 6) Weight clipping / resetting can hide the actual amount of weight motion
After every weight calculation, the code can reset or clip detector weights relative to a group median:

- `include/citlali/core/timestream/ptc/ptcproc.h:2864-2960`

Config controls:

- `median_map_weight_factor`
- `lower_map_weight_factor`
- `upper_map_weight_factor`

from:

- `data/config.yaml:304-308`

This is not necessarily wrong, but it means:

- the raw variance response to fruit-loops subtraction is not always what reaches mapmaking
- some changes are compressed back toward the group median
- some detectors can be zeroed/flagged if thresholds are active

So if flux shifts are seen only after fruit-loops and weighting changes, part of the effect may be:

- subtraction changed variance estimate
- reset/clip logic changed which detectors dominated mapmaking

rather than a direct gain calibration bug.

### 7) Approximate weighting has an additional flux-scale consistency risk
This is lower priority because default weighting is `full`, but it is still worth recording.

Approximate weights use:

- `apt["sens"]`
- `in.fcf`

in:

- `include/citlali/core/timestream/ptc/ptcproc.h:2189-2214`

Observation-level `flxscale_correction` applies only to:

- `apt["flxscale"]`

in:

- `src/citlali/cli/main.cpp:352-376`

and does **not** update:

- `apt["sens"]`

By contrast, beammap calibration updates both together:

- `include/citlali/core/engine/beammap.h:2941-2945`

So under `weighting.type: approximate`, observation-level flux-scale changes can alter signal amplitude without keeping the approximate noise model fully consistent.

This does not explain a generic `mJy/beam` problem under the default `full` weighting, but it is a real concern if approximate weighting is ever used in fruit-loops comparisons.

## Secondary Concerns

### Cleaning and weighting can be driven by slightly different source treatments
`subtract_mean()` can use a mask if `mask_radius_arcsec > 0`, but otherwise source contributes to mean subtraction:

- `ptcproc.h:1058-1100`

Then cleaning groups may use masked flags if masking is enabled:

- `ptcproc.h:1198-1204`
- `ptcproc.h:1354-1362`

Then full weights may or may not use the fruit-loops source mask:

- `ptcproc.h:2223-2274`

So the source treatment is not a single global switch. It is a composition of:

- subtraction quality
- mean-subtraction masking
- cleaning masking
- full-weight masking

That makes it easy for subtle configuration changes to move the effective transfer function.

### Noise maps in the fruit-loops pass are only populated from the source-subtracted state
If `run_mapmaking && run_noise`, the code explicitly populates noise maps from the source-subtracted pass before add-back:

- `lali.h:307-319`

That means any downstream use of those noise maps is tied to the noise-only state, not the final signal-added-back state.

This is probably intentional, but it reinforces the general theme:

- fruit-loops creates a split between the state used for noise/weights and the state used for final signal maps

## What I Think Is Most Likely

### Most likely major driver
For fruit-loops-related gain or apparent flux loss, the most likely high-impact driver is:

- **post-addback weight semantics**

specifically:

- keeping source-subtracted weights by default

combined with:

- projection/subtraction mismatch

This is the first thing I would try to falsify.

### Next most likely driver
If the field is not strictly a single compact positive source, then:

- the current source-mask construction is probably too crude

because it only masks one peak-centered circular region.

### Lower-probability but real contributor
If someone is testing with:

- `weighting.type: approximate`

then:

- `flxscale_correction` / `sens` inconsistency

becomes another plausible source of inter-run weighting differences.

## Recommended Validation Matrix
If the goal is to isolate whether fruit-loops weighting is responsible for apparent flux loss, I would compare the same observation with:

1. `fruit_loops.enabled: false`
2. `fruit_loops.enabled: true`, `recompute_weights_after_addback: false`
3. `fruit_loops.enabled: true`, `recompute_weights_after_addback: true`
4. repeat 2 and 3 with:
   - `interp_mode_override: trunc`
   - `legacy_center: true`
5. repeat 2 and 3 with:
   - `mask_radius_arcsec: 0`
   - then a nonzero source mask

The most informative outputs to compare are:

- per-scan detector-weight distributions before mapmaking
- number of detectors clipped/reset by `reset_weights()`
- raw obs-map source amplitude
- raw coadd source amplitude
- fruit-loops iteration-to-iteration amplitude change

## Recommended Logging / Diagnostics To Add Later
If this becomes the main line of investigation, the most useful diagnostics to add would be:

1. per scan, save summary stats of weights:
   - before fruit-loops subtraction
   - after source-subtracted weight pass
   - after add-back recomputation when enabled
2. save the fraction of detector samples masked by the fruit-loops source mask during full weighting
3. log the chosen fruit-loops source coordinates per map and whether the source mask was active
4. log the number of pixels passing the fruit-loops S/N gate vs flux gate
5. log weight changes by array/network after each fruit-loops iteration

## Bottom Line
My main concern is not that fruit-loops contains one obvious arithmetic bug in `mJy/beam`.

My main concern is that fruit-loops currently changes the **state on which weights are estimated**, and that state is highly sensitive to:

- map-to-TOD projection semantics
- source masking choices
- whether weights are frozen before or after add-back

That is exactly the kind of behavior that can produce apparent flux loss or gain drift without any explicit calibration mistake in the raw `mJy/beam` path.
