# Citlali Compact Config Prototype - 2026-06-29

This prototype turns the config simplification plan into runnable files without
changing Citlali runtime behavior.

## What Was Added

- Profile YAML definitions under `tools/config/profiles/`.
- Compact example configs under `tools/config/examples/`.
- `tools/config/expand_compact_config.py`, which expands a compact config into
  the current full `data/config.yaml` shape.

The tool is an authoring prototype only. The C++ parser, CLI behavior, default
config, build system, and reduction flow are unchanged.

## Expansion Model

Expansion is deterministic:

1. Load `data/config.yaml` as the full-schema baseline.
2. Apply the selected profile's `full:` patch.
3. Apply compact user fields.
4. Apply `expert:` overrides verbatim.
5. Emit the expanded full YAML.

The result is intentionally compatible with the current full YAML shape so it
can later be compared against hand-authored configs on Unity.

## Supported Compact Surface

The prototype supports these user-facing fields:

- `mode`
- `profile`
- `inputs.legacy`, `inputs.full`, `inputs.file`, or `inputs.manifest`
- `output.dir`, `output.subdir`, `output.verbose`
- `runtime.threads`, `runtime.parallel`
- `map.unit`, `map.method`, `map.grouping`, `map.pixel_axes`,
  `map.pixel_size_arcsec`, `map.center`, `map.size`, `map.coadd`
- `products.maps`, `products.noise`, `products.noise_count`,
  `products.noise_products`, `products.noise_realizations`, `products.tod`,
  `products.diagnostics`
- `processing.tod`, `processing.clean`, `processing.clean_grouping`,
  `processing.weighting`, `processing.fruitloops`,
  `processing.fruitloops_iters`
- `pointing.source_strategy`, `pointing.source_protection_radius_arcsec`,
  `pointing.fit_radius_arcsec`, `pointing.fit_box_arcsec`
- `beammap.iterations`, `beammap.convergence_tolerance`,
  `beammap.convergence_radius_arcsec`, `beammap.derotate`,
  `beammap.subtract_reference_det`, `beammap.reference_det`,
  `beammap.detector_weighting`, `beammap.detector_tod`, `beammap.priors`
- `expert`, a verbatim full-schema patch applied last

Unknown compact keys currently produce warnings rather than hard failures,
except inside `inputs`, where unsupported forms are rejected.

## Profiles

Initial profiles:

- `science_standard`
- `science_diagnostic`
- `pointing_standard`
- `beammap_detector`
- `tod_export`

These profile defaults are starting points for review, not final policy. They
should be calibrated against real configs and `gw_dev` outputs before any CLI
accepts compact configs directly.

## Usage

List profiles:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py --list-profiles
```

Expand an example:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/science_standard_compact.yaml \
  --output /tmp/science_standard.expanded.yaml \
  --summary-out /tmp/science_standard.summary.yaml
```

The examples use `inputs.legacy: ../../../data/config.yaml` only so the tool can
run locally. Real compact configs should point at an observation-specific
legacy `inputs:` block or include `inputs.full` directly.

## Recommended Next Step

Collect representative full configs for science, pointing, beammap, and
TOD-output use cases. For each one, write a compact equivalent, expand it, and
diff the expanded YAML against the existing full config. Any diff should be
classified as:

- expected profile default,
- compact user override,
- missing compact field,
- expert-only override, or
- bug in the prototype mapper.

Only after those comparisons should this become a runtime parser path.
