# Citlali/TolTECA Reduction Intents

Date: 2026-06-30

This note describes the compact authoring model for normal TolTECA-driven
Citlali reductions. It is a prototype layer only: the generated low-level YAML
still targets the current Citlali schema.

## Intent Model

Normal users should choose one of four reduction intents:

| Intent | Compact mode | Legacy Citlali reduction type | Purpose |
| --- | --- | --- | --- |
| Pointing | `pointing` | `pointing` | Compact-source pointing/focus reductions |
| Out-of-focus holography | `oof` | `pointing` | PSF-preserving OOF/holography reductions |
| Beammap | `beammap` | `beammap` | Detector beam characterization |
| Science | `science` | `science` | Normal science maps and diagnostics |

`oof` is intentionally first-class in the compact layer even though it still
expands to `runtime.reduction_type: pointing`. This keeps the user-facing
language aligned with observing/reduction practice while avoiding a behavior
change in Citlali before OOF-specific validation exists.

## TolTECA Layering

The compact layer should respect TolTECA's existing merge model:

```text
40_setup.yaml              TolTECA-managed setup state
60_citlali_profile.yaml    generated or shipped mode/profile defaults
70_reduce.yaml             normal user-facing choices as targeted overrides
72_target.yaml             target/input/APT selection overrides
80_expert.yaml             optional raw low-level escape hatch
```

TolTECA reads all `NN*.yaml` files in leading-number order, with higher
numbered files overriding lower numbered files. Citlali should continue to
receive a generated legacy-compatible `citlali_*.yaml` file.

## Prototype Tooling

`tools/config/expand_compact_config.py` now supports the four intent modes and
three output formats:

```bash
# Standalone legacy Citlali-style tree.
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/oof_standard_compact.yaml \
  --output /tmp/oof_full.yaml

# Bare low_level block, suitable for embedding by another tool.
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/oof_standard_compact.yaml \
  --output-format low_level \
  --output /tmp/oof_low_level.yaml

# TolTECA NN*.yaml wrapper.
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/oof_standard_compact.yaml \
  --output-format tolteca \
  --output /tmp/60_citlali_profile.yaml
```

The `tolteca` output format writes:

```yaml
reduce:
  steps:
    0:
      config:
        low_level:
          ...
```

The indexed `steps: 0:` form matches the overlay style used by higher-numbered
TolTECA files such as `72_reduce.yaml`, so later files can patch the same step
instead of replacing a list wholesale.

It deliberately omits top-level `inputs`, because TolTECA owns input discovery
and writes the generated `inputs` block in the final `citlali_*.yaml`.

## Current Profiles

| Profile | Mode | Notes |
| --- | --- | --- |
| `pointing_standard` | `pointing` | Compact-source pointing with source-aware protection |
| `pointing_compat_passthrough` | `pointing` | No-op validation profile for existing TolTECA point baselines |
| `oof_standard` | `oof` | PSF-preserving OOF defaults mapped through the pointing engine |
| `oof_compat_passthrough` | `oof` | No-op validation profile for existing TolTECA OOF baselines |
| `beammap_detector` | `beammap` | Detector-grouped beammap defaults |
| `beammap_compat_passthrough` | `beammap` | No-op validation profile for existing TolTECA beammap baselines |
| `science_standard` | `science` | Normal science map production |
| `science_diagnostic` | `science` | Science maps with verbose diagnostics and TOD products |
| `tod_export` | `science` | TOD export/debugging profile |
| `science_compat_passthrough` | `science` | No-op validation profile for existing TolTECA science baselines |

## User-Facing Knob Groups

The compact layer should keep normal authoring focused on these groups:

| Group | Example compact keys |
| --- | --- |
| Runtime | `runtime.threads`, `runtime.parallel` |
| Output | `output.dir`, `output.subdir`, `output.verbose` |
| Map | `map.method`, `map.grouping`, `map.pixel_axes`, `map.pixel_size_arcsec`, `map.center`, `map.size`, `map.coadd` |
| Products | `products.maps`, `products.noise`, `products.tod`, `products.diagnostics` |
| Processing | `processing.clean`, `processing.clean_grouping`, `processing.weighting`, `processing.fruitloops` |
| Pointing | `pointing.source_strategy`, `pointing.source_protection_radius_arcsec`, `pointing.fit_radius_arcsec` |
| OOF | `oof.source_strategy`, `oof.fit_gaussian`, `oof.fruitloops_center_mode`, `oof.center_keep_radius_arcsec` |
| Beammap | `beammap.iterations`, `beammap.convergence_tolerance`, `beammap.detector_weighting`, `beammap.priors` |
| Expert | Raw legacy low-level overrides under `expert` |

Everything else remains profile-managed unless a validation case shows it needs
to become a common advanced user knob.

## Validation Strategy

Before changing TolTECA or Citlali runtime behavior, validate the prototype by
expanding a compact profile to a `low_level` block and comparing it to the
current `70_reduce.yaml` low-level block for the same reduction intent.

For the protected point 152389 case, the existing boundary inventory showed:

- generated Citlali leaves: 489
- copied from `70_reduce.yaml` low-level: 443
- TolTECA-generated `inputs`: 45
- TolTECA-rewritten low-level paths: 1 (`runtime.output_dir`)

The first behavior-preserving target is therefore not a Unity compile. It is a
YAML equivalence check: compact pointing should expand to the same low-level
tree as the current point `70_reduce.yaml`, excluding TolTECA-owned
`runtime.output_dir` handling.

The first run of that comparison is recorded in
`doc/POINTING_COMPACT_EQUIVALENCE_2026-06-30.md`. It found 178 low-level
differences between the prototype `pointing_standard` profile and the current
point `70_reduce.yaml`, mostly in `timestream` and inactive `beammap` defaults.
The no-op `pointing_compat_passthrough` profile now expands against that same
`70_reduce.yaml` baseline with zero differences. That is the safe starting
point for moving point configuration groups behind compact names before asking
Unity reductions to compare science products.

The same no-op compatibility harness now exists for all four user-facing
intents. Representative local checks passed with zero low-level YAML
differences after ignoring `runtime.output_dir`:

| Intent | Baseline | Leaf keys |
| --- | --- | ---: |
| Pointing | `/Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/70_reduce.yaml` | 444 |
| OOF | `/Users/gwilson/work_toltec/local_data/OOF/149056/70_reduce.yaml` | 164 |
| Beammap | `/Users/gwilson/work_toltec/local_data/beammaps/3c273/70_reduce.yaml` | 485 |
| Science | `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/70_reduce.yaml` | 219 |

The stronger compact-key fixtures below also passed the same zero-difference
check. These files use compact runtime, map, product, processing, and
intent-specific fields where the selected baseline already has matching
low-level destinations:

| Intent | Compact-key fixture |
| --- | --- |
| Pointing | `tools/config/examples/pointing_compat_point70_compact.yaml` |
| OOF | `tools/config/examples/oof_compat_149056_compact.yaml` |
| Beammap | `tools/config/examples/beammap_compat_3c273_compact.yaml` |
| Science | `tools/config/examples/science_compat_goodsn_compact.yaml` |
