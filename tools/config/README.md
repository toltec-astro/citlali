# Config Refactor Tools

This directory contains lightweight tools for auditing Citlali's YAML config
surface during the structural refactor.

`config_inventory.py` reads `data/config.yaml`, counts leaf keys, and scans the
source tree for simple `std::tuple{...}` config-key references. It is a static
aid only; it does not validate that every dynamic config access was found.

Example:

```bash
$HOME/tolteca/bin/python tools/config/config_inventory.py \
  --config data/config.yaml \
  --source-root . \
  --markdown-out /tmp/citlali_config_inventory.md \
  --json-out /tmp/citlali_config_inventory.json
```

The output is useful for deciding which keys belong in normal user profiles,
which keys should remain expert-only overrides, and which keys are candidates
for deprecation after compatibility validation.

## TolTECA Low-Level Boundary Inventory

`tolteca_lowlevel_inventory.py` compares TolTECA authoring YAML files with the
generated `citlali_*.yaml` low-level file that Citlali actually ingests.
TolTECA reads all `NN*.yaml` files in a reduction directory and lets higher
numbered files override lower numbered files, so the preferred invocation is a
directory scan:

```bash
$HOME/tolteca/bin/python tools/config/tolteca_lowlevel_inventory.py \
  --authoring-dir /path/to/reduction_dir \
  --generated-file /path/to/reduced/redu08/citlali_o152389_0_2_c1.yaml \
  --markdown-out /tmp/tolteca_lowlevel_inventory.md \
  --csv-out /tmp/tolteca_lowlevel_inventory.csv \
  --json-out /tmp/tolteca_lowlevel_inventory.json
```

Use repeated `--authoring-file` arguments only when comparing a specific subset
of files by hand. The tool flattens `reduce.steps.*.config.low_level`,
normalizes list indexes to `[]`, and classifies generated keys as copied from
low-level authoring, rewritten by TolTECA, generated input assembly, or absent
from the authoring files.

## Compact Config Prototype

`expand_compact_config.py` expands a compact user-facing YAML file into the
current full `data/config.yaml` shape. It supports the normal TolTECA/Citlali
reduction intents:

| Mode | Default profile | Legacy Citlali reduction type |
| --- | --- | --- |
| `pointing` | `pointing_standard` | `pointing` |
| `oof` | `oof_standard` | `pointing` |
| `beammap` | `beammap_detector` | `beammap` |
| `science` | `science_standard` | `science` |

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/science_standard_compact.yaml \
  --output /tmp/science_standard.expanded.yaml \
  --summary-out /tmp/science_standard.summary.yaml
```

Use `--output-format low_level` to write only the Citlali low-level block, or
`--output-format tolteca` to wrap that block as indexed
`reduce.steps.0.config.low_level` for a TolTECA `NN*.yaml` file:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/oof_standard_compact.yaml \
  --output-format tolteca \
  --output /tmp/60_citlali_profile.yaml
```

Profiles live in `tools/config/profiles/`, and example compact configs live in
`tools/config/examples/`.

`--base-config` accepts either a full Citlali YAML file or a TolTECA YAML file
containing `reduce.steps.*.config.low_level`. This is useful for compatibility
work against an existing `70_reduce.yaml` baseline:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/pointing_compat_passthrough_compact.yaml \
  --base-config /path/to/70_reduce.yaml \
  --output-format low_level \
  --output /tmp/pointing_compat.low_level.yaml
```

The `*_compat_passthrough` profiles are intentionally empty. Use them as
validation harnesses when moving an existing TolTECA low-level block behind
compact knobs:

| Intent | Passthrough profile | Example compact file |
| --- | --- | --- |
| Pointing | `pointing_compat_passthrough` | `pointing_compat_passthrough_compact.yaml` |
| OOF | `oof_compat_passthrough` | `oof_compat_passthrough_compact.yaml` |
| Beammap | `beammap_compat_passthrough` | `beammap_compat_passthrough_compact.yaml` |
| Science | `science_compat_passthrough` | `science_compat_passthrough_compact.yaml` |

`pointing_compat_point70_compact.yaml` is a stronger point fixture: it uses the
same passthrough profile but expresses the normal point-152389 runtime, map,
product, processing, and source-fitting choices through compact keys. It still
expands to zero differences against that point `70_reduce.yaml` baseline.

The same stronger-fixture pattern exists for representative OOF, beammap, and
science baselines:

| Intent | Compact-key fixture |
| --- | --- |
| Pointing | `pointing_compat_point70_compact.yaml` |
| OOF | `oof_compat_149056_compact.yaml` |
| Beammap | `beammap_compat_3c273_compact.yaml` |
| Science | `science_compat_goodsn_compact.yaml` |

This is a prototype authoring layer only. It does not change the Citlali C++
parser, CLI, build system, or default runtime behavior.

## Low-Level YAML Equivalence

`compare_lowlevel_yaml.py` compares two low-level Citlali YAML trees and exits
with status 0 only when they match after ignored paths are removed. It accepts
either a bare low-level block or a TolTECA wrapper under
`reduce.steps.*.config.low_level`.

```bash
$HOME/tolteca/bin/python tools/config/compare_lowlevel_yaml.py \
  /path/to/70_reduce.yaml \
  /tmp/compact_profile.low_level.yaml \
  --ignore runtime.output_dir \
  --json-out /tmp/lowlevel_compare.json \
  --markdown-out /tmp/lowlevel_compare.md
```

Use this before Unity reduction tests when converting a TolTECA workflow to a
compact profile. The first target for behavior-preserving work is zero
differences against the existing `70_reduce.yaml` low-level block, excluding
TolTECA-owned paths such as `runtime.output_dir`.
