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
current full `data/config.yaml` shape:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/science_standard_compact.yaml \
  --output /tmp/science_standard.expanded.yaml \
  --summary-out /tmp/science_standard.summary.yaml
```

Profiles live in `tools/config/profiles/`, and example compact configs live in
`tools/config/examples/`.

This is a prototype authoring layer only. It does not change the Citlali C++
parser, CLI, build system, or default runtime behavior.
