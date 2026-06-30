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
