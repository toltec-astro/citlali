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

## Low-Level Key Classification

`classify_lowlevel_config.py` classifies low-level Citlali YAML leaves as
`user-facing`, `expert`, `hidden/internal`, or `deprecated` using the policy in
`config_key_classification.yaml`. It accepts either bare Citlali low-level YAML
or TolTECA YAML containing `reduce.steps.*.config.low_level`.

Classify the representative compact compatibility baselines:

```bash
$HOME/tolteca/bin/python tools/config/classify_lowlevel_config.py \
  --cases tools/config/compact_compatibility_cases.yaml \
  --require-all \
  --json-out /tmp/citlali_config_classification.json \
  --csv-out /tmp/citlali_config_classification.csv \
  --markdown-out /tmp/citlali_config_classification.md
```

The classification policy is review metadata only. It does not change the C++
parser, TolTECA config loading, default YAML, or reduction behavior.
The current policy is provisional baseline v1; see
`doc/CONFIG_POLICY_BASELINE_V1_2026-07-02.md` for the compact authoring
philosophy and mode-specific user/expert surfaces. The handoff summary for the
main refactor thread is `doc/CONFIG_SIMPLIFICATION_HANDOFF_2026-07-02.md`.

`audit_compact_surface_coverage.py` compares the provisional `user-facing`
classification against the compact translator. A path is counted as covered
only when normal compact fields expand back to that low-level path without
using the `expert:` escape hatch.

```bash
$HOME/tolteca/bin/python tools/config/audit_compact_surface_coverage.py \
  --cases tools/config/compact_compatibility_cases.yaml \
  --require-all \
  --json-out /tmp/citlali_compact_surface_coverage.json \
  --csv-out /tmp/citlali_compact_surface_coverage.csv \
  --markdown-out /tmp/citlali_compact_surface_coverage.md
```

Use this audit before adding typed config mirrors or new compact keys. Gaps are
not automatically bugs: some provisional user-facing paths are conditional
product families, inactive defaults in a given reduction mode, or policy
questions that may stay profile-owned.

`render_policy_review_dashboard.py` turns the same classification data into a
standalone interactive HTML review page. It keeps the YAML policy as the source
of truth, but makes policy review easier by showing rules, observed paths,
example values, per-mode counts, filters, generated confidence scores, review
status fields, and an exported review-decision JSON file.

```bash
$HOME/tolteca/bin/python tools/config/render_policy_review_dashboard.py \
  --cases tools/config/compact_compatibility_cases.yaml \
  --require-all \
  --output /tmp/citlali_config_policy_review/index.html
```

The dashboard is static HTML with embedded data. Rules can be sorted by
confidence, and any rule scored 8/10 or lower has hover text explaining the
uncertainty. Review choices are stored in browser local storage until exported;
exports include every rule id so unvisited rules are still visible in the next
review cut. Applying those decisions to `config_key_classification.yaml` is
still a deliberate follow-up edit.

## TolTECA Low-Level Boundary Inventory

`tolteca_lowlevel_inventory.py` compares TolTECA authoring YAML files with the
generated `citlali_*.yaml` low-level file that Citlali actually ingests.
TolTECA reads leading-number YAML files named like `70_reduce.yaml` in a
reduction directory and lets higher numbered files override lower numbered
files, so the preferred invocation is a directory scan:

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
`reduce.steps.0.config.low_level` for a TolTECA leading-number YAML file:

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

### Expert Overrides

Compact configs can still reach any low-level Citlali key through the
top-level `expert` block. The expander deep-merges `expert` after the selected
profile and normal compact keys, and lists the resulting paths in
`expert_override_paths` in the expansion summary.

In a TolTECA reduction directory, expert low-level settings should live in an
optional high-numbered overlay such as `80_expert.yaml`, after profile/default
material and normal reducer-facing choices:

```text
40_setup.yaml
60_citlali_profile.yaml
70_reduce.yaml
72_target.yaml
80_expert.yaml
```

The file should use the normal TolTECA low-level location:

```yaml
reduce:
  steps:
    0:
      config:
        low_level:
          timestream:
            raw_time_chunk:
              line_audit:
                enabled: true
```

For example, a beammap reduction can keep the normal compact surface while
temporarily tuning detailed beammap controls:

```yaml
schema: citlali-reduction-v2
mode: beammap
profile: beammap_detector

beammap:
  iterations: 6
  derotate: true
  priors: true

expert:
  beammap:
    flagging:
      array_upper_fwhm_arcsec: [10, 15, 20]
      array_network_robust_z: [0, 3.4, 0]
    priors:
      score_lambda_after_iter0: 3.0
    rfi_mask:
      enabled: true
      sigma_threshold: 6.0
```

The policy classification treats these detailed beammap fields as `expert`,
not unavailable; they remain deliberately reachable for beammap operations and
development without making them part of the default compact surface.

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

## Compact Compatibility Suite

`run_compact_compatibility.py` runs a manifest of compact examples against
representative local TolTECA `70_reduce.yaml` baselines. The checked-in manifest
uses `${HOME}/work_toltec/local_data/...` paths for the point, OOF, beammap,
and science examples discussed in the refactor notes.

```bash
$HOME/tolteca/bin/python tools/config/run_compact_compatibility.py \
  --work-dir /tmp/citlali_compact_compat \
  --json-out /tmp/citlali_compact_compat/results.json \
  --markdown-out /tmp/citlali_compact_compat/results.md
```

The suite skips missing baseline files by default, which keeps the tool usable
on machines that do not have every local reduction copied over. Use
`--require-all` when a full local validation data set is expected.

The current suite includes eight cases:

- passthrough and compact-key point fixtures
- passthrough and compact-key OOF fixtures
- passthrough and compact-key beammap fixtures
- passthrough and compact-key science fixtures

All cases are expected to have zero low-level differences after ignoring
`runtime.output_dir`.

## Translation Utilities

These translators live in the Citlali refactor tree on purpose. They treat
TolTECA YAML as an interchange format, but they do not require changes in the
TolTECA repository.

`lowlevel_to_compact_config.py` generates a compact compatibility YAML file
from an existing low-level Citlali or TolTECA `70_reduce.yaml` file. It only
emits compact keys whose low-level destinations already exist in the input, and
uses the `*_compat_passthrough` profile by default. By default, low-level paths
that are not represented by compact keys are preserved under `expert:` so the
result can round-trip without using the original `70_reduce.yaml` as a hidden
base config.

```bash
$HOME/tolteca/bin/python tools/config/lowlevel_to_compact_config.py \
  /path/to/70_reduce.yaml \
  --mode science \
  --output /tmp/science_compact.yaml \
  --summary-out /tmp/science_compact.summary.yaml
```

`--mode` is optional for `science` and `beammap` baselines because those map
directly from `runtime.reduction_type`. Use `--mode oof` for OOF reductions,
because legacy Citlali still represents OOF through
`runtime.reduction_type: pointing`.

The generated compact file should be validated immediately with the equivalence
tools:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  /tmp/science_compact.yaml \
  --base-config /path/to/70_reduce.yaml \
  --output-format low_level \
  --output /tmp/science_compact.low_level.yaml

$HOME/tolteca/bin/python tools/config/compare_lowlevel_yaml.py \
  /path/to/70_reduce.yaml \
  /tmp/science_compact.low_level.yaml \
  --ignore runtime.output_dir
```

By default, the bootstrapper omits `output.dir` when the low-level value is an
absolute path. Pass `--include-output-dir` when reproducing site-specific output
paths is desired. Pass `--preserve-unmapped none` only when intentionally
dropping unmapped low-level content.

To expand a loss-preserving compact file back to an old-style low-level or
TolTECA YAML file, use an empty base:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  /tmp/science_compact.yaml \
  --base-config none \
  --output-format tolteca \
  --output /tmp/science_roundtrip_70_reduce.yaml
```

`run_translation_roundtrip.py` automates the full old -> compact -> old check
and compares the result against the input. This is the preferred smoke test
when validating older Citlali branches:

```bash
$HOME/tolteca/bin/python tools/config/run_translation_roundtrip.py \
  /path/to/70_reduce.yaml \
  --mode science \
  --work-dir /tmp/citlali_config_translation_roundtrip \
  --expected-diff-count 0
```
