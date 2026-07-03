# Config Simplification Handoff - 2026-07-02

This is the handoff point for the config-simplification support thread. The
work here is deliberately non-disruptive: it adds policy metadata, review
tooling, and documentation only. It does not change Citlali runtime parsing,
TolTECA workflow behavior, engine logic, defaults, build files, or reduction
outputs.

## Current State

The low-level key exposure policy is now declared **provisional baseline v1**.
The policy is conservative: compact configs expose normal reducer-facing
choices, while detailed algorithm controls remain reachable through explicit
expert overrides.

The bidirectional translation utilities are owned by this Citlali refactor
tree. They read and write TolTECA-shaped YAML, but the implementation should
stay here because the refactor branch controls the compact schema and old-branch
compatibility tests. No TolTECA repository changes are required.

The project-queue catalog/audit layer should live in `~/GitHub/tolproj`, not
here. `tolproj` knows which projects, reduction directories, and queued
deliverables matter; Citlali owns only the config semantics and translation
tools. Once reduction-type defaults settle, `tolproj` also needs its own YAML
templates updated in its templates archive so new project scaffolding emits the
current compact/default config shape.

Source of truth:

- `tools/config/config_key_classification.yaml`

Supporting docs:

- `doc/CONFIG_POLICY_BASELINE_V1_2026-07-02.md`
- `doc/CONFIG_SIMPLIFICATION_BASELINE_INVENTORY_2026-07-02.md`
- `tools/config/README.md`

Review/reporting tools:

- `tools/config/classify_lowlevel_config.py`
- `tools/config/render_policy_review_dashboard.py`

Translation tools:

- `tools/config/lowlevel_to_compact_config.py`
- `tools/config/expand_compact_config.py`
- `tools/config/run_translation_roundtrip.py`

Generated local review page:

- `/private/tmp/citlali_config_policy_review/index.html`

## Baseline Counts

The policy was generated from the representative compact-compatibility
baselines for pointing, OOF, beammap, and science.

Across those four `70_reduce.yaml` inputs:

| Metric | Count |
| --- | ---: |
| Leaf occurrences | 1312 |
| Unique normalized paths | 559 |
| Fallback-classified leaf occurrences | 0 |

Unique normalized paths by class:

| Class | Unique paths |
| --- | ---: |
| `user-facing` | 101 |
| `expert` | 424 |
| `hidden/internal` | 3 |
| `deprecated` | 31 |

Leaf occurrences by class:

| Class | Leaves |
| --- | ---: |
| `user-facing` | 298 |
| `expert` | 962 |
| `hidden/internal` | 12 |
| `deprecated` | 40 |

## Main Decisions Captured

- Expert classification does not remove access. Expert keys remain reachable
  through compact-config `expert:` overrides or later TolTECA low-level overlay
  files such as `80_expert.yaml`.
- TolTECA/Tollan leading-number YAML loading supports this overlay model: later
  numeric YAML files recursively merge over earlier files.
- `beammap.*` is split into a small user-facing setup/product surface plus
  detailed expert families for prior tuning, flagging, mask thresholds, fitting
  support, phase strategy, sensitivity limits, and split-FITS flag selection.
- `timestream.*` is split into user-facing high-level toggles/choices and
  expert families for raw/PTC sidecar details, line-audit parameters, filtering,
  despiking, flagging, cleaning, weighting, learning, and diagnostic output.
- `runtime.use_subdir`, `runtime.n_threads`, and `source.map_regime` are
  treated as user-facing for baseline v1.
- Remaining observed `mapmaking.*` and `post_processing.*` questions were split
  into explicit expert families. Broad catchalls remain only as safety nets for
  future baselines.
- Deprecated keys remain accepted during the structural refactor. The safe path
  is warning or translation only after YAML-level and product-level equivalence
  tests cover the affected baselines.

## Files To Carry Forward

Include these changes when handing work to the main refactor thread:

| Path | Purpose |
| --- | --- |
| `tools/config/config_key_classification.yaml` | Baseline v1 low-level exposure policy. |
| `tools/config/classify_lowlevel_config.py` | Policy classifier/report generator. |
| `tools/config/render_policy_review_dashboard.py` | Static review dashboard generator. |
| `tools/config/lowlevel_to_compact_config.py` | Old low-level/TolTECA YAML to compact translator; preserves unmapped paths under `expert:` by default. |
| `tools/config/expand_compact_config.py` | Compact to full low-level/TolTECA YAML translator; supports `--base-config none` for lossless round trips. |
| `tools/config/run_translation_roundtrip.py` | Old -> compact -> old round-trip validation driver for older-branch testing. |
| `tools/config/README.md` | Usage notes for classification, dashboard, compact config, and expert overlays. |
| `doc/CONFIG_POLICY_BASELINE_V1_2026-07-02.md` | Short policy philosophy and mode-surface summary. |
| `doc/CONFIG_SIMPLIFICATION_BASELINE_INVENTORY_2026-07-02.md` | Detailed inventory, counts, review history, and implications. |
| `doc/CONFIG_SIMPLIFICATION_HANDOFF_2026-07-02.md` | This handoff note. |

Do not mix unrelated engine/refactor files into a config-simplification commit.

## Regeneration Commands

Use the local TolTECA environment:

```bash
$HOME/tolteca/bin/python -m py_compile \
  tools/config/classify_lowlevel_config.py \
  tools/config/render_policy_review_dashboard.py \
  tools/config/expand_compact_config.py
```

Regenerate classification outputs:

```bash
$HOME/tolteca/bin/python tools/config/classify_lowlevel_config.py \
  --cases tools/config/compact_compatibility_cases.yaml \
  --require-all \
  --json-out /private/tmp/citlali_config_policy_review/classification_cases.json \
  --csv-out /private/tmp/citlali_config_policy_review/classification_cases.csv \
  --markdown-out /private/tmp/citlali_config_policy_review/classification_cases.md
```

Regenerate the dashboard:

```bash
$HOME/tolteca/bin/python tools/config/render_policy_review_dashboard.py \
  --cases tools/config/compact_compatibility_cases.yaml \
  --require-all \
  --output /private/tmp/citlali_config_policy_review/index.html
```

Rerun compact compatibility:

```bash
$HOME/tolteca/bin/python tools/config/run_compact_compatibility.py \
  --work-dir /private/tmp/citlali_compact_compat_handoff \
  --json-out /private/tmp/citlali_compact_compat_handoff/results.json \
  --markdown-out /private/tmp/citlali_compact_compat_handoff/results.md \
  --require-all
```

Expected current result:

```text
compact compatibility: passed=8 failed=0 skipped=0
```

Run an old-branch translation round trip:

```bash
$HOME/tolteca/bin/python tools/config/run_translation_roundtrip.py \
  /path/to/70_reduce.yaml \
  --mode science \
  --work-dir /private/tmp/citlali_config_translation_roundtrip \
  --expected-diff-count 0
```

The round-trip driver generates a compact file, expands it from an empty base
with `--base-config none`, writes low-level and TolTECA-style outputs, and
compares the result with the original input.

## Next Useful Step

The next Citlali-side practical step is a compact-surface coverage audit:

1. Map each `user-facing` low-level path to an existing compact config key or
   mark it as a gap.
2. Split gaps by mode: pointing, OOF, beammap, science, or shared.
3. Identify expert families that deserve example `80_expert.yaml` snippets.
4. Keep the audit as docs/tooling until the main refactor thread is ready to
   wire any missing compact fields into production behavior.

That work should still avoid engine changes unless the main refactor thread is
ready for them.

The next `tolproj`-side step is separate: add a project config catalog/audit
tool there, backed by manifests of active project reductions, and update the
templates archive YAMLs after the final reduction defaults are chosen.
