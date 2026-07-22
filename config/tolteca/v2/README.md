# Four-Mode Authoring Kit V2

This is the human-facing Phase 4.1 configuration structure for pointing, OOF,
Beammap, and science reductions. The files are ordinary TolTECA YAML. TolTECA
merges them directly in numeric order and supplies one generated low-level YAML
document to Citlali; no runtime translator is required.

Every mode merges exactly to its accepted Phase 4 policy when the operator
files are unchanged. The identities are pinned in `manifest.yaml`.

## File Roles

Each mode uses the same seven roles with a mode name in every filename. The
point directory uses `pointing` in filenames because that is the user-facing
reduction intent.

| Number and suffix | Audience | Purpose |
| --- | --- | --- |
| `60_MODE_internal_policy.yaml` | Citlali maintainers | Complete accepted policy. Generated, hash-checked, and not normally edited. |
| `71_MODE_runtime.yaml` | Site operator | Executable, thread count, output layout, and verbosity. |
| `72_MODE_observation.yaml` | TolPROJ | Data path, observation selection, APTs, calibrator fluxes, and pointing support. |
| `81_MODE_defaults.yaml` | Reducer | Routine mode-specific analysis choices. |
| `82_MODE_products.yaml` | Reducer | Mode-appropriate product and retained-data choices. |
| `90_MODE_advanced_overrides.yaml` | Advanced reducer | Additional supported user-facing settings omitted from the short defaults. Empty by default. |
| `99_MODE_expert_overrides.yaml` | Citlali expert | Detailed algorithm or diagnostic overrides. Empty by default and requires validation rationale. |

TolPROJ owns the observation file and refreshes the internal policy on a
same-kit setup. It preserves runtime, defaults, products, advanced, and expert
files so reducer edits are not lost.

## Normal Editing

Most reducers inspect or edit only:

1. `81_MODE_defaults.yaml` for analysis choices;
2. `82_MODE_products.yaml` for requested products; and
3. `71_MODE_runtime.yaml` when the executable or CPU allocation differs.

TolPROJ generates `72_MODE_observation.yaml` from project metadata and the
actual project directory layout. In the normal science/OOF layout
`<root>/<user>/<source>`, shared project data resolve through `../../data`.

The normal low-level surfaces remain bounded:

| Mode | Runtime leaves | Analysis leaves | Product leaves |
| --- | ---: | ---: | ---: |
| Pointing | 4 | 44 | 26 |
| OOF | 4 | 44 | 26 |
| Beammap | 4 | 43 | 5 |
| Science | 4 | 27 | 30 |

Pointing and OOF expose source strategy, source protection, map geometry,
cleaning, weighting, fruit loops, and learning. Beammap exposes its iteration,
convergence, reference-detector, prior, mask, cleaner, and fruit-loop policy,
but not point/OOF controls. Its product file is intentionally short: detector
TOD, split FITS, line-audit, and retained RTC/PTC TOD switches. Science keeps
the previously accepted structure, including consolidated fruit-loop controls.

Source finding is visible where relevant but marked experimental and remains
disabled in every accepted policy. Detailed thresholds and algorithm internals
remain available through the explicit advanced or expert files rather than the
routine surface.

## Validation

Validate every canonical kit from the Citlali repository root:

```bash
$HOME/tolteca/bin/python tools/config/tolteca_mode_kit.py validate-all \
  --config-root config/tolteca/v2 \
  --manifest config/tolteca/v2/manifest.yaml
```

Inspect a deployed project while allowing deliberate operator changes:

```bash
$HOME/tolteca/bin/python tools/config/tolteca_mode_kit.py merge \
  --mode science \
  --mode-dir /path/to/reduction \
  --manifest config/tolteca/v2/manifest.yaml \
  --yaml-out /tmp/science-merged.yaml \
  --json-out /tmp/science-config-report.json
```

The full config preflight additionally enforces user/expert classification,
mode-inapplicable control exclusion, file-size bounds, non-overlapping defaults
and products, exact versioned policy hashes, and byte-for-byte generator
reproduction.

## Regeneration And TolPROJ

`tools/config/generate_tolteca_v2_mode_kits.py` regenerates all four modes from
the mechanically exact V1 accepted baselines plus documented successor
controls. Review and validate a generated change before replacing this
directory.

Citlali is the canonical policy source. TolPROJ vendors an exact, hash-checked
snapshot and installs it only under `--refactor`; non-refactor setup retains
the established `70_reduce.yaml`/`72_reduce.yaml` workflow.
