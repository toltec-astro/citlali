# Citlali Config Simplification Plan - 2026-06-29

This note scopes a user-facing config rework for Citlali. It is intentionally a
design and migration plan only; no runtime config parsing has been changed.

## Why This Is Worth Doing

The current default config is a full expert/debug surface:

- `data/config.yaml` has 491 YAML leaf keys by static inventory.
- 353 of those leaves are under `timestream`.
- Many keys are operationally useful only for diagnostics, experiments,
  instrument debugging, or one-off failure analysis.
- New users have to distinguish core choices from implementation details before
  they can run a normal science, pointing, or beammap reduction.

The refactor should therefore separate two concerns:

- **User-facing config:** small, stable, task-oriented, and profile-based.
- **Internal config:** typed, complete, validated, and able to represent every
  expert/debug knob needed by maintainers.

This should be a compatibility-preserving migration, not an abrupt schema
break.

## Current Surface Snapshot

Generated with:

```bash
$HOME/tolteca/bin/python tools/config/config_inventory.py \
  --config data/config.yaml \
  --source-root .
```

Top-level YAML leaf counts:

| Node | Leaf keys |
| --- | ---: |
| `timestream` | 353 |
| `beammap` | 59 |
| `mapmaking` | 22 |
| `post_processing` | 22 |
| `wiener_filter` | 10 |
| `runtime` | 8 |
| `kids` | 6 |
| `noise_maps` | 6 |
| `coadd` | 1 |
| `inputs` | 1 |
| `interface_sync_offset` | 1 |
| `pointing` | 1 |
| `source` | 1 |

This count is not itself a problem; many expert controls are legitimate. The
problem is that the expert surface is also the first surface a normal user sees.

## Design Principle

Keep the full schema valid, but stop making it the normal authoring interface.

The intended path:

```text
compact user config + named profile + optional expert overrides
  -> typed config loader
  -> expanded legacy-compatible config tree
  -> existing Engine / RTC / PTC / mapmaking fields during migration
```

The compact config should be treated as an authoring layer. The expanded config
should still be copied into reduction outputs for provenance.

## Config Surface Levels

Every key in the full schema should eventually be classified into one of these
levels:

| Level | Meaning | User Exposure |
| --- | --- | --- |
| Core | Needed for ordinary reductions | Present in compact templates |
| Common advanced | Often adjusted by experienced reducers | Available in profile docs |
| Expert | Legitimate but uncommon tuning/debug knob | Allowed under `expert:` override |
| Diagnostic | Expensive or specialized inspection feature | Hidden by default, explicit profile or override |
| Experimental | Under active development or not broadly validated | Explicit opt-in with warning |
| Deprecated | Retained for compatibility but replaced | Warn, document replacement, remove only after validation |

This classification can start as documentation and later become metadata used
by config validation and generated docs.

## Proposed Compact Schema

The compact schema should describe user intent rather than every implementation
knob.

Example shape:

```yaml
schema: citlali-reduction-v2
mode: science        # science, pointing, beammap
profile: standard   # standard, fast, diagnostic, jinc, tod_export, etc.

inputs:
  # Either keep current TolTECA-provided input structure or reference an
  # external input manifest. Do not redesign data discovery in this PR.
  legacy: /path/to/input_manifest_or_full_inputs_block.yaml

output:
  dir: /path/to/redu
  subdir: true
  verbose: false

runtime:
  threads: 8
  parallel: omp

map:
  unit: mJy/beam
  method: naive
  grouping: auto
  pixel_size_arcsec: 1.0
  center: auto
  size: auto
  coadd: false

products:
  maps: true
  noise: false
  noise_products: true
  tod: none          # none, rtc, ptc, both
  diagnostics: normal

processing:
  clean: standard
  weighting: full
  fruitloops: off

expert:
  # Optional raw legacy YAML subtree patches for maintainers.
  timestream:
    raw_time_chunk:
      filter:
        enabled: true
```

The exact field names can change, but the core idea should stay: a compact
front door plus an explicit expert escape hatch.

## Suggested Profiles

Profiles should be small named bundles that expand into the current full YAML
schema.

### `science_standard`

Purpose: normal science map production.

Expose:

- output directory and subdir behavior
- thread count and parallel policy
- map unit, grouping, method, pixel size, optional center/size
- coadd on/off
- noise maps on/off and number of noise maps
- clean on/off and cleaning grouping
- weighting type
- fruitloops off/on and iteration count
- TOD output off by default

Keep profile-managed:

- despike/filter/downsample internals
- line-audit internals
- adaptive selector internals
- learning thresholds
- diagnostic sidecars
- map filtering convergence thresholds

### `science_diagnostic`

Purpose: investigate artifacts or suspicious reductions.

Expose:

- all `science_standard` fields
- TOD output mode
- selected chunk policy
- diagnostic product level
- optional line audit and mapdiag toggles

Keep profile-managed:

- exact thresholds for line families, impulsive coincidence, learned masks, and
  contributor tracing unless explicitly overridden.

### `pointing_standard`

Purpose: compact-source pointing/focus-style reductions.

Expose:

- source strategy: `standard` or `psf_preserve`
- source protection radius
- fruitloops on/off and iteration count
- fit radius
- map method, pixel size, and output products

Keep profile-managed:

- detailed learning thresholds
- busy-network acceptance thresholds
- high-weight validation thresholds
- map-pixel contributor tracing internals

### `beammap_detector`

Purpose: detector-grouped beammap reductions.

Expose:

- iteration count and convergence tolerance
- detector grouping mode
- derotation on/off
- reference detector policy
- prior use on/off and prior file
- flagging strictness preset
- detector TOD output on/off

Keep profile-managed:

- soft-prior scoring thresholds
- RFI sample-mask thresholds
- scan-band mask thresholds
- fit-bound diagnostics
- sensitivity PSD internals

### `tod_export`

Purpose: produce selected RTC/PTC products for diagnostics without forcing a
large custom config.

Expose:

- RTC/PTC/both
- full/mini/full_outer/mini_outer
- chunk selection: all, explicit indices, uniform count, source-dense count
- output subdirectory

Keep profile-managed:

- netCDF writer internals
- sidecar diagnostic details

## Initial Core Key Set

The first compact templates should expose only keys that users commonly need.

Candidate core fields:

- `mode`
- `profile`
- `inputs`
- `output.dir`
- `output.subdir`
- `output.verbose`
- `runtime.threads`
- `runtime.parallel`
- `map.unit`
- `map.method`
- `map.grouping`
- `map.pixel_axes`
- `map.pixel_size_arcsec`
- `map.center`
- `map.size`
- `map.coadd`
- `products.maps`
- `products.noise`
- `products.noise_count`
- `products.noise_realizations`
- `products.tod`
- `products.diagnostics`
- `processing.clean`
- `processing.clean_grouping`
- `processing.weighting`
- `processing.fruitloops`
- `processing.fruitloops_iters`
- `pointing.source_strategy`
- `beammap.iterations`
- `beammap.derotate`
- `beammap.priors`

Everything else should remain available through expert overrides until we have
usage evidence that it belongs in the compact layer.

## Typed Internal Config Model

The structural refactor should still build a complete typed config model. The
compact user surface is not a substitute for internal validation.

Suggested internal structs:

- `RuntimeConfig`
- `InputConfig`
- `OutputConfig`
- `MapmakingConfig`
- `NoiseConfig`
- `TimestreamConfig`
- `RTCConfig`
- `PTCConfig`
- `FruitLoopsConfig`
- `LearningConfig`
- `PointingConfig`
- `BeammapConfig`
- `PostProcessingConfig`
- `WienerFilterConfig`

Each typed config section should support:

- parse from existing full YAML
- parse from expanded compact YAML
- validation with path-rich error messages
- defaults from profile metadata
- serialization back to expanded YAML for provenance

During migration, the typed model can populate the existing `Engine`, `RTCProc`,
`PTCProc`, and `MapBuffer` fields without changing processing behavior.

## Profile Expansion Rules

Profile expansion should be deterministic and inspectable:

1. Load base defaults from the current full schema.
2. Apply mode-specific profile defaults.
3. Apply compact user fields.
4. Apply explicit `expert:` overrides.
5. Validate the complete expanded config.
6. Write the expanded config to the reduction output directory.

Conflicts should be rejected early. For example, if `profile: fast` disables
noise products and the user sets `products.noise: true`, the compact field
should win, but the expanded config should record that override.

## Backward Compatibility

The current full YAML config should remain supported throughout the structural
refactor.

Recommended compatibility policy:

- Full legacy configs keep working.
- Compact configs are accepted only after they can be expanded to an identical
  full config manifest for baseline cases.
- Existing key names are not removed during early refactor PRs.
- Deprecated keys produce warnings only after a validated replacement exists.
- Output directories include both the user-authored compact config and the
  expanded full config.

## Migration PR Sequence

### Config PR 0: Inventory and Classification

Deliverables:

- Static key inventory tooling.
- Key classification document or CSV: core, advanced, expert, diagnostic,
  experimental, deprecated.
- Review with maintainers before implementation.

Validation:

- Documentation/tool syntax checks only.

### Config PR 1: Profile Metadata and Templates

Deliverables:

- Profile definitions for `science_standard`, `pointing_standard`, and
  `beammap_detector`.
- Compact example configs for each profile.
- Documentation for expert overrides.

Validation:

- Static expansion dry-run once the expander exists.

### Config PR 2: Full YAML Typed Parser

Deliverables:

- Typed config structs that parse the current full YAML schema.
- Unit tests for missing, invalid, out-of-range, enum, and fixed-vector errors.
- No runtime behavior changes beyond validation error quality.

Validation:

- Local unit tests if/when build is available.
- Unity compile and config smoke cases.

### Config PR 3: Compact Config Expander

Deliverables:

- Tool/library function that expands compact config to current full YAML.
- Golden tests comparing expanded configs to hand-authored full configs.
- No pipeline execution path change yet.

Validation:

- Expansion tests.
- Baseline manifest comparison after manually running expanded full config.

### Config PR 4: CLI Acceptance of Compact Configs

Deliverables:

- CLI detects full vs compact schema.
- Compact config expands before existing pipeline construction.
- Output includes compact source config and expanded full config.

Validation:

- Unity baseline reductions using compact and equivalent full configs.
- Manifest comparisons against `gw_dev` and current full-schema reductions.

### Config PR 5: Default Config Split

Deliverables:

- Keep `data/config.yaml` as expert reference.
- Add small profile templates as the default user-facing examples.
- Update docs and `--dump_config` behavior only after compatibility review.

Validation:

- CLI behavior review.
- Maintainer sign-off on user-facing default changes.

## Risks

- Hiding knobs too aggressively can make operational debugging harder.
- Compact profile defaults can accidentally drift from established full-config
  behavior.
- A profile layer can obscure provenance unless the expanded config is always
  saved.
- Some current knobs may be required by TolTECA-generated configs even if they
  look expert-only from Citlali alone.
- Dynamic config reads and mode-specific paths mean static inventory is only a
  starting point, not proof that a key is unused.

## Recommendation

Do not attempt a complete config rewrite as the first runtime refactor.

Do start the config rework now as a parallel design track:

1. Inventory and classify the current 491-key surface.
2. Define compact profile templates with explicit expert escape hatches.
3. Build typed full-schema parsing and validation.
4. Add compact-to-full expansion and prove equivalence with Unity reductions.
5. Only then change the normal user-facing default config.

This gives users a much smaller interface without taking away the expert
controls maintainers need for difficult reductions.
