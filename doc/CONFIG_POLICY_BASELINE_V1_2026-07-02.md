# Config Policy Provisional Baseline v1 - 2026-07-02

This note declares the current low-level key classification policy as
**provisional baseline v1**. It is intentionally low stakes: the policy is
review metadata for the compact-config/refactor work, not a runtime schema
change. No Citlali parser behavior, TolTECA loading behavior, defaults, build
files, or reduction engine logic are changed by this baseline.

The source of truth is
`tools/config/config_key_classification.yaml`, with generated review/reporting
provided by:

- `tools/config/classify_lowlevel_config.py`
- `tools/config/render_policy_review_dashboard.py`
- `/private/tmp/citlali_config_policy_review/index.html`

## Policy Philosophy

The compact authoring surface should be small, stable, and reduction-oriented.
Normal reducers should see intent, product, geometry, resource, and common
algorithm choices. They should not have to see the full low-level Citlali
implementation surface.

The low-level surface remains fully reachable. Anything classified `expert` is
not removed or forbidden; it is deliberately placed behind explicit expert
overrides. In a TolTECA reduction directory, those overrides should normally
live in a high-numbered overlay such as `80_expert.yaml`, after the normal
profile/default and reducer-facing YAML files.

The working classes are:

| Class | Meaning |
| --- | --- |
| `user-facing` | Normal compact profile fields or normal TolTECA authoring choices. |
| `expert` | Valid tuning, diagnostic, or algorithm-control key reachable through explicit expert overrides. |
| `hidden/internal` | TolTECA/profile/runtime-owned plumbing, not a normal authoring knob. |
| `deprecated` | Compatibility or historical key retained until validation supports warning, translation, or removal. |

This baseline prefers conservative exposure. A key can be promoted later when a
real workflow needs it often enough to justify a compact field. A key can also
move back to expert if experience shows it is mostly profile policy or
debugging machinery.

## Baseline Inputs

The baseline was generated from the representative compact-compatibility
fixtures:

| Intent | Baseline |
| --- | --- |
| Pointing | `/Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/70_reduce.yaml` |
| OOF | `/Users/gwilson/work_toltec/local_data/OOF/149056/70_reduce.yaml` |
| Beammap | `/Users/gwilson/work_toltec/local_data/beammaps/3c273/70_reduce.yaml` |
| Science | `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/70_reduce.yaml` |

Across these four baselines there are 559 unique normalized low-level paths:

| Class | Unique paths |
| --- | ---: |
| `user-facing` | 101 |
| `expert` | 424 |
| `hidden/internal` | 3 |
| `deprecated` | 31 |

No observed baseline path is classified by the fallback rule.

## Mode Surfaces

Counts below are unique normalized paths within each representative baseline.
They are path-policy counts, not claims that every key is active for that mode.
Some generic low-level scaffolding appears across multiple reduction types.

| Mode | Unique paths | User-facing | Expert | Hidden/internal | Deprecated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pointing | 413 | 92 | 315 | 3 | 3 |
| OOF | 142 | 56 | 75 | 3 | 8 |
| Beammap | 441 | 85 | 350 | 3 | 3 |
| Science | 187 | 61 | 97 | 3 | 26 |

### Pointing

User-facing surface:

- runtime intent, output path/layout, thread count, and verbosity
- map unit, method, pixel axes, and pixel size
- coadd and noise-map product controls
- pointing source strategy and source-context metadata
- source fitting, source finding, filtered-map product controls, and Wiener
  template controls
- high-level TOD chunking, raw/PTC output toggles, raw filtering/despiking
  switches, calibration switches, cleaner selection, standard-PCA depth, map
  weighting type, source masks, and fruit-loop iteration controls

Expert surface:

- manual sync offsets
- detailed raw/PTC TOD flagging, filtering, cleaner, weighting, learning, and
  fruit-loop internals
- mapmaking manual geometry, coverage, jinc-filter, and maximum-likelihood
  controls
- KIDs internals and post-processing edge/fit model details

### OOF

User-facing surface:

- runtime intent, output path/layout, thread count, and verbosity
- map unit, method, pixel axes, and pixel size
- coadd, noise-map controls, source fitting/finding, filtered-map controls, and
  Wiener template controls
- high-level TOD chunking, raw/PTC output toggles, raw filtering/despiking,
  calibration switches, cleaner selection, weighting type, and fruit-loop
  source/iteration controls

Expert surface:

- manual sync offsets
- mapmaking manual geometry, jinc-filter, and maximum-likelihood controls
- beammap/shared low-level flagging and sensitivity details that remain
  profile or expert policy for OOF workflows
- detailed raw/PTC TOD processing, KIDs internals, and source-fit Gaussian
  model details

### Beammap

User-facing surface:

- beammap iterations, convergence radius, derotation, reference-detector
  policy, detector weighting, detector-TOD output toggle, prior enablement/file,
  RFI/scan-band mask enablement, and split-FITS enablement
- runtime intent, output path/layout, thread count, and verbosity
- map unit, method, pixel axes, pixel size, coadd, and noise-map controls
- source-context metadata, source fitting/finding, filtered-map controls, and
  Wiener template controls
- high-level TOD chunking, raw/PTC output toggles, raw filtering/despiking,
  line-audit mode toggle, calibration switches, cleaner selection, weighting
  type, and fruit-loop controls

Expert surface:

- detailed beammap prior scoring/alignment, detector flagging, RFI and scan-band
  thresholds, fitting support, phase strategy, sensitivity PSD limits, and
  split-FITS flag selection
- line-audit thresholds and fixed-notch lists
- detailed raw/PTC TOD flagging, filtering, cleaner, weighting, learning, and
  fruit-loop internals
- manual sync offsets, mapmaking expert controls, KIDs internals, and
  post-processing edge/fit model details

### Science

User-facing surface:

- runtime intent, output path/layout, thread count, and verbosity
- map unit, method, pixel axes, pixel size, coadd, and noise-map controls
- source fitting/finding, filtered-map controls, and Wiener template controls
- high-level TOD chunking, raw/PTC output toggles, raw filtering/despiking,
  calibration switches, cleaner selection, weighting type, and fruit-loop
  source/iteration controls

Expert surface:

- manual sync offsets
- detailed raw filtering, alt-az destriping, raw kernel policy, processed
  weighting, cleaner, flagging, and fruit-loop internals
- mapmaking manual geometry, jinc-filter, and maximum-likelihood controls
- KIDs internals, shared beammap flagging details, and source-fit Gaussian
  model details

Science currently carries the largest deprecated surface because its observed
baseline includes legacy cleaner and weighting aliases. Those keys should stay
accepted until product-level validation supports translation or warning.

## Expert Override Placement

For compact configs, expert controls live under a top-level `expert` block and
are deep-merged after the selected profile and normal compact keys.

For TolTECA reduction directories, expert low-level controls should normally be
placed in a later leading-number YAML file:

```text
40_setup.yaml
60_citlali_profile.yaml
70_reduce.yaml
72_target.yaml
80_expert.yaml
```

The low-level path remains the normal TolTECA location:

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

TolTECA/Tollan loads leading-number YAML files in numeric order and recursively
merges them, so later files override earlier files. That is the basis for the
`80_expert.yaml` convention.

## Iteration Rules

Baseline v1 is good enough to support ongoing refactor work. It should change
when one of these things happens:

1. A reducer needs an expert key often enough that hiding it slows normal work.
2. A newly added representative baseline produces fallback-classified paths.
3. A profile starts owning a key so consistently that user exposure creates
   confusion.
4. Validation shows a deprecated key can be safely warned, translated, or
   removed.

The expected workflow is:

1. Regenerate the dashboard.
2. Review changed or low-confidence rules.
3. Export review JSON.
4. Apply deliberate policy edits to `config_key_classification.yaml`.
5. Rerun classification and compact compatibility checks.

