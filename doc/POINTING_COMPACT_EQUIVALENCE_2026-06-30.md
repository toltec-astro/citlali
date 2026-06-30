# Pointing Compact Profile Equivalence Check

Date: 2026-06-30

This note records the first YAML-level equivalence check between the compact
`pointing_standard` profile prototype and the current TolTECA point reduction
configuration for obsnum 152389.

## Baseline

Local TolTECA reduction directory:

```text
/Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor
```

Baseline authoring file:

```text
/Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/70_reduce.yaml
```

Validated generated Citlali file from the same reduction:

```text
/Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/reduced/redu08/citlali_o152389_0_2_c1.yaml
```

The TolTECA boundary inventory showed that this generated file contains 489
leaf values:

- 443 copied from the `70_reduce.yaml` low-level block
- 45 generated under top-level `inputs`
- 1 TolTECA-rewritten low-level value: `runtime.output_dir`

That makes `70_reduce.yaml` the useful baseline for behavior-preserving compact
profile work, with `runtime.output_dir` ignored.

## Reproduction

Expand the compact profile to a bare low-level block:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/pointing_standard_compact.yaml \
  --output-format low_level \
  --output /private/tmp/pointing_standard_compact.low_level.yaml \
  --summary-out /private/tmp/pointing_standard_compact.low_level.summary.yaml
```

Compare that low-level block with the TolTECA point baseline:

```bash
$HOME/tolteca/bin/python tools/config/compare_lowlevel_yaml.py \
  /Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/70_reduce.yaml \
  /private/tmp/pointing_standard_compact.low_level.yaml \
  --ignore runtime.output_dir \
  --json-out /private/tmp/pointing_standard_vs_70_lowlevel_compare.json \
  --markdown-out /private/tmp/pointing_standard_vs_70_lowlevel_compare.md
```

`compare_lowlevel_yaml.py` exits with status 0 only when the effective
low-level trees match after ignored paths are removed.

## First Result

The first comparison did not pass:

| Metric | Count |
| --- | ---: |
| Baseline leaf keys | 444 |
| Compact candidate leaf keys | 532 |
| Differences | 178 |

Differences by kind:

| Kind | Count |
| --- | ---: |
| Changed value | 76 |
| Candidate-only path | 95 |
| Missing candidate path | 7 |

Differences by top-level node:

| Node | Count |
| --- | ---: |
| `timestream` | 100 |
| `beammap` | 55 |
| `wiener_filter` | 5 |
| `mapmaking` | 4 |
| `pointing` | 4 |
| `post_processing` | 4 |
| `noise_maps` | 2 |
| `runtime` | 2 |
| `kids` | 1 |
| `source` | 1 |

Representative differences:

| Path | `70_reduce.yaml` | Compact profile |
| --- | --- | --- |
| `mapmaking.method` | `jinc` | `naive` |
| `mapmaking.pixel_axes` | `altaz` | `radec` |
| `post_processing.map_filtering.type` | `convolve` | `wiener_filter` |
| `source.map_regime` | `source_dominant` | `unknown` |
| `runtime.n_threads` | `6` | `8` |
| `pointing.source_strategy.fit_gaussian` | `true` | missing |
| `timestream.fruit_loops.max_iters` | `10` | `1` |
| `beammap.derotate` | `true` | `false` |

## Interpretation

This is not a Citlali runtime regression. It is a profile-definition mismatch:
the prototype `pointing_standard` profile is seeded from the current
`data/config.yaml` tree and only applies compact overrides, while the TolTECA
point `70_reduce.yaml` carries a validated historical low-level block.

The largest groups of differences are in `timestream` and inactive `beammap`
defaults. Those still matter for YAML equivalence because Citlali receives the
whole low-level tree, not just the branch implied by `runtime.reduction_type`.

## Compatibility Harness

The expander can also use an existing TolTECA `70_reduce.yaml` file as its base.
When the base contains `reduce.steps.*.config.low_level`, the tool extracts that
low-level block before applying profile and compact overlays.

The no-op point compatibility profile is:

```text
tools/config/profiles/pointing_compat_passthrough.yaml
```

It has an empty `full` overlay. This makes every compact override visible as a
deliberate diff from the selected baseline.

Reproduction:

```bash
$HOME/tolteca/bin/python tools/config/expand_compact_config.py \
  tools/config/examples/pointing_compat_passthrough_compact.yaml \
  --base-config /Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/70_reduce.yaml \
  --output-format low_level \
  --output /private/tmp/pointing_compat_passthrough.low_level.yaml \
  --summary-out /private/tmp/pointing_compat_passthrough.summary.yaml

$HOME/tolteca/bin/python tools/config/compare_lowlevel_yaml.py \
  /Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/70_reduce.yaml \
  /private/tmp/pointing_compat_passthrough.low_level.yaml \
  --ignore runtime.output_dir \
  --json-out /private/tmp/pointing_compat_passthrough_vs_70_lowlevel_compare.json \
  --markdown-out /private/tmp/pointing_compat_passthrough_vs_70_lowlevel_compare.md
```

Result:

| Metric | Count |
| --- | ---: |
| Baseline leaf keys | 444 |
| Compatibility candidate leaf keys | 444 |
| Differences | 0 |

The stronger point compatibility fixture is:

```text
tools/config/examples/pointing_compat_point70_compact.yaml
```

It uses the same empty compatibility profile, but expresses the normal point
152389 runtime, map, products, processing, and source-fitting values through
compact keys. It also expands against the point `70_reduce.yaml` baseline with
zero differences:

| Metric | Count |
| --- | ---: |
| Baseline leaf keys | 444 |
| Compact point-70 candidate leaf keys | 444 |
| Differences | 0 |

This confirms that those compact mappings are behavior-preserving for the
validated point case when the historical low-level block is used as the base.

## Next Decision

Before replacing TolTECA point authoring with compact profiles, choose one of
these compatibility targets:

1. Preserve the validated point `70_reduce.yaml` exactly, then make a
   `pointing_standard` compatibility profile that expands to zero differences
   against this baseline, excluding `runtime.output_dir`.
2. Intentionally adopt newer `data/config.yaml` defaults, then treat the 178
   differences as behavior changes requiring Unity reduction comparison.

For a safe refactor path, use option 1 first. After the compatibility profile
matches the existing point baseline, user-facing simplification can happen by
moving individual low-level choices behind compact names without changing the
expanded YAML.

The recommended next implementation step is to repeat the stronger fixture
pattern for OOF, beammap, and science: start with the passthrough profile,
choose one representative `70_reduce.yaml`, express a compact group with the
same values, and keep `compare_lowlevel_yaml.py` at zero differences before the
mapping is promoted into the normal profile.
