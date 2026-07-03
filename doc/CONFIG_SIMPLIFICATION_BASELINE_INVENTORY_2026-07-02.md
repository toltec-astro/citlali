# Config Simplification Baseline Inventory - 2026-07-02

Status: provisional baseline v1. The current classification policy is recorded
in `tools/config/config_key_classification.yaml`; the short policy summary is
`doc/CONFIG_POLICY_BASELINE_V1_2026-07-02.md`.

This note records a low-impact config simplification pass. It only adds
classification metadata and reporting tools; it does not change Citlali runtime
parsing, TolTECA workflow behavior, engine logic, defaults, or build files.

## Scope

The inventory covers the representative TolTECA baselines already used by the
compact compatibility suite:

| Intent | Baseline `70_reduce.yaml` |
| --- | --- |
| Pointing | `/Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/70_reduce.yaml` |
| OOF | `/Users/gwilson/work_toltec/local_data/OOF/149056/70_reduce.yaml` |
| Beammap | `/Users/gwilson/work_toltec/local_data/beammaps/3c273/70_reduce.yaml` |
| Science | `/Users/gwilson/work_toltec/local_data/2025-C1-COM-04/GOODS-N/70_reduce.yaml` |

The checked-in compatibility suite still passes before and after this inventory:

```bash
$HOME/tolteca/bin/python tools/config/run_compact_compatibility.py \
  --work-dir /private/tmp/citlali_compact_compat_config_simplification \
  --json-out /private/tmp/citlali_compact_compat_config_simplification/results.json \
  --markdown-out /private/tmp/citlali_compact_compat_config_simplification/results.md \
  --require-all
```

Result:

| Case group | Cases | Result |
| --- | ---: | --- |
| Pointing passthrough + compact keys | 2 | 0 diffs |
| OOF passthrough + compact keys | 2 | 0 diffs |
| Beammap passthrough + compact keys | 2 | 0 diffs |
| Science passthrough + compact keys | 2 | 0 diffs |

## Tools Added

- `tools/config/config_key_classification.yaml`: first-pass path classification
  policy for low-level keys.
- `tools/config/classify_lowlevel_config.py`: report generator that accepts
  TolTECA-wrapped YAML or bare low-level YAML, applies the policy, and writes
  Markdown/JSON/CSV.
- `tools/config/render_policy_review_dashboard.py`: static HTML dashboard
  generator for interactive policy review.

Example:

```bash
$HOME/tolteca/bin/python tools/config/classify_lowlevel_config.py \
  --cases tools/config/compact_compatibility_cases.yaml \
  --require-all \
  --json-out /private/tmp/citlali_config_simplification_inventory/classification_cases.json \
  --csv-out /private/tmp/citlali_config_simplification_inventory/classification_cases.csv \
  --markdown-out /private/tmp/citlali_config_simplification_inventory/classification_cases.md
```

Interactive review page:

```bash
$HOME/tolteca/bin/python tools/config/render_policy_review_dashboard.py \
  --cases tools/config/compact_compatibility_cases.yaml \
  --require-all \
  --output /private/tmp/citlali_config_policy_review/index.html
```

The dashboard is standalone HTML. It lets reviewers filter by rule id,
classification, top-level group, and reduction mode; inspect observed paths and
example values; sort rules by generated confidence; hover over rules scored
8/10 or lower to see the uncertainty rationale; record accept/change/discussion
notes in browser local storage; and export those decisions as JSON for a later
policy-file update. Exports include every rule id, including rules that were
not visited in the current filtered view.

## TolTECA Boundary Inventory

Generated `citlali_*.yaml` files were compared against the TolTECA authoring
YAML files with `tools/config/tolteca_lowlevel_inventory.py`.

| Intent | Generated leaves | Authored low-level leaves | Copied low-level leaves | TolTECA-generated input leaves | Rewritten leaves | Generated non-input leaves absent from authoring |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Pointing | 489 | 444 | 443 | 45 | 1 | 0 |
| OOF | 293 | 164 | 160 | 129 | 4 | 0 |
| Beammap | 945 | 485 | 485 | 460 | 0 | 0 |
| Science | 856 | 219 | 201 | 633 | 1 | 21 |

Notes:

- Pointing, OOF, and beammap are clean authoring-layer targets: generated
  non-input processing keys are copied from the low-level block, aside from
  TolTECA rewrites such as `runtime.output_dir`.
- Beammap was inventoried with explicit `70_reduce.yaml` and `72_reduce.yaml`.
  The directory also contains `70_reduce_perf_mapaccum.yaml`, so blind
  directory scans include an extra low-level authoring file.
- Science has 21 generated non-input leaves absent from authoring, all under
  `timestream.processed_time_chunk.clean.*`. These look like injected defaults
  or legacy-cleaner compatibility expansion and should be handled before using
  science as a strict authoring-equivalence target.

## Classification Summary

The classifier applies four exposure classes:

| Class | Meaning |
| --- | --- |
| `user-facing` | Should be expressible by compact profile fields or normal TolTECA authoring choices. |
| `expert` | Valid tuning, diagnostic, or algorithm-control key, hidden behind explicit expert overrides. |
| `hidden/internal` | TolTECA-assembled, runtime/profile-owned, or not a normal authoring knob. |
| `deprecated` | Historical or compatibility key with a current replacement or ignored behavior. |

Across the four `70_reduce.yaml` baselines:

| Scope | User-facing | Expert | Hidden/internal | Deprecated |
| --- | ---: | ---: | ---: | ---: |
| Leaf occurrences | 298 | 962 | 12 | 40 |
| Unique normalized paths | 101 | 424 | 3 | 31 |

By baseline:

| Intent | Leaves | User-facing | Expert | Hidden/internal | Deprecated |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pointing | 444 | 92 | 346 | 3 | 3 |
| OOF | 164 | 56 | 97 | 3 | 8 |
| Beammap | 485 | 85 | 394 | 3 | 3 |
| Science | 219 | 65 | 125 | 3 | 26 |

Unique paths by top-level node:

| Node | User-facing | Expert | Hidden/internal | Deprecated | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| `timestream` | 44 | 331 | 1 | 29 | 405 |
| `beammap` | 13 | 46 | 0 | 0 | 59 |
| `mapmaking` | 4 | 16 | 0 | 2 | 22 |
| `post_processing` | 11 | 11 | 0 | 0 | 22 |
| `interface_sync_offset` | 0 | 14 | 0 | 0 | 14 |
| `wiener_filter` | 10 | 0 | 0 | 0 | 10 |
| `runtime` | 5 | 1 | 2 | 0 | 8 |
| `kids` | 1 | 5 | 0 | 0 | 6 |
| `noise_maps` | 6 | 0 | 0 | 0 | 6 |
| `pointing` | 5 | 0 | 0 | 0 | 5 |
| `coadd` | 1 | 0 | 0 | 0 | 1 |
| `source` | 1 | 0 | 0 | 0 | 1 |

No observed baseline paths fell through to the classifier fallback rule.

## User-Facing Surface

The first-pass compact/user-facing set is intentionally small relative to the
full low-level schema. It includes:

- runtime intent, output layout, threads, and verbosity;
- source map regime/context metadata;
- map products: units, method, pixel axes, pixel size, coadd, and noise-product
  controls;
- pointing/OOF source strategy and fit radii;
- beammap setup and product toggles: iterations, convergence, derotation,
  reference-detector policy, detector weighting, prior enablement/file, detector
  TOD output enablement, optional RFI/scan-band mask enablement, and split-FITS
  enablement;
- timestream chunking, high-level RTC/PTC output toggles, high-level
  raw/filtered/cleaning switches, diagnostic line-audit mode, cleaner mode,
  standard-PCA depth, weighting type, source masks, fruit-loop iteration
  controls, and OOF support radii;
- source fitting, map filtering, source finding, and observed Wiener-filter
  authoring fields.

Everything else remains expert unless a profile deliberately promotes it.
Detailed beammap tuning remains reachable through compact-config `expert`
overrides.

## Review Import Status

The first policy-review export, created on 2026-07-02 at 15:32:35 UTC, was
checked against the current rule ids. It contained 96 decisions for 98 policy
rules, with no unknown ids. The missing ids were
`deprecated-legacy-standard-pca-ncalc` and `internal-tolteca-input-assembly`;
future dashboard exports initialize all rule ids before download.

Applied explicit `change` decisions from the first export:

- `runtime.parallel_policy`: `user-facing` -> `expert`
- `mapmaking.enabled`: `user-facing` -> `expert`
- `mapmaking.grouping`: `user-facing` -> `expert`
- `mapmaking.cr*`: `user-facing` -> `expert`
- `mapmaking.*_size_pix`: `user-facing` -> `expert`
- `noise_maps*`: `expert` -> `user-facing`
- `wiener_filter*`: `expert` -> `user-facing`

The broad `beammap*` promotion was refined in a follow-up review with the
beammap reducer. The current policy keeps 13 unique observed `beammap.*` paths
user-facing and classifies the other 46 observed `beammap.*` paths as expert
controls. User-facing beammap keys are high-level setup/product choices; expert
beammap families include detector-TOD sidecar detail, prior scoring/alignment,
flagging thresholds, mask thresholds, fitting support, phase strategy,
sensitivity PSD limits, and split-FITS flag selection.

The broad `timestream*` catch-all was also refined. The current policy keeps 44
unique observed `timestream.*` paths user-facing and classifies 331 observed
`timestream.*` paths as expert controls. `timestream.enabled` is now
profile-owned/expert because standard reductions keep TOD processing enabled;
`timestream.raw_time_chunk.line_audit.enabled` is user-facing as a diagnostic
mode toggle. Expert timestream families now cover raw/processed TOD sidecar
details, line-audit parameters, raw despiking/flagging/filtering/downsampling,
kernel and alt-az destriping policy, PTC cleaning/flagging/weighting details,
fruit-loop tuning, learning internals, and timestream diagnostic output. The
final `timestream*` rule remains only as an unobserved safety net.

The remaining observed policy-review items have been resolved with conservative
best guesses. `runtime.use_subdir` and `runtime.n_threads` stay user-facing
because they are normal local output/resource choices. `source.map_regime` is
user-facing source-context metadata. The broad `mapmaking*` catch-all has been
split into explicit expert families for `coverage_cut`, `jinc_filter`, and
`maximum_likelihood`; the final `mapmaking*` rule is now an unobserved safety
net. The broad `post_processing*` catch-all has likewise been split into
explicit expert families for map-filtering edge guards and source-fitting
Gaussian-model details; the final `post_processing*` rule is now an unobserved
safety net.

One accepted rule still had a stale proposed class in the export:
`internal-timestream-type` was accepted as `hidden/internal`, so its proposed
`expert` value was ignored.

## Deprecated Candidates Observed

Observed deprecated paths are mostly historical aliases or ignored fields:

- `timestream.precompute_pointing`
- `mapmaking.tan_ra`, `mapmaking.tan_dec`
- legacy PTC cleaner locations outside
  `timestream.processed_time_chunk.clean.*`, including
  `timestream.processed_time_chunk.standard_pca.*`,
  `timestream.processed_time_chunk.null_model.*`, and
  `timestream.processed_time_chunk.marchenko_pastur.*`
- legacy standard-PCA aliases under
  `timestream.processed_time_chunk.clean.n_eig_to_cut.*`,
  `clean.n_calc`, and `clean.stddev_limit`
- legacy weighting-correlation aliases under
  `timestream.processed_time_chunk.weighting.pair_corr.*`,
  `cm_el_corr.*`, and `cm_low_mid_ratio.*`

These should not be removed during the structural refactor. The safe path is to
warn or translate after YAML-level and product-level equivalence tests cover the
affected baselines.

## Implications

1. Keep the existing low-level schema accepted by Citlali while refactor work is
   active.
2. Treat `70_reduce.yaml` low-level blocks as compatibility baselines, not as
   the long-term user authoring surface.
3. Let compact profiles own most `timestream` and post-processing internals;
   expose only the classified user-facing paths.
4. Keep `inputs` and generated absolute `runtime.output_dir` behavior in the
   TolTECA boundary layer.
5. Resolve the science generated-cleaner defaults before enforcing strict
   generated `citlali_*.yaml` equivalence for science workflows.
