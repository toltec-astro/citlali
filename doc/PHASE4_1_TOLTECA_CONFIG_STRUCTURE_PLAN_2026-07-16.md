# Phase 4.1 TolTECA Config Structure Plan - 2026-07-16

## Goal

Provide a checked-in, operator-facing numbered-YAML structure for the four
supported observing modes:

- pointing;
- out-of-focus holography (OOF);
- Beammap; and
- science.

The structure must work with TolTECA's existing rule that all `NN_*.yaml`
files in a reduction directory are merged in ascending numeric order, with
higher-numbered values taking precedence. Citlali continues to receive one
generated low-level YAML document. This phase improves the human authoring
surface; it does not move TolTECA's merge responsibility into Citlali.

## Current Problem

The validated work directories contain a large mode-specific `70_reduce.yaml`,
observation/calibration content in `72_reduce.yaml`, and accumulated `80`-series
debug or validation overlays. The files work, but normal controls, deployment
paths, observation selection, calibration records, product choices, and expert
experiments are not separated consistently. A user has to understand too much
of the complete low-level schema to make a routine edit safely.

## Proposed Checked-In Kit

Create four ready-to-copy directories under `config/tolteca/`:

```text
config/tolteca/
  README.md
  point/
  oof/
  beammap/
  science/
```

Each mode directory has the same authoring roles:

| File | Owner and normal use |
| --- | --- |
| `70_pipeline_MODE.yaml` | Repository-managed canonical mode policy and full Citlali step skeleton. Users normally do not edit it. |
| `71_runtime.yaml` | Site/workspace values such as executable path, job key, and input root. This is a normal deployment edit. |
| `72_observation.yaml` | Observation selection, APT records, pointing calibration objects, and Beammap source/calibrator records. This is a normal dataset edit. |
| `80_products.yaml` | Named standard product/output policy for the mode. Users select or copy a documented product profile rather than finding scattered switches. |
| `90_user_overrides.yaml` | Small, optional user override surface. Every entry is intentional and is expected to win by merge order. |

TolTECA-generated `40_setup.yaml` remains outside the kit and must not be
edited by hand. Diagnostic, performance, and scientific experiments use named
optional overlays above `80`; they are not accumulated silently in the normal
mode template.

The exact filenames may be adjusted if a real TolTECA fixture exposes a merge
constraint, but the five roles and their precedence remain fixed.

The implemented filenames are `70_pipeline.yaml`, `71_runtime.yaml`,
`72_observation.yaml`, `80_products.yaml`, and `90_user_overrides.yaml`.

## Editing Contract

A routine user should usually edit only:

1. `71_runtime.yaml` when deployment paths differ;
2. `72_observation.yaml` for the selected data and calibration context; and
3. `90_user_overrides.yaml` for a deliberate analysis change.

The mode policy and product profile remain readable standard TolTECA YAML, not
a new runtime language. Existing compact profiles and translators may generate
or verify repository-managed templates, but production execution must not
depend on running a Python translator. The expanded numbered files are what
TolTECA loads and what users can inspect.

Expert access is retained. Any valid low-level Citlali leaf may be placed in
`90_user_overrides.yaml`, but the merged-config report must identify the
override, its authority class, and whether it changes activation or products.
This avoids hiding a necessary control while keeping rare controls out of the
normal editing path.

## Mode Responsibilities

| Mode | Policy that must be obvious in the kit |
| --- | --- |
| Point | Point-source strategy, fitting, pointing products, and compact validation outputs |
| OOF | Pointing-engine execution with OOF-specific selection, PSF-preserving source policy, and deliberately bounded outputs |
| Beammap | Calibrator/source records, detector weighting, priors, iterations, fitting/flagging, split maps, and detector diagnostics |
| Science | Multi-observation calibration/pointing support, fruit loops, mapmaking, coadd, Wiener/filtering, noise products, and science outputs |

Mode-inapplicable controls do not appear in the normal user override example.
Repository-managed base files may still carry complete low-level defaults when
Citlali's schema requires them.

## Implementation Sequence

1. Inventory the effective merged YAML for the four accepted validation
   profiles and classify every authored leaf by file role.
2. Build the four directory templates without changing effective values.
3. Add a merge/inspection tool that reports source file, final value, and
   overridden value for each authored leaf.
4. Add hermetic fixtures for precedence, list replacement, null/deletion,
   aliases, unknown keys, multiple reduction steps, and expert overrides.
5. Prove each template expands to the accepted low-level config, ignoring only
   TolTECA-owned runtime paths, or record any intentional successor change.
6. Run one real TolTECA smoke reduction per mode using the new files and apply
   the existing mode validation profile.

## Implementation Status - 2026-07-16

The Citlali-owned portion is implemented under `config/tolteca/`:

- all four mode kits are generated from the named accepted Phase 4 low-level
  baselines and pinned by baseline and normalized-policy SHA-256 identities;
- `tolteca_mode_kit.py` reproduces TolTECA/Tollan ordered recursive updates,
  numeric list indexing, and list-slice operations without a TolTECA runtime
  dependency;
- strict validation checks file roles, reduction type, the checked config-leaf
  contract, and exact accepted policy identity;
- deployed-project inspection allows deliberate policy drift while retaining
  unknown-key and reduction-mode failures and reporting final source,
  authority, owner, and expert overrides;
- the checked leaf contract now uses the four kits as hermetic observed
  sources and covers 576 leaves, including two previously omitted science
  cleaner-grouping leaves; and
- the full config preflight passes 107 focused tests, all four kit identities,
  all eight compact-compatibility cases, 100% compact-surface coverage, and all
  typed-boundary audits.

TolPROJ integration is implemented at commit `a33d26a`. Existing behavior
remains the default. The vendored, hash-checked kits are instantiated only when
the operator selects `--refactor`; this path generates
`72_observation.yaml` and selects the refactor executable while the existing
`70_reduce.yaml`/`72_reduce.yaml` path remains unchanged. Unknown numbered
overlays and mixed config families fail before installation. Same-mode,
same-kit reruns preserve operator-owned `71_runtime.yaml` and
`90_user_overrides.yaml`; mode or kit changes require a fresh directory. All
96 TolPROJ tests pass, including legacy-default, four-mode installation,
reinstallation, conflict, CLI-help, scannum, and Beammap calibration coverage.
Phase 4.1 remains open only until point, OOF, Beammap, and science smoke
reductions pass with the new files.

## Exit Gate

Phase 4.1 is complete when:

- all four ready-to-copy mode directories exist and share the same file-role
  convention;
- a user can identify where to edit deployment, observation/calibration,
  outputs, and expert overrides without reading the full low-level tree;
- the hermetic overlay tests cover ordering, precedence, lists, null/deletion,
  aliases, unknown keys, multiple steps, and expert overrides;
- the generated low-level YAML is exact against the accepted profile unless a
  change is explicitly entered in the science-change ledger;
- point, OOF, Beammap, and science TolTECA smoke runs complete with requested
  products and no unexpected errors; and
- the README explains installation, editing, inspection, and restoration of a
  canonical mode template.

This closes retained-debt item D09. Later scientific development may version
the mode policy or product profile; it does not silently edit an accepted
template in place.
