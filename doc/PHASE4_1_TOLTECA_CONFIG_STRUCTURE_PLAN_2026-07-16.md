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

## Original V1 Proposal

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

The implemented V1 filenames are `70_pipeline.yaml`, `71_runtime.yaml`,
`72_observation.yaml`, `80_products.yaml`, and `90_user_overrides.yaml`. V1 is
retained as a mechanically exact reference. Project-owner review subsequently
found that its generic names and large base policy did not make routine versus
expert editing approachable enough. The accepted V2 structure below supersedes
V1 for operator use.

## Accepted V2 Structure

Each mode has seven mode-named files under `config/tolteca/v2/MODE/`:

| File role | Owner and normal use |
| --- | --- |
| `60_MODE_internal_policy.yaml` | Generated complete accepted policy; Citlali maintainers only |
| `71_MODE_runtime.yaml` | Executable, thread count, output layout, and verbosity |
| `72_MODE_observation.yaml` | TolPROJ-generated data, observation, APT, flux, and pointing binding |
| `81_MODE_defaults.yaml` | Routine mode-specific analysis choices |
| `82_MODE_products.yaml` | Mode-appropriate requested and retained products |
| `90_MODE_advanced_overrides.yaml` | Optional supported controls beyond the short routine surface |
| `99_MODE_expert_overrides.yaml` | Deliberate implementation or diagnostic overrides requiring validation rationale |

Point uses `pointing` in its filenames. TolPROJ refreshes the generated policy
and observation files on a same-kit setup and preserves the five operator-owned
files. V2 remains ordinary TolTECA YAML; no runtime translator is introduced.

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

## Implementation Status - 2026-07-17

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
Subsequent project-owner review rejected V1 as the final operator interface.
The V2 pattern was first reviewed with science and is now implemented for all
four modes. The generator produces the same seven roles for pointing, OOF,
Beammap, and science, with mode-specific routine analysis and product surfaces.
Every unchanged V2 directory merges exactly to its accepted V1 policy hash:

- point: 445 leaves, `f2d124d40ac7ad9e6351a647253050e5146659c666feed07262125e6fa5415c8`;
- OOF: 444 leaves, `414a5d16ceba8b6f9163851c139affa486f50397e1ead56fc480ed53475b76f4`;
- Beammap: 485 leaves, `75eaf79fb5ce45b383f48bbb6a4715209fbb25cb29a1ac595a3afb2a7df4e0b0`; and
- science: 404 leaves, `10095418b09100f15c90af173ee34ea7bfcf12260cec41d80f43f6f50473a347`.

Hermetic tests enforce exact identities, user/expert classification, bounded
file size, disjoint analysis/product ownership, mode-inapplicable exclusions,
consolidated fruit-loop controls, TolPROJ-owned data binding, and byte-for-byte
generator reproducibility. The full config preflight passes 116 focused tests,
all four exact mode identities, and every config-authority audit. Citlali
commit `6b6be9f57` is vendored byte-for-byte by TolPROJ commit `8490f09`.
TolPROJ's manifest-driven `--refactor` path installs all four V2 modes,
preserves the legacy default, preserves all five operator files on same-kit
reruns, and passes all 100 tests. Phase 4.1 remains open only until point, OOF,
Beammap, and science smoke reductions pass with the installed V2 files.

TolPROJ commit `39f724d` adds the reproducible four-mode acceptance-suite layer
above those installers. The canonical suite contains one portable
`suite.yaml` plus human-readable point, OOF, Beammap, and science
`project.yaml` files. Cluster-specific executable, data, APT, prior, and Slurm
paths are resolved from the normal TolPROJ site config. The installer creates
one refactor-only workspace, runs `tolteca setup`, installs the verified V2
mode kits, generates observation overlays and Slurm helpers, and records exact
source, kit, site, config, and resource identities in `suite.lock.yaml`.
Mode-aware verification permits point, OOF, and Beammap readiness to be checked
independently while correctly holding science until the suite's eight pointing
support products exist. It refuses in-place suite, site, or config drift. The
suite does not create an OG baseline, copy data, submit jobs, or run Citlali.
All 103 TolPROJ tests pass. The remaining gate is a fresh Unity installation
followed by successful point, OOF, Beammap, and science reductions and their
existing product-profile comparisons.

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
