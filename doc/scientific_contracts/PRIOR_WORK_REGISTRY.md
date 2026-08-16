# Prior-Work Discovery Registry

Status: discovery seed; re-verify at the start of every package

This registry prevents repeated searches and forgotten scientific work. It is
not itself scientific authority, an author packet, or a substitute for a
package-specific [`PRIOR_WORK.md`](templates/PRIOR_WORK.md). Topic branches and
historical audits may contain later or more detailed reasoning than the
integrated application line, but their status must be classified before use.

## Reference Snapshot

The initial recovery used these exact repository references:

| Shorthand | Reference | Role |
| --- | --- | --- |
| `MAIN` | `origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20` | Current integrated application/documentation discovery line at recovery time |
| `COORD` | `codex/register-sci-map-003-audit-disposition@8c581bfb26f01b187f4f1e0565f4457bcc25f099` | Later coordination decisions, package ledger, and handoff registry |
| `FRAME` | `codex/scientific-audit-framework@dd5894679bf12bf4a5fb551e871b3c6010ef9b9b` | Historical scientific-audit framework and early package corpus |
| `METHOD` | `codex/science-doc-framework@4a7916a8ec459f050de236211e5bacfc95695412` | Reusable scientific-method documentation framework |

These references are serial numbers for recovery, not permanent authority.
Each package must check the living status, relevant current branch, and later
owner decisions before adopting material.

## Program-Level Recovery Anchors

- `COORD:doc/audits/audit-ledger.yaml`
- `COORD:doc/audits/packages/SCIENTIFIC_AUDIT_PROGRAM_CHECKPOINT_2026-08-08.md`
- `FRAME:doc/audits/README.md`
- `FRAME:doc/audits/AUDIT_MANAGER_INSTRUCTIONS.md`
- `METHOD:doc/science/README.md`
- `MAIN:doc/ARCHITECTURE.md`
- `MAIN:doc/SCIENTIFIC_CONVENTIONS.md`
- `MAIN:validation/product_contracts.json`
- `MAIN:tools/config/config_leaf_contract_resolved.json`

The audit framework is historical process and evidence, not the governing
process for this library. Its package inventory, independent cores, decisions,
and cross-package handoffs are valuable recovery inputs.

## Package-Family Seed Map

### CAL

- Package-specific recovery:
  [`packages/SCI-CAL/v0.1/PRIOR_WORK.md`](packages/SCI-CAL/v0.1/PRIOR_WORK.md)
  at `2026-08-16`. It records the later layered-identity supersession,
  accuracy hierarchy, passband decision, opacity quality policy, and the
  distinction between the retained structural atmosphere operator and its
  unestablished physical/observational authority.
- Current shared authority: `MAIN:doc/SCIENTIFIC_CONVENTIONS.md`,
  `MAIN:validation/product_contracts.json`, and
  `MAIN:tools/config/config_leaf_contract_resolved.json`.
- Owner/coordination decisions to reconcile: `COORD:doc/audits/packages/`
  entries beginning `SCI-CAL-001_`, especially the coordinator decision,
  opacity, APT identity, accuracy/model-scope, atmosphere, and successor
  owner-acceptance records.
- Earlier derivation and audit evidence:
  `codex/audit-sci-cal-001@27b0916e725696597c3ba84fb6a82bf6cf0ea356:doc/audits/packages/SCI-CAL-001_INDEPENDENT_CORE.tex`
  and `SCI-CAL-001_SCIENTIFIC_CONTRACT_AUDIT.tex` beside it.
- Scope material: `MAIN:doc/astrometry_photometry_config_transition.md` and
  the later APT end-to-end and detector-identity audit branches.

### MAP And Coaddition

- Current shared authority plus
  `MAIN:doc/adr/0009-science-map-bundle-admission-and-validity.md`.
- Approved/integration decisions:
  `MAIN:handoff/SCI-MAP-001_APPLICATION_INTEGRATION_DECISION_2026-08-05.md`
  and the later MAP-002/MAP-003 decision records under `COORD`.
- Earlier independent cores and audits on `codex/audit-sci-map-001`,
  `codex/audit-sci-map-002`, and `codex/audit-sci-map-003` under
  `doc/audits/packages/`.
- Reusable method note:
  `codex/doc-map-001-phase-a@abf3ba2fd15e5941c571f41d495a58c78d6adf63:doc/science/SCI-METHOD-WEIGHTED-MAP-NORMALIZATION-001.md`.
- Scope maps: `MAIN:doc/ANALYSIS_FLOW_RAW_TO_SCIENCE_PRODUCTS_2026-07-01.md`
  and `MAIN:doc/MAPMAKING_CONFIG_AUTHORITY.md`.

### Beammap

- Current shared authority plus `MAIN:doc/BEAMMAP_CONFIG_AUTHORITY.md` and
  `MAIN:handoff/BEAMMAP_AUTHORITY_DESIGN_REVIEW_2026-07-14.md`.
- Package and dependency scope: `FRAME:doc/audits/audit-ledger.yaml` entry
  `SCI-BEAM-001` and `COORD:doc/audits/handoffs/SCI-BEAM-001/`.
- Historical-only evidence includes Beammap prior-audit and dated handoff
  material under `MAIN:handoff/`.
- No dedicated approved `SCI-BEAM-001` scientific contract was recovered in
  the initial search.

### RTC And PTC

- Current shared authority plus the RTC/PTC decisions and amendments under
  `COORD:doc/audits/packages/SCI-RTC-001_*` and `SCI-PTC-001_*`.
- Earlier independent cores and audits on `codex/audit-sci-rtc-001` and
  `codex/audit-sci-ptc-001` under `doc/audits/packages/`.
- Earlier scope/reference material: `MAIN:doc/RTC_FLAGGING_AUDIT_2026-03-16.md`
  and `MAIN:doc/PTC_MODEL_PROTECTED_NOTCH_PLAN_2026-05-21.md`.

### Alignment And Astrometry

- Current frame, indexing, and pointing-correction authority in
  `MAIN:doc/SCIENTIFIC_CONVENTIONS.md`.
- Owner/coordination decisions under
  `COORD:doc/audits/packages/SCI-ALIGN-001_*` and `SCI-AST-001_*`.
- Earlier independent cores and audits on `codex/audit-sci-align-001` and
  `codex/audit-sci-ast-001`.
- Scope material: `MAIN:doc/astrometry_photometry_config_transition.md`.
- Recovery must use frozen records only and must not absorb active ALIGN work.

### Noise And Filtering

- Current shared authority plus `MAIN:doc/NOISE_PRODUCTS_CONFIG_AUTHORITY.md`.
- Existing noise integration and filter-amendment decisions under `MAIN` and
  the `codex/coordinate-sci-flt-001-amendment` topic branch.
- Reusable scientific candidates:
  `MAIN:doc/citlali_noise_estimation_plan.tex`, earlier NOI-001/NOI-002
  independent cores, and
  `codex/convolve-contract-audit@800e8ae433f87d3fb7521fcb1a7fdf1d32532949:doc/CONVOLVE_SIGNAL_UNCERTAINTY_AND_RESPONSE_CONTRACT.tex`.
- The Convolve document is mixed: recover its independent mathematics, but
  exclude its source audit, candidate verdicts, repair requirements, and
  validation history from authorship.
- No dedicated approved Wiener/lowpass `SCI-FLT-002` contract was recovered.

### Source Fitting, Pointing, And OOF

- Current authority in `MAIN:doc/SCIENTIFIC_CONVENTIONS.md` and
  `MAIN:validation/product_contracts.json`.
- Scope records in the `SCI-SRC-001` and `SCI-MODE-001` ledger and handoff
  entries under `FRAME` and `COORD`.
- Earlier reference/evidence:
  `MAIN:handoff/HANDOFF_2026-06-18_POINTING_SOURCE_AWARE.md` and
  `MAIN:doc/POINTING_COMPACT_EQUIVALENCE_2026-06-30.md`.
- No dedicated approved `SCI-SRC-001` or `SCI-MODE-001` scientific contract
  was recovered.

### Fruit Loops

- Current iteration, checkpoint, significance, and morphology authority in
  `MAIN:doc/SCIENTIFIC_CONVENTIONS.md` and
  `MAIN:doc/adr/0006-fruit-loop-restart-checkpoint.md`.
- Reusable investigations under `MAIN:doc/` beginning
  `FRUIT_LOOP_FEEDBACK`, `FRUIT_LOOP_CALIBRATION_REFERENCE`,
  `FRUIT_LOOP_CONVERGENCE`, and `FRUIT_LOOP_POPULATION_EXTENSION`.
- Historical evidence under `MAIN:validation/fruit_loop_*` and the
  `codex/fruit-loop-calibration-reference` branch.
- Package/dependency scope in the `SCI-FRUIT-001` ledger and handoff records.
- No dedicated approved `SCI-FRUIT-001` contract or production stopping rule
  was recovered.

## Required Use

At package startup, copy relevant entries into the package's `PRIOR_WORK.md`,
resolve each exact reference, search for later material, classify every item,
and record its disposition. Do not pass this registry wholesale to an author:
the approved author packet contains only sanitized scientific references named
in the Scope Brief.
