# SCI-RTC-001 learned-sampling Stage A successor repair handoff

Date: 2026-08-11

Handoff ID: `SCI-RTC-001-LEARNED-SAMPLING-STAGE-A-SUCCESSOR-REPAIR-READY-007`

Status: owner disposition accepted; successor repair authorized for a separate
High-effort task; implementation not yet approved past its mandatory READY
checkpoint

## 1. Exact immutable authority

- Application repair base:
  `66c96757164af2c83ee1449d00fea30d131a7e3f` (parent
  `6cbe119a59f8915c5aecf5eaf333425dd592993d`, tree
  `f7ec30c021f30101f453043cedbd0f6773763ff1`).
- Application patch SHA-256:
  `d0ce6490baad81e447f08dab04e434d771d0e83e5a6def57068c86743f1d6805`,
  using `git diff --binary --full-index HEAD^ HEAD | shasum -a 256`.
- Frozen predecessor repair authority:
  `3132d5d8c001ef32f185d4ece2038aa6d7ce1b5c`.
- Independent successor re-audit:
  `923eca42a8f892a18774df8f483dd78404807d59` (parent exact application
  base above, tree `17cc4d00b75ed06beecb64776c4d7b8fba4381ee`).
- Re-audit report object:
  `923eca42a8f892a18774df8f483dd78404807d59:handoff/RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REAUDIT_2026-08-10.md`,
  SHA-256
  `a4f5a662be4b4a7d9569152dd6220444b45eb490a523a74cf5367e639c987214`.
- Re-audit evidence object:
  `923eca42a8f892a18774df8f483dd78404807d59:handoff/RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REAUDIT_EVIDENCE_2026-08-10.md`,
  SHA-256
  `d368c9f3f699c60c1f27a33d53db1f64b2b932aaf03319969646abb6ffe6133f`.
- Re-audit findings object:
  `923eca42a8f892a18774df8f483dd78404807d59:handoff/RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REAUDIT_FINDINGS_2026-08-10.csv`,
  SHA-256
  `ef42bb65c1763f8453e9a26c0452cfcc3665234767f1b5176f8f1a8788459b9b`.
- Machine-readable successor ledger:
  `doc/audits/proposals/SCI-RTC-001_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REPAIR_FINDING_LEDGER_2026-08-11.yaml`.

The repair branch shall be
`codex/repair-rtc-learned-sampling-stage-a-successor-2`, created from the exact
application base above in a fresh worktree. The audit commit is evidence, not
an application base.

## 2. Supersession and preservation

This handoff is a narrow successor overlay on the frozen 2026-08-10 Stage A
handoff. It replaces only the portions contradicted by findings SRA-001 through
SRA-009 and the owner decisions recorded below. All unaffected scientific
authority and positive controls remain binding, including:

- fixed diffraction-limited Airy intensity FWHM values `4.66`, `5.94`, and
  `8.48 arcsec` for `a1100`, `a1400`, and `a2000`;
- non-HWPR eligible source-telescope `v95` using pre-interpolation source rows
  with `v >= 1 arcsec/s`, speeds above `3600 arcsec/s` invalid, and higher
  percentiles/max diagnostic only;
- exact physically derived candidate range, factor-1 floor, and no silent
  truncation;
- coherent phase-zero astronomical folding, factor-1 zero alias, and the
  observation's existing realized filter as a counterfactual rather than an
  `M`-specific filter design;
- analytical or deterministic bounded/adaptive characterization without
  ranking, recommendation, acceptance, or selection;
- no absolute stopband threshold and no Stage A use of the provisional future
  `epsilon_alias` target as a gate;
- observe-only behavior: no sample, flag, timestamp, RTC/PTC/map input, map,
  weight, filter, cadence, factor, or production-execution change; and
- no `ptcdiag`, Stage B, Unity, reduction, production authorization, BEAM, or
  downstream work.

The predecessor statement requiring a new requested/effective/realized HWPR
lifecycle inside Stage A is **withdrawn**. The predecessor permission to mutate
the shared `rtc_diagnostics` executable contract in place is **withdrawn**.

## 3. Accepted re-audit disposition

The owner accepts the independent `RETURN FOR REPAIR` disposition. The repair
must close all nine successor findings:

| Finding | Required closure |
| --- | --- |
| SRA-001 | Bind reset, motion-support capture, and diagnostic consumption to the same observation. |
| SRA-002 | Remove Stage A dependence on stale or inferred legacy HWPR state and enforce the unsupported capability boundary below. |
| SRA-003 | Derive context from the exact total-intensity RTC validity domain and report truthful exclusion categories. |
| SRA-004 | Bound actual factor/image/extrema/tap work and serialized storage before evaluation. |
| SRA-005 | Introduce a versioned successor `rtcdiag` contract/profile with executable conditional cardinality and complete identity requirements. |
| SRA-006 | Persist and validate the full 40-hex application commit. |
| SRA-007 | Replace helper-only or circular claims with independent production-boundary evidence. |
| SRA-008 | Preserve requested/effective/realized cadence separately and report deterministic consistency. |
| SRA-009 | Publish the required complete `rtcdiag` successor atomically without destructive replacement or post-publication append. |

## 4. Owner decisions

### 4.1 HWPR analysis remains unsupported

Citlali is currently authorized only for total-intensity analysis. HWPR analysis
is unsupported and deferred until total-intensity mapping has established the
required confidence. This repair shall not invent an observation-scoped HWPR
lifecycle, infer HWPR analysis from file or hardware presence, or claim HWPR
sampling support.

- A configuration that requires enabled HWPR/polarimetry analysis remains
  rejected before scientific execution by the existing capability boundary.
- If the Stage A diagnostic API is nevertheless presented with an explicitly
  HWPR-analysis-dependent state, it returns deterministic status
  `unsupported_hwpr` and emits no learned-sampling candidate metrics for that
  state.
- Supported total-intensity diagnostics must not depend on stale
  `calib.run_hwpr`, serialized legacy enum spellings, physical HWPR-file
  presence, or newly invented realized-HWPR state.
- No HWPR correction, modulation transfer, phase convention, candidate range,
  product, or validation claim is authorized.

This decision closes the SRA-002 architecture choice without enabling HWPR.

### 4.2 Exact total-intensity context

Stage A complete-context accounting must consume exactly the total-intensity
RTC domain admitted after the applicable science flags, finite-value checks,
motion eligibility, realized filter guards, factor, phase, boundaries, and
per-detector validity. Motion eligibility alone is not complete context.

The implementation must report counts and fractions separately for at least:

- fully supported outputs;
- boundary/context exclusions;
- internal gaps;
- low-velocity motion exclusions;
- invalid/over-limit motion;
- science-flag exclusions;
- non-finite input exclusions;
- realized filter-guard exclusions; and
- any remaining category, with no silently unused `other` bucket.

Aggregation must preserve scan and array identity and derive each category from
the production total-intensity RTC conventions. Per-metric availability remains
independent: failure of one metric does not invalidate unrelated diagnostics.
The predecessor `N_full == 0` applicability meaning remains unchanged; Stage A
reports it but does not alter production flags or mapmaking inputs.

### 4.3 Versioned successor `rtcdiag` contract

The learned-sampling diagnostic must use a new versioned successor executable
contract and a corresponding preparing validation profile. It must not mutate
the existing `rtc_diagnostics` check in place. Existing consumers migrate
explicitly; backward-compatibility aliases are not required for the new Stage A
fields.

When a candidate table is declared available, the executable contract must
conditionally require and validate:

- the candidate dimension and complete candidate rows;
- declared count, dimension, and every per-candidate array cardinality;
- factor, phase, status, reason, and independent per-metric validity;
- requested/effective/realized cadence and filter identity;
- exact realized FIR coefficient vector and digest;
- exact observation/scan/array and source-support identity;
- exact 40-hex Citlali commit and canonical raw-input-manifest join; and
- all compact setup/method state required to recreate the run and diagnostic
  calculation.

Integer NetCDF scalar conditions must be normalized or otherwise handled
truthfully by the validator. Malformed available and unavailable products are
required adversarial fixtures. The successor remains a diagnostic product, not
an exhaustive intermediate replay.

### 4.4 Cadence provenance and calculation authority

Persist three distinct values:

1. requested cadence from the observation request;
2. effective cadence resolved by configuration; and
3. realized cadence measured from the native time grid.

Valid Stage A calculations use realized cadence. Requested-to-effective and
effective-to-realized consistency statuses are deterministic and separately
reported; a mismatch is never hidden by overwriting one state with another. A
mismatch does not prevent diagnostics that truthfully describe valid realized
data. Missing, non-finite, non-positive, or otherwise invalid realized cadence
makes dependent metrics unavailable with a cause-specific reason.

### 4.5 Atomic successor publication

Keep the successor `rtcdiag` self-contained where practical. Construct all
metadata and candidate tables in one temporary artifact, complete and validate
it, close/sync it, and then atomically replace the final file. Never delete the
previous valid regular file before replacement. Never append to a product after
publication.

If an auxiliary artifact is unavoidable, publish generation-specific artifacts
and commit one manifest or pointer last. Any create, write, sync, validation,
close, replacement, append-equivalent, or provenance failure must:

- propagate as a required-output failure;
- leave no newly advertised partial generation;
- preserve the previous complete generation, if one existed; and
- remove only task-created temporary artifacts.

The owner explicitly approves the bounded path-scope expansion needed in the
shared NetCDF atomic helper and the existing ordered rtcdiag append owner. This
does not authorize unrelated writer refactoring.

## 5. Technical closure requirements

The remaining design is technical and may be approved by the coordinator at
READY checkpoints without further scientific decisions:

1. reset and recapture the typed pre-interpolation motion carrier at the same
   production observation boundary that consumes it;
2. use overflow-checked resource bounds that account for each factor, every
   folded image, extrema/adaptive evaluations, all FIR taps, candidate rows,
   coefficient vectors, and serialized auxiliary state;
3. reject with `candidate_range_resource_limit` before partial evaluation when
   the full derived range exceeds the technical guard; retain the derived
   `Mmax` and never truncate;
4. publish a full 40-hex commit from a bounded build/source identity owner;
5. use independent analytical fixtures rather than the candidate's own helper
   to create expected numerical answers;
6. execute real production-boundary A/B, B/A, sequential, failed/partial,
   repeated, and permitted parallel/OpenMP tests;
7. run writer/reopen/executable-validator tests for complete, unavailable, and
   adversarial malformed successor products; and
8. inject every required-output failure stage against absent and pre-existing
   valid final products.

The repair may restructure cold diagnostic/product code where required. It may
not broadly rewrite the mature RTC numerical operator or add cross-cutting
public state to `Engine`.

## 6. Mandatory READY checkpoint

Before editing, the separate repair task must return `READY` with:

- exact live origin/local base, parent, tree, patch digest, branch, fresh
  worktree, and clean state;
- proof that the new repair branch did not pre-exist;
- a bounded changed-path proposal and finding/decision-to-path traceability;
- the exact observation owner for reset/capture/consume;
- proof that supported total-intensity routing uses no stale or inferred HWPR
  state, plus the exact pre-execution capability gate and diagnostic
  `unsupported_hwpr` seam;
- exact total-intensity flag, detector-validity, finite-value, motion, and
  realized-guard authorities used for complete context;
- overflow-safe CPU/evaluation/storage resource formulas and configured
  technical guard;
- the new successor product-contract ID, schema version, validation-profile ID,
  conditional cardinality design, and complete stable reason vocabulary;
- the exact full-commit build/source identity path;
- the complete temporary-to-final atomic lifecycle and failure matrix; and
- production-boundary and deterministic gate traceability.

The coordinator may approve continuation only when this checkpoint is exact,
bounded, and introduces no new scientific choice.

## 7. Initial path envelope

The task may propose only paths needed by the nine findings. The expected
envelope is:

- the existing Stage A diagnostic/math/state/writer headers changed by the
  predecessor candidate;
- observation input/setup boundaries needed for SRA-001;
- the mature RTC header only for read-only validity/guard exposure and atomic
  rtcdiag finalization, not numerical redesign;
- `include/citlali/core/utils/netcdf_io.h` for bounded replacement atomicity;
- top-level build/version identity only if needed to supply the full 40-hex
  commit;
- `tests/test_rtc_learned_sampling_metrics.cpp`, relevant ordered-writer or
  production-boundary tests, and `tests/CMakeLists.txt` only as needed;
- `validation/product_contracts.json`,
  `validation/validation_profiles.json`, and the product-contract validator and
  its tests for the versioned successor route;
- Stage A documentation/example and `doc/REFACTOR_STATUS.md` for truthful
  handback status.

No file is approved merely by appearing in this envelope. The task must name
each exact path and why it is the smallest owner before editing. Any other path
or any architecture/science conflict returns to the owner.

## 8. Validation and handback

After one approved READY checkpoint and one first-viable-artifact technical
checkpoint, the task may run deterministic local validation only:

- independent focused numerical and resource tests;
- production observation-lifecycle and context tests;
- successor writer/reopen/validator and atomic failure-injection tests;
- exact enabled/disabled Stage A non-interference checks;
- full enabled CTest with disabled/skipped tests reported;
- complete config preflight;
- baseline, validation-ledger, intended-science-change, validation-profile,
  authority, product-contract, raw-execution-census, and session-exit gates;
- exact changed-path and `ptcdiag` blob inventories;
- `git diff --check`; and
- clean committed state.

Return one exact repair commit, parent, tree, patch digest, branch, changed
paths, all gate commands/results, resource/cardinality evidence, successor
schema/profile/contract identity, stable reasons, and confirmation that Stage A
remains observe-only. Stop for coordinator verification and owner push. The
task must not push, merge, integrate, launch a re-audit, access/request Unity,
run a reduction, enter Stage B, contact external parties, authorize production,
or launch downstream work.
