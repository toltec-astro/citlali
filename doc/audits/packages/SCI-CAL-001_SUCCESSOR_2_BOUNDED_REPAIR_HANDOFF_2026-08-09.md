# SCI-CAL-001 successor-2 bounded repair handoff

Date: 2026-08-09

Handoff ID: `SCI-CAL-001-SUCCESSOR-2-REPAIR-DISPATCH-READY-001`

Status: prepared for coordinator verification and owner launch approval; repair
not authorized or launched

## Exact authority and proposed base

- Owner disposition record:
  `doc/audits/packages/SCI-CAL-001_SUCCESSOR_2_OWNER_DISPOSITIONS_2026-08-09.md`
- Owner disposition SHA-256:
  `f0e0500e0ba809c1b51a36f69a97a71ab980d66337f62a9ff6985309b43df1d6`
- Immutable exact-repair re-audit report:
  `doc/audits/SCI-CAL-001_EXACT_REPAIR_REAUDIT_2026-08-08.md`, SHA-256
  `7a9eeae603871f3e2c157b123c15970dd2b2e472257479d100b02bea43101d34`.
- Immutable re-audit ledger proposal:
  `doc/audits/proposals/SCI-CAL-001_REAUDIT_LEDGER_PROPOSAL_2026-08-08.yaml`,
  SHA-256
  `47a63a5c2a2fcc1000547dd5cdc64d24382818666e299b6629e92afff28e9ee2`.

Proposed exact application base:
`7894346a91fa78ceb2a8b3d625335f466e5e1756`.

- Parent: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.
- Tree: `991f96c64e4d2d973ed5fc02630bfe29149109d9`.
- Verified source ref at preparation:
  `origin/codex/repair-sci-cal-001-successor`.
- Proposed repair branch: `codex/repair-sci-cal-001-successor-2`.
- Proposed fresh worktree:
  `/private/tmp/citlali-repair-sci-cal-001-successor-2`.

This base is rejected as complete CAL closure but proposed because it contains
the retained, re-audited F002 fixed atmosphere operator. Selecting it does not
accept its other changes, implementation status, validation, or production
state. The owner launch must explicitly confirm this base before any branch or
worktree is created.

The following exact successor material must be preserved unless a regression
forces a coordinator stop:

- operator contract SHA-256:
  `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a`;
- operator nodes SHA-256:
  `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f`;
- generated header SHA-256:
  `d322bdc863ccb1292325c739865f772ef53f4e9f4101967752027ea0a2413262`;
- operator ID: `am12_fixed_djf25_piecewise_linear_los_tau_v1`.

## Mandatory initial scope checkpoint

If launched, the repairer must first verify the exact base/ref/object,
fresh-worktree path, branch, parent/tree, and clean state, then return this
checkpoint before editing:

- exact files proposed for change from the initial allowlist below;
- finding-to-file and finding-to-test mapping;
- whether any new persisted field or file is required and its exact existing
  owner/consumer;
- confirmation that no new uncertainty product, APT extension, per-product
  factor-table duplication, response algorithm, or scientific estimator is
  proposed;
- local Citlali science reduction: prohibited;
- Unity request/access: prohibited in the repair task;
- delegation or independent review: prohibited unless separately approved;
- first viable artifact: focused deterministic F003 failure-boundary and F005
  existing-recipient transfer fixtures; and
- next return: after those fixtures and before writer/provenance expansion or
  broad test execution.

Silence prohibits a capability. Any needed path, product, schema, helper,
uncertainty object, response computation, or cross-package expansion outside
this handoff requires a stop and separate coordinator/owner decision.

## Initial implementation path allowlist

The repair may propose changes only within these exact existing paths at the
initial checkpoint:

Configuration and admission:

- `include/citlali/core/config/calibration_config.h`
- `include/citlali/core/config/calibration_config_validation.h`
- `include/citlali/core/config/reduction_config_validation.h`
- `include/citlali/core/pipeline/calibration_config_read.h`
- `include/citlali/core/engine/detail/citlali_config_impl.h`
- `include/citlali/core/engine/detail/observation_setup_impl.h`

Calibration application and realized state:

- `include/citlali/core/timestream/rtc/calibrate.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`
- `include/citlali/core/pipeline/raw_timestream_provenance.h`
- `include/citlali/core/pipeline/raw_timestream_provenance_lifecycle.h`
- `include/citlali/core/pipeline/raw_timestream_observation_resolution.h`
- `include/citlali/core/pipeline/raw_timestream_execution_plan.h`

Existing product metadata/writer boundaries:

- `include/citlali/core/engine/detail/map_phdu_output_helpers.h`
- `include/citlali/core/engine/detail/beammap_setup_metadata_impl.h`

Focused existing tests and deterministic contract tools:

- `tests/test_calibration_atmosphere_operator.cpp`
- `tests/test_config_scaffold.cpp`
- `tests/test_science_map_fits_products.cpp`
- `tests/CMakeLists.txt`
- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/test_audit_reduction_run.py`
- `tools/config/config_leaf_contract.yaml`
- `tools/config/config_leaf_contract_resolved.json`
- `tools/config/config_key_classification.yaml`
- `tools/config/config_authority_inventory.json`
- `include/citlali/core/pipeline/config_leaf_schema_generated.h`

Status documentation may be updated only at coherent repair handback:
`doc/REFACTOR_STATUS.md`.

No new source, test, schema, or output-format path is authorized by this
prepared handoff. If the canonical package-lineage record cannot be realized
within an existing owner above, return one exact minimal path proposal and stop
before creating it.

## Finding-to-change/test traceability

| Finding | Bounded implementation change | Required focused evidence | Explicit exclusion |
| --- | --- | --- | --- |
| F001 | No estimator/response expansion. Preserve admitted sample factor semantics and interfaces used by in-scope recipients. | Existing scalar/operator regression only; record dependency state. | Kernel/response redesign, new downstream weight product, ALIGN/AST science, Unity, astronomical standard, empirical response fidelity. |
| F002 | Preserve the exact fixed operator and prevent regression. | Generator check; exact nodes/endpoints/monotonicity/seam and low-opacity regression. | Atmosphere model redesign, truth/fidelity claim, new operator/domain. |
| F003 | Reject unsupported requested units at configuration/startup; reject malformed/incompatible exact inputs before calibration; preserve approved RTC-only/uncalibrated skips; emit typed persistent cause and zero scientific output. Admit the complete calibration state atomically before mutation/publication. | Unsupported-unit startup fixture; APT/raw mismatch/reorder/missing/duplicate fixtures; invalid-factor matrix; engine/CLI cause propagation; zero-writer/output assertion. | Requiring a preconstructed `CalibrationProduct` for every earlier failure; late unit rejection; relabeling skips as calibrated. |
| F004 | Consume, validate, and propagate existing legacy ECSV and available TolAPT manifest/row lineage; retain selected-source association and validity; prove verified-row or explicit-key binding. | Realistic legacy-header and modern-manifest fixtures; row permutation, missing/duplicate/mismatch, unavailable optional-detail, source-association/eligibility round trip. | New lineage system, duplicated APT extensions, perfect design identity requirement, invented optional provenance. |
| F005 | Scale each existing affected conditional variance by `a^2` and inverse-variance weight by `1/a^2`, coupled to the same valid factor/support/stage. Mark nuisance/total uncertainty unavailable. | Per-recipient signal/variance/weight fixture; omitted/duplicate factor; invalid/missing support; conditional-versus-total metadata and reopen check. | New uncertainty/covariance product; calibration/atmosphere/beam/donor-target/common-mode/cross-detector covariance. |
| F006 | Enforce only the approved top-of-atmosphere `mJy/beam` configuration boundary through F003. | Supported `mJy/beam` admission and unsupported-unit startup rejection. | Implementing other units, extended-source, integrated-photometry, or temperature contracts. |
| F007 | Write one canonical package calibration-lineage record; copy the exact selected APT locally with ECSV metadata and digest; bind raw identity and factor definitions; add stable package/calibration joins and minimal product links. | Actual package writer/reopen; APT byte/digest check; exact raw/factor resolution; unique/missing/stale/cross-package join cases; reconstruct without per-product table duplication. | Duplicate APT extensions or full per-detector factor tables in every FITS/TOD/Beammap file; competing lineage authorities. |
| F008 | Preserve exact once-only factor composition and record selected APT beam, mapmaker/kernel class, and filtering state as realized response basis; route existing variance/weight through F005. | Composition reconstruction; omission/duplication/inversion/recipient mismatch; response-basis identity reopen; unavailable stronger-claim metadata. | Empirical response calculation/claim, MAP/BEAM work, total uncertainty, donor-target or common-mode covariance. |
| F009 | No additional scientific feature. Produce exact-candidate local validation handback after all bounded changes. | Focused fixtures, actual writer/reopen, full CTest, config preflight, applicable authority/baseline/product-contract gates; exact commands/results/digests/skips/errors. | Local science reduction; astronomical response-fidelity campaign; Unity request from repair task. |
| F010 | Implement only approved abstract input interfaces already required by F003/F004; retain dependency. | Deterministic interface/eligibility fixtures possible from local approved abstractions; record conditioning. | Choosing/rederiving ALIGN/AST semantics or absorbing their evidence work. |

## Package/product contract minimum

One canonical calibration-lineage record per coherent reduction package must
resolve:

- exact package/calibration identity;
- exact raw-observation identity;
- package-local exact selected APT path/digest and preserved ECSV lineage;
- selected-source association and applicable validity/eligibility;
- exact applied factor definitions and once-only composition;
- approved target unit and complete calibration validity;
- selected APT beam and realized mapmaker/kernel/filtering response basis;
- existing conditional variance/weight availability and transfer; and
- explicit unavailability of total/nuisance uncertainty and empirical response
  fidelity.

Each individual FITS, TOD, and Beammap product retains only its
package/calibration identity, calibration validity, target unit, and exact link
needed to resolve the canonical record. No product-local duplication of the
complete APT or per-detector factor table is required or authorized.

## Local validation and handback

Before repair handback, run only the local mechanical/implementation gates:

1. all focused successor-2 success/failure and writer/reopen fixtures;
2. fixed-operator generator check and focused operator tests;
3. full CTest, recording disabled tests and treating no required skip as pass;
4. `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all`;
5. applicable raw/config authority and interface synchronization checks;
6. applicable baseline and product-contract tests; and
7. `git diff --check`, exact changed-path inventory, and clean final worktree.

Return one exact repair commit/parent/tree, changed paths, artifact and evidence
digests, test commands/results/skips, finding traceability, and remaining
dependencies. Stop for coordinator review. Do not request Unity, prepare a
Unity runbook, start a re-audit, merge, or push.

After a later owner integration and push, the coordinator may separately
prepare a small exact-SHA human-run Unity operational-confirmation request.
Astronomical response-fidelity validation remains later MAP/BEAM work.

## Explicit exclusions

- no atmosphere-operator redesign or new scientific domain;
- no kernel, mapmaker, filter, response, ALIGN, AST, RTC, PTC, MAP, BEAM, or
  covariance algorithm change;
- no new unit beyond approved top-of-atmosphere `mJy/beam`;
- no new uncertainty, covariance, nuisance, response, or per-product lineage
  product;
- no duplicate APT extensions or per-detector factor tables in every product;
- no local science reduction, Unity request/access, or astronomical campaign;
- no production-status change, integration, merge, push, repair re-audit, or
  downstream launch; and
- no claim of atmosphere truth, absolute response fidelity, total calibrated
  uncertainty, precision/accuracy closure, or scientific production readiness.

## Launch and stop rule

Only the project owner may approve launch against the proposed base and
branch. A launch authorization must cite this exact handoff digest after
coordinator verification. Until then, do not create the branch/worktree or
modify application/config/test code.
