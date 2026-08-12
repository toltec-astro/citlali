# SCI-CAL-001 successor-6 bounded technical repair handoff

Date: 2026-08-12

Handoff ID: `SCI-CAL-001-SUCCESSOR-6-REPAIR-DISPATCH-001`

Status: frozen owner-authorized scope; role-separated repair not launched

## Exact authority and proposed base

- Owner acceptance:
  [`SCI-CAL-001_SUCCESSOR_5_OWNER_ACCEPTANCE_2026-08-12.md`](SCI-CAL-001_SUCCESSOR_5_OWNER_ACCEPTANCE_2026-08-12.md),
  SHA-256
  `825f67f874f0ec444f2d3250b08174ef9c1b9ceaf35e87e8a33e236af991ea5f`.
- Immutable re-audit report:
  [`../SCI-CAL-001_SUCCESSOR_5_REAUDIT_2026-08-12.md`](../SCI-CAL-001_SUCCESSOR_5_REAUDIT_2026-08-12.md),
  SHA-256
  `7f3a484cf5d446647313659a3d6d3103805837ecbb3a9d77034d49bb5762234a`.
- Immutable local evidence:
  [`../evidence/SCI-CAL-001_SUCCESSOR_5_LOCAL_EVIDENCE_2026-08-12.yaml`](../evidence/SCI-CAL-001_SUCCESSOR_5_LOCAL_EVIDENCE_2026-08-12.yaml),
  SHA-256
  `e6d8d721c02d22683a0ca8500efcd66d1e53c00b85d9529f92bd1c7ccbc64206`.
- Machine-readable finding ledger:
  [`../proposals/SCI-CAL-001_SUCCESSOR_6_REPAIR_FINDING_LEDGER_2026-08-12.yaml`](../proposals/SCI-CAL-001_SUCCESSOR_6_REPAIR_FINDING_LEDGER_2026-08-12.yaml).

The exact pushed application base is
`5dfc414a13fe69e6b063608906d87e3b30491ec7` on
`origin/codex/repair-sci-cal-001-successor-5`:

- parent: `693f1b107855e3ae9b36617323ca14aac868f304`;
- tree: `72e4df08bc3677290b03d1c39457ea049f8db813`; and
- parent-to-candidate binary-patch SHA-256:
  `1c9e634c574da60c40cf7e2808b1ec1ac25d1fa8f80cd4de7cb31230365cf7d8`.

The proposed repair branch is
`codex/repair-sci-cal-001-successor-6`. It was absent locally and in
remote-tracking state at the coordination checkpoint. This handoff does not
create the branch, worktree, or repair task.

## Frozen scope and retained states

Successor-6 is limited to F005, F007, F008, and the local implementation
portion of F009. F002, F003, F004, and F006 retain their accepted bounded
closures. F001 and F010 remain open, conditioned, and outside repair. The CAL
axes remain `approved`, `nonconformant`, `in_progress`, and `fail_closed`,
with verdict `amend`, until later independent re-audit and owner disposition.

No new CAL science, calibration arithmetic, weighting or mapmaking design,
uncertainty/covariance product, empirical response-fidelity claim,
cross-package architecture, heterogeneous coadd membership schema, or global
cross-output transaction architecture is authorized.

## Mandatory READY checkpoint before edits

A later role-separated repair task must start from the exact pushed base in a
fresh clean worktree and return before editing with:

1. exact local and live-origin base/ref, parent, tree, patch digest, proposed
   branch, and clean worktree/index state;
2. confirmation that the successor-6 branch was absent before creation;
3. finding-to-change and finding-to-test traceability within the exact
   33-path ceiling below;
4. first viable production-path evidence for F005 and the scope checkpoint
   before writer, lifecycle, contract, profile, or broad-test expansion;
5. preservation evidence for F002/F003/F004/F006 and confirmation that F001
   and F010 remain conditioned and outside scope;
6. confirmation that per-artifact atomic publication remains mandatory; and
7. confirmation that the active v1--v3 contradiction is a stop gate, not an
   implementation choice.

Any path or behavior outside this handoff requires a stop and separate
coordinator/owner review. Silence prohibits a capability.

## Exact initial changed-path ceiling — 33 paths

F005 production correction, admission, state, and tests:

- `src/citlali/core/engine/calib.cpp`
- `include/citlali/core/pipeline/flxscale_correction.h`
- `include/citlali/core/pipeline/initial_observation_setup.h`
- `include/citlali/core/pipeline/reduction_observation_calibration.h`
- `include/citlali/core/pipeline/reduction_observation_inputs.h`
- `include/citlali/core/pipeline/calibration_product_admission.h`
- `include/citlali/core/timestream/calibration_product.h`
- `tests/test_calibration_product.cpp`
- `tests/test_science_map_fits_products.cpp`

F007/F008 applied-state identity, joins, lifecycle, and publication:

- `include/citlali/core/engine/detail/beammap_setup_impl.h`
- `include/citlali/core/engine/detail/beammap_setup_metadata_impl.h`
- `include/citlali/core/engine/detail/map_phdu_output_helpers.h`
- `include/citlali/core/engine/detail/tod_file_output_impl.h`
- `include/citlali/core/engine/detail/ptc_line_audit_impl.h`
- `include/citlali/core/pipeline/raw_observation_outputs.h`
- `include/citlali/core/pipeline/raw_timestream_execution_plan.h`
- `include/citlali/core/pipeline/raw_timestream_observation_shadow.h`
- `include/citlali/core/pipeline/raw_timestream_provenance.h`
- `include/citlali/core/pipeline/raw_timestream_provenance_lifecycle.h`
- `include/citlali/core/pipeline/reduction_observation_pipeline.h`
- `include/citlali/core/pipeline/tod_metadata_mapmaker_tau.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`

The admission, product-state, and C++ test paths in the F005 group may also
serve F007/F008 but count only once.

Local F009 executable audit, fixtures, contracts, profiles, and baselines:

- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/examples/sci_cal_001_raw_timestream_provenance_v4.yaml`
- `tools/baseline/examples/sci_cal_001_selected_calibration_apt.ecsv`
- `tools/baseline/test_audit_reduction_run.py`
- `tools/baseline/test_compare_reduction_products.py`
- `tools/baseline/test_validate_product_contract.py`
- `tools/baseline/test_validation_profiles.py`
- `tools/baseline/validate_product_contract.py`
- `validation/product_contracts.json`
- `validation/validation_profiles.json`

Coherent repair handback only:

- `doc/REFACTOR_STATUS.md`

These groups contain exactly 33 unique paths and all exist at the exact
candidate. No build-system path or new file is authorized. A demonstrated
need outside the ceiling stops the task before editing that path.

## F005 — production correction and truthful state

Make the production correction setup reachable in the correct observation
lifecycle. Preserve the source APT and source sensitivity as immutable.
Carry the per-observation correction as explicit applied state, reject
non-finite or non-positive composed factors before scientific mutation, and
prevent reuse across observation boundaries or duplicate application.
Generic fallback behavior must not mutate the source APT.

Persist the truthful applied scalar, its state, source/recipient identity, and
exactly-once application. Focused evidence executes corrected and uncorrected
production setup, extreme composition failure, observation reuse, supported
fallback, nonzero map realization, and `noise_variance_I` transfer. Preserve
the accepted approximate/hybrid/validated inverse-square transfer and full
weight behavior without redesign.

## F007 — immutable identities and observation-owned joins

Bind requested-state identity to immutable requested input, distinct from
effective and realized state. Complete operator identity with the application
sample rate and the already authorized fixed/shared/detector notch details.
Only executed stages are realized.

Carry finalized CALID/PKGID through TOD-only operation. Own joins by the
reduced observation and the actual contributing observations, not by a reused
selected-APT identity. A coadd may publish one CALID only when every
contributing observation has that CALID; differing CALIDs fail closed.
Fruit/iteration and observation boundaries reset or bind the applicable join
and response state without collision.

Focused evidence covers immutable-request mutation probes, equal geometry at
different sample rates, TOD-only writer/reopen, calibrator/science APT reuse,
homogeneous and heterogeneous coadds, repeated scan numbers, and fruit
iterations.

## F008 — finalization order and mandatory per-artifact atomicity

Resolve calibrated Beammap ordering so admission, finalized joins, canonical
package publication, and dependent metadata occur in a valid lifecycle.
Publish and validate the canonical package before dependent linked products.
Every dependent artifact must be staged and atomically published with its
complete CALID/PKGID link. An orphan canonical package after later failure is
acceptable; an unresolved or partially linked dependent product is not.

Per-artifact atomic publication is mandatory and may not be weakened. Do not
add a global cross-output transaction or rollback architecture. Reset join
ownership at observation boundaries and cover true unavailable, inactive,
interrupted, repeated-finalization, reused-scan, multiscan, and dependent
writer-failure cases.

## Local F009 — lineage integrity and mandatory authority stop

Accept valid legacy selected-APT lineage without inventing unavailable modern
manifest details. Reject impossible associations. Recompute serialized raw
acquisition, ordered-row, admitted-factor, response, CALID, component, and
package identities from their actual payloads and sibling bytes. Use
production-shaped single- and multi-observation fixtures with distinct
observation-owned identity.

The active v1--v3 contract contradiction is not delegated to implementation.
If accepted legacy behavior cannot coexist with the v4 requirement, the
repair must stop for owner choice among:

1. preserving the legacy epoch;
2. creating a successor contract/profile epoch; or
3. explicitly superseding the affected authority.

No active contract, profile, baseline, or accepted compatibility authority
may be silently weakened, broadened, or rewritten to avoid this stop.
Focused evidence covers legacy no-manifest and optional-modern lineage,
missing/tampered/stale/conflicting/forged identities, serialized-payload
mutation, distinct multi-observation packages, supported uncalibrated v4,
and every accepted active legacy baseline affected by the contract.

## Validation and handback

The successor reruns accepted F002/F003/F004/F006 controls without broadening
their claims. It then runs all focused fixtures above, actual package/map/TOD/
Beammap writer/reopen paths, focused tests, full CTest, configuration
preflight, and applicable authority/baseline/product-contract/profile/
validation-ledger gates. Record every disabled or skipped test.

Return the exact repair commit, parent, tree, standard binary-patch digest,
changed paths, commands/results/skips, finding traceability, and unresolved
dependencies. Stop for coordinator review. Do not start a re-audit, merge,
push, access or request Unity, run a reduction, or launch downstream work.

## Non-authorization

This handoff does not launch repair. It authorizes no application edit by this
coordination task, science/arithmetic redesign, production change,
cross-package expansion, RTC/PTC/MAP/BEAM/TEL/ALIGN change, Unity access or
request, reduction, external contact, downstream launch, re-audit, merge, or
push.
