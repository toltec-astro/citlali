# SCI-CAL-001 successor-5 bounded technical repair handoff

Date: 2026-08-11

Handoff ID: `SCI-CAL-001-SUCCESSOR-5-REPAIR-DISPATCH-001`

Status: frozen owner-authorized scope; role-separated repair not launched

## Exact authority and proposed base

- Owner acceptance:
  [`SCI-CAL-001_SUCCESSOR_4_OWNER_ACCEPTANCE_2026-08-11.md`](SCI-CAL-001_SUCCESSOR_4_OWNER_ACCEPTANCE_2026-08-11.md),
  SHA-256
  `d01d329d08e3622f7e2329e1f889a19f47d0b24858453b2e60ed85e6b8bbce2d`.
- Immutable re-audit report:
  [`../SCI-CAL-001_SUCCESSOR_4_REAUDIT_2026-08-11.md`](../SCI-CAL-001_SUCCESSOR_4_REAUDIT_2026-08-11.md),
  SHA-256
  `47a7bc888a2c5b1981e96287b46c20392146f073f1cdb5cc092e842ddeeb9d9c`.
- Immutable local evidence:
  [`../evidence/SCI-CAL-001_SUCCESSOR_4_LOCAL_EVIDENCE_2026-08-11.yaml`](../evidence/SCI-CAL-001_SUCCESSOR_4_LOCAL_EVIDENCE_2026-08-11.yaml),
  SHA-256
  `597f8fe2c9106e287cc1eda9e341ff8c50d0d90d151add8cf703904cbfae44a0`.
- Machine-readable finding ledger:
  [`../proposals/SCI-CAL-001_SUCCESSOR_5_REPAIR_FINDING_LEDGER_2026-08-11.yaml`](../proposals/SCI-CAL-001_SUCCESSOR_5_REPAIR_FINDING_LEDGER_2026-08-11.yaml).

The exact pushed application base is
`693f1b107855e3ae9b36617323ca14aac868f304` on
`origin/codex/repair-sci-cal-001-successor-4`:

- parent: `3af6faf996fa002b2647adca8f33991002d49ff1`;
- tree: `fb317a7862ff474c118d229bb45320adb560b3bc`; and
- parent-to-candidate binary-patch SHA-256:
  `59ba71493377630be5bff5164d779aa43d14ee1e788c002707f1f4fe62d5902d`.

The proposed repair branch is
`codex/repair-sci-cal-001-successor-5`. It was absent locally, in
remote-tracking state, and at live origin when this handoff was frozen. This
document does not create the branch, worktree, or task.

## Frozen scope and retained states

Successor-5 is limited to F005, F007, F008, and the local implementation
portion of F009. F002, F003, F004, and F006 retain their accepted bounded
closures. F001 and F010 remain open, conditioned, and outside repair. The CAL
axes remain `approved`, `nonconformant`, `in_progress`, and `fail_closed`,
with verdict `amend`, until a later independent re-audit and owner disposition.

No new CAL science, estimator, weighting or mapmaking design, uncertainty or
covariance product, empirical response-fidelity claim, heterogeneous coadd
membership schema, or global cross-output transaction architecture is
authorized.

## Mandatory READY checkpoint before edits

A later role-separated repair task must start from the exact pushed base in a
fresh clean worktree and return before editing with:

1. exact local and live-origin base/ref, parent, tree, patch digest, proposed
   branch, and clean worktree/index state;
2. confirmation that the successor-5 branch was absent before creation;
3. finding-to-file and finding-to-test traceability within the exact 27-path
   ceiling below;
4. the exact first viable F005 production-route test and the scope checkpoint
   before metadata, writer, profile, or broad-test expansion;
5. preservation evidence for F002/F003/F004/F006 and confirmation that F001
   and F010 remain outside scope;
6. confirmation that no source APT/sensitivity mutation, new product,
   scientific choice, heterogeneous coadd schema, global transaction layer,
   or cross-package change is proposed; and
7. confirmation that local science reductions and Unity access or requests
   are prohibited.

Any path or behavior outside this handoff requires a stop and separate
coordinator/owner review. Silence prohibits a capability.

## Exact initial changed-path ceiling — 27 paths

F005 production correction, admission, state, and tests:

- `include/citlali/core/pipeline/flxscale_correction.h`
- `include/citlali/core/pipeline/initial_observation_setup.h`
- `include/citlali/core/pipeline/reduction_observation_calibration.h`
- `include/citlali/core/pipeline/calibration_product_admission.h`
- `include/citlali/core/timestream/calibration_product.h`
- `tests/test_calibration_product.cpp`
- `tests/test_science_map_fits_products.cpp`

F007/F008 applied-state, lifecycle, joins, and publication:

- `include/citlali/core/engine/detail/beammap_setup_metadata_impl.h`
- `include/citlali/core/engine/detail/map_phdu_output_helpers.h`
- `include/citlali/core/engine/detail/ptc_line_audit_impl.h`
- `include/citlali/core/pipeline/raw_timestream_execution_plan.h`
- `include/citlali/core/pipeline/raw_timestream_observation_shadow.h`
- `include/citlali/core/pipeline/raw_timestream_provenance.h`
- `include/citlali/core/pipeline/raw_timestream_provenance_lifecycle.h`
- `include/citlali/core/pipeline/reduction_observation_pipeline.h`
- `include/citlali/core/pipeline/tod_metadata_mapmaker_tau.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`

The shared admission, product-state, and two C++ test paths listed under F005
may also serve F007/F008 but count only once.

Local F009 executable audit, fixtures, contracts, profiles, and baselines:

- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/examples/sci_cal_001_raw_timestream_provenance_v4.yaml`
- `tools/baseline/examples/sci_cal_001_selected_calibration_apt.ecsv`
- `tools/baseline/test_audit_reduction_run.py`
- `tools/baseline/test_compare_reduction_products.py`
- `tools/baseline/test_validate_product_contract.py`
- `tools/baseline/test_validation_profiles.py`
- `validation/product_contracts.json`
- `validation/validation_profiles.json`

Coherent repair handback only:

- `doc/REFACTOR_STATUS.md`

These groups contain exactly 27 unique paths. No build-system path or new file
is authorized. A demonstrated need outside the ceiling stops the task before
editing that path.

## F005 — explicit observation correction

Keep the source APT and its source sensitivity immutable. Carry the
per-observation flxscale correction `a` as explicit applied calibration state.
Apply `a` exactly once to calibrated samples and exactly once as
`W' = W/a^2` in the approximate baseline inherited by approximate, hybrid,
and validated modes. Preserve full-weight behavior.

Persist truthful applied-factor and recipient provenance. Focused evidence
must execute the real production correction path through nonzero map
realizations and `noise_variance_I`; manually scaling a compatibility factor
or testing only a scalar helper is insufficient.

Do not mutate source APT fields or redesign calibration arithmetic, weighting,
variance, or mapmaking.

## F007 — actual applied state and joins

Record fixed, shared, and detector notch applications at the actual RTC/PTC
application point. Each record carries RTC-versus-PTC phase, scan, PTC
iteration, model-subtraction state, scope, detector/ordinal, geometry, and
phase convention. Immutable requested state is distinct from actual effective
and realized state, and only executed stages are labelled realized.

Carry finalized CALID/PKGID joins through supported TOD-only operation. A
coadd may publish one CALID only when all contributing observations have the
same CALID. Differing contributing CALIDs fail closed. Do not introduce a
heterogeneous coadd-membership schema.

Focused evidence covers fixed/shared/detector RTC and PTC applications,
mutable-versus-immutable state separation, TOD-only writer/reopen joins,
homogeneous multi-observation coadds, and heterogeneous fail-closed behavior.

## F008 — observation lifecycle and publication order

Applied-response history is observation-owned and resets at observation
boundaries. Repeated finalization either rejects or idempotently preserves the
immutable consumed snapshot and original CALID/PKGID. Cover interrupted,
unavailable, reused-scan-number, and multiscan lifecycles.

Publish and validate the canonical calibration package before dependent linked
products. Preserve atomic publication per artifact. An orphan canonical
package after a later dependent-output failure is acceptable; an unresolved
linked product is not. Do not add a global transaction or rollback layer.

Focused evidence must cover repeat finalization, observation reuse and
failure boundaries, package-first ordering, dependent-output failure, and the
absence of unresolved linked products.

## Local F009 — per-observation layout and integrity

Publish, contract, and validate `{obs}/selected_calibration_apt.ecsv` once per
calibrated observation. Lineage and member requirements are conditional on
effective calibration. Supported uncalibrated v4 remains valid without that
member.

Validation is path-aware, hashes the actual sibling member, requires canonical
lineage schema and complete components, verifies source/component/package
digest joins, and recomputes package identity. Synchronize v4, product
contracts, profiles, baselines, and production-shaped fixtures.

Focused tests cover actual single- and multi-observation layouts; missing,
tampered, stale, conflicting, and forged members or identities; supported
uncalibrated v4; package and dependent-output partial failures; and complete
digest-join and package-identity recomputation.

Preserve v1-v3 compatibility unless an exact contradiction is demonstrated;
if one is found, stop before changing compatibility policy.

## Preservation, validation, and handback

The successor reruns the accepted F002 atmosphere-operator, F003 startup and
admission, F004 APT lineage/association, and F006 `mJy/beam` boundary controls
without broadening their claims. It then runs:

1. every focused successor-5 fixture above;
2. actual package, map FITS, TOD NetCDF, and Beammap ECSV writer/reopen paths;
3. focused tests and full CTest, recording disabled and skipped tests;
4. `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all`;
5. applicable authority, raw-execution, baseline, product-contract, profile,
   validation-ledger, and science-change-ledger gates; and
6. `git diff --check`, exact changed-path inventory, and clean final state.

Return the exact repair commit, parent, tree, standard binary-patch digest,
changed paths, artifact digests, commands/results/skips, finding traceability,
and remaining dependencies. Stop for coordinator review. Do not start a
re-audit, merge, push, request Unity work, run a reduction, or launch
downstream activity.

## Non-authorization

This handoff does not launch repair. It authorizes no application edit by this
coordination task, CAL scientific expansion, production change, RTC/PTC/MAP/
BEAM/TEL/ALIGN scientific change, Unity access or request, reduction, external
contact, downstream launch, re-audit, merge, or push.
