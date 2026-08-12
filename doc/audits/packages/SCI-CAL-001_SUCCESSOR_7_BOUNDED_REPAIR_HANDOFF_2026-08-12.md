# SCI-CAL-001 successor-7 bounded technical repair handoff

Date: 2026-08-12

Handoff ID: `SCI-CAL-001-SUCCESSOR-7-REPAIR-DISPATCH-001`

Status: frozen owner-authorized scope; role-separated repair not launched

## Exact authority and proposed base

- Owner acceptance:
  [`SCI-CAL-001_SUCCESSOR_6_OWNER_ACCEPTANCE_2026-08-12.md`](SCI-CAL-001_SUCCESSOR_6_OWNER_ACCEPTANCE_2026-08-12.md),
  SHA-256
  `84a0bdc4a867a079a13db83200475c13237a9befaefde1ecec75165b2d9f0092`.
- Immutable re-audit report:
  [`../SCI-CAL-001_SUCCESSOR_6_REAUDIT_2026-08-12.md`](../SCI-CAL-001_SUCCESSOR_6_REAUDIT_2026-08-12.md),
  SHA-256
  `9f9305d3324f67dbcb4bac6a510115dc9c1108be7d7cc2ddd376e724653cfbbd`.
- Immutable local evidence:
  [`../evidence/SCI-CAL-001_SUCCESSOR_6_LOCAL_EVIDENCE_2026-08-12.yaml`](../evidence/SCI-CAL-001_SUCCESSOR_6_LOCAL_EVIDENCE_2026-08-12.yaml),
  SHA-256
  `0953e3030a27c2b3d6bd7c623413d4f4290b5d1e7ca2c7ca9608c2a8e10ab9be`.
- Machine-readable finding ledger:
  [`../proposals/SCI-CAL-001_SUCCESSOR_7_REPAIR_FINDING_LEDGER_2026-08-12.yaml`](../proposals/SCI-CAL-001_SUCCESSOR_7_REPAIR_FINDING_LEDGER_2026-08-12.yaml).

The exact pushed application base is
`211e2f16f6354609de3ce6c6ee526d8aa4c6c59c` on
`origin/codex/repair-sci-cal-001-successor-6`:

- parent: `5dfc414a13fe69e6b063608906d87e3b30491ec7`;
- tree: `5ed203711ad5242aafb029373b416afdd1232081`; and
- parent-to-candidate binary-patch SHA-256:
  `2ff5928b48c516e70500e6f3190366e4e05243a3ba6327e18cad3251e1858019`.

The proposed repair branch is
`codex/repair-sci-cal-001-successor-7`. It was absent locally, in
remote-tracking state, and at live origin at the coordination checkpoint.
This handoff does not create the branch, worktree, or repair task.

## Frozen scope and retained states

Successor-7 is limited to F007, F008, and the local implementation portion of
F009. F002, F003, F004, F005, and F006 retain their accepted bounded
closures. F001 and F010 remain open, conditioned, and outside repair. The CAL
axes remain `approved`, `nonconformant`, `in_progress`, and `fail_closed`,
with verdict `amend`, until a later independent re-audit and owner
disposition.

No new CAL science, calibration arithmetic, RTC/PTC/mapmaking numerical
behavior, uncertainty/covariance product, empirical response-fidelity claim,
cross-package architecture, global transaction/rollback layer, accepted-run
rewrite, or production promotion is authorized.

## Mandatory READY checkpoint before edits

A later role-separated repair task must start from the exact pushed base in a
fresh clean worktree and return before editing with:

1. exact local and live-origin base/ref, parent, tree, patch digest, proposed
   branch, and clean worktree/index state;
2. confirmation that the successor-7 branch was absent before creation;
3. finding-to-change and finding-to-test traceability within the exact
   22-path ceiling below;
4. the first viable F007 identity-preimage tests and the scope checkpoint
   before publication, consumer, profile, or broad-test expansion;
5. preservation evidence for F002/F003/F004/F005/F006 and confirmation that
   F001/F010 remain conditioned and outside scope;
6. confirmation that historical readiness and profile/run authorities remain
   immutable and that `sig2noise_pixel_I` stays prohibited for new/current
   products; and
7. confirmation that no RTC numerical behavior, global transaction,
   production promotion, or new accepted profile/run is proposed.

Any path or behavior outside this handoff requires a stop and separate
coordinator/owner review. Silence prohibits a capability.

## Exact initial changed-path ceiling — 22 paths

F007 response-identity preimages:

- `include/citlali/core/pipeline/calibration_product_admission.h`
- `tests/test_science_map_fits_products.cpp`

F008 per-artifact publication:

- `include/citlali/core/engine/detail/beammap_apt_table_output_impl.h`
- `include/citlali/core/engine/detail/beammap_map_product_writers_impl.h`
- `include/citlali/core/engine/detail/lali_output_impl.h`
- `include/citlali/core/engine/detail/map_phdu_output_helpers.h`
- `include/citlali/core/pipeline/atomic_yaml_output.h`
- `include/citlali/core/pipeline/raw_timestream_provenance.h`
- `include/citlali/core/pipeline/reduction_observation_pipeline.h`
- `include/citlali/core/utils/ecsv_io.h`
- `include/citlali/core/utils/fits_io.h`

The F007 C++ test path may also serve F008 but counts only once.

Local F009 authoritative inputs, joins, fixtures, and profile restoration:

- `include/citlali/core/pipeline/raw_timestream_execution_plan.h`
- `include/citlali/core/pipeline/raw_timestream_observation_shadow.h`
- `include/citlali/core/pipeline/raw_timestream_provenance_lifecycle.h`
- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/examples/sci_cal_001_raw_timestream_provenance_v4.yaml`
- `tools/baseline/examples/sci_cal_001_selected_calibration_apt.ecsv`
- `tools/baseline/test_audit_reduction_run.py`
- `tools/baseline/test_compare_reduction_products.py`
- `tools/baseline/test_validation_profiles.py`
- `validation/validation_profiles.json`

The F007 admission path and F008 provenance path may also serve F009 but count
only once.

Coherent repair handback only:

- `doc/REFACTOR_STATUS.md`

These groups contain exactly 22 unique paths and all exist at the exact
candidate. No build-system path or new file is authorized. A demonstrated
need outside the ceiling stops the task before editing that path.

## F007 — response identity only

Complete the realized-response identity preimage by binding the
coefficient-defining application sample rate for the realized FIR and
IIR-highpass stages and by binding reduced-observation identity in every
canonical actual-notch serialization. Requested, effective, and realized
states remain distinct, and only executed stages are represented as realized.

Focused deterministic tests must show that equal frequency settings at
different sample rates produce distinct FIR and IIR-highpass response
identities and that otherwise equal actual-notch records from different
reduced observations remain distinct. Preserve dormant-stage semantics and
all existing requested/effective/realized distinctions.

F007 authorizes no RTC coefficient calculation, filter/notch selection,
filtering, flagging, cadence, time-grid, signal, or transfer behavior change.
Source traces through the unchanged coefficient implementations are evidence;
those implementation files are not in the edit ceiling.

## F008 — mandatory per-artifact publication

For the canonical calibration YAML, CAL-linked FITS, and Beammap ECSV, the
required output route must stage, synchronize, close, reopen, validate the
complete artifact and CALID/PKGID join, and atomically replace the final path.
Publication failure must leave no newly represented complete artifact and
must not destroy a previously valid final artifact before replacement is
ready. The canonical package remains ordered before dependent products.

Focused tests cover canonical-YAML parse and identity recomputation before
dependent publication; FITS late-HDU/write/close/reopen failures; complete
PHDU join readback; Beammap ECSV structural and metadata readback; interruption
and replacement of an existing valid final; cleanup of stages; and regression
of the already staged TOD publication route.

Per-artifact atomicity is mandatory. No global cross-output transaction,
rollback architecture, or change to scientific output content is authorized.

## Local F009 — authoritative inputs, ownership, and profile restoration

Parse the exact package-local selected-APT ECSV and bind its detector/factor
basis to the serialized factor state so the accepted audit's self-consistent
`flxscale` forgery fails even when the sibling digest remains unchanged.
Serialize an authoritative requested-configuration preimage and independently
recompute its requested-response identity. Bind package observation identity
to the owning observation directory. Verify every material observation versus
realized calibration identity/state join, not only CALID and PKGID.

Focused tests cover the demonstrated factor forgery; selected-APT row/value
tampering; requested-preimage mutation and digest mismatch; copying a valid
package to a different observation directory; mismatches in factor state,
response, raw identity, target unit, schema, and correction state; valid
legacy optional-modern lineage; distinct production-shaped multi-observation
packages; and supported uncalibrated v4.

Restore exactly these seven pre-existing validation-profile snapshots by
removing the unauthorized broad `selected_calibration_apt.ecsv` basename
exclusion and comparing them against their predecessor authority:

- `phase4-point-152389-v1`;
- `phase4-oof-152385-152387-v1`;
- `phase4-beammap-148670-v1`;
- `phase5-point-152389-v2`;
- `phase5-oof-152385-152387-v2`;
- `sci-map-001-point-152389-v1`; and
- `sci-map-001-oof-152385-152387-v1`.

Existing accepted profiles, contracts, runs, and historical snapshots may not
be rewritten or weakened. A genuinely new profile or contract must remain a
successor under the existing `preparing` current-production authority, must
have an exact in-scope need, and requires a stop before any path outside this
ceiling is edited. No profile or epoch is promoted.

## Historical readiness and preservation gates

`phase5_readiness.py --verify-fixtures` remains
`failed_owner_waived_never_passed`, solely for six point and twelve science
errors from immutable historical FITS products containing the prohibited
`sig2noise_pixel_I` extension. The waiver covers only that pre-existing
historical drift. It does not admit the extension for any new/current product.

The following exact objects remain byte-immutable:

- `validation/accepted_runs.json`, SHA-256
  `4a134dcdd14e0444d96875547f628a3353574cc66574dd9a559bcf59dafb94bb`;
- `validation/phase5_validation_readiness.json`, SHA-256
  `b9daf6ab3973d2d35968ab27d2b7c75eca8534d2baeb6af9bb43725261f04755`;
- `tools/baseline/phase5_readiness.py`, SHA-256
  `3a27fa5279c75432aa0939cbcc2add2db4d30df92379f6bf511ff281202b2af7`.

Historical product contracts and declared outcomes remain unchanged. The v4
current-production candidate epoch, contracts, and profiles remain
`preparing`; accepted-run records are not rewritten.

## Validation and handback

After the focused F007/F008/F009 fixtures pass, run the preserved F002--F006
controls, full CTest, configuration preflight, baseline Python suites,
validation-profile listing, validation and science-change ledgers, raw
execution census, atmosphere generator, ordinary readiness, and all other
applicable deterministic authority gates. Record disabled and skipped tests.
The historical write-producing fixture gate retains its exact failed/waived
truth and is never relabeled pass.

Return the exact repair commit, parent, tree, standard binary-patch digest,
changed paths, commands/results/skips, finding traceability, and unresolved
dependencies. Stop for coordinator review. Do not start a re-audit, merge,
push, access or request Unity, run a reduction, contact external parties, or
launch downstream work.

## Non-authorization

This handoff does not launch repair. It authorizes no application edit by this
coordination task, RTC/PTC/mapmaking numerical change, science/arithmetic
redesign, production promotion, accepted-evidence rewrite, cross-package
expansion, global transaction architecture, Unity access or request,
reduction, external contact, downstream launch, re-audit, merge, or push.
