# Citlali combined development candidate — current state

As of 2026-10-30. This page distinguishes the collaborator target, implemented
engineering candidate, and accepted operational evidence. Historical records
remain authoritative for their exact scope and source identities.

## Current isolated preparation — 2026-10-30

Grant approved pointing cutoff `797d013f115e403a30663eb92254082e8e742826`
and the [acceptance table](../validation/pointing_engineering_candidate_2026-10-30/acceptance-table.json)
for one candidate combining engineering `5c46201057d6b6d1048aac2949a64e9a34e9bad3`.
The [current work order](POINTING_ENGINEERING_CANDIDATE_2026-10-30.md)
records exact scope, preservation, settings and held uses. Candidate assembly is
complete at implementation merge `c83375dc030095821999a0e0ff435f13bf82b914`, followed
by documentation/evidence closure. [Local validation](../validation/pointing_engineering_candidate_2026-10-30/LOCAL_VALIDATION.json)
passes: 1,388 enabled CTests, 207 baseline and 300 Python application tests, full
configuration preflight and supporting checks. One inherited test is disabled.
Independent source review has no findings. Supported Mac Spack qualification is
pending 44 dependency installations for the current OS; fallback checks remain
supplemental. Canonical remains `e32c4584dec07617d6caa857d39bf13047b6c4b0`.
No push or canonical movement is authorized. The active pointing workspace may
continue independently; this checkout never replaces its frozen runtime.

The older engineering identities below describe that lane's original delivery.
They do not identify the new combined executable or certify its validation.
Current scientific, supported-environment and production limitations remain.

## Identities and ownership

| Role | Exact identity | Disposition |
| --- | --- | --- |
| Canonical integration | `codex/refactor-mainline` at `e32c4584dec07617d6caa857d39bf13047b6c4b0` | Existing integration authority; unchanged by this work |
| Reviewed POINT candidate / engineering base | `77063575c4fac2bb207ca462435009358be396f6` | Historical engineering base; 262 commits beyond canonical; live pointing work continues separately |
| Frozen weighted POINT trials | `331ff29ea1720b1454e671cbbab93814d8df8321` | Preserved runtime and products; Grant reviews science elsewhere |
| Engineering implementation snapshot | `8c3b676ec9c3f5d8c7ccbb0f0f2ede97a959c983` on `codex/citlali-engineering-20261029` | Locally verified stages 1–4; subsequent documentation commits are recorded in Git; owner integration pending |

The engineering worktree is
`/Users/gwilson/.codex/worktrees/citlali-engineering-20261029/citlali-refactor`;
the shared store is `/Users/gwilson/GitHub/citlali/.git`. The directory named
`/Users/gwilson/GitHub/citlali-refactor` is an older, protected scientific
checkout. A folder name does not establish authority. Cached remote refs are
historical local information. Later read-only GitHub inspection confirmed the owner-pushed CI repair; this lane has not pushed.

Grant alone launches reductions, accepts scientific results, integrates
canonical, and pushes. This engineering lane does not modify the original
pointing checkout, frozen trials, reference reductions or scientific policies.

## Collaborator workflow and release boundary

The release target is POINT and SCIENCE on a1100, a1400 and a2000, using the
familiar TolProj project/configuration/validation/guarded-submission workflow
and TolTECA preparation/launch layer. Grant supplies beammap/APT products;
beammap production and OOF are outside this release target.

1. Prepare project inputs and matched/calibrated APT associations with TolProj.
   Its opt-in `--refactor` workflow preserves numbered TolTECA configuration.
2. Select an exact reviewed native executable and prepared Python runtime.
   See [Python deployment](../requirements/README.md). The copyable runtime
   launcher preserves TolProj snapshot semantics and verifies its bound files.
3. Select the explicit configured POINT overlay:
   [uniform v1](../data/configured_point_v1.yaml) or
   [weighted v2](../data/configured_point_v2.yaml). The v2 optical-response,
   weighting and FRUIT rules are the existing scientific candidate, unchanged
   by engineering cleanup. TolTECA supplies the merged low-level YAML.
4. Use existing configuration/input/resource checks and guarded submission.
   Grant launches only after those checks and the separate scientific decision.
   Repeated configured POINT calls reserve a fresh `reduNN` atomically.
5. Inspect complete required products and truthful unavailable measurements.
   Qualified POINT fit diagnostics do not by themselves establish the
   conventional calibration table or downstream calibration acceptance.

| Release requirement | Present evidence / remaining gate |
| --- | --- |
| All-array configured POINT | Implemented; local synthetic/engineering gates pass; current trial science reviewed in the original chat |
| Established pointing-derived calibration consumed by SCIENCE | Compatibility/owner acceptance remains open |
| All-array SCIENCE and genuine multi-observation coadds | Required release milestone; not established by configured POINT engineering tests |
| Existing downstream consumers | Product/units/layout qualification remains open |
| Native environment | Supported Spack profiles are `macos-llvm20` and `unity-gcc13`; this task's AppleClang/Homebrew checks are supplemental |
| Python deployment | Exact Mac recreation verified; target-platform qualification is stated in the deployment record |
| Production/release | No promotion or expanded authorization from this work |

## Engineering and integration

[Engineering work order and stage results](ENGINEERING_CLEANUP_2026-10-29.md)
record build parity, ordinary test coverage, publication/resource ownership,
shared input services, packaging and cleanup disposition. The configured POINT
entry remains explicit; the retained Engine and scientific algorithms are
unchanged. Source lists/strict-FP policy are shared by the dependency entry
points. Ordinary tests build through `citlali_test_binaries`; sanitizer and
observational suites remain separately selected.

Canonical is an ancestor of the reviewed candidate, but the 262-commit gap
contains application work, experimental controls and evidence. Proposed route:
review that bounded continuation against its recorded owner dispositions,
resolve release blockers in their scientific/application lane, then let Grant
accept an exact candidate for canonical integration. Prefer preserving accepted
ancestry if the whole continuation is accepted. If any unit is not accepted,
prepare a separately reviewed selective integration plan. Do not fast-forward
canonical, cherry-pick the engineering tip onto the old base, or infer scientific
acceptance merely from graph containment or passing tests.

[Owner-run target qualification](ENGINEERING_OWNER_QUALIFICATION_2026-10-29.md)
provides isolated Unity build/check instructions. The
[cleanup record](REPOSITORY_CLEANUP_PROPOSAL_2026-10-29.md) documents Grant's
subsequent exact approval and removal of 33 missing registrations. At that
checkpoint, all 20 existing worktrees, 104 branches, three stashes and recovery
artifacts were verified intact. The separately approved October 30 consolidation
is recorded below.

[CI](../.github/README.md) separates hosted source checks from a prepared native
Spack job. No CI step launches an observation. Required data cannot be skipped
and counted as successful qualification.

## Historical authority and evidence

- [Living chronological status](REFACTOR_STATUS.md) and
  [integration ledger](INTEGRATION_LEDGER.md).
- [Application landmarks and distinct acceptance axes](APPLICATION_BASELINES.md).
- [Architecture](ARCHITECTURE.md), [scientific conventions](SCIENTIFIC_CONVENTIONS.md),
  [retained debt](RETAINED_DEBT.md), [durable decisions](adr/README.md).
- [Engineering governance](governance/ENGINEERING_GOVERNANCE.md),
  [successor governance](governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md),
  [review/conformance](governance/REVIEW_AND_CONFORMANCE.md).
- [Configured POINT handoff](../handoff/CONFIGURABLE_POINT_LOCAL_RESULT_2026-10-28.md)
  and [frozen weighted trials](../handoff/WEIGHTED_POINT_PREPARED_2026-10-29.md).
- [Accepted Spack foundation](adr/0014-spack-build-foundation.md) and
  [V2 campaign binding](../validation/citlali_v2_spack_validation_authority.json).

The chronological August no-work-order/held descriptions record their historical
checkpoints. The September 30 standing owner amendment and the present bounded
engineering work order govern their own later scope; neither changes scientific
contracts or grants integration/production authority.

## Refactor working-folder consolidation completed — 2026-10-30

The [completed grouped cleanup](REPOSITORY_CONSOLIDATION_2026-10-30.md) retired 13
historical 2026-refactor working folders after full recovery checks. Seven folders
remain: five refactor workspaces and two protected legacy/performance workspaces.
At that checkpoint all 104 local branches, three stashes and original recovery
evidence remained. The later branch consolidation below supersedes that branch
count. 4.x/5.x source and refs were untouched. Current pointing and its pilot fallback
remain available; archived historical OOF checks/replays require exact-path
restoration. Physical consolidation does not change scientific or integration
acceptance. The October 29 one-folder approval proposal is superseded by Grant's
October 30 grouped-consolidation directive and completed operation record.

## Local branch consolidation completed — 2026-10-30

The [approved two-batch branch cleanup](BRANCH_CONSOLIDATION_2026-10-30.md) reduced
104 local branches to 41, then 10. Ninety-four historical names and 12 additional
commit roots remain preserved by 106 archive refs and an independently restored
backup. Seven working folders remain; pointing, 4.x/5.x, all stashes, old recovery
evidence and scientific status are protected. Historical branch-dependent
verifiers require a separate recovery repository with original names and exact
historical checkout state. GitHub cleanup is a separate owner-controlled next
step; the live remote audit and recommendation are in the operation record.
