# Citlali collaborator candidate — current state

As of 2026-10-29. This page distinguishes the collaborator target, implemented
engineering candidate, and accepted operational evidence. Historical records
remain authoritative for their exact scope and source identities.

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
subsequent exact approval and removal of 33 missing registrations. All 20
existing worktrees, 104 branches, three stashes and recovery artifacts were
verified intact. Further cleanup requires its own exact approval.

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

## Working-folder consolidation preparation — 2026-10-29

Grant authorized safe preparation toward five core working folders, preserving
pointing and recovery evidence, with small approval batches before deletion.
The [consolidation record](REPOSITORY_CONSOLIDATION_2026-10-29.md) names all
20 folders, verified independent backups, dependencies and the first one-folder
proposal. No existing folder or branch has been deleted under this new request.
The pointing chat is active; inventory and backup cutoffs are recorded separately.
Current engineering/pointing builds still use the older science checkout’s
dependency sources, so that folder is retained.
