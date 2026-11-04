# Configured POINT qualification preparation: 152389

Status: owner-run preflight completed and reviewed; single-observation reduction
ready for Grant to invoke, with reduction/scientific results still pending.
Grant executes every build, test, probe, reduction and validation runner.
Codex performed file/Git inspection, preparation and static review only.

## Bound purpose and source

### Preflight assessment — 2026-11-04

Grant's attempt `preflight-152389-20261104T150300.353722Z` completed native
version and configured-input preparation with exit 0 and no warnings/errors in
either log. Receipt and log hashes match their recorded identities. Current
source remains clean at `19b6fc462`; native, helper/config and preserved
environment hashes match the bound packet. Static rehashing of all 39 external
input files found no differences. The new output root does not yet exist.

The prepared observation is 152389, source **1146+399**, with all three arrays
and networks 0,1,2,3,4,5,7,8,9,11,12. Effective settings match the selected
weighted recipe below, with coadd and the weight-comparison arm disabled.
Pointing metadata is available; supplied reduction offsets are zero. Recorded
telescope User/Paddle metadata is preserved separately and is not relabeled as
zero. Preflight did not execute RTC, CAL, cleaning, mapping or fitting.

Detailed receipt SHA-256:
`c9b8b700dd2e7336c14c15a907d4acd651b26b8774e14e37be62bb4796866a89`.
The packet's `PREFLIGHT_REVIEW-152389.json` records the evidence and separate
scientific, architectural and hygiene dispositions. Ready for the existing
`owner_run.py run 152389` command below; reduction/product/scientific assessment
remains pending. No change to the prepared runner, config or binding was needed.

On 2026-11-04 Grant selected **152389 only**, preserving its completed weighted
POINT fixture's settings: all three arrays, 2 arcsec pixels, coverage cut 0.15,
ranks 3/4/6, zero applied az/alt offsets, optical-response threshold 0.10,
three upstream workers and at most ten feedback applications after M0.
Normal source-guard termination and intermediate-retention false remain intact.
The earlier draft for 152418 is retained unused; it is not in the active batch.

Documentation preparation starts at clean
`5dc124d1acf8997434f9b75133ac3d7f6e9e86cf` on `codex/refactor-mainline`.
Concurrent documentation-only team approval at
`2ae76819848852a041ca2bb07937be04852a6a25` was inspected and preserved before
committing this preparation. It changes neither the pinned application nor the
scope of this run. The approved strategy still awaits bounded implementation.
Application source remains accepted
`19b6fc462930af6f0a7bf2ca5b5350dd3c72af6e`, tree
`3b1d9e0ae067e502faa1e02895be3695f5166a66`, in the preserved stage-3 checkout.
Use the matching stage-3 Mac Spack binary, SHA-256
`40cf99ab50f3d65f3a37b878ea08e28d39d1f535e9ca8c339ebcf356b0c7358a`.
No branch/worktree, application, installed runtime or dependency is changed.

Governance read: Engineering Governance, Timestream Successor Governance, and
Review and Conformance, effective through the integration ledger. This is
bounded command/configuration preparation using existing scientific settings
and the preserved environment. It makes no conformance conclusion. Future
scientific qualification needs its scoped evidence, independent review and
owner disposition. The [qualification plan](POINT_SCIENCE_QUALIFICATION_PLAN_2026-11-03.md)
owns sequencing; [POINT method/use direction](POINT_METHOD_AND_USE_ACCEPTANCE_2026-11-04.md)
owns the new policy and unresolved acceptance questions.

## Prepared packet and first command

Packet:
`/Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-point-qualification-20261104`.
Its README explains the command boundaries and later reduction step.
`BINDING.json` SHA-256:
`fe94f367dbdc12c71f97dc18fb22561a83da6537e58beb0553be13ebb1ab2dd6`.

Run only preflight first:

```bash
"$HOME/tolteca/bin/python" -B /Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-point-qualification-20261104/owner_run.py preflight 152389
```

This checks source, executable version, preserved environment, configuration,
raw/telescope metadata, compact-v2 APT and KMP associations and file hashes.
It invokes the existing configured input-preparation module but no scientific
processing stages. Fresh logs and native input requests are written within the
packet. Return the printed preflight directory for review before reduction.

After preflight assessment, the separate owner command is:

```bash
"$HOME/tolteca/bin/python" -B /Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-point-qualification-20261104/owner_run.py run 152389
```

The only change to the frozen fixture YAML is its output directory. Its
original launch directory is preserved for relative KMP lookup. All outputs
go into a new packet-owned root, and an existing root prevents execution.
Failed attempts are retained; there is no automatic retry or batch continuation.
Preflight binds external inputs, and run checks those identities again before
and after processing. Source, config and native identity, basic all-array
product completeness and explicit error-level messages are checked. Full
product/log review remains necessary even if these checks succeed.

## Evidence boundaries and review checklist

The original merged YAML hash matches its completed historical receipt:
`a13a6630e748e1cee7d79331bf36a1df1500fb90785b97a15066dd06c5af499d`.
That old run used application `331ff29ea` and a non-Spack dependency
realization. It is a comparison reference, not same-source regression evidence.
Do not require bitwise equality or treat an unexplained difference as acceptable.
Report changed settings/implementations/environments explicitly when interpreting
differences. No new numerical tolerance is selected by this preparation.

Review the exact received/effective/realized configuration; input and observation
identity; published Alt/Az frame, units, validity and empirical weights; each
array's fit or explicit unavailable/failure state; feedback termination and
budget; required products; table eligibility or explicit withholding; and all
logs, including warnings and native receipts. A complete map can coexist with
an unavailable pointing fit or withheld table; report these separately.

The new array-selection rules, beam override, observer report and
TolTECA adaptation are not implemented in this source. This run cannot qualify
those policies or establish the approximately 1.5 arcsec combined-error target.
It also does not establish flux-calibration eligibility, settle broad-source
recovery, qualify SCIENCE, or authorize production. The retained SCIENCE driver
remains on hold.

Preparation dispositions:

- Scientific/behavioral: existing weighted POINT recipe preserved; observation
  selected; no new policy or accuracy claim; all runtime evidence pending.
- Architecture/ownership: application boundaries unchanged. The external wrapper
  owns only invocation, input/source guards and evidence capture. Existing POINT
  code owns configuration interpretation, scientific processing and publication.
- Repository/evidence: clean exact documentation base inspected; source and
  native hash pinned; original configs/results preserved; only output path
  changed in the copied selected YAML. No executable checks, builds, tests,
  reductions, Unity access, pushes, runtime replacement or cleanup by Codex.
