# Individual POINT and celestial SCIENCE qualification

Prepared 2026-11-03; owner clarification on 2026-11-04 supersedes the earlier
two-route selection. Active scope: individual configured successor POINT now.
SCIENCE/RA-Dec/coadd/known-offset qualification is deferred until successor
SCIENCE is connected. The retained-driver build command is on hold; do not run
it for this plan. POINT observation/configuration binding remains pending.
No qualification runs have been assessed under this plan.

## Owner workflow clarification

Grant clarified that POINT is run on individual observations in delta-source
Alt/Az. Observations to be coadded are reduced as SCIENCE in RA/Dec. Applied
pointing offsets can then be checked against the source's known celestial
coordinates. A full POINT-coadd qualification is therefore not part of this
workflow. The accepted stage-1 invalid-fit handling for unfitted POINT/OOF
coadds remains a defensive software repair; it does not establish a supported
POINT-coadd observing workflow and should not be reverted.

This records intended use of the existing frame and astrometry contracts in
[SCIENTIFIC_CONVENTIONS](SCIENTIFIC_CONVENTIONS.md#map-frames). It changes no
algorithm, calibration policy, numerical default, or acceptance threshold.

## Preflight and exact scope

- Owner: Grant. He executes every build, test, executable probe, reduction and
  validation runner on Mac and Unity. Codex prepares and statically reviews
  commands and assesses owner-produced evidence. No push or direct Unity access.
- Documentation base: `45757b009f33aabfaf691c7ece133e98902f3a41`, branch
  `codex/refactor-mainline`, clean at inspection, in
  `/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor`.
  The November 4 route-selection and command-preparation update starts from
  clean documentation commit `e425ab715c044a99340941818f92a1009bf2e300` on that
  same branch. It adds a separate external owner-build packet; no application
  source, dependency realization or installed runtime is changed.
  The subsequent owner clarification starts from
  `e988f7be19891bb114d77e757bd756c568792ee0`; unrelated untracked
  `doc/POINT_METHOD_AND_USE_ACCEPTANCE_2026-11-04.md` and
  `doc/POINT_TEAM_SUMMARY_2026-11-04.md` are preserved untouched. This update
  changes documentation only and places the external build preparation on hold.
- Qualification application: accepted stage-3 source
  `19b6fc462930af6f0a7bf2ca5b5350dd3c72af6e`, tree
  `3b1d9e0ae067e502faa1e02895be3695f5166a66`. The later acceptance receipt
  changes documentation only. Bind Python and native code to this same source;
  do not follow a moving mainline or relabel an older executable.
- Read governance: `ENGINEERING_GOVERNANCE`, `TIMESTREAM_SUCCESSOR_GOVERNANCE`
  and `REVIEW_AND_CONFORMANCE`, with effectiveness recorded in
  [INTEGRATION_LEDGER](INTEGRATION_LEDGER.md). Current sequencing and scope
  come from `CURRENT_STATE` and `REFACTOR_STATUS`; scientific and architectural
  boundaries come from `SCIENTIFIC_CONVENTIONS`, `ARCHITECTURE`,
  `APPLICATION_BASELINES` and `RETAINED_DEBT`.
- This preparation is Tier 1 documentation of existing owner intent and a
  proposed qualification sequence. A scientific qualification conclusion needs
  the applicable Tier 2 evidence and independent exact-revision review. This
  document neither declares conformance nor authorizes missing implementation.
- Preserve repair worktrees, existing runs, original inputs, runtimes, backups,
  ongoing pointing work and all historical acceptance records. No application
  changes, dependency installation, runtime replacement or cleanup are included.

## Execution routes must remain explicit

| Route | What exists at the pinned application | Claim a run could support |
| --- | --- | --- |
| Configured POINT | Python `tools/timestream_successor/configured_point_cli.py --native ...` consumes one merged POINT request; the native services implement the selected stages | Configured successor POINT for the selected observations and settings |
| Native development CLI | Explicit `citlali-development-v1` request with bounded terminal stages | Only the named native stage/route, not an ordinary SCIENCE request |
| Retained standard SCIENCE | Retained standard reduction has RA/Dec and coadd controls; it is reached by a separate validation driver | Retained-standard SCIENCE/coadd behavior at its exact source/build |
| Connected successor SCIENCE | Current connected input/configuration, calibration propagation and genuine multi-observation celestial route remain unfinished | No executable qualification command is claimed here |

The configured POINT parser requires `delta-source-AltAz` and disables coadd.
Changing its YAML to SCIENCE does not connect a successor SCIENCE pipeline.
The native CLI also does not silently fall back to the retained driver.
See [current release boundaries](CURRENT_STATE.md#application-and-release-boundaries)
and [handoff scope](POINTING_HANDOFF_INTEGRATION_2026-10-31.md).

**Current owner selection, 2026-11-04:** after clarifying the recommendation,
Grant chose individual configured successor POINT now, with SCIENCE/coadd/offset
qualification deferred until successor SCIENCE is connected. This supersedes
the earlier same-day selection of configured POINT plus retained-standard
SCIENCE. The retained-driver packet is preserved on hold, not part of the
active qualification. It should be resumed only for a separately selected
reference/comparison purpose. Historical V2 SCIENCE acceptance remains valid
within its recorded mixed-revision operational scope.

## Bind the experiment before launch

Prepare one immutable manifest and separate fully merged YAML for every case.
For the active POINT step, bind the selected individual observations, raw/APT
and calibration inputs, POINT settings, outputs, budget and applicable
acceptance criteria. Prefer observations reusable for the later celestial
comparison, but do not make the deferred coadd or known-offset decisions a
prerequisite to qualifying individual POINT. The full eventual packet must name:

- At least two distinct observations of the selected source, raw/telescope
  inputs, compact-v2 APTs and calibration inputs, with file hashes. Repeating
  one observation does not demonstrate genuine multi-observation coadding.
- Source reference RA/Dec, reference frame/equinox, epoch and provenance.
  For a moving source, resolve time-dependent reference and alignment before
  launch; do not assume a common fixed sky position across observations.
- Explicit per-observation az/alt corrections in arcseconds, their provenance,
  and any MJD interpolation support. Keep the known correction separate from
  an optional deliberate perturbation. Do not select offset signs by looking
  at the desired result.
- Exact mode/route, arrays, cleaning/weighting/feedback settings and iteration
  budget, pixel scale, map center/extent, calibration and required products.
  Freeze the common SCIENCE WCS for comparing individual and coadded products.
  Existing general POINT defaults and frozen trial settings differ; neither
  silently becomes the chosen qualification configuration.
- New output roots, resource/retention budget, source/build/environment
  identities, launch directory and complete commands. Claim destinations
  before processing and preserve failed attempts rather than reusing them.
- Owner-approved acceptance criteria or named existing criteria for positional
  residuals, repeatability and other scientific comparisons. Until these are
  bound, measurements are descriptive, not scientific passes.

The source name, selected observations, coordinates, offset values and numerical
acceptance criteria are currently unbound. Local POINT fixtures including
152388/152389 exist, but have not been selected or asserted to satisfy the
source/reference requirements. GOODS-N compatibility YAML is not automatically
a compact-source astrometry fixture.

## Ordered qualification cases

Only steps 0 and 1 are active. Steps 2–4 describe the deferred successor SCIENCE
qualification; they are not instructions to launch retained-standard SCIENCE.
Connecting successor SCIENCE requires its own bounded implementation work and
review before those qualification commands can be prepared.

| Step | Case and purpose | Required evidence |
| --- | --- | --- |
| 0 | Freeze and preflight | Exact source, native hash/version, environment, input/APT/config identities, new output roots and fully resolved settings; no missing required inputs or substituted defaults |
| 1 | Individual configured POINT for each selected observation, no coadd | Complete run receipt; Alt/Az FITS axes/units; valid fit status and per-array offsets; requested/effective/realized settings and iteration outcome; required products present; zero unexpected error-level messages |
| 2 | Individual SCIENCE in RA/Dec using the same selected observations | Named SCIENCE route; celestial FITS WCS, units, calibration/validity and source positions; sufficient saved weight/coverage information to assess coadds; observation-specific astrometry provenance |
| 3 | Genuine multi-observation SCIENCE coadd | Distinct contributing observation identities; common WCS and map units; recorded weights, valid support and exclusions; agreement with the route's documented coadd operation using the individual contributions; no duplicated or stale observation state |
| 4 | Matched known-offset check, individual SCIENCE then coadd | Control and corrected cases differing only in declared astrometry; requested/effective/applied offsets per observation; measured celestial residuals to the independent reference and predicted signed change; corrected coadd retains registration |

Review each step before progressing. A zero process exit is necessary but is
not evidence that the right products, frame, offsets or coadd were realized.
Unexpected errors, missing products, invalid astrometry, changed inputs or a
source/build mismatch stop the sequence; preserve logs and return for review.

For step 4, use independently established corrections if the claim is recovery
of the known source position. Record the uncorrected residual and the corrected
residual for every observation and array before assessing the combined map.
An arbitrary extra offset should move the source by its predicted displacement;
it should not be required to leave the source at its known position. A deliberate
perturbation is an additional owner-selected treatment, not an implicit case.
POINT-derived offsets tested on the same source can show internal closure, but
do not by themselves establish independent astrometric accuracy.

Apply corrections through the existing astrometry input, before celestial
mapping. Az/alt offsets are not constant RA/Dec offsets. Use the accepted
coordinate transformation and the product WCS to predict and measure motion;
account for RA wrapping and tangent-plane angular distance. Do not infer signs
from image array order or use a raw RA difference as an angular separation.

The offset check is astrometric. It does not qualify POINT-derived flux
calibration, resolve unavailable calibration uncertainties, or justify using a
withheld already-corrected POINT table. Use the independently accepted APT and
calibration inputs selected for this experiment. Broad-source recovery and
calibration policies remain separately held.

## Owner-run command preparation

The existing stage-3 source and native executable can be reused without
rebuilding or replacing a runtime. These exact paths were statically checked:

```bash
QUAL_SOURCE='/Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-stage3-20261103/source'
QUAL_NATIVE='/Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-stage3-20261103/owner-run-20261103T205227.678366Z/build/citlali'
git -C "$QUAL_SOURCE" rev-parse HEAD
git -C "$QUAL_SOURCE" status --porcelain
shasum -a 256 "$QUAL_NATIVE"
```

Expected source: `19b6fc462930af6f0a7bf2ca5b5350dd3c72af6e`; status empty;
native SHA-256:
`40cf99ab50f3d65f3a37b878ea08e28d39d1f535e9ca8c339ebcf356b0c7358a`.
Stop on any difference. The existing owner version log reports `g19b6fc462`,
Clang 20.1.8, Spack DAG `kqk3pl4f6jgsb5yksobnynlx2gmnqks6`, unmanaged
deployment. This is Mac evidence, not Unity or a portable installed release.

The following is the verified **POINT invocation form, not a launch-ready
command**. `QUAL_POINT_YAML` must first name a reviewed, newly prepared merged
request whose outputs are in the new qualification packet. It must run inside
the bound Spack process environment, with logs/receipts captured:

```bash
"$HOME/tolteca/bin/python" -B \
  "$QUAL_SOURCE/tools/timestream_successor/configured_point_cli.py" \
  --native "$QUAL_NATIVE" "${QUAL_POINT_YAML:?Bind a reviewed merged POINT YAML first}"
```

Preparation must reuse and verify the preserved environment rather than invent
a new one: mainline `build/spack-candidate/environment/spack.lock` SHA-256
`00964e0c57c1df96b0ddc3f6d214bef1c1ab8f59cc4e5c4da653cefbcb915238`, and
`build/spack-candidate/process-environment.json` SHA-256
`255ae04aa1ae7991349caee609a34675221083055d767e25b57b4b69684c9c3e`.
Reuse the owner-run stage-3 environment setup, but not its old mainline-base
guard or its `all` command. A new final packet must bind the observations,
configs and environment and record a fresh native version check before launch.

### Retained SCIENCE driver build — preserved on hold, 2026-11-04

Do not run the previously supplied `owner_build_driver.py build` command for
this plan. It was prepared for the superseded two-route selection. Preserve
the packet at
`/Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-point-science-qualification-20261104`.
The build files and their binding remain unchanged; the hold is recorded in
its README and `ON_HOLD.md`. The description below is historical preparation,
not an active execution request.

The external packet contains `README.md`, `BUILD_BINDING.json`, the owner
wrapper and `driver/{CMakeLists.txt,driver.cpp}`. Binding-file SHA-256:
`e8bd1bd990bd2750ef57b46cebfb4b38bad89ffe7fd4b76c21ca9beecffbfe85`.
It records the prepared-file hashes and preserved environment digests above.
The wrapper checks source/tree and dependency identities before and after each
command, configures/builds in a new timestamped attempt directory, checks the
driver version, and records logs, receipts and the executable digest. The final
`DRIVER_BUILT.json` means commands completed with log assessment still pending;
it does not declare scientific acceptance. No tests or reductions are launched.

The driver source is byte-identical to the preserved
`citlali-beammap-iteration-20261101/driver/driver.cpp` under the October 29
visualizations tree; the CMake recipe changes only the validation-project name.
It calls `load_and_run_default_citlali_reduction`, not the successor entry.
Its target is `retained_validation_driver`, with `REVIEWED_SOURCE` bound to the
accepted stage-3 checkout. It builds a new executable without installing it or
replacing the preserved native POINT executable. This recipe reuses the existing
Mac Spack realization and does not reproduce Unity V2 acceptance.

Static review only: checked source identities, dependency locator, environment
digests, command construction and driver equivalence. The new wrapper has not
been executed, even for syntax/help checks; Grant owns its first execution and
all resulting build/version gates. No fresh observational result exists.

No SCIENCE reduction command or offset YAML is supplied. Those steps await the
connected successor implementation and their own input/reference/correction
bindings. The retained-driver build is not a prerequisite for POINT.

## Reporting and disposition

Publish a compact scorecard with links to exact configs, input manifest,
executable/source/environment identities, logs, complete run receipts, required
product inventory and positional/weight/validity comparisons. Each row states
the execution route, claim, result, limitations and owner disposition.

- Scientific/behavioral: workflow clarified; new observational gates pending.
  No scientific threshold or acceptance changed. Retained SCIENCE evidence
  cannot qualify configured successor SCIENCE or configured POINT.
- Architecture/ownership: no implementation changes. Configuration remains an
  upfront request; learned/effective/realized changes must be reported. Python
  POINT orchestration, native services and retained standard reduction retain
  their separate owners and entry points.
- Repository/evidence: documentation-only preparation from the clean exact
  base; accepted stages 1–3 and historical V2/012cc7adc evidence are unchanged.
  Existing software gate passes remain scoped to their original candidates.
  New qualification gates have not run. No push, production authorization,
  deployment, runtime replacement or cleanup follows.

Next: select individual POINT observations, bind their existing configuration,
inputs and applicable scientific acceptance criteria, then prepare and review
the exact per-case owner-run packet for steps 0 and 1. Preserve its evidence
for later reuse. SCIENCE implementation, celestial coadds and the known-offset
check are deferred; their final qualification must use the connected successor
route. The three accepted software repairs remain accepted within their
recorded scope. No additional scientific acceptance follows from this sequencing
decision. Grant retains all execution; no push.
