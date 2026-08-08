# SCI-PTC-001 independent correlated-mode interface audit dispatch

This is one frozen, not-yet-launched Tier B scientific-interface audit. It
authorizes an independent audit only after owner/coordinator launch. It does
not authorize implementation, repair, re-audit, integration, production,
external contact, Unity, a local Citlali reduction, or another package audit.

## Execution profile

- MODEL: `sol`
- EFFORT: `ultra`
- TASK SHAPE: `decomposable_scientific`
- MISSION: independently derive and then audit the mature PTC
  correlated-mode cleaning, detector-weight, validity, response, covariance,
  product, and consumer interface at exact application SHA
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.
- SCIENTIFIC AUTHORITY: before the core freeze, only the approved RTC owner
  decisions and MAP precision/nonprecision contract named in the frozen
  manifest may be opened. All RTC, ALIGN, CAL, AST, and MAP observations are
  post-core evidence.
- ULTRA TRIGGER: one verdict must reconcile four independent hard
  investigations: mathematical cleaning/weight/covariance derivation, exact
  source-to-contract tracing, product/metadata/validity tracing, and
  VAL/MAP/NOI/BEAM consumer synthesis. Each can falsify a different part of
  the interface, and none substitutes for the others.
- PARALLELISM: `none`. Delegation, subagents, and external review are
  prohibited. The auditor keeps the four workstreams logically separate and
  is the sole synthesis owner.
- STOP RULE: stop at the initial checkpoint, after the independent-core
  artifact is frozen and before source/post-core exposure, before any broad
  test execution, and after the final audit for coordinator/owner review.
- EXPECTED OUTPUT: the exact eight documentation artifacts listed below,
  their digests, independent-core and final commit identities, and one concise
  owner-decision brief.
- DOWNSTREAM RESTRICTIONS: VAL, MAP, NOI, and BEAM remain within their current
  bounded `existing_use_only` or fail-closed states. BEAM is not launched.

The Ultra allocation is recorded under `FRAMEWORK-EFFORT-001` and ends when
the four workstreams are reconciled into the final PTC audit and owner brief.

## Mandatory scope checkpoints

Before substantive work, return a concise checkpoint and stop for coordinator
direction. State:

- allowed paths: only the eight deliverables below, an untracked temporary PDF
  rendering, and ordinary audit-branch Git metadata;
- local Citlali reductions and Unity: prohibited;
- application, test, build, validation, configuration, production, canonical
  ledger, and canonical handoff-registry edits: prohibited;
- allowed evidence: digest/YAML/link/static checks, offline LaTeX compile and
  render, focused deterministic static probes, and existing narrow PTC tests
  only after the independent core is frozen;
- delegation, external review, contact, or execution: none;
- first viable artifact: the committed frozen
  `SCI-PTC-001_INDEPENDENT_CORE.tex`; and
- next return: immediately after that freeze and before opening source or any
  post-core record.

The three required coordinator returns are:

1. before source inspection, after freezing the independent core;
2. at that first viable artifact with exact digest/commit; and
3. before any broad, costly, local-reduction, or external execution.

Broad or costly execution is not authorized. Return before adding an artifact
class, helper, schema, verifier, test campaign, delegation, reduction,
external request, or scientific interpretation not named here.

## Exact assignment

- package ID: `SCI-PTC-001`
- package name: correlated-mode cleaning and detector-weight interface
- tier: `B`
- repository: `/Users/gwilson/GitHub/citlali-refactor`
- canonical application ref: `origin/codex/refactor-mainline`
- governing application SHA: `46ad23888a40f5102cdfd50c06e49a549bdf8a20`
- governing parent: `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`
- governing tree: `ab230a93b8fb310d58aefd7ac5da92e5d5e0f408`
- audit branch: `codex/audit-sci-ptc-001`
- suggested fresh worktree: `/private/tmp/citlali-audit-sci-ptc-001`
- coordinator ledger/registry snapshot:
  `47c4632008c4fd78f2ab88106372d7ee11ee7711`
- frozen manifest:
  `doc/audits/handoffs/SCI-PTC-001/SCI-PTC-001_INBOX_AUTHORITY_MANIFEST_2026-08-08.yaml`
- manifest SHA-256:
  `23913b37dfac7a106bcb281f9a1870616c99acf912a5a7aad59aed39e6bd67d3`
- pre-core IDs: `SCI-RTC-001-D001-D004` and
  `SCI-PTC-001-XAUD-001`
- post-core IDs: `SCI-PTC-001-XAUD-002`,
  `SCI-PTC-001-XAUD-003`, `SCI-PTC-001-XAUD-004`,
  `SCI-PTC-001-XAUD-005`, and `SCI-PTC-001-XAUD-006`
- planned numerical/external studies: none. Static probes and narrow existing
  deterministic tests are not costly. Any broad/costly proposal remains held
  behind the complete `FRAMEWORK-NUM-001` launch gate and separate owner
  authorization.

The current line contains only the bounded integrated MAP-001 and NOI-002
states recorded on application mainline; both remain `existing_use_only`.
CAL and MAP-002 repairs are active from this exact base. ALIGN evidence remains
owner-managed; AST repair waits behind ALIGN. No active-task result may be
inferred. The completed RTC audit and exact ALIGN processed-PTC repair are
post-core evidence only.

## Included scientific interface

Audit PTC from admitted RTC-stage detector timestreams, kernels/responses,
flags, sample/detector identities, APT state, and source masks through cleaned
TOD, evolved validity/flags, detector weights, diagnostics, processed
products, and downstream contracts. Cover:

- mean/common-mode/PCA or other correlated-mode estimation, selection,
  subtraction, iteration, convergence/state, normalization, degeneracies,
  excluded modes, and sequential/OpenMP equivalence;
- source masking, second-pass selection, detector/sample admission, flag and
  non-finite handling, replacement/synthesis influence, scan/segment
  boundaries, and feedback paths;
- detector-weight construction, factor identity, units, normalization,
  lifecycle, formal variance versus empirical scatter, precision claims, and
  retained temporal/detector/cross-observation covariance;
- exact signal, kernel or realized local response, transfer, causal support,
  phase/rate/time identity, response uncertainty, and explicit unavailable
  states through data-dependent cleaning;
- requested, effective, observation-resolved, and realized state, including
  method choice, components, thresholds, masks, selected detectors, passes,
  coefficients, convergence, and random/simulation state;
- raw/outer/inner/full/mini/diagnostic/simulated/processed stage identity,
  scan/sample extents, flags, units, product links, atomicity, and provenance;
  and
- exact VAL eligibility, MAP weight/significance, NOI noise/covariance, and
  BEAM response/uncertainty contracts.

## Explicit exclusions and non-authorizations

- Preserve mature PTC numerical internals unless a concrete interface,
  response, covariance, amplitude, selection, or feedback contradiction gives
  a named Tier-A reopen trigger.
- Do not audit RTC, ALIGN, CAL, AST, VAL, MAP, NOI, BEAM, JINC, Wiener, fruit
  loops, or source fitting beyond their direct PTC boundary.
- Do not accept or dispose the RTC handoff merely by launching this audit. Its
  closure remains gated by an accepted exact RTC successor and fresh re-audit.
- Do not treat ALIGN repair `5c6309...` as integrated or Unity-validated.
- Do not modify or repair application code, tests, builds, configurations,
  production records, audit coordination, or another audit package.
- Do not push, merge, rebase, cherry-pick, install/download software, use the
  network, contact people, connect to Unity, run a local Citlali reduction,
  delegate, repair, re-audit, or launch VAL/BEAM.

## Frozen evidence partition

Verify every manifest path, digest, Git object, status, and review phase. Open
only `SCI-RTC-001-D001-D004` and `SCI-PTC-001-XAUD-001` before the core freeze.
The former supplies approved upstream product authority with open
implementation/evidence gates; the latter supplies the approved rule that MAP
and coadd weights are nonprecision coefficients unless PTC proves the required
precision and covariance conditions.

Do not open the contents, diffs, callers, tests, or references of these
quarantined implementation paths before freeze:

- `include/citlali/core/timestream/ptc/ptcproc.h`
- `include/citlali/core/timestream/ptc/clean.h`
- `include/citlali/core/pipeline/processed_timestream_execution_plan.h`
- `include/citlali/core/pipeline/processed_timestream_provenance.h`

Do not open `SCI-PTC-001-XAUD-002` through `-006` before freeze. In
particular, `-005` is the completed RTC evidence handoff and `-006` is the
ALIGN processed-PTC scan-metadata repair evidence. Neither may alter the
independent core. Record unavoidable prior exposure truthfully.

## Phase 1 — independent mathematical core

Create exactly:

`doc/audits/packages/SCI-PTC-001_INDEPENDENT_CORE.tex`

Derive the interface from product intention and the two approved pre-core
authorities, not source names. Use a data-dependent/stateful model such as

```text
z = F(x, m, a, theta(x, m, a));
w = W(z, m, a, theta);
state_(k+1) = G(state_k, x, m, a)
```

and define every term. At minimum:

1. define sample/detector/scan/stage identities, shapes, order, time/rate,
   units, frames, indexing, validity, missing/non-finite policy, and upstream
   assumptions;
2. classify masks, flags, components, coefficients, thresholds, factors,
   detector weights, selected samples/detectors, passes, convergence, and
   simulation/random state as fixed, fitted, selected, data-derived, random,
   or conditioned upon;
3. derive estimator expectation, formal variance, full covariance, empirical
   uncertainty, calibration/systematic uncertainty, and response uncertainty;
4. derive temporal, amplitude, spatial/source-mask, common-mode/PCA,
   detector, extended-mode, iteration, and downstream map/beam response,
   including when a global linear transfer does not exist and what realized
   local response is required;
5. distinguish variance, inverse variance, detector weight, normalization,
   support, hits, coverage, direct validity, causal influence, confidence,
   response, and scientific eligibility;
6. define requested/effective/observation-resolved/realized state and immutable
   parent/processing product links;
7. define provisional fail-closed VAL/MAP/NOI/BEAM contracts; and
8. preregister analytic limits and deterministic falsification cases for
   identity/no-clean, single detector, identical detectors, rank-deficient and
   orthogonal modes, scale/sign invariance, source-mask boundaries, selected
   and rejected samples/detectors, flags/non-finite/influence, unequal
   calibration factors, constant/impulse/sinusoid, scan segmentation,
   full/mini variable chunk extents, one/two-pass behavior, sequential/OpenMP,
   simulation, covariance, and map-weight interpretation. Give an exact
   `not_applicable` rationale for each omitted standard method.

Freeze the exact bytes, compute SHA-256, commit the core alone, report the
freeze identity/timestamp/exposure record and next intended open, and stop.
Any later correction is a preserved successor revision.

## Phase 2 — source-to-contract review

Proceed only after explicit coordinator approval. Record first source and
post-core opening times. Inspect exact source at `46ad2388...`, then
disposition all five post-core records without promoting their claims.

Trace numbered independent equations through PTC admission, cleaning,
weights, flags/validity, kernel/response, first/second pass, simulation,
sequential/OpenMP paths, configuration resolution, product writers,
scan/sample metadata, provenance, and direct VAL/MAP/NOI/BEAM consumers.
Record any additional source path and why it was needed.

Focused deterministic static probes and narrow existing tests are permitted
after freeze. Record commands, exact SHA, inputs, and limits in local evidence.
Stop before any broad test/build campaign; a pass cannot replace the contract,
close a handoff, or authorize production. Classify findings independently by
class, severity, basis, confidence, owner, dependency, and falsifiable gate.

## Exact deliverables

Create only:

1. `doc/audits/packages/SCI-PTC-001_INDEPENDENT_CORE.tex`
2. `doc/audits/packages/SCI-PTC-001_SCIENTIFIC_CONTRACT_AUDIT.tex`
3. `doc/audits/evidence/SCI-PTC-001_LOCAL_EVIDENCE_2026-08-08.yaml`
4. `doc/audits/proposals/SCI-PTC-001_LEDGER_PROPOSAL_2026-08-08.yaml`
5. `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-008.yaml`
6. `doc/audits/handoffs/SCI-MAP-001/SCI-MAP-001-XAUD-004.yaml`
7. `doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001-XAUD-001.yaml`
8. `doc/audits/packages/SCI-PTC-001_OWNER_DECISION_BRIEF_2026-08-08.md`

Create each outgoing handoff only with a bounded supported claim; record an
explicit no-material-handoff conclusion in the report if one target has none.
BEAM-relevant facts must be included in the VAL handoff and owner brief for
later routing; do not create or launch a BEAM package here.

The final report must include the frozen core/exposure log, source/equation
trace, product/consumer matrix, uncertainty/response/covariance treatment,
all findings/decisions/dependencies, four status axes and verdict, allowlist
and fail-closed restrictions, every inbound disposition, applicable outgoing
handoffs, local evidence, and a concise owner brief. Compile/render offline
and validate YAML, links, digests, controlled vocabulary, Git objects, diff
scope, and clean state.

## Final return and stop

Return exact branch/worktree/commit/parent/tree; core digest/freeze/timestamp
and exposure events; final report digest/render result; compact claimed and
implemented operators; findings, decisions, dependencies, axes, verdict, and
restrictions; inbound dispositions and outgoing artifacts/digests; evidence
limits and cost classification; and confirmation that no application,
validation, Unity, external, repair, re-audit, production, canonical
coordination, or other audit was changed or launched.

Stop for coordinator/owner review. Do not repair, integrate, re-audit, push,
launch VAL/BEAM, or claim production authorization.
