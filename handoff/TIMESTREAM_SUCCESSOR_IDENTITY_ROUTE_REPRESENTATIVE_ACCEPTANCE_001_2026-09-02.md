# Timestream Successor Identity Route Representative Acceptance 001

Status: acceptance-tooling candidate under local validation; representative
Spack/Unity execution, owner disposition, push, canonical integration, and
activation have not occurred

Work-order identity:
`TIMESTREAM-SUCCESSOR-IDENTITY-ROUTE-REPRESENTATIVE-ACCEPTANCE-001`

Owner: Citlali project owner

## Work order and preflight

Purpose: prepare the smallest acceptance-only tooling increment that can
execute and machine-validate the complete accepted typed identity witness on
the bounded NGC4449 observation-152390 data. The tooling must distinguish its
own source identity from the accepted implementation subject and must not
change that implementation.

Risk tier: Tier 2. This is executable acceptance evidence for a cross-stage
typed route, even though it changes no scientific or application component.

Applicable governance read:

- `doc/governance/ENGINEERING_GOVERNANCE.md`;
- `doc/governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md`; and
- `doc/governance/REVIEW_AND_CONFORMANCE.md`.

These documents are effective through the owner-accepted canonical governance
integration and effectiveness record. The governing subject-specific sources
also include `AGENTS.md`, `doc/REFACTOR_STATUS.md`, `doc/ARCHITECTURE.md`,
`doc/SCIENTIFIC_CONVENTIONS.md`, `doc/RETAINED_DEBT.md`, ADRs 0014, 0017,
0018, 0021, and 0023, the WP-7.1 authority router, and the Spack-backed V2
validation authority.

Current sequencing authority: the project owner's acceptance of exact Identity
Route 001 candidate `b57d9f606549d524ab6bb61faf0cd3d52ac27db6`, followed
by explicit approval to proceed with this bounded acceptance-only plan.

WIP slot: the existing single Timestream Successor integration/spine
increment. No module probe, ordinary-route change, or build-adaptation lane is
opened.

Branch and worktree:

- branch: `codex/timestream-successor-identity-route`;
- worktree: `/private/tmp/citlali-timestream-successor-identity-route`;
- accepted implementation subject:
  `b57d9f606549d524ab6bb61faf0cd3d52ac27db6`;
- subject tree: `32de9791255c6c52032c0f05d64054b83ff44de5`;
- subject sole parent:
  `3b1d56a87bebb4aebc02db223f36fc7f5eeb83b7`; and
- local and live remote feature refs were clean and equal to the subject before
  the separately authorized process-policy change.

The owner clarified that all GitHub pushes are user-performed. Standalone
commit `c2da49fc912b6bb789c8a58d682f4614f3ad5a7c`, whose sole parent is
the accepted subject, records that rule in `AGENTS.md`. Codex must provide the
exact non-force push command or arguments but must not execute a push.

## Accepted implementation disposition carried forward

The subject candidate is the VAL-repaired, unactivated Identity Route 001
implementation accepted by the owner. Its reviewed ancestry is:

```text
4c06fe83d8c9627d87f97c6a77d6a4ca99156e5a
-> 3b1d56a87bebb4aebc02db223f36fc7f5eeb83b7
-> b57d9f606549d524ab6bb61faf0cd3d52ac27db6
```

Post-commit validation reported for exact subject `b57d9f606...` passed:

- all 52 Timestream Successor focused tests: Paired-D1 6/6, AST/ALIGN
  17/17, VAL/identity RTC/RTC-only 21/21, and route context 8/8;
- isolated public-header compilation for the complete successor surface;
- `citlali_cli` and `citlali_safety_test` builds;
- all 884 runnable CTests out of 885 registered, with the one established
  `MapFitterLifecycle.ExactProductSequence` test disabled;
- configuration preflight 130/130, all four mode kits, 8/8 compact
  compatibility cases, full surface coverage, and every authority/boundary
  audit;
- baseline tools 207/207;
- build tools 62/62;
- historical WP-7 tools 26/26;
- then-current identity-route validator 7/7;
- validation ledger with 60 valid records; and
- science-change ledger with 3 changes and 5 valid integration commits.

Independent fresh-context review of the exact subject passed with no
scientific/behavioral, architecture/ownership, or repository/evidence
findings. The owner accepted it and authorized its exact normal push. The live
remote feature ref was then verified at
`b57d9f606549d524ab6bb61faf0cd3d52ac27db6` with clean local branch,
index, and worktree.

The subject validation used the local AppleClang/cache-backed dependency
realization and is supplemental regression evidence, not Spack/V2
representative evidence. The reused KIDs dependency carried the pre-existing
`04088da-dirty` provenance limitation. No Spack/V2 representative execution,
canonical integration, route activation, ordinary-route change, MAP action,
production use, or representative science claim occurred.

## Included and excluded scope

Included:

- repair the already opt-in acceptance runner so it builds after the accepted
  always-present VAL repair;
- execute the complete existing typed route rather than the earlier RTC-only
  component;
- load the exact previously accepted real telescope artifact and build the
  accepted AST raw product and network-native ALIGN views;
- construct VAL generation zero immediately after Paired-D1 and prove that the
  same immutable snapshot reaches ALIGN, RTC input, evidence, plan, applied
  product, RTC output, and the MAP-facing bundle;
- prove identity RTC reports AST motion dependency `not_applicable` while AST
  remains present in the interface;
- prove exact midpoint occurrence assignment, factor-one/phase-zero identity,
  exact Paired-D1 values, identities, causes, support, and partition
  invariance;
- prove truthful unavailable CAL/PTC products and their unavailable optional
  future VAL evaluations, unavailable MAP admission, and no MAP action;
- bind the output record to the exact subject candidate and its tree separately
  from the acceptance-tooling revision;
- require a clean Spack build under profile `unity-gcc13`, retain a new
  `spack.lock` by exact bytes and SHA-256, record the realized root DAG and
  executable/dependency identities, and validate the JSON record
  mechanically; and
- preserve the exact owner disposition and environment limitations in this
  durable record.

Excluded:

- any change to Paired-D1, VAL, AST, ALIGN, RTC, CAL, PTC, MAP, or route-context
  implementation;
- filtering, factor selection, downsampling, resampling, common-grid
  projection, RTC/PTC science, MAP processing, or route activation;
- YAML/CLI selection, ordinary-route changes, persistent product publication,
  production use, canonical integration, release, or legacy retirement;
- reconstruction or substitution of the unavailable historical V2
  `spack.lock` bytes;
- build-system redesign or dependency repair; and
- the deferred substantive-RTC questions of typed VAL finding targets and
  operator-specific relevance/invalidation beyond the conservative VAL 0.1
  exact-generation-requires-relearn rule.

The exact historical V2 lock SHA and DAG remain evidence of that accepted
campaign, not inputs to be fabricated. A new run must retain its own exact
lock bytes, digest, and concrete DAG.

## Reassessment: stale provisional runner

Trigger: the opt-in representative runner had not been built after the VAL
repair. A direct local build failed because its aggregate
`RtcOnlyRouteRequest` omitted the now-required VAL snapshot. It also exercised
only the RTC-only component and therefore could not evidence AST presence or
the complete typed MAP-facing boundary.

Disposition: repair within scope. The affected paths are evidence tooling
already named by the approved plan. The repair changes no public header,
pipeline source, application route, configuration, or build environment. The
existing standalone and Spack target registrations remain adequate and no
generic framework is introduced.

## Tooling design

The runner's bounded representative input is exactly:

- dataset `SCI_ALIGN_STAGE7_NGC4449_152390`;
- canonical observation scope `(152390, 0, 2)`;
- 11 APT-bound KIDs networks;
- native row interval `[20000, 22048)` on each participant;
- real telescope file
  `tel_toltec_2026-02-19_152390_00_0002.nc`, 24,157,872 bytes, SHA-256
  `2845455a620635955c00a4731e0d9720cfa456fece79d1729cf755a366a1ad6b`;
- exact producer-interface artifact
  `TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE v0.1/r0.1`, SHA-256
  `f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`;
  and
- owner-approved midpoint support assignment
  `wp7-provisional-integration-center-152390-v1`, SHA-256
  `6fc4e9009b98190c42cc3f6e7e030fa317e8ae5f9e707cd968110a696fac2b6c`.

The v2 record and validator require:

- exact subject SHA/tree and a distinct exact tooling SHA;
- clean compiled-source identity and executable SHA-256;
- `build_environment=spack`, `build_profile=unity-gcc13`, retained lock byte
  count and SHA-256, and concrete root DAG;
- exact observation, telescope, APT, Tune, producer-interface, and support
  identities;
- one complete typed route publication and zero unexpected error/critical
  logs;
- complete Paired-D1/RTC value, identity, cause, support, midpoint, AST, VAL,
  and partition comparison cardinalities with zero mismatches;
- VAL generation zero with zero committed findings and eight exact-handle
  binding checks;
- AST present, identity RTC AST dependency `not_applicable`, truthful
  unavailable CAL/PTC/MAP states, and no MAP action;
- zero RTC-owned numerical bytes and zero ALIGN/RTC-output/VAL numerical or
  finding ownership; and
- explicit false claims for route activation, ordinary-route change,
  canonical integration, and representative science.

## Gates and current evidence

Focused local gates:

- compile the opt-in standalone acceptance executable;
- run its `--help` boundary without representative data;
- run all v2 validator unit tests;
- syntax-compile the validator and its tests; and
- prove the implementation-subject blobs are unchanged from exact
  `b57d9f606...`.

Broader local gates:

- build `citlali_cli`, the complete Timestream Successor focused targets, and
  `citlali_safety_test`;
- run all runnable CTests with the established disabled test reported;
- run full configuration preflight;
- run baseline-, build-, historical WP-7-, and validation-ledger suites;
- after commit, verify exact ancestry, tree, changed paths, CLI revision, and
  clean state.

Local evidence is AppleClang/cache-backed and supplemental. It cannot close
the representative gate.

Representative gate: after exact-SHA review and separate owner authorization,
the project owner performs one clean Unity `unity-gcc13` Spack build and the
bounded complete typed witness. The retained package must contain the new
lock bytes, lock SHA-256, root DAG, exact source and executable identities,
exact input/support artifact identities, runner output, validator PASS,
zero unexpected errors, and clean source state. This is an acceptance-tool
execution only; it is not a Citlali reduction or representative science claim.

## Local candidate validation

Pre-commit validation in the existing AppleClang/cache-backed worktree passed:

- the standalone acceptance executable compiled and its non-data `--help`
  boundary completed successfully;
- all 52 Timestream Successor focused tests passed;
- all 884 runnable tests out of 885 registered CTests passed, with only the
  established disabled `MapFitterLifecycle.ExactProductSequence` not run;
- the v2 machine-record validator passed all 9 focused unit tests and both
  Python files passed syntax compilation and Ruff checks;
- `citlali_cli`, `citlali_safety_test`, every focused successor target, and
  the acceptance target built successfully;
- configuration preflight passed 130/130 tests, all four mode kits, all 8
  compact compatibility cases, 100% surface coverage, and every reported
  authority/boundary audit;
- baseline tools passed 207/207, build tools 62/62, and historical WP-7 tools
  26/26;
- the validation ledger reported 60 valid records and the science-change
  ledger reported 3 changes and 5 valid integration commits;
- `git diff --check` passed; and
- all public implementation, implementation-source, focused-test, and test
  registration blobs under `include/`, `src/`, and `tests/` remained identical
  to exact accepted subject `b57d9f606...`.

No representative acceptance run was performed locally. No actual Spack lock,
root DAG, real-data output record, executable digest, or runtime memory result
is claimed by this candidate validation. Exact committed ancestry/tree, clean
state, post-commit CLI provenance, and independent review remain post-commit
gates.

## Completion authority and stop point

This work order authorizes local preparation, validation, and one coherent
acceptance-tooling candidate. It does not authorize Codex to access Unity,
perform representative execution, push, integrate, activate, clean refs, or
make a production/scientific claim. The candidate stops for exact-SHA review
and owner disposition. All GitHub push operations remain owner-performed.
