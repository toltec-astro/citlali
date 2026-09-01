# Timestream Successor Identity Route 001

Status: owner-steered Tier-2 spine candidate assembled locally from the pushed
RTC identity and AST/ALIGN components; pending exact-SHA independent review
and owner disposition, not integrated or activated

Work-order identity: `TIMESTREAM-SUCCESSOR-IDENTITY-ROUTE-001`

Owner: Citlali project owner

## Effective owner steering / disposition

The project owner's 2026-09-01 steering and five subsequent decisions
supersede any broader completion implication in the original work-order text
below:

1. This work may end at a complete typed MAP-facing bundle while MAP admission
   remains explicitly unavailable. It does not perform or activate MAP.
2. Canonical Paired-D1 remains intact. Later route assembly adds an ALIGN-owned
   wrapper/bundle around it; Paired-D1 is not split, rewritten, or made to own
   downstream context.
3. AST is part of the RTC input context. Whether an RTC operator consumes AST
   motion is operator-dependent. Identity RTC declares that dependency
   `not_applicable`; AST therefore remains present in the interface and is not
   represented by deleting AST from the route topology.
4. Until real CAL and PTC components are admitted, contract-derived fixtures
   and typed unavailable CAL/PTC states are the only permitted representation.
   Fake identity CAL or PTC stages are prohibited.
5. A network occurrence is assigned the midpoint of its occurrence interval
   until the policy can be revisited using beammap data. This is the bounded
   owner-approved timing policy, not an inferred common analysis grid.

The implementation already present on this branch is consequently preserved
as an RTC identity **component checkpoint**, not closed or accepted as the
complete `TIMESTREAM-SUCCESSOR-IDENTITY-ROUTE-001` vertical route. Its
representative-data runner and record validator remain provisional scaffolding
and must not be run or treated as acceptance evidence for the incomplete
topology.

The next approved stage on this same branch is a literal bounded replay of the
accepted AST v2 native-motion product and ALIGN-owned per-network mapped views:

- initial AST product: `0046092125075cd498c0eb3888429a74b968d2a4`;
- route-family v2: `672f907355a3f15f3ee987d92a5f7e95bbdc38b5`;
- reviewed cause-precedence repair:
  `adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7`; and
- historical review/evidence record:
  `37c2adfa84762cb8cef5dc66d5b1fbc6753331f6`.

Only the accepted AST raw product, ALIGN-owned network views, isolated headers,
focused tests, and minimal build registration are admitted in that stage. The
historical WP-7 status/control material, acceptance/census/filter tooling,
authority-ledger changes, `Engine` changes, common-grid work, and all RTC
coupling remain excluded. The historical header name
`timestream_native_timing.h` is adapted to the canonical
`timestream_native_alignment.h`; no timing implementation is duplicated.

## Original work order / preflight

Original purpose: recover the preserved first vertical Timestream Successor witness as
one unactivated route from an atomic native KIDs `x/r` result through canonical
Paired-D1, identity RTC Learn--Consider--Plan--Apply, exact logical
finalization, and one in-memory RTC-only terminal publication.

Risk tier: Tier 2. This increment introduces a new cross-stage typed spine and
exercises scientific lifecycle boundaries, although its numerical operator is
strict identity.

Applicable governance read:

- `doc/governance/ENGINEERING_GOVERNANCE.md`;
- `doc/governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md`; and
- `doc/governance/REVIEW_AND_CONFORMANCE.md`.

Their incorporation-time `candidate` banners are superseded for effectiveness
by the accepted exact digests and canonical incorporation recorded in
`doc/INTEGRATION_LEDGER.md` and the 2026-08-31 governance-effectiveness status
record.

Current sequencing authority: canonical `doc/REFACTOR_STATUS.md` at exact base
`6d6e5d570e2a311687ede8e954c996046772af6f`, plus the project owner's
2026-09-01 approval of this exact work order.

Scientific authority:

- WP-7.1 Timestream Contract Baseline source
  `170ecea9de1ee810da7d7e45a489a4545ccd623d` and closure
  `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`;
- canonical authority router
  `validation/wp7_timestream_successor_authority.json`;
- canonical ADR 0017's native-axis identity-RTC and RTC-only route decision;
- canonical ADR 0018's network-native timing ownership and separation from a
  separately requested common analysis grid; and
- Paired-D1 implementation `30c42528f86bb9b7d8104bbda63834ce72595798`
  and canonical admission record
  `6d6e5d570e2a311687ede8e954c996046772af6f`.

Architectural authority: `AGENTS.md`, `doc/ARCHITECTURE.md`,
`doc/SCIENTIFIC_CONVENTIONS.md`, canonical ADRs, and the three effective
governance documents above.

Exact canonical base: `6d6e5d570e2a311687ede8e954c996046772af6f`.

Branch and worktree:

- branch: `codex/timestream-successor-identity-route`;
- worktree: `/private/tmp/citlali-timestream-successor-identity-route`;
- initial tree: `f6fa552231f60f62bd2bd2e91be98f5ec0de59a2`; and
- initial staged, unstaged, and untracked state: clean.

WIP slot: the one active Timestream Successor integration/spine increment. No
scientific-module probe is opened.

Owned products, interfaces, and lifecycle:

- the KIDs ingress adapter atomically transfers the two coordinate planes from
  one solver-result instance into one canonical Paired-D1 network;
- canonical Paired-D1 remains the observation/network-owned input authority;
- RTC owns immutable learned evidence, consideration, the factor-one plan,
  exact plan application, and realized-operation facts;
- application orchestration owns exact logical finalization and one no-replace
  in-memory terminal slot; and
- `Engine` gains no state or scientific behavior.

Included scope:

- one rvalue-only, exact-layout KIDs-to-Paired-D1 network adapter;
- native-axis immutable views and exact engineering-partition coverage;
- sparse cause-preserving identity evidence;
- a factor-one, phase-zero identity operator and immutable plan;
- a zero-copy RTC result referencing canonical Paired-D1 values, identities,
  supports, validity, causes, mapping authority, and lineage;
- run- and exact-input-bound finalization;
- one in-memory, once-only, no-replace RTC-only terminal publication;
- isolated-header and behavioral tests; and
- opt-in representative-data acceptance tooling and a machine-checkable
  acceptance-record validator.

Original work-order excluded scope: route activation; CLI or YAML selection; persistent products;
filtering; factor selection; incidence or glitch handling; leakage correction;
notches; low-pass design; resampling; downsampling; AST; a common analysis
grid; CAL; PTC; VAL; MAP; FRUIT; ordinary-route wiring; production use;
build-environment repair; integration; push; ref cleanup; and any scientific
or architectural framework in anticipation of later work.

Expected changed ownership areas:

- `include/citlali/core/pipeline/timestream_native_paired_readout_kids_adapter.h`;
- `include/citlali/core/pipeline/timestream_identity_rtc.h`;
- `include/citlali/core/pipeline/timestream_identity_rtc_only_route.h`;
- corresponding focused tests and isolated-header translation units;
- bounded CMake registration;
- `tools/timestream_successor/` acceptance runner and validator;
- this handoff; the living status record remains unchanged until a later
  reviewed route state warrants a canonical status transition.

Preserved recovery authority:

- identity implementation foundation
  `dd06fc27251ad8925a3cf2bdfd82661edfec6a43`, accepted historical tip
  `641f75ad8ce6d026e30e29dc31f50028a1ac757d`;
- exact-context terminal finalization correction
  `cddfea28f89d3ca51ba52930a82f51c270905874`, preserved closure tip
  `0574d9a50fe6df6f7ded07c1d229bcb8ca04309d`.

Recovery is literal at path and concept level, followed only by enduring-name,
canonical Paired-D1 interface, and exact-context terminal adaptation. No
historical status, control record, common-grid spike, anticipatory application
context, or later scientific module is imported.

Focused gates:

- all new isolated headers compile without a precompiled-header dependency;
- the canonical Paired-D1 focused tests remain green;
- adapter tests prove one atomic rvalue source, exact matrix layout, ownership
  transfer, and fail-closed mismatches;
- identity RTC tests prove exact value bits, coordinate-local validity and
  causes, pair consequence, factor one, zero numeric duplication, deterministic
  partition invariance, and absence of pointing/common-grid dependencies;
- route tests prove one Learn--Consider--Plan--Apply entry, exact-input
  finalization, complete logical counts, truthful failure, and once-only
  publication; and
- acceptance-validator tests prove exact source/environment/data/evidence
  binding and fail closed on incomplete records.

Broader gates: build `citlali_cli`, the focused target, and explicitly
`citlali_safety_test`; run all runnable CTests with the established disabled
test reported; run the complete configuration preflight; run baseline-tool,
build-tool, historical WP-7-tool, and validation-ledger suites; verify CLI
revision identity after the candidate commit; verify ancestry, tree identity,
changed paths, and final worktree state.

Affected-mode / representative-environment gate: no existing mode is switched,
so ordinary Point, OOF, Science, and Beammap behavior is not affected by local
route execution. Completion of the full vertical increment nevertheless
requires an owner-run, accepted Spack-backed representative
`citlali-validation/v2` package showing real KIDs ingress, exact identity
output, partition invariance, bounded cardinality/memory, zero unexpected
error-level output, and no default-route change. Local AppleClang or cached
dependency results are supplemental only.

Review triggers: any need to modify canonical Paired-D1; any scientific choice
beyond identity; new cross-network timing or common-grid dependency; growth of
`Engine`; YAML, CLI, persistent schema, or ordinary route wiring; matrix copy;
loss of an admitted identity/cause/support relation; build-environment changes;
or performance/memory behavior inconsistent with zero-copy apply.

Stop conditions: an absent or contradictory contract; inability to adapt the
preserved route without duplicating canonical Paired-D1; need for any excluded
stage or scientific operator; a moving base; or a required sibling-repository
change.

Integration, push, activation, and cleanup authority: none. The work order
authorizes this one branch/worktree and a coherent candidate commit only. The
candidate must stop for independent fresh-context exact-SHA review and then
owner disposition.

## Component-checkpoint / conformance state

The current RTC identity implementation and provisional tooling are retained as
a local component checkpoint only. This record makes no full-route
conformance, representative-environment, integration, activation, production,
or push claim. A later complete typed topology must incorporate the owner
dispositions above and undergo its own validation and exact-SHA review.

The exact RTC component checkpoint is
`b2ad615b8` (`Preserve RTC identity component checkpoint`), based literally on
canonical `6d6e5d570e2a311687ede8e954c996046772af6f`. It is contained in the pushed
feature branch through `4c06fe83d8c9627d87f97c6a77d6a4ca99156e5a` and is not an integration or
activation candidate by itself.

## Bounded AST / ALIGN replay stage

Replay parent: exact local RTC component checkpoint `b2ad615b8`.

Source disposition:

- `ast_scan_motion.h`, `ast_scan_motion.cpp`,
  `ast_scan_motion_alignment.h`, `ast_scan_motion_alignment.cpp`, both isolated
  header translation units, and `test_ast_scan_motion.cpp` were recovered from
  the accepted state at `adbc013e2d4287fb5a32db8bc7f2b0112c1c88d7`;
- both implementation translation units and both isolated-header translation
  units are blob-identical to that accepted source;
- the two public headers differ only by replacing the historical
  `timestream_native_timing.h` include with canonical
  `timestream_native_alignment.h`;
- the focused behavioral test differs only by removing its dependency on
  `Engine::Telescope` and the assertion for the historical Engine registry
  mutation, which is deliberately not imported; and
- minimal standalone and Spack library-source registration plus one isolated,
  non-default focused test target were added.

Ownership and exclusion audit:

- AST owns one immutable raw telescope-motion source/product and its typed
  validity/cause/support facts;
- ALIGN owns the per-network mapped views, which reference the exact canonical
  `NativeNetworkAlignment` and raw AST product without copying a network time
  axis or creating a common grid;
- no RTC type includes or consumes AST in this stage;
- no Paired-D1, `Engine`, YAML, CLI, route-selection, persistence, filtering,
  factor-selection, resampling, downsampling, CAL, PTC, MAP, or ordinary-route
  implementation is changed; and
- historical WP-7 status/control records, acceptance/census/filter tooling,
  authority-ledger mutations, and the historical review-record content remain
  evidence only and are not replayed.

Local supplemental validation:

- focused AST/ALIGN target configured and built successfully;
- `citlali_cli` built successfully without route activation;
- AST raw-product and ALIGN-mapping tests: 17/17 passed;
- canonical Paired-D1 regression: 6/6 passed;
- preserved identity-RTC component regression: 15/15 passed; and
- canonical SCI-ALIGN regression: 94/94 passed.

These results use the existing local AppleClang/cache-backed build and are not
Spack/V2, representative-data, integration, activation, or production
evidence. The representative identity acceptance executable was not run. The
project owner accepted exact AST/ALIGN replay commit
`4c06fe83d8c9627d87f97c6a77d6a4ca99156e5a` and pushed the feature branch at
that exact ref. This accepted component state is the literal base for the next
bounded stage; it is not integrated or activated.

## Bounded route-context assembly stage

Stage work-order identity:
`TIMESTREAM-SUCCESSOR-IDENTITY-CONTEXT-ASSEMBLY-001`.

Owner authorization: the project owner accepted exact component base
`4c06fe83d8c9627d87f97c6a77d6a4ca99156e5a`, pushed that exact feature ref,
and on 2026-09-01 explicitly authorized proceeding with the previously
proposed bounded typed route-context assembly.

Risk and WIP: Tier 2. This remains the one active Timestream Successor spine
increment. It opens no scientific-module probe.

Exact base, branch, and worktree:

- base: `4c06fe83d8c9627d87f97c6a77d6a4ca99156e5a`;
- base tree: `afb0714acea48cc3c477718a3ef44728e8926f9a`;
- branch: `codex/timestream-successor-identity-route`;
- worktree: `/private/tmp/citlali-timestream-successor-identity-route`; and
- initial staged, unstaged, and untracked state: clean, with local branch and
  `origin/codex/timestream-successor-identity-route` verified at the exact
  base.

Scientific and owner authority: the effective engineering, Timestream
Successor, and review/conformance governance package; canonical ADRs 0017,
0018, 0021, and 0023; the accepted Paired-D1 and AST/ALIGN component contracts;
and the five owner decisions at the beginning of this record. ADR 0022's
nonidentity occurrence-level RTC speed admission is not exercised: identity
RTC's AST motion dependency is explicitly `not_applicable`.

Owned products, interfaces, and lifecycle:

- canonical Paired-D1 remains the sole observation/network owner of paired
  signal values, detector and occurrence identities, native timing, and
  integration support;
- ALIGN owns a concrete immutable route-context wrapper that binds exact
  Paired-D1 and AST network-view handles, requires identical observation and
  participant scopes, and assigns each occurrence the midpoint of its
  Paired-D1 integration interval without creating a common grid;
- AST remains present as the coordinate/motion authority in the RTC input
  context and on the RTC output domain; the identity operator declares AST
  motion consumption `not_applicable` rather than deleting AST from the
  interface;
- RTC retains its existing Learn--Consider--Plan--Apply lifecycle and
  factor-one, phase-zero, zero-copy realization unchanged;
- application orchestration assembles an immutable MAP-facing context bound
  to the exact RTC terminal, while CAL and PTC remain typed unavailable and
  their use-specific VAL dispositions remain explicitly unevaluated because
  those scientific products do not exist; and
- MAP admission remains explicitly unavailable and performs no MAP action.

Included paths and scope:

- one concrete `timestream_identity_route_context` public interface and
  implementation;
- one backward-compatible optional exact-view field in the preserved RTC-only
  request so the AST-bearing input context's admitted Paired-D1 view is the
  exact Learn--Consider--Plan--Apply input;
- one isolated-header translation unit and focused behavioral tests;
- minimal standalone/Spack source and focused-test registration, including
  reconciliation of the source-graph audit's stale pre-AST source count; and
- this handoff record.

The context stores handles and compact span bindings only. It owns no signal,
coordinate, motion, response, uncertainty, or common-grid numerical plane.
It reuses the exact Paired-D1 native timing handle already referenced by each
AST network view. Because the accepted AST view maps at that timing handle's
event time, route admission fails closed unless that event time is exactly the
owner-selected midpoint of the paired occurrence interval.

Explicit exclusions: changes to canonical Paired-D1; changes to AST raw or
network-view algorithms; nonidentity RTC; filtering; factor selection;
downsampling; response or kernel construction; real CAL, VAL, PTC, or MAP
behavior; common-grid projection; route activation; CLI/YAML or persistent
schema; `Engine`; representative-data execution; build-environment repair;
status or integration-ledger closure; integration; push; cleanup; and generic
stage/framework work.

Focused gates:

- isolated compilation of the new public header;
- exact Paired-D1/AST handle, scope, participant, span, native-row, and
  midpoint bindings fail closed on mismatch;
- AST remains inspectable in the identity RTC input/output contexts while the
  dependency disposition is exactly `not_applicable`;
- identity execution preserves exact `x/r` bits, identities, validity,
  causes, support, factor one, phase zero, and zero RTC-owned numerical bytes;
- CAL, CAL-use VAL, PTC, PTC-use VAL, and MAP admission expose only typed
  unavailable states and manufacture no products, units, responses, or
  uncertainties; and
- the MAP-facing bundle binds the exact completed RTC terminal and does not
  perform or publish a MAP action.

Broader gates: build the new focused target, canonical Paired-D1, AST/ALIGN,
identity-RTC, `citlali_cli`, and explicitly `citlali_safety_test`; run all
runnable CTests with the established disabled test reported; run the complete
configuration preflight and the baseline-, build-, historical WP-7-, and
validation-ledger suites; after committing, verify the CLI exact short SHA,
ancestry, tree, changed paths, and clean worktree.

Build and representative evidence: local AppleClang/cache-backed C++ results
remain supplemental. This stage changes no active route or existing mode, so
no affected-mode reduction is triggered. Owner-run Spack/V2 representative
acceptance remains deliberately deferred until the full typed identity witness
has an independently reviewed owner-disposed candidate.

Review triggers and stop conditions: stop on any need to mutate Paired-D1 or
AST algorithms, manufacture missing CAL/PTC/MAP facts, weaken exact midpoint
or handle binding, introduce a common grid, grow `Engine`, activate a route,
add persistent configuration/schema, repair the build environment, or make a
new scientific choice. A contradictory or absent contract, moving base, or
required sibling-repository change is also a stop.

Integration, push, activation, cleanup, and production authority: none. A
coherent local candidate commit is authorized. It must receive independent
fresh-context read-only review at its exact full SHA and stop for owner
disposition.

## Route-context construction and local validation

The bounded implementation adds one concrete route context and makes one
backward-compatible exact-input extension to the preserved RTC-only request:

- ALIGN admits only exact Paired-D1/AST scope, participant, timing-handle,
  support, identity, and owner-selected midpoint agreement;
- the RTC input context owns the exact admitted logical Paired-D1 view, always
  carries the AST views, and declares only the identity operator's AST motion
  dependency `not_applicable`;
- the existing RTC-only lifecycle uses that exact view instance for evidence,
  plan, apply, finalization, and terminal publication while retaining its old
  self-admission behavior for existing callers;
- the RTC output context binds the exact factor-one terminal to the same AST
  views and exposes the native occurrence-to-source support association;
- CAL and PTC product, unit/conditioning, response, and uncertainty states are
  typed unavailable; their two VAL-owned use dispositions are distinct typed
  objects and remain unevaluated because the required products do not exist;
  and
- the immutable bundle reaches the MAP-facing boundary with admission
  unavailable and no MAP action or product.

No Paired-D1 value, state, identity, support, or ownership implementation was
changed. No AST raw or mapping algorithm, RTC numerical operator, ordinary
route, configuration, `Engine`, CAL/PTC/MAP behavior, response, uncertainty,
common grid, filter, factor, or downsampling implementation was added.

The production source-graph audit had retained its pre-AST hard-coded count of
11 even though the accepted AST replay had already raised both matching CMake
graphs to 13 sources. Registering this stage raised both to 14; the bounded
audit assertion now names that actual matching count. No build dependency,
toolchain, environment, or source composition changed beyond the three
already-declared production implementation translation units.

Pre-candidate local AppleClang/cache-backed validation passed:

- new isolated header and route-context target: 7/7 tests;
- preserved identity component including exact-view compatibility: 16/16;
- accepted AST/ALIGN component: 17/17;
- canonical Paired-D1: 6/6;
- `citlali_cli` and the explicitly requested `citlali_safety_test` built;
- full CTest registration: 879 tests, 878 runnable tests passed, and the one
  established disabled `MapFitterLifecycle.ExactProductSequence` remained
  disabled;
- configuration preflight: 130/130 tests, all four mode kits, 8/8 compact
  compatibility cases, complete surface coverage, and every authority/boundary
  audit passed;
- baseline tools: 207 tests and 137 subtests passed;
- build tools: 62/62 passed;
- historical WP-7 plus provisional identity-record verifier: 33 tests and 11
  subtests passed; and
- validation ledger: 60 records valid; science-change ledger: 3 changes and 5
  integration commits valid.

These are supplemental local regression and contract-conformance results, not
Spack/V2 representative-data, independent-review, integration, activation,
release, or production evidence. The provisional representative acceptance
runner was not run.

## Owner-directed VAL state-plane repair work order

Repair work-order identity:
`TIMESTREAM-SUCCESSOR-IDENTITY-VAL-REPAIR-001`.

Purpose and owner: the Citlali project owner's 2026-09-01 directive preserves
exact reviewed pre-repair candidate
`3b1d56a87bebb4aebc02db223f36fc7f5eeb83b7` and requires one local child
candidate that makes VAL an always-present parallel state plane. The repair
must stop after exact-SHA validation and fresh independent review.

Risk tier and WIP: Tier 2. This is a repair of the one active Timestream
Successor spine increment, not a new spine or scientific-module probe.

Applicable authority:

- effective `doc/governance/ENGINEERING_GOVERNANCE.md`,
  `doc/governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md`, and
  `doc/governance/REVIEW_AND_CONFORMANCE.md` through their accepted canonical
  incorporation record;
- WP-7.1 contract source
  `170ecea9de1ee810da7d7e45a489a4545ccd623d`, closure
  `20ba6ae5dcf0b90a24ac3e778a75eff0a1bbe2aa`, and canonical authority router;
- canonical ADRs 0017, 0018, 0021, and 0023; and
- the owner's VAL repair directive, which corrects this feature candidate's
  representation without changing Paired-D1, ALIGN/AST science, or identity
  RTC numerics.

Exact base, branch, worktree, and initial state:

- base and preserved pre-repair candidate:
  `3b1d56a87bebb4aebc02db223f36fc7f5eeb83b7`;
- base tree: `3bda4548b92e33fa648e8452cb3d2d21b38425ab`;
- sole parent of the preserved candidate:
  `4c06fe83d8c9627d87f97c6a77d6a4ca99156e5a`;
- branch: `codex/timestream-successor-identity-route`;
- worktree: `/private/tmp/citlali-timestream-successor-identity-route`;
- initial staged, unstaged, and untracked state: clean; and
- cached remote feature ref remains at accepted component base
  `4c06fe83d8c9627d87f97c6a77d6a4ca99156e5a`; local is exactly one commit
  ahead before this repair.

Reassessment finding: the pre-repair candidate has no core VAL container,
snapshot, generation, delta, or RTC phase binding. Its two VAL-named objects
represent only later CAL-for-PTC and PTC-for-MAP use dispositions as
`not_evaluated_product_unavailable`. That representation makes the only VAL
surface look unavailable alongside CAL and PTC and does not satisfy the
owner-directed always-present state-plane architecture.

Owned boundary and lifecycle:

- Paired-D1 remains the exact immutable product and identity authority;
- VAL 0.1 owns only an immutable route-associated snapshot container,
  producer-owned fact records, deterministic staged deltas, and generation
  mechanics; it owns no scientific inference, RTC plan, or MAP admission;
- ALIGN binds the exact Paired-D1, AST view set, and already-existing VAL
  snapshot into one route context without changing midpoint or AST behavior;
- identity RTC owns its conservative exact-generation dependency: evidence,
  plan, and product bind one exact immutable VAL snapshot, and any different
  generation requires relearning/reconsideration before apply; and
- application orchestration passes exact handles at phase boundaries and
  retains typed unavailable CAL/PTC products and unavailable MAP admission.

Included scope and expected paths:

- new isolated `timestream_val_state.h` container contract and header test;
- exact VAL-snapshot participation in `timestream_identity_rtc.h` and
  `timestream_identity_rtc_only_route.h`;
- VAL binding and clarification of optional unavailable downstream VAL
  evaluation in the identity route-context header/implementation;
- focused RTC, RTC-only, and route-context tests, including a synthetic
  producer-owned spike fact used only to prove stale-generation rejection;
- minimal focused-test CMake registration; and
- this handoff record.

The snapshot implementation will share its immutable Paired-D1 handle and
prior generation and own only each committed delta. It will not copy signal,
AST, support, provenance, or a complete prior VAL container per stage.

Focused gates:

- VAL isolated-header compilation, exact identity binding, deterministic
  insertion/lookup, staged invisibility, and immutable generation tests;
- RTC learn/consider/apply exact-snapshot tests, zero identity findings, and
  V0-plan rejection against a synthetic V1 spike finding followed by a valid
  V1 relearn/consider/apply lifecycle;
- always-present route VAL with exact Paired-D1/AST/RTC addressability while
  CAL/PTC remain unavailable and MAP performs no action; and
- retained Paired-D1, ALIGN/AST, identity RTC, RTC-only, and route-context
  focused regressions.

Broader gates: build the focused targets, `citlali_cli`, and
`citlali_safety_test`; run every runnable CTest with the established disabled
test reported; run configuration preflight plus baseline-, build-, historical
WP-7-, and validation-ledger suites; after the child commit, verify exact CLI
revision, ancestry, tree, changed paths, and clean state.

Affected-mode and representative-environment disposition: no route is active
and no numerical operator, configuration, or product is changed, so no
affected-mode reduction is triggered. Local AppleClang/cache-backed results
are supplemental. Spack/V2 representative execution is explicitly prohibited
for this repair.

Review triggers and stop conditions: stop on any need for a real spike or
incidence detector, scientific fact interpretation, convergence rule,
filtering, decimation, common-grid work, Paired-D1 or AST algorithm mutation,
nonidentity RTC behavior, CAL/PTC/MAP implementation, `Engine` growth,
persistent schema, route activation, or a generic workflow/validity framework.

Integration, push, activation, representative execution, cleanup, and
production authority: none. One coherent local child commit and fresh
read-only exact-SHA review are authorized. The preserved candidate must not be
amended, squashed, rebased, accepted, pushed, or integrated.

## VAL repair implementation and pre-commit validation

The repair adds `ValSnapshot` as the always-present route-associated state
plane. Generation zero binds one exact immutable Paired-D1 product. Each later
generation shares that product and its prior snapshot and owns only one sorted
producer-scoped delta. A `ValAddress` resolves exact sample, network,
occurrence, and optional detector identity through the bound Paired-D1
authority. Finding keys additionally bind the producer, its product instance,
and its opaque producer-local fact code; VAL itself assigns no scientific
meaning, score, operation, or downstream-admission decision.

The only mutable object is a phase-local `ValDeltaBuilder`. Proposals are not
visible through the immutable base snapshot before or after the builder is
frozen. Commit creates a new immutable generation; deterministic sorting and
duplicate-key rejection make insertion and lookup unambiguous. The container
does not include RTC plan types, MAP admission types, signal/AST payloads, a
workflow engine, or cross-producer inference.

The repaired route constructs VAL before ALIGN admission and binds that exact
snapshot beside Paired-D1 and AST. ALIGN, the RTC input/output contexts, the
RTC-only request and terminal product, and the MAP-facing boundary all retain
the same snapshot handle. CAL and PTC products remain typed unavailable. Their
formerly VAL-named downstream dispositions are clarified as optional future
CAL-for-PTC and PTC-for-MAP VAL evaluations whose absence does not make the
core state plane unavailable. MAP admission remains unavailable and the route
still performs no MAP action.

Identity RTC now receives an immutable VAL snapshot explicitly at Learn,
Consider, and Apply. Evidence retains the exact snapshot and exposes zero
identity-operator proposals; the plan retains the evidence and declares the
typed `exact_generation_requires_relearn` policy; Apply requires the exact
resolved snapshot instance and records the generation in both realization and
product. A different snapshot, including a child generation of the same
Paired-D1 product, is rejected as stale. Repeating Learn, Consider, and Apply
against the new generation is supported without adding a generic iteration
engine or convergence policy.

The synthetic spike fixture uses only an opaque test-owned RTC fact. It learns
and plans against V0, stages and commits the test fact into V1, proves the V0
snapshot never observes the proposal, rejects both V0 evidence consideration
and V0 plan application against V1, then completes a new explicit lifecycle
against V1. No spike detector or other incidence science is present in the
implementation.

Pre-commit local AppleClang/cache-backed validation passed:

- every Timestream Successor focused test: 52/52, comprising canonical
  Paired-D1 6/6, accepted AST/ALIGN 17/17, VAL/identity RTC/RTC-only 21/21,
  and repaired route context 8/8;
- isolated public-header compilation for the new VAL contract plus all
  retained successor headers;
- `citlali_cli` and `citlali_safety_test` builds;
- full CTest registration: 885 tests, 884 runnable tests passed, and the one
  established disabled `MapFitterLifecycle.ExactProductSequence` remained
  disabled;
- configuration preflight: 130/130 tests, all four mode kits, 8/8 compact
  compatibility cases, complete surface coverage, and every
  authority/boundary audit passed;
- baseline tools: 207/207 passed;
- build tools: 62/62 passed;
- historical WP-7 tools: 26/26 passed;
- provisional identity-route verifier: 7/7 passed;
- validation ledger: 60 records valid; and
- science-change ledger: 3 changes and 5 integration commits valid.

These results are supplemental local contract and regression evidence. No
Spack/V2 representative run, affected-mode reduction, route activation,
canonical integration, remote operation, or MAP action occurred. Exact-child
ancestry, committed tree, CLI provenance, clean state, and fresh independent
review remain post-commit gates.
