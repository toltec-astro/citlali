# Timestream Successor Identity Route 001

Status: owner-steered Tier-2 component work; RTC identity component preserved
locally, full Identity Route candidate not yet assembled, reviewed, integrated,
activated, or pushed

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
- this handoff and the living status record.

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
