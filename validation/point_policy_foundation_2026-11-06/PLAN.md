# POINT assessment foundation work order

Current disposition, 2026-11-06: Grant accepted application
`ec3a447c1666fddb6be5c5823cee1b438f9c57c7` as a bounded software increment and
authorized local integration. The [owner acceptance](OWNER_ACCEPTANCE.json)
records the passing software and matched 152389 gates and independent reviews.
The original preflight and planned gate states below are preserved as history;
their pending/integration restrictions are superseded only for this accepted
increment. Scientific rule development and future integrations remain separate.

## Original work order and preflight

Original status: prepared candidate, runtime gates pending. Date: 2026-11-06.
Execution owner: Grant, for every build, test, probe, benchmark, reduction and
validation runner. Codex is limited to preparation, static review and authorized
local commits. No Unity access, push, integration, runtime replacement or cleanup.

## Preflight and authority

- Purpose: implement the owner-approved first increment collecting existing
  POINT facts for independent rule development in “Assess contract compliance”.
- Risk: Tier 2, new persistent diagnostic schema and required publication.
- Read governance: `doc/governance/ENGINEERING_GOVERNANCE.md`,
  `TIMESTREAM_SUCCESSOR_GOVERNANCE.md` and `REVIEW_AND_CONFORMANCE.md`;
  incorporated authority is indexed in `doc/INTEGRATION_LEDGER.md`.
- Sequencing: `doc/CURRENT_STATE.md`, `doc/REFACTOR_STATUS.md`; narrow owner
  approval supersedes the earlier documentation-only boundary for this increment.
- Base: `dfe8b3ea81d874ff4196807c165b6e56493af13f`, clean
  `codex/refactor-mainline`; application behavior remains the accepted
  `fc9c1f2935b433782ae3cfa1a79e2ac0247dd8f1` generation.
- Candidate: `codex/point-policy-foundation-20261106`, one active spine
  increment. Initial staged/unstaged/untracked state clean. Other retained
  candidates remain evidence; no scientific-module probe is started.
- Worktree: `/Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-point-policy-foundation-20261106/source`.
- Scientific authorities preserved: `doc/SCIENTIFIC_CONVENTIONS.md`, existing
  POINT-EXPECTED method and accepted POINT application/metadata contracts.
  The decision table at `doc/POINT_METHOD_AND_USE_ACCEPTANCE_2026-11-04.md`
  records agreed future rules; unresolved proposals in the broader scope are
  not adopted by this work. `validation/wp7_timestream_successor_authority.json`
  remains the native scientific authority router; no native algorithm changes.
- Architecture: `doc/ARCHITECTURE.md`; observation-local final publication
  owns the small fact snapshot. Existing fit/map producers retain scientific
  authority. No `Engine` state, global mutable state, generic policy framework,
  extra fitting or new runtime dependency.
- Evidence boundaries: `doc/APPLICATION_BASELINES.md`, `doc/RETAINED_DEBT.md`;
  preparation, software conformance and scientific/use acceptance stay separate.

## Scope, interfaces and lifecycle

Add a pure final-measurement adapter plus one required staged JSON product.
Reuse the caller's in-memory maps/guards and existing table outcome. Bind the
serialized report in product records and `RUN.json`; fail publication if the
report cannot be written. Inputs remain unmodified and scope is one observation.
Add the module to the explicit bundle/wheel inventory. Keep parameter names and
units shared with the existing fit diagnostic so the two descriptions agree.

The [handoff and schema](../../doc/POINT_POLICY_FOUNDATION_HANDOFF_2026-11-06.md)
define validity, identity, units, axes and missing-value meanings. No thresholds,
selection, annular statistics, calibration, partial-array route, feedback,
TolTECA consumer, SCIENCE, CFITSIO, numerical kernels or environment changes.
The existing per-array table continues to own its current handoff decision.

## Gates and risks

| Risk | Gate and disposition |
| --- | --- |
| Changed scientific values or signs | Focused preservation assertions for all seven parameters, centroid axes and uncertainty meanings; normal publisher reuses guards. Pending Grant. |
| False success or ambiguous unavailable state | Failed/unavailable/not-requested/not-produced cases; malformed accepted guards fail; future evaluation remains explicitly pending. Pending Grant. |
| Report write failure publishes a completed observation | Real publisher and CLI with injected file-open failure; final rename and transport cleanup must not occur. Pending Grant. |
| Missing packaged module or broken entry path | Full configured POINT Python suite, packaging inventory/launcher tests under retained Mac Spack environment. Pending Grant. |
| Broader regression | Build/version binding, enabled CTest, baseline-tool tests, full required config preflight. Pending Grant. |
| New per-pass I/O or performance loss | Static caller trace: one final report, no additional scientific input reads or fits. Matched 152389 M0–M5 time/memory and exact scientific products after software evidence. Pending Grant. |
| Concurrent scientific design changes | Separate checkout ownership; exact approved design revision required before rule implementation. No automatic merge or policy choice. |

Unexpected numerical, failure, scientific-policy, dependency or scope effects
require recorded reassessment. A scientific choice, consumer change or broader
algorithm change is split to its own owner-reviewed increment. The intentional
new schema is bounded by the explicit owner authorization and handoff above.

Before requesting acceptance: fresh-context independent read-only review of one
exact candidate SHA; all runtime execution stays with Grant. Report separate
scientific/behavioral, architecture/ownership and repository/evidence dispositions.
No runtime gate has been run by Codex or inferred to pass from preparation.
Owner acceptance is required for subsequent local integration.
