# Phase 4 Closeout Census - 2026-07-16

## Purpose

This census maps every required broader-refactor criterion in section F.2 of
the adopted
[`EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md`](../handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md)
to current evidence, an explicit project-owner scope decision, a deliberate
deferral, or a concrete remaining action.

It is a stop-control document. It prevents Phase 4 from expanding into more
pipeline decomposition after the required runtime architecture is already
supported by evidence. The living phase decision remains
[`REFACTOR_STATUS.md`](REFACTOR_STATUS.md).

## Result

The 15 required F.2 criteria currently divide as follows:

- **10 closed by implementation and evidence**;
- **2 closed by an explicit scope decision or proportionality exception**;
- **3 compilation-dependent criteria deliberately deferred** pending review
  of TolTECA's revised C++ integration approach; and
- **0 open compilation-independent criteria**.

These counts are not a percentage-complete claim. The three deferred build
criteria remain material unresolved work despite completion of the non-build
package.

## Status Vocabulary

- **Closed:** the implementation, local gate, and required operational
  evidence exist.
- **Closed by scope/exception:** the project owner made the choice allowed by
  F.2 and the limitation is recorded.
- **Deferred:** the criterion remains required if its scope is retained, but
  work is paused by an explicit sequencing decision.
- **Open:** work can proceed now without crossing a deferral or changing
  scientific execution.

## F.2 Evidence Matrix

| # | Required criterion | Status | Current evidence and remaining boundary |
| ---: | --- | --- | --- |
| 1 | Non-CLI session accepts explicit input and returns structured result; CLI alone selects exit code | **Closed** | `ReductionSession` and `ReductionResult` own sequential run status, diagnostics, product roots, and published provenance paths. Standard loading and mode selection run inside the session operation. Session/CLI separation, exceptions, nested use, and sequential runs are tested. Phase 3 point `redu66` accepts the boundary. The standard reusable adapter remains under `citlali::cli`, a naming/layout debt rather than process-policy leakage. |
| 2 | No supported non-CLI process termination | **Closed** | `tools/refactor/audit_session_exits.py --fail-on-growth` reports zero library and zero CLI exits in 691 supported dependencies and scans every core implementation source. Remaining success exits are confined to active CLI help/version handling and unbuilt legacy mains. |
| 3 | Required output failure propagation, cancellation, partial-product policy, and recovery | **Closed** | Required NetCDF, FITS, ECSV/CSV, manifest, and provenance failures use canonical failure boundaries. Ordered-writer tests cover same-stream and cross-stream cancellation, waiter wake-up, explicit partial products, nonzero CLI result, and successful recreation in the same process. Product contracts independently check requested output completeness. |
| 4 | Explicit run, iteration, observation, scan, and writer owners; sequential recovery | **Closed** | The session owns run state and profiling; local iteration and observation contexts own bounded lifecycle state; scan cursors are invocation-owned; writers and the output-root lease own publication state. Tests cover two sequential runs, profiler reset, injected-failure recovery, cursor restart, writer recovery, and lease release. |
| 5 | `Engine` frozen as a compatibility adapter; new stages use narrow plans/contexts | **Closed** | The rule is governing policy in `AGENTS.md` and [`ARCHITECTURE.md`](ARCHITECTURE.md). Phase 2 plans and Phase 3 lifecycle contexts provide the new authority while one-way adapters preserve established processors. Existing broad engine access remains transitional debt; adding new cross-cutting public state is prohibited. |
| 6 | Public-header isolation, multi-TU linkage, and bounded private contextual fragments | **Deferred** | The new session, validation, Wiener, fruit-loop, and output-root boundaries have isolation and multi-TU tests, and concrete include-order defects were repaired. The repository does not mechanically compile all 700 historical headers in isolation, and 171 contextual `engine/detail` fragments remain under the public include tree. Expanding the compile matrix or physically privatizing fragments is compilation-side work and remains paused with that deferral. No new contextual-fragment family may be added in the meantime. |
| 7 | Meaningful measured cold `.cpp` boundary that reduces CLI closure/build time | **Deferred** | Timestream enum parsing moved from a 946-line header into `src/citlali/core/config/timestream_enums.cpp`; the public header fell to 712 lines and products/runtime were unchanged. The immediate compile pair was 62.4 versus 63.7 seconds, so no build-time improvement was demonstrated and the CLI remains header-dominant. A broader boundary must wait for the TolTECA build-direction review. The existing result is retained as neutral evidence, not described as a speedup. |
| 8 | Active, legacy, generated, and transitional files are distinguishable | **Closed** | [`ARCHITECTURE.md`](ARCHITECTURE.md) prominently classifies the active target/entry, active transitional engine/header graph, unbuilt historical mains, placeholders, generated config headers, and deferred paths. `refactor_inventory.py` independently reports the active header/source shape and seven commented CMake entries. Physical removal or relocation may occur in Phase 5 after preserving the validated tree. |
| 9 | Checked cold-boundary scientific invariants for touched subsystems | **Closed** | [`SCIENTIFIC_CONVENTIONS.md`](SCIENTIFIC_CONVENTIONS.md), `validation/product_contracts.json`, the 574-leaf config contract, and focused C++ safety/config tests define identity, shape, units, frames, index bases, finite/missing policy, and product schema for the boundaries touched by the refactor. Incomplete NetCDF fill/unit and ECSV unit metadata are explicit successor-schema debt, not unspecified current semantics. |
| 10 | Clean pinned-dependency build and real tests/config tools in CI with exact version/dependency identity | **Deferred** | Local CLI and test builds work and 460 CTests plus config/baseline gates are active. Dependencies, clean CI lanes, embedded version regeneration, and build reproducibility are not closed. The project owner explicitly deferred CMake, dependency, preset, CI-build, install, and cluster-helper work until TolTECA's revised C++ integration direction is reviewed. |
| 11 | Install/export smoke client, or explicit statement that external library consumption is not a goal | **Closed by scope** | The supported deployment is the Citlali CLI on the small set of collaborating clusters. The static `citlali` target is currently an internal composition and test boundary, not a promised stable installed API/ABI. Replicable cluster deployment remains desirable, but external library consumption is not an accepted requirement. Revisit this decision after the TolTECA integration review rather than adding install/export rules speculatively. |
| 12 | Current strict point, Beammap, science, and OOF gates; enabled polarimetry validated or rejected | **Closed** | `validation/validation_profiles.json` contains four active immutable mode profiles. The unified validator runs completion/provenance, exact config, product-contract, and numerical gates and fails on missing, extra, skipped, changed, or error records. Enabled polarimetry is mechanically rejected before execution until an approved reference contract exists. |
| 13 | Controlled performance/RSS evidence meets policy or has approved exception | **Closed by exception** | The wrapper records exact run identity, GNU Time wall/RSS/I/O, Citlali time, stages, config/input identity, and runtime policy. Point `redu67` demonstrated the wrapper with 908,316 KB peak RSS and exact products. Twelve Beammap checkpoints show no sustained regression. The owner approved a proportionality exception to a dedicated campaign on shared VAST; Beammap RSS and profiler-overhead experiments are trigger-based requirements for the next natural run or a real regression signal, not Phase 4 prerequisites. |
| 14 | Intended non-structural/science imports have a checked ledger | **Closed** | `validation/intended_science_changes.json` records the determinism repair, Wiener optimization series, and active-detector PCA optimization with source/integration commits, expected effect, affected modes/products, and accepted evidence. The validator checks ancestry, patch identity where applicable, evidence IDs, and product-family references. |
| 15 | Durable section-H documentation is current and retained debt has owner/exit condition | **Closed** | `AGENTS.md`, [`ARCHITECTURE.md`](ARCHITECTURE.md), [`SCIENTIFIC_CONVENTIONS.md`](SCIENTIFIC_CONVENTIONS.md), [`REFACTOR_STATUS.md`](REFACTOR_STATUS.md), [`RETAINED_DEBT.md`](RETAINED_DEBT.md), the five focused [`adr`](adr/README.md) records, validation ledger, profiles, product contracts, and science-change ledger form the canonical set. Root `CODEX.md` is now a concise redirect with no contradictory historical workflow. Each retained debt has a role owner, reopening trigger, and observable exit condition. |

## Deferred Compilation Package

Criteria 6, 7, and 10 must be reconsidered together after the TolTECA build
direction is available. Treating them as independent cleanup tasks would risk
optimizing the wrong build topology.

That later review must answer:

1. Is Citlali built as a standalone project, a TolTECA-managed dependency, or
   part of a larger C++ workspace?
2. Which dependency pins and patches are authoritative on the supported
   clusters?
3. Which headers are truly public to another target, and which implementation
   fragments can become private?
4. Is the static library an internal target only, or must it be installed and
   consumed externally?
5. Which clean-build and incremental-build measurements reflect the intended
   deployment workflow?
6. Which CI environment can reproduce at least one supported cluster/compiler
   lane without pretending to emulate all four sites?

Until those answers exist, do not alter CMake targets, presets, dependency
fetching, CI build lanes, install/export rules, or cluster build helpers.

## Completed Compilation-Independent Closeout

The finite non-build package is complete:

1. Five focused ADRs record the immutable config transition, structured
   result/output-failure contract, frozen `Engine` and session ownership, first
   compiled-boundary/header policy, and deferred measured R channel.
2. `CODEX.md` is a concise redirect to `AGENTS.md`,
   [`ARCHITECTURE.md`](ARCHITECTURE.md), and
   [`REFACTOR_STATUS.md`](REFACTOR_STATUS.md); historical invalid build
   instructions remain only in Git history.
3. [`RETAINED_DEBT.md`](RETAINED_DEBT.md) assigns every deliberate limitation
   a role owner, reopening trigger, and exit test.
4. The complete local CTest, config-preflight, baseline-tool, registry, and
   session-exit gates pass after the documentation set was assembled.
5. Compilation-independent Phase 4 work now stops. The next Phase 4 action is
   the build-direction review, not another structural tranche.

## Retained Debt

[`RETAINED_DEBT.md`](RETAINED_DEBT.md) is the canonical register. It includes
the broader `Engine`/pipeline/header graph, deferred build and CI work, product
metadata, polarimetry, measured R, compact config, fruit-loop run identity,
external library scope, concurrent sessions, triggered performance evidence,
and the future Beammap corpus. Do not duplicate or silently relax those exit
conditions here.

## Phase Boundary

Criterion 15 is closed and compilation-independent Phase 4 work is done. Phase
4 as a whole remains active but waiting on the deferred compilation package.
Phase 5 must not begin by silently waiving criteria 6, 7, and 10; it may begin
only after those criteria are closed or the project owner records a final
explicit exception based on the reviewed TolTECA integration model.

While the TolTECA build owner is unavailable, the project may perform the
bounded preparation in
[`PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md`](PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md).
Preparation inventories source disposition and defines the same-SHA validation
and integration packet. It does not change compilation infrastructure, close
the deferred criteria, freeze/tag a candidate, or authorize integration.
