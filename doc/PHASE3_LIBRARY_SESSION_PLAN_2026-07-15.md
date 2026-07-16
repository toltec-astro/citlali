# Phase 3 Library And Session Plan

This document records the bounded ownership and compilation work for Phase 3.
It supplements the governing exit gates in `REFACTOR_STATUS.md`; it does not
authorize scientific algorithm changes or renewed open-ended header splitting.

## Starting Shape

The standard executable already has a thin `main`, and each invocation creates
a selected TOD processor that owns its mode-specific engine by value. The
scientific pipeline returns a Boolean completion signal, but loading, processor
selection, exception policy, diagnostics, provenance finalization, and process
exit status still meet in the CLI header layer.

The first Phase 3 boundary introduces:

- `citlali::session::ReductionSession` as the non-copyable owner of one
  sequential run lifecycle;
- `citlali::session::ReductionResult` as the structured success/failure,
  diagnostic, product-root, and provenance-artifact result;
- a standard session entry that constructs fresh reduction inputs and a fresh
  selected engine for each run; and
- a CLI-only adapter that prints result diagnostics and chooses the process
  exit code.

This is deliberately a narrow facade over the validated pipeline. It does not
claim that all internal failures are classified or that every produced file is
enumerated yet.

## Lifecycle Ownership Census

| Scope | Current owner | Phase 3 disposition |
| --- | --- | --- |
| Process/CLI | argument parsing and run-environment helpers | Keep process control, help/version exits, terminal reporting, and exit-code policy here. |
| Reduction run | new `ReductionSession`; selected TOD processor owns a fresh engine by value | Make this the stable non-CLI lifecycle/result boundary. Reject nested use; support clean sequential runs. |
| Fruit-loop iteration | local `ReductionIterationState` plus compatibility fields in `Engine::iteration` | Preserve the local owner; move only lifecycle facts needed to remove scattered resets. Do not redesign fruit-loop numerics. |
| Observation | local KIDs processor and `ReductionObservationContext`; `ObservationRuntimeState` remains embedded in `Engine` | Name the context as owner and narrow adapters into engine state one cluster at a time. |
| Scan/chunk | scan contexts and processor-local cursors, with remaining mutable state spread through RTC/PTC and engine detail | Convert only boundaries needed for reentrancy or reachable-exit removal. Keep hot numerical loops unchanged. |
| Ordered output | `OrderedWriter` and explicit output/provenance execution plans | Preserve Phase 1 cancellation and required-write contracts; make failures populate the session result. |
| Profiling | run-owned `StageProfileCollector`, passed explicitly through production scopes | Verify unchanged sidecar behavior on Unity. Profiling remains optional and must not change reduction success. |
| Configuration | Phase 2 requested/effective/realized plans, stored through the compatibility engine | Freeze Phase 2 authority. Do not resume control migration in Phase 3. |

`Engine` is frozen as a compatibility boundary: Phase 3 may remove or narrow
public state, but it must not add a new cross-cutting state bag merely to avoid
passing an explicit owner or context.

## Bounded Work Sequence

1. Establish and test the session/result facade without changing successful
   scientific execution or output ordering.
2. Inventory process termination reachable from the standard non-CLI entry.
   Convert exits by coherent boundary, starting with setup/validation paths;
   retain CLI-only help/version termination.
3. Move the process-static stage profiler under run ownership and prove two
   sequential runs cannot share records or output paths.
4. Tighten observation and scan ownership only where the exit/reentrancy work
   identifies a concrete stale-state or dependency hazard.
5. Add independent-header and multi-translation-unit checks. Repair discovered
   include-order or ODR defects without broad textual extraction.
6. Select one measured coherent declaration/validation tranche for a real
   `.cpp` boundary. Record dependency/build evidence before and after.
7. Re-run local gates, then use a Unity point reduction for the session cut;
   add science or Beammap only when the touched boundary is mode-specific.

Step 3 is implemented locally as one atomic ownership cutover.
`ReductionSession` owns and resets the collector, and that explicit owner now
reaches every production profiling scope and sidecar operation. The temporary
process-static collector and implicit adapter are deleted. Sequential-run tests
prove reset isolation, and a pipeline test verifies representative reduction,
observation, and map-output records in the supplied collector. Local CLI and
test builds and full config preflight pass. Unity point `redu63` accepts step 3
with exact scientific products and the same 76 profile stage/context records as
`redu62`; elapsed values and concurrent chunk completion order are intentionally
volatile.

Step 4 found and repaired one concrete ownership defect: the scientific
pipeline reset the collector already owned and reset by `ReductionSession`.
The inner reset is deleted, and a regression test proves pre-pipeline records
survive. No broader observation or scan state moved without evidence.

The first step 6 candidate moves timestream enum name tables and parse/format
definitions from the public header into a compiled source. The header shrank
from 946 to 712 lines while retaining small predicates inline. An immediate
single before/after CLI compile pair was 62.4 versus 63.7 seconds, which shows
no demonstrated speedup or material regression. CLI, primary test, and safety
test targets build. Unity compile and point `redu63` accept the boundary with
zero product differences and no attributable runtime regression. A larger cold
tranche may be considered later only if dependency and build evidence justify
it.

The first mature-implementation contract tranche retires the two PTC weighting
exits and one RTC kernel setup exit without changing valid-path arithmetic.
The following bounded tranches retire all 40 fruit-loop and all eleven Wiener
exits without changing valid-path arithmetic. Their public validation contracts
compile in isolation; focused category, recovery, geometry, identity, and
allocation tests pass. OpenMP allocation failure is captured within workers
and rethrown only after the parallel region. The conservative session audit now
reports zero library and zero CLI exits. All three local targets build, all 455
CTests pass, and full config preflight passes. The audit was then widened from
session-reachable headers to every core implementation source; it exposed and
retired three APT input exits and one invalid Lissajous chunk exit. Manual
review confines the remaining textual exits to successful CLI help/version
handling and two main programs excluded from CMake. Standard point, full-Wiener
point with noise maps and `lowpass_only: false`, fruit-loop science, and Beammap
validation remain before Phase 3 closeout. A science config that selects
`type: wiener_filter` but retains `lowpass_only: true` exercises the convolution
path and is not evidence for denominator construction.

Unity refactor point `redu65` and matched OG point `redu10` close the full-
Wiener production gate. Their 490-leaf low-level trees differ only in the OG/
refactor output and telescope-file paths; the two telescope files have the same
SHA-256. Both runs execute six Wiener core calls with five noise realizations
and `lowpass_only: false`. All seven filtered products pass the strict
`2e-8 + 1e-10 * abs(reference)` gate without skipped records, and the complete
three-array pointing-fit table is exact. The candidate reports no issues. The
OG run retains its known twelve legacy NetCDF errors. Fruit-loop science and
Beammap validation remain before Phase 3 closeout.

## Stop Rules

Phase 3 is complete when the governing exit gates pass and the following are
true:

- the reusable entry returns a structured result without process policy;
- every process termination reachable through supported library execution is
  removed or mechanically proven unreachable and allowlisted;
- run-owned mutable state, including profiling, is reset by construction;
- two sequential session runs recover after an injected failure;
- public boundary headers compile independently and link from multiple
  translation units; and
- one evidence-backed compiled boundary reduces exposed dependencies without
  material local-build, Unity-build, or reduction-runtime regression.

Further file movement stops when it would only rename or relocate contextual
implementation. Concurrent reductions, install/export support, compact-config
rollout, broad numerical cleanup, and R execution are not Phase 3 requirements.
