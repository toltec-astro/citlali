# ADR 0003: Session Lifecycle And Engine Compatibility

- **Status:** Accepted
- **Recorded:** 2026-07-16
- **Decision owners:** Citlali project owner and engineering

## Context

Citlali's established mode engines combine calibration, telescope, RTC/PTC,
mapmaking, buffers, output, config, and progress state. Replacing that
aggregate while simultaneously changing lifecycle and scientific algorithms
would create excessive validation risk. At the same time, process-static
profiling, cursors, scattered resets, and ambiguous observation state prevented
reliable repeated use.

The operational requirement is sequential repeated reductions in one process,
including a successful run after a failed run. Concurrent sessions in one
process are not currently required.

## Decision

`citlali::session::ReductionSession` is the non-copyable owner of one
sequential run lifecycle. It owns run state and the `StageProfileCollector`,
rejects nested use, converts escaping errors to `ReductionResult`, and resets
run-owned services before each operation.

Each standard operation loads fresh reduction inputs and constructs a fresh
selected `TimeOrderedDataProc`. That processor owns its `Lali`, `Pointing`, or
`Beammap` engine by value. Iteration, observation, scan/chunk, and ordered
writer state use the narrowest available local owner or context. A filesystem
`OutputRootLease` prevents competing processes from publishing into the same
root.

The current `Engine` remains an active compatibility boundary around the
validated numerical implementation. It is frozen for growth:

- no new public cross-cutting mutable state;
- no process-lifetime run or observation state;
- new orchestration receives narrow requests, plans, contexts, and results;
- typed objects stored in `Engine` retain their named lifecycle authority; and
- fields leave the aggregate only through bounded, tested, mode-validated
  changes.

## Consequences

- Sequential runs start with fresh config, coordinator, mode, profiling,
  cursor, and output ownership.
- The project can improve ownership without rewriting mature RTC/PTC,
  mapmaking, or Beammap algorithms.
- Existing arbitrary engine access is retained transitional debt, not an API
  pattern for new work.
- Concurrent in-process reductions require a separate decision and proof for
  logger, FFTW, dependency, memory, and output isolation.
- A smaller file or class is not automatically a better boundary; extraction
  needs a named owner or contract benefit.

## Rejected Alternatives

- **Rewrite `Engine` before establishing the session:** changes too many
  scientific and lifecycle variables at once.
- **Add another global service/state bag:** relocates rather than fixes
  ownership.
- **Scatter cleanup calls after each run:** recovery remains path-dependent and
  cannot be guaranteed by construction.
- **Require concurrent sessions now:** adds constraints with no operational
  caller.

## Supersession

`Engine` may eventually disappear or become a narrow mode facade. Such a
change is consistent with this ADR if the explicit session and subordinate
lifetime owners remain. Concurrent-session support requires a new ADR.

## Evidence

- [`../PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md`](../PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md)
- [`../PHASE3_SESSION_EXIT_CENSUS_2026-07-15.md`](../PHASE3_SESSION_EXIT_CENSUS_2026-07-15.md)
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- `tests/test_reduction_session.cpp`
- `tests/test_output_root_lease.cpp`
