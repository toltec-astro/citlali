# Phase 3 Session Exit Census

This is the starting process-termination census for the standard non-CLI
reduction entry. Run:

```bash
$HOME/tolteca/bin/python tools/refactor/audit_session_exits.py \
  --fail-on-growth
```

The audit follows Citlali project-header includes from
`standard_reduction_execution.h`. It is deliberately conservative: dependency
reachability does not prove that every reported function is runtime-call
reachable for every mode. It does prove that the reusable boundary still
exposes those process-termination definitions and gives Phase 3 a mechanical
no-growth gate.

## Starting Result

- Project dependencies: 667
- Direct dependency-reachable exits: 94
- Library exits: 94
- CLI exits: 0
- Files containing library exits: 22

The initial Phase 3 setup slice removed eight direct exits before this baseline
was frozen: two unsupported top-level config paths now enter structured config
diagnostics; two map-filter prerequisite failures throw canonical config
errors; three tone-frequency inventory failures throw canonical I/O errors;
and a missing timing-gap mask throws a canonical runtime error.

The first post-baseline retirement removes the six TOD output-selection config
exits. Invalid values now append path-aware diagnostics without partially
installing chunk lists or counts. The current audit reports 88 library exits;
the 94-exit baseline remains intentionally frozen as a no-growth ceiling that
permits monotonic retirement.

## Retirement Order

1. Config and output-selection readers: six exits in TOD selection config and
   three in TOD row selection. Invalid requests should reach `ReductionResult`
   with actionable paths.
2. Observation/input setup: fourteen exits across observation setup, KIDs/raw
   loading, gap alignment, and Beammap fit preparation. Convert by owning
   boundary with focused malformed-input and recovery tests.
3. FITS/ECSV and output-slot validation: fourteen exits. Required output
   failures should become canonical output errors and preserve the Phase 1
   ordered-writer cancellation contract.
4. Mapmaking policy and template setup: five exits outside the mature filter
   kernels. Treat these as config/runtime preconditions.
5. Mature numerical implementations: 40 exits in `timestream.h`, eleven in the
   serial/OpenMP Wiener implementations, two in PTC, and one in the RTC kernel.
   Convert these only in coherent algorithm-boundary tranches with matched
   mode validation; do not mechanically replace them en masse.

The checked baseline is per file, so exit counts may decrease but cannot grow
or move to a new dependency-reachable library file unnoticed. Phase 3 closure
still requires manual proof for unreachable legacy paths and zero process
termination on every supported non-CLI execution path.
