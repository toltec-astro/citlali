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
installing chunk lists or counts. The adjacent TOD row-selection tranche
removes three more exits: invalid effective modes and out-of-range chunks are
canonical config errors, while an empty source-crossing selection is a
canonical runtime error. Valid row assignment is unchanged. The current audit
reports 85 library exits; the 94-exit baseline remains intentionally frozen as
a no-growth ceiling that permits monotonic retirement.

The first observation-input tranche removes six duplicated exits after KIDs
matrices are populated by direct, loaded, and gap-aligned paths. A shared
finite-input contract preserves valid matrices and classifies NaN or infinite
input as canonical I/O failures. The current audit reports 79 library exits.

The second observation/input tranche removes the remaining eight exits in
this census group. Explicit contracts now cover detector cardinality,
cross-network sample-rate agreement and units, valid gap-alignment rate,
derived extinction, polarization calibration availability, IIR/Nyquist
compatibility, and Beammap fit-map shape. Valid setup paths are unchanged and
the current audit reports 71 library exits.

The first required-output tranche removes nine FITS image and PHDU slot exits.
Missing map, file, Stokes, array, FWHM, noise, and PHDU slots now log their
specific diagnostic and throw a canonical output error. Valid lookup and write
paths are unchanged. The current audit reports 62 library exits.

The FITS/ECSV adapter tranche removes four more executable exits and classifies
other adapter exceptions that were not visible to the exit scan. CCfits input
and output operations now catch its actual `FitsException` base hierarchy;
ECSV input and required atomic publication use canonical I/O and output errors.
The final apparent exit in this group belonged to a fully commented, unused
Gaussian transfer-function prototype and was removed with that dead block. The
current audit reports 57 library exits.

The three remaining non-kernel mapmaking preconditions are retired next.
Unsupported polarization/grouping and Beammap pixel-axis requests plus missing
Wiener template FWHM values now use canonical config errors. The current audit
reports 54 library exits, all in mature numerical implementations.

## Retirement Order

1. Config and output-selection readers: the six TOD selection-config exits and
   three TOD row-selection exits are retired. Invalid requests now reach
   `ReductionResult` through canonical errors or path-aware config diagnostics.
2. Observation/input setup: all fourteen exits across observation setup,
   KIDs/raw loading, gap alignment, and Beammap fit preparation are retired
   behind focused, canonical validation contracts.
3. FITS/ECSV and output-slot validation: all fourteen reported exits are
   retired or, for one commented prototype, proven non-executable and removed.
   Required failures preserve the Phase 1 ordered-writer cancellation contract.
4. Mapmaking policy and template setup: all three exits remaining after the
   baseline setup slice are retired as config preconditions.
5. Mature numerical implementations: 40 exits in `timestream.h`, eleven in the
   serial/OpenMP Wiener implementations, two in PTC, and one in the RTC kernel.
   Convert these only in coherent algorithm-boundary tranches with matched
   mode validation; do not mechanically replace them en masse.

The checked baseline is per file, so exit counts may decrease but cannot grow
or move to a new dependency-reachable library file unnoticed. Phase 3 closure
still requires manual proof for unreachable legacy paths and zero process
termination on every supported non-CLI execution path.
