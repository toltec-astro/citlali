# Phase 3 Session Exit Census

This is the starting process-termination census for the standard non-CLI
reduction entry. Run:

```bash
$HOME/tolteca/bin/python tools/refactor/audit_session_exits.py \
  --fail-on-growth
```

The current audit follows Citlali project-header includes from
`standard_reduction_execution.h` and scans every implementation source under
`src/citlali/core`. It is deliberately conservative: dependency reachability
does not prove that every reported function is runtime-call reachable for every
mode, while source inclusion does not prove that every definition is linked by
every target. Together they prevent header-only or source-only library exits
from escaping the mechanical no-growth gate.

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

The first bounded mature-implementation tranche retires three more exits.
Non-contiguous PTC network grouping is now a canonical input I/O failure,
impossible PTC weight counters are an internal failure, and mismatched RTC
kernel-image cardinality is invalid configuration. Focused tests cover valid
and invalid contracts; all 451 CTests and full config preflight pass. The
current audit reports 51 library exits.

The fruit-loop map-ingestion tranche retires all 37 exits in `TCProc::load_mb`.
Unsupported requested grouping is invalid configuration; file discovery,
required FITS metadata and schema, map identity/cardinality, WCS compatibility,
and filesystem failures are input I/O errors. Optional `GROUPING` and `RADESYS`
keywords now ignore only `CCfits::HDU::NoSuchKeyword`, so canonical exceptions
cannot be swallowed by the legacy optional-key boundary. Valid map loading and
numerical processing are unchanged. Focused category and sequential-session
recovery tests pass, all 453 CTests and full config preflight pass, and the
current audit reports 14 library exits. Matched science and Beammap fruit-loop
reductions remain required before accepting this tranche.

The adjacent fruit-loop feedback tranche retires its final three exits without
changing interpolation or map-to-TOD arithmetic. Non-contiguous calibration
groups, unknown detector-array identities, and out-of-range map indices are
explicit input-I/O invariants checked before affected data access. The public
contract compiles in isolation, focused valid/invalid tests pass, all 454
CTests pass, and the current audit reports 11 library exits. This tranche uses
the same pending matched science and Beammap fruit-loop acceptance runs.

The Wiener tranche retires the final eleven exits behind a shared, header-
isolated runtime-invariant boundary. Serial and OpenMP template geometry,
pixel spacing, kernel/weight identity and shape, and finite kernel peak checks
now propagate through `ReductionSession`. FFTW allocation is checked before
plan creation and partial resources are reset before failure. Because C++
exceptions may not escape an OpenMP worker, each worker captures allocation
failure inside the parallel region, all workers synchronize before entering
the worksharing loop, and the owning thread rethrows only after the region
ends. Valid FFTW, template, and numerical paths are unchanged. All three local
targets build, all 455 CTests and full config preflight pass, and the audit now
reports zero reachable library or CLI exits. Point, science, and Beammap mode
validation remain required before accepting the final mature tranches.

The closeout audit widened the original header-only baseline scope to include
all core implementation sources. That correction exposed four exits that the
first census missed: three invalid APT-table paths in `Calib::get_apt` and one
invalid Lissajous chunk request in `Telescope::get_tel_data`. They now throw
canonical I/O or invalid-config errors without changing valid inputs. The only
remaining textual exits are successful `--help` and `--version` handling in the
active CLI parser and the same successful paths in `main_old.cpp` and
`kids_main.cpp`; CMake builds neither legacy main. No supported non-CLI
execution path retains explicit process termination.

## Remaining Tranche Classification

The mature exit census is locally closed. The table retains the tranche-level
acceptance obligations rather than treating a zero static count as sufficient.

| Tranche | Exits | Status | Boundary | Minimum validation before acceptance |
| --- | ---: | --- | --- | --- |
| Fruit-loop map ingestion | 0 of 37 | Retired locally | Required map-file discovery, FITS metadata/schema, grouping identity, WCS, and map cardinality in `TCProc::load_mb` | Focused malformed-input tests pass; matched science and Beammap fruit-loop reductions pending |
| Fruit-loop grouping/application | 0 of 3 | Retired locally | Non-contiguous detector grouping and map/array identity during map-to-TOD feedback | Focused invariant tests pass; science and Beammap fruit-loop reductions pending |
| Wiener filtering | 0 of 11 | Retired and full-Wiener point-validated | Template geometry, kernel-map identity, finite kernel peak, and OpenMP FFTW allocation | Matched OG `redu10` and refactor `redu65` execute all six expected OpenMP filter calls with noise maps and `lowpass_only: false`; seven filtered products pass the strict scientific-tolerance gate and the pointing-fit table is exact |
| PTC weighting | 0 of 2 | Retired and point-validated | Network-group contiguity and impossible weight-counter state | Add science when the active weighting policy differs |
| RTC kernel setup | 0 of 1 | Retired locally | FITS kernel image cardinality | Kernel-enabled point or Beammap run |

The preferred retirement order is PTC weighting, RTC kernel setup, fruit-loop
map ingestion, fruit-loop application, then Wiener filtering. This order starts
with small contracts that are exercised by inexpensive validation and leaves
the broadest file-I/O and implementation-specific boundaries until their
fixtures and mode runs are ready. A tranche may move earlier only when a real
failure or an already scheduled mode validation makes its evidence cheaper.

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
5. Mature numerical implementations: the PTC, RTC kernel, all 40 fruit-loop,
   and all eleven serial/OpenMP Wiener exits are retired locally in coherent
   boundary tranches. Their matched mode validation remains open.

The checked baseline is per file, so any new dependency-reachable or core-source
exit fails the no-growth gate. The source and legacy-main review completes the
manual stop-line proof. Phase 3 closure still requires the listed matched mode
validation; every supported non-CLI execution path is now free of explicit
process termination.
