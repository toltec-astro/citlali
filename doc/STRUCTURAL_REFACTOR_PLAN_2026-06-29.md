# Citlali Structural Refactor Plan - 2026-06-29

This plan starts the long-running structural refactor from
`handoff/HANDOFF_2026-06-29_STRUCTURAL_REFACTOR.md`.

Branch/worktree:

- Worktree: `/Users/gwilson/GitHub/citlali-refactor`
- Branch: `codex/structural-refactor`
- Baseline commit: `376e0022` (`adding weight cuts to fruitloops`)
- Remote branch: `origin/codex/structural-refactor`

## Scope

Refactor Citlali toward a maintainable library and CLI structure while
preserving science behavior and runtime performance.

Primary goals:

- Preserve the existing YAML config format and user-facing CLI behavior.
- Move orchestration out of `src/citlali/cli/main.cpp` into library code.
- Introduce typed config adapters and validation boundaries around the current
  YAML tree.
- Replace direct process exits in library paths with typed failures handled at
  the CLI boundary.
- Split very large public headers into smaller declarations, implementation
  files, and narrowly included detail headers.
- Reduce shared mutable state by introducing explicit reduction contexts for
  observations, calibration, timestreams, mapmaking, output products, and
  iteration state.
- Add characterization tests and Unity reduction baselines before changing
  risky behavior.

## Non-Goals

- This is not a rewrite of the science pipeline.
- Do not change numerical algorithms unless a separate science reason is
  documented and reviewed.
- Do not change the YAML schema in a breaking way during the structural phase.
- Do not add abstraction inside detector/sample/map-pixel hot loops without a
  Unity benchmark.
- Do not attempt authoritative local configure/build/test validation on the
  workstation; use Unity for compile, runtime, and reduction validation.

## Current Architecture Findings

The refactor should be driven by the current code shape:

- `include/citlali/core/engine/engine.h` is about 9.7k lines and exposes a
  broad public mutable surface by inheriting from multiple state/control bags.
- `include/citlali/core/engine/beammap.h`,
  `include/citlali/core/timestream/rtc/rtcproc.h`, and
  `include/citlali/core/timestream/ptc/ptcproc.h` are each several thousand
  lines and mix configuration, state, diagnostics, IO, and runtime algorithms.
- `src/citlali/cli/main.cpp` is about 1.3k lines and currently owns argument
  handling, config merging, logger/runtime setup, observation preflight,
  fruit-loops iteration control, output directory setup, map/noise allocation,
  coadd/filtering, and product writing.
- `CMakeLists.txt` has natural source files for engine/todproc/kidsproc/lali,
  pointing, beammap, and wiener filtering commented out, so much code is forced
  through public headers.
- Library code still contains many `std::exit(EXIT_FAILURE)` calls, especially
  in engine, timestream, mapmaking, and utility headers.
- Tests are currently utility/filter smoke tests and do not yet protect
  calibration, config validation, alignment, mapmaking, metadata, or end-to-end
  reduction behavior.

## Target Shape

The target structure should be reached incrementally:

```text
CLI
  parse args, initialize logging, call runner, translate failures to exit codes

citlali::pipeline
  PipelineRunner / ReductionSession
  ObservationPlan / ObservationContext
  IterationContext
  OutputLayout and product-writer coordination

citlali::config
  typed structs parsed from existing YAML
  validators with path-rich diagnostics
  adapters for legacy Engine/RTC/PTC/MapBuffer consumers during migration

citlali::engine
  mode-specific science/pointing/beammap behavior
  narrower state contexts instead of one giant public Engine bag

citlali::timestream
  RTC/PTC processors with explicit input/output chunk contracts
  diagnostics and learning state separated from hot kernels where practical

citlali::mapmaking
  stable hot-loop APIs
  map buffer state split from map diagnostics and product writers over time
```

Initial PRs should add boundaries around the existing code rather than moving
large amounts of code at once.

## Validation Policy

All behavior-preserving PRs must state what was compared and where validation
was run.

Authoritative validation:

- Compile and reduction validation happen on Unity.
- Local workstation checks are limited to static inspection, text searches,
  formatting checks that do not require the full toolchain, and documentation.
- If Unity is unavailable, the PR must be marked as unvalidated for runtime
  correctness.

Default behavior tolerance:

- Sequential runs should be bitwise identical where the existing pipeline is
  deterministic.
- Parallel runs may require numeric tolerances because floating accumulation
  order can vary; any tolerance must be recorded per product.
- A refactor PR should default to "no science behavior change." If outputs
  change, the PR must explain the science or bug-fix reason.

Performance budget for runtime-path PRs:

- Standard reductions should not regress wall time by more than about 3-5%
  without maintainer approval.
- Peak RSS should not materially increase.
- Diagnostics and tracing must remain off by default unless they are already
  part of the operational product contract.
- No virtual dispatch, string lookup, YAML access, dynamic allocation, or extra
  logging may be added inside per-sample/per-detector/per-pixel inner loops
  without a benchmark.

## Unity Baseline Protocol

Before structural code changes, establish a baseline at commit `376e0022`.

For each run record:

- Git SHA and branch.
- Exact config files and any local overrides.
- Unity module/environment details.
- Exact command line.
- Number of threads and parallel policy.
- Wall time and peak RSS. Use Slurm accounting or `/usr/bin/time -v` if
  available.
- Output directory.
- Checksums for key FITS/netCDF/ECSV products.
- Numeric summaries for maps and TOD products.
- Science metrics relevant to the reduction type: flux, FWHM, pointing offsets,
  map RMS, detector counts, flagged fractions, source-fit validity, and
  beammap convergence/flag counts.

Suggested baseline matrix:

| Case | Purpose | Required Products |
| --- | --- | --- |
| Science, naive mapmaking, noise off | Standard fast path | raw obs/coadd maps, stats, map summaries |
| Science, naive mapmaking, noise on | Noise-map memory/runtime path | noise maps/products, weight summaries, map RMS |
| Pointing with fruitloops/learning path | Iteration and source-aware path | PPT/ECSV, fitted flux/FWHM/offsets, learning CSVs |
| Beammap detector grouping | Beammap iteration and detector state | APT/PPT outputs, detector maps, flags, convergence |
| Jinc mapmaking | Performance-sensitive gridding path | jinc maps, headers, diagnostics, timing |
| TOD-output-heavy reduction | netCDF writer and IO locking path | RTC/PTC TOD files, stats, sidecar diagnostics |
| Optional polarimetry case | HWPR/polarized map path if operational | polarized maps, HWPR metadata, Stokes summaries |

For each baseline, save a small comparison manifest in the worktree or a
maintainer-approved diagnostics location. Do not commit bulky reduction outputs.

## Artifact Comparison Plan

Create lightweight comparison scripts or notebooks only after the Unity output
locations are known.

Compare:

- FITS HDU presence, dimensions, WCS keywords, unit keywords, and checksums.
- Map arrays: finite count, NaN count, sum, mean, median, stddev, RMS,
  min/max, robust MAD, peak location, coverage support, and selected pixel
  residual statistics.
- netCDF TOD products: dimensions, variable names, attributes, chunk counts,
  flag counts, selected variable checksums/summaries.
- ECSV/CSV tables: row counts, required columns, sorted keys, key metrics, and
  exact values for integer/status columns.
- Logs: missing/invalid config reporting, reduction mode, thread settings,
  timing sections, warning/error counts.

Use exact comparisons for metadata and integer/status products. Use numeric
tolerances only for floating arrays where parallel accumulation or upstream
library behavior prevents bitwise stability, and record the tolerance in the
manifest.

## Refactor Sequence

### PR 0: Planning and Baseline Definition

Deliverables:

- This plan.
- Unity baseline run matrix and record template.
- Tracking issue or epic: `Citlali structural refactor`.
- PR checklist template for affected modes, validation, products compared,
  performance result, and residual risk.

Validation:

- Documentation review only.

### PR 1: Baseline Harness and Comparison Utilities

Deliverables:

- Unity run manifest template.
- Scripts for checksum and numeric product summaries.
- A small comparator that can diff two reduction manifests.
- Optional stage-timing log parser if current logs are sufficient.

Constraints:

- No runtime behavior changes.
- Keep large outputs outside git.

Validation:

- Run the matrix at `376e0022` and store manifests.

### PR 2: Typed Config Adapters

Deliverables:

- Add typed config structs under a new config namespace while preserving the
  existing YAML schema.
- Start with low-risk top-level structs: runtime controls, mapmaking controls,
  noise/coadd controls, and output controls.
- Add path-rich validation errors that can be converted to the current
  missing/invalid key reporting.
- Adapt `Engine::get_citlali_config` to consume typed values in small sections,
  keeping legacy fields populated.

Constraints:

- Do not change config defaults or key names.
- Do not access YAML during processing once typed runtime values are available.

Validation:

- Unit tests for valid config, missing key, bad type, out-of-range value,
  fixed-length vector, and enum validation.
- Unity baseline comparison for any PR that changes runtime parsing behavior.

### PR 3: PipelineRunner / ReductionSession Extraction

Deliverables:

- Move config-file merge, reduction-type dispatch, observation preflight,
  fruit-loops iteration control, output directory setup, and coadd/filtering
  orchestration out of `src/citlali/cli/main.cpp`.
- Keep the CLI responsible for parse/help/version, logging bootstrap, calling
  the runner, and translating typed failures to exit codes.
- Introduce `ReductionSession` or `PipelineRunner` as the single library entry
  point for full reductions.

Constraints:

- Preserve CLI flags, default config dumping, logger names, output directory
  layout, and return codes.
- Move code mechanically first; redesign only after baseline comparisons pass.

Validation:

- Unity compile.
- Full baseline matrix for science/pointing/beammap entry points.

### PR 4: Typed Failure Boundary

Deliverables:

- Define a small failure hierarchy, for example `ConfigError`,
  `DataIOError`, `ReductionError`, and `InternalInvariantError`.
- Convert direct `std::exit(EXIT_FAILURE)` calls by subsystem, starting with
  non-hot config/IO/preflight paths.
- Ensure the CLI boundary catches failures and reports concise messages with
  context.

Constraints:

- Do not convert all exits in one sweep.
- Do not throw through hot loops for expected per-sample control flow.
- Preserve existing failure messages where operational scripts may depend on
  them.

Validation:

- Unit tests for failure conversion in config and synthetic IO/preflight cases.
- Unity smoke reductions that intentionally fail on bad config/input.

### PR 5: Header and CMake Boundaries

Deliverables:

- Move non-template implementation from large headers into `.cpp` files.
- Re-enable natural source entries in `CMakeLists.txt` one at a time:
  engine/todproc/kidsproc/lali/pointing/beammap/wiener-filter paths as they
  become safe.
- For template-heavy code, split declarations from implementation into
  focused `detail` or `impl` headers and keep include surfaces small.
- Add forward declarations and reduce transitive includes where practical.

Constraints:

- Avoid broad namespace/file renames in the same PR as implementation moves.
- Avoid changing algorithms while moving code.
- Coordinate with the performance worktree before mapmaking hot-loop edits.

Validation:

- Unity compile for every source-boundary PR.
- Runtime baseline for any moved code used by active reductions.

### PR 6: Explicit Reduction Contexts

Deliverables:

- Introduce narrow context structs that wrap existing state first:
  `RuntimeContext`, `ObservationContext`, `CalibrationState`,
  `TimestreamState`, `MapmakingState`, `IterationContext`, and `OutputLayout`.
- Replace broad `Engine` field reads with context access in orchestration code
  first.
- Keep `Engine` as a compatibility owner until downstream code has migrated.

Constraints:

- Do not force a large inheritance-to-composition rewrite in one PR.
- Protect invariants explicitly: detector indexing, array/network grouping,
  sample ranges, map geometry, units, and output product type.

Validation:

- Unit tests for context construction and invariant failures.
- Unity baseline for affected reduction modes.

### PR 7: RTC/PTC Interface Cleanup

Deliverables:

- Define explicit RTC input/output chunk contracts.
- Define explicit PTC input/output chunk contracts.
- Separate diagnostics/learning snapshots from live mutable processor state.
- Keep source-protection and learning controls typed and stage-local.

Constraints:

- No extra allocations in per-sample/per-detector loops without measurement.
- Preserve deterministic despike/learning behavior added in prior work.

Validation:

- Synthetic tests for flags, sample ranges, detector exclusion, source mask
  behavior, and deterministic learning summaries.
- Unity pointing and science fruitloops baselines.

### PR 8: Mapmaking State and Product Writers

Deliverables:

- Split map geometry, accumulation buffers, noise products, diagnostics, and
  FITS/netCDF product writing into narrower interfaces.
- Keep naive and jinc hot loops stable until benchmarks and characterization
  tests cover them.
- Gate expensive diagnostics explicitly and document defaults.

Constraints:

- Coordinate with `codex/perf-map-accumulation-noise-lifecycle` before changing
  map accumulation/noise lifecycle internals.
- No product-format changes without explicit migration notes.

Validation:

- Microbenchmarks or stage timings for naive, jinc, noise products, PSD/hist,
  and filtering.
- Unity science/noise/jinc baseline comparisons.

### PR 9: Public API Cleanup and Deprecation

Deliverables:

- Reduce public mutable fields after contexts and adapters are in use.
- Remove obsolete helper paths and dead commented-out build entries only after
  replacements are validated.
- Add contributor docs describing the new flow and validation expectations.

Validation:

- Full Unity baseline matrix.
- Maintainer review of remaining public API and operational scripts.

## Test Coverage Plan

Add tests in layers:

- Config parsing/validation unit tests for the existing YAML schema.
- Unit tests for calibration conversions and vector-length/grouping invariants.
- Synthetic telescope/KIDs alignment tests for overlap, gaps, HWPR lengths, and
  scan boundaries.
- Synthetic RTC/PTC tests for despiking, filtering guard behavior, flags,
  deterministic diagnostics, and source protection.
- Synthetic mapmaking tests for map geometry, naive accumulation, jinc metadata,
  source finder behavior, noise products, and product headers.
- Small end-to-end synthetic reduction once the runner is extractable.
- Unity characterization reductions for real operational data products.

Test priority should follow risk: config, alignment, calibration, flags,
mapmaking, metadata, then broad structural cleanup.

## PR Checklist

Each refactor PR should state:

- Affected reduction modes: science, pointing, beammap, polarimetry if relevant.
- Expected behavior change: ideally none.
- Files/modules touched.
- Hot paths touched, if any.
- Unity build result.
- Unity reduction result when runtime paths are touched.
- Performance comparison when runtime paths or product generation are touched.
- Products compared and comparison tolerance.
- Residual risk.
- Follow-up tests or refactor work.

## Risks

- Header movement can hide template instantiation or include-order problems
  until Unity compile time.
- CLI extraction can accidentally change output directory naming, fruitloops
  iteration semantics, logging, or product write order.
- Typed config adapters can introduce default drift if they are not compared
  against `data/config.yaml`.
- Error conversion can change operational failure modes if messages or return
  codes drift.
- State-context extraction can change lifetime or per-observation reset
  behavior.
- Mapmaking/noise refactors carry performance and numerical risk and should
  wait for baselines and coordination with performance work.

## Immediate Next Actions

1. Create the tracking issue/epic and PR checklist.
2. Define the Unity baseline run matrix with exact datasets/configs.
3. Run baseline reductions at `376e0022` and save manifests.
4. Add manifest/comparison utilities.
5. Start PR 2 only after the baseline harness can detect config/output drift.
