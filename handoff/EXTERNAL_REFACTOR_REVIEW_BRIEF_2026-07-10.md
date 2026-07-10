# Citlali Refactor External Review Brief - 2026-07-10

## Review Assignment

Please act as an independent architecture, validation, and engineering-process
reviewer for the long-running Citlali refactor in:

```text
/Users/gwilson/GitHub/citlali-refactor
```

Branch:

```text
codex/structural-refactor
```

Baseline commit:

```text
376e0022  adding weight cuts to fruitloops
```

Snapshot described by this brief:

```text
84670829  Record typed output validation checkpoint
```

The maintainer is compiling this snapshot on Unity now. The compile and point
reduction for `84670829` are not yet part of the validated evidence below.

The primary question for this review is:

> What does an external reviewer need to know about this project in order to
> offer sound advice and check that our plan and execution are sound?

Please answer that question by independently checking the repository and then
assessing the plan, implementation shape, validation evidence, performance
discipline, and remaining work. Do not assume that the existing handoff notes
are correct merely because they are detailed. Challenge their conclusions.

This is a review task, not an authorization to perform another broad refactor.
Please report findings and recommendations before making code changes.

## Requested Review Output

Please produce:

1. An executive verdict: sound / sound with material reservations / course
   correction needed.
2. Findings ordered by severity, with concrete file references and evidence.
3. An assessment of whether the new module boundaries are coherent or have
   become over-fragmented.
4. An assessment of whether the validation matrix is strong enough for the
   changes already made.
5. An assessment of the typed-config migration and its dual-state risks.
6. A recommended definition of done for the current config phase and for the
   broader refactor.
7. A prioritized roadmap for the next three to five phases, including explicit
   validation and performance gates.
8. A short list of work that should stop or be deferred because it is producing
   churn without enough architectural value.
9. Any missing evidence needed before migrating analysis-control paths or
   presenting the branch as production-ready.

## What Citlali Is

Citlali is a C++ high-performance reduction pipeline for TolTEC astronomical
data. It processes KIDs timestreams, telescope data, detector/calibration
tables, and reduction configuration into products including:

- FITS maps and filtered/coadded science maps
- pointing fit tables
- beammap detector maps, fit/QC tables, flags, and calibration products
- RTC/PTC timestream NetCDF files
- RTC/PTC/map diagnostic NetCDF sidecars
- statistics, PSD, histogram, learning, and index products

The normal operational entry point is not a hand-written low-level Citlali
command. On Unity, the maintainer runs:

```text
tolteca reduce
```

TolTECA reads `70_reduce.yaml` and all other numbered `NN*.yaml` files in the
reduction directory. Higher-numbered files recursively override lower-numbered
files. TolTECA then constructs the low-level `citlali_o*.yaml` file consumed by
Citlali. Any config redesign must remain compatible with this overlay workflow.

The routinely important reduction types are:

- pointing
- out-of-focus holography (OOF)
- beammap
- science

## Refactor Goals And Constraints

The original plan is in:

- `doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md`
- `doc/REFACTOR_DESIGN_SCAFFOLDING_2026-06-29.md`
- `doc/REFACTOR_INVENTORY_2026-06-29.md`

The governing goal is to make Citlali a maintainable library and CLI without
changing science behavior or materially degrading runtime performance.

Key goals:

- move orchestration out of the CLI and giant engine methods
- introduce a complete typed internal config model
- preserve the existing low-level YAML schema during migration
- simplify the normal user config surface while preserving expert overrides
- separate config, state, algorithms, diagnostics, and output ownership
- replace process exits in library code with typed failures at appropriate
  boundaries
- move suitable non-template implementation from headers into `.cpp` files
- preserve hot-loop performance and avoid allocations/YAML/string dispatch in
  per-sample, per-detector, and per-pixel paths
- add enough characterization and product comparison tooling to distinguish
  refactor regressions from existing numerical behavior

Important non-goals:

- this is not a rewrite of the science algorithms
- config keys and expert controls are not to be removed early
- numerical changes require a separate scientific reason and validation
- the giant RTC/PTC/mapmaking headers are not automatically in the current
  engine/config stage merely because they are large
- optional R/quadrature-channel execution is future work, not part of the
  current behavior contract

Performance policy from the original plan:

- runtime-path changes should normally stay within roughly 3-5% wall time
- peak memory should not materially increase
- diagnostics should not add default hot-path cost
- compile-time and dependency behavior matter, but should not be optimized by
  obscuring architecture or weakening validation

## Starting Architecture

At baseline `376e0022`, the main concentration points were:

| File | Baseline lines | Responsibility problem |
| --- | ---: | --- |
| `src/citlali/cli/main.cpp` | 1,270 | CLI, config merge, runtime setup, observation/iteration orchestration, output |
| `include/citlali/core/engine/engine.h` | 9,674 | broad public mutable engine state plus config, runtime, and output behavior |
| `include/citlali/core/engine/beammap.h` | 6,916 | beammap setup, mapmaking, fitting, convergence, flags, tables, and output |
| `include/citlali/core/timestream/rtc/rtcproc.h` | 6,337 | RTC config, algorithms, diagnostics, and state |
| `include/citlali/core/timestream/ptc/ptcproc.h` | 5,841 | PTC config, cleaning, weighting, diagnostics, and state |
| `include/citlali/core/timestream/ptc/clean.h` | 2,136 | cleaner algorithms and policy |
| `include/citlali/core/mapmaking/wiener_filter.h` | 1,600 | serial Wiener filtering implementation |
| `include/citlali/core/mapmaking/jinc_mm.h` | 1,358 | performance-sensitive gridding/mapmaking |

The baseline inventory counted 144 direct exit calls and seven natural CMake
implementation sources commented out. Test coverage was mostly utility/filter
smoke coverage rather than end-to-end pipeline protection.

## Current Architecture Snapshot

At `84670829`:

| File | Current lines | Change from baseline |
| --- | ---: | ---: |
| `src/citlali/cli/main.cpp` | 54 | -1,216 |
| `include/citlali/core/engine/engine.h` | 298 | -9,376 |
| `include/citlali/core/engine/beammap.h` | 833 | -6,083 |
| `include/citlali/core/timestream/rtc/rtcproc.h` | 6,476 | +139 |
| `include/citlali/core/timestream/ptc/ptcproc.h` | 5,930 | +89 |
| `include/citlali/core/timestream/ptc/clean.h` | 2,208 | +72 |
| `include/citlali/core/mapmaking/wiener_filter.h` | 1,600 | unchanged |
| `include/citlali/core/mapmaking/jinc_mm.h` | 1,286 | -72 |

The major top-level concentration has been reduced dramatically. The current
shape uses:

- a small CLI entry point
- `citlali::pipeline` helpers for reduction, iteration, observation, setup,
  output, and validation orchestration
- `citlali::config` enums, structs, validation, and activation policies
- a small `Engine` facade plus focused engine detail implementation headers
- mode-specific pointing/beammap/lali helpers
- product-specific FITS, NetCDF, ECSV, diagnostic, and timing helpers

However, the decomposition has created a very large header surface:

- 171 files under `include/citlali/core/engine/detail`
- 348 files under `include/citlali/core/pipeline`
- 616 headers scanned by the current inventory tool
- 145 direct exit calls remain, 137 classified as high-risk library-header
  exits
- the same seven natural `.cpp` entries remain commented out in `CMakeLists.txt`

The branch history is also unusually granular:

- 2,887 commits since `376e0022`
- 695 changed files
- approximately 73,592 inserted and 21,557 deleted lines

Many early commits were intentionally tiny because Unity compilation was the
only dependable gate. The workflow later changed to coherent development
stages, one local build after roughly two stages, and a larger Unity checkpoint
after multiple commits. The external review should explicitly judge whether
the present file and commit granularity is still reviewable and maintainable.

## Major Work Completed

### CLI And Pipeline Orchestration

The large CLI body was moved behind reusable reduction/iteration/observation
boundaries. The flow now closely matches the conceptual pipeline:

```text
TolTECA config generation
  -> Citlali CLI/runtime setup
  -> initial geometry
  -> reduction iteration loop
  -> observation loop
  -> observation preflight
  -> RTC/PTC processing and map accumulation
  -> observation/coadd products
  -> filtering/source products
  -> learning/index finalization
```

See:

- `doc/ANALYSIS_FLOW_RAW_TO_SCIENCE_PRODUCTS_2026-07-01.md`
- `doc/analysis_flow_html/index.html`
- `include/citlali/core/pipeline/reduction_pipeline.h`
- `include/citlali/core/pipeline/reduction_iteration_loop.h`
- `include/citlali/core/pipeline/reduction_observation_loop.h`

### Engine And Beammap Structure

Engine config, setup, observation, timestream, map, diagnostic, and output
implementation was split into named detail units. Beammap was decomposed around
setup, timestream processing, mapmaking passes, fitting, priors, convergence,
flagging, APT/QC tables, detector TOD, and map products.

The broad beammap subdivision phase has intentionally stopped. Further splits
should be justified by a specific ownership or validation problem rather than
line count alone.

One real output bug was found by validation: detector-grouped beammap FITS
writing used a map/detector index where a TolTEC array id was required, causing
a late segmentation fault. The narrow fix used the already computed array
identity. This is a useful example of the validation workflow finding a real
index-ownership defect.

### Typed Config Model

The engine owns an aggregate `citlali::config::ReductionConfig` with typed
sections for runtime, timestream, mapmaking, coadd/noise, post-processing,
pointing, beammap, and astrometry.

Typed enums and helpers replace many scattered string comparisons. Legacy
parsers and processor fields remain authoritative where execution has not yet
migrated. Current work is moving in three tiers:

1. parse and mirror existing YAML into typed state
2. use typed values for summaries, filenames, output layout, and provenance
3. only after validation, use typed values for execution/control decisions

Recent work has moved learning config, timestream core/output parsing,
polarimetry/HWPR policy parsing, and FITS/NetCDF provenance behind pipeline and
typed-config boundaries. Calibration, filtering, despiking, downsampling,
HWPR loading, map allocation, and other analysis execution still use the
legacy RTC/PTC/engine runtime state unless a prior narrow migration was already
validated.

The reviewer should look for dual-state hazards:

- raw user request versus effective runtime value
- typed mirror versus legacy processor field
- reset/order dependencies during config loading
- aliases normalized in only one representation
- typed values used before all processor policy rewrites have completed
- provenance emitted from a representation different from the one executing

### User-Facing Config Simplification

Config simplification is deliberately split from execution migration.

The current policy classifies representative low-level keys as:

- user-facing
- expert
- hidden/internal
- deprecated

Expert keys remain available through `expert:` or later TolTECA overlays such
as `80_expert.yaml`. The compact translator can round-trip representative
pointing, OOF, beammap, and science configs to the old low-level schema.

Current preflight result:

```text
compact compatibility: 8 passed, 0 failed, 0 skipped
actionable surface coverage: 100%
covered=265, profile_owned=17, gaps=0
```

This proves YAML translation equivalence for the representative fixtures. It
does not yet prove that normal TolTECA production should switch to compact
configs. Runtime rollout and TolTECA template changes remain future work.

See:

- `doc/CONFIG_SIMPLIFICATION_HANDOFF_2026-07-02.md`
- `doc/CONFIG_POLICY_BASELINE_V1_2026-07-02.md`
- `doc/CONFIG_SIMPLIFICATION_BASELINE_INVENTORY_2026-07-02.md`
- `tools/config/README.md`

### Validation And Profiling Tooling

The branch now includes:

- reduction audit tooling
- strict deterministic manifests
- FITS/NetCDF/ECSV/CSV product comparison
- low-level YAML comparison
- compact config compatibility and surface audits
- stage profile sidecars and log timing extraction
- analysis-flow diagrams and validation hook documentation

Important tools:

- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/compare_reduction_audits.py`
- `tools/baseline/compare_reduction_products.py`
- `tools/baseline/summarize_outputs.py`
- `tools/baseline/compare_deterministic_manifests.sh`
- `tools/config/compare_lowlevel_yaml.py`
- `tools/config/run_config_preflight.py`

The product comparator intentionally ignores large TOD files by default unless
`--include-timestream` is requested. A reviewer should judge whether that
default is appropriate for the next phase.

### Build Workflow

There is now a local Apple Silicon/Homebrew build environment and a Unity
GitHub pull/build helper. Citlali refactor executables live separately from the
OG comparison build:

```text
citlali_dev/citlali/build/...          OG comparison executable
citlali_dev/citlali_refactor/build/... refactor executable
```

The Unity helper skips CMake configure when inputs/cache do not require it.
This matters because configure regenerates timestamped version headers and can
invalidate the PCH/large CLI translation unit.

Observed costs:

- local no-op build: about 1.2 seconds
- local configure: about 22-23 seconds
- local rebuild after configure: roughly 90-119 seconds in the audit
- current local header-change build: commonly about 60-70 seconds
- Unity compile reported by the maintainer: about 6 minutes
- standard point run: tens of seconds to roughly one minute
- beammap run: roughly one hour, sometimes longer

The current header-heavy structure still makes a small header edit rebuild the
large CLI translation unit. Moving non-template implementation into real `.cpp`
targets has not yet begun in a material way.

## Validation Evidence

### Pointing

Pointing observation 152389 is the strongest and most frequent regression
gate. It exercises RTC/PTC processing, learning, mapmaking, TOD output,
diagnostics, FITS output, source finding/fitting, and pointing products.

The latest completed validation before this review brief is:

```text
point/refactor/reduced/redu21
Citlali revision 71635a7e
```

Compared with `redu20`:

- both low-level configs had 489 leaves and zero differences
- expected product sets matched
- compared FITS, NetCDF, and ECSV values were exactly unchanged
- only stage-profile elapsed times differed
- the run completed normally

There are 12 recurring log messages reading `NetCDF: Not a valid ID` in each
recent point run. They have not changed across the latest comparisons, but they
remain a known defect and should not be normalized away merely because the
products complete.

### Beammap

Beammap validation is expensive and therefore less frequent. Detailed OG versus
refactor comparisons have shown:

- matching FITS product/HDU structure
- matching ECSV row/column structure and detector flag classifications
- only very small floating differences in most fit/table/map values
- matching NetCDF schemas
- no current product-structure red flag

The detailed `f278bd32` validation compared OG `redu01` with refactor `redu07`.
It found mostly threshold-level differences and a refactor wall time about 7%
slower than the cited OG run (`3535.6 s` versus `3297.2 s`). Another comparison
was roughly `58m38s` versus `55m04s`. This is close, but not conclusively inside
the original 3-5% budget. Peak RSS evidence is incomplete.

More recent config/provenance work has usually been point-validated but has not
always received a new full beammap run.

### Science

Science validation is weaker than pointing and beammap. Individual-observation
products have been compared successfully in prior cycles. Some filtered/coadded
science comparisons showed few-percent scale differences, especially in
filtered signal/noise products. Investigations involved weight normalization,
fruitloops configuration, and Wiener-filter behavior/performance. That history
is not yet a clean, automated science acceptance gate for the current branch.

A reviewer should treat current science/coadd equivalence as incomplete rather
than infer safety from point equivalence.

### OOF And Polarimetry

OOF has representative config translation fixtures but less runtime validation
than pointing/beammap. Polarimetry now has typed config vocabulary, but no
current operational polarized reduction baseline protects enabled execution.
Do not recommend migrating HWPR/polarized analysis control solely on the basis
of the ordinary point run, where polarimetry is disabled.

## Current Phase And In-Flight Work

The broad engine/beammap structural subdivision phase is considered complete
enough to stop splitting by line count. The active phase is typed-config
ownership and consumer migration.

Current strategy:

- move YAML reads out of engine execution methods
- preserve legacy parsing and effective runtime behavior
- use typed config first in read-only summaries, filenames, metadata, and
  provenance
- locally build after approximately two coherent development stages
- batch several validated commits before a Unity compile/point run
- require mode-appropriate validation before changing analysis-control paths

The current Unity compile is for `84670829`, containing the latest typed output
provenance batch. A point reduction should directly compare FITS and NetCDF
products because this batch changes which in-memory representation supplies
provenance, even though the represented values should be identical.

## Work Not Yet Complete

### Authoritative Typed Config

The typed model is not yet the single authoritative source for all execution.
The reviewer should recommend a safe endpoint and transition strategy:

- when should legacy processor config reads stop?
- should typed config populate legacy processors once, or should processors
  consume typed section objects directly?
- how should effective runtime rewrites be represented?
- how can mirror divergence be detected automatically?
- when should serialized typed config become authoritative provenance?

### Header To `.cpp` Migration

The project still compiles much implementation through headers and a large CLI
translation unit. The seven natural source entries in `CMakeLists.txt` remain
commented. A future phase should move suitable non-template implementation into
real source files, but only with a deliberate dependency and ABI plan.

The reviewer should identify the best first `.cpp` boundary. Candidates should
be non-template, non-hot, have stable ownership, and produce measurable compile
benefit without forcing a risky broad move.

### Failure Handling

The original goal of replacing library `std::exit` calls remains largely
unfinished. The current census finds 145 direct exits, including 137 in
high-risk library-header locations. A typed error/failure hierarchy and CLI
translation boundary still need a staged design.

### Core RTC/PTC/Mapmaking Architecture

The giant RTC/PTC/clean/Wiener/JINC headers remain largely outside the completed
engine/beammap stage. This was intentional to avoid destabilizing hot paths.
They still represent major long-term maintenance and compile-time debt.

The reviewer should decide whether the next broad phase should address them or
whether typed config, `.cpp` boundaries, tests, and error handling must come
first.

### Tests And CI

`tests/test_config_scaffold.cpp` has grown useful policy/helper coverage, but
the normal local build tree used in this thread does not expose the
`citlali_test` target because tests are added only when `CITLALI_STANDALONE` and
`CITLALI_BUILD_TESTS` are enabled. The strongest gate is still Unity reduction
comparison rather than a fast unit/integration suite.

The reviewer should recommend a practical test pyramid:

- pure policy/config/helper unit tests
- product-schema writer tests
- small fixture integration tests
- deterministic point reduction
- periodic beammap/science performance and product gates

### Compact Config Runtime Rollout

Translation and policy tooling are mature enough for review, but compact config
is not yet the normal production input. TolTECA template/catalog work belongs
partly in `tolproj`, not solely in this repository. Expert overlays and old
full-schema compatibility must remain available during rollout.

### Auxiliary R/Quadrature Channel

The future R-channel design is documented in:

```text
doc/R_ANALYSIS_AUXILIARY_CHANNEL_NOTE_2026-07-08.md
```

The key architectural requirement is to preserve a first-class optional
measured sidecar channel rather than treating R as a synthetic kernel. Typed
config scaffolding exists and defaults off, but execution is not implemented.

The review should check whether current TCData, RTC/PTC cleaning, calibration,
output, and learning boundaries are becoming harder to extend to multiple
measured channels. It should not recommend implementing full R analysis merely
to complete this refactor review.

## Known Risks And Ambiguities

1. Over-fragmentation: hundreds of small headers may have replaced monoliths
   with navigation and compile-time costs rather than stable modules.
2. Dual config state: typed and legacy representations can diverge by ordering,
   defaults, aliases, or effective runtime rewrites.
3. Validation imbalance: pointing is strong; science, OOF, and polarimetry are
   not equivalently protected.
4. Performance evidence: stage timing exists, but controlled repeated runtime
   and peak-memory comparisons are incomplete.
5. Header-only implementation: compile time remains sensitive to broad header
   changes, and real `.cpp` ownership has not landed.
6. Error handling: process exits remain common inside library code.
7. Commit/PR reviewability: the branch is extremely long and granular, with no
   clean series of small external PRs matching the original PR plan.
8. Known recurring NetCDF errors are tolerated by current runs but unresolved.
9. Source finding has limited operational exercise; it is enabled in the point
   validation but its scientific findings are not yet treated as a mature gate.
10. Some earlier changes incorporated verified `gw_dev` determinism, cleaning,
    and Wiener-filter improvements, so not every branch difference is purely
    structural even though each such import was intended and separately tested.

## Questions The Reviewer Should Answer

### Architecture

- Do the current `pipeline`, `config`, `engine/detail`, and mode-specific
  boundaries reflect real ownership, or are they mostly textual extraction?
- Which small headers should be consolidated into subsystem modules?
- Is `Engine` now an acceptable compatibility facade, or does its inherited
  state model still block safe evolution?
- Should the next architecture phase prioritize `.cpp` boundaries, explicit
  contexts/state ownership, or RTC/PTC core cleanup?

### Typed Config

- Is mirror-first migration still the safest strategy at this point?
- What invariant should define authoritative config after parsing?
- How should requested values, normalized values, and effective runtime values
  be represented without duplication?
- What tests are required before typed config controls filtering, calibration,
  weighting, mapmaking, HWPR, noise, and beammap execution?

### Validation

- Is exact point-product equality a sufficient frequent gate for the current
  output/config changes?
- Which changes require beammap, science, OOF, TOD-heavy, or polarimetry runs?
- Should large RTC/PTC timestream products be compared more routinely?
- What science metrics and tolerances should become explicit acceptance rules?
- How should the unresolved NetCDF errors affect release readiness?

### Performance And Build

- Does the current evidence satisfy the 3-5% runtime budget?
- What repeated-run and peak-RSS protocol is needed to distinguish code changes
  from Unity/storage variation?
- Which first implementation move would most reduce rebuild cost?
- Are hundreds of focused headers helping or harming PCH/incremental builds?

### Process

- How can this 2,887-commit development branch be reviewed and merged safely?
- Should it be consolidated into a smaller logical integration series, or
  reviewed as a validated branch snapshot plus subsystem follow-ups?
- What explicit completion criteria prevent endless subdivision?
- Which deferred tasks belong in separate issues rather than this refactor?

### Future Evolution

- Do current interfaces preserve a clean path for optional measured R data?
- Does compact config policy integrate cleanly with TolTECA numbered overlays?
- Is the codebase moving toward a modern high-performance pipeline architecture
  or merely redistributing legacy coupling?

## Suggested Reading Order

1. `handoff/EXTERNAL_REFACTOR_REVIEW_BRIEF_2026-07-10.md`
2. `doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md`
3. `doc/REFACTOR_INVENTORY_2026-06-29.md`
4. `handoff/HANDOFF_2026-07-06.md`
5. `handoff/HANDOFF_2026-07-09.md`
6. `doc/CONFIG_SIMPLIFICATION_HANDOFF_2026-07-02.md`
7. `doc/R_ANALYSIS_AUXILIARY_CHANNEL_NOTE_2026-07-08.md`
8. `doc/ANALYSIS_FLOW_RAW_TO_SCIENCE_PRODUCTS_2026-07-01.md`
9. `doc/STRUCTURAL_REFACTOR_PR_CHECKLIST.md`
10. `tools/baseline/README.md` and `tools/config/README.md`

Then inspect the current code shape beginning with:

- `src/citlali/cli/main.cpp`
- `include/citlali/core/pipeline/reduction_pipeline.h`
- `include/citlali/core/engine/engine.h`
- `include/citlali/core/engine/beammap.h`
- `include/citlali/core/pipeline/reduction_config_accessors.h`
- `include/citlali/core/config/reduction_config.h`
- `include/citlali/core/engine/detail/citlali_config_impl.h`
- `include/citlali/core/engine/detail/timestream_top_config_impl.h`
- `include/citlali/core/engine/detail/rtc_config_impl.h`
- `include/citlali/core/engine/detail/ptc_config_impl.h`

## Useful Independent Checks

Run Python with the project environment:

```bash
$HOME/tolteca/bin/python tools/refactor/refactor_inventory.py \
  --repo . \
  --json-out /private/tmp/citlali_review_inventory.json \
  --markdown-out /private/tmp/citlali_review_inventory.md
```

Inspect branch scale:

```bash
git rev-list --count 376e0022..HEAD
git diff --shortstat 376e0022..HEAD
```

Run the local compile/config gate without changing configuration:

```bash
cmake --build build --target citlali_cli -j 8
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all
```

Inspect the most recent completed point validation:

```bash
$HOME/tolteca/bin/python tools/baseline/audit_reduction_run.py \
  /Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/reduced/redu21 \
  --expected-mode point \
  --expected-label refactor
```

Compare it with the preceding checkpoint:

```bash
$HOME/tolteca/bin/python tools/baseline/compare_reduction_products.py \
  /Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/reduced/redu20 \
  /Users/gwilson/work_toltec/local_data/2026-refactor/point/refactor/reduced/redu21 \
  --mode point \
  --baseline-label refactor \
  --candidate-label refactor
```

Do not use Unity from the review thread. The maintainer performs Unity builds
and reductions and can supply new products when a review finding requires them.

## Final Framing

The strongest evidence in favor of the work is that the CLI/engine/beammap
monoliths are much smaller, the runtime flow has named boundaries, and thousands
of changes have repeatedly passed exact point comparisons plus periodic
beammap checks.

The strongest reason for skepticism is that the branch has become very large
and header-fragmented while major goals remain unfinished: authoritative typed
config, `.cpp` implementation boundaries, typed failure handling, broad tests,
science/polarimetry validation, and core RTC/PTC architecture.

The reviewer should determine whether the current branch is a sound foundation
for those next phases, or whether it needs consolidation and stronger gates
before more refactoring continues.
