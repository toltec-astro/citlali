# Citlali Legacy Test Suite Modernization Handoff

Date: 2026-07-10

Repository: `/Users/gwilson/GitHub/citlali-refactor`

Branch in the main refactor working tree: `codex/structural-refactor`

## Assignment

Modernize and reactivate the pre-existing `citlali_test` suite without changing
Citlali production behavior merely to accommodate stale tests.

The desired endpoint is:

1. `citlali_test` builds cleanly in the supported local build environment.
2. All still-relevant tests run through CTest and pass.
3. Obsolete tests are updated to current contracts, or removed only with a
   written justification showing that their behavior is covered elsewhere or
   no longer exists by design.
4. The existing `citlali_safety_test` target remains active and passing.
5. `citlali_cli` still builds.
6. No numerical pipeline behavior, output schema, or user-facing config
   semantics change as an incidental part of test repair.

This is a parallel workstream supporting Phase 1, Safety Stabilization, of the
adopted five-phase roadmap in `doc/REFACTOR_STATUS.md`.

## Why This Work Matters

The legacy target contains substantial, useful characterization coverage:

- `tests/test_config_scaffold.cpp`: 186 tests.
- `tests/test_utils.cpp`: 15 tests.

These cover typed config parsing and validation, CLI setup, runtime policy,
preflight, calibration setup, scan and observation preparation, fruit-loop
lifecycle, output orchestration, FFT helpers, Wiener-filter behavior, and
timestream filters. They are not disposable simply because their scaffolding
has decayed.

The separate focused target currently contains 18 passing safety tests:

- `tests/test_config_safety.cpp`: 9 tests.
- `tests/test_ordered_writer.cpp`: 6 tests.
- `tests/test_output_schema.cpp`: 1 test.
- `tests/test_scan_cursor.cpp`: 2 tests.

The focused target was introduced to establish urgent safety contracts while
the old suite was unavailable. It must not become an excuse to abandon the
broader suite.

## Current Refactor Context

The active roadmap is recorded in `doc/REFACTOR_STATUS.md`. Read it and the
external review before making changes:

- `handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md`
- `handoff/EXTERNAL_REFACTOR_REVIEW_BRIEF_2026-07-10.md`

Recent relevant commits are:

- `2bf87107` Record accepted point safety checkpoint
- `d3080eb2` Own scan cursors per reduction run
- `a899e318` Verify required timestream output cardinality
- `c2ec8ae5` Add strict reduction product gate
- `f06766f6` Verify failed writes return CLI failure

The project is intentionally pausing broad typed-config migration while Phase 1
contracts are stabilized. Test modernization may update fake objects to match
the current typed architecture; it must not resume production config migration
or introduce a second authority model.

## Working-Tree Coordination

At the time this note was written, the main refactor working tree contains
uncommitted partial test modernization in exactly these files:

- `tests/CMakeLists.txt`
- `tests/main.cpp` (deleted in the partial work)
- `tests/test_config_scaffold.cpp`

No other files are dirty.

The main refactor task will avoid editing `tests/` while this work is active.
Coordinate before touching shared roadmap or handoff files. Do not push unless
the project owner explicitly asks; make local coherent commits for review.

If working in a separate Git worktree, either reproduce the partial changes
described below or transfer them deliberately. Do not assume uncommitted edits
are present in a fresh worktree.

## What Has Already Been Diagnosed

### 1. Obsolete Benchmark Coupling

The original `tests/main.cpp` initialized both GTest and Google Benchmark:

```cpp
#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
```

It ran all tests and then `benchmark::RunSpecifiedBenchmarks()`. However, neither
legacy test source contains benchmark definitions. Linking `tula::testing`
therefore pulled Google Benchmark 1.6.0 into an ordinary test executable for no
functional reason.

On the local Clang environment, Benchmark 1.6.0 fails under project warnings:

```text
benchmark-src/src/mutex.h:79:40: error: mutex 'mut_' is still held at the end
of function [-Werror,-Wthread-safety-analysis]
```

The current partial fix:

- Removes `tests/main.cpp` from `citlali_test` and deletes the file.
- Replaces `tula::testing` with `GTest::gtest_main` and `GTest::gmock`.

This is the recommended direction. Tests and benchmarks should be separate
targets if actual Citlali benchmarks are introduced later.

### 2. Removed Transitive Validation Include

`test_config_scaffold.cpp` used `ValidationReport`, `format_path`, and config
validation APIs through a transitive include. The partial work adds:

```cpp
#include <citlali/core/config/reduction_config_validation.h>
```

Keep explicit includes. Do not restore accidental production-header coupling.

### 3. GTest Macros And Explicit Template Arguments

Current GTest macros parse commas in explicit template argument lists as macro
argument separators. Several existing assertions therefore fail before C++
compilation, for example:

```cpp
EXPECT_TRUE(citlali::pipeline::run_reduction_pipeline<
    false, RawObs, FilteredObs, RawCoadd, FilteredCoadd, false, FakeKidsProc>(...));
```

The partial work adds local variadic wrappers:

```cpp
#define EXPECT_TEMPLATE_TRUE(...) EXPECT_TRUE((__VA_ARGS__))
#define EXPECT_TEMPLATE_FALSE(...) EXPECT_FALSE((__VA_ARGS__))
```

and mechanically updates the affected assertions. This is acceptable as a
local test compatibility device. A named local bool followed by `EXPECT_TRUE`
is also acceptable if it improves clarity.

### 4. Validation API Drift

`ReductionConfig` retains the one-argument convenience overload:

```cpp
ValidationReport validate(const ReductionConfig &);
```

Nested config types use the current two-argument API:

```cpp
void validate(const SomeConfig &, ValidationReport &);
```

The partial work updates the Beammap default-validation test accordingly. The
Astrometry test around the current line 1644 still needs the same treatment.
Do not add broad production convenience overloads solely to preserve stale
test syntax.

### 5. Fake Engine Predates Typed Config Authority

The largest remaining compile problem is `FakeEngine` in
`tests/test_config_scaffold.cpp`. It still exposes legacy flat fields such as:

- `n_threads`
- `verbose_mode`
- `map_method`
- `map_grouping`

Current production helpers intentionally read through
`engine.typed_config` using `reduction_config_accessors.h`. Compilation now
fails with messages such as:

```text
error: no member named 'typed_config' in FakeEngine
```

Modernize the fake to model the current contract. Add a real
`citlali::config::ReductionConfig typed_config` and update tests to set/assert
the corresponding typed paths, including at least:

- `typed_config.runtime.n_threads`
- `typed_config.runtime.verbose`
- `typed_config.mapmaking.method`
- `typed_config.mapmaking.grouping`

Do not make production accessors fall back to the fake's old flat fields. One
authority per migrated field is an explicit architectural requirement.

Some legacy flat fields may still be required by unmigrated production paths.
Retain them only where compilation or a current contract demonstrates that
need. Avoid a wholesale fake rewrite before the compiler identifies the next
missing contract.

### 6. Alignment State Moved Under Explicit Ownership

`FakeEngine` still places these fields at top level:

- `start_indices`
- `end_indices`
- `hwpr_start_indices`
- `hwpr_end_indices`

Current code uses `engine.alignment`, represented by
`citlali::pipeline::TimestreamAlignmentState` in
`include/citlali/core/pipeline/timestream_alignment_state.h`, with fields such
as:

- `start_indices`
- `end_indices`
- `hwpr_start_index`
- `hwpr_end_index`

Update the fake and affected expectations to the owned alignment state. Do not
restore aliases or fallback access in production code. This ownership change
supports sequential same-process reentrancy.

### 7. Fruit-Loop Terminology Changed From Model To Map

The production APIs were deliberately renamed because these artifacts are map
products, not abstract models. Stale tests call:

- `load_previous_fruit_loop_model_if_needed`
- `load_observation_fruit_loop_models_if_needed`

Current APIs are:

- `load_previous_fruit_loop_maps_if_needed`
- `load_observation_fruit_loop_maps_if_needed`

Update test names and calls consistently from `model(s)` to `map(s)`. Do not add
legacy aliases to production headers just to compile old tests.

## Recommended Work Sequence

Use compiler-driven, coherent slices:

1. Finish test-target decoupling from Google Benchmark.
2. Finish explicit validation includes and nested-config validation call
   updates.
3. Finish variadic/template assertion compatibility.
4. Add `typed_config` to `FakeEngine`; update only tests whose setup currently
   writes migrated flat fields.
5. Move fake alignment data and expectations under `engine.alignment`.
6. Update fruit-loop model-to-map API names and corresponding test names.
7. Rebuild `citlali_test` and repair the next compiler-reported fake/API drift
   in small thematic groups.
8. Once it links, run the executable directly and capture the exact pass/fail
   inventory.
9. Repair behavioral expectation failures only after deciding whether each
   expectation reflects the current intended contract. Do not blindly change
   expected values.
10. Run CTest discovery, the focused safety suite, and the CLI build.
11. Update `doc/REFACTOR_STATUS.md` only when the target genuinely builds and
   the disposition of all 201 tests is known.

## Commands

Use the established local build environment and do not reconfigure unless
required:

```bash
cd /Users/gwilson/GitHub/citlali-refactor
cmake --build build --target citlali_test -j 8
./build/tests/citlali_test
cmake --build build --target citlali_safety_test -j 8
./build/tests/citlali_safety_test
cmake --build build --target citlali_cli -j 8
ctest --test-dir build --output-on-failure
```

If Python is needed, use:

```bash
$HOME/tolteca/bin/python
```

Do not use Unity. The project owner handles Unity compilation and scientific
reduction validation after reviewed commits are pushed.

## Acceptance Criteria

Minimum acceptance for this workstream:

- `citlali_test` builds without Google Benchmark.
- The suite's test count and pass/fail/disabled inventory are reported.
- Every enabled legacy test passes, or each remaining failure has a precise
  documented production-contract question for the project owner.
- No test is disabled, deleted, or weakened merely to make the count green.
- `citlali_safety_test` continues to pass all 18 tests.
- `citlali_cli` builds locally.
- CTest discovers both targets reliably.
- Changes are restricted primarily to `tests/` and test CMake wiring.
- Any production-file change is separately justified as a real defect or
  missing public contract, with a focused regression test.
- No numerical algorithm, output format, reduction config semantics, or
  runtime policy changes without explicit project-owner review.

## Questions That Require Escalation

Ask the project owner or main refactor task before:

- Removing a test because the covered behavior is believed obsolete.
- Changing a scientific expected value or tolerance.
- Adding a production compatibility alias for an old API.
- Restoring fallback from typed config to raw YAML or legacy flat state.
- Modifying RTC/PTC, JINC, Wiener-filter, beammap-fit, or mapmaking algorithms.
- Changing test dependency versions or the main build dependency strategy.

## Deliverable

Provide a concise final report containing:

- Commits created, without pushing.
- Files changed.
- Original failure categories and their resolutions.
- Final number of tests discovered, passed, failed, skipped, and disabled.
- Exact build and test commands run.
- Any remaining contract questions or residual risk.
- Confirmation that the focused safety suite and `citlali_cli` still pass.

