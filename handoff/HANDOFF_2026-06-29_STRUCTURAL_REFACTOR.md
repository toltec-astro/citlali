# Structural Refactor Handoff - 2026-06-29

Local timestamp: 2026-06-29 16:26:07 EDT

This note is for the next Codex thread taking over the long-running Citlali structural refactor.

## Workspace And Branch

Use this worktree for the refactor:

`/Users/gwilson/GitHub/citlali-refactor`

Branch:

`codex/structural-refactor`

Remote:

`origin git@github.com:toltec-astro/citlali.git`

The branch was created from `gw_dev` at commit `376e0022` (`adding weight cuts to fruitloops`) and pushed to GitHub as:

`origin/codex/structural-refactor`

GitHub PR creation URL:

`https://github.com/toltec-astro/citlali/pull/new/codex/structural-refactor`

The active development worktree remains separate and should not be disturbed:

`/Users/gwilson/GitHub/citlali` on `gw_dev`

There is also an existing performance worktree:

`/Users/gwilson/GitHub/citlali-perf` on `codex/perf-map-accumulation-noise-lifecycle`

Coordinate with that work before changing mapmaking performance-sensitive code.

## Repository Rules To Preserve

Read `CODEX.md` before doing work. Important points:

- Do not attempt normal local configure/build/test commands on this machine.
- Authoritative compile, runtime, and reduction validation must happen on Unity.
- Use local edits and lightweight static checks only unless Unity is available.
- When running Python locally, use `$HOME/tolteca/bin/python` and `$HOME/tolteca/bin/pip`.
- After editing Citlali, append a current local timestamp note to a same-day handoff file before ending the session.

## Goal

Refactor Citlali toward modern professional structure without changing science behavior or materially increasing runtime.

This is not a rewrite. Treat it as a behavior-preserving refactor program split into small reviewable PRs. Every merged change should either preserve output numerics or explicitly document the science reason for changing them.

## Main Structural Findings

The review identified these core problems:

1. Giant public headers dominate the architecture.
   - `include/citlali/core/engine/engine.h` is about 9.7k lines.
   - `beammap.h`, `rtcproc.h`, `ptcproc.h`, and `timestream.h` are also very large.
   - Much implementation lives in headers, while natural `.cpp` entries are commented out in `CMakeLists.txt`.

2. The CLI is also the pipeline runner.
   - `src/citlali/cli/main.cpp` parses config, configures runtime, loops observations, mutates engine state, runs fruitloops, creates output directories, and dispatches reductions.
   - Extract orchestration into library code; keep the CLI as parse/log/return-code glue.

3. Library code exits the process directly.
   - There are many `std::exit(EXIT_FAILURE)` calls in library headers and source files.
   - Convert these gradually to typed errors/exceptions returned to the CLI boundary.

4. Config is untyped mutable runtime state.
   - `data/config.yaml` is large and config parsing is imperative inside `Engine` and subsystem methods.
   - Introduce typed config structs and schema-style validation while preserving the existing YAML format.

5. Major objects are public bags of state.
   - `Engine`, `Calib`, `MapBuffer`, `RTCProc`, and `PTCProc` expose large mutable surfaces.
   - Add narrower contexts and explicit invariants around units, detector indices, sample ranges, and map geometry.

6. Tests do not match risk.
   - Current tests are mostly utility/filter smoke tests.
   - Add characterization tests and synthetic fixtures before changing risky behavior.

## Performance Requirement

Runtime speed is critical. The refactor must not make reductions dramatically slower.

Default performance budget for runtime-path PRs:

- Standard reductions should not regress wall time by more than about 3-5% without maintainer approval.
- Peak memory should not materially increase.
- Expensive diagnostics must stay off by default.
- No abstraction should be added inside detector/sample/map-pixel hot loops without a Unity benchmark.

Hot-loop guardrails:

- No virtual dispatch in per-sample/per-detector inner loops.
- No per-sample string or map lookups.
- No repeated YAML/config access during processing.
- No extra logging in hot paths unless tightly gated.
- Preserve static/templated dispatch where profiling says it matters.
- Move boundaries first; avoid changing numerical kernels until covered by tests and benchmarks.

## Recommended Refactor Program

Use a GitHub epic plus small PRs. Suggested sequence:

1. Baseline and performance harness.
2. Typed config structs and validation adapters.
3. Extract `PipelineRunner` or `ReductionSession` from `src/citlali/cli/main.cpp`.
4. Convert library `std::exit` paths to typed failures subsystem by subsystem.
5. Move non-template implementation from headers into `.cpp` files and re-enable clean CMake source boundaries.
6. Split engine state into narrower contexts: `ObservationContext`, `CalibrationState`, `TimestreamState`, `MapmakingState`, and product writers.
7. Clean up RTC/PTC/mapmaking interfaces after behavior is protected.

Do not start with broad file movement unless there is already a validation harness. First make it possible to prove behavior and speed did not change.

## First Milestone For The New Thread

Create a baseline/validation plan before making structural code changes.

Suggested deliverable:

`doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md`

Include:

- scope and non-goals
- branch/worktree workflow
- Unity validation protocol
- representative reduction matrix
- performance budget
- artifact comparison plan
- proposed PR sequence

Suggested Unity baseline cases:

- science, naive mapmaking, noise off
- science, naive mapmaking, noise on
- pointing with fruitloops/learning path
- beammap detector grouping
- jinc mapmaking
- TOD-output-heavy case if used operationally

For each baseline record:

- git SHA
- config files
- Unity module/environment details
- exact command
- wall time
- peak RSS if available
- output directory
- key map/TOD product checksums or numeric summaries
- science metrics such as flux, FWHM, pointing offsets, map RMS, detector counts, flagged fractions

## GitHub Management

Use one tracking issue or epic: `Citlali structural refactor`.

Each PR should include:

- affected reduction modes: science, pointing, beammap
- expected behavior change: ideally none
- Unity build result
- Unity reduction result when runtime paths are touched
- performance comparison when hot paths are touched
- list of products compared
- residual risk and follow-up tests

Prefer short-lived branches from `codex/structural-refactor`, for example:

- `codex/refactor-baseline-harness`
- `codex/refactor-config-types`
- `codex/refactor-cli-runner`
- `codex/refactor-error-handling`
- `codex/refactor-header-boundaries`

Avoid a single giant refactor PR.

## Files The Next Thread Should Read First

- `CODEX.md`
- `CMakeLists.txt`
- `src/citlali/cli/main.cpp`
- `include/citlali/core/engine/engine.h`
- `include/citlali/core/engine/todproc.h`
- `include/citlali/core/engine/lali.h`
- `include/citlali/core/engine/pointing.h`
- `include/citlali/core/engine/beammap.h`
- `include/citlali/core/timestream/rtc/rtcproc.h`
- `include/citlali/core/timestream/ptc/ptcproc.h`
- `include/citlali/core/mapmaking/map.h`
- `data/config.yaml`
- `tests/CMakeLists.txt`
- `tests/test_utils.cpp`
- `doc/CODE_AUDIT_2026-05-14.md`
- `doc/PERFORMANCE_AUDIT_2026-05-14.md`
- `doc/REDUCTION_LEARNING_REFACTOR_PLAN.md`

## Suggested Opening Prompt For The Next Codex Thread

Use this prompt in a new Codex thread opened in `/Users/gwilson/GitHub/citlali-refactor`:

```text
We are starting the long-running Citlali structural refactor on branch codex/structural-refactor in /Users/gwilson/GitHub/citlali-refactor. Read CODEX.md and handoff/HANDOFF_2026-06-29_STRUCTURAL_REFACTOR.md first. Do not make broad code changes yet. Create the initial structural refactor plan and Unity validation/performance baseline plan, with runtime speed as a hard constraint.
```

## Current Status

- Refactor worktree exists and is clean.
- Branch `codex/structural-refactor` exists locally and on GitHub.
- No refactor code changes have been made yet.
- Next step is planning and baseline definition, not code movement.

## Update - 2026-06-29 16:31:49 EDT

- Created `doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md`.
- The plan is documentation-only and preserves the next step as baseline
  definition before structural code changes.
- No build/test was attempted locally, per `CODEX.md`.

## Update - 2026-06-29 16:48:18 EDT

- Added the first baseline-harness files under `tools/baseline/`:
  - `README.md`
  - `run_manifest_template.yaml`
  - `summarize_outputs.py`
  - `compare_manifests.py`
- The summarizer records file metadata/checksums and optional structured
  FITS/netCDF/ECSV/CSV/log summaries.
- The comparator compares two generated manifests with configurable numeric
  tolerances and optional checksum ignoring for timestamp-noisy products.
- Local lightweight checks only:
  - Python syntax parsed with `$HOME/tolteca/bin/python`.
  - Both scripts' `--help` paths ran.
  - A self-check manifest was generated for `tools/baseline/`.
  - The comparator reported that the self-check manifest matches itself.
- No Citlali build/test/reduction was attempted locally, and no Unity access was
  attempted.

## Update - 2026-06-29 21:46:00 EDT

- Continued with compile-neutral overnight refactor preparation only.
- Added config simplification artifacts:
  - `doc/CONFIG_SIMPLIFICATION_PLAN_2026-06-29.md`
  - `doc/CONFIG_KEY_CLASSIFICATION_START_2026-06-29.md`
  - `tools/config/README.md`
  - `tools/config/config_inventory.py`
- The config inventory reports 491 YAML leaf keys in `data/config.yaml`, with
  353 under `timestream`.
- Added structural refactor inventory artifacts:
  - `doc/REFACTOR_INVENTORY_2026-06-29.md`
  - `tools/refactor/README.md`
  - `tools/refactor/refactor_inventory.py`
- The refactor inventory reports 144 direct exit calls, 43 headers scanned,
  seven commented CMake source entries, and 234 simple config references.
- Added design scaffolding:
  - `doc/REFACTOR_DESIGN_SCAFFOLDING_2026-06-29.md`
  - `doc/STRUCTURAL_REFACTOR_PR_CHECKLIST.md`
- Polished baseline harness:
  - `tools/baseline/validation_record_template.md`
  - `tools/baseline/examples/tiny_reduction/`
  - `tools/baseline/examples/tiny_manifest.json`
  - updated `tools/baseline/README.md`
- Local lightweight checks only:
  - Python syntax parsed for all new Python tools with
    `$HOME/tolteca/bin/python`.
  - Config inventory tool ran successfully.
  - Refactor inventory tool ran successfully.
  - Tiny baseline manifest generated successfully.
  - Comparator reported the tiny generated manifest matches itself.
  - `tools/baseline/examples/tiny_manifest.json` passed `json.tool`.
  - `git diff --check` passed.
- No C++ source, CMake behavior, Citlali build/test/reduction, or Unity access
  was attempted.

## Update - 2026-06-29 21:57:01 EDT

- Inspected the copied Unity development tree under
  `/Users/gwilson/foo/citlali_dev/citlali`.
- The copied tree is on `gw_dev` at `376e0022`; its copied Unity build cache
  points at:
  - repo: `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali`
  - build dir: `build_unity_release_native_lto`
  - `FETCHCONTENT_SOURCE_DIR_TULA=/home/toltec_umass_edu/work_toltec/citlali_dev/tula`
  - GCC/GCC-ar 13 tooling and TolTEC shared extern include/lib paths
- Added Unity sync/build helpers:
  - `tools/unity/README.md`
  - `tools/unity/sync_to_unity.sh`
  - `tools/unity/build_on_unity.sh`
  - `tools/unity/make_source_bundle.sh`
  - `tools/unity/rsync_filter.txt`
  - `tools/unity/unity.env.example`
- Added workflow documentation:
  - `doc/UNITY_SYNC_BUILD_WORKFLOW_2026-06-29.md`
- Recommended morning path:
  - dry-run source copy to the refactor tree with
    `tools/unity/sync_to_unity.sh --dry-run`
  - live source copy to the refactor tree with
    `tools/unity/sync_to_unity.sh --live`
  - build on Unity with `tools/unity/build_on_unity.sh --live`
  - use `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor/build`
    for the refactor build
  - leave `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali` and its
    existing `build_unity_release_native_lto` untouched for gw_dev comparison
- Also added an scp-friendly source-bundle fallback via
  `tools/unity/make_source_bundle.sh`.
- Local checks only:
  - `bash -n` passed for all Unity shell helpers.
  - helper `--help` paths ran.
  - `tools/unity/build_on_unity.sh` dry-run printed the intended ssh command.
  - `tools/unity/make_source_bundle.sh` created a local test bundle under
    `/private/tmp`.
  - tarball listing confirmed source/docs/tools are included while `.git`,
    build dirs, reductions, and coadds are excluded.
  - `git diff --check` passed.
- No ssh, scp, rsync-to-Unity, Citlali build/test/reduction, or Unity access was
  attempted.

## Update - 2026-06-29 22:21:32 EDT

- Added a compact-config expansion prototype without changing Citlali runtime
  behavior:
  - `tools/config/expand_compact_config.py`
  - profile definitions under `tools/config/profiles/`
  - compact examples under `tools/config/examples/`
  - `doc/CONFIG_COMPACT_PROTOTYPE_2026-06-29.md`
  - updated `tools/config/README.md`
- Initial profiles:
  - `science_standard`
  - `science_diagnostic`
  - `pointing_standard`
  - `beammap_detector`
  - `tod_export`
- The expander loads `data/config.yaml`, applies the selected profile, applies
  compact user fields, then applies `expert:` overrides verbatim.
- Updated Unity sync/build helpers so the refactor tree is isolated from the
  gw_dev comparison checkout:
  - default remote refactor repo:
    `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor`
  - default refactor build dir: `build`
  - protected comparison repo:
    `/home/toltec_umass_edu/work_toltec/citlali_dev/citlali`
  - `sync_to_unity.sh` and `build_on_unity.sh` refuse to target the protected
    repo unless `ALLOW_BASELINE_REPO=true` is set.
- Updated Unity docs and env example to point at
  `citlali_dev/citlali_refactor/build`, leaving
  `citlali_dev/citlali/build_unity_release_native_lto` untouched for
  comparison.
- Local lightweight checks only:
  - Python syntax parsed for config/refactor/baseline Python tools with
    `$HOME/tolteca/bin/python`.
  - `expand_compact_config.py --list-profiles` loaded all profiles.
  - All compact examples expanded into `/private/tmp/*.expanded.yaml`.
  - Expanded YAML values were spot-checked for mode, output dir, thread count,
    mapmaking enabled/grouping, TOD output flags, and list-shaped inputs.
  - All expansion summaries had zero warnings.
  - `bash -n` passed for Unity shell helpers.
  - Unity helper `--help` paths ran.
  - `tools/unity/build_on_unity.sh` dry-run prints
    `citlali_refactor/build`.
  - Sync and build guard checks refused the protected gw_dev comparison repo.
  - `git diff --check` passed.
- No ssh, scp, rsync-to-Unity, Citlali build/test/reduction, or Unity access was
  attempted.
