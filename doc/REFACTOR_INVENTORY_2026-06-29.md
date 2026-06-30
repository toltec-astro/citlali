# Citlali Refactor Inventory - 2026-06-29

This is a static inventory for planning structural refactor work. It is not a
compile or runtime validation result.

Generated with:

```bash
$HOME/tolteca/bin/python tools/refactor/refactor_inventory.py \
  --repo . \
  --json-out /private/tmp/citlali_refactor_inventory.json \
  --markdown-out /private/tmp/citlali_refactor_inventory.md
```

## Summary

| Item | Count |
| --- | ---: |
| Direct exit calls | 144 |
| Headers scanned | 43 |
| Commented CMake source entries | 7 |
| Simple config references | 234 |

## Direct Exit Calls

Direct process exits are still common in library code. They should migrate to
typed failures subsystem by subsystem, not as one broad mechanical sweep.

By subsystem:

| Subsystem | Count |
| --- | ---: |
| `engine` | 64 |
| `timestream` | 40 |
| `rtc` | 15 |
| `mapmaking` | 11 |
| `utils` | 5 |
| `other` | 4 |
| `ptc` | 3 |
| `cli` | 2 |

By risk label:

| Risk label | Count |
| --- | ---: |
| `library_header_high` | 134 |
| `cli_boundary` | 6 |
| `library_runtime_review` | 4 |

Interpretation:

- CLI exits are acceptable as a boundary behavior.
- Library header exits are the highest-value migration target.
- Start with config/IO/preflight exits where throwing typed failures does not
  touch hot loops.
- Do not throw exceptions for ordinary per-sample/per-detector control flow.

## Large Headers

| Header | Lines | Includes |
| --- | ---: | ---: |
| `include/citlali/core/engine/engine.h` | 9674 | 67 |
| `include/citlali/core/engine/beammap.h` | 6916 | 18 |
| `include/citlali/core/timestream/rtc/rtcproc.h` | 6337 | 21 |
| `include/citlali/core/timestream/ptc/ptcproc.h` | 5841 | 27 |
| `include/citlali/core/timestream/timestream.h` | 3432 | 27 |
| `include/citlali/core/utils/matplotlibcpp.h` | 2986 | 11 |
| `include/citlali/core/timestream/ptc/clean.h` | 2136 | 20 |
| `include/citlali/core/engine/todproc.h` | 1671 | 9 |
| `include/citlali/core/mapmaking/wiener_filter.h` | 1600 | 16 |
| `include/citlali/core/mapmaking/wiener_filter_omp.h` | 1599 | 18 |
| `include/citlali/core/timestream/rtc/despike.h` | 1420 | 10 |
| `include/citlali/core/utils/utils.h` | 1372 | 16 |
| `include/citlali/core/mapmaking/jinc_mm.h` | 1358 | 14 |

Practical sequence:

1. Do not start by moving `engine.h` wholesale.
2. Extract orchestration from the CLI first so runtime flow is easier to test.
3. Move non-template, non-hot helper implementations into `.cpp` files one
   source target at a time.
4. Leave template-heavy and hot-loop code in headers until Unity build and
   benchmark coverage exists.

## Headers With Simple Non-Template Definitions

The inventory script found simple non-template definitions in these headers.
This is only a regex starting point; every candidate needs manual review.

| Header | Non-template member defs | Non-template free defs | Lines |
| --- | ---: | ---: | ---: |
| `include/citlali/core/utils/matplotlibcpp.h` | 99 | 5 | 2986 |
| `include/citlali/core/engine/engine.h` | 3 | 30 | 9674 |
| `include/citlali/core/engine/beammap.h` | 13 | 10 | 6916 |
| `include/citlali/core/engine/learning.h` | 0 | 17 | 434 |
| `include/citlali/core/timestream/ptc/ptcproc.h` | 7 | 6 | 5841 |
| `include/citlali/core/timestream/ptc/clean.h` | 1 | 9 | 2136 |
| `include/citlali/core/timestream/rtc/despike.h` | 6 | 3 | 1420 |
| `include/citlali/core/engine/io.h` | 0 | 9 | 666 |
| `include/citlali/core/timestream/rtc/rtcproc.h` | 4 | 4 | 6337 |
| `include/citlali/core/utils/utils.h` | 1 | 6 | 1372 |
| `include/citlali/core/mapmaking/map.h` | 0 | 6 | 252 |

Move candidates should be prioritized by low risk, not just count. Good early
targets are non-template utility/config/preflight helpers. Avoid moving
mapmaking hot loops or template-heavy processor code until validation is ready.

## Commented CMake Source Entries

`CMakeLists.txt` currently lists several natural implementation files as
commented-out sources:

- `src/citlali/core/engine/todproc.cpp`
- `src/citlali/core/engine/kidsproc.cpp`
- `src/citlali/core/engine/engine.cpp`
- `src/citlali/core/mapmaking/wiener_filter.cpp`
- `src/citlali/core/engine/lali.cpp`
- `src/citlali/core/engine/pointing.cpp`
- `src/citlali/core/engine/beammap.cpp`

These are good eventual boundaries, but they should be re-enabled only after a
small implementation move has Unity compile coverage.

## Config Ownership

Simple config-reference counts by subsystem:

| Subsystem | References |
| --- | ---: |
| `rtc` | 131 |
| `engine` | 48 |
| `other` | 45 |
| `cli` | 10 |

Simple config-reference counts by top-level node:

| Node | References |
| --- | ---: |
| `timestream` | 135 |
| `pointing_offsets` | 8 |
| `filepath` | 7 |
| `source` | 5 |
| `fitter` | 4 |
| `fsmp` | 4 |
| `map` | 4 |
| `meta` | 4 |
| `solver` | 4 |
| `type` | 4 |
| `runtime` | 3 |

Interpretation:

- `timestream` is both the largest YAML surface and the largest direct config
  reader surface.
- Typed config work should start with top-level runtime/map/noise/output values
  and then move into RTC/PTC sections.
- CLI extraction should keep config merge and reduction-type dispatch simple
  while the typed config model is introduced.

## Inventory Conclusions

The first structural PRs should not be file movement. The safer order is:

1. Baseline and comparison harness.
2. Config inventory/classification and typed config design.
3. `PipelineRunner`/`ReductionSession` extraction plan.
4. Typed failure hierarchy and staged exit migration.
5. Small `.cpp` boundary moves with Unity compile coverage.

This keeps science behavior and performance protected while making each later
diff reviewable.
