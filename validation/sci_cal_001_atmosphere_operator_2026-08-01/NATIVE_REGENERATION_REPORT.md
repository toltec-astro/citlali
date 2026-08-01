# SCI-CAL-001 AM 12.2 native regeneration report

## Verdict

The complete `180`-case annual AM 12.2 matrix was structurally valid; all parsed fields match exactly. Exact parsed-field matches: `180`; parsed-field mismatches: `0`. Numeric data-line byte matches: `180`; byte mismatches: `0`.

This is a software/numerical regeneration check for the copied AM 12.2 annual profiles. It does not establish that these profiles are the exact legacy `am_q25/q50/q75/q95` inputs, and it does not select or authorize an atmosphere operator.

A predecessor parallel attempt used one shared `AM_CACHE_PATH` and was rejected from canonical evidence after 28 of 180 cases emitted cache-mutation warnings (22 `insert_as_mru`, 9 `promote_to_mru`, with overlap). Its numeric data lines were still exact, but those warnings fail the software-execution contract. A second numerically exact sharded attempt with zero cache, unknown-warning, or error diagnostics was superseded because it did not yet bind the complete execution context or commit warning-bearing output identity. The canonical matrix reported here satisfies both requirements; its unresolved-line warnings and status 1 remain explicit and are not described as a clean software success.

## Build identity

The copied Linux reference binary is SHA-256 `3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c` and identifies itself as `am version 12.2 (build date Aug 26 2022 19:20:13)`. The regeneration executable is classified as `native_macos_build_distinct_from_copied_linux_binary`, format `mach-o`, SHA-256 `78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb`. Same bytes as the copied Linux binary: `false`.

Native compiler provenance status: `supplied_by_operator_as_build_compiler`. Native build command supplied: `make -j8 gcc-omp COMPILER_GCC=gcc-15`.

## Execution and comparison contract

Each case used the historical AM argv body `profile 0 GHz 500 GHz 10 MHz ZA deg 1.0`, without `srun`, with pinned `LANG=C`, `LC_ALL=C`, the requested `OMP_NUM_THREADS`, and a deterministically assigned cache shard below `--cache-dir/am_cache`. One whole-cache POSIX writer lock excludes other processes. Within each of two nonoverlapping phases, one ordered worker queue owns each shard; the q95 ZA10/ZA70 smoke phase must complete exactly before the remaining matrix begins. Generated combined stdout/stderr and execution sidecars remain in the external cache. The comparison parses all 50,001 rows and requires exact binary64 equality independently for frequency, tau, transmission, Rayleigh-Jeans temperature, and brightness temperature.

Committed identity includes both normalized numeric data text and normalized warning-bearing combined output. The latter replaces only the volatile runtime and dcache-counter header lines; it preserves the AM identity, configuration, numeric grid, warning lines, and all other output. Each sidecar binds its raw and normalized output digests to the immutable cache execution-context SHA-256.

AM return code 1 is accepted only when the complete grid accompanies the canonical unresolved-narrow-line warning with count 86, 87, or 88. Cache insert/promote warnings, unknown warning classes, error lines, and other nonzero statuses fail closed.

## Aggregate differences

| Field | Maximum absolute difference |
| --- | ---: |
| `frequency` | `0.00000000000000000e+00` |
| `tau` | `0.00000000000000000e+00` |
| `tb` | `0.00000000000000000e+00` |
| `trj` | `0.00000000000000000e+00` |
| `tx` | `0.00000000000000000e+00` |

Return-code counts: `{"1": 180}`. Warning-count distribution: `{"86": 72, "87": 108}`.

Regeneration AM identity distribution: `{"am version 12.2 (build date Aug  1 2026 11:20:29)": 180}`.

Warning-class totals: `{"cache_insert_as_mru_warning_line_count": 0, "cache_promote_to_mru_warning_line_count": 0, "other_warning_line_count": 0, "unresolved_column_warning_line_count": 6480, "unresolved_summary_warning_line_count": 180}`. Error-line total: `0`. Normalized numeric-output aggregate SHA-256: `18abf7fb57f335637c7cb2e105aea910f491d74dcd485df01c63ef759a28cd5c`. Normalized warning-bearing full-output aggregate SHA-256: `fc465133e1cc2ac7458f593209dd8b0adbf320ba79a233fcf852f018883aefaf`.

## Provenance closure

`native_regeneration_metrics.csv` SHA-256 is `1d6f099383880207bca94cc0f0236a379a158a0be17e4a365b62371cb1ebca87`. `native_regeneration_manifest.json` SHA-256 is `128d2b8481d64120be2fac020658f9f6abbe3de620438563572e6d40d8493ac4`. The external cache execution context is `execution_context.json` SHA-256 `8ff9af2fa844db88f94ca27585e2f33854dc38fe5422935dc57865a669e60093`, and its complete content is copied into the committed manifest. It binds the runner, copied and regeneration binaries, compiler and build command, AM source, five annual profiles, all 180 copied reference grids, frozen historical scripts, argv, run scope, ordered shard topology, locale, and actual execution host.

Uploader logs and credentials are deliberately excluded. No network or Unity access is part of this workflow.
