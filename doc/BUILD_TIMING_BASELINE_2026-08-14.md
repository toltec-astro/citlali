# Citlali Spack Build Timing Baseline

Date: 2026-08-14

## Purpose

This baseline measures the development build costs required by the build-
integration acceptance plan. It separates build-system startup from a clean
compile and from the header-dominant production CLI translation unit. It is a
measurement record, not an optimization claim.

## Method

The campaign used `tools/build/measure_spack_build_times.py` from clean source
commit `3a4defda5308fa86a89fa7be24a991a17c9755b7`, the exact native
`macos-llvm20` profile, concrete Spack root
`u2qcns6o5rtnzzzblqvqsnbdr7uldxgf`, Release mode, Ninja, and eight build jobs
on the 14-logical-CPU Apple Silicon development host.

The harness:

1. configured an empty disposable build tree;
2. built the complete library, CLI, and compiled test targets;
3. immediately repeated the build without changes;
4. changed only the modification time of `src/citlali/cli/main.cpp` and rebuilt;
5. restored the original timestamp and verified the source SHA-256; and
6. removed the disposable build tree.

The exact manifest and logs are retained under
`validation/build_timing/20260814T111058-0400-macos-llvm20/`.

## Results

| Stage | Elapsed time | Result |
| --- | ---: | --- |
| Clean configure | 5.93 s | Pass |
| Clean build, 44 Ninja actions | 161.60 s | Pass |
| No-op build | 0.94 s | Pass; Ninja reported no work |
| CLI translation-unit incremental build | 173.22 s | Pass; compile plus link |

The clean build can compile independent targets concurrently. The incremental
case has only one expensive compile available, so eight-way build parallelism
cannot shorten it. The 161.60 versus 173.22 second difference is a single-run
spread and is not evidence that incremental compilation performs more work;
the logs show 44 clean-build actions versus two incremental actions.

## Interpretation

Build-system overhead is small: configure is about six seconds and a no-op
build is under one second. The dominant local edit/build cost is the production
CLI translation unit, which includes much of the application through the
header graph and takes roughly three minutes to compile by itself. This is
direct evidence for the existing header-boundary debt, but it does not by
itself justify a particular source split or optimization.

The logs also retain duplicate-RPATH linker warnings and one upstream C++23
`aligned_union` deprecation warning. Neither changed the result or explains the
compile duration; warning cleanup should remain separate from timing work.

## Remaining Evidence

Run the same harness under the Unity `unity-gcc13` profile. A single controlled
Unity campaign is sufficient for this acceptance checkpoint; repeated timing
trials are only needed if the first result is anomalous or if a later build-
boundary change claims an improvement.
