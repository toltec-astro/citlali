# Citlali Spack Build Timing Baseline

Date: 2026-08-14

## Purpose

This baseline measures the development build costs required by the build-
integration acceptance plan. It separates build-system startup from a clean
compile and from the header-dominant production CLI translation unit. It is a
measurement record, not an optimization claim.

## Method

The native campaign used `tools/build/measure_spack_build_times.py` from clean
source commit `3a4defda5308fa86a89fa7be24a991a17c9755b7`, the exact
`macos-llvm20` profile, concrete Spack root
`u2qcns6o5rtnzzzblqvqsnbdr7uldxgf`, Release mode, Ninja, and eight build jobs
on the 14-logical-CPU Apple Silicon development host. The Unity campaign used
clean source commit `4f9c7e55ab41c2d7ca88ffee86bc9e60e8c5727b`, the exact
`unity-gcc13` profile, concrete root `elcot6ah6jtv3onjjmqerxaznnshumqh`, Unix
Makefiles, and six jobs on `toltec-cpu002`.

The two revisions have identical timed C++ source and build commands; the
intervening commit records the native evidence and hardens failed-stage
recording. Both manifests report clean source status and the same SHA-256 for
`main.cpp`. The host results characterize their own development lanes and are
not treated as a controlled compiler-performance comparison.

The harness:

1. configured an empty disposable build tree;
2. built the complete library, CLI, and compiled test targets;
3. immediately repeated the build without changes;
4. changed only the modification time of `src/citlali/cli/main.cpp` and rebuilt;
5. restored the original timestamp and verified the source SHA-256; and
6. removed the disposable build tree.

The exact manifests and logs are retained under:

- `validation/build_timing/20260814T111058-0400-macos-llvm20/`; and
- `validation/build_timing/20260814T152240+0000-unity-gcc13/`.

## Results

| Stage | macOS LLVM 20 | Unity GCC 13 | Result |
| --- | ---: | ---: | --- |
| Clean configure | 5.93 s | 20.16 s | Pass |
| Clean build | 161.60 s | 315.56 s | Pass |
| No-op build | 0.94 s | 4.98 s | Pass |
| CLI translation-unit incremental build | 173.22 s | 277.38 s | Pass |

The clean build can compile independent targets concurrently. The incremental
case has only one expensive compile available, so eight-way build parallelism
cannot shorten it. The 161.60 versus 173.22 second difference is a single-run
spread and is not evidence that incremental compilation performs more work;
the logs show 44 clean-build actions versus two incremental actions.

## Interpretation

Build-system overhead is small relative to compilation on both hosts. The
dominant edit/build cost is the production CLI translation unit, which includes
much of the application through the header graph and takes about three minutes
on the Mac and 4.6 minutes on Unity to compile by itself. This is direct
evidence for the existing header-boundary debt, but it does not by itself
justify a particular source split or optimization.

The logs retain host-specific compiler and linker warnings, including duplicate
RPATH warnings on macOS and upstream template warnings on both platforms. None
changed the result or explains the compile duration; warning cleanup should
remain separate from timing work.

## Conclusion

The formal build-timing acceptance gate is complete for the native LLVM 20 and
Unity GCC 13 development lanes. Repeat a campaign only when diagnosing an
anomaly or evaluating a build-boundary change that claims an improvement.
