# Citlali multi-observation SIGBUS investigation

## Status

Investigation in progress. The native failure has been narrowed to
`ceres::Solve`, but no native frame from the failed Unity process exists yet.
No fitter or scientific-policy change is justified at this point.

Primary external handoff:

`~/GitHub/tolapt/docs/citlali_multi_observation_sigbus_handoff_2026-07-24.md`

## Established evidence

- Both failed 108-observation jobs completed 45 observations and entered the
  same a1400 pointing fit for observation 137372.
- Each failed on the 137th `ceres::Solve` invocation. The a1100 fit was
  invocation 136; a1400 was invocation 137.
- The final pre-evaluation was complete and finite. The crash occurred after
  `ceres_fit solve start` and before `ceres_fit solve done`.
- Standalone observation 137372 completed with identical map dimensions,
  seed, limits, residual statistics, and initial Ceres cost.
- `785d18ec..d339053c` changes runtime CPU-resource discovery and capping.
  These jobs all realized six OpenMP threads, one Eigen thread, and one FFTW
  thread. The diff does not change fitter, map, solver, or observation
  ownership code.
- The fitter implementation at failed commit `785d18ec` is byte-identical to
  current HEAD.
- Current-RSS logs at observation boundaries range from approximately 0.8 to
  1.1 GiB after correcting an existing KiB-to-GiB reporting error. They do not
  show Citlali retaining 26 or 41 GiB.
- Slurm `MaxRSS` for a batch step can include the TolTECA Python parent, which
  remains alive while Citlali runs, as well as transient usage. The reported
  26/41 GiB values therefore cannot be assigned to the Citlali child without
  PID-level evidence.

## Exact fit replay

The failed v4 product tree contains all 135 maps from the 45 completed
observations. The standalone 137372 tree contains its three maps. The opt-in
`MapFitterLifecycle.DISABLED_ExactProductSequence` test reads `signal_I` and
`weight_I` from those products and replays all 138 fits in original
observation/array order.

The exact replay completes locally in one process, including invocation 137.
An AddressSanitizer/UndefinedBehaviorSanitizer build of the fitter test
translation unit also completes both:

- 512 repeated synthetic fits; and
- the exact 138-product sequence.

No sanitizer finding occurs. This rules out fit count plus real fit data as a
sufficient cause. It also makes an ordinary Ceres cost-function,
parameterization, or fit-local backing-storage leak unlikely.

## Remaining hypotheses

1. An observation-pipeline resource or state outside the fitter corrupts the
   process before invocation 137.
2. The old Unity executable or one of its mapped dependencies was damaged,
   replaced, or unavailable through its backing filesystem. A mapped-object
   failure is consistent with SIGBUS, no C++ exception, and the absence of
   output from the old backtrace handler.
3. Both failed jobs ran on the same unhealthy node or encountered the same
   persistent storage fault.
4. The runtime-resource commit has a platform-specific effect despite
   identical realized thread counts. This requires a controlled old/new
   experiment, not inference from the standalone success.

The project owner reports that developer binaries are sometimes copied and
the build tree recompiled while reductions may be active. Copying a binary to
a backup name is harmless. Rewriting or truncating the path mapped by a
running process could produce `SIGBUS` on a later page fault; a normal linker
replacement should instead leave the running process attached to its old
inode. The exact command and timing for the failed jobs have not been
recovered, so this remains a plausible operational hypothesis rather than the
established root cause.

## Added diagnostics

- `process_resource_snapshot.h` records current and peak RSS, virtual memory,
  thread count, open descriptor count, mapping count, and live executable
  inode/path at every observation start and Ceres solve when
  `CITLALI_PROCESS_RESOURCE_DIAGNOSTICS=1`. The native reproducer enables it;
  ordinary reductions do not pay solve-level diagnostic cost.
- The physical-memory summary now converts KiB to GiB correctly.
- The fatal-signal handler writes signal number, `si_code`, and fault address
  before attempting a backtrace. `SIGBUS` codes distinguish address,
  alignment, and mapped-object failures.
- `subset_multi_observation_config.py` creates contiguous-prefix or
  repeated-observation reproducers from the original effective YAML.
- `run_citlali_native_reproducer.sh` verifies and copies the executable to
  node-local storage, bypasses the TolTECA parent, records linked libraries,
  and runs directly under `/usr/bin/time -v` or GDB.

TolPROJ commit `76daaa6` adds a separate operational safeguard.
`tolproj submit-reduction` resolves and snapshots the selected Citlali
executable before `sbatch`, addresses it by SHA-256, passes that immutable
identity in the queued job environment, and runs a checksum-verified
node-local copy. Newly generated refactor reductions also use the launcher
for direct `sbatch`; that fallback selects the binary when the job starts and
warns that it does not freeze the queued version. This removes mutable
build-tree mappings from future running jobs but does not by itself establish
the cause of the historical failures.

## Next controlled runs

Use the current instrumented commit and the v4 effective config.

1. Run 137218 followed by 137372 from a node-local executable copy.
2. If that passes, binary-search the immediately preceding history using
   history counts 23, 34, 40, and 45.
3. If only the full history fails, repeat one benign observation 45 times
   before 137372. This separates count from a particular predecessor.
4. Run the shortest failing case with
   `run_citlali_native_reproducer.sh --gdb`.
5. Compare direct-VAST and node-local execution only if a current failure is
   reproduced.
6. Complete the full 108-observation run after the fix and compare every
   completed observation against its accepted standalone/old-process product.

The two-observation config is generated with:

```bash
python tools/diagnostics/subset_multi_observation_config.py \
  ORIGINAL_EFFECTIVE.yaml two_observation.yaml \
  --terminal-obsnum 137372 \
  --history-count 1 \
  --output-dir /work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1/sigbus-reproducer/two-observation
```

Run it directly with:

```bash
tools/unity/run_citlali_native_reproducer.sh \
  build/bin/citlali two_observation.yaml
```

Before interpreting the next run, recover node placement for jobs 62131625,
62131626, and 62151845 and determine whether the Unity executable was rebuilt
while either long process was active.
