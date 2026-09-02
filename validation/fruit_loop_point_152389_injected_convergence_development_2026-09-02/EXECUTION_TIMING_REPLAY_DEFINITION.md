# Iterations 5–6 execution-timing replay

Status: **completed diagnostic replay; timing anomaly did not repeat; exact
restart failed on the second continued iteration; not a performance
qualification or benchmark**

The continuous injected branch completed scientifically, but iteration wall
times changed from approximately 41–51 seconds for iterations 1–4 to 1361 and
2649 seconds for iterations 5 and 6. Log gaps occurred in several unrelated
scan-processing phases rather than one isolated timed operator. The cause is
unavailable from that execution alone.

This replay starts from the exact completed injected iteration-4 checkpoint:

- path:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-iter1-6-r0.1/injected/reduced/redu03/citlali_restart_checkpoint.nc`
- SHA-256:
  `697c13734e28204183276c24dc1fbf530c2024a66bb80ad5fe933e75b4814fc0`

It retains the injected-branch science configuration, 100 mJy/beam source,
sequential execution, one thread, diagnostics, output suppression, and
exclusive stop at 7. It changes the restart and output paths. Citlali also
requires a restarted injection's `start_iteration` to equal the checkpoint's
`next_iteration`, so the replay binds that restart marker to 5. The carried
checkpoint already contains iterations 1--4; this is restart bookkeeping, not
a new source or recurrence. It reruns absolute iterations 5 and 6 under
`/Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-iter1-6-r0.1/timing-replay-from-iter4/attempt-02`.

The replay is successful only as a diagnostic if:

1. its products carry absolute iterations 5 and 6;
2. `signal_I`, `kernel_I`, and `weight_I` are bitwise equal to the continuous
   injected branch for all arrays at both iterations; and
3. `/usr/bin/time -l` and Citlali's per-iteration profiles are retained without
   turning a single local run into a performance claim.

If numerical identity fails, the scientific convergence record requires
investigation. If identity passes and timing is ordinary, the earlier elapsed
times are recorded as a transient unresolved execution anomaly. If the large
slowdown repeats, it becomes a reproducible performance-development finding
requiring separate diagnosis.

Attempt 1 retained `start_iteration: 1`. Citlali rejected it before scan
processing because the iteration-4 checkpoint requires `start_iteration: 5`.
No scientific product was produced. Attempt 2 applies the required binding and
uses a distinct output root.

## Attempt 2 result

Attempt 2 completed absolute iterations 5 and 6 in `39.843821` and
`40.379260` seconds, respectively; `/usr/bin/time -l` reports `80.74` seconds
elapsed and zero swaps. The original `1361.209607`- and `2649.503333`-second
iterations did not repeat. They are retained as a transient, unresolved host
or execution anomaly and are not used as algorithm-performance evidence.

The numerical-identity condition failed. Iteration 5 is bitwise equal to the
continuous branch in every `signal_I`, `kernel_I`, and `weight_I` image.
Iteration 6 remains exact for `a1400` and `a2000`, but all three `a1100`
images differ:

| Product | Difference RMS / reference RMS | Maximum absolute difference |
| --- | ---: | ---: |
| `a1100 signal_I` | `0.1212060215` | `151.6869421` |
| `a1100 kernel_I` | `0.0001447336` | `0.0001945065` |
| `a1100 weight_I` | `0.0045009378` | `0.0016464048` |

The first changed state appears in the iteration-5 checkpoint. The continuous
checkpoint contains three effective detector penalties, while the replay has
two. The missing row is a `mapdiag:raw_obs`
`map_pixel_outlier_detector_dominance` exclusion learned at iteration 5 for
scan 11, UID 1489, array 0, with score 8. The continuous log records 11
targeted contributor pixels sourced from iteration-4 outliers and then learns
this exclusion. The replay begins with no restored map-pixel-outlier history,
does not enable that targeted tracing, and does not learn the row.

This is consistent with the implementation: operational masks and existing
penalties are serialized by
`src/citlali/core/pipeline/reduction_restart_checkpoint.cpp`, while the
`map_pixel_outliers` diagnostic vector is not. That vector is nevertheless
consumed across iterations by
`include/citlali/core/engine/detail/learning_targets_impl.h` to choose targeted
contributor tracing. The history is therefore causal under this configuration,
despite being classified as diagnostic in `ReductionLearningState`.

Exact values are in
[`restart_replay_comparison.csv`](restart_replay_comparison.csv), and
[`restart_replay_manifest.json`](restart_replay_manifest.json) hashes the
compared products, checkpoints, configs, logs, timing, and comparison tool.
