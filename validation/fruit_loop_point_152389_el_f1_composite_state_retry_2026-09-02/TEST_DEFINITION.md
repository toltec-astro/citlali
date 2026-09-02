# FRUIT EL-F1 r1 composite-state retry

Status: **complete; valid primary screen, neither candidate promising**

Test ID: `SCI-FRUIT-EL-F1-R1-COMPOSITE-STATE-RETRY-R0.1`

The owner approved the exact `EL_F1_R1_BUNDLE_MANIFEST_R0.1.md` on
`2026-09-02`. This retry keeps the original EL-F1 recurrence, source,
scientific metrics, prospective thresholds, resource limits, and non-effects.
It changes only the causal-state bookkeeping authorized by the r1 packet.

## Composite state

- checkpoint-bound ordinary complete map `Q_k` is the sole authority for
  newest weights and `MEDRMS`;
- separately checkpointed `F_k` contains relaxed signal/kernel plus exact
  method, alpha, observation, iteration, grouping, ordered-plane, spatial WCS,
  grid, and support identity; and
- the existing checkpoint contains learned operational state.

The next pass reloads `Q_k` normally and replaces only the accepted feedback
signal/kernel with `F_k`. No duplicate weight or RMS state is permitted.

## Frozen inputs and run matrix

- admitted data root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389`;
- fresh source YAML SHA-256:
  `dc0df89b706f1af9f32d747861f8c23975ded7cb0cf5c706110e7a96126d5909`;
- fixed alpha values: `1.00`, `1.25`, and `1.50`;
- injection disabled/enabled pair at 100 mJy/beam in every array from
  absolute iteration 1;
- every trajectory starts at iteration 0 and saves through iteration 6;
- sequential local execution with one configured thread; and
- new retained-output root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/el-f1-composite-state-retry-r0.1`.

The six variant YAML files differ only in output root, alpha, and injection
switch. `COMMON_LOCAL.yaml` retains the previous one-thread and optional-I/O
settings. One executable must be copied into the new setup directory, hashed,
and used for every trajectory without rebuild.

## Technical preflight and stop rules

Focused tests must establish alpha-one bit identity; Q-owned weight/RMS;
signal/kernel-only F state; fail-closed identity/support; absence of duplicate
weight/RMS checkpoint fields; harmless FITS-decimal RMS round trip; and v3
ordinary versus versioned experimental checkpoint separation.

The original six-primary/two-replacement, time, memory, disk, unexpected-log,
non-finite, route/grid/support, and checkpoint stop rules remain in force. The
analysis is frozen in `ANALYSIS_MANIFEST_R0.1.yaml`. An unfavorable scientific
outcome is retained and not rerun.

## Completed disposition

All six primary trajectories completed on the first attempt under the frozen
executable. Both non-unity candidates failed only the prospective final a1100
annular-residual check and are classified
`not_promising_on_this_compact_case`. No restart follow-up is required. See
`EXECUTION_RESULT_R0.1.md` for the complete bounded result and non-effects.
