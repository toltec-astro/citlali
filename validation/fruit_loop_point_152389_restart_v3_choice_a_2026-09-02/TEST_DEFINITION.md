# FRUIT checkpoint-v3 Choice-A exact-restart validation

Status: **completed; PASS; D19 closure evidence**

Test ID: `SCI-FRUIT-POINT-152389-D19-CHOICE-A-V3-R0.1`

## Question

Does the owner-approved bounded resolved-target state in checkpoint v3 make a
split FRUIT trajectory exactly equal to an uninterrupted trajectory for three
iterations after the checkpoint?

## Frozen setup

- Observation: `152389`, using only the development copy under
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389`.
- Base configuration:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-iter1-6-r0.1/setup/citlali_injected_source_injected.yaml`,
  SHA-256 `cf8899b0c9348c3a1b61fe1a00ee8aefdaa2422cecc90f63ad5eda19c921b007`.
- Repair source commit:
  `2b59ad642ffebb8798a5bcf2b2bf228916180dc0`.
- Executable: `build/bin/citlali`, reported version
  `sci-noi-v0.1-stage-a-27-g2b59ad642`, SHA-256
  `8665f09a44e57b5c3ba5fe24fb51e06af94bf3f92e98878a55a9c48b146a2418`.
- Execution: local `--grppiex seq` and `runtime.n_threads: 1`.
- Injection: the existing centered 100 mJy/beam source in every array,
  beginning at absolute iteration 1.
- New, isolated run root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/d19-choice-a-v3-r0.1`.

The base configuration's older restart is deliberately cleared. The
uninterrupted branch runs absolute iterations 0--7. Its completed iteration-4
version-3 checkpoint is the exact split point, and the restarted branch runs
absolute iterations 5--7 from that checkpoint. The two trajectories therefore
share one identical prefix. Apart from output location, restart binding,
one-thread execution, and the restart-required injection marker, the
scientific and learning settings are unchanged.

## Pass conditions

1. Restarted absolute iterations 5, 6, and 7 are bitwise equal to the matching
   uninterrupted products for every array and all three image extensions.
2. The effective sample-mask intervals, effective detector penalties,
   weight-validation state, and resolved next-iteration target state are
   identical at each compared completed-iteration checkpoint.
3. Every checkpoint reports schema
   `citlali-reduction-restart-checkpoint-v3`, the expected completed/next
   iteration identity, and a resolved target scope for observation `152389`.
4. Both runs complete without an unexpected error-level message.

Any mismatch fails D19 closure. Passing this test validates exact restart only
for this enabled feature combination and observation; it does not qualify the
FRUIT recurrence, choose stopping rules, or establish scientific superiority.

## Result

The uninterrupted and restarted branches both completed without an unexpected
error- or critical-level message. The iteration-4 split checkpoint contains
one `mapdiag:raw_obs` scope for observation `152389`, with source iteration 4,
apply iteration 5, a `3 x 375 x 371` map grid, and 12 resolved targets. The
restarted process applied those 12 targets at iteration 5.

All 27 required map-plane comparisons are bitwise equal: `signal_I`,
`kernel_I`, and `weight_I` for `a1100`, `a1400`, and `a2000` at absolute
iterations 5, 6, and 7. All checkpoint variables are also bitwise equal at
each of those three completed-iteration boundaries. Both branches contain one
resolved target scope, with 13, 3, and 3 targets after iterations 5, 6, and 7,
respectively, and seven effective detector penalties at every compared
boundary.

This closes D19 for the tested enabled path. It demonstrates that checkpoint
v3 carries the previously missing causal state for a three-iteration real
pointing replay. The result remains development evidence only and does not
promote the injected source, the per-iteration targets, or any checkpoint
product into an independently calibrated scientific sky product.

The exact comparisons are in
[`restart_replay_comparison.csv`](restart_replay_comparison.csv). The version-2
replay manifest in
[`restart_replay_manifest.json`](restart_replay_manifest.json) hashes the
executable, comparison tool, configurations, split checkpoint, logs, every
compared map product, and every compared checkpoint.
