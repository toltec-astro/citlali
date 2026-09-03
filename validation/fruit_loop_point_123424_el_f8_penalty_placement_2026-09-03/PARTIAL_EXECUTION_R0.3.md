# EL-F8 R0.3 Partial Execution

## Status

R0.3 stopped during the third ordered trajectory and has no scientific
penalty-placement interpretation.  Both current-placement trajectories
completed and passed the mandatory compatibility gate.  `c5-map` then
stopped before iteration 5 because exact restart correctly rejected a
`pre_mapmaking` run configuration paired with a checkpoint whose learning
policy records the historical `pre_cleaning` placement.  `a5-map` was not
started and the full analysis was not run.

No R0.3 product will be reused by a later registration.

## Valid current-placement compatibility evidence

| Trajectory | Exit | Wall (s) | Maximum resident bytes | Log SHA-256 |
|---|---:|---:|---:|---|
| `c5-current` | 0 | 30.91 | 857,849,856 | `0542bc4c89f6e739d429c9ebfd0ac77c40c714f3b3154431d5112266b1dd36e5` |
| `a5-current` | 0 | 30.19 | 859,226,112 | `fc27070a4ce7c95c29a800b83ec2e688e6705131fb64e92bfaada52676b1cacc` |

Both trajectories produced absolute iteration 5 in local `redu00`
directories and had no unexpected error-level messages.  Against the
existing EL-F5 C5/A5 references, each reproduced all nine signal, kernel, and
weight planes bitwise.  Every scientific checkpoint value was identical;
the only observed allowed difference was normalized `learning_policy_yaml`.

## Aborted moved-placement trajectory

- trajectory: `c5-map`
- exit status: `1`
- wall time: `0.56 s`
- maximum resident set size: `46,678,016 bytes`
- log size: `14,639 bytes`
- log SHA-256:
  `63ed8cee0a23f06048bfa1c6e02e92fcbeb332f75aa9d425533677a20a6c50b8`
- failure: exact restart rejected a learning-policy mismatch
- scientific products: none; only the output-root lock was created

## Bounded intervention correction

Exact-restart checking must not be weakened.  A later registration may use a
fail-closed editor on fresh copies of the two map-placement checkpoints.  The
editor must change only
`learning_policy_yaml.map_pixel_outlier_detector_exclusion_application` from
its normalized historical value `pre_cleaning` to `pre_mapmaking`, verify all
other values, types, dimensions, and attributes unchanged, and emit a
machine-readable audit.  The transformed checkpoint hashes and audits must
be frozen before replay.

This makes the already-approved placement intervention explicit in the
complete restart state.  It does not change the feedback maps, learned
penalty records, source injection, recurrence, operator implementation,
measurements, or claim limits.
