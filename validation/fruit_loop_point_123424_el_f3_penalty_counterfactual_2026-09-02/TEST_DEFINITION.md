# FRUIT EL-F3 late-penalty counterfactual

Status: **registered before checkpoint intervention or execution**

Test ID: `SCI-FRUIT-EL-F3-LATE-PENALTY-COUNTERFACTUAL-R0.1`

## Question

Did the newly learned zero-factor exclusion of detector UID 4460 materially
cause the EL-F2 alpha-1.25 a1400 collapse from iteration 4 to iteration 5, or
was the exclusion merely a warning of some other instability?

This is a mechanism test on an already exposed development case. It is not a
new candidate-method screen.

## Intervention

The original EL-F2 control and injected iteration-4 reduction directories are
copied into a new isolated development root. The originals remain read-only.

- The control checkpoint is an unmodified sham copy.
- From the injected checkpoint copy, a validated tool removes exactly one
  effective detector-penalty record and no other state:
  `producer=mapdiag:raw_obs`,
  `reason=map_pixel_outlier_detector_dominance`, `iteration=4`, `scan=5`,
  `uid=4460`, `network=-1`, `array=1`, `factor=0`, `score=4`, and
  `scan_local=true`.

Every other checkpoint dimension, variable, value, type, attribute, and
learning/processing-policy snapshot must remain equal. A machine-readable
audit records the source and transformed hashes and the exact removed row.
The transformed checkpoint remains a deliberately intervened scientific
state, not an exact historical checkpoint.

Both copied states advance exactly once, from absolute iteration 4 to 5, with
the already frozen EL-F2 executable and configuration. The injected source is
enabled from the restarted iteration 5 as required by the restart contract.
Any newly rediscovered penalty at the end of iteration 5 cannot affect that
iteration's already completed map and is retained as evidence.

## Required controls and comparisons

1. The control source directory and checkpoint copy must be byte-identical to
   the original source before execution.
2. The sham control iteration-5 signal, kernel, and weight images must be
   bitwise equal to the original EL-F2 control iteration 5 in every array.
   All checkpoint variables must also be value-identical.
3. The counterfactual injected iteration-5 products are paired with the sham
   control products. The known 100 mJy/beam source is measured with the same
   fitting, kernel normalization, finite-support, and 40--120 arcsec annular
   residual definitions used by EL-F2.
4. All arrays and all inherited EL-F2 science protections are reported, but
   they do not override the mechanism classification.

The primary a1400 effect fractions are

\[
q_R = \frac{R_{5,\mathrm{cf}}-R_{5,\mathrm{original}}}
             {R_{4,\mathrm{original}}-R_{5,\mathrm{original}}},
\qquad
q_A = \frac{A_{5,\mathrm{original}}-A_{5,\mathrm{cf}}}
             {A_{5,\mathrm{original}}-A_{4,\mathrm{original}}},
\]

where \(R\) is kernel-normalized central recovery and \(A\) is annular
residual over injected truth. The frozen original values are
`R4=0.8904511775839052`, `R5=0.8228281905201593`,
`A4=0.005591061898999446`, and `A5=0.02147410602676817`.

The prospective mechanism classification is:

- `substantial_causal_contribution` when both `q_R >= 0.5` and `q_A >= 0.5`;
- `partial_causal_contribution` when both are positive but the substantial
  condition is not met;
- `mixed_effect` when the fractions do not have the same sign, including one
  being zero; or
- `no_support_for_causal_contribution` when both are non-positive.

`q_R >= 1` and `q_A >= 1` is additionally reported as full reversal of these
two observed losses. These deterministic fractions describe this one
intervention; they are not population-level effect estimates.

## Bounds and stop rules

- new output root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f3-penalty-counterfactual-r0.1`;
- exactly two one-iteration local replays, sham control followed by injected
  counterfactual;
- one configured thread and sequential execution;
- the exact preserved EL-F2 executable, without a rebuild;
- at most one replacement of a trajectory for an environmental or interrupted
  start, never for an unfavorable result;
- 2 hours and 64 GiB per replay, 5 hours and 20 GiB retained in aggregate; and
- stop on checkpoint-audit failure, sham-control mismatch, non-finite required
  products, unexpected error/critical logging, or resource-limit breach.

No source checkpoint or reduction product may be edited in place. No Citlali
algorithm or configuration default may change. No further iteration,
candidate, observation, or tuning follows automatically.

## Claim limit

Even `substantial_causal_contribution` would establish only that this one hard
penalty materially contributed to this one exposed a1400 failure. EL-F1
already demonstrates that late degradation can occur without a new a1100
penalty. This test therefore cannot establish a universal guard, select a
recurrence, rescue EL-F2, qualify a method, or authorize production use.
