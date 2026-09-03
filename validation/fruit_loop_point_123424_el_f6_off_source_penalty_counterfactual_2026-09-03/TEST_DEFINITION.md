# FRUIT EL-F6 off-source penalty counterfactual

Status: **registered before checkpoint intervention or execution**

Test ID: `SCI-FRUIT-EL-F6-OFF-SOURCE-PENALTY-COUNTERFACTUAL-R0.1`

## Question

Did applying the carried, factor-zero UID 4460 a1400 penalty materially cause
the off-source response-shape and residual-leakage degradation observed from
EL-F5 iteration 4 to iteration 5?

This is a causal mechanism test on one already exposed development trajectory.
It is not a new method screen, detector-quality decision, penalty-policy
proposal, or FRUIT qualification.

## Intervention and sham

The complete EL-F5 off-source injected iteration-4 reduction directory is
copied twice into a new isolated development root. The original remains
unchanged.

- The first copy is an untouched sham restart.
- From the second copy, the existing fail-closed checkpoint editor removes
  exactly one effective detector-penalty record and no other state:
  `producer=mapdiag:raw_obs`,
  `reason=map_pixel_outlier_detector_dominance`, `iteration=4`, `scan=5`,
  `uid=4460`, `network=-1`, `array=1`, `factor=0`, `score=4`, and
  `scan_local=true`.

Every other checkpoint dimension, variable, value, type, and attribute must
remain equal. A machine-readable audit records both hashes, the removed row,
and the equality checks. The edited checkpoint is a deliberate causal
intervention, not an exact historical state.

Both copied states advance exactly once, from absolute iteration 4 to 5, with
the frozen EL-F5 executable and configuration. The 100 mJy/beam source remains
at FITS map-world position `(AZOFFSET, ELOFFSET) = (0, -60) arcsec` and is
enabled for the restarted iteration. A penalty learned again at the end of
iteration 5 cannot affect that iteration's completed map and is retained as
evidence.

## Required validity gates

1. Each copied source directory and checkpoint must equal the EL-F5 source
   before intervention.
2. The sham iteration-5 signal, kernel, and weight planes must be bitwise equal
   to the original EL-F5 off-source injected iteration 5 in all arrays. All
   checkpoint variables must be value-identical.
3. The intervention audit must prove exactly one matching penalty removed and
   all other checkpoint content preserved.
4. Counterfactual a1100 and a2000 signal, kernel, and weight planes must remain
   bitwise equal to the original EL-F5 off-source injected iteration 5. Only
   the targeted a1400 trajectory may change.
5. Required products must be finite and complete, and both runs must have zero
   unexpected error- or critical-level messages.

Failure of any validity gate invalidates the causal comparison and stops the
test without retuning.

## Measurement and prospective causal classification

The existing EL-F5 no-injection control iteration 5 remains the fixed paired
control. For each array, the counterfactual source response is

`T_5,cf = signal_I(counterfactual injected, 5) - signal_I(EL-F5 control, 5)`.

The response is measured at the registered off-source position using the
same fitting, same-iteration kernel normalization, finite support, and
40--120 arcsec annulus as EL-F5. The complete response is retained.

The primary a1400 reversal fractions are

\[
q_K = \frac{K_{5,\mathrm{original}}-K_{5,\mathrm{cf}}}
             {K_{5,\mathrm{original}}-K_{4,\mathrm{original}}},
\qquad
q_A = \frac{A_{5,\mathrm{original}}-A_{5,\mathrm{cf}}}
             {A_{5,\mathrm{original}}-A_{4,\mathrm{original}}},
\]

where `K` is kernel-residual relative RMS and `A` is annular residual over
injected truth; lower values are better for both. The frozen EL-F5 values are
`K4=0.32067306391201056`, `K5=0.7278038848217114`,
`A4=0.0032704798085909136`, and `A5=0.02313442849087168`.

The prospective classification is:

- `substantial_causal_contribution` when both `q_K >= 0.5` and `q_A >= 0.5`;
- `partial_causal_contribution` when both are positive but the substantial
  condition is not met;
- `mixed_effect` when the fractions have different signs, including one being
  zero; or
- `no_support_for_causal_contribution` when both are non-positive.

Both fractions at least one are additionally reported as full reversal. These
deterministic fractions apply only to this intervention.

Central and full-kernel recovery, widths, and centroid error are reported as
secondary diagnostics. Recovery correctness is distance from unity; a larger
or smaller raw amplitude is not intrinsically preferred. These secondary
quantities do not replace the registered causal classification.

## Bounds and stop rules

- new output root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f6-off-source-penalty-counterfactual-r0.1`;
- exactly two sequential one-iteration replays: untouched sham, then the
  UID-4460-removed counterfactual;
- one configured thread and `--grppiex seq`;
- the exact preserved EL-F5 executable, without a rebuild;
- at most one replacement per replay for an environmental or interrupted
  start, never for an unfavorable result;
- 2 hours and 64 GiB per replay, 5 hours and 8 GiB retained in aggregate; and
- stop after the registered analysis without adding variants, iterations,
  thresholds, or tuning.

## Claim limit

A favorable result would establish only that applying this one carried hard
penalty causally contributed to the off-source a1400 degradation in this one
observation and source location. It would strengthen the same-observation
mechanism found at map center, but would not establish generic behavior across
detectors, geometries, source amplitudes, or observations; judge UID 4460;
select a safeguard; qualify a recurrence; or authorize production use.

