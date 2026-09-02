# FRUIT centered-source convergence development test: pointing 152389

Status: **completed exploratory development test; not qualification,
candidate ranking, or a stopping-rule decision; exact-restart follow-up
failed**

Test ID: `SCI-FRUIT-POINT-152389-INJECT-CENTER-100MJY-ITER1-6-R0.1`

Parent test:
`SCI-FRUIT-POINT-152389-INJECT-CENTER-100MJY-R0.1`

## Question

For the same known centered 100 mJy/beam compact source, how do amplitude,
shape, centroid, and the complete injected-minus-control response evolve from
absolute FRUIT iteration 1 through iteration 6?

This is a complete replay and extension, not a splice. Both the control and
injected branches restart from the same frozen iteration-0 checkpoint and run
continuously through iteration 6. This preserves a single causal trajectory
for each branch and makes the previously measured iterations 1 and 2 exact
replay checks.

## Frozen inputs and settings

- Observation: `152389`, development-only copy under
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389`.
- Source configuration SHA-256:
  `dc0df89b706f1af9f32d747861f8c23975ded7cb0cf5c706110e7a96126d5909`.
- Shared restart:
  `attempt-04/reference/reduced/redu00/citlali_restart_checkpoint.nc`, SHA-256
  `85419a82e050ae5d3685313abf413e239ffab272d75d336bd43633fc845dfdf8`.
- Executable: `build/bin/citlali`, reported version
  `sci-noi-v0.1-stage-a-22-g92d174630`, SHA-256
  `5a6f6741ee81c0b78ff718d2d4b6674f3b9a27a476a6d4a29105d3b68b319c38`.
- Execution: local `--grppiex seq`, `runtime.n_threads: 1`, and the same
  optional full-timestream output suppression used by the parent pair.
- Injection: enabled only in the injected branch, active from absolute
  iteration 1, with `100 mJy/beam` in `a1100`, `a1400`, and `a2000`.
- Source identity: the central pointing kernel already created by the
  reduction; no off-center or extended source is introduced.
- Saved iterations: every absolute iteration from 1 through 6.
- New run root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-152389/fruit-injection-development/centered-100mjy-iter1-6-r0.1`.

The only intended changes from the parent pair are the new output root and
raising the exclusive iteration stop from 3 to 7. No threshold, mask,
filtering, learning, weighting, injection, or mapmaking setting is retuned.

## Predeclared checks and measurements

1. The new control at absolute iterations 1 and 2 must be bitwise equal to the
   parent control in `signal_I`, `kernel_I`, and `weight_I` for all arrays.
2. The new injected branch at absolute iterations 1 and 2 must be bitwise equal
   to the parent injected branch in the same arrays and extensions.
3. Every new product must carry the expected absolute `FRUITLOOPS_ITER` value
   1, 2, 3, 4, 5, or 6 exactly once per branch.
4. At each iteration, form
   `transfer = injected signal_I - control signal_I`.
5. At the known central injection position, report Gaussian peak recovery
   normalized by both 100 mJy/beam and the same-iteration injected `kernel_I`.
6. Report the full-map projection of the response onto the same-iteration
   injected `kernel_I`, divided by 100 mJy/beam.
7. Report transfer/kernel FWHM ratios, centroid separation, projection
   residual, successive response-map change, and control/injected kernel and
   weight differences.

No convergence tolerance or stopping threshold is introduced after seeing the
parent result. This test describes whether the trajectories appear to approach
a plateau and whether shape or centroid degrades; it cannot by itself select a
production stopping rule.

## Scientific boundary

As in the parent test, the source enters after RTC processing, despiking,
calibration, initial learned masks, and detector selection. It tests only the
PTC-cleaning/FRUIT/mapmaking recurrence for one bright, positive, centered
compact source co-located with the real pointing source. It supplies no claim
about pre-RTC losses, off-center response, dynamic range, extended angular
scales, faint emission, atmosphere leakage, historical superiority,
qualification, or production readiness.

## Result

Both continuous branches completed absolute iterations 1--6. Their iterations
1 and 2 are bitwise equal to the corresponding parent branches in `signal_I`,
`kernel_I`, and `weight_I` for all arrays, so the extension reproduces the
parent trajectory.

Kernel-normalized central recovery at iteration 6 is `97.19%`, `97.87%`, and
`96.40%` for `a1100`, `a1400`, and `a2000`. The full-kernel projections are
`97.55%`, `98.23%`, and `96.83%`. From iteration 5 to 6, the central estimates
increase by only `0.12`, `0.16`, and `0.35` percentage points. Fitted major
and minor axes are within 1.8% of the same-iteration kernel and centroid
separations are below 0.09 arcsec. This is a descriptive late-iteration
plateau for one bright centered source, not an approved convergence threshold
or stopping rule. The complete transfer still contains non-kernel structure,
especially in `a2000`.

The timing replay described in
[`EXECUTION_TIMING_REPLAY_DEFINITION.md`](EXECUTION_TIMING_REPLAY_DEFINITION.md)
then exposed a causal checkpoint defect. A restart from the completed
iteration-4 checkpoint reproduces iteration 5 bit-for-bit but not iteration 6:
`a1100` signal, kernel, and weight differ, with signal relative RMS `0.121206`.
The uninterrupted iteration-5 checkpoint contains a newly learned scan-local
detector exclusion for UID 1489; the replay checkpoint does not. Prior
map-pixel-outlier records drive targeted contributor tracing but are not
restored from the checkpoint. Exact restart is therefore unavailable for this
enabled learning path, and the development result cannot advance to
qualification or establish restart-safe stopping behavior.
