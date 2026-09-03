# FRUIT EL-F7 shared-start response decomposition

Status: **registered after exact owner approval and before staging or output
inspection**

Test ID: `SCI-FRUIT-EL-F7-SHARED-START-RESPONSE-DECOMPOSITION-R0.1`

## Question

What part of the EL-F5 iteration-5 total adaptive response is produced during
one transition from a common incoming state, what part is associated with
other state learned along the earlier injected history, and what part is the
already isolated UID 4460 penalty effect?

This is a measurement-clarification test on one development observation. It
is not a candidate recurrence, a safeguard test, a detector-quality decision,
or a qualification experiment.

## One new map and three existing maps

Let `C5` be the EL-F5 control iteration-5 signal map, `A5` the EL-F5 adaptive
injected iteration-5 map, and `N5` the EL-F6 iteration-5 map from the injected
checkpoint with only the carried UID 4460 penalty removed. Create one new map,
`P5`, by starting from an exact copy of the EL-F5 control iteration-4 state and
adding the 100 mJy/beam source at FITS map-world `(0, -60)` arcsec only during
iteration 5.

For every array, retain:

`T5 = A5 - C5` — total adaptive trajectory response;

`S5 = P5 - C5` — shared-incoming-state one-step response;

`H5 = N5 - P5` — other earlier injected-history state and interactions; and

`D4460,5 = A5 - N5` — the previously isolated UID 4460 effect.

Require the exact telescoping closure `T5 = S5 + H5 + D4460,5` to the
registered floating-point roundoff bound. The decomposition is an identity
among maps, not a linearity, orthogonality, independent-calibration, or
intervention-order claim.

## Execution order

1. Verify and copy the complete EL-F5 no-injection `redu04` directory twice.
2. Restart the first copy with injection disabled and advance once.
3. Require all nine signal/kernel/weight planes to equal the existing EL-F5
   control iteration 5 bitwise and every checkpoint variable to be
   value-identical.
4. Only after that gate passes, restart the second copy with injection enabled
   from absolute iteration 5 and advance once.
5. Apply the frozen descriptive decomposition and stop.

Both trajectories use the exact preserved EL-F5 executable, common first six
configuration files, one thread, and `--grppiex seq`. The only paired
effective-configuration differences allowed are output/restart paths and the
declared injection enabled/start/amplitude/position fields.

## Measurement

For all four component maps and all three arrays, record common support,
complete-map and declared-region RMS, fixed-kernel projection and residual,
20-arcsec injected-source and Neptune-region RMS, the 40--120 arcsec annulus
after excluding 25 arcsec around Neptune, and all pairwise component inner
products and cosines. Fit compact-source amplitude, aperture-integrated
response, centroid, and width for `T5` and `S5` relative to their respective
processed kernels. Record all checkpoint variables that differ between `C5`
and `P5` and whether UID 4460 is learned at the end of `P5`.

The report is descriptive. It applies no unregistered dominance threshold and
must retain mixed and cancelling terms. `S5` remains a shared-incoming-state
one-step response, not a fully matched-operator transfer function.

## Bounds

- exactly two sequential one-iteration local replays, sham first;
- no rebuild and no production-code change;
- at most one replacement per replay only for interruption or environmental
  failure;
- 1 hour and 64 GiB per replay, 3 hours and 8 GiB retained in aggregate; and
- no extra variant, iteration, tuning choice, threshold, or algorithm change.

The complete scientific and authorization limits remain those in the exact
approved owner-review bundle.
