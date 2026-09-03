# EL-F8 R0.2 Partial Execution

## Status

R0.2 stopped during the second ordered trajectory and has no scientific
interpretation.  `c5-current` completed successfully, but `a5-current`
stopped before iteration 5 because the inherited injected-source overlay
retained `start_iteration: 1`.  An exact restart at iteration 5 requires the
already-established restart convention `start_iteration: 5`.

The two map-placement trajectories were not started and the registered
analysis was not run.  The completed control output is retained but will not
be reused by a later registration.

## Completed control trajectory

- trajectory: `c5-current`
- exit status: `0`
- absolute output iteration: `5` (local directory `redu00`)
- wall time: `30.98 s`
- maximum resident set size: `858,406,912 bytes`
- log size: `484,755 bytes`
- log SHA-256:
  `8805b52eb0dd3d8fc7e3f551f57395c5d40b2d465f89b3da883aebfbe426c876`
- retained product size: `58,392 KiB`
- unexpected error-level messages: none

## Aborted injected trajectory

- trajectory: `a5-current`
- exit status: `1`
- wall time: `0.55 s`
- maximum resident set size: `46,383,104 bytes`
- log size: `14,438 bytes`
- log SHA-256:
  `c1cafa209a4e4aa8f0f45784124ab5adbdde74600a634975520d4aeee1699302`
- failure: `fruit-loop injected-source restart requires start_iteration 1
  to equal checkpoint next_iteration 5`
- scientific products: none; only the output-root lock was created

## Bounded registration correction

The next registration may restate the injected source in both injected
trajectory overlays with `enabled: true`, `start_iteration: 5`, and the
unchanged `[100, 100, 100] mJy/beam` amplitudes.  This is the same restart
convention used by the validated EL-F6 and EL-F7 iteration-4-to-5 replays.  It
does not alter the incoming A4 feedback state, source location, amplitude,
recurrence, penalty intervention, measurements, order, or claim limits.
