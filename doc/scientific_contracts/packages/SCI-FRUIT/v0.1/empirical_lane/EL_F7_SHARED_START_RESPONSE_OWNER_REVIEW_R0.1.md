# SCI-FRUIT v0.1 — EL-F7 Shared-Start Response Review r0.1

Decision candidate:
`SCI-FRUIT-EL-F7-SHARED-START-RESPONSE-DECOMPOSITION-R0.1`

Status: **owner-review proposal; no setup, replay, analysis implementation, or
execution is authorized**

## The short version

EL-F5 compared an injected FRUIT history with an uninjected FRUIT history.
That is a useful end-to-end test, but by iteration 4 the two histories no
longer have the same internal state. EL-F6 proved that one of those state
differences, the UID 4460 detector penalty, caused most of the new arc-shaped
a1400 structure at iteration 5.

The proposed EL-F7 test asks a simpler missing question:

> What happens if both iteration-5 reductions start from the exact same
> uninjected iteration-4 state, and the off-source test signal is added to
> only one of them for that single iteration?

This requires two short one-iteration restarts: one no-injection sham and one
100 mJy/beam off-source probe. Together with the EL-F5 and EL-F6 products we
already have, that one new probe map completes an exact map-level
decomposition of the observed iteration-5 response.

## Why this comes before a proposed fix

The read-only checkpoint comparison found that the EL-F5 control and injected
iteration-4 states differ in all of the following:

- more than 304,000 values in each stored feedback signal/kernel state;
- 2,973 accumulated atmospheric-correlation entries;
- 2,372 effective detector-weight penalties;
- one effective sample-mask interval;
- three of 24 learned target-pixel coordinates; and
- one additional effective detector penalty, the factor-zero a1400 UID 4460
  record isolated by EL-F6.

The relevant arrays have the same shapes except for the expected extra mask
and penalty rows, but the injected history is not simply “the control plus UID
4460.” Testing a new penalty rule now would mix its effect with those other
history differences. EL-F7 first separates the parts that can be separated
with the existing evidence and one common-start probe.

## Exact states and maps

Use the frozen EL-F5 executable and observation 123424. Let:

- `C4` be the complete EL-F5 no-injection checkpoint after iteration 4;
- `C5` be the existing EL-F5 no-injection iteration-5 signal map;
- `A5` be the existing EL-F5 adaptive injected iteration-5 signal map;
- `N5` be the existing EL-F6 iteration-5 signal map made from the injected
  iteration-4 checkpoint after removing only the carried UID 4460 penalty;
  and
- `P5` be the one new map made by starting from an exact copy of `C4`, adding
  the registered off-source source only during iteration 5, and advancing once.

The injected source is unchanged from EL-F5: 100 mJy/beam in each array at
FITS map-world `(AZOFFSET, ELOFFSET) = (0, -60)` arcsec.

The maps define:

\[
T_5 = A5-C5
\]

for the existing total adaptive trajectory response,

\[
S_5 = P5-C5
\]

for the new shared-incoming-state one-step response,

\[
H_5 = N5-P5
\]

for the contribution of all earlier injected-history state other than the
removed UID 4460 record, including its downstream interactions, and

\[
D_{4460,5} = A5-N5
\]

for the exact UID 4460 intervention effect already established by EL-F6.

These maps obey the telescoping identity

\[
T_5 = S_5 + H_5 + D_{4460,5}.
\]

The identity is bookkeeping across observed maps; it does not assert that
FRUIT is linear, that the components are orthogonal, or that they are
independently calibrated sky products.

## Runs and exact controls

Both replays copy the complete `C4` reduction directory into a new isolated
development root. The original EL-F5 products remain read-only.

1. **No-injection sham.** Restart from the copied `C4`, leave injection
   disabled, and advance once. Its signal, kernel, weight, and complete
   checkpoint must reproduce existing `C5` exactly. This is not a second
   scientific control; it proves that the copied common starting state and
   restart route are exact.
2. **Shared-start probe.** Restart from a second exact copy of `C4`, enable the
   off-source injection beginning at absolute iteration 5, and advance once.
   No other scientific or learned-state input is edited.

Run the sham first. If it fails, stop before opening or interpreting the probe
output. No fresh iteration-0 trajectory, centered injection, blank field,
Neptune subtraction, or additional iteration is part of this test.

## Validity gates

1. Both restart-source copies and checkpoints must recursively match the
   frozen `C4` source before execution.
2. The sham must reproduce all a1100, a1400, and a2000 signal, kernel, and
   weight planes bitwise and every checkpoint variable value-identically to
   existing `C5`.
3. The two effective configuration stacks may differ only in output/restart
   paths and the declared injection enabled/start/amplitude/position fields.
4. All required products must be finite and complete, with zero unexpected
   error- or critical-level messages.
5. `C5`, `A5`, `N5`, and `P5` must have identical array identity, units, WCS,
   grid, normalization, and declared common finite support before subtraction.
6. The four-component identity must close elementwise to a prospectively
   recorded floating-point roundoff bound on every common-support pixel. A
   failure invalidates the decomposition rather than being absorbed by a
   looser post hoc tolerance.

## Measurements

Retain `T5`, `S5`, `H5`, and `D4460,5` for all three arrays. For each component
report:

- complete-map and declared-region RMS;
- best-fit amplitude, integrated response, centroid, and major/minor width
  relative to the processed injected-source kernel where that fit is meaningful;
- best-kernel residual map and relative RMS;
- RMS inside 20 arcsec of the injected source;
- RMS inside 20 arcsec of the fitted real Neptune position;
- RMS in the 40--120 arcsec injection-centered annulus after excluding the
  25-arcsec Neptune neighborhood; and
- component inner products and cross terms in those regions, so the report
  does not imply that squared RMS values add independently.

Record all checkpoint differences newly produced during iteration 5. Report
whether the shared-start probe itself learns UID 4460 again, but do not treat
an end-of-iteration record as causal for the already completed `P5` map.

This is a descriptive decomposition, not a winner/loser screen. No unregistered
dominance threshold will be applied. The report must say where the non-kernel
structure appears among `S5`, `H5`, and `D4460,5`, including mixed or
cancelling contributions.

## Scientific interpretation limits

`S5` will be the cleanest source-response measurement available in the present
data because it removes all incoming-state differences. It is still only a
shared-incoming-state one-step response. The injected data can change
data-dependent processing during iteration 5, so EL-F7 cannot call `S5` a
fully matched-operator transfer function without separately demonstrating
that equality.

`H5` combines every incoming-state difference other than the removed UID 4460
record. It cannot identify those state variables one by one. `D4460,5` remains
conditional on the injected iteration-4 state used by EL-F6. The exact sum is
a decomposition of maps, not a claim that the causal effects commute under
other intervention orders.

A clean `S5` would support proceeding to a prospectively defined, narrow
penalty-safeguard experiment. A structured `S5` would show that current-step
source-dependent processing also needs investigation first. Either outcome is
informative, but neither selects a recurrence, safeguard, or production rule.

## Bounds and stop rules

- new output root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f7-shared-start-response-r0.1`;
- exactly two sequential one-iteration local replays, sham first;
- the preserved EL-F5 executable with SHA-256
  `6431c6653ed46ff6e1dfa5512cd27e8169525f7a110207b0b24505786f39dbbe`;
- one configured thread and `--grppiex seq`;
- at most one replacement per replay for an environmental or interrupted
  start, never for an unfavorable scientific result;
- 1 hour and 64 GiB per replay, 3 hours and 8 GiB retained in aggregate; and
- stop after the registered decomposition without adding variants, thresholds,
  iterations, tuning, or an algorithm change.

The expected active reduction time is minutes; the larger hard bounds exist
only to stop an abnormal run safely.

## Owner choices

### Choice A — Approve the bounded shared-start decomposition (recommended)

Approve
`SCI-FRUIT-EL-F7-SHARED-START-RESPONSE-DECOMPOSITION-R0.1` exactly as bound by
its manifest. This authorizes staging the frozen inputs, creating only the
restart overlays and bounded analysis support needed for this test, executing
the two local one-iteration replays, and recording the result.

### Choice B — Adopt only the measurement language

Accept the response-measurement distinctions for future proposals, but do not
stage or run EL-F7.

### Choice C — Revise the test

Return a new proposal with different states, source parameters, metrics,
bounds, or sequencing. No work in Choice A is authorized.

Silence, a general request to continue, or approval of the surrounding
analysis is not approval of Choice A. The exact decision identifier and
manifest must be approved.
