# Impulsive Stress-Test Findings

This note summarizes the first impulsive-focused read of the `redu40` `rtcdiag`
products for:

- `152526`
- `151928`
- `152524`

using the `a1100` slot reports generated from:

- `tools/blank_sky/rtc_impulsive_slot_report.py`

## Main Result

The dominant impulsive failure mode in these stress-test obsnums is not
"isolated detector glitches."

It is a mixed population with two clear classes:

1. Short same-sample bursts that appear across many detectors and often across
   several networks at once.
2. Repeat-offender detectors that recur across many scans, but are secondary to
   the burst class in the worst rows.

The current detector-local despike logic is missing many of these events:

- `152526`: `1095` captured events, `639` untouched by despike counters.
- `151928`: `1272` captured events, `673` untouched.
- `152524`: `938` captured events, `470` untouched.

## Evidence

### `152526`

- Strong repeated-detector component:
  - `apt_uid 1368` dominates `8` of the top `20` events.
- Strong burst/coincidence component:
  - scan `69`, samples `1179-1181`: `17` captured events across networks
    `0-5`
  - scan `105`, samples `655-658`: `17` events across networks `0-5`
  - scan `82`, samples `1036-1040`: `16` events across networks `0-5`

### `151928`

- Strong coherent burst at one exact sample:
  - scan `73`, sample `1087`: `18` captured events across all six networks
    `0-5`
- Secondary repeated-detector component:
  - `apt_uid 2738` appears `55` times
  - `apt_uid 2734` appears `44` times

### `152524`

- The most impulsive-looking top rows are concentrated in `nw1`, but they also
  sit inside broader same-sample multi-network clusters:
  - scan `12`, samples `714-716`: `15` events across networks `0-5`
  - scan `82`, samples `1214-1217`: `15` events across networks `0-5`
  - scan `36`, samples `528-530`: `15` events across networks `0-5`

## Interpretation

The next runtime change should not be another detector-local threshold tweak.

The first new generalized metric we need is network-level impulsive
coincidence, analogous to the existing step-alignment metrics:

- detector impulsive-event score fraction per network
- dominant impulsive sample per network
- aligned-fraction around that dominant sample

Those metrics are now a better fit to the observed failure mode than raw/local
candidate counts alone.

## Code Direction

The next diagnostic layer should:

1. Compute network-level impulsive coincidence summaries from the detector
   `TransientEvent` products.
2. Persist them in `rtcdiag` and RTC TOD products.
3. Use them to decide whether a future `impulsive_coincidence_mask` should flag
   short shared windows, in the same spirit as `network_step_mask`.

The actual masking action should be validated separately. The current change is
diagnostics-first.
