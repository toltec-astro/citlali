# Science Raw-I/Q Event-Vector Analysis

Date: 2026-07-30

## Question

What kind of readout change produces the late-night, network-selective events
seen in the NGC4449 data, and do the events occur at a repeatable PPS or
digital-sample boundary?

This analysis does not use target sweeps to judge tune quality or to argue
that tuning caused the events. It uses each observation's processed target
sweep only to measure the local complex derivative
`d(I + iQ) / df` at each raw tone slot.

## Inputs and Selection

The analysis covers science observations 152390, 152392, 152419, 152431, and
152433. Networks 1, 2, 3, 4, 8, and 9 are the previously identified affected
set; networks 0, 5, 7, 11, and 12 are controls.

Persisted RTC diagnostics select at most 18 candidate ten-second chunks per
observation. Raw `I + iQ` then independently supplies an operational event
time: the epoch giving the largest coherent pre/post phase change across
APT-usable tones. Candidates within 0.35 s are clustered when at least three
affected networks participate. Every available network is evaluated at the
shared cluster time, including networks that did not trigger the cluster.

The fit population is the set of APT-usable tones with a phase change above
both eight times the robust per-tone phase noise and 5 mrad. A network fit is
reported only when at least eight tones respond and at least 10% of its usable
tones move coherently with the same phase sign.

## Chronology and Network Partition

The raw-I/Q result independently recovers the abrupt night transition:

- No qualifying cross-network raw event is found in the selected chunks of
  152390 or 152392.
- There are 13 event clusters in 152419, 20 in 152431, and 19 in 152433.
- All 52 late event clusters cross the RACKA/RACKO boundary.
- Each cluster contains three to six affected networks. Forty-eight of 52
  contain no control network. Network 12 is the only control that participates,
  and it does so in four clusters; networks 0, 5, 7, and 11 never qualify.
- The median span between the participating networks' independently estimated
  event times is 81 ms; 90% span no more than 162 ms.

These counts describe an RTC-guided sample, not an unbiased event rate. The
result nevertheless confirms that the late-night disturbance is simultaneous
to the available raw-I/Q time resolution, crosses the two electronics racks,
and preserves the same affected/control partition seen in the maps.

## Complex-Vector Result

For each responsive network/event, the measured fractional complex change is
fit with real coefficients against:

1. common gain, `delta_z / z = a`;
2. common phase rotation, `delta_z / z = i phi`;
3. frequency-like motion,
   `delta_z / z = delta_f (d z / d f) / z`;
4. arbitrary common complex motion; and
5. the combined gain, phase, and frequency-like model.

The reported R-squared is zero-baseline explained complex-change energy. It is
a descriptive model score, not a statistical-significance measure.

There are 292 accepted affected-network fits. Median model support by network
is:

| Network | Gain R2 | Phase R2 | Frequency-like R2 | Combined R2 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.091 | 0.331 | 0.349 | 0.446 |
| 2 | 0.077 | 0.329 | 0.487 | 0.603 |
| 3 | 0.213 | 0.153 | 0.231 | 0.408 |
| 4 | 0.193 | 0.145 | 0.335 | 0.415 |
| 8 | 0.025 | 0.638 | 0.625 | 0.765 |
| 9 | 0.023 | 0.128 | 0.145 | 0.285 |

The event is therefore not a single common gain change or a single common
phase rotation:

- Network 2 has the clearest frequency-like preference.
- Network 8 has a very coherent complex response; common phase and
  frequency-like motion both describe much of it, and their combination is
  substantially better.
- Networks 3 and 4 contain appreciable gain-like motion, but frequency-like
  structure remains important, especially in network 4.
- Networks 1 and 9 are mixed and less completely described.
- Controls 0, 5, 7, and 11 never have enough coherent response to fit.
  Network 12 qualifies only four times and has a median combined R2 of 0.205.

The combined fit estimates median frequency-like coefficients near -2.9 kHz
for network 2 and +4.8 kHz for network 8, but these are local linearized
equivalent shifts, not direct resonance-frequency measurements. Their broad
event distributions and the sensitivity of network 3 to the sweep-fit
neighborhood preclude treating every coefficient as a calibrated physical
frequency shift.

Repeating all 292 fits with local sweep half-widths of two and five steps
preserves the main result. Best-single-mode labels agree for 89% of fits
between the narrow and nominal estimates and 85% between the nominal and wide
estimates. The exact network-3 frequency coefficient is not stable, so only
the broader mixed-mode conclusion is retained.

## Timing Result

There is no robust evidence that the events lock to PPS or a tested digital
sample boundary:

- UTC event time modulo one second has circular resultant length 0.071 and an
  approximate Rayleigh p-value of 0.772.
- Sample-index moduli 8, 16, 32, 128, and 256 are not concentrated.
- Modulus 64 alone gives resultant length 0.268 and nominal p = 0.023.
  It is one of six exploratory modulus tests and does not survive even the
  simple six-test Bonferroni threshold.

The operational event times are maxima of a finite pre/post comparison, not
exact physical onset times. PPS telemetry and hardware block-boundary metadata
are unavailable, so this result rules out only a strong alignment at the
tested boundaries.

## Physical Interpretation

The best current model is a shared disturbance or shared enabling condition
whose expression depends on each network's analog/readout operating point.
That model naturally permits:

- synchronization across the two racks;
- a stable subset of susceptible networks;
- only a fraction of tones responding within each network; and
- increasing severity as an unmeasured thermal or electrical susceptibility
  changes through the night.

A purely downstream common gain or phase jump is disfavored because the
per-tone response contains repeatable information aligned with the measured
resonance slope, and because each network expresses a different mixture and
sign. A common LNA-bias-supply perturbation remains possible, but it would
need a network-dependent transfer function and cannot presently be tested
from telemetry. An observatory clock/PPS trigger is not supported by the event
timing. Recorded array thermometry does not show a corresponding spike.

The analysis does not identify a unique hardware component. It narrows the
failure from "Citlali step pathology" to a real, synchronized raw-complex
disturbance upstream of Citlali calibration, with strong network-dependent
susceptibility.

## Follow-up Discriminator

The bounded tone-susceptibility analysis is complete and is documented in
`handoff/SCIENCE_IQ_TONE_SUSCEPTIBILITY_ANALYSIS_2026-07-30.md`.

It finds highly repeatable UID susceptibility and strong banding versus the
signed digital tone offset from each network LO. A pure delay/sample-slip
model is weak, while an event-amplitude times stable tone-transfer model is
especially strong in network 8 and substantial in networks 1 through 4.
The next bounded discriminator is therefore to map the digital tone offset to
the authoritative ROACH/PFB and analog signal-path coordinates.

For a future instrument test, the decisive missing telemetry is time-resolved
LNA-bias voltage/current, IF power or gain state, and ADC statistics recorded
through an event.

## Outputs

The complete artifact set is stored locally at:

`/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-event-vector-20260730`

It includes the scan selection, every raw network candidate, clustered event
times, network-level fits, tone-level complex vectors, timing tests, two
figures, a manifest carrying thresholds and input identities, and two
sweep-window sensitivity audit tables. The schema-v2 tone table retains every
model-valid tone at every event, including nonresponders, and records APT UID,
observation-local tone slot, LO center, signed digital tone offset, and probe
frequency.

## Reproduction

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/citlali-event-vector-mpl \
XDG_CACHE_HOME=/private/tmp/citlali-event-vector-cache \
$HOME/tolteca/bin/python \
  tools/diagnostics/science_iq_event_vector_analysis.py \
  --data-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data \
  --reduction-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/NGC4449-network-seed-20260729/reduced_network_seed/redu00 \
  --apt-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/apts \
  --obsnums 152390 152392 152419 152431 152433 \
  --max-scans-per-observation 18 \
  --null-scans-per-observation 2 \
  --output-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-event-vector-20260730
```
