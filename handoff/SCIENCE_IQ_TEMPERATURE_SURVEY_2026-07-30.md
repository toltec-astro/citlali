# Science I/Q and Cryostat-Temperature Survey — 2026-07-30

## Scope

This survey uses the complete locally available NGC4449 raw-data directory to
test two questions:

1. Does the network-selective phase-event population seen in the pointing
   observations persist through the five approximately 20-minute science
   observations?
2. Does its onset or incidence track any recorded cryostat temperature?

The science observations are 152390, 152392, 152419, 152431, and 152433. The
night chronology also includes the eight reduced pointings with matched
diagnostics: 152389, 152391, 152393, 152418, 152420, 152430, 152432, and
152434.

The affected-network set is `[1, 2, 3, 4, 8, 9]`; the control set is
`[0, 5, 7, 11, 12]`. This grouping was fixed from the preceding pointing-map
and raw-I/Q evidence, not selected from the temperature results.

The implementation is
`tools/diagnostics/science_iq_temperature_survey.py`. The generated products
are stored outside the repository under
`docs/science-iq-temperature-survey-20260730` in the local project data tree.

## Measurement Semantics

The primary population measurement is Citlali's persisted
`rtc_network_step_det_frac` for every network and every approximately
10-second science chunk. It is a fraction of detectors satisfying the RTC
step classifier, not a probability or statistical significance.

Housekeeping samples are retained at their measured epochs. Science chunks
are joined to the nearest measured MC-plate (`T8`) epoch only when it lies
within 35 seconds. The roughly 60-second housekeeping cadence is sufficient
for slow susceptibility tests, not subsecond trigger timing.

As an independent check, the survey reads raw `I + iQ` for the highest-RTC and
lowest-RTC interior science chunk in each observation and network. It finds
the maximum fraction of matched-APT usable tones with a same-sign phase change
larger than both eight robust sigma and 5 mrad. This operational classifier is
independent of Citlali's x/r conversion and RTC step statistic.

## Dominant Result: An Abrupt Transition

The network pathology is not merely a gradual worsening across the five
science observations. It is sharply bracketed by pointing 152418 and science
observation 152419:

| Observation | Type | Affected median fraction | Control median fraction | Contrast |
| ---: | :--- | ---: | ---: | ---: |
| 152393 | pointing | 0.000 | 0.007 | -0.007 |
| 152418 | pointing | 0.016 | 0.009 | +0.007 |
| 152419 | science | 0.223 | 0.046 | +0.177 |
| 152420 | pointing | 0.198 | 0.018 | +0.181 |
| 152430 | pointing | 0.184 | 0.027 | +0.157 |
| 152431 | science | 0.456 | 0.079 | +0.377 |
| 152432 | pointing | 0.212 | 0.017 | +0.196 |
| 152433 | science | 0.489 | 0.078 | +0.411 |
| 152434 | pointing | 0.267 | 0.014 | +0.253 |

Pointing 152418 ends approximately 8.8 minutes before science data begin for
152419. Its final three scan medians are only 0.0015, 0.0014, and 0.0049 in
the affected networks. In contrast, the first three 152419 science chunks
have affected-network medians of 0.370, 0.228, and 0.202. The disturbance is
therefore already active at the start of 152419 and persists through that
20-minute observation.

Pointing and science chunks have different durations, so their absolute
fractions are not perfectly interchangeable. Three safeguards make the
transition robust: early pointings and early science observations are both
quiet, affected-minus-control contrast is used, and selected science chunks
are checked directly in raw I/Q.

## Independent Raw-I/Q Check

Across 110 stratified science chunks, the raw coherent phase fraction and RTC
step-detector fraction have a Spearman rank correlation of 0.629. More
importantly, the population partition is preserved:

- In the two early science observations, none of the selected raw chunks
  crosses the 10% coherent-phase threshold.
- In observations 152419, 152431, and 152433, all 18 selected event-rich
  affected-network chunks cross 10%; their median raw coherent fraction is
  0.398.
- Thirteen of the 18 nominally lowest-RTC affected-network chunks in those
  late observations also cross 10%, with a median raw coherent fraction of
  0.202. "Lowest RTC" is therefore not a genuinely clean late baseline for
  many affected networks.
- None of the 15 lowest-RTC late control-network chunks crosses 10%; their
  median raw coherent fraction is 0.0047.

The raw evidence confirms that the late science data contain a pervasive
network-selective phase disturbance before Citlali's calibrated x/r
conversion, filtering, or mapmaking. The imperfect one-to-one relation is
expected because the raw operational classifier and RTC statistic use
different representations and event definitions.

## Thermometry Result

There is no recorded array-temperature spike that explains the transition.
Between the median housekeeping states of observations 152418 and 152419:

- MC plate (`T8`): **-0.070 mK**
- MC bar (`T12`): **+0.052 mK**
- 1.1-mm top (`Temperature5`): **+0.210 mK**
- 2.0-mm foot (`Temperature6`): **+0.715 mK**

Those are small changes, and the MC-plate change has the wrong sign for a
simple warming-spike account.

Two warmer channels do rise across the transition:

- 4K busbar (`Temperature4`): **+8.84 mK**
- LS front (`Temperature13`): **+3.82 mK**

Both then continue a smooth night-long rise. Across the full chronology they
track time of night and event severity, so they remain plausible
susceptibility variables or proxies for an unmeasured thermal state. They do
not identify a discrete trigger. Within individual 20-minute observations,
the strongest descriptive temperature/event correlations change channel and
sign; the autocorrelated minute samples do not provide a consistent thermal
response law.

The association CSV retains Spearman and multiple-testing values for audit,
but they are explicitly descriptive. Treating the 21 correlated samples
within each observation as independent trials would overstate the statistical
evidence.

## Interpretation

The new evidence changes the priority of the failure hypotheses:

1. **A setup- or operating-point transition between 152418 and 152419 is now
   the primary discriminator.** Each observation has a target sweep/tune, and
   152419 begins in the bad state immediately after its setup. Compare the
   152418 and 152419 tune products in affected networks against the stable
   controls.
2. **A thermally changing susceptibility remains plausible, especially through
   a warm-stage or unmeasured electronics temperature.** The recorded array
   and mixing-chamber temperatures do not supply the trigger. The monotonic
   4K-busbar and LS-front drift cannot presently be separated from time of
   night.
3. **A one-off bad tune is insufficient by itself.** The same network
   partition persists through later retunes. A tune-dependent response to a
   changed thermal/readout state is more consistent with the chronology than
   an isolated fit accident.
4. **Citlali is measuring an existing raw-readout phenotype.** It can change
   the map expression of the events, but it is not creating the underlying
   late-night phase disturbances.

## Follow-up Analysis

The next analysis followed the raw-I/Q event physics rather than interpreting
target-sweep quality. The target sweeps were used only as empirical
`d(I + iQ) / df` measurements at the matching raw tone slots. The resulting
complex-vector and timing analysis is recorded in
`handoff/SCIENCE_IQ_EVENT_VECTOR_ANALYSIS_2026-07-30.md`.

It confirms synchronized cross-rack events in the late observations, finds a
mixed common-complex and resonance-slope-aligned response that differs by
network, and finds no robust PPS or tested sample-boundary alignment. This
supersedes the tune/setup comparison as the immediate diagnostic priority.

## Reproduction

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/citlali-science-survey-mpl \
XDG_CACHE_HOME=/private/tmp/citlali-science-survey-cache \
$HOME/tolteca/bin/python \
  tools/diagnostics/science_iq_temperature_survey.py \
  --data-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data \
  --reduction-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/NGC4449-network-seed-20260729/reduced_network_seed/redu00 \
  --apt-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/apts \
  --obsnums 152390 152392 152419 152431 152433 \
  --pointing-reduction-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/pointings/hero_trial_v4/redu00 \
  --pointing-obsnums \
  152389 152391 152393 152418 152420 152430 152432 152434 \
  --raw-validation-per-class 1 \
  --output-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-temperature-survey-20260730
```
