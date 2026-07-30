# Science Raw-I/Q Continuous Event Catalog and Temporal Morphology

Date: 2026-07-30

## Question and Bounded Verdict

What happens when the previously learned raw-I/Q event mode is searched over
the complete duration of every identity-safe observation, and what is the
time-domain shape of the disturbance?

The full-duration result changes the physical description:

> The late-night pathology is a frequent, synchronized, bidirectional,
> telegraph-like transition in a stable raw-I/Q projection. It is not well
> described as a sparse population of independent impulses followed by simple
> recovery.

The strongest observations contain 15.8 to 16.0 primary cross-rack
transitions per minute. Participating networks agree on the direction of each
transition, while successive event directions alternate 77% to 85% of the
time. A direct projected-phase trace visibly switches between broad levels.
The transition itself takes roughly 0.1 to 0.18 s, but much of the displaced
level remains after five seconds when no intervening catalog event censors the
measurement.

This is a raw-I/Q result upstream of Citlali calibration and mapmaking. It
does not identify the triggering hardware component.

## Detector and Event Definitions

Each affected network is projected onto the previously measured UID rank-1
loading. Detector identity comes only from the exact observation-specific
matched APT:

- event-rich science observations 152419, 152431, and 152433 use a
  leave-one-observation-out template trained on the other two observations;
- all other observations use the fixed all-event science UID loading;
- no observation-local event time is used to fit or select the template; and
- the 52 prior RTC-guided events are used only after detection to measure
  recall.

The projected phase is an operational coordinate in radians per
RMS-normalized UID loading. It is not calibrated detector phase.

A symmetric step filter compares 0.20 s pre- and post-event windows separated
by a 0.05 s guard. Single-network candidates require:

- absolute robust step score at least 8;
- score prominence at least 3; and
- separation of at least 0.60 s within a network.

Candidates are clustered within 0.25 s, consuming a 0.50 s neighborhood. A
primary event requires at least three affected networks and both electronics
racks. Tier A requires five or six affected networks; tier B requires three
or four.

The coincidence null circularly shifts each network's candidate times
independently. One hundred shifts preserve each network's candidate count,
score distribution, and within-network timing while removing true
cross-network alignment.

## Coverage and Identity Gate

The raw archive contains 18 observations. Thirteen have exact matched APTs
and are analyzed. All five 20-minute science observations are included.

Pointings 152398, 152403, 152408, 152413, and 152425 lack exact matched APTs
and are explicitly excluded. Their tone combs are not identical to an
APT-bearing observation, so borrowing another APT would compromise the stable
UID identity used by the detector. The observation inventory preserves these
exclusions and their raw-file provenance.

This leaves an important chronology limitation: the first cataloged
unanimous-sign events occur in pointing 152418, but the missing pointings
prevent locating the exact onset between 152393 and 152418.

## Full-Duration Chronology

The five science observations give:

| Obsnum | Primary | Tier A | Primary / min | Shift-null primary p95 | Tier-A null p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 152390 | 3 | 0 | 0.145 | 0.00 | 0.00 |
| 152392 | 1 | 0 | 0.048 | 0.00 | 0.00 |
| 152419 | 83 | 58 | 4.010 | 5.00 | 0.00 |
| 152431 | 334 | 181 | 16.031 | 80.00 | 2.00 |
| 152433 | 328 | 165 | 15.836 | 89.05 | 2.05 |

Across the 13 analyzed observations there are 787 primary events and 426
tier-A events. The five science scans contain 749 and 404, respectively.

The dense-observation primary counts greatly exceed the shift null. The
tier-A comparison is particularly clean: 58, 181, and 165 observed events
against null 95th percentiles of 0, 2, and 2.05.

The continuous catalog recovers 49 of the 52 prior RTC-guided events within
0.50 s without using those times for detection:

- 12 of 13 in 152419;
- 20 of 20 in 152431; and
- 17 of 19 in 152433.

The earlier RTC result therefore sampled a real population but severely
understated its size. Its selected chunks contained 52 events; the complete
durations contain 745 primary events in the same three science observations.

The transition develops within the night:

- 152390 and 152392 contain only four weak primary coincidences total, none
  with unanimous participating-network sign;
- pointing 152418 contains three unanimous-sign events;
- 152419 contains few events in its first eight minutes and then becomes
  active;
- 152431 and 152433 are active nearly throughout; and
- the intervening and following pointings retain high rates.

This is progressive susceptibility with observation-to-observation
modulation, not a strictly monotonic ramp.

## Temporal Morphology

### Cross-network timing

After refining each network candidate to the maximum projected-phase
derivative:

- the median participating-network onset span is 0.131 s;
- the 16th to 84th percentile span is 0.064 to 0.271 s; and
- the 95th percentile span is 0.411 s.

Median network lags from the event center are between -2 ms and +35 ms.
Networks 1 and 4 have the broadest lag distributions. These values support a
shared transition but do not establish a propagation order.

### Step and persistent level

Median 10% to 90% rise times and normalized levels are:

| Network | Rise (s) | Level at 1 s | Level at 3 s | Level at 5 s |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.16 | 0.911 | 0.752 | 0.667 |
| 2 | 0.16 | 0.959 | 0.844 | 0.790 |
| 3 | 0.16 | 1.010 | 0.888 | 0.831 |
| 4 | 0.10 | 0.683 | 0.539 | 0.544 |
| 8 | 0.18 | 0.961 | 0.808 | 0.691 |
| 9 | 0.16 | 0.981 | 0.838 | 0.715 |

The normalization is the signed immediate post-minus-pre step. Neighboring
events censor each waveform before the next transition. The five-second
levels therefore describe events with enough isolated post-event coverage;
they do not assume that an unobserved second transition is recovery.

Exponential fits with descriptive median time constants of roughly 2 to 7 s
often fit part of the post-transition drift, but they are not the primary
physical model. Some fits reach the 30 s bound or imply nonzero asymptotes.
The direct level fractions are the more robust result.

### Bidirectional state switching

Every primary event in science observations 152419, 152431, and 152433 has
unanimous projected sign across its participating networks. The event
direction itself is nearly balanced between positive and negative. Successive
directions alternate much more often than a balanced independent-sign
sequence:

| Obsnum | Events | Positive fraction | Adjacent alternation | IID-sign expectation |
| --- | ---: | ---: | ---: | ---: |
| 152419 | 83 | 0.494 | 0.854 | 0.500 |
| 152431 | 334 | 0.581 | 0.769 | 0.487 |
| 152433 | 328 | 0.524 | 0.801 | 0.499 |

The median interval between primary events falls from 6.75 s in 152419 to
2.61 s in 152431 and 2.40 s in 152433. Thus the typical recurrence in the
worst observations is shorter than the isolated five-second persistence.

The automatically selected direct example is network 8 in observation
152431. Its densest 30 s interval contains 17 primary events. The linearly
detrended projected raw phase visibly occupies broad high and low levels,
with cataloged positive and negative transitions moving between them.

The independent-sign expectation is only a descriptive reference. A
step-derivative detector acting on a bounded correlated signal can favor
alternation. The direct raw-projection trace, persistent levels, cross-network
sign unanimity, and repeated transfer across observations are the stronger
evidence for the telegraph-like description.

## Citlali Consequence

The enabled RTC step mask changes affected-network maps but does not recover
networks 1 through 4 because the failure is not confined to a narrow
transition sample:

1. a sharp synchronized transition occurs;
2. the projected raw state remains displaced and drifts;
3. another, usually opposite transition often arrives within a few seconds;
4. the cycle repeats hundreds of times in a science observation.

Masking only the transition edge leaves substantial displaced-state data in
the timestream. Expanding every mask through the state dwell would discard a
large fraction of the worst observations. The correct production behavior is
therefore not to imply that short step masking repairs the data.

For immediate science recovery, the existing network guard remains justified:
exclude affected network/observation combinations rather than allow their
artifact-dominated maps into the coadd.

For Citlali QA, this result motivates:

- a full-duration event-rate and cross-network-coincidence diagnostic;
- a hard warning or rejection when persistent state switching exceeds a
  validated threshold;
- explicit reporting of masked transition duration versus inferred
  displaced-state duration; and
- no claim that an observation was repaired when the surviving state
  occupancy remains pathological.

This handoff does not prescribe a production threshold. The present detector
uses a science-derived template and is a forensic tool, not yet a general
online validator.

## Limitations and Next In-Can Analysis

- The catalog is complete in time for exact-APT observations but selected for
  the learned UID mode. It is not complete for orthogonal raw-I/Q
  pathologies.
- Five pointings are excluded by the detector-identity gate.
- The robust step score is an operational threshold, not a statistical
  significance map or calibrated probability.
- Time-shift counts are an empirical coincidence reference, not a complete
  colored-noise null model.
- Cross-network onset estimates inherit the 122 Hz sampling, projection
  noise, and onset-refinement window.
- Event-time telemetry from the LNA bias, IF chain, ADC interior, clocks, and
  electronics temperatures is unavailable.

The most useful next analysis using only data in hand is a bounded
change-point or hidden-state model on the projected raw traces. It should
measure state levels, dwell-time distributions, transition matrices, and
cross-network state correspondence without assuming exponential recovery.
Generating exact matched APTs for the five excluded pointings would then
close the chronology gap, provided the APT provenance and matching validation
are acceptable.

That hidden-state follow-up is now complete. See
`handoff/SCIENCE_IQ_HIDDEN_STATE_ANALYSIS_2026-07-30.md`. It finds a
baseline plus two broad cross-rack response modes, with an approximate
nw1/nw2/nw9 versus nw3/nw4/nw8 partition rather than one identical state
shared by all networks.

## Outputs

The artifact set is stored locally at:

`/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-continuous-event-morphology-20260730`

Central products are:

- `continuous_observation_inventory.csv`
- `continuous_observation_summary.csv`
- `continuous_observation_network_summary.csv`
- `continuous_single_network_candidates.csv`
- `continuous_event_catalog.csv`
- `continuous_event_network_members.csv`
- `continuous_known_event_recall.csv`
- `event_network_temporal_morphology.csv`
- `event_temporal_morphology_summary.csv`
- `event_observation_network_temporal_morphology_summary.csv`
- `event_projected_waveform_stack.csv`
- `telegraph_example_projected_phase.csv`
- `telegraph_example_events.csv`
- five diagnostic figures; and
- `manifest.json`, containing input identities, exclusions, definitions,
  parameters, output names, and row counts.

## Reproduction

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/citlali-continuous-event-mpl \
XDG_CACHE_HOME=/private/tmp/citlali-continuous-event-cache \
$HOME/tolteca/bin/python \
  tools/diagnostics/science_iq_continuous_event_morphology.py \
  --data-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data \
  --apt-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/apts \
  --event-vector-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-event-vector-20260730 \
  --tone-analysis-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-tone-susceptibility-20260730 \
  --output-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-continuous-event-morphology-20260730 \
  --time-shift-permutations 100
```
