# Pointing Raw-I/Q Event Survey — 2026-07-29

## Scope

This bounded survey tested whether the raw-I/Q phase events seen in pointing
observation 152434 were isolated or population behavior. It analyzed every
persisted Citlali scan and every available network in observations 152420,
152432, and 152434.

The diagnostic implementation is
`tools/diagnostics/pointing_iq_event_survey.py`. The analyzed data remain
external local products and are not part of this repository.

## Classifier

For each observation, one-based persisted `output_scan_index`, and explicit
network ID, the classifier:

1. joins raw tone slots to detector identity and validity through the matched
   APT `kids_tone`, `uid`, `kids_flag`, and `flag` fields;
2. estimates each APT-usable tone's phase-noise scale from first differences;
3. compares 0.20-second pre-event and post-event raw-I/Q windows separated by a
   0.05-second guard;
4. searches for the time that maximizes the fraction of usable tones with a
   same-sign phase change exceeding both eight robust sigma and 5 mrad; and
5. records that operational raw candidate independently of Citlali's RTC
   network dominant-step sample.

The candidate is not an exact physical onset time. The v1 survey retains only
the dominant raw-I/Q candidate per scan and network, so it is an incidence
classifier rather than a complete within-scan event counter.

The event-population threshold used below is a coherent same-sign fraction of
10%. A severe event is reported descriptively at 30%; neither threshold is a
Citlali masking policy.

## Results

| Observation | Network-event scan cells >=10% | Severe cells >=30% | Cross-rack event clusters |
| ---: | ---: | ---: | ---: |
| 152420 | 36 | 18 | 6 |
| 152432 | 38 | 18 | 7 |
| 152434 | 54 | 18 | 9 |

The increasing pathology is primarily an increase in recurrence or duty cycle,
not in maximum amplitude. The severe-cell count remains 18 in each observation,
while the total event-cell count grows from 36 to 54. In observation 152434,
network 3 is affected in all 12 scans; networks 2, 4, 8, and 9 are affected in
10, 9, 8, and 9 scans, respectively.

The network selection is stable:

- affected: networks 1, 2, 3, 4, 8, and 9;
- quiet controls at the 10% threshold: networks 0, 5, 7, 11, and 12.

Varying the noise multiplier from six to ten sigma preserves this affected
versus quiet partition. At ten sigma, the observation event-cell counts remain
36, 36, and 49.

The events are phase-dominant. Among event cells, the median absolute phase
shift of the strong tones is 14.8, 14.3, and 12.5 mrad in time order, while the
median absolute all-tone amplitude changes are 0.014%, 0.049%, and 0.024%.
Across the population, the median fraction of strong tones sharing the
reference phase sign is 0.968 and the median strong-change direction coherence
is 0.935.

There are 18 clusters involving at least four networks. Every one crosses the
RACKA/RACKO boundary, with a median raw-candidate span of 0.113 seconds. The
most common participating sets are subsets of networks 1–4, 8, and 9. Networks
can respond with opposite phase signs in the same cluster, consistent with a
shared trigger followed by a network-specific transfer response.

For qualifying events with an RTC candidate, the raw classifier commonly
precedes the RTC dominant-step sample. The median RTC-minus-raw offset is
0.304, 0.069, and 0.264 seconds by observation. This confirms that the RTC
sample should not be treated as the physical onset.

## Interpretation

The population evidence disfavors telescope motion and isolated detector
physics as primary explanations. It also disfavors a simple whole-rack power
disturbance: the same selected networks recur across two racks while adjacent
networks remain quiet.

The evidence is consistent with a shared physical or timing disturbance
reaching multiple readout chains whose susceptibility is network dependent.
The remaining common paths to test are:

- observatory-derived 10 MHz/PPS distribution;
- the power supply common to the LNA-bias circuits, followed by
  network-dependent LNA/detector/readout response; and
- a cryogenic operating-point change that alters network susceptibility.

The project owner clarified on 2026-07-30 that LMTMC configures TolTEC through
explicit observing-script actions and then commands only the telescope. TolTEC
is passive during an ordinary observation; state changes require an explicit
script command such as Tune or atypical manual TolTEC MC intervention.
Therefore a synchronized mid-observation control action is not a viable
ordinary-night explanation. The NetCDF headers are the available setup record.
The instrument does not collect synchronized timestream telemetry for the
suggested timing, LNA-bias, electronics-temperature, or FPGA/control-status
signals.

This survey establishes the raw-I/Q phenotype and its population structure; it
does not yet distinguish those mechanisms.

## Reproduction

```bash
$HOME/tolteca/bin/python \
  tools/diagnostics/pointing_iq_event_survey.py \
  --data-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data \
  --reduction-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/pointings-refactor-wide-psf/reduced/redu00 \
  --apt-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/apts \
  --obsnums 152420 152432 152434 \
  --output-dir /private/tmp/pointing-iq-event-survey
```

The command writes a JSON manifest with explicit semantics and provenance,
scan-network CSV, event-cluster CSV, population CSV, and summary PNG.

## Next Discriminator

Completed on 2026-07-30 using the five approximately 20-minute science
observations plus the reduced pointing chronology. See
`handoff/SCIENCE_IQ_TEMPERATURE_SURVEY_2026-07-30.md`.

The recorded array and mixing-chamber temperatures do not show a spike that
explains the event onset. The network-selective transition is sharply
bracketed by the clean 152418 pointing and affected 152419 science
observation. Warm-stage channels drift upward with time of night and remain
possible susceptibility variables, but they do not isolate the trigger. The
next discriminator is therefore a controlled affected-versus-control
comparison of the 152418 and 152419 tune/setup products.
