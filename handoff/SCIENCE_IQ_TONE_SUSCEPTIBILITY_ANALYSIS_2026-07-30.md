# Science I/Q Tone-Susceptibility Analysis

Date: 2026-07-30

## Question

Do the late-night raw-I/Q events repeatedly affect the same detectors, follow
a fixed readout-frequency band, or reduce to a simple common phase/delay
change?

This is a follow-up to
`handoff/SCIENCE_IQ_EVENT_VECTOR_ANALYSIS_2026-07-30.md`. It uses all
model-valid tones at each of the 52 previously identified events in science
observations 152419, 152431, and 152433. The affected networks are 1, 2, 3, 4,
8, and 9.

Detector identity is APT `uid`. Tone slot is retained only as an
observation-local readout coordinate. The frequency coordinate used for
banding is the signed digital tone frequency relative to each network's LO
center, not absolute sky/probe frequency.

The phase-response flag is the event-vector analysis threshold: a phase
change exceeding both eight times the robust per-tone noise estimate and
5 mrad. All nonresponding but model-valid tones remain in the denominator.

## Result 1: Susceptibility Is Stable by Detector/Tone Identity

The same UIDs respond repeatedly. Split-half Spearman correlations of response
rate are:

| Network | Response-rate rho | Continuous amplitude rho | Top-20% overlap |
| --- | ---: | ---: | ---: |
| 1 | 0.853 | 0.883 | 0.783 |
| 2 | 0.740 | 0.764 | 0.554 |
| 3 | 0.724 | 0.610 | 0.568 |
| 4 | 0.873 | 0.935 | 0.868 |
| 8 | 0.954 | 0.986 | 0.857 |
| 9 | 0.786 | 0.777 | 0.550 |

Within-event permutations preserve every event's severity and availability
mask while shuffling response labels among its tones. UID heterogeneity,
split-half response-rate correlation, and top-20% overlap all exceed all 2,000
permutations in every network (empirical p <= 1/2001). These are operational
model checks, not independent detector-level significance claims.

The continuous amplitude statistic is each tone's absolute raw phase change
normalized by the median absolute phase change of all valid tones in that
network/event. Its repeatability shows that the result is not an artifact of
the binary response threshold alone.

## Result 2: The Stable Structure Follows Digital Tone Offset

Every affected network has response-rate banding versus signed digital tone
offset stronger than all 2,000 within-event permutations
(empirical p <= 1/2001). In the 12 equal-count offset bins, the broad response
minima occur at:

| Network | Response-rate minimum | Continuous-amplitude minimum |
| --- | ---: | ---: |
| 1 | +43.7 MHz | +43.7 MHz |
| 2 | +22.5 MHz | -22.3 MHz |
| 3 | +28.1 MHz | +28.1 MHz |
| 4 | +18.4 MHz | +64.0 MHz |
| 8 | +76.9 MHz | +76.9 MHz |
| 9 | +101.1 MHz | +58.1 MHz |

The useful observation is not that all minima have an identical coordinate;
they do not. It is that the response in each independently centered network
has a broad envelope tied to the digital offset from its own LO. This replaces
the initially misleading appearance of structure at an absolute probe
frequency.

The response threshold is not the main cause. Response-rate correlation with
the per-tone threshold ranges from +0.024 to -0.458. It is moderate in
networks 2 and 9 and small in the others. Likewise, the local target-sweep
phase derivative partially explains networks 8, 2, and 9 but is weak in
networks 1, 3, and 4. The target sweep is used only as a local complex
derivative basis; this analysis does not attribute the events to a failed
tune.

## Result 3: A Pure Delay or Sample Slip Does Not Fit

A path-delay perturbation predicts

`phase(tone) = common_phase + 2 pi tone_offset delay`

and a delay-dominated event should tend toward opposite signs across the LO
when common phase is not dominant. The median extra phase energy explained by
adding this delay term beyond a common phase is:

| Network | Median incremental R2 | Events with opposite signs across LO |
| --- | ---: | ---: |
| 1 | 0.001 | 0.038 |
| 2 | 0.012 | 0.077 |
| 3 | 0.031 | 0.154 |
| 4 | 0.016 | 0.462 |
| 8 | 0.098 | 0.000 |
| 9 | 0.026 | 0.288 |

Network 8 carries a modest phase-slope contribution, but its median response
on the two sides of the LO never changes sign. A simple path-delay jump,
sample slip, or timing-offset change is therefore not the dominant mechanism.

## Result 4: Events Pass Through Stable Network/Tone Transfer Functions

An uncentered singular-value decomposition tests the model

`tone response(event) = event amplitude(event) * fixed tone loading`

using only UIDs present in every event. The first-mode energy fractions and
split-half reproducibility of the phase loading are:

| Network | Phase rank-1 energy | Complex rank-1 energy | Loading cosine |
| --- | ---: | ---: | ---: |
| 1 | 0.584 | 0.427 | 0.760 |
| 2 | 0.598 | 0.486 | 0.700 |
| 3 | 0.693 | 0.637 | 0.718 |
| 4 | 0.602 | 0.550 | 0.852 |
| 8 | 0.895 | 0.809 | 0.985 |
| 9 | 0.577 | 0.415 | 0.329 |

The rank-1 loading sign is arbitrary and is aligned only for plotting.
Network 8 is exceptionally close to a single event-amplitude times stable
tone-transfer mode. Networks 1 through 4 show a substantial but incomplete
version of the same structure. Network 9 has stable response magnitude but a
poorly reproducible signed phase loading, so it requires more than one mode.

Residuals after the existing common gain, phase, and frequency-like event fit
retain tone-offset structure. This is descriptive rather than a formal
residual significance test, but it confirms that a single network-wide complex
coefficient is incomplete.

## Result 5: Networks Form Coupled Transfer Families

The fitted frequency-like coefficients vary together from event to event but
with network-dependent signs:

- Networks 1 and 2 are tightly coupled: Spearman rho = +0.914 over 46 paired
  fits and no opposite-sign events.
- Networks 3 and 4 have rho = +0.682 over 40 paired fits.
- Networks 2 and 8 have rho = -0.529 and opposite signs in all 52 paired
  events.
- Networks 1 and 8 have opposite signs in all 46 paired events.
- Networks 3 and 8 have phase-coefficient rho = +0.878 and the same phase sign
  in all 52 paired events.

These pairings support a shared event whose sign and shape are transformed by
stable per-network transfer functions. They do not by themselves locate the
physical component.

## Physical Interpretation

The strongest current model is:

`shared event amplitude/polarity -> network transfer -> stable tone-dependent response`

This accounts simultaneously for cross-rack timing, the stable affected
network subset, stable within-network UID susceptibility, and the fixed
digital-tone-offset envelope. It is more specific than a generic detector
instability and is inconsistent with random isolated glitches.

The data disfavor the following as complete explanations:

- Citlali or mapmaking, because the disturbance is in raw I/Q;
- telescope motion, because event onset does not causally follow the motion
  diagnostics;
- a tune failure, because the conclusion survives without treating sweep fits
  as causal and most tone structure does not track the local sweep slope;
- an array-temperature spike, because the recorded thermometry has no matching
  transient;
- a pure common gain, common phase, delay, or digital sample slip.

A perturbation of the common LNA-bias supply remains possible as the shared
trigger, but a scalar bias change alone does not explain the digital-tone
envelope. It would have to couple through different network/IF transfer
functions. Mixer/LO/IF gain, attenuation, analog filtering, ADC behavior, or a
digital channelization response are also plausible locations for that
transfer. The present data do not uniquely choose among them.

## Next Bounded Discriminator

Map `tone_offset_frequency_hz` to the actual ROACH/PFB channel, bin edge,
sideband, DAC lane, and ADC/firmware signal-path coordinates. Then ask whether
the response trough and rank-1 loading align in that hardware coordinate
across networks and whether the same structure appears in healthy control
networks or another observing night.

Do not infer that mapping from tone slot alone: tone slot is
observation-local, while UID is the detector identity. The next analysis
requires an authoritative description of the firmware/channel mapping or a
controlled readout test.

Future instrument tests should record time-resolved LNA-bias voltage/current,
IF power or gain state, and ADC statistics through the event. Those signals
are not present in the current `.nc` headers.

## Outputs

The complete artifact set is stored locally at:

`/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-tone-susceptibility-20260730`

The most useful figures are:

- `science_tone_susceptibility_vs_offset.png`
- `science_tone_relative_phase_vs_offset.png`
- `science_tone_response_heatmap.png`
- `science_tone_rank_one_modes.png`
- `science_network_pair_coupling.png`
- `science_tone_model_residuals.png`

The CSVs preserve the UID-level statistics, offset-bin summaries, direct
delay fits, rank-1 tone modes, cross-network coefficient coupling, and
model residuals. `manifest.json` records identities, semantics, inputs, and
parameters.

## Reproduction

First generate the complete schema-v2 event-vector tone table:

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/citlali-event-vector-v2-mpl \
XDG_CACHE_HOME=/private/tmp/citlali-event-vector-v2-cache \
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

Then run the tone analysis:

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/citlali-tone-susceptibility-mpl \
XDG_CACHE_HOME=/private/tmp/citlali-tone-susceptibility-cache \
$HOME/tolteca/bin/python \
  tools/diagnostics/science_iq_tone_susceptibility_analysis.py \
  --input-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-event-vector-20260730 \
  --output-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-tone-susceptibility-20260730 \
  --networks 1 2 3 4 8 9 \
  --frequency-bins 12 \
  --minimum-opportunities 20 \
  --n-permutations 2000
```
