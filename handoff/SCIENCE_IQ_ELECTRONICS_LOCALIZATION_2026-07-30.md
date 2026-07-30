# Science I/Q Electronics-Coordinate Localization

Date: 2026-07-30

## Question and Bounded Verdict

Does the repeatable raw-I/Q event mode follow detector identity, a digital or
hardware coordinate, or neither?

The current dataset reaches the third outcome:

> It decisively rejects ownership by the observation-local tone-list slot,
> but it cannot distinguish detector UID from signed tone offset or absolute
> probe/RF frequency.

The reason is an identifiability limit, not weak event statistics. The same
UIDs changed tone-list slot frequently, providing a useful list-slot test, but
their LO and broad frequency coordinates did not move independently:

- every affected network retained one LO center in all 18 inspected
  observations;
- 53% to 94% of common UIDs changed tone-list slot between adjacent event
  observations;
- median UID tone-offset changes were only 5.9 to 31.3 kHz between adjacent
  observations;
- the largest measured common-UID shift in any comparison was 0.300 MHz; and
- no common UID changed sideband.

Consequently, UID, signed digital tone offset, and absolute probe frequency
remain nearly collinear. Since LO is constant, binned absolute-RF and
tone-offset models are mathematically identical in this dataset.

This is a localization result, not a production correction. No mitigation is
proposed or applied.

## Coordinate Authority

The all-tone inventory covers 54,306 raw tone records from 18 observations and
affected networks 1, 2, 3, 4, 8, and 9. Every raw tone remains in the
inventory, including unmatched and nonresponding tones.

The following coordinates are supported:

- detector identity: APT `uid`, when a matched APT is available;
- observation-local tone slot: zero-based array index in
  `Header.Toltec.ToneFreq`;
- signed digital tone offset: recorded
  `Header.Toltec.ToneFreq`;
- LO center: recorded `Header.Toltec.LoCenterFreq`;
- probe/RF frequency: derived as LO plus signed tone offset;
- readout board identity: network, ROACH index, and recorded MAC address;
- synthesizer, attenuator, compile-time, and selected-mask header values.

The raw files and APTs do not contain an authoritative PFB channel, position
within a PFB bin, DAC/ADC lane, channelizer path, or deployed firmware map. A
provisional DAC-comb FFT bin is included only as a clearly labeled software
calculation using the dated `2**21` and 512 MHz convention found in
`tolteca_web` and `taco_recipes`. It is not claimed to be the deployed
firmware/PFB coordinate.

## Held-Out Ownership Tests

Each of the three event-rich science observations 152419, 152431, and 152433
is held out in turn. Models are trained on the other two observations and
then evaluated on all events in the held-out observation. Only the per-event
mode amplitude is fit on the test event.

The score is zero-baseline phase energy explained. It is a descriptive
prediction score, not a statistical-significance measure. The UID and
tone-slot scores below use exactly the same held-out tones, eliminating a
denominator advantage.

| Network | Phase + delay | Offset, 12 bins | List slot, rank 1 | UID, rank 1 | UID, rank 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.238 | 0.263 | 0.153 | 0.510 | 0.544 |
| 2 | 0.405 | 0.451 | 0.318 | 0.430 | 0.578 |
| 3 | 0.325 | 0.334 | 0.207 | 0.508 | 0.587 |
| 4 | 0.135 | 0.133 | 0.065 | 0.373 | 0.486 |
| 8 | 0.408 | 0.472 | 0.473 | 0.881 | 0.939 |
| 9 | 0.123 | 0.125 | 0.019 | 0.437 | 0.618 |

The rank-1 UID model beats the rank-1 list-slot model in every network on the
same tones. The median advantage ranges from 0.112 in network 2 to 0.417 in
network 9. Network 8 is the clearest result: its rank-1 UID score is 0.825,
0.912, and 0.883 in the three held-out observations, while the corresponding
list-slot scores are 0.323, 0.576, and 0.456.

Pairwise observation-mode comparisons give the same answer. In network 8 the
median loading cosine is 0.937 when matched by UID, 0.925 when interpolated at
the nearly unchanged tone offset, and 0.640 when matched by list slot.
Networks 1 and 4 similarly retain UID/offset identity while list-slot
agreement collapses. Network 9 has a less stable signed first mode, but its
held-out UID prediction remains far stronger than list slot.

This rules out the raw list index as the owning coordinate. It does not choose
between UID and frequency because the same UID remained at essentially the
same frequency.

## Independent Pointing and Null Validation

The science-derived UID loading is projected, without refitting its tone
shape, onto:

- 47 separated raw-I/Q events from independent pointings 152420, 152432, and
  152434; and
- 40 fixed epochs from previously classified clean pointings 152390 and
  152418.

All APT-usable finite tones remain in each vector. The event/null AUC below is
the descriptive probability that a selected event score exceeds a selected
clean-epoch score; vectors within an observation are not treated as
independent statistical trials.

| Network | Event median R2 | Clean-epoch median R2 | Event/null AUC |
| --- | ---: | ---: | ---: |
| 1 | 0.408 | 0.009 | 0.913 |
| 2 | 0.489 | 0.048 | 0.930 |
| 3 | 0.417 | 0.012 | 0.937 |
| 4 | 0.426 | 0.020 | 0.885 |
| 8 | 0.668 | 0.045 | 0.961 |
| 9 | 0.420 | 0.028 | 0.928 |

For network 8, each event observation separately has a median projection R2
between 0.518 and 0.731. The two clean observations have medians of 0.034 and
0.054. The independently estimated pointing first modes also agree with the
science template: their loading cosines are 0.959, 0.971, and 0.988.

Thus the stable tone loading is neither overfit to the three science
observations nor manufactured by selecting only responsive tones.

## Rank and Residual Structure

Network 8 is close to, but not exactly, rank one:

- rank 1 explains 89.47% of the 52-event phase energy;
- ranks 1 and 2 explain 94.93%;
- the second mode accounts for 5.46 percentage points, or 51.8% of the
  rank-1 residual;
- a conservative threshold-derived measurement-noise upper bound is 4.22% of
  total energy; and
- the remaining rank-2 residual is 5.07%.

The network-8 loading is stable under event polarity and amplitude splits.
Its cosine with the all-event mode is 0.994 for positive events, 0.934 for
negative events, 0.994 for the lower-amplitude half, and 0.998 for the
higher-amplitude half. This argues against a strong amplitude-driven
deformation of the dominant mode.

Networks 1 through 4 are more complex: rank 1 explains 58% to 69% and rank 2
explains 73% to 79%. Network 9 is also multi-mode; its lower-amplitude mode
has cosine 0.404 with the all-event mode, versus 0.994 for high-amplitude
events.

Held-out UID-mode residuals remain structured versus tone offset, especially
at the negative-frequency edge of networks 8 and 9. They are much smaller
than phase-plus-delay residuals, but the remaining structure says that UID
rank 1 is not the whole transfer function. Apparent residual structure versus
list slot cannot be interpreted as slot ownership because slot and frequency
ordering are not independent in a single observation.

## Trigger Telemetry Boundary

The files contain enough setup information to establish that LO, attenuation,
compile-time, and selected-mask states are constant at the observation level.
They do not contain event-time telemetry for:

- LNA-bias voltage or current;
- per-network electronics temperatures;
- ROACH/PFB status or control registers;
- 10 MHz/PPS lock state;
- IF power, gain, or compression state; or
- interior ADC waveform statistics beyond the beginning/end snapblocks.

Recorded cryogenic temperatures and telescope motion have already given
negative trigger evidence. The event-time modulus tests give no robust PPS or
tested block-boundary locking. LMTMC normally issues no mid-observation
TolTEC command. These results constrain triggers but do not replace the
missing electronics telemetry.

## Ranked Path Localization

The evidence supports the following ranking:

1. **Stable detector/RF or frequency-dependent per-network transfer,
   excited by a shared trigger.** This is the strongest surviving class. It
   explains cross-rack timing, stable network selection, repeatable all-tone
   loading, and independent pointing transfer. The present observations
   cannot divide this class into detector/resonator identity, absolute RF,
   signed digital offset, analog filtering, mixer/IF response, ADC response,
   or channelizer response.
2. **A shared LNA-bias or other common electrical perturbation passed through
   different network transfer functions.** This remains a plausible trigger,
   especially because the affected set crosses racks, but it lacks direct
   telemetry and does not by itself explain the tone envelope.
3. **A multi-mode network response layered on the dominant transfer.** This is
   required for networks 1 through 4 and 9 and is weakly required even in
   network 8. It could arise from multiple analog/digital paths or operating
   point dependence, but the existing coordinate inventory cannot localize
   it further.
4. **Observation-local tone-list slot or simple software index ownership.**
   This is strongly disfavored by the slot-reassignment and held-out tests.
5. **Pure common gain, phase, delay/sample slip, or clock-boundary event.**
   These are incomplete explanations and are disfavored by held-out scores,
   phase-sign behavior, and timing tests.

Citlali/mapmaking is excluded as the origin because the disturbance is in raw
I/Q. A tune failure, telescope motion, or a recorded cryostat-temperature
spike is not supported as the trigger by the preceding analyses.

## Full-Duration Temporal Follow-up

The full-duration UID-template catalog is complete and documented in
`handoff/SCIENCE_IQ_CONTINUOUS_EVENT_MORPHOLOGY_2026-07-30.md`.

It finds 745 primary cross-rack events in the three event-rich science
observations, compared with the 52-event RTC-guided sample. The two worst
observations contain about 16 transitions per minute. Participating networks
have unanimous sign within each event, while successive directions alternate
77% to 85% of the time. A direct projected-phase trace visibly switches
between broad levels.

This strengthens the shared-trigger, network-dependent-transfer conclusion
and changes the temporal description from sparse impulses to persistent,
bidirectional, telegraph-like state transitions. It does not break the
remaining UID-versus-frequency-coordinate degeneracy.

## Minimal Controlled Retune

The smallest decisive experiment is an A/B/A LO retune that moves the same
detector UIDs through digital offset while keeping them on their physical
resonances:

1. Choose one event-prone network, preferably network 8, and a stable
   comparison network.
2. At state A, tune and record a short raw-I/Q scan with the normal LO.
3. At state B, move the LO enough that common UIDs shift by several broad
   susceptibility-bin widths, while regenerating the comb so those same UIDs
   remain on resonance. A total digital-offset displacement of order
   50--100 MHz is the useful target if the hardware and detector bandwidth
   permit it; a repeat of state A closes the drift test.
4. Preserve the same detector set and record the exact APT, LO, tone list,
   attenuations, compile-time/firmware identity, and an authoritative
   tone-to-PFB/lane map.
5. Acquire enough repeated events in each state to estimate a loading. If
   natural events cannot be reproduced on demand, a controlled perturbation
   must be defined and approved by the instrument team; this analysis does
   not prescribe one.

The discriminator is direct:

- if loading stays with UID while digital offset moves, detector/resonator or
  absolute-RF ownership wins over digital-offset ownership;
- if loading stays at digital offset/PFB coordinate and transfers to different
  UIDs, the digital/channelizer path wins; and
- if neither transfers cleanly, fit the justified rank-2 model and test the
  analog/RF and digital coordinates separately.

An authoritative firmware channel map is necessary to distinguish a PFB bin,
bin edge, lane, or channelizer path. Tone-list slot must not be substituted
for that map.

## Outputs

The complete artifact set is stored locally at:

`/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-electronics-localization-20260730`

The central files are:

- `electronics_coordinate_authority.csv`
- `tone_electronics_coordinates.csv`
- `coordinate_identifiability.csv`
- `heldout_model_summary.csv`
- `heldout_model_event_scores.csv`
- `mode_mapping_stability.csv`
- `low_rank_decomposition.csv`
- `mode_sign_amplitude_stability.csv`
- `heldout_residual_coordinate_bins.csv`
- `independent_pointing_population_comparison.csv`
- `independent_pointing_mode_summary.csv`
- `trigger_telemetry_inventory.csv`
- five diagnostic figures; and
- `manifest.json`, which records input identities, coordinate semantics,
  thresholds, and output counts.

## Reproduction

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/citlali-electronics-localization-mpl \
XDG_CACHE_HOME=/private/tmp/citlali-electronics-localization-cache \
$HOME/tolteca/bin/python \
  tools/diagnostics/science_iq_electronics_localization.py \
  --data-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data \
  --apt-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/apts \
  --event-vector-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-event-vector-20260730 \
  --tone-analysis-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-tone-susceptibility-20260730 \
  --pointing-consensus-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/pointing-iq-level-shift-consensus-20260730 \
  --output-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-electronics-localization-20260730
```
