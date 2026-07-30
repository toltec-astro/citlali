# Science Raw-I/Q Held-Out Mode Detection

Date: 2026-07-30

## Question and Bounded Verdict

Can the recurrent network response modes be detected without giving the
detector the event catalog, and do they transfer to unseen time and other
science observations?

Yes, at the level of event timing and response shape:

> A three-state model trained without catalog times on the first half of
> observation 152431 recovers 96.2% of catalog events in the held-out half
> after target-intrinsic shape normalization. The frozen model also recovers
> 78.3% of events in observation 152419 and 81.1% in observation 152433. All
> matched transitions have the catalog-expected direction, with median timing
> residuals of 0.096--0.127 s.

The observed match counts in all three event-rich evaluations exceed every
one of 200 circular-shift null trials. The minimum resolvable empirical
probability is therefore 1/201, or 0.004975, in each case.

The early quiet controls 152390 and 152392 have much lower shape-detector
transition rates, 1.74 and 0.87 per minute, and zero catalog matches. The rich
evaluations range from 6.43 to 22.08 transitions per minute.

This independently validates the event timing implied by the prior
catalog-conditioned hidden-state analysis. It does not establish literal,
observation-invariant hardware states, and it is not yet a production
flagging rule.

## Causality and Information Boundary

The detector never receives catalog event times, catalog boundaries, or event
labels during fitting or decoding.

For each observation and affected network nw1, nw2, nw3, nw4, nw8, and nw9:

1. the established stable-UID projected raw-I/Q phase coordinate is
   reconstructed from the exact observation APT;
2. non-overlapping 0.5 s medians are formed on one fixed time grid;
3. the first half of observation 152431 is detrended and robustly scaled;
4. a three-state diagonal-Gaussian HMM is fit to those six-network vectors;
5. the fitted centers, variances, and transition matrix are frozen;
6. the frozen model decodes the held-out half and the other observations; and
7. only after decoding are predicted transitions compared with the catalog.

The stable-UID projection remains informed by the prior event-mode analysis,
which is the main remaining conditioning boundary. For each event-rich target,
however, the projection template is leave-one-observation-out: its own events
did not train its projection. Observation 152431 uses templates trained on
152419 and 152433; 152419 uses 152431 and 152433; and 152433 uses 152419 and
152431. The two quiet controls use the fixed template learned from all three
event-rich observations.

Catalog matching is a one-to-one maximum-cardinality match within 0.75 s,
with the minimum total timing residual used to break ties. Circularly shifting
the predicted transition path relative to the catalog provides the timing
null.

The projected-phase coordinate is an operational radians-per-RMS-normalized-
UID-loading quantity. It is not calibrated detector phase.

## Two Transfer Questions

Each target is decoded twice because two scientifically different questions
must not be conflated.

### Frozen training scale

The target is locally detrended and centered, but divided by the training
scale. This preserves changes in absolute disturbance severity.

This evaluation degrades sharply when the night becomes nonstationary. In the
held-out half of 152431, the intrinsic scale relative to the training half is
10.89 in nw3 and 4.08 in nw4, while it is 0.65--1.28 in the other four
networks. The frozen-scale decoder therefore parks 89.0% of held-out bins in
one response mode and recovers only 30.3% of catalog events.

That is useful evidence, not merely a failed normalization: the response
amplitude changes substantially within one observation.

### Target-intrinsic shape scale

The target is locally detrended, centered, and robustly scaled without event
labels. This asks whether the relative six-network response pattern
transfers after allowing its overall severity to change.

This evaluation recovers the event timing strongly and is the basis of the
main verdict.

## Timing-Transfer Result

| Obsnum and role | Predicted transitions | Catalog events | Matched | Recall | Precision | Direction agreement | Median absolute timing residual | Null p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 152431 first-half training | 200 | 149 | 143 | 0.960 | 0.715 | 1.000 | 0.124 s | 66.0 |
| 152431 held-out, frozen scale | 125 | 185 | 56 | 0.303 | 0.448 | 0.857 | 0.184 s | 49.1 |
| 152431 held-out, shape scale | 230 | 185 | 178 | 0.962 | 0.774 | 1.000 | 0.122 s | 93.1 |
| 152390 quiet, shape scale | 36 | 3 | 0 | 0.000 | 0.000 | -- | -- | 1.0 |
| 152392 quiet, shape scale | 18 | 1 | 0 | 0.000 | 0.000 | -- | -- | 0.0 |
| 152419 transfer, shape scale | 133 | 83 | 65 | 0.783 | 0.489 | 1.000 | 0.096 s | 19.1 |
| 152433 transfer, shape scale | 365 | 328 | 266 | 0.811 | 0.729 | 1.000 | 0.127 s | 132.1 |

The precision is lower than the recall, especially in 152419. The model finds
additional state transitions that do not have a catalog partner within
0.75 s. They could be lower-amplitude real transitions, HMM fragmentation,
or false positives; this analysis does not relabel them as events.

## Learned Response Patterns

The training-state center differences relative to the baseline are:

| Response mode | nw1 | nw2 | nw3 | nw4 | nw8 | nw9 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mode 129 | 1.90 | 2.12 | 0.77 | 0.04 | 1.36 | 1.64 |
| mode 348 | 0.76 | 1.12 | 4.23 | 1.60 | 1.71 | 1.17 |

The first mode is strongest in nw1/nw2/nw9, while the second is dominated by
nw3 and includes nw4/nw8. This reproduces the broad cross-rack partition from
the catalog-conditioned analysis without using its boundaries.

Both response modes preferentially return through the baseline hub. The
training path contains only two direct mode-129 to mode-348 transitions and
two in reverse, compared with 37 and 61 baseline-to-mode transitions. The
pattern is therefore again more consistent with a baseline plus two response
families than with a scalar three-level staircase.

## What Transfers and What Does Not

Transition timing transfers more strongly than literal decoded-state
identity.

The adjusted Rand agreement with the prior event-conditioned state path is
0.640 in the held-out half of 152431, but only 0.041 in 152419 and 0.223 in
152433 under shape normalization. Normalized mutual information is similarly
modest outside the held-out half.

Therefore:

- the cross-network shape is a useful independent event detector;
- the event sequence is real and is not an artifact of catalog boundary
  placement;
- disturbance amplitude and occupancy are nonstationary;
- the labels `mode 129` and `mode 348` are phenomenological response-pattern
  names, not physical switch positions; and
- one observation's state occupancy must not be interpreted literally in
  another observation merely because the frozen template was used.

## Citlali Consequence

The result supports continued observe-only integration of coherent raw-I/Q
evidence. It also sharpens the required production contract:

- event detection and severity assessment are separate quantities;
- a narrow transition mask cannot correct the displaced dwell that follows;
- a validator must preserve the network response vector, not only a
  detector-by-detector event count;
- target-scale adaptation cannot silently become a flagging threshold because
  it can normalize severe pathology into a familiar shape; and
- any future automatic action must report the affected occupancy and
  data-loss consequence before changing samples or weights.

The current model remains forensic. It must not yet mask, subtract, exclude
networks, or alter map products.

## Limitations and Next In-Can Step

- The projection basis was learned from the event-rich science corpus even
  though each rich target uses a leave-one-observation-out template.
- The HMM state count is fixed at three from the prior bounded result.
- Target-intrinsic scaling improves shape transfer but removes absolute
  severity information.
- Quiet controls still contain 0.87--1.74 predicted transitions per minute,
  so a production decision threshold is not established.
- The 0.5 s binning and 0.75 s match window are appropriate to this forensic
  test but are not an approved runtime cadence or tolerance.
- Only five observations and six affected networks are included.
- The circular-shift null tests timing coincidence, not every form of
  structured HMM false positive.

The next in-can step should combine the two outputs rather than fit a more
complicated HMM: use the shape match as event evidence and the frozen-scale
residual as a separate severity/occupancy measure. Evaluate a predeclared
decision point on alternating time blocks and the quiet controls, preserving
observation-level false-positive rate and proposed data loss. This can inform
the observe-only Citlali sidecar but should remain non-mutating until a broader
corpus passes.

## Outputs

The artifact set is stored locally at:

`/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-held-out-mode-detection-20260730`

Central products are:

- `fixed_bin_network_levels.csv`
- `segment_normalization.csv`
- `detector_template_state_parameters.csv`
- `detector_template_transitions.csv`
- `detector_template.json`
- `decoded_fixed_bins.csv`
- `predicted_state_transitions.csv`
- `catalog_event_matches.csv`
- `circular_shift_null.csv`
- `segment_detection_summary.csv`
- three diagnostic figures; and
- `manifest.json`, containing semantics, parameters, input identities, output
  names, row counts, and a payload SHA-256.

## Reproduction

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/citlali-held-out-mode-mpl \
XDG_CACHE_HOME=/private/tmp/citlali-held-out-mode-cache \
$HOME/tolteca/bin/python \
  tools/diagnostics/science_iq_held_out_mode_detection.py \
  --data-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/data \
  --apt-root \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/apts \
  --event-vector-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-event-vector-20260730 \
  --tone-analysis-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-tone-susceptibility-20260730 \
  --continuous-analysis-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-continuous-event-morphology-20260730 \
  --hidden-state-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-hidden-state-20260730 \
  --output-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-held-out-mode-detection-20260730
```
