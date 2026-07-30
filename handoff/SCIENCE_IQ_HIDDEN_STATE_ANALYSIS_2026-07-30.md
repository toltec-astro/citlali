# Science Raw-I/Q Event-Bounded Hidden-State Analysis

Date: 2026-07-30

## Question and Bounded Verdict

Do the frequent raw-I/Q transitions occupy recurrent levels, how long do the
levels persist, and do the affected networks share one common state?

The event-rich science observations decisively reject a one-level
description. The most compact bounded description is:

> The disturbance repeatedly occupies a baseline and two broad response
> modes. Those modes are synchronized in time but do not have one identical
> amplitude or state assignment in every network. The strongest recurring
> partition is approximately nw1/nw2/nw9 versus nw3/nw4/nw8, crossing both
> electronics racks.

The three-state joint model is a phenomenological summary of recurrent raw-I/Q
levels. It is not evidence for a literal three-position hardware switch.
Individual traces retain drift and excursions around the fitted centers.

This analysis uses the validated continuous-event catalog to define candidate
boundaries. It therefore measures state structure, occupancy, dwell, and
cross-network correspondence conditional on those boundaries. It is not an
independent measurement of event rate.

## Model Boundary

For each exact-APT science observation and affected network:

1. the stable-UID raw-I/Q projection and leave-one-observation-out template
   from the continuous-event analysis are reconstructed;
2. the primary cross-rack catalog events divide the complete observation into
   intervals;
3. an adaptive guard of at most 0.35 s is removed on either side of each
   transition;
4. the robust projected-phase median is measured in each surviving interval;
5. a Theil-Sen linear trend is separated from each observation/network; and
6. one-, two-, and three-state diagonal-Gaussian HMM candidates are fit to the
   interval medians.

The selected model is the smallest eligible state count within 6 BIC units of
the minimum. Eligibility requires convergence, at least 3% posterior
occupancy in every state, and at least one pooled-sigma separation between
every pair of state centers. Eight randomized initializations protect against
local optima.

A separate joint model uses the six robustly standardized network levels.
State labels are observation-local ordinals sorted by the nw8 center. They
are not hardware-state identities and must not be matched by ordinal alone
across observations.

The projected-phase unit is radians per RMS-normalized UID loading. It is an
operational raw-I/Q coordinate, not calibrated detector phase.

Observations 152390 and 152392 contain only four and two catalog intervals.
They are retained as baseline-only references but are explicitly marked
`insufficient_catalog_intervals`; no state-count comparison is claimed for
them.

## State-Selection Result

| Obsnum | Intervals | Network state counts | Joint states | Joint ΔBIC vs one | Joint boundary changes | Median uncensored joint dwell |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| 152419 | 84 | nw1:2 nw2:1 nw3:2 nw4:1 nw8:2 nw9:2 | 3 | -395.8 | 0.687 | 8.40 s |
| 152431 | 335 | nw1:2 nw2:2 nw3:3 nw4:3 nw8:2 nw9:2 | 3 | -1964.7 | 0.784 | 2.94 s |
| 152433 | 329 | nw1:2 nw2:2 nw3:3 nw4:3 nw8:3 nw9:2 | 3 | -2238.1 | 0.625 | 2.73 s |

All selected rich-observation models pass their convergence, occupancy, and
center-separation gates. The joint minimum center separations are 5.63, 4.89,
and 4.40 pooled sigma, respectively.

When a participating network's decoded state changes at a catalog boundary,
the state-center direction matches the catalog step sign 98.9%, 95.8%, and
97.0% of the time in 152419, 152431, and 152433. This consistency supports
the level interpretation, but it is conditional on catalog-defined
boundaries and is not an independent validation of their timing.

## Dwell and Transition Structure

The uncensored joint-state dwell distributions are broad:

- in 152419, state medians range from 7.8 to 15.9 s;
- in 152431, state medians range from 2.0 to 3.6 s; and
- in 152433, the two common states have medians of 2.4 and 3.1 s, while the
  least common state has a 25.7 s median across six uncensored runs.

The two non-baseline response modes almost never transition directly into one
another. In the decoded joint paths:

- 152419 contains one state-1 to state-2 transition and none in reverse;
- 152431 contains one state-1 to state-2 transition and none in reverse; and
- 152433 contains six state-1 to state-2 transitions and none in reverse.

Most mode changes return through state 0. This is better described as a
baseline plus two response patterns than as a scalar system stepping
monotonically among low, middle, and high levels.

## Cross-Network Response Modes

Pairwise interval levels are often strongly correlated even when independent
network HMMs choose different state counts. The median pairwise Spearman
correlation is 0.55 to 0.64 in the three rich observations. Selected examples
are:

- nw3/nw8 level correlations of 0.970, 0.873, and 0.981 in 152419, 152431,
  and 152433;
- nw3/nw8 adjusted Rand state agreement of 0.732, 0.434, and 0.849; and
- nw1/nw2 adjusted Rand agreement of 0.973 in 152433.

The joint center vectors expose an approximate cross-rack partition:

- one response mode is strongest in nw1/nw2/nw9;
- a second is strongest in nw3/nw4/nw8; and
- in 152433 the second mode is negative in nw1/nw2/nw9 while positive in
  nw3/nw4/nw8.

This partition is not absolute. In 152431 one mode is positive in all six
networks, though strongest in nw1/nw2/nw3/nw9, while the other is dominated
by nw3/nw4/nw8. The result constrains the failure topology: it is neither
simply RACKA versus RACKO nor a single identical state applied to all
networks.

The most defensible physical description is a shared or synchronized
disturbance filtered through network-specific susceptibility, amplitude, and
possibly sign. The data do not identify whether the shared origin is the LNA
bias supply, observatory timing, another common coupling path, or a common
software/readout response.

## Citlali Consequence

This result reinforces the existing science guard:

- short transition masks cannot repair a persistent displaced state;
- one global rack flag is too coarse because the expression is
  network-specific;
- retaining every interval would allow recurrent offset modes into the map;
  and
- masking every inferred dwell would discard a large fraction of the worst
  observations.

A production validator should preserve network-level evidence, then form a
joint confidence or mode estimate. It should report at least event rate,
network participation, displaced-state occupancy, dwell duration, and the
fraction of data that would survive any proposed mask. It must not claim
repair merely because the narrow transition edge was removed.

The present hidden-state model is forensic and must not yet become an
automatic flagger. Its boundaries and UID template were learned from this
science dataset.

## Limitations and Next In-Can Test

- Catalog boundaries are supplied rather than independently inferred.
- Interval medians intentionally suppress within-interval waveform shape and
  weight intervals equally in the HMM likelihood.
- The state search is capped at three for bounded interpretation.
- Diagonal-Gaussian emissions are descriptive; visible drift and excursions
  remain around the centers.
- Only the six affected networks and five exact-APT science observations are
  modeled.
- State ordinal and projected sign are operational conventions, not hardware
  labels.
- No time-resolved internal readout, bias, clock, or electronics-temperature
  telemetry exists to identify the trigger.

That held-out test is now complete. A three-state model trained without
catalog times on the first half of observation 152431 recovers 96.2% of
catalog events in the held-out half after target-intrinsic shape
normalization, and transfers with 78.3% and 81.1% recall to observations
152419 and 152433. See
`handoff/SCIENCE_IQ_HELD_OUT_MODE_DETECTION_2026-07-30.md`.

## Outputs

The artifact set is stored locally at:

`/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-hidden-state-20260730`

Central products are:

- `observation_state_summary.csv`
- `state_model_comparison.csv`
- `network_state_parameters.csv`
- `joint_state_parameters.csv`
- `network_interval_state_assignments.csv`
- `joint_interval_state_assignments.csv`
- `state_dwell_runs.csv`
- `state_transition_matrices.csv`
- `network_event_state_audit.csv`
- `joint_event_state_audit.csv`
- `cross_network_state_correspondence.csv`
- interval-measurement and trajectory-example tables;
- five diagnostic figures; and
- `manifest.json`, containing definitions, input identities, parameters,
  output names, and row counts.

## Reproduction

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/private/tmp/citlali-hidden-state-mpl \
XDG_CACHE_HOME=/private/tmp/citlali-hidden-state-cache \
LOKY_MAX_CPU_COUNT=8 \
$HOME/tolteca/bin/python \
  tools/diagnostics/science_iq_hidden_state_analysis.py \
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
  --output-dir \
  /Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/science-iq-hidden-state-20260730
```
