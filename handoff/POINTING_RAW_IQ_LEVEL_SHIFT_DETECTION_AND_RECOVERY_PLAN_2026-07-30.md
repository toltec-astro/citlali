# Pointing raw-I/Q level-shift detection and recovery plan

Date: 2026-07-30

Status: active; offline detection development

## Scientific objective

The immediate objective is to produce the most scientifically defensible
NGC4449 maps possible from the data already collected. We will recover
additional detector-time only when it improves usable sensitivity without
biasing source flux, PSF, morphology, background structure, or reported
uncertainty.

The detector is not an end in itself. Its purpose is to replace avoidable
whole-network exclusion with conservative transition masking and stable-chunk
recovery when map-level evidence supports doing so.

The longer-term hardware-cause investigation is proceeding separately. This
plan does not attempt to identify the physical source of the level shifts.

## Scope and safety boundary

Until the validation milestones below pass:

- all analysis remains offline and detection-only;
- no candidate changes Citlali flags, detector weights, or maps;
- thresholds remain tunable and are not production constants;
- raw events, transition boundaries, and accepted chunks remain distinct
  objects;
- reference-network subtraction is supporting evidence, not a primary trigger;
- a map-level comparison, rather than event count, is the final acceptance
  test.

The existing diagnostics and their current contracts are documented in:

- `handoff/POINTING_RAW_IQ_LEVEL_SHIFT_IDENTIFIER_2026-07-30.md`;
- `handoff/POINTING_RAW_IQ_REFERENCE_RESIDUAL_REVIEW_2026-07-30.md`.

## Working detection model

### Detector response

At every candidate time, compare robust pre- and post-event complex-I/Q states
for each APT-usable detector. A detector has a strong response only when its
phase change exceeds both:

1. a detector-relative robust-noise threshold; and
2. an absolute phase-change floor.

Positive- and negative-going detector fractions are preserved separately.
Opposite response signs between networks are allowed.

The current high-confidence starting point is:

- significance threshold: 8 sigma;
- absolute phase floor: 5 mrad;
- same-sign network participation: 10%;
- pre/post comparison windows: 0.20 s;
- guard interval: 0.05 s.

These values are initial review settings, not accepted production policy.

### Observation-level event identity

The master event timeline is the union of independently detected network
candidates. It is not a requirement that every network cross a threshold.

For every clustered event, the diagnostic will measure every network in a
common time neighborhood and preserve a response vector containing:

- positive and negative participating fractions;
- maximum same-sign fraction and its time;
- characteristic phase-change amplitude;
- affected detector identities;
- threshold-sweep support;
- raw and reference-residual evidence;
- explicit near-zero or missing response.

Provisional review classes are:

- `strong_single_network`;
- `corroborated`;
- `threshold_sensitive`;
- `overlapping_or_unresolved`;
- `false_or_non_step`.

A 5% same-sign participation level may be tested as corroborating evidence
when temporally aligned with another network. It must not become an
independent trigger without review evidence.

### Transition boundary

The current pre/post score maximum is a candidate locator, not a physical
change-point estimate. A separate local analysis will determine:

- transition onset;
- transition duration;
- settling interval;
- affected detectors;
- credible pre- and post-transition plateaus;
- whether the feature is a step, impulse, drift, or overlapping event.

Mask boundaries must be derived from this local transition analysis, not from
the width or center of the 0.20-second detection windows.

### Stable chunks

A stable chunk is an interval between accepted change points that independently
passes criteria for:

- minimum useful duration;
- sufficient usable detectors;
- stationary robust center and slope;
- stable noise;
- no unresolved transition;
- plausible relation to the quiet-network atmospheric shape;
- stable behavior in Citlali's calibrated timestream coordinates.

Simply flagging transition samples may be insufficient. Filtering,
common-mode estimation, and baseline operations must not bridge different
level states. Citlali integration may therefore require explicit segmentation
or an equivalent mask-aware boundary contract.

## Milestones and gates

### Milestone 0 — Freeze the scientific and detection contract

Status: **complete**

- [x] State that scientific map recovery is the primary objective.
- [x] Separate hardware diagnosis from data mitigation.
- [x] Separate candidate detection, transition localization, chunk acceptance,
      and map acceptance.
- [x] Keep all current work offline and detection-only.
- [x] Establish the initial three-threshold detector and threshold sweep.
- [x] Establish nw0/nw5 reference residuals as supporting evidence only.

Gate: the written contract distinguishes scientific success from detector
candidate count.

### Milestone 1 — Build the observation-level consensus report

Status: **complete**

- [x] Cluster per-network candidates into a union event timeline.
- [x] Measure every network at every union-event time without requiring it to
      cross the normal selection threshold.
- [x] Preserve signed response, detector membership, and near-zero responses.
- [x] Generate aligned multi-network cutouts for by-eye review.
- [x] Write machine-readable event, network-response, strong-detector, and
      provenance products.
- [x] Distinguish strong single-network events from weakly corroborated events.
- [x] Attach compatible reference-residual evidence to primary seeds when its
      manifest is supplied.

Gate: a reviewer can determine from one event product which networks and
detectors responded, with what sign and strength, without inferring presence
from unequal per-network event counts.

The gate passed on observation 152434. The classification remains review
triage: in particular, `corroborated` does not mean that a physical common
cause or a valid transition boundary has been established.

### Milestone 2 — Build and label the validation corpus

Status: **in progress**

- [x] Select early-, middle-, and late-night pointing observations.
- [x] Include quiet controls and visibly pathological networks.
- [ ] Label candidates as clear, ambiguous, false, or non-step.
- [ ] Review all raw-only and residual-only candidates.
- [ ] Review representative high- and low-support stable matches.
- [x] Freeze editable review-table copies separate from regenerated outputs.

Gate: the corpus spans both benign and pathological conditions and contains
enough reviewed events to measure failure modes, not merely candidate yield.

### Milestone 3 — Tune event selection

Status: **pending**

- [ ] Measure event recall and false-positive rate over the threshold grid.
- [ ] Evaluate the proposed 5% multi-network corroboration level.
- [ ] Test sensitivity to event-clustering tolerance.
- [ ] Confirm that single-network strong events are not lost.
- [ ] Confirm that reference subtraction does not become an unguarded trigger.
- [ ] Record accepted thresholds and known blind spots.

Gate: selected thresholds achieve high transition recall on pathological
networks while retaining a low false-positive rate on quiet controls, and
small threshold perturbations do not radically change the accepted corpus.

### Milestone 4 — Estimate transition boundaries

Status: **pending**

- [ ] Implement local high-time-resolution change-point estimation.
- [ ] Estimate onset, duration, and settling separately.
- [ ] Determine affected detectors for each event.
- [ ] Reject impulses, smooth drift, and unresolved overlaps from the
      level-shift class.
- [ ] Validate boundary placement by eye on the labeled corpus.

Gate: transition intervals conservatively contain the visible disturbance
without discarding an unjustifiably large fraction of adjacent stable data.

### Milestone 5 — Classify stable chunks

Status: **pending**

- [ ] Define a minimum useful chunk duration.
- [ ] Measure within-chunk center, slope, noise, and unresolved-event evidence.
- [ ] Measure detector survival and network usability.
- [ ] Check stability after conversion to calibrated timestream coordinates.
- [ ] Produce an inspectable chunk ledger with acceptance reasons.
- [ ] Test threshold sensitivity of chunk acceptance.

Gate: every accepted and rejected chunk has explicit evidence and provenance,
and accepted chunks remain stable under reasonable parameter perturbations.

### Milestone 6 — Perform offline map-recovery ablations

Status: **pending**

For representative pointings and then NGC4449 science observations, compare:

1. the current whole-network-exclusion baseline;
2. all-network maps without mitigation;
3. transition-masked maps;
4. stable-chunk-recovered maps;
5. reasonable variations of the detection and chunk thresholds.

Evaluate:

- usable detector-time and map weight;
- point-source centroid, PSF, peak, and curve of growth;
- integrated flux;
- source-free background RMS and structure;
- bowls, stripes, ghosts, and edge artifacts;
- NGC4449 regional flux and S/N stability.

Gate: stable-chunk recovery measurably improves sensitivity or flux recovery
over whole-network exclusion without reintroducing pathological morphology or
creating threshold-sensitive scientific results.

If this gate fails, whole-network exclusion remains the scientifically
preferred mitigation.

### Milestone 7 — Design the Citlali integration contract

Status: **in progress**

- [x] Choose raw complex I/Q at the KIDs solve boundary as the earliest
      scientifically valid mode-detection coordinate.
- [x] Define a versioned coherent-event diagnostic and mode-template contract.
- [ ] Define affected-detector, transition-boundary, and chunk data structures.
- [ ] Define how segmentation interacts with filtering and common-mode
      estimation.
- [x] Define the initial requested compatibility metadata and realized
      diagnostic provenance.
- [x] Define fail-closed template matching and observe-only QA outputs.
- [ ] Decide which thresholds are user-facing and which are derived.
- [ ] Record the intended numerical behavior change.

Gate: an architecture review confirms that the design has one authority for
each fact, preserves identity and units, and cannot silently claim successful
mitigation when required validation is absent.

### Milestone 8 — Implement and validate in Citlali

Status: **pending**

- [ ] Add focused unit and synthetic tests.
- [ ] Add offline-versus-Citlali event and boundary equivalence tests.
- [ ] Add configuration validation and truthful realized-state reporting.
- [ ] Run local build, focused tests, full CTest, baseline-tool tests, and
      config preflight.
- [ ] Run bounded Unity pointing reductions.
- [ ] Run the NGC4449 guarded comparison.
- [ ] Update scientific conventions, retained debt, and refactor status when
      the validated behavior changes.

Gate: Citlali reproduces the accepted offline decisions and passes the
map-recovery gate with zero unexpected error-level messages.

## Decision rules

The costs are intentionally asymmetric:

- transition detection favors recall because one missed step can contaminate
  a large map area;
- stable-chunk acceptance favors precision because accepting unstable data can
  bias the science;
- whole-network exclusion requires map-level justification because it loses
  substantial sensitivity.

No single threshold should control all three decisions.

## Current evidence

For pointing observation 152434:

- nw0 and nw5 have no selected events at the initial thresholds;
- nw0 and nw5 median phase traces correlate at 0.9988;
- nw1, nw2, nw3, and nw4 have 14, 29, 48, and 18 raw candidates;
- almost all target-network candidates survive guarded reference subtraction;
- no target raw candidate is reference-active at the current reference-event
  threshold;
- the dominant level-shift population is therefore not explained by the
  nw0/nw5 atmospheric template;
- clustered target-network detections include both shared and single-network
  events;
- many shared events have network-dependent response sign and strength.

The first consensus-report validation adds:

- 109 primary per-network candidates collapse into 55 union events;
- 32 union events contain normal-threshold seeds from at least two networks,
  while 23 initially contain only one seed network;
- measuring all networks near every event increases the count with at least
  two 10%-level responses to 38;
- the provisional 5% corroboration rule labels 49 events as corroborated;
- among the 23 singleton-seed events, 17 receive a second 5%-level response,
  four remain threshold-sensitive, and two remain high-support strong
  single-network candidates;
- nw0 reaches 5.4% at one union event while nw5 remains below 0.8%, so 5% is
  demonstrably a review threshold rather than a frozen acceptance threshold;
- all 109 primary seeds match the companion reference-residual table: 104
  survive reference subtraction, five are raw-only threshold-sensitive, and
  none are coincident with a selected reference-network event;
- some apparently corroborated cutouts contain broad or overlapping activity,
  confirming that consensus triage cannot replace the later change-point and
  morphology review.

These observations motivate the union timeline and continuous per-network
response vector. They do not yet freeze a corroboration threshold.

## Progress log

### 2026-07-30

- Established an observation-long raw-I/Q phase-change detector.
- Added an explicit 27-point threshold sweep and by-eye review products.
- Demonstrated strong separation between quiet nw0 and pathological nw2 in
  observation 152434.
- Added guarded nw0/nw5 atmospheric-reference residual analysis.
- Confirmed that the dominant nw1--nw4 candidates survive reference removal.
- Adopted the science-first recovery objective and the staged gates above.
- Implemented the Milestone 1 observation-level consensus diagnostic in
  `tools/diagnostics/pointing_iq_multinetwork_consensus_review.py`.
- Added union-event, per-network response, strong-detector, manifest, overview,
  and aligned event-cutout products.
- Added optional, parameter-checked attachment of the existing
  reference-residual evidence.
- Passed 16 focused synthetic tests and Ruff checks for the new diagnostic.
- Completed the first real-data run on observation 152434 and passed the
  Milestone 1 review-product gate.
- Added a successful zero-event output contract for quiet observations; the
  focused suite now contains 17 passing tests.
- Created the initial Milestone 2 corpus under
  `/Users/gwilson/work_toltec/local_data/2025-C1-COM-01/docs/`
  `pointing-iq-level-shift-consensus-20260730/`.
- The corpus currently contains:
  - early quiet control 152390: zero primary events in nw0--nw5;
  - middle pathological 152420: 56 per-network candidates forming 27 union
    events;
  - later pathological 152432: 73 per-network candidates forming 35 union
    events;
  - late pathological 152434: 109 per-network candidates forming 55 union
    events.
- Event recurrence therefore rises `0 -> 27 -> 35 -> 55` across this selected
  sequence. This is descriptive corpus structure, not yet a threshold-quality
  measurement.
- At union-event times, quiet-control nw0/nw5 remain below 1.7% in 152420 and
  below 1.2% in 152432. In 152434 nw0 reaches 5.4% once, reinforcing that the
  provisional 5% rule requires event-by-event review.
- Integrated the 52-event raw-I/Q tone-mode handoff from Investigate commit
  `422f25f5f`.
- Added a versioned, fail-closed, observe-only coherent-mode template and
  classifier with no data/flag/weight mutation.
- Alternating-half event cross-validation at the descriptive 0.6 cosine /
  5 mrad operating point recognizes 167/212 independent cluster-member
  network responses and 52/100 shared-epoch nonmember responses.
- None of 660 fixed quiet-scan epochs pass that descriptive point.
- nw8 passes the proof-of-concept gate (44/44 member events; split-half loading
  cosine 0.985); nw9 does not pass a one-mode stability gate (23/37 at the
  descriptive point; split-half loading cosine 0.329).
- The existing RTC accepted-event fanout has an estimated median of 150.5
  per coherent event, demonstrating the compact-record benefit.
- Recorded the audit, schema, evaluation, masking design, and subtraction
  stop rules in
  `handoff/COHERENT_RAW_IQ_MODE_OBSERVE_ONLY_ARCHITECTURE_2026-07-30.md`.

## Immediate next action

Continue Milestone 2 by labeling a stratified event set:

1. all six events not corroborated at 5%;
2. all eleven singleton-seed events promoted only by 5--10% evidence;
3. all events with broad, overlapping, or visibly non-step morphology;
4. representative high-support, multi-seed events.

Then apply the same diagnostic to selected early- and middle-night quiet and
pathological pointing observations before tuning any threshold.
