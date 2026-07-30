# Coherent raw-I/Q mode: observe-only architecture and evaluation

Date: 2026-07-30

Status: implemented offline; production mutation disabled

Investigate evidence: commit `422f25f5f`, integrated on this branch as
`ad842d6cf`

## Decision

Mode projection is useful enough to become a first-class all-network
diagnostic event, but the evidence does not support automatic subtraction or
interchangeable templates between networks.

At the descriptive operating point of absolute cosine at least 0.6 and
absolute projected amplitude at least 5 mrad, alternating-half
cross-validation recognizes 167 of 216 network/event pairs that independently
participated in the 52 synchronized event clusters (77.3% recall). It selects
none of 1,210 fixed epochs in selected scans of the quiet observations 152390
and 152392. It also identifies 52 of 356 network responses measured at a
shared event epoch that were below the original independent network trigger.

Those numbers answer the immediate architectural question:

- the projection adds recognition beyond independent threshold crossings;
- one compact event record can replace a large per-detector diagnostic fanout;
- the classifier is not complete enough to replace the existing detector
  protections;
- current null evidence is promising but not a production false-positive
  bound; and
- every network represented in the raw corpus should participate in further
  observe-only trials; nw8 is the strongest benchmark, while nw9 is not
  adequately represented by one stable rank-one template.

No code in this slice changes raw data, RTC/PTC flags, detector weights,
learning state, map inputs, or maps.

## Existing repair-path audit

### Raw data and loss of coherent identity

`KidsDataProc::populate_rtc_from_rawobs` reads one network file at a time,
runs the KIDs solver, selects one requested scalar channel (`xs`, `rs`, `is`,
or `qs`), and concatenates it into the sample-by-detector RTC matrix. In the
ordinary `xs` path the simultaneous complex `I + iQ` vector is no longer
available after this boundary.

This is the earliest and narrowest point at which the observed pathology can
be classified in its measured coordinate system. It is before despiking,
filtering, detector removal, cleaning, and mapmaking.

### Current RTC detection and flagging

RTC despiking produces detector-local raw and delta events. Network summaries
then aggregate detector step and impulsive scores by fraction and temporal
alignment. Optional network masks set detector/sample flags around the
dominant interval. The impulsive capture product retains a bounded set of
detector snippets, but it does not retain the simultaneous tone-response
vector or a physical coherent-event identity.

### Current learning collection

`collect_rtc_learning_diagnostics` records every accepted local raw or delta
event as a separate `LearnedSampleMask` keyed by detector UID.

`collect_ptc_learning_diagnostics` likewise records each accepted
second-pass detector event separately. The top event can also be emitted as an
additional diagnostic row.

The effective learning state is safer than the old hard-limit warning
suggested: `record_learned_sample_mask` merges intervals into an uncapped
canonical union keyed by observation, scan, application stage, and UID before
the bounded diagnostic row is appended. Filling `max_records_per_type`
therefore drops diagnostic history, not effective operational masks.

However, the effective representation still has one interval series per UID.
It cannot say that hundreds of intervals are observations of one network
event, cannot preserve the stable tone-loading vector, and cannot express
template compatibility or residual energy.

### Current application

Later iterations query canonical UID intervals from earlier iterations,
propose a sample-by-detector mask, apply source protection, reject the entire
application when the configured new-flagged fraction is exceeded, and
otherwise set the existing flags. Detector exclusions follow a separate
penalty path.

The mode-aware observer must not bypass these safeguards. A future event-level
mask should expand through an explicit adapter into the existing proposal and
source-protection path.

### Diagnostics and record multiplicity

RTC NetCDF diagnostics preserve detector and network summaries. Learning CSVs
preserve the bounded event log and application summaries. Neither contains a
compact coherent-event record.

For the 52-event corpus, the existing RTC accepted local-event count,
apportioned among coherent clusters that share a scan/network, has a median of
150.5 records per coherent event, a 90th percentile of 673, and a maximum of
1454. This is an estimate because the RTC count is scan-wide, not already
associated with the physical event. It nevertheless demonstrates the
representation mismatch.

## Narrow integration point

The production integration should be an observation-local sidecar at the raw
KIDs solve boundary:

1. receive simultaneous raw `I` and `Q`, receive time, network ID, tone slot,
   APT UID, signed digital tone offset, and declared readout compatibility
   metadata;
2. compute a bounded mode-projection score trace or candidate list before the
   complex vector is discarded;
3. persist compact scan/network event diagnostics;
4. after RTC, correlate those event times with detector-local and network RTC
   summaries for comparison only; and
5. leave the primary RTC matrix and all flags unchanged.

The sidecar has no network allow-list. It attempts every network present in
the observation. Each network uses its own compatible template and emits an
explicit `template_unavailable` or compatibility status when it cannot be
scored. Template quality changes the interpretation of a score; it does not
silently suppress observe-only acquisition.

This state should be owned by the observation pipeline, not added as new
cross-cutting mutable state on `Engine`. The generator currently processes raw
scans sequentially before the scan farm, so it can produce deterministic
scan-keyed diagnostics without process-lifetime state.

Using only the current RTC candidate time would require retaining raw-I/Q
vectors or snippets until after `rtcproc.run`. Computing the bounded
observe-only projection while raw I/Q is already present avoids that memory
growth and can still be correlated with RTC candidates by time.

## Template contract

The executable schema is
`validation/coherent_iq_mode_template.schema.json`. The reference
implementation is
`tools/diagnostics/coherent_iq_mode_observer.py`.

A template declares:

- template ID, version, lifecycle state, and creation time;
- authoritative network and readout identity;
- detector UID as the join identity;
- tone slot as a readout coordinate, not a detector identity;
- signed digital tone offset from the network LO;
- one or more tone-loading modes;
- RMS-unity normalization and deterministic sign convention;
- training dataset, method, event count, and version;
- required compatibility metadata and unresolved compatibility fields;
- minimum compatible tone coverage and tone-frequency tolerance;
- stability and uncertainty statements; and
- provenance to the Investigate commit, handoff, and source artifact.

The current sign convention makes the largest-absolute loading positive.
Projection amplitude therefore carries the repeatable event sign.

Input order is not trusted. Reordered tones are joined by UID and accepted only
when their frequency coordinate is compatible. Duplicate UIDs, the wrong
network, incompatible required metadata, insufficient coverage, and
out-of-tolerance tone coordinates fail closed with an explicit status.
Partial coverage is reported as a count and fraction.

The schema supports multiple modes. This is intentional: a weak or unstable
rank-one network must not be forced into the nw8 model.

The templates produced from the current corpus remain `observe_only`.
Firmware, readout-software, and detailed IF state were unavailable and are
listed as unresolved compatibility metadata. They are not production
templates.

## Event diagnostic contract

One scored network event reports:

- template ID and version;
- network and event time supplied by the caller;
- primary mode ID;
- signed projection amplitude in mrad RMS phase change;
- signed and absolute cosine similarity;
- primary-mode explained-energy fraction;
- total and residual phase energy;
- multi-mode explained-energy fraction;
- common-phase explained-energy fraction;
- common-phase-plus-tone-offset-slope explained-energy fraction;
- compatible tone count and fraction;
- rejected tone count and compatibility notes; and
- distinct coincident network count and identities when records are grouped.

The common-phase and delay/slope fits are comparisons, not significance maps
or causal labels. A high mode score says that the event vector resembles the
versioned template. It does not identify the hardware cause.

## Real-data evaluation

The evaluation tool is
`tools/diagnostics/coherent_iq_mode_evaluation.py`.

It uses alternating event halves: every event score is produced with a mode
fit to the other half. This avoids scoring an event with a mode directly
trained on that event. The positive label is independent network membership in
one of the 52 synchronized raw-I/Q clusters. A network measured at the shared
epoch but not independently triggering is retained as an ambiguous
low-amplitude class, not silently relabeled positive.

The null set contains five fixed epochs in each selected quiet scan for every
evaluated network. The first affected-network evaluation contained 660
network/epoch examples. The all-network extension contains 1,210. Those
samples include real
sky, atmosphere, tune state, and telescope scan motion. They are not a
continuous all-night scan and do not test another night, firmware state, or IF
configuration. Zero false positives in 1,210 samples corresponds to an
approximate 95% binomial upper bound of 0.25% overall; the per-network null
sample is only 110.

### Descriptive confusion table

| Actual class | Selected | Not selected | Total |
| --- | ---: | ---: | ---: |
| Independent cluster-member response | 167 | 49 | 216 |
| Quiet-scan fixed epoch | 0 | 1,210 | 1,210 |

This table uses absolute cosine at least 0.6 and absolute amplitude at least
5 mrad. It is a descriptive review point, not accepted production policy.

The threshold grid shows the expected trade:

- cosine 0.5, amplitude 2 mrad: 84.26% recall, 3/1,210 null selections;
- cosine 0.6, amplitude 5 mrad: 77.31% recall, 0/1,210 null selections;
- cosine 0.7, amplitude 5 mrad: 73.15% recall, 0/1,210 null selections; and
- cosine 0.6, amplitude 10 mrad: 61.11% recall, 0/1,210 null selections.

The 0.6/5 mrad point also selects 52 of 356 shared-epoch nonmember network
responses. These are candidates for the “below the old threshold but aligned
in time” population; they require by-eye review before being counted as
recovered events.

### nw8 versus nw9

nw8 is the clean proof of concept:

- training rank-one energy fraction: 0.895;
- split-half loading cosine: 0.985;
- cross-validated median absolute cosine for its 44 cluster-member events:
  0.950;
- all 44 pass the descriptive 0.6/5 mrad point; and
- none of its 110 quiet epochs pass.

nw9 is qualitatively different:

- training rank-one energy fraction: 0.577;
- split-half loading cosine: 0.329;
- cross-validated median absolute cosine for its 37 member events: 0.822;
- only 23 of 37 pass 0.6/5 mrad; and
- none of its 110 quiet epochs pass.

The reasonable event-level nw9 cosine does not rescue the unstable loading
estimate. A rank-two, state-conditioned, or separate event-family model should
be tested before nw9 receives any masking authority. nw9 should still be
scored and persisted in observe-only mode.

### All-network extension

The evaluation now infers and scores every network represented in the
event-tone corpus:

`nw0, nw1, nw2, nw3, nw4, nw5, nw7, nw8, nw9, nw11, nw12`.

Networks 6 and 10 have no raw/event-tone rows in this dataset and are reported
as unavailable. They are not excluded by policy.

The extension contains:

- 572 event/network vectors at the 52 shared event epochs;
- 216 independently threshold-triggering cluster-member responses;
- 356 measured shared-epoch nonmember responses; and
- 1,210 fixed quiet-scan network/epoch samples.

At the same descriptive cosine 0.6 / amplitude 5 mrad point, 167/216
cluster-member responses are selected (77.3%) and 0/1,210 quiet epochs are
selected. The four additional cluster-member responses come from nw12 and
remain below the 5 mrad amplitude floor.

The nominal control networks demonstrate why both shape and amplitude belong
in the diagnostic:

- nw7 has a cross-validated median quiet-epoch absolute cosine of 0.680 and
  60% of its quiet epochs exceed cosine 0.6, but its median projected
  amplitude is only 0.20 mrad and none exceed the joint 0.6/5 mrad point;
- nw11 has a stable loading (split-half cosine 0.996) and a median shared-epoch
  cosine of 0.952, but median amplitude is only 1.62 mrad and none pass the
  joint point; and
- nw12 has a stable loading (split-half cosine 0.998) and high shared-epoch
  cosine, but median amplitude is 3.08 mrad and no response passes 5 mrad.

These modes may represent repeatable low-level atmosphere/readout response,
not the destructive pathology. They are scientifically useful controls. A
high cosine by itself must never be called a pathology detection.

## Coherent masking design — disabled

The first optional masking experiment should operate at event level:

1. require an observe-only score that passes a reviewed template,
   compatibility, coverage, amplitude, cosine, and residual policy;
2. require a separately validated transition interval rather than reusing the
   pre/post scoring window;
3. declare affected UIDs from the measured response and template support;
4. expand the event through an adapter into the existing learned-mask
   proposal;
5. retain source protection, maximum new-flagged fraction, existing
   detector-local fallbacks, and application summaries;
6. record one coherent event plus the deterministic expanded mask count; and
7. compare maps against current whole-network exclusion and existing masking.

Mask acceptance requires improved usable weight or map noise without changes
to point-source centroid, PSF, peak/integrated flux, curves of growth, extended
source morphology, bowls, stripes, or uncertainty calibration. Until that
ablation passes, mode scores remain diagnostics only.

## Subtraction risk and acceptance plan — disabled

Subtraction is substantially riskier because the template may overlap real
astronomical, atmospheric, or calibration response and may change with tune,
LO placement, firmware, IF state, temperature, and event family.

It must remain disabled until all of the following pass:

1. template stability across nights, retunes, LO placements, firmware and
   readout states;
2. explicit rejection or separation of common phase, delay-like slope,
   detector-local glitches, and multiple coherent modes;
3. injected point-source and extended-source tests through the raw-I/Q
   classifier and subtraction path;
4. negative-amplitude and overlapping-event tests;
5. residual whiteness without coherent template leakage;
6. no measurable attenuation or shape change in astronomical and atmospheric
   signals;
7. map-level flux, PSF, morphology, noise, and uncertainty acceptance;
8. explicit configuration, template version, fitted amplitude, affected tone
   coverage, and subtraction provenance; and
9. a fallback that leaves data unchanged on every compatibility or fit
   failure.

No current result authorizes subtraction.

## Focused tests

The synthetic suite covers:

- positive and negative rank-one amplitudes;
- missing tones and explicit coverage;
- safe UID-based reorder;
- incompatible tone coordinates;
- detector-local events;
- common phase;
- delay/slope events;
- a two-mode mixture;
- zero-energy null input;
- wrong network;
- wrong required metadata;
- non-mutation of template and input arrays; and
- deterministic compact records.

The seven focused tests pass.

## Artifacts

Repository:

- `tools/diagnostics/coherent_iq_mode_observer.py`;
- `tools/diagnostics/coherent_iq_mode_evaluation.py`;
- `tools/diagnostics/test_coherent_iq_mode_observer.py`;
- `validation/coherent_iq_mode_template.schema.json`.

Generated evaluation:

- `coherent_mode_scores.csv`;
- `coherent_mode_threshold_grid.csv`;
- `coherent_mode_network_summary.csv`;
- `coherent_mode_template_nwN.json`;
- `manifest.json`.

The generated files belong under the project artifact directory
`docs/coherent-iq-mode-evaluation-20260730`.

## Next production step

Implement only the observation-local raw-I/Q sidecar and required diagnostic
writer. Run it for every network present in the observation, with one
versioned network-compatible template per score and explicit unavailable or
incompatible records. Use nw8 as the positive benchmark and nw0/nw5/nw7/nw11/
nw12 as control behavior, but do not encode either group as an allow-list.
Correlate scores with RTC events and prove byte-identical flags, weights, and
maps with the observer enabled versus disabled. Do not add masking in the same
change.
