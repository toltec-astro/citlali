# SCI-BEAM — Detector Beam Inference, Calibration Candidates, QC, And Products Scope Brief

Status: draft Stage A output for scientific-owner review; **not approved**

Scientific owner: Grant Wilson

Proposed version/date: `v0.1`, drafted `2026-08-16`

Approved source identifier: unavailable until owner approval and packet
content-binding

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: Codex manager, `2026-08-16`; scientific-owner review pending
- Existing material adopted for scope: stable Beammap frame/unit/identity
  conventions and current cross-repository ownership boundaries
- Existing material reused conditionally: the TolAPT soft-prior producer
  contract, subject to owner approval of a sanitized extract
- Existing material not found: a dedicated approved implementation-independent
  SCI-BEAM scientific core
- Existing material deferred or excluded: source, audit ledger, raw handoffs,
  repairs, tests, A/B claims, validation, Unity evidence, tracked prior catalogs,
  active ALIGN material, implementation conformity, and production status
- Genuinely new work: derive the BEAM estimand, source/beam model, objective,
  identifiability, covariance, prior role, iteration/convergence, QC, product
  validity, and promotion boundaries
- Proposed author references: this brief; a future sanitized convention and
  ownership extract; two primary TolTEC references; and at most one
  owner-selected methodological analogue
- Author-packet exclusions: [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md), this
  full recovery record, repository sources/interfaces, raw audits and handoffs,
  tests/validation, current A/B or production state, and active ALIGN/AST work

Confirm that this opening was reviewed before launching scientific authorship:
`NO — scientific-owner approval is the current gate`.

## 1. Package Name And Scientific Purpose

**SCI-BEAM — Detector Beam Inference, Calibration Candidates, QC, And
Products** defines how admitted detector-resolved observations of a declared
calibration source support inference of an observation-local source response
and beam model for each detector, how uncertainty and quality state are
assigned, and how the resulting product bundle may be consumed or promoted.

The package must separate measured fit parameters from detector focal-plane
coordinates, telescope pointing, absolute calibration, sensitivity, and later
matched/reference APT products. It must make unavailable or conditional every
claim that depends on upstream identity, coordinates, response, covariance,
calibration, or validity not established by BEAM itself.

## 2. Proposed Scientific Boundary

V0.1 begins with:

- an immutable observation and calibrator identity;
- a declared calibrator brightness/flux model with epoch, spectral/passband
  convention, angular extent, uncertainty, and upstream authority;
- detector-resolved conditioned signal and/or map bundles with stable identity,
  declared unit, response status, support, validity, and provenance;
- an externally supplied sample/map coordinate relation in a declared frame,
  with identity binding, validity, and uncertainty;
- an observation-local APT or equivalent detector metadata input whose role is
  explicit rather than inferred from row position;
- requested/effective/observation-resolved model, support, prior, iteration,
  convergence, and QC policy; and
- optional compatible soft priors with producer identity, frame, unit,
  reliability, covariance/scale, and fallback semantics.

It ends with an atomic observation-local result bundle containing, for every
attempted detector:

- fit attempt and terminal state;
- declared source/beam-model identity and parameters;
- centroid and beam-shape estimates in the admitted relative frame;
- amplitude/normalization estimate and its exact estimand;
- uncertainty/covariance or an explicit unavailable state;
- prior influence and fallback state;
- convergence, diagnostics, QC causes, and BEAM-specific validity;
- a typed calibration/sensitivity candidate state when scientifically defined,
  without automatic promotion to SCI-CAL authority; and
- provenance and compatibility identity sufficient for TolAPT,
  `toltec_beammap`, CAL, and later observations to avoid silent mixing.

V0.1 may include observation-local internal iteration needed to realize this
estimator. It excludes general fruit-loop feedback and restart; source
selection and catalog flux estimation; physical timing and absolute
astrometry; upstream conditioning, validity, and mapmaking; downstream
matched/reference APT construction; generic planet calibration and sensitivity
analysis; and production threshold selection without owner decisions.

## 3. Legitimate Inputs

Legitimate inputs may include:

- a point-like or finite angular calibrator model evaluated under one declared
  bandpass and source convention, with nuisance parameters and uncertainty;
- a detector-resolved signal or map in one declared calibrated or provisional
  unit, plus its response/kernel status and calibration lineage;
- explicit stable detector/acquisition identity and observation role;
- a declared AltAz tangent-plane coordinate system about the source, in
  arcseconds, with WCS/axis sign and handedness externally supplied;
- upstream sample, detector, map, response, and numerical validity states;
- an explicit likelihood/noise/covariance input or a typed approximation whose
  limitations are stated;
- an optional initial APT/reference coordinate with role and validity;
- a soft spatial prior that is compatible in frame, array, network, slot,
  scale, reliability, and version, and that cannot masquerade as exact UID or
  measured-position truth; and
- immutable requested/effective/realized policy and parent identities.

Missing, disabled, invalid, rejected, and unavailable are distinct states.
A finite payload does not by itself admit a detector or parameter.

## 4. Required Outputs

The contract must define, without copying the current table schema:

1. the estimand and model family for source, background, beam, and amplitude;
2. parameter units, reference frame, orientation convention, indexing, and
   normalization;
3. likelihood/objective, admitted support, residual meaning, and covariance
   assumptions;
4. per-detector attempt, candidate, fitted, converged, rejected, invalid, and
   unavailable states with explicit causes;
5. fit covariance and propagated nuisance uncertainty, or exact limitations
   when unavailable;
6. prior identity, influence, compatibility, fallback, and causality;
7. a distinction among fitted source centroid, detector relative coordinate,
   telescope pointing, beam width/shape, and absolute astrometric claims;
8. a distinction among fitted amplitude, source-model flux, response, candidate
   flux factor, promoted calibration, and sensitivity;
9. atomic product identity and stable detector binding across fit, QC, APT,
   map, and optional TOD companions;
10. requested/effective/observation-resolved/realized model and iteration state;
11. product-level validity and failure rules that do not infer authority from
    a flag integer, filename, row order, or finite number; and
12. falsifiable response, uncertainty, convergence, prior, edge, and
    observation-order predictions.

## 5. Upstream And Downstream Responsibilities

- **TolProj/TolTECA photometry boundary** selects the calibrator, estimates the
  per-array source flux/model, and supplies it. BEAM records and uses that input
  but does not select the catalog source.
- **SCI-CAL** owns calibrated signal meaning and promotion of any detector
  calibration factor. BEAM may publish a typed candidate only under a declared
  source/amplitude/response convention and uncertainty.
- **ALIGN/AST** owns sample timing, coordinate relations, detector-coordinate
  truth, pointing correction, absolute placement, and astrometric uncertainty.
  BEAM may infer a relative centroid in an admitted frame but cannot promote it
  to those meanings.
- **RTC/PTC** own conditioned signal, causal validity, response, coefficient,
  and covariance identities. BEAM cannot call an incomplete upstream kernel a
  complete realized beam response.
- **VAL** owns upstream sample/detector eligibility and non-finite policy. BEAM
  owns model-fit admission, diagnostics, and BEAM-specific result validity.
- **SCI-MAP** owns detector map estimator, WCS, support, response companion,
  covariance status, and map validity. BEAM consumes a complete admitted bundle.
- **SCI-BEAM** owns the observation-local source/beam inference, fit state,
  covariance statement, prior influence, internal convergence, QC causes, and
  atomic result bundle defined by this package.
- **TolAPT** owns matched/reference APT construction and producer-side soft
  priors. It may consume BEAM outputs under an explicit artifact contract but
  does not retroactively redefine the observation-local fit.
- **`toltec_beammap`** owns downstream calibration analysis, APT diagnostics
  and updates, planet workflows, and sensitivity utilities. It must preserve
  BEAM identities, states, and limitations.
- **FRUIT** owns general feedback, learning, recurrence, restart, and
  cross-iteration science-product semantics. BEAM's internal estimator loop is
  not a fruit loop.

## 6. Externally Imposed Conventions Proposed For Approval

1. Detector-map coordinates are declared AltAz tangent-plane azimuth/elevation
   offsets about the Beammap source in arcseconds. Persisted WCS determines
   axis sign and handedness.
2. Stable detector identity is explicit. A row, slot, `det_N` label, array, or
   network is not by itself an external detector identity.
3. Signal/kernel units and coefficient/noise units retain the meaning supplied
   by their producers. `mJy/beam` is the current admitted map boundary but does
   not by itself establish absolute calibration.
4. Requested, effective, observation-resolved, and realized states are
   distinct and one-way. A later observation cannot inherit an earlier fit,
   source, flux, prior, convergence, or validity state.
5. A TolAPT prior is soft initialization/gating information at
   array/network/slot level, not exact UID assignment or measured-position
   truth. Blind fallback remains required unless a successor decision changes
   that policy.
6. Relative source centroid, detector focal-plane coordinate, pointing offset,
   and absolute astrometry are distinct quantities.
7. Fitted amplitude, source flux/model, response, candidate conversion factor,
   promoted calibration, and sensitivity are distinct quantities and states.
8. Invalid input is excluded before payload evaluation. Missing, disabled,
   rejected, and unavailable states are not valid zeros.
9. Result publication is atomic across the required BEAM bundle; an optional
   diagnostic cannot substitute for a required scientific companion.
10. Internal BEAM iteration is observation-local estimator state and does not
    authorize general fruit-loop feedback or restart semantics.

These are proposed scope conventions, not yet owner-approved scientific
decisions.

## 7. Questions The Contract Must Answer

### A. Estimand and model

- Is v0.1 a point-source response model, a finite-disc-convolved beam model, or
  a declared family selected by calibrator class?
- Is the base beam circular Gaussian, elliptical Gaussian, Gaussian plus
  background, multi-component beam, or another owner-approved model?
- What amplitude is estimated: peak density, integrated flux, template
  coefficient, or another quantity, and in what normalization?
- Which background terms are identifiable without absorbing filtered modes?
- Which parameterization and orientation convention avoid label or angle
  degeneracies?

### B. Support, likelihood, response, and uncertainty

- What map/TOD samples enter the objective, and how are edges, missing pixels,
  correlated noise, heteroscedastic weights, flags, and non-finite values
  handled?
- What response has already been applied upstream, and when may the fitted
  profile be interpreted as the detector-plus-telescope beam rather than a
  conditioned effective response?
- What likelihood or objective is authorized, and when are reported fit errors
  valid covariance estimates?
- How are calibrator-model, bandpass, finite angular size, atmosphere,
  calibration, coordinate, and response uncertainties propagated or marked
  unavailable?
- What synthetic and limiting cases distinguish parameter bias, uncertainty
  miscalibration, and model inadequacy?

### C. Priors, candidates, iteration, and convergence

- May a prior initialize only, define candidate windows, enter the objective,
  or veto a fit? Which roles may be combined?
- How are prior compatibility and influence recorded, and what guarantees that
  a strong observed peak can defeat a wrong soft prior?
- What blind fallback is required when priors are missing, incompatible, weak,
  or unsuccessful?
- Which changes between locator and measurement phases define one estimator,
  and which would constitute a different scientific operator?
- Is convergence parameter stability, objective stability, detector-set
  stability, response stability, or a conjunction? What are the failure and
  maximum-iteration states?

### D. QC, validity, and products

- Which diagnostics are quantitative scientific validity conditions and which
  are review cues only?
- Are thresholds physical constants, instrument expectations, empirical
  policies, or production choices, and who may change them?
- How are failed, unconverged, low-S/N, prior-dominated, edge-truncated,
  nonphysical, and model-inadequate results represented without collapsing
  causes into a binary flag?
- What is the minimum atomic result bundle, and are fit, QC, APT, map, and TOD
  artifacts different views of one result or separate products?
- What identities and compatibility checks permit combining observations or
  using one observation as a prior for another?

### E. Calibration and consumers

- Under what exact amplitude and source-model convention may BEAM form a
  candidate detector conversion factor?
- Which nuisance terms and covariance must accompany that candidate before CAL
  can consider promotion?
- Is sensitivity a BEAM measurement, a downstream combination with noise and
  observing conditions, or only a carried diagnostic in v0.1?
- Which BEAM outputs may TolAPT use for matching/reference construction, and
  which may `toltec_beammap` update without changing BEAM authority?

## 8. Edge Cases And Falsifiable Predictions

The author must include at least:

- noiseless circular and elliptical model recovery;
- axis swap and orientation periodicity;
- constant background and admissible gradient/background cases;
- finite calibrator angular size approaching zero and approaching beam scale;
- cropped/edge source, masked core, disconnected support, and zero support;
- wrong, broad, weak, missing, incompatible, and strongly conflicting priors;
- duplicate/permuted detector rows and slot labels with stable UID binding;
- missing or incomplete response/kernel state;
- correlated noise and deliberately misspecified diagonal covariance;
- non-convergence, alternating candidates, and maximum-iteration termination;
- repeated observation order and sequential-run independence;
- candidate calibration under source-model and fit-amplitude uncertainty; and
- identical numerical payloads carrying different validity or authority states.

Each case needs a predicted invariant, failure state, or explicitly unavailable
claim—not merely a test suggestion.

## 9. Proposed Author Deliverables

After approval, a fresh implementation-blind author shall produce:

1. the shared normative LaTeX modules;
2. a science-team rationale following the library house standard;
3. an engineering conformance view generated from the same authority;
4. a complete requirement and edge/prediction crosswalk;
5. a scientific-owner decision ledger separating author choices from owner
   policy choices; and
6. two rendered PDFs subjected to mechanical and full visual QA.

The author must state dependencies and unavailable claims rather than inspect
software, infer current behavior, or propose repairs.

## 10. Scientific-Owner Approval Gate

Before Stage B, Grant must:

- approve or revise sections 1--6;
- disposition `BEAM-SCOPE-Q001--Q012` in the owner ledger;
- select the exact primary literature references and approve a content-bound
  sanitized convention/ownership extract;
- decide whether CAL/MAP interface extracts may enter the packet despite their
  still-open authority questions; and
- confirm the information-firewall exclusions.

Until then, this brief is a proposal. No author packet manifest or author task
may claim approval.
