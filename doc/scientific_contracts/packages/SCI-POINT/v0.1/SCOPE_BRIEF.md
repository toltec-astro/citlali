# SCI-POINT — Bright-Source Pointing Inference Scope Brief

Status: Stage A candidate for scientific-owner review; not approved for Stage B

Scientific owner: Grant Wilson

Version/date: `v0.1`, `2026-09-02`

Starting source identifier:
`codex/sci-point-v0.1-stage-a@0b977a90a0bae6a68dadcf7824c9b7a0c80a7f45`

Approved source identifier: unavailable pending exact owner approval

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Working-wheel dispositions:
  [`WORKING_WHEEL_ADOPTION_REGISTER.md`](WORKING_WHEEL_ADOPTION_REGISTER.md)
- Recovery date: `2026-09-02`
- Existing material proposed for adoption: the mature targeted per-array
  bright-source Pointing fit, its six-parameter Gaussian compatibility model,
  its result vocabulary, and the already governed diagnostic meanings
- Existing material abstracted: frozen predecessor boundaries, map-parent and
  support identity, fit method identity, and measurement/correction separation
- Existing material deferred or excluded: per-detector Beammap fitting,
  blank-field source work, OOF inference, implementation and validation evidence
- Genuinely new scientific work: only the unresolved parent, estimator,
  acceptance, and uncertainty choices identified below; aggregation and the
  correction boundary are now owner-decided
- Owner-approved author references: none yet; proposed sanitized inputs are
  listed separately and are not dispatchable
- Author-packet exclusions: implementation, configuration, schemas, audit
  findings, repairs, tests, reductions, validation, Unity, defaults, historical
  performance, and current production behavior

Opening reviewed before Stage B launch: **no**. This is the owner-review draft.

## 1. Package Name And Scientific Purpose

SCI-POINT defines inference from a known, isolated, bright pointing source in
an observation-local map. Its central scientific quantity is the apparent
source displacement in an exact declared map tangent plane, reported first by
TolTEC array and accompanied by enough model, support, response, uncertainty,
and parent identity to interpret the measurement honestly.

The package exists to formalize the working Pointing fitter, not replace it
with a generic source-analysis system. Its result may later be used by a named
pointing-support producer to construct a correction record, by CAL/TolProj for
an authorized pointing-source photometric workflow, or by other declared
consumers. Those uses do not alter the original measurement or its claims.

## 2. Scientific Boundary

The operation begins with:

- one immutable, observation-local, per-array map product from one exact
  admitted map or transformation method;
- a known source identity and declared expected-location/search relation;
- the map's signal, support/validity, weight or covariance representation,
  WCS/grid/frame, unit/calibration, effective source response, parent, method,
  version, and generation; and
- a complete requested fit method and policy.

It ends with one atomic per-array Pointing measurement bundle containing the
fit result or explicit unavailability. Under owner-approved ODQ-001,
observation-level cross-array aggregation is not a SCI-POINT v0.1 role.

The operation does not begin from a blind field and does not end by applying a
correction to a different observation.

## 3. Legitimate Inputs

Every candidate parent shall identify:

- observation and array;
- exact observation-local MAP, JINC, FLT-FIXED, or FLT-MATCHED
  method/product identity;
- exact terminal FRUIT method/iteration/generation lineage when the admitted
  map product was produced through FRUIT;
- terminal versus intermediate state where applicable;
- signal quantity and unit;
- target WCS, AltAz tangent-plane basis, pixel metric, orientation, handedness,
  and continuous-coordinate convention;
- support, missing/non-finite policy, and exact usable fit domain;
- map weight/covariance meaning and availability;
- effective source response, normalization, null-space, and limitations;
- calibration and beam/template convention;
- exact source identity, assumed morphology, expected position or search
  center, and any position uncertainty; and
- immutable parent/version/generation provenance.

Listing a parent family does not admit it. A directory labelled `raw` or
`filtered`, or a map-shaped object with compatible dimensions, is insufficient
scientific identity.

The ordinary v0.1 frame is proposed to be an AltAz tangent plane with source
displacements in arcseconds, consistent with the current Pointing convention.
RA/Dec spherical-coordinate fitting is not proposed for base v0.1.

## 4. Required Outputs

For each requested array, the candidate package requires either a complete
measurement or an explicit unavailable result. A complete measurement records:

- observation, array, parent, route, method, model, version, and generation;
- fitted amplitude and its signal-unit meaning;
- two-component fitted centroid in the declared tangent basis;
- fitted major/minor effective widths and orientation;
- fit-domain, seed/search, support, validity, boundary, weight/covariance, and
  parameter-constraint identity;
- estimator-status and fit-acceptance state;
- formal parameter uncertainty or explicit unavailability, with covariance
  representation and limitations;
- effective source-response and calibration limitations;
- the legacy amplitude/full-map-RMS dynamic-range value when retained;
- the explicit amplitude/formal-amplitude-error diagnostic when available;
  and
- immutable product/parent provenance and lifecycle state.

The per-array measurement remains valid as its own product even when no
observation-level aggregate or downstream correction is available.

Any observation-level aggregate belongs to the named pointing-support producer,
which must declare its participating arrays, weighting, covariance treatment,
partial-array policy, failure behavior, and exact POINT ancestry. Under
owner-approved ODQ-002, neither a correction candidate nor an applied pointing
correction is a POINT output.

## 5. Upstream And Downstream Responsibilities

| Owner | Retained responsibility |
| --- | --- |
| MAP or JINC | map estimand, gridding, normalization, WCS, support, response/covariance disclosure |
| FLT-FIXED / FLT-MATCHED | exact transformed signal, response, support, covariance, and method state |
| FRUIT | iteration, feedback, terminal-product selection, recurrence response, and restart/lineage |
| ALIGN / AST | detector/sample coordinate realization, frame transformations, pointing-support interpretation/application, and astrometric uncertainty |
| CAL / TolProj | calibrated amplitude meaning and any authorized pointing-derived flux-scale transfer |
| SCI-BEAM | per-detector Beammap fits, effective PSF, sensitivity, and APT products |
| NOI | empirical uncertainty companion or transformed-product uncertainty under its declared method |
| VAL | evaluation and Registry mechanism for exact POINT-owned named-use policy |
| pointing-support producer | cross-array aggregation, selection, sign/user-offset composition, and correction-record publication |

POINT consumes these exact meanings and may not redefine them.

## 6. Externally Imposed Conventions

- Pointing uses AltAz tangent-plane azimuth/elevation offsets in arcseconds for
  base v0.1.
- The WCS and pixel metric of the exact parent govern conversion from fitted
  continuous pixel coordinates to tangent coordinates.
- Array identity is a key, not an assumed row position.
- Requested, effective, and realized fit state are distinct.
- `fitted_amplitude_over_full_map_rms` is the canonical descriptive identity
  for the fitted-amplitude/full-map-RMS dynamic-range diagnostic.
- `sig2noise` is retained only as a legacy alias for that diagnostic and is
  not statistical significance.
- `peak_over_full_map_rms` is unavailable unless the approved compatibility
  method establishes that fitted amplitude is the relevant positive peak for
  the exact source model and parent route.
- `fit_sig2noise` is fitted amplitude divided by formal fitted-amplitude error;
  it is not empirical Gaussian significance or detection probability without
  separate justification and validation.
- Fitted effective widths are properties of the exact parent response and fit
  model. They are not automatically intrinsic telescope beam parameters.
- Formal fit uncertainty, astrometric/correction uncertainty, calibration
  uncertainty, and empirical repeatability are distinct.
- A measured source displacement and a correction to apply have opposite roles;
  the sign and composition transition must be explicit wherever it occurs.

## 7. Questions The Contract Must Answer

1. How shall the contract preserve the owner-approved boundary that base v0.1
   ends at authoritative per-array measurements and cross-array aggregation
   remains downstream?
2. How shall the contract preserve the owner-approved boundary that POINT
   stops at measured displacement while the pointing-support producer owns
   aggregation, sign, user-offset composition, selection, and correction
   records and AST owns application?
3. How shall the contract preserve the owner-approved eligibility of distinct
   observation-local MAP, JINC, FLT-FIXED, and FLT-MATCHED routes, with no
   automatic selection, substitution, equivalence, or fallback and with
   terminal FRUIT retained as lineage on the exact map type?
4. How shall the contract fully state the owner-approved six-parameter
   `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1` method without redesigning
   its estimand or importing implementation text?
5. How shall the contract preserve the owner-approved configurable expected-
   center, weighted-peak initialization, central/global search realization,
   bounded fit domain, and parameter-constraint state without hiding defaults
   or freezing one universal numerical configuration?
6. How shall the contract implement the owner-approved independent per-array
   complete/diagnostic-only/unavailable states without erasing sibling results
   or inventing a POINT-owned whole-observation success state?
7. How shall the contract publish the owner-approved marginal formal errors,
   state unavailable joint covariance without implying zero or independence,
   and attach later uncertainty companions without rewriting the fit result?
8. How shall the contract preserve amplitude and effective shape as required
   fit-result components and telescope/observing-condition quality-control
   metrics while bounding their processed-map, CAL/TolProj, and non-beam
   meanings?
9. How shall the contract preserve the approved separate ownership of POINT
   fit completeness, pointing-support displacement use, telescope/observing
   QC use, and CAL/TolProj amplitude use while VAL only registers/evaluates
   and no aggregate profile enters base v0.1?

## 8. Non-Goals

SCI-POINT does not perform source detection, candidate selection in a blank
field, deblending, cataloging, completeness estimation, or faint distributed-
source inference. It does not perform per-detector Beammap fitting or OOF
optical inference. It does not make or filter maps, run FRUIT, define
calibration, infer an intrinsic beam, create empirical uncertainty, select
bracketing Pointing observations, interpolate corrections, or apply a
correction to science data.

Stage A does not implement, audit, repair, optimize, execute, validate, or
authorize production behavior. It makes no achieved-accuracy, uncertainty-
coverage, response-fidelity, performance, readiness, or Unity claim.

## 9. Allowed References

No Stage B input is owner-approved yet. The proposed future author sees only
the sanitized objects named in
[`PROPOSED_SANITIZED_AUTHOR_INPUTS.md`](PROPOSED_SANITIZED_AUTHOR_INPUTS.md).
The internal dossier, implementation, configuration, schemas, audit findings,
tests, reductions, validation, and operational code are prohibited.

## 10. Owner Decisions And Remaining Ambiguities

The owner has already decided:

- launch SCI-POINT now;
- preserve and recover working prior art before new derivation;
- keep per-detector Beammap fitting in SCI-BEAM; and
- defer blank-field faint-source detection/fitting; and
- end SCI-POINT v0.1 at authoritative per-array measurements, leaving any
  cross-array aggregate to the named pointing-support producer; and
- stop at measured displacement, leaving correction sign, telescope-offset
  composition, selection, publication, and AST application downstream;
- treat a terminal FRUIT result by its exact terminal MAP, JINC, or FLT map
  type while preserving complete FRUIT lineage; FRUIT is not a separate
  POINT parent family; and
- exclude coadd parents from base v0.1; and
- admit observation-local MAP, JINC, FLT-FIXED, and FLT-MATCHED as distinct
  eligible parent families without automatic selection, substitution,
  equivalence, or fallback; exact numerical availability and binding remain
  separate gates; and
- adopt the established six-parameter zero-background elliptical-Gaussian
  Pointing fit as `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1`, with no
  additional profile family in base v0.1; and
- preserve its existing configurable expected-center/central-search,
  weighted-peak initialization, global fallback, bounded fit-domain, and
  parameter-constraint machinery while making every requested, effective,
  and realized state explicit; and
- treat each requested array fit independently as complete, diagnostic-only,
  or unavailable, preserving sibling-array results and leaving partial-array
  aggregate admission to the downstream pointing-support producer; and
- require the established marginal formal errors when available, permit joint
  covariance to be explicitly unavailable without implying zero, diagonal, or
  independent covariance, and allow later uncertainty estimates only as
  separately versioned companions; and
- retain fitted amplitude, widths, and angle as required fit-result components
  and, together with centroid and fit state, as telescope/observing-condition
  quality-control metrics, without promoting amplitude to universal flux or
  effective fitted shape to an intrinsic beam; and
- assign fit-result completeness policy to POINT, correction-construction
  displacement policy to the pointing-support producer, parameter-QC policy
  to the named telescope/observing QC process, and photometric-transfer
  amplitude policy to CAL/TolProj, while VAL registers/evaluates without
  authoring and Stage B defines exact collision-free mechanics for later owner
  approval.

The closed bounded decision set is recorded in
[`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).
The scope and exact author packet still require separate final owner approval.

## 11. Independence Statement

This brief uses implementation knowledge to identify the working wheel and the
real gaps, but does not prescribe source code as the scientific answer. A
future implementation-blind author will receive only exact owner-approved
sanitized inputs. Stage B is not authorized by this draft.
