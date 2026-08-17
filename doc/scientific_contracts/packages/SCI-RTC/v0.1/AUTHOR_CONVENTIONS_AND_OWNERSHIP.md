# SCI-RTC v0.1 — Sanitized Conventions And Ownership

Status: proposed author reference; scientific-owner approval pending

Prepared: `2026-08-17`

This document contains stable conventions and abstract
producer-transformer-consumer boundaries relevant to raw-timestream
conditioning. It contains no Citlali source behavior, audit finding, repair,
test, validation result, or production-status claim.

Proposed source basis:

- `origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20:doc/SCIENTIFIC_CONVENTIONS.md`,
  content SHA-256
  `1970d7e31ccbcf77f890ea7c0854fde59d25b2fc745f909a74150360605d3049`;
- the approved scientific decisions consolidated in the accompanying
  supersession cover; and
- only the conditional interfaces of SCI-CAL v0.1, SCI-MAP v0.1, frozen
  SCI-BEAM v0.1/r0.3, and approved PTC decisions.

## Capability And Quantity Boundary

- V0.1 concerns Stokes-I primary `xs` only.
- The physical observable, sign, zero/reference, baseline, unit, and valid
  operating domain of `xs` must be declared by the producing authority.
- The primary standardized Beammap detector signal is raw fractional frequency
  shift `Delta f/f`.
- A role that imports an approved CAL operator may carry calibrated
  `mJy/beam`; dimensional unit preservation through a filter does not by itself
  prove preserved point-source amplitude or response.
- Enabled polarimetry and measured R-channel execution are outside v0.1.
- RTC transforms an admitted detector stream. It does not manufacture missing
  timing, coordinate, calibration, beam, or eligibility authority.

## Identity And Ordering

- Observation, Tune, array, network/interface, detector occurrence, selected
  APT row/binding, native row, aligned slot, scan, science interval, context
  interval, conditioned output slot, stage, and product identity are distinct.
- Array identity is one of `a1100`, `a1400`, and `a2000`. Array ID, array
  index, network ID, detector index, and container position are not
  interchangeable.
- A detector index is a zero-based location in the current artifact. Cross-
  artifact identity requires an explicit occurrence/product-scoped binding;
  equal row numbers are insufficient.
- Primary detector timestreams and their sample flags use samples on rows and
  detectors on columns unless a product declares a different ordered shape.
- Native acquisition rows, aligned input slots, and phase-zero output slots do
  not silently renumber one another.
- Science windows, physical scans, processing chunks, filter context, output
  selections, and learned-plan scan sets are separate interval identities.

## Time, Coordinates, And Frames

- RTC consumes an ALIGN-assigned detector-reference grid in seconds with exact
  cadence, phase, mapping, gaps, support, and origin/synthesis state.
- The supported approximately 8-ms rate relationships and `0.5x`, `2x`, and
  `4x` family are software-grid conventions. They do not establish physical
  detector integration-event timing, absolute epoch accuracy, or an
  astrometric correction.
- Pointing, OOF, and Beammap coordinate-dependent controls use a declared AltAz
  tangent plane with azimuth/elevation offsets in arcseconds.
- Science coordinate-dependent controls use a declared equatorial J2000 TAN
  relation. RTC performs no implicit frame conversion.
- Coordinate finiteness is not coordinate validity. Invalid, unavailable,
  wrong-frame, or ambiguously bound coordinates do not become outside-source
  samples.

## Signal, Response, And Statistical Labels

- Calibration factor, RTC temporal/detector-mixing response, PTC response, map
  response, and Beammap effective PSF are different quantities with distinct
  parentage.
- Frozen SCI-BEAM gives legacy `responsivity` no canonical scientific role.
  RTC cross-detector replacement may use only a separately declared
  donor-to-target transfer with exact direction, unit domain, support,
  uncertainty, validity, and provenance.
- A scalar LTI transfer is a special fixed-state, translation-invariant,
  detector-separable interior case. A realized local/factorized detector-time
  response is required when selection, detector mixing, masking, state, edges,
  time variation, or multirate behavior makes the scalar claim false.
- Conditional covariance follows from the admitted input covariance and fixed
  realized operator. Unknown covariance is unavailable; it is not diagonal,
  white, stationary, or zero by default.
- Conditional inverse variance, a scalar PTC coefficient, full precision,
  total uncertainty, response, and confidence are not synonyms.
- Selection, calibration/nuisance, response, and model uncertainty remain
  separate terms. Missing terms block only the claims that require them.
- A reused donor or overlapping temporal support creates correlations; it does
  not create independent exposure.

## Validity, Influence, And Missing State

- Acquisition support, original validity, ALIGN synthesis, numerical
  computability, coordinate validity, flags/causes, operator masks, RTC
  replacement, transitive influence, complete temporal support, response
  status, consumer eligibility, uncertainty availability, and provenance are
  distinct.
- Any output influenced by an ALIGN-synthesized or RTC-replaced cell is
  scientifically ineligible under the approved RTC rule, even when finite.
  Its numerical influence and cause/support remain visible.
- A flag describes a cause; it is not automatically a mask, invalidity,
  weight, or universal eligibility decision.
- A source mask controls an operator. It is not an acquisition fact or
  confidence measure.
- Missing, disabled, automatic, rejected, invalid, failed, unavailable,
  unsupported, and omitted are semantic states. NaN, zero, a negative value,
  an endpoint clamp, a stale state, or a prior observation cannot substitute
  silently.
- Required malformed identity/state or unavailable authority fails the
  affected required product or claim. Optional unavailable detail is
  scientifically inert.

## State And Sampling

- Fixed sampling and optional learned sampling are distinct requested modes.
- Requested, effective, observation-resolved, learned/resolved when
  applicable, and realized state flow one way. Later state never overwrites
  the accepted request.
- The phase-zero decimator selects `y[n]=u[M n]` with exact factor, zero phase,
  input/output rates, support, representative time, flag/influence
  propagation, edge/cardinality rule, and alias response.
- Arithmetic-mean downsampling is not a v0.1 operator.
- Learned sampling resolves maximum safe reduction from an admitted candidate
  set under owner-approved response, alias, sampling, realizability, and
  compatibility constraints. The first applied scope is one immutable common
  observation plan.
- Oversampling is scientifically valid and may be advisory. A learned request
  fails or uses native cadence only under its explicitly selected fallback
  policy.
- A learn-to-apply transfer change is not convergence evidence. Restart must
  restore a state-complete resolved plan and reject incompatible inputs.

## Producer–Transformer–Consumer Responsibilities

### Upstream producers

- **ALIGN** owns the native-to-assigned mapping, assigned grid and cadence,
  scan/gap meaning, synthesis/origin state, mapping response/covariance, and
  physical-timing availability.
- **AST** owns coordinate meaning, frame/topology, detector binding, validity,
  and astrometric uncertainty.
- **SCI-BEAM** owns source Beammap APT production, including scientifically
  accepted `flxscale` and its uncertainty under the frozen BEAM contract; it
  does not supply canonical legacy `responsivity` authority.
- **SCI-CAL** owns application meaning for a selected calibration factor,
  target atmosphere, unit, validity, uncertainty, and lineage. Its present
  scientific authority remains conditional where its owner ledger is open.

### SCI-RTC transformer

- RTC owns selected despike/replacement, source-protection, ordered filtering,
  state/edge/non-finite handling, fixed or learned phase-zero sampling,
  complete conditioned response or unavailability, support/influence, and the
  atomic RTC output bundle.
- Applying an imported upstream quantity does not transfer ownership of its
  physical meaning to RTC.

### Downstream consumers

- **PTC** owns optional correlated-mode cleaning, fitted-state admission,
  processed-sample meaning, scalar coefficient families, and covariance
  status. When PTC is disabled, RTC may be a terminal requested product and
  no PTC or map product follows.
- **VAL** owns reusable sample/detector eligibility policy and cause
  precedence; RTC supplies causal facts.
- **MAP** owns sample-to-map estimation, map support/response/validity, and
  coaddition.
- **BEAM** consumes the conditioned raw Beammap signal/response state for its
  standardized maps and owns beam/source inference; RTC does not own the
  effective PSF.
- **NOI, MAP-003, FLT, SRC/MODE, and FRUIT** own empirical noise, response
  tracer/transfer, map filtering, fit products, and recurrence/restart science.

No consumer may strengthen an unavailable RTC response, covariance, timing,
coordinate, calibration, or eligibility claim.

## Claim Layers

The scientific contract, implementation conformity, representation fidelity,
observational performance, science-impact qualification, and production
readiness are separate claims. An analytic equation, compiled program, test
pass, or existing-use status establishes only its named layer.

If approved, this extract is content-bound in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).
