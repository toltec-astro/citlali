# SCI-FLT v0.1 Stage A Scope Brief

Status: sanitized scientist-readable owner-review packet; Stage B not
authorized

## Program Adherence And Prior-Work Recovery

This Scope Brief follows the Scientific Contract Library Program, pilot
process, and owner-approved downstream roadmap. Recovery and classification
preceded this brief. Implementation, configuration, schema, audit, repair,
test, validation, reduction, and performance material is quarantined outside
this packet.

The prior scientific candidate is not supplied wholesale. Only fixed affine
and convolution identities have been abstracted into
[`AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md`](AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md).
Applicable frozen/approved boundaries are restated in
[`AUTHOR_BOUNDARY_INPUTS.md`](AUTHOR_BOUNDARY_INPUTS.md). Historical
implementation findings, verdicts, repair requirements, and old uncertainty
estimators are excluded.

## Scientific Purpose

Define scientifically typed map-domain transformations and successor products
without treating every operation called a filter as one method. Preserve exact
parent, transformation, response, support, validity, uncertainty, and lifecycle
identity so downstream users know what quantity was produced and what claims
it can support.

## Recommended Contract Split

Before Stage B, split the tranche into:

- `SCI-FLT-DET`, for fully fixed deterministic map-domain transformations; and
- `SCI-FLT-INF`, as a holding tranche for inference-bearing methods.

Within `SCI-FLT-INF`, Wiener transformation, matched/template-amplitude
estimation, source-learned filtering, and data-derived spectral selection
should remain distinct unless the owner establishes that they share the same
estimand, prior/learned-state semantics, response, uncertainty, and lifecycle.

The first proposed Stage B assignment is `SCI-FLT-DET`, not the whole tranche.
That assignment remains blocked on the owner decisions.

## In Scope For `SCI-FLT-DET` v0.1

- exact immutable parent-product identity for MAP or an explicitly admitted
  JINC route;
- a fixed affine operator and fixed convolution as a structured subtype;
- scientifically named method purpose, including any approved fixed low-pass
  subtype;
- kernel/template identity, units, sampling, centering, normalization, support,
  and provenance;
- operator order and whether the parent is an observation or a coadd;
- edge, padding, fill, missing, non-finite, and partial-support behavior;
- stored transformed-amplitude identity and units;
- transformed unit-source response or another exact response/transfer object,
  including honest absence;
- numerical support and scientific validity as distinct products;
- deterministic propagation of an available declared covariance model,
  including off-diagonal consequences and honest absence;
- fixed-state attachment to SCI-NOI uncertainty for the exact transformed
  product; and
- immutable requested/effective/resolved/applied/realized lineage sufficient
  to identify the exact transformation without prescribing an engineering
  schema.

## Deferred To Separate Inference-Bearing Stage A Work

- Wiener estimand, signal/noise model, prior, regularization, learned-state
  source, transfer/response, bias, and uncertainty;
- matched or generalized least-squares template-amplitude estimation;
- source-learned templates, positions, masks, morphologies, or subtraction;
- input-data-derived spectral/mode selection and map-domain destriping;
- per-member relearning and other adaptive uncertainty methods; and
- automatic selection among deterministic and inference-bearing methods.

The deterministic author may name these boundaries but must not derive or
select their science.

## Outside SCI-FLT Ownership

- RTC temporal/timestream filtering and related flags;
- MAP/JINC parent estimator and parent validity;
- CAL absolute calibration, passband/color correction, and calibration
  covariance;
- SCI-NOI generation, uncertainty/covariance inference, inverse conditional
  scale, and standardized-signal semantics;
- Beammap/source-fit amplitudes, morphologies, source positions, and mode-
  specific Pointing/OOF interpretation;
- FRUIT source modeling, recurrence, learning, stopping, restart, and
  downstream admission; and
- VAL policy authorship.

## Required Scientific Outputs Of Future Stage B

If the owner approves `SCI-FLT-DET`, the implementation-blind author must
produce a shared normative core and scientist-facing/formal views that state:

1. exact estimand and parent-product compatibility;
2. operator, parameter, kernel/template, order, domain, and lifecycle identity;
3. units, normalization, response/transfer, and source-imprint assumptions;
4. support, validity, edge/padding/missing/non-finite rules;
5. covariance propagation and honest absence, separated from NOI inference;
6. fixed-state NOI transformation-parity requirements;
7. observation versus coadd product identities and any noncommutation; and
8. explicit edge cases, unavailable states, failure conditions, requirements,
   and testable predictions without implementation knowledge.

## Dependencies And Availability

- SCI-MAP and SCI-JINC parent claims remain exactly as frozen, including any
  unavailable numerical, response, or covariance state.
- SCI-NOI Stage A controls the transformation/uncertainty boundary; its draft
  Stage B material is not an input.
- CAL, Beammap, source/mode, and FRUIT claims remain conditional on their own
  exact authorities.
- No filtered JINC, Wiener, matched, source-learned, destripe, or FRUIT route is
  made available by this brief.

## Owner Decisions Required Before Authorship

The bounded decision set is in
[`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).
The first question is the package split. No future author receives this packet
until all decisions that affect their scope are transcribed into the dedicated
owner-authorship record and the packet hashes are finalized.

## Nonclaims

This Stage A brief does not define final scientific truth, authorize Stage B,
select a numerical route, modify an algorithm, establish implementation
conformity, validate a product, establish calibration or achieved response,
approve performance, or claim readiness, production use, or freeze.
