# SCI-FLT-FIXED v0.1 SCI-VAL Profile Drafts

Status: owner-policy drafts for future immutable Registry binding; not
registered and not evaluable as a current numerical route

VAL binds and evaluates these FLT-owned policies. It authors neither producer
facts nor FLT policy and performs no transformation.

## `SCI-FLT-FIXED:input_admission@1`

- **Policy owner:** SCI-FLT-FIXED scientific owner.
- **Object:** one exact immutable MAP observation, MAP coadd, or JINC
  observation parent plus one exact resolved fixed-linear plan.
- **Request:** `requested` only for an accepted plan explicitly requesting
  SCI-FLT-FIXED; otherwise `not_requested`.
- **Applicability:** `applicable` only for one supported parent role and the
  strict-linear same-grid full-footprint method; another role is
  `inapplicable`; missing/conflicting identity is `applicability_unknown`.
- **Eligibility:** `eligible` only when package/version/generation, quantity,
  units/beam, WCS/grid/metric/shape, parent row domain, support/validity,
  response/covariance availability state, exposure/influence, operator and
  parameters, normalization, edge rule, lifecycle, failure, and provenance
  meet the exact applicable boundary.
- **Decision unavailable:** used when no decisive false exists but a required
  fact is unknown, unavailable, or conflicting.
- **Decisive exclusions:** approximate WCS, reprojection/resampling,
  incomplete parent bundle, affine term, data-derived state, unsupported edge
  method, inferred parameter/default, or incompatible generation.
- **Consumer action:** after an eligible realized decision, SCI-FLT-FIXED may
  apply the exact resolved operator. VAL performs no arithmetic.
- **Missing/conflict behavior:** fail closed for the affected route and retain
  exact causes.
- **Supersession:** any change to object, role, source binding, operator family,
  restriction, exception, or failure action requires a new immutable profile.

## `SCI-FLT-FIXED:output_publication@1`

- **Policy owner:** SCI-FLT-FIXED scientific owner.
- **Object:** one realized atomic SCI-FLT-FIXED product bundle and its exact
  admitted parent/operator decision.
- **Request:** `requested` only when publication of that exact product role is
  requested; otherwise `not_requested`.
- **Applicability:** limited to the strict-linear same-grid full-footprint
  product; another method is `inapplicable`; missing/conflicting identity is
  `applicability_unknown`.
- **Eligibility:** requires transformed signal on exact `S_out`; exact
  operator/parameter, parent/output WCS, units/beam, transfer state, transformed
  response/covariance availability, null/mode state, influence, support,
  FLT-local validity, edge state, lifecycle, causes, provenance, and failure.
- **Unavailable companions:** a base signal role may publish an honestly
  unavailable response/covariance state only when the exact role permits it;
  a response- or covariance-qualified role requires the complete named object.
- **Disabled state:** produces no FLT product and is not publication-eligible.
- **Identity/zero state:** a realized identity or zero operator is a real,
  separately parented FLT product and is not disabled.
- **Consumer action:** after an eligible realized decision, SCI-FLT-FIXED may
  publish the exact atomic bundle. This does not authorize any downstream
  source, BEAM, NOI, catalog, Pointing, OOF, or FRUIT use.
- **Missing/conflict behavior:** no partial bundle or inferred placeholder;
  retain exact causes and fail the required publication.
- **Supersession:** any changed role, companion policy, source digest,
  transformation generation, support/validity rule, or failure action requires
  a new immutable profile.

## Registration Gate

These drafts require exact source hashes, owner approval, and new immutable
SCI-VAL Registry/source-binding successors before they can be evaluated. Such
registration would not make a missing MAP/JINC parent, operator realization,
response/covariance object, implementation, validation, or production state
available.
