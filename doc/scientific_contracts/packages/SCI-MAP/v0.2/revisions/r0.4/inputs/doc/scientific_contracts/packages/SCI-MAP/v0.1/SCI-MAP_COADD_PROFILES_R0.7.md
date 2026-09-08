# SCI-MAP v0.1/r0.7.1 Coadd Coefficient And Admission Profiles

Status: owner-approved scientific-policy disposition carried into the r0.7.1
freeze-only errata draft; no
implementation, validation, response-fidelity, readiness, or production claim

Scientific owner: Grant Wilson

## `SCI-MAP:uniform_observation_coadd_coefficient@1`

Canonical identifier spelling is exactly
`SCI-MAP:uniform_observation_coadd_coefficient@1`.

- **Index/domain:** exact observation-output row `(o,p)` after centered-integer
  placement and atomic coadd admission.
- **Value/unit:** `u_op = 1`, dimensionless.
- **Normalization:** per output pixel over the exact set `O_p` of atomically
  admitted observation rows; `Q_p^c = sum_o u_op`.
- **Support/lifecycle:** fixed by the immutable effective coadd plan and valid
  only for the exact admitted observation/product generation and row domain.
- **Meaning:** equal-observation arithmetic averaging.
- **Prohibited meanings:** inverse variance, precision, covariance summary,
  equal noise, empirical weight, exposure, or scientific significance.
- **Uncertainty relation:** covariance is separately typed and, when complete,
  propagates through `B_out C_obs B_out^T`; the coefficient is not derived
  from `C_obs`.

## `SCI-MAP:observation_coadd_admission@1`

Canonical identifier spelling is exactly
`SCI-MAP:observation_coadd_admission@1`.

This is a MAP-authored, VAL-governed aggregate profile. Exact Registry
revision `SCI-VAL_PROFILE_REGISTRY
v0.1/r0.3-map-r0.7.1-2026-08-28` contains the immutable evaluable record;
VAL supplies no policy content.

- **Scientific-policy owner:** Grant Wilson, SCI-MAP scientific-policy owner;
  VAL registers and evaluates but does not author this rule.
- **Object and population:** one immutable complete observation MAP bundle,
  its exact support-authorized row domain, and the target centered-integer
  coadd plan. The source atomic MAP profile is exactly
  `SCI-MAP:map_upstream_admission@2`.
- **Request axis:** `requested` only when the accepted effective plan requests
  the centered-integer base coadd; otherwise `not_requested`.
- **Applicability axis:** `applicable` only for a complete base/unfiltered
  observation bundle on the declared nonpolarimetric quantity and candidate
  common-grid family; `inapplicable` for another product role; and
  `applicability_unknown` for missing or conflicting identity, source, parent,
  profile, or plan binding.
- **Eligibility axis:** `eligible` only when all required restrictions pass;
  `ineligible` on a decisive incompatibility; and `decision_unavailable` when
  no decisive false exists but a required fact is unknown or conflicting.
- **Realization axis:** `realized`, `incomplete`, `failed`, or `not_produced`
  describes the decision artifact, never the eligibility meaning.
- **Source versions and compatibility:** exact SCI-MAP v0.1/r0.7.1 shared
  authority and source manifest; `SCI-MAP:map_upstream_admission@2`; frozen
  SCI-PTC v0.1/r0.5; frozen SCI-CAL v0.1 science r0.5/engineering r0.4;
  frozen SCI-AST v0.1/r0.3; SCI-VAL Core v0.1/r0.3 plus exact Registry revision
  `SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-r0.7.1-2026-08-28` and exact
  source-binding revision `SCI-VAL_SOURCE_BINDING_REGISTER
  v0.1/r0.3-map-r0.7.1-2026-08-28`, SHA-256
  `7b91a324f35196a8c8a6e23c8abbbf5322fc601798e36d4ac821907a6090eadf`.
  A digest or compatibility mismatch is
  unavailable and is not repaired by similar naming.
- **Required compatibility:** quantity role; `mJy/beam` and exact
  fixed-nominal-beam/template identity; realized PTC route and compatible
  product/application generation; exact manifest-bound MAP profile; complete
  AST frame/WCS and identical grid; centered-integer shape/reference-pixel
  relation; support policy and admitted `coverage_cut` state; exact uniform
  coadd coefficient; response/covariance role; null/additive-reference and
  removed-subspace state; exposure convention; required companions; lifecycle;
  and immutable parentage.
- **Decisive exclusions:** different quantity, beam, grid, frame, response
  basis for a response-bearing role, fractional shift, unauthorized crop/pad,
  reprojection, mosaic, incompatible PTC/MAP/profile generation, missing
  required companion, or conflicting identity.
- **Response roles:** `advisory` and unavailable-compatible for
  `SCI-MAP:base_signal_coadd_without_required_response@1`; if any member
  response is unavailable or basis-incompatible, the signal membership may
  remain unchanged while the coadd response is typed unavailable. Response is
  `required_permission` for a response-bearing coadd role, which requires every
  member and exact compatible source domain, basis, family, units,
  normalization, parent, and row identity. No hidden response subset or zero
  response is allowed.
- **Covariance roles:** partial or unavailable covariance is compatible with
  the base signal role when disclosed honestly. A covariance-qualified role
  requires every named within- and cross-observation block before the complete
  numerical equation is claimed. Otherwise covariance is explicitly partial,
  symbolic, summarized, lineage-resolvable, or unavailable; unknown is never
  zero or independence.
- **Exposure/count:** admitted-observation count follows the atomic bundle and
  centered-integer row placement. Physical exposure uses the separate
  unique-original, own-coordinate construction and observation-scoped union;
  causal influence does not duplicate seconds.
- **Missing/conflict and failure:** preserve every cause and reject the entire
  observation before any coadd state changes. No plane or row rescues a bundle
  rejected at observation scope.
- **Aggregation/propagation:** this record authorizes only the stated
  centered-integer aggregate and forward propagation of exact compatible
  companions. It authorizes no reverse propagation, crop, pad, interpolation,
  reprojection, GLS, or mosaic.
- **Consumer action:** after an eligible realized decision, MAP performs the
  exact centered-integer placement, equal-observation accumulation, support
  resolution, original-occurrence exposure union, and MAP-local coadd validity.
  VAL performs none of that arithmetic.
- **Supersession:** any changed source digest, population, role, restriction,
  exception, aggregation rule, or missing/conflict behavior requires a new
  immutable profile version and decision generation.
