# SCI-MAP coadd aggregate profile v0.2/r0.3 candidate

Profile identity: `SCI-MAP:observation_coadd_admission@2`.

Status: **CANDIDATE successor profile semantics**. Scientific-owner approval,
Registry registration, and source binding are `pending`; activation/
evaluability and object-specific decision realization are `unavailable`. It
supersedes no record until accepted and canonically activated. Historical
`SCI-MAP:observation_coadd_admission@1` and every evaluation under it remain
immutable and are not aliases.

Policy owner: Grant Wilson for SCI-MAP. After exact owner acceptance,
canonical Registry/source binding, and activation, VAL may evaluate this
MAP-authored policy for a requested object; it performs no coadd arithmetic.

Source binding: paired candidate records
`SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-v0.2-r0.3-candidate-2026-09-08` and
`SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-v0.2-r0.3-candidate-2026-09-08`.

## Object, population, and axes

The object is one immutable complete observation MAP bundle, its exact
support-authorized row domain, one exact selected product role, and the target
centered-integer coadd plan. The population is the plan's ordered observation
bundle list before any coadd mutation. The source occurrence profile is exactly
`SCI-MAP:map_upstream_admission@2` under the r0.3 candidate binding proposal.

The four VAL axes remain request, applicability, eligibility, and
decision-artifact realization. Only requested/applicable/eligible/realized
passes. They remain distinct from each selected role's ordered MAP lifecycle
`requested/effective/observation_resolved/applied/realized` and from ECS
requirement results.

## Request and applicability

Request is `requested` only when the accepted effective coadd plan names
`SCI-MAP:base_coadd@1`, `SCI-MAP:response_bearing_coadd@1`, or
`SCI-MAP:covariance_qualified_coadd@1` and centered-integer coaddition;
otherwise it is `not_requested`. Observation-role products are inputs and base
parents, not selectable aggregate outputs. Applicability is `applicable` only
for a complete base/unfiltered observation bundle of the declared
nonpolarimetric quantity and candidate common-grid family. Another product
role is `inapplicable`. Missing
or conflicting identity, source, parent, role, profile, plan, or lifecycle
binding is `applicability_unknown`.

## Required base compatibility

Every selected role requires compatible quantity; `mJy/beam` unit and exact
fixed-nominal-beam identity; realized PTC route and product/application
generation; exact MAP occurrence/source binding; complete AST frame/WCS and
WCS generation; centered-integer shape/reference-pixel relation; support
policy and admitted applied `coverage_cut`; exact equal-observation coefficient
`SCI-MAP:uniform_observation_coadd_coefficient@1`; null/additive-reference and
removed-subspace state; exposure convention; role and base-parent identity;
ordered lifecycle; and immutable parentage.

Different quantity, beam, grid, frame, fractional shift, unauthorized crop or
pad, reprojection, mosaic, incompatible PTC/MAP/profile generation, missing
base-role required fact, or conflicting identity is decisive. All causes are
retained. A complete base-bundle incompatibility rejects the observation at
the declared role scope before mutation.

## Role-qualified companion rules

- `SCI-MAP:base_coadd@1` requires response and covariance identities/statuses,
  but unavailable or basis-incompatible member response and incomplete
  covariance do not change numerical signal membership. Coadd response becomes
  unavailable with exact cause. No hidden response subset, zero fill, or
  assumed covariance independence is permitted.
- `SCI-MAP:response_bearing_coadd@1` binds one exact response family `R` and
  requires every exact member response `R_o^(R)` with compatible source
  domain, perturbation definition, basis, class, unit, normalization,
  reference/gauge/null state, parent, WCS/grid, and row maps. Membership,
  centered-integer placement, `B_out`, support, and coefficients are fixed,
  and an exact composition theorem is required. Mixed families and
  finite-difference-as-Jacobian propagation are prohibited. Coadd-procedure
  and whole-chain responses remain separately unavailable.
- `SCI-MAP:covariance_qualified_coadd@1` requires every within- and
  cross-observation covariance block named by that selected request on exact
  stacked rows. Missing blocks are unknown.

For an already-realized base coadd, a required companion incompatibility
blocks the stronger role and never selects a subset or mutates the parent. A
separately requested stronger-role admission may reject an observation only
before constructing its own explicitly identified base/signal parent. The
resulting parent and stronger product share the identical ordered membership.

## Aggregation, output, and failure

After an eligible realized decision, MAP performs exact centered-integer
placement, equal-observation accumulation with dimensionless `u_op=1`, support
resolution, unique-original exposure union, and role-local validity. Signal,
count, exposure, response, covariance, provenance, and product cardinality
follow one selected atomic admission. The profile authorizes no reverse
propagation, crop, pad, interpolation, reprojection, GLS, or mosaic.

If the applied observation or coadd support policy authorizes no output rows,
the attempt outcome is `not_produced` with cause
`no_support_authorized_output_rows`; there is no realized product, empty
bundle, no-row member, dense zero, or fabricated companion.

Missing required facts with no decisive false yield `decision_unavailable`;
decisive incompatibility yields `ineligible`. Decision-artifact realization
describes only the evaluation artifact. Pre-application failure creates no
applied/realized MAP record. An applied terminal failure retains its exact
attempt and cause without fabricating a realized product.

These rules are prospective. This candidate generation cannot presently be
evaluated and asserts no current applied attempt or realized object decision.
