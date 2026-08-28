# SCI-PTC To SCI-JINC Boundary

Profile identity: `SCI-PTC_TO_SCI-JINC v0.1/r0.3`

Status: ODQ-101/103/107 Stage A successor boundary candidate; exact-byte
owner approval required; numerical JINC route unavailable until a registered,
JINC-permitted family is selected and realized

Prepared: `2026-08-28`

Scientific owner: Grant Wilson

## Exact Authority And Purpose

This boundary binds frozen SCI-PTC v0.1/r0.5, freeze-record SHA-256
`8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`,
to the proposed SCI-JINC v0.1 observation estimator. The exact PTC freeze
promotes commit `8f0ecccfacbdce0543141c4289ec06c702065f5e`.

The ordinary `SCI-PTC_TO_SCI-MAP v0.1/r0.1` boundary is predecessor context
only. It is neither renamed nor inherited. SCI-JINC consumes PTC and AST
parents directly and consumes no SCI-MAP product.

The owner-approved ODQ-101 successor decision uses the exact post-freeze
predecessor object
`54475956f6aefb839d43b2f0fb019a142cb64310:doc/scientific_contracts/packages/SCI-MAP/v0.1/POST_FREEZE_SCIENTIFIC_OWNER_DECISIONS_2026-08-28.md`,
SHA-256
`4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c`,
only under
[`AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md`](AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md).
It establishes controlled successor architecture; it does not edit frozen PTC
r0.5 or MAP r0.7.1.

The handoff is atomic for one exact occurrence `i`. Its identity binds the
observation, detector occurrence/UID, stable RTC output sample `n`, exact PTC
contract and application generation, PTC segment, TolTEC array/group, and
associated time. Time, row, shape, cardinality, detector label, or numerical
coordinate equality cannot establish identity.

## Required Logical Handoff

Every requested occurrence supplies each facet or an exact typed unavailable
state and cause:

| Facet | Required meaning |
| --- | --- |
| PTC authority and lifecycle | Exact PTC contract/revision, request, effective plan, observation-resolved state, learned/fitted evidence, resolved-selected state, applied/realized product, publication state, application generation, and immutable parents. |
| Transformed signal | `z_i=Z_i^PTC`; quantity role, unit, availability/cause, exact originating fixed nominal beam/template identity, CAL/PTC ancestry, and finite-payload classification performed only by the named consumer gate. |
| Output retention | Exact `SCI-PTC:output_retention@1` profile/evaluation identity, request/applicability/eligibility/realization fields, direct causes, scope, and preserved CAL classification. |
| JINC-facing coefficient | Exact single-registry identity/version and PTC owner; exact family/version and explicit `SCI-JINC` permission; requested, effective, observation-resolved and realized selection identities; user selection or exact versioned mode-policy default; generation; detector, sample, or detector-to-sample broadcast index; compatibility with `z_i`; payload availability/cause; statistic and factors; unit; normalization operator/domain; estimation population; support; lifecycle; coefficient/QC profile/evaluation; covariance meaning/assumptions; uncertainty; and prohibited interpretations. Availability alone makes no finiteness or positivity claim. |
| PTC transform state | Nonrestored additive reference; fitted correlated removed component; total removed component; realized removed/null-subspace identity; fixed-state null space; full-procedure invariant/unidentifiable modes when claimed; and exact causes. |
| Influence and cause | Direct causes and complete transitive influence preserved without inventing a universal downstream veto. |
| Response | Exact upstream response family and state remain producer facts when present. ODQ-107 does not authorize a base-v0.1 JINC response product or a response-role availability object. |
| Covariance and uncertainty | Exact upstream covariance/uncertainty meaning remains a producer fact when present. ODQ-107 does not authorize a base-v0.1 JINC covariance, uncertainty or formal-weight product. Unknown is not zero. |
| Coordinate association | Exact frozen AST role `SCI-AST:rtc_output_grid_coordinates@1` associated with the same processed sample realization entering JINC, under [`SCI-AST_TO_SCI-JINC_BOUNDARY.md`](SCI-AST_TO_SCI-JINC_BOUNDARY.md). The scientific association is exact; its data-model realization is not prescribed here. |
| Admission and failure | Exact JINC-owned `SCI-JINC:jinc_map_contribution@1` evaluation and its established input, decision and cause semantics. Ordinary MAP admission/validity and producer-owned JINC-usability decisions do not cross this boundary. These facts do not create a JINC bundle-role availability or provenance product. |

## Approved Registry And Conditional Numerical Route

PTC owns one versioned registry of positive analysis/gridding coefficient
families. Every exact family/version declares permission for named consumers
`SCI-MAP`, `SCI-JINC`, or both. Permission is not transitive between
consumers. The user selects from the exact allowed list; only an explicit
versioned mode policy may provide a default. Requested, effective,
observation-resolved and realized family identities remain distinct.

When an exact family permits SCI-JINC, JINC consumes the same positive
PTC-produced `omega_i` and its separately typed availability/QC, identity,
normalization, support, provenance and covariance meaning. The family must
declare every coefficient facet in the handoff table. JINC does not reproduce
or infer the generating formula.

No numerical JINC route exists until an exact registered family permits
`SCI-JINC`, is selected by the user or an authorized versioned mode default,
and supplies a compatible realized payload and QC state. SCI-JINC must not
infer unity, a MAP-permitted family, `sens`, loading, scatter, inverse
variance, precision or significance. It must not infer a coefficient from the
signal unit, inverse-square units or another family's availability.

JINC separately classifies an authorized coefficient value as finite strictly
positive, exact zero, finite negative, non-finite, or unrepresentable. Only a
finite strictly positive value may enter `omega_i`; other classifications
produce the profile-defined nonmembership or failure with cause. A new
coefficient generation never mutates an earlier transformed product.

SCI-JINC alone applies the signed point-phase coefficient `kappa_ip` and owns
`w_ip=kappa_ip omega_i`, signed normalization, conditioning, support and the
fixed JINC bundle semantics. Response and covariance remain SCI-JINC-owned
scientific questions if a later concrete use authorizes corresponding
products; ODQ-107 does not include them in base v0.1. No MAP projection,
normalization, support, exposure, coadd, response, covariance or validity rule
is inherited.

## Exact Occurrence And Admission Sequence

For one route candidate, the following remain separately typed:

1. transformed-signal availability;
2. PTC output-retention disposition;
3. coefficient-family/value availability;
4. PTC coefficient/QC disposition;
5. JINC map-contribution sample-admission disposition;
6. AST coordinate validity and exact same-processed-sample association;
7. finite `z_i`;
8. finite positive `omega_i`;
9. sample-pixel finite support;
10. signed-kernel placement and finite `kappa_ip`, including normal zero and
    negative coefficients;
11. cancellation and formal JINC support; and
12. final fixed-bundle validity.

Passing the profile admits only the sample for JINC consideration. Sample-
pixel support is a separate JINC decision. All pixel-local gates pass before
payload accumulation, and every coupled accumulator uses the same admitted
sample-pixel pair and coefficient identity. Outside support and contract-
defined zero are ordinary no-contribution results; a finite negative
coefficient is normal. A producer flag or cause has no universal action unless
an exact restriction names it.

## Disabled, Missing, And Failure Routes

PTC-disabled terminates on the RTC-terminal export route and supplies no PTC
product or JINC product. Missing positive-rank PTC realization; unavailable,
duplicate, or ambiguous exact signal-coordinate association; missing
selection with no authorized mode default;
unregistered family; missing `SCI-JINC` permission; missing coefficient value
or QC; unavailable or mismatched payload; incompatible generation; or
unresolved required lifecycle/provenance prevents formation and publication
of the affected complete JINC bundle. There is no direct CAL fallback, inferred no-op PTC,
zero substitution, hidden unity/alternate-family fallback, neighboring-product
borrowing, coefficient reconstruction or generation repair.

## Compatibility And Change

Compatibility requires this exact r0.3 boundary identity, frozen PTC r0.5
semantics plus the controlled ODQ-101 successor, the exact same-processed-
sample AST boundary, exact `SCI-JINC:jinc_map_contribution@1` identity,
registry/family versions,
explicit JINC permission, requested/effective/observation-resolved/realized
selection identities, exact coefficient/QC profile, and preservation of typed
quantity, causes, upstream response/covariance meaning, lifecycle and failure scopes.
Any changed registry, consumer permission, selection/default rule, quantity,
identity, coefficient, response class, profile or missing/conflict rule
requires a versioned successor. No prior product or evaluation is rewritten.
