# Frozen JINC coefficient handoff constraints

## Source cover

J01's controlling SCI-JINC v0.1/r0.3 freeze manifest binds the exact J04
boundary, including its inherited coefficient framework. The boundary's
retained candidate labels are historical. Only the excerpts below are
admitted. JINC consumes the positive PTC value and separately owns its signed
spatial coefficient; no numerical JINC kernel design is part of this task.
JINC permission is independent of MAP permission. Other JINC bundle,
coordinate and local numerical gates remain constraints, not new derivation
scope. A proposed coefficient successor must not promise route availability
or new products before all independently governed prerequisites hold.

## Exact occurrence identity

Source: J04; whole-source SHA-256 `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e`.

<!-- EXCERPT JINC_BOUNDARY-Exact occurrence identity -->
The handoff is atomic for one exact occurrence `i`. Its identity binds the
observation, detector occurrence/UID, stable RTC output sample `n`, exact PTC
contract and application generation, PTC segment, TolTEC array/group, and
associated time. Time, row, shape, cardinality, detector label, or numerical
coordinate equality cannot establish identity.

<!-- END EXCERPT JINC_BOUNDARY-Exact occurrence identity -->

## Required logical handoff

Source: J04; whole-source SHA-256 `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e`.

<!-- EXCERPT JINC_BOUNDARY-Required logical handoff -->
```markdown
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

```
<!-- END EXCERPT JINC_BOUNDARY-Required logical handoff -->

## Approved registry and conditional route

Source: J04; whole-source SHA-256 `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e`.

<!-- EXCERPT JINC_BOUNDARY-Approved registry and conditional route -->
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

<!-- END EXCERPT JINC_BOUNDARY-Approved registry and conditional route -->

## Disabled routes and change

Source: J04; whole-source SHA-256 `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e`.

<!-- EXCERPT JINC_BOUNDARY-Disabled routes and change -->
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
<!-- END EXCERPT JINC_BOUNDARY-Disabled routes and change -->
