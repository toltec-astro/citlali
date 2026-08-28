# SCI-PTC To SCI-JINC Boundary

Profile identity: `SCI-PTC_TO_SCI-JINC v0.1/r0.1`

Status: final Stage A boundary candidate; awaiting scientific-owner approval;
numerical JINC route unavailable

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
| JINC-facing coefficient | Exact family/version and PTC owner; generation; detector, sample, or detector-to-sample broadcast index; compatibility with `z_i`; payload availability/cause; statistic and factors; unit; normalization operator/domain; estimation population; support; lifecycle; JINC permission; coefficient/QC profile/evaluation; covariance assumptions; uncertainty; and prohibited interpretations. Availability alone makes no finiteness or positivity claim. |
| PTC transform state | Nonrestored additive reference; fitted correlated removed component; total removed component; realized removed/null-subspace identity; fixed-state null space; full-procedure invariant/unidentifiable modes when claimed; and exact causes. |
| Influence and cause | Direct causes and complete transitive influence preserved without inventing a universal downstream veto. |
| Response | Exact response family and state: fixed-state upstream-to-PTC; PTC full-procedure finite difference; JINC re-resolved procedure; separately authorized whole-chain RTC-to-CAL-to-PTC-to-JINC; another named family; or typed unavailable. A realized PTC-grid companion starts at JINC and receives the JINC operator exactly once. |
| Covariance and uncertainty | Conditional PTC-domain covariance when available, with exact conditioning state, domain/axes, units, support, approximation, omitted terms, response uncertainty, selection/nuisance/parameter uncertainty, and typed unavailable components. Unknown is not zero. |
| Coordinate join | Exact frozen AST role `SCI-AST:rtc_output_grid_coordinates@1` for the same stable RTC `n` and complete parent chain, under [`SCI-AST_TO_SCI-JINC_BOUNDARY.md`](SCI-AST_TO_SCI-JINC_BOUNDARY.md). |
| Admission and failure | Exact `SCI-JINC:upstream_admission@1` evaluation, all input facts and per-predicate outcomes, cause union, missing/conflict behavior, failure scope, and immutable provenance. |

## Conditional Coefficient Route

Frozen PTC defines coefficient declaration obligations but does not select the
JINC-facing family. Therefore:

> No numerical JINC route exists until one exact PTC-owner-selected
> JINC-facing coefficient family is versioned, source-bound, and admitted.

The family must declare every coefficient facet in the table above. SCI-JINC
must not infer unity, the MAP coefficient, `sens`, loading, scatter, inverse
variance, precision, or significance. It must not infer a coefficient from
the signal unit or from a payload with inverse-square units.

JINC separately classifies an authorized coefficient value as finite strictly
positive, exact zero, finite negative, non-finite, or unrepresentable. Only a
finite strictly positive value may enter `omega_i`; other classifications
produce the profile-defined nonmembership or failure with cause. A new
coefficient generation never mutates an earlier transformed product.

## Exact Occurrence And Admission Sequence

For one route candidate, the following remain separately typed:

1. transformed-signal availability;
2. PTC output-retention disposition;
3. coefficient-family/value availability;
4. PTC coefficient/QC disposition;
5. JINC upstream-admission disposition;
6. same-parent AST coordinate validity;
7. finite `z_i`;
8. finite positive `omega_i`;
9. signed-kernel placement and finite `kappa_ip`;
10. cancellation and formal JINC support;
11. required-companion availability; and
12. final JINC product validity.

Passing the upstream profile creates only a JINC route candidate. All
pixel-local gates pass before payload accumulation. A producer flag or cause
has no universal action unless an exact restriction names it.

## Disabled, Missing, And Failure Routes

PTC-disabled terminates on the RTC-terminal export route and supplies no PTC
product or JINC product. Missing positive-rank PTC realization; missing exact
signal/coordinate join; missing coefficient family, value, or QC; incompatible
generation; or unresolved required lifecycle/provenance makes the affected
JINC route unavailable with cause. There is no direct CAL fallback, inferred
no-op PTC, zero substitution, neighboring-product borrowing, coefficient
reconstruction, or generation repair.

## Compatibility And Change

Compatibility requires this exact boundary identity, frozen PTC r0.5
semantics, the same-`n` AST boundary, exact `SCI-JINC:upstream_admission@1`
identity, exact coefficient family and QC profile, and preservation of typed
quantity, causes, response/covariance states, lifecycle and failure scopes.
Any changed quantity, identity, coefficient, response class, profile, or
missing/conflict rule requires a versioned successor. No prior product or
evaluation is rewritten.
