# SCI-PTC To SCI-MAP Boundary

Profile identity: `SCI-PTC_TO_SCI-MAP v0.1/r0.1`

Status: targeted owner-review boundary authority; implementation conformity,
validation, performance, readiness, and production authorization not assessed

Prepared: `2026-08-27`

Scientific owner: Grant Wilson

## Purpose

This boundary defines the representation-independent logical handoff from the
frozen SCI-PTC v0.1/r0.5 transformed product to SCI-MAP v0.1/r0.5. It neither
prescribes a class or persistence format nor authorizes a numerical MAP while
an exact PTC MAP-facing coefficient family or another hard gate remains open.

The handoff is atomic at one exact occurrence. `i` binds the observation,
detector occurrence and UID, stable RTC output sample `n`, exact PTC
product/application generation, PTC segment, and array/network/group. Sample
time is an attribute. Column, row, and container positions are locators only.

## Logical Handoff

Every requested occurrence supplies or explicitly types unavailable:

| Facet | Required meaning |
| --- | --- |
| Product and lifecycle | Exact PTC contract/revision, request, effective plan, observation-resolved state, learned/fitted evidence, resolved-selected state, applied/realized product, publication state, application generation, and immutable parents |
| Signal | `z_i = Z_i^PTC`, its availability/cause, unit, quantity role, exact originating fixed-nominal-beam/template identity, and CAL/PTC ancestry |
| Retention | Exact `SCI-PTC:output_retention@1` profile/evaluation identity, decision, direct causes, scope, and preserved CAL classification |
| MAP coefficient | Exact family/version and owner; detector or detector-time index; broadcast relation; value and availability; statistic and all factors; unit; normalization operator/domain; estimation population; support; lifecycle/generation; permitted MAP use; coefficient/QC decision; relation to retention; uncertainty; and prohibited meanings |
| PTC transform state | Nonrestored additive reference `lambda`; fitted correlated removed component `Uhat_corr`; total removed component `U_total,Theta`; realized removed-subspace identity; fixed-state null space; full-procedure invariant/unidentifiable modes when claimed; and causes |
| Influence | Direct causes and the complete transitive influence relation, preserved without inventing a universal downstream veto |
| Response | One exact class: complete fixed-state upstream-to-PTC response, PTC full-procedure response, whole-chain injection, another named class, or typed unavailable. A fixed-state companion includes exact domain, parent, operator state, membership, support, and limitations. A realized PTC-grid companion starts at MAP and is not passed through upstream response again |
| Uncertainty | Conditional PTC-domain covariance when available, exact conditioning/state/domain/axes/units/support/approximation and omitted terms, response uncertainty, selection/nuisance uncertainty status, and typed unavailable terms. Unknown is not zero |
| Coordinate join | Exact `SCI-AST:rtc_output_grid_coordinates@1` result for the same stable `n` and complete parent chain, with coordinate validity/causes and complete frame identity. Time, row, shape, and coordinate equality cannot establish the join |
| Exposure | Exact original-occurrence lineage carrying ALIGN-owned `e_acq` and `e_vo` under `TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY v0.1/r0.1`; accounting unit/convention, deduplication identity, availability, and causes |
| Policy and failure | Exact MAP admission profile/version/evaluation, all input facts and per-predicate outcomes, cause union, missing/conflict behavior, failure scope, and immutable provenance |

## Quantity Boundary

The ordinary signal is a calibrated-`x`-derived nonpolarimetric
total-intensity-equivalent detector-time quantity in the inherited
top-of-atmosphere, point-source-equivalent `mJy/beam` convention. Its identity
includes the originating fixed nominal beam/template and spectral/calibration
lineage. Neither the unit nor a STOKES token establishes Stokes I. A future
Stokes-I MAP role requires separate scientific authority.

## Coefficient Disposition

Frozen PTC r0.5 defines the coefficient type and declaration obligations in
SCI-PTC-REQ-052--055 but leaves selection open in `PTC-OD-010`. Consequently
this boundary carries a typed coefficient slot but selects no numerical family.
Until the owner selects and versions one exact family, coefficient availability
is unresolved and the ordinary numerical MAP route is unavailable. MAP shall
not infer unity, loading, `sens`, scatter, inverse variance, or precision.

A new coefficient generation never mutates an earlier transformed product.
Cross-generation use requires an explicit compatibility record binding both
generations and preserving the original product claims.

## Admission And Contribution

`SCI-MAP:map_upstream_admission@1` is MAP-owned, registered by VAL, and
evaluated by VAL Core. An eligible result creates only a MAP-route candidate.
For pixel `p`, MAP separately requires signal availability, PTC retention,
coefficient availability and QC, MAP admission, same-parent AST validity,
finite `z_i`, finite positive coefficient, admitted one-hot placement/boundary,
and MAP-local companion/contribution gates. Every gate is screened before
payload evaluation. A producer flag or cause has no universal MAP action.

## Exposure Carriage

ALIGN owns physical acquired `e_acq` and valid-original `e_vo` seconds on
stable original occurrences. RTC, CAL, and PTC preserve them and their causes.
MAP owns upstream-eligible, retained, projected, and coadded exposure under its
exact admission and placement. It unions and deduplicates original-occurrence
parents; invalid-original contributes zero `e_vo`; synthesis, replacement,
reused donors, and overlapping support create no new exposure. The one-hot
projection applies; outer-boundary loss contributes nowhere. Exposure is not
reconstructed from duration, cadence, sample count, hits, `Q`, coefficient, or
formal precision.

## Disabled, Missing, And Failure Routes

PTC-disabled terminates on the RTC-terminal export route. It supplies no PTC
product and therefore no MAP product. Missing positive-rank PTC realization,
missing exact signal/coordinate join, missing coefficient family/value/QC,
or unresolved required lifecycle/provenance makes the affected MAP route
unavailable with cause. There is no direct CAL fallback, inferred no-op PTC,
zero substitution, neighboring-product borrowing, or generation repair.

## Compatibility And Change

Compatibility requires this exact boundary identity, frozen PTC r0.5
semantics, the exact registered MAP profile, the same-`n` AST role, exact CAL
nominal-beam lineage, the exposure-lineage boundary, and preservation of typed
causes, response/covariance states, lifecycle, and failure scopes. Any changed
quantity, identity, coefficient, response class, exposure convention, profile,
or missing/conflict rule requires a versioned successor. No earlier product or
evaluation is rewritten.
