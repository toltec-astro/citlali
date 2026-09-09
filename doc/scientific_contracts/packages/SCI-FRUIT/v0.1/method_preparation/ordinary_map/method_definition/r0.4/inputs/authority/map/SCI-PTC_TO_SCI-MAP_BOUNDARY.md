# SCI-PTC To SCI-MAP Boundary

Profile identity: `SCI-PTC_TO_SCI-MAP v0.1/r0.1`

Canonical identifier spelling is exactly `SCI-PTC_TO_SCI-MAP v0.1/r0.1`;
the single space before `v0.1/r0.1` is part of the identity.

Status: targeted owner-review boundary authority; implementation conformity,
validation, performance, readiness, and production authorization not assessed

Prepared: `2026-08-28`

Scientific owner: Grant Wilson

## Purpose

This boundary defines the representation-independent logical handoff from the
frozen SCI-PTC v0.1/r0.5 transformed product to SCI-MAP v0.1/r0.7.1. It neither
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
| MAP coefficient | Exact family/version and owner; generation; detector or detector-time index; broadcast relation; compatibility with the transformed PTC product; payload presence/readability and typed availability/cause; statistic and all factors; unit; normalization operator/domain; estimation population; support; lifecycle; coefficient/QC decision under the exact PTC-owned profile; relation to retention; uncertainty; and prohibited meanings. Availability makes no finiteness claim |
| PTC transform state | Nonrestored additive reference `lambda`; fitted correlated removed component `Uhat_corr`; total removed component `U_total,Theta`; realized removed-subspace identity; fixed-state null space; full-procedure invariant/unidentifiable modes when claimed; and causes |
| Influence | Direct causes and the complete transitive influence relation, preserved without inventing a universal downstream veto |
| Response | One exact class: complete fixed-state upstream-to-PTC response, PTC full-procedure finite-difference family with separately typed state-change record, PTC+MAP re-resolved procedure response, separately authorized whole-chain RTC-to-CAL-to-PTC-to-MAP response, another named class, or typed unavailable. The whole-chain class is unavailable and not claimed here. A fixed-state object binds source domain, PTC-sample codomain, units, parent, membership, support, coefficient state, response basis, and limitations. A realized PTC-grid companion starts at MAP and receives the MAP operator exactly once, never the upstream response again |
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
Coefficient availability establishes the exact family/generation,
index/broadcast, product compatibility, readable payload, typed state, and
cause; it does not require finiteness. MAP alone classifies an authorized value
as finite strictly positive, exact zero, negative finite, non-finite, or
unrepresentable. Cross-generation use requires an explicit compatibility record binding both
generations and preserving the original product claims.

## Admission And Contribution

`SCI-MAP:map_upstream_admission@2` is MAP-owned, registered by VAL, and
evaluated by VAL Core. An eligible result creates only a MAP-route candidate.
For pixel `p`, MAP separately requires signal availability, PTC retention,
coefficient availability, a PTC coefficient/QC decision that is requested,
applicable, eligible, and realized under its exact PTC-owned profile, separate
MAP admission, same-parent AST validity,
finite `z_i`, finite positive coefficient, admitted one-hot placement/boundary,
and MAP-local companion/contribution gates. MAP admission is never replaced or
rescued by the PTC coefficient/QC proposition. Evaluation proceeds through
structural binding, authorized payload retrieval/classification,
coordinate/product gates, membership, and then accumulation; all gates pass
before payload arithmetic. A producer flag or cause has no universal MAP action.

## Exposure Carriage

ALIGN owns physical acquired `e_acq` and valid-original `e_vo` seconds on
stable original occurrences. RTC, CAL, and PTC preserve them and their causes.
MAP owns upstream-eligible and retained original-footprint exposure and coadded
original-footprint exposure under its
exact admission and placement. It deduplicates by observation, detector
occurrence/UID, stable original/native occurrence, stable ALIGN slot, and
ALIGN mapping generation. Each positive-`e_vo` original is placed exactly once
through `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`, using
that original's layered AST ALIGN-grid direction/tangent/continuous-pixel
parent in the exact target MAP WCS and the one-hot half-open rule. A descendant
RTC-output coordinate is never substituted. An original may
causally influence several RTC/PTC outputs, but those edges neither relocate
nor duplicate physical seconds. Invalid-original contributes zero `e_vo`;
synthesis, replacement, reused donors, overlapping filters, and decimation
create no exposure. Outer-boundary loss contributes nowhere. Coadd exposure
unions observation-scoped original identities after atomic admission. Exposure
is not reconstructed from duration, cadence, sample count, hits, `Q`,
coefficient, or formal precision. These planes are original acquisition
footprints, not complete temporal support, effective integration time,
precision, or normalized-map influence.

## Disabled, Missing, And Failure Routes

PTC-disabled terminates on the RTC-terminal export route. It supplies no PTC
product and therefore no MAP product. Missing positive-rank PTC realization,
missing exact signal/coordinate join, missing coefficient family/value/QC,
or unresolved required lifecycle/provenance makes the affected MAP route
unavailable with cause. There is no direct CAL fallback, inferred no-op PTC,
zero substitution, neighboring-product borrowing, or generation repair.

## Compatibility And Change

Compatibility requires this exact boundary identity, frozen PTC r0.5
semantics, the exact registered MAP profile `SCI-MAP:map_upstream_admission@2`,
the same-`n` AST role, exact CAL
nominal-beam lineage, the exposure-lineage boundary, and preservation of typed
causes, response/covariance states, lifecycle, and failure scopes. Any changed
quantity, identity, coefficient, response class, exposure convention, profile,
or missing/conflict rule requires a versioned successor. No earlier product or
evaluation is rewritten.
