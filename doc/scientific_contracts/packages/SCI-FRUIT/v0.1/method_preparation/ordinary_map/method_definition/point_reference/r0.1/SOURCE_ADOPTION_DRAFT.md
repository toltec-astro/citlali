# Controlled POINT boundary amendment draft

Identity: `SCI-FRUIT-POINT-BOUNDARY-DRAFT@r0.1`, 2026-09-10.
State: **proposed exact text; not adopted upstream authority**.

## Program adherence and prior-work recovery

Follow the [charter](../../r0.4/inputs/program/README.md),
[reviewed recovery](../../r0.4/PRIOR_WORK.md),
[Q02-A/B decision with S1](../../q02_review/r0.2/README.md), and
[limited POINT-reference acceptance](../../POINT_REFERENCE_BRIEF_ACCEPTANCE_2026-09-10.md).
No new scientific-substance vote on A/B/C is requested. This is the controlled
text and binding proposal needed to carry those decisions into future sources.
The existing PTC/CAL/MAP/VAL sources and profiles remain unchanged.

## Exact construction and applicability

This draft consists of the exact [B01–B12 predecessor](../../r0.4/BOUNDARY_AMENDMENTS.md)
(SHA-256 `528ca373cf70c480f97280de4a47a160d93992fc3f550f74bfd49b62f21daf28`),
with the explicit scoped replacements and additions below. All other clause
text is inherited unchanged as proposed text. The versioned [profile draft](PROFILE_BINDING_DRAFT.json)
is part of this proposal. Predecessor proposal labels remain traceable;
this draft identity distinguishes the POINT restriction and S1 correction.

For every new permission in B01–B12, applicability shall additionally require
the accepted initial POINT reference scope and an exact separately approved
method instance. This does not create permission merely because a request is
called POINT. It shall bind the exact calibrated-x quantity, nominal beam,
correction space, occurrence ancestry, AST association, target WCS, all support
roles and required current producer decisions. Actual mode/input applicability
under B12 is still unresolved; no mode-name shortcut is supplied here.

Bootstrap retains the ordinary CAL → PTC branch. A feedback pass uses the
explicit residual route. These routes and their decision generations shall
never be interchangeable. The draft does not authorize numerical simulation,
empirical input populations, full-procedure response execution or production.

## S1: replacement for B05's second paragraph

For this explicit residual route, output-retention predicates and four-axis
decisions shall apply to the residual-derived PTC product with its actual
parents and causes under the **new immutable**
`SCI-PTC:output_retention@2` candidate. Residual basis fitting, loading fitting
where used, operator application and requested conditional response companions
shall likewise use the applicable candidate @2 versions in the profile draft.
The ordinary @1 records and their prior decisions shall remain unchanged.
No residual result may borrow an ordinary-route decision or present an @1
decision as proof of changed-domain permission. CAL classification, direct
origin, model influence and every required failure cause shall persist.
Retention shall establish neither MAP membership nor independence, sky truth,
model fitness or stronger response/uncertainty.

The first paragraph of B05, including linear-part response and data-equivalent
guards, remains unchanged. A requested companion needs its own named-use
decision; output retention does not provide that permission.

## Current-decision additions to B09–B11

Every reference to current PTC retention shall bind ordinary @1 at bootstrap
or residual @2 at k≥1, with the actual source/profile version and current
product generation. Current QC shall bind that decision, not an old bootstrap
eligibility result. The uniform gamma values may retain their original value
generation only with a new retained-use record and current QC. Residual PTC
output and FRUIT-rejoined output remain distinct products.

For `SCI-MAP:map_upstream_admission@3`, the ordinary route preserves the exact
@2 predicates and applicable ordinary PTC @1 records. The added POINT-rejoined
route shall require current residual `output_retention@2`, the exact uniform
family, requested/applicable/eligible/realized current QC, and all B01–B10
compatibility and lineage facts. All inherited direct synthesized/replaced
representative vetoes and CAL engineering classification remain. Transitive
model influence is recorded and is not a universal direct-origin veto.

MAP owns this predicate; VAL only registers/evaluates the owner proposition.
Missing or conflicting applicability/source authority gives unavailable
admission, not permission. Required false predicates give ineligibility;
unresolved required facts cannot become eligible. Realization remains an
independent axis. MAP's own finite-positive classification and local support
gates remain additional requirements, not consequences of producer QC.

## Clause-by-clause disposition

| Clause | Exact treatment in this draft |
| --- | --- |
| B01–B02 | Preserve residual-pass permission and actual-parent/typed identity requirements; add POINT applicability above |
| B03–B04 | Preserve one current-residual fit, positive configured rank, identity scaling, binary centering/nonrestoration and complete affine Apply. Q03 values remain separate |
| B05 | First paragraph unchanged; replace second paragraph with S1 above |
| B06 | Preserve requirement-ID crosswalk; changed source/domain uses the candidate @2 profiles, never ordinary @1 decisions |
| B07 | All ordinary REQ-099 exclusions remain; only the explicitly scoped residual parent and outer invocation are added. No simulation, empirical-population or full-procedure numerical response permission |
| B08 | Preserve exact correction difference space, matching containing pixels, scale one/no added offset, residual-only outside D_a and unknown-versus-rejection distinction |
| B09 | Preserve constant occurrence family, four-axis fresh QC, retained-use and forbidden precision meanings; add explicit current @1/@2 binding above |
| B10 | Preserve proposed boundary v0.1/r0.2, separate rejoined identity and original exposure carriage; add current @1/@2 binding above |
| B11 | Preserve proposed MAP @3 and all old-route predicates; add POINT and exact @2 residual-profile dependencies above. Ordinary MAP arithmetic is unchanged |
| B12 | Preserve unresolved actual mode/WCS/input registration. Only POINT numerical reference scope is approved in substance; no OOF/BEAM/SCIENCE route is inferred |

The PTC common named-use semantics fragment remains byte-identical, identity
`SCI-PTC-COMMON-NAMED-USE-SEMANTICS-v0.1/r0.1`, SHA-256
`c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c`.
It contains no parent-domain or estimator permission. Its reuse does not
license reuse of changed-domain @1 profiles or decisions.

## Source-release bindings still required before use

| Owner surface | Required controlled successor content / current state |
| --- | --- |
| PTC source | B01–B07 and B09, exact residual profiles and uniform-family/QC declarations. Draft above; a published reviewed source counterpart is unavailable |
| MAP source / boundary | B08 compatibility, B10 boundary v0.1/r0.2 and B11 @3, plus exact POINT B12 applicability. Draft above; actual WCS/input binding and published counterpart unavailable |
| VAL registry | Five candidate PTC @2 records, uniform QC candidate and MAP @3, with exact adopted owner sources. Profile draft supplied; no key is registered by this packet |
| VAL source register | Bind final PTC/MAP/boundary/common-fragment hashes and adopted owner decisions; no floating latest source and no fabricated future hash. Existing register unchanged |
| FRUIT method record | Bind adopted source/profile keys and complete numerical effective plan, input, evidence and execution decision. Present record exposes the remaining unavailable slots |

Final owner-source counterparts, registry/source rows, release manifests and
POINT applicability must agree before numerical use. Their final identities
are unavailable until controlled review/adoption; the hashes of this draft
do not substitute for them. Changes to scientific method, gate, population,
input or scope return to the owner. Routine defects inside a later authorized
experiment remain governed by the standing repair direction.
