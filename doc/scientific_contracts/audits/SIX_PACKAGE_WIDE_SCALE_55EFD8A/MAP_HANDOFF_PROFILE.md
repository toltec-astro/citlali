# Draft ALIGN–AST–RTC–CAL–PTC–VAL to MAP Handoff Profile

Status: **normative-in-spirit, non-authoritative draft**
Pinned library revision: `55efd8a54464636a24e621f6d1b60486d235b20e`
Proposed profile identity: `SCI-MAP-HANDOFF/ordinary-xs-stokes-I-mJy-beam/draft-0.1`
Authority effect: **none until scientific-owner approval and registry binding**

## 1. Purpose and force

This draft states the minimum cross-package handoff that would be needed to
admit the ordinary positive-coefficient SCI-MAP v0.1 route without inventing
missing upstream science. It is deliberately written with normative terms so
that it can be reviewed as a candidate profile. It is not itself a scientific
contract, owner decision, profile registration, implementation prescription,
or conformance result.

`MUST`, `SHALL`, `MUST NOT`, and `SHALL NOT` below express proposed admission
gates. They acquire authority only through an owner-approved successor that
binds an exact profile version to exact package and external-authority
identities.

The governing safety rule is:

> **No numerical SCI-MAP route exists unless every prerequisite in the
> selected, version-bound profile is present, mutually consistent, and
> authoritative for the same sample occurrences and observation.**

Missing authority is not an automatic value, a default coefficient, a zero
response, an empty mask, an inferred coordinate, or permission to bypass a
producer. The route stops before contribution-set construction or payload
arithmetic.

## 2. Authority boundary

This profile composes the independently extracted ALIGN, AST, RTC, CAL, PTC,
and VAL findings at the pinned revision and tests them against the admitted MAP
downstream reference. It does not certify any implementation, test,
configuration, schema, generated product, external package, validation result,
or prior audit. A primary-package guarantee is claimed only where its exact
admitted source supplies it; TEL, APT, BEAM, and other outside facts remain
external dependencies, and every source, binding, profile, or owner gap remains
explicit.

SCI-MAP remains the authority for its ordinary sample-to-map transformation,
MAP-specific support and final validity, complete raw-map bundle, and atomic
centered-integer observation coadd. It does not acquire authority to repair or
reinterpret upstream quantity, coordinate, calibration, coefficient,
eligibility, response, or uncertainty meaning.

## 3. Profile and version binding

An admitted execution SHALL carry one immutable profile record with at least:

| Field | Required binding |
|---|---|
| `profile_id`, `profile_revision` | Registry identity and immutable revision; no unversioned `auto` profile |
| `scientific_library_revision` | Exact Git object, here `55efd8a54464636a24e621f6d1b60486d235b20e` |
| Package authority identities | Exact contract version, document revision, and canonical-source digest for RTC, MAP, and every required ALIGN/AST/CAL/PTC/VAL authority |
| External authority identities | Exact TEL/timing/field/pointing, detector-geometry/APT, BEAM, calibration-policy, source/provenance, and map-grid/projection artifacts used |
| Owner-decision state | Exact disposition/version for every gate affecting this profile, especially MAP OD-003, OD-004, OD-006, OD-007, OD-008, and OD-009 |
| Route state | Explicit producer sequence, including whether PTC is required, disabled, or replaced by a separately authorized profile |
| Quantity and identity | Stream, Stokes, unit, observation, array, network, group, input column, detector occurrence, sample-axis identity, and parent products |
| Geometry | Coordinate authority, frame, WCS, pixel basis, shape/order, extent, boundary rule, projection class, and `G_{pi}` normalization identity |
| Policies | VAL eligibility/action profile, support-policy population and exact admitted `coverage_cut`, required companions, required products, and failure scope |
| Lifecycle | Requested, effective, observation-resolved, applied, and realized parents without reverse mutation or cross-observation leakage |

Any change to one bound authority, owner decision, route, projection class,
support population, required companion, or policy parameter creates a new
effective profile identity. Matching filenames, shapes, local keys, or
configuration tokens do not establish profile equivalence.

## 3A. Exact logical `B_MAP` member table

The logical bundle is:

```text
B_MAP = (
  exact sample signal,
  sample identity and parent chain,
  AST RTC-grid coordinate/WCS,
  MAP projection request or exact G_pi parent relation,
  upstream analysis coefficient,
  VAL map-admission decision,
  support and exposure,
  causes and influence,
  calibration and beam convention,
  response and null-space state,
  uncertainty/covariance availability,
  lifecycle and provenance,
  availability and failure state
)
```

“Available” below means scientifically available under the exact cited source,
not merely numerically present. The current-disposition column is an audit
result, not a proposed default.

| `B_MAP` member | Scientific owner | Producer | Exact parent | Quantity / unit / frame | Support | Current availability | Required profile | MAP action permitted | MAP action prohibited | Source clauses |
|---|---|---|---|---|---|---|---|---|---|---|
| Exact sample signal | CAL owns calibrated ordinary-`xs` meaning; PTC owns its admitted conditioned transform | PTC for the ordinary PTC-dependent route | Exact CAL output, RTC-conditioned `x`, and frozen PTC operator/application generation | Calibrated Stokes-I sample; declared active `mJy/beam` convention and sign; detector/sample domain | Exact retained PTC output occurrence and application support | **Not route-admissible now:** the numerical value can exist, but PTC conflicts and missing profiles remain | Exact CAL, PTC application/output, and MAP-admission profiles | Consume the exact finite retained value after all upstream gates | Recalibrate, restore a removed reference, substitute raw `r`, or use a default CAL-only/PTC-disabled route | CAL `SCI-CAL-REQ-001--003`, `015--016`, `039--043`; PTC `SCI-PTC-REQ-001`, `010`, `022--023`, `046--047`, `061--063`, `083`; MAP `SCI-MAP-REQ-002`, `004`, `010` |
| Sample identity and parent chain | Each producer owns its stage identity; observation/source authority owns acquisition identity | ALIGN → RTC → CAL → PTC provenance records | Observation/acquisition, detector occurrence, ALIGN slot `s`, RTC output `n`, CAL/PTC generation, and immutable product parents | Identity, not a numerical quantity; no unit or coordinate frame | One exact occurrence through every retained stage | **Structurally specified; externally incomplete** for native tune/readout, TEL, and APT identities | One exact route/profile and source binding | Join branches only by the declared occurrence and parent relation | Infer identity from row, shape, time value, filename, detector label, or equal coordinates | ALIGN `SCI-ALIGN-REQ-003--006`, `050`; RTC `SCI-RTC-REQ-003`, `028--029`, `084--085`, `115`, `139`; CAL `SCI-CAL-REQ-003`, `005`; PTC `SCI-PTC-REQ-002--005`, `017`, `046`, `050`, `071`; MAP `SCI-MAP-REQ-003`, `009`, `043`, `046` |
| AST RTC-grid coordinate/WCS | AST owns coordinate construction; RTC owns the output-grid/sample parent; external TEL/APT authorities own their inputs | AST RTC-output-grid role | Exact RTC product, plan, grid, representative ALIGN slot, time/phase/delay/segment/support, plus external pointing and geometry parents | Continuous coordinate with declared celestial frame, units, WCS, pixel basis, shape/order, and extent | Exact RTC-output occurrences for which the coordinate role is valid | **Blocked:** exact RTC→AST boundary body and geometry/field-rotation authority are absent | Version-bound RTC→AST boundary and exact TEL/APT/AST coordinate profile | Admit the declared finite coordinate and WCS as a projection input | Treat equal numeric coordinates as identical parents; rerun ALIGN silently; interpolate, wrap, clamp, or invent placement | RTC `SCI-RTC-REQ-028--029`, `037`, `041`, `086`, `114`; AST `SCI-AST-REQ-074--083`; MAP `SCI-MAP-REQ-003`, `005`, `043--045` |
| MAP projection request or exact `G_{pi}` parent relation | MAP scientific owner owns admissible projection classes, normalization, boundary, and conservation semantics | MAP owns the request; a future named producer may materialize `G_{pi}` from AST coordinates | Exact AST RTC-grid coordinate/WCS plus immutable MAP grid/projection request | Dimensionless sample-to-pixel coefficient on named pixel frame; exact class and boundary semantics | All and only candidate sample–pixel pairs defined by the request | **Unavailable:** MAP OD-008 is open and no authorized materialized relation is bound | Owner-approved projection profile resolving OD-008 | After resolution, construct membership and weights from exact `G_{pi}` | Equate continuous coordinate or containing pixel with `G_{pi}`; assume `sum_p G_{pi}=1`; choose an implementation default | AST `SCI-AST-REQ-080--083`; MAP `SCI-MAP-REQ-005`, `010--011`, `025--030`; `SCI-MAP-OD-008` |
| Upstream analysis coefficient | PTC owns coefficient family, estimation population, lifecycle, factors, and QC facts; MAP owns only its use in `a_{pi}` | PTC coefficient product | Exact PTC fit/model generation and same conditioned sample occurrence | Declared `omega_i` family and unit; **not precision merely by unit or inverse-square form** | Exact coefficient-eligible occurrence and estimation population | **Not MAP-admissible:** PTC named-use profiles are reserved and PTC has frozen internal conflicts | Registered PTC coefficient/QC and MAP-admission profiles, with conflict disposition | Use a finite positive coefficient under the exact authorized family; treat zero as noncontributing | Default to unity; infer precision; transfer a coefficient across generation/support; use negative/nonfinite values | PTC `SCI-PTC-REQ-012--013`, `047`, `052--060`, `069`; VAL `SCI-VAL-REQ-009--010`, `026`, `044--046`; MAP `SCI-MAP-REQ-006`, `019--024` |
| VAL map-admission decision | The named MAP use owner owns policy; VAL owns evaluation semantics, not policy | VAL evaluation under a registered immutable profile | Source-current producer facts, complete cause knowledge, exact object/use, profile, and evaluation generation | Typed eligibility/action result; unitless; object/use-scoped | Exact sample occurrence and named MAP upstream-admission use | **Unavailable:** `SCI-MAP:map_upstream_admission` is reserved only and source bindings are stale | Complete registered MAP-admission profile with current digests and compatibility | Use the returned decision as one upstream predicate | Substitute finiteness, package-local validity, another use's decision, or a reserved profile name; treat eligibility as numerical contribution | VAL `SCI-VAL-REQ-001--010`, `023--026`, `035--038`, `044--049`; MAP `SCI-MAP-REQ-004`, `010` |
| Support and exposure | Each package owns its stage support; use owner owns policy population; MAP owns incidence, contribution, retained exposure, and final support | Upstream package facts plus VAL evaluation; MAP derives only MAP-local states | Exact occurrence, acquisition interval, policy population, projection, and coefficient state | Membership plus explicitly named exposure quantity/unit; neither is weight, precision, significance, or validity | Acquired, valid-original, fit, application, retained, incidence, contribution, and pixel support remain distinct | **Partly specified, not fully bridged:** cross-package exposure carrier and coefficient-support relation need owner decisions | Exact support/exposure population and MAP-admission profiles | Carry upstream states separately; derive MAP-local incidence/contribution/support under MAP rules | Collapse exposure into support or weight; call retained exposure acquired exposure; promote a pixel from a nonzero counter alone | ALIGN `SCI-ALIGN-REQ-015`, `027--031`; CAL `SCI-CAL-REQ-030`, `033`; PTC `SCI-PTC-REQ-012`, `052--057`; VAL `SCI-VAL-REQ-030--034`; MAP `SCI-MAP-REQ-007`, `010--013`, `025--033` |
| Causes and influence | Each fact producer owns causes/origin/influence; use owner owns action policy; VAL evaluates | ALIGN/RTC/CAL/PTC fact records and VAL evaluation | Exact original or representative occurrence, producer generation, and direct/transitive influence graph | Typed facts/relations; no unit or coordinate frame | Cause and influence domains declared by the producer, not inferred from payload | **Facts partly available; use decision unavailable** because bindings/profiles are incomplete | Source-current cause taxonomy plus exact consumer profile | Apply only the action returned for the named use | Convert silence to cause absence; turn one cause into a universal action; collapse direct origin into transitive influence | ALIGN `SCI-ALIGN-REQ-027--030`, `034--036`; RTC `SCI-RTC-DEF-011--013`, `SCI-RTC-EQ-020a/b`, `SCI-RTC-REQ-019--020`, `041`, `052`, `131--143`; VAL `SCI-VAL-REQ-004--007`, `018--022` |
| Calibration and beam convention | CAL owns its calibrated transformation; external BEAM/flxscale owners own nominal-beam and source-model meanings | CAL plus bound external beam/calibrator artifacts | Exact RTC-conditioned `x`, occurrence-associated calibration factors/operators, and beam/passband/source parents | Declared calibrated quantity, unit, sign, reference plane, and nominal-beam convention; `mJy/beam` does not by itself assert a literal point-source peak | Exact calibration-valid occurrences and factor/operator validity domains | **Conditionally specified, externally incomplete:** CAL core is admitted; external beam/input realizations and complete route profile are not | Exact CAL use/route profile plus versioned BEAM/flxscale and source/calibrator authorities | Preserve the declared calibrated meaning and beam identity | Reinterpret units, assume literal peak response, invent factors, or apply CAL twice | CAL `SCI-CAL-REQ-001--003`, `013--016`, `021--024`, `039--044`; MAP `SCI-MAP-REQ-002`, `008`, `014`, `017`, `022`, `044`, `050` |
| Response and null-space state | Each transformer owns its local response; MAP owns deposition/normalization consequences and final representation | ALIGN, RTC, CAL, and PTC companions; later MAP on identical membership/placement | Exact signal parent, frozen state, masks/support/order, projection, normalization, application counts, and source/beam template | Operator/tracer with declared template, normalization, and units; null/additive modes typed separately | Exactly the signal support and rows for the declared response tier | **Incomplete or typed unavailable:** complete chain is not closed; PTC conflict and MAP OD-003 remain | Exact response-companion, source/beam, and MAP consumer profiles | Propagate a declared companion on identical rows, or preserve typed unavailability | Infer response from unit; encode unavailable as zero/sentinel; reuse fixed-state response as full-procedure response; erase null space | ALIGN `SCI-ALIGN-REQ-037`; RTC `SCI-RTC-REQ-037--045`, `108--125`; CAL `SCI-CAL-REQ-039--043`; PTC `SCI-PTC-REQ-061--068`, `087`; MAP `SCI-MAP-REQ-008`, `016--018`, `036`, `050`; `SCI-MAP-OD-003` |
| Uncertainty/covariance availability | Each producer owns its conditional terms and omissions; MAP owns authorized propagation/persistence | ALIGN/AST/RTC/CAL/PTC uncertainty records; MAP conditional operator | Exact signal/response/support/model/selection generation and declared covariance domain | Variance/covariance in corresponding squared units, plus typed calibration, astrometric, response, selection, model, and bias terms | Exact authorized rows/domain; consequential correlations retained | **Typed but numerically incomplete:** total uncertainty unavailable and MAP OD-004 open | Exact uncertainty representation/persistence and consumer-claim profile | Propagate only supplied conditional covariance and retain explicit omissions | Treat missing terms/correlations as zero; call normalization precision; claim total significance | ALIGN `SCI-ALIGN-REQ-038--039`; AST `SCI-AST-REQ-022`, `065--072`; RTC `SCI-RTC-REQ-042--045`, `135`; CAL `SCI-CAL-REQ-032--038`; PTC `SCI-PTC-REQ-057--060`; VAL `SCI-VAL-REQ-025`; MAP `SCI-MAP-REQ-019--024`, `036`, `050`; `SCI-MAP-OD-004` |
| Lifecycle and provenance | Every package owns its requested/effective/resolved/applied/realized generations; registry owner owns profile versioning | Immutable stage records and product bundle | Exact source artifacts/digests, decisions, learned evidence, resolved/frozen model, application, and publication parents | Typed state and generation identities; no unit | Observation- and product-scoped, without cross-observation leakage or reverse mutation | **Semantically coherent, not freeze-complete:** CAL/VAL/MAP status and VAL source bindings prevent final authority | Exact source bindings and one immutable route/profile revision | Replay the bound generation and publish immutable parents | Rewrite an earlier decision from a later generation; infer profile equivalence from names; mix observations | RTC `SCI-RTC-EQ-028`, `SCI-RTC-REQ-035`, `056--057`, `117--125`; CAL `SCI-CAL-REQ-005`; PTC `SCI-PTC-REQ-046--050`, `071`; VAL `SCI-VAL-REQ-027`, `029`, `040`, `044`, `048--049`; MAP `SCI-MAP-REQ-009`, `036--049` |
| Availability and failure state | Producer owns fact/product availability; named use owner owns action; MAP owns required-product/final-validity failure | Every stage's typed state and atomic publication record | Exact requested/effective route, required authorities, companions, products, and publication generation | Distinct disabled, not-requested, not-applicable, unavailable, invalid, failed, and not-produced states; no numeric sentinel | Exact fact, branch, product, or atomic bundle scope | **Fail-closed:** distinctions are broadly coherent, but disabled PTC and route/failure closure remain unresolved | Exact route, required-products, disabled, and failure profile | Exclude or fail only at the declared scope; keep optional `r` failure independent | Convert unavailable to zero, finite to valid, disabled to a bypass route, or publish partial required bundles as complete | ALIGN `SCI-ALIGN-REQ-034`, `040`, `044`, `052`; AST `SCI-AST-REQ-057--060`, `069`; RTC `SCI-RTC-REQ-026`, `037`, `045`, `049`, `131--136`; CAL `SCI-CAL-REQ-004`, `034--038`, `045--046`; PTC `SCI-PTC-REQ-066`, `076--077`, `088`; VAL `SCI-VAL-REQ-004`, `011--013`, `036`, `038`; MAP `SCI-MAP-REQ-047--050` |

The table makes several non-equivalences explicit: AST continuous coordinate is
not `G_{pi}`; a MAP request is not a materialized projection; a PTC coefficient
is not precision; VAL eligibility is not contribution; support is not exposure;
response is not unit; normalization is not variance; calibrated `mJy/beam` is
not by itself a literal point-source peak; and numerical signal availability is
not total-uncertainty availability.

## 4. Identity and coordinate diamond

All branches SHALL close on one explicit occurrence identity before MAP
admission:

```text
Tune/readout x/r mapping + TEL/source occurrence
                        │
                        v
               ALIGN paired relation ──> RTC conditioned-x/raw-r bundle ──> CAL/PTC facts ──┐
                        │                         │                              │             │
                        └──> AST ALIGN role ──────┴──> AST RTC-grid coordinate ──┤             │
                                                                               v             │
                                                           MAP projection request             │
                                                                               │             │
                                                       unresolved G_pi materializer           │
                                                                               │             │
VAL: producer facts + use-owner policy + Registry binding + evaluation ─────────┴─────────────┤
                                                                                              v
                                                           one MAP candidate i=(t,d), pixel p
```

The join is scientific identity, not row position.

1. **Tune/readout and Source/TEL identity.** External authorities SHALL bind the
   native `x/r` mapping, observation, acquisition, sample-time occurrence,
   detector occurrence, field/source, and requested/realized pointing facts.
2. **ALIGN pair identity.** ALIGN SHALL construct the stable paired relation
   from the exact native parents and preserve occurrence, slot, original versus
   inserted state, source/support, and replacement/drop mappings
   (`SCI-ALIGN-REQ-001--007`, `REQ-027--031`, `REQ-050`).
3. **RTC pair identity.** RTC SHALL preserve the atomic conditioned-`x` /
   raw-`r` pair, its original-pair parentage, and stable output sample identity
   `n`. Required identity or authority failure is atomic, not a partial branch
   output (`SCI-RTC-REQ-003`, `REQ-049`, `REQ-084`, `REQ-115`, `REQ-139..143`).
4. **RTC grid mapping.** The phase-zero relation
   `rho_dn=(d,Mn)` SHALL identify the detector and mapped stable sample on the
   exact common grid with full declared support (`SCI-RTC-REQ-028..029`). The
   profile SHALL NOT infer equality from equal lengths or row numbers.
5. **Independent coordinate validity.** Coordinate validity remains an
   independent fact joined to the pair identity (`SCI-RTC-REQ-086`). RTC order
   does not authorize a second silent ALIGN pass or a calibrated RTC branch
   (`SCI-RTC-REQ-013`).
6. **ALIGN/AST handoff.** The MAP candidate SHALL receive the AST continuous
   RTC-grid coordinate on the same occurrence identity, with declared frame,
   full WCS, pixel basis, extent, and validity. That coordinate is not
   `G_{pi}`. MAP owns the projection request and estimator semantics; the
   materializer and exact projection authority remain unresolved under OD-008
   (`SCI-AST-REQ-074--083`; `SCI-MAP-REQ-003`, `REQ-005`). Missing
   coordinates SHALL NOT be interpolated, wrapped, clamped, chosen, or replaced
   by invented placement.
7. **MAP identity.** Observation, array, network when applicable, map group,
   Stokes, input column, occurrence/product-scoped detector acquisition,
   estimator, response version, WCS, shape/order, boundary, parent, and product
   identity SHALL be explicit (`SCI-MAP-REQ-002..003`, `REQ-043`).

A mismatch anywhere in this diamond is a profile-admission failure. It SHALL
NOT be repaired by container position, a local detector key, row equality, a
WCS tolerance, or a late reprojection.

## 5. Required producer prerequisites

### 5.1 ALIGN/AST

The selected ALIGN/AST authority SHALL supply, for the same occurrence set:

- stable common sample-axis binding and every replacement/drop mapping;
- independently valid finite coordinates and cause-specific invalidity;
- coordinate frame, time/pointing/field authority, astrometric uncertainty,
  complete WCS, pixel basis, indexing convention, and extent;
- a lossless relation from occurrence identity to the input of the declared
  MAP projection; and
- a guarantee that no second unrecorded alignment, interpolation, wrapping,
  clamping, or recentering occurs.

MAP's exact consumer anchor is `SCI-MAP-REQ-005`; identity/WCS persistence is
`REQ-043..045`. These clauses require the facts but do not prove that a current
ALIGN/AST artifact supplies them.

### 5.2 RTC

The admitted RTC state SHALL provide or preserve:

- the atomic conditioned-`x`/raw-`r` original pair and stable sample `n`
  (`SCI-RTC-REQ-003`, `REQ-084`, `REQ-115`);
- the exact `rho_dn=(d,Mn)` mapping, common grid, and full support
  (`REQ-028..029`);
- the immutable learn–resolve–apply chain and realized parentage
  (`SCI-RTC-EQ-028`, `REQ-035`, `REQ-056..057`, `REQ-117..125`);
- explicit validity, availability, origin, cause, representative occurrence,
  direct exclusion/consumer eligibility, and full transitive influence
  (`SCI-RTC-DEF-011..013`, `SCI-RTC-EQ-020a/b`, `REQ-019..020`, `REQ-041`,
  `REQ-052`);
- complete fixed/full response and uncertainty state, including typed
  unavailable response, fixed-state zero cross-branches, omitted/null spaces,
  conditional assumptions, and application count where the canonical state
  requires them (`REQ-037..045`, `REQ-108..125`);
- direct and inferred influence/support provenance sufficient for a downstream
  consumer to decide eligibility without reconstructing it from a payload
  (`REQ-139..143`); and
- atomic failure for malformed identity or missing required authority
  (`REQ-049`).

The exceptional donor-recovery route is constrained: the locked RTC contract
permits an **x-only donor-recovery exception**. Across the complete donor causal
influence, conditioned-`r`, its response, and dependent covariance remain typed
unavailable while the common grid, raw-`r` parent, causes, and otherwise valid
`x` are preserved (`SCI-RTC-DEF-048..049`, `SCI-RTC-REQ-026`,
`REQ-131..136`, `REQ-141..142`). A recovered `x` SHALL NOT fabricate a
conditioned `r`, erase origin/cause/influence, or make a nonrepresentative
occurrence eligible under a consumer profile that explicitly excludes it
(`SCI-RTC-REQ-020`, `REQ-052`).

### 5.3 SCI-CAL

The CAL authority SHALL bind the ordinary-`xs` Stokes-I quantity, calibrated
`mJy/beam` scale and unit, calibration quality/validity, uncertainty transfer,
response basis, beam/passband/atmosphere policy, and lineage for the exact
selected occurrences (CAL `SCI-CAL-REQ-001--003`, `013--016`,
`021--024`, `039--044`; MAP `SCI-MAP-REQ-002`, `REQ-004`, `REQ-008`).

RTC hands CAL the conditioned `x` branch only (`SCI-RTC-REQ-103`). The raw
`r` companion does not thereby become a calibrated signal, a second CAL
branch, or a numerical response product. Any CAL response tracer handed to
MAP SHALL retain its template, normalization, unit, processing parent, and
membership, or be typed unavailable (`SCI-MAP-REQ-008`, `REQ-017`).

### 5.4 PTC

The ordinary coefficient `omega_i` SHALL have a declared producer, family,
unit, normalization scope, lifecycle, applied factors, and statistical status.
It SHALL be finite and strictly positive to contribute; zero is
noncontributing, and negative/nonfinite is invalid (`SCI-MAP-REQ-006`). PTC
also supplies the conditioned-sample and covariance identity required by the
current MAP boundary.

No coefficient is precision merely because it has an inverse-square unit.
MAP normalization `Q` may be labeled marginal inverse variance only under the
complete true-inverse-variance, independence, and projection condition in
`SCI-MAP-REQ-020`; formal weight, full covariance/precision, and empirical NOI
weight remain distinct (`REQ-019..024`).

### 5.5 VAL

The handoff SHALL preserve the Core/Registry/Consumer split without MAP/RTC
redefinition:

- package producers own facts, causes, origin, influence, and availability;
- the named use owner owns sample/detector policy, nonfinite policy, and action
  precedence;
- the Registry binds that policy to one exact immutable profile and current
  producer-source identities;
- VAL Core evaluates the bound proposition, preserves reasons and knowledge
  state, and does not acquire fact or policy ownership; and
- the consumer owns its numerical action after receiving the evaluation.

The resulting record SHALL expose sample and detector eligibility; exact
policy/profile version; cause-specific invalid, unavailable, excluded, and
failed states; and direct, representative, inferred, and transitive influence
needed by the chosen use.

RTC representative/replacement and donor exceptions remain governed by their
explicit origin/influence clauses; MAP consumes upstream eligibility before
arithmetic (`SCI-RTC-DEF-011..013`, `DEF-048..049`, `SCI-RTC-EQ-020a/b`,
`SCI-RTC-REQ-019..020`, `REQ-052`, `REQ-131..143`;
`SCI-MAP-REQ-004`, `REQ-010`, `REQ-034..035`). A profile SHALL NOT collapse
cause, origin, influence, support, exposure, availability, and final validity
into one mask.

## 6. Disabled PTC and the direct-CAL ambiguity

The current MAP clauses create a direct **semantic quantity edge** from CAL:
`SCI-MAP-REQ-002` calls calibrated SCI-CAL ordinary-`xs` the ordinary input.
The same current boundary assigns conditioned-sample, coefficient, and
covariance meaning to PTC. It does not specify a PTC-disabled profile, a
CAL-only coefficient, or permission to bypass PTC.

Therefore this draft adopts the only fail-closed disposition supported by the
locked findings:

| Route request | Draft disposition |
|---|---|
| PTC active and exact PTC authority/profile present | May proceed to the remaining gates |
| PTC disabled but a separately owner-approved replacement profile supplies the complete conditioned-sample, coefficient, covariance, lifecycle, and validation authority | May proceed only under that exact future profile |
| `direct CAL`, PTC absent, `omega_i` defaulted/uniform without authority, or final `x_i` producer ambiguous | **No numerical MAP route; hard stop** |

The direct CAL semantic edge is not a complete CAL-only execution route. A
configuration token such as `ptc=false`, a uniform array, or historical
behavior cannot answer the missing scientific ownership question.

## 7. Projection coefficient `G_{pi}` and OD-008 blocker

For MAP, `a_{pi}=G_{pi} omega_i`; contribution membership and every signal,
response, covariance, hit, exposure, and edge result depend on the exact
projection.

`SCI-MAP-OD-008` is still open. It asks:

- which sample-to-pixel projection classes ordinary MAP v0.1 authorizes;
- what normalization applies to `G_{pi}`, including whether and where
  `sum_p G_{pi}=1` is required;
- how boundary loss is represented; and
- which upstream authority owns those facts.

Pending an owner-approved answer, a profile may name only the actual declared
one-hot or fractional class already used by the contract examples; it SHALL
record class, normalization, extent, and boundary convention and SHALL assume
no unrecorded conservation property. This does **not** close OD-008. Because
the owner and admissible normalization are not fixed, no production numerical
route may be registered from this draft alone.

MAP performs no crop, pad, fractional shift, reprojection, interpolation,
wrap, recenter, or mosaic. Canonical-grid preparation and future
reprojection/mosaic ownership remain OD-009 (`SCI-MAP-REQ-005`, `REQ-038`).

## 8. Contribution, support, cause, influence, exposure, and validity

These states SHALL remain separately represented and joined by explicit
identity:

| State | Producer/transformer meaning | Admission rule |
|---|---|---|
| RTC origin/cause/influence | Original vs donor/replacement origin; direct, representative, inferred, and full transitive influence | Preserve exact provenance and consumer-policy disposition; never infer from payload (`SCI-RTC-REQ-019..020`, `REQ-041`, `REQ-052`, `REQ-131..143`) |
| Upstream eligibility | VAL/CAL facts before estimator-specific rejection | MAP consumes without promotion or recoding (`SCI-MAP-REQ-004`) |
| Geometric incidence | Upstream-eligible occurrence whose declared projection touches pixel | Explicit fractional touch convention; may differ from estimator count (`SCI-MAP-REQ-025..026`) |
| Estimator contribution | Every MAP predicate passes and `a_{pi}>0` | Resolve before payload arithmetic (`SCI-MAP-REQ-010`, `REQ-027`, `REQ-034`) |
| Upstream-eligible exposure | Named exposure over eligible projected population before coefficient rejection | Finite, nonnegative, named accounting unit (`SCI-MAP-REQ-007`, `REQ-029`) |
| Retained exposure | Same exposure over estimator-admitted membership/placement | Not precision (`SCI-MAP-REQ-030`) |
| Numerical support | Explicit finite positive Q and effective-policy threshold | Separate from science support and final validity (`SCI-MAP-REQ-031..032`) |
| Science-policy support | Separate stricter effective-policy predicate | Not automatically final validity (`SCI-MAP-REQ-032`) |
| Final raw validity | Exact output-row membership + admitted identity + applicable support + finite signal/required companions + no product failure | No finite payload, hit, exposure, Q, support alias, or compatibility alias can promote it (`SCI-MAP-REQ-033`) |

The exact dimensionless `coverage_cut=c` value SHALL be admitted by the
owner-authorized effective policy before support-row construction. CI-001
fixes only dimensional status; OD-007 leaves numeric domain, boundary cases,
recommended range, authority, and failure behavior open. An unauthorized
value is a hard stop, not a threshold default.

## 9. Response and uncertainty disposition

### 9.1 RTC state

RTC fixed/full response, uncertainty, null-space/omission, cross-branch,
application-count, and direct/transitive support state SHALL be carried as an
immutable conditional record. Required complete response may instead be a
typed unavailable state; unavailable SHALL NOT be represented by a zero array
or sentinel (`SCI-RTC-REQ-037..045`, `REQ-108..125`). The x-only donor
exception preserves otherwise valid `x` but makes conditioned-`r`, its
response, and dependent covariance unavailable across the complete donor causal
influence; it preserves the common grid, raw-`r` parent, and cause/influence
record and fabricates none of those unavailable products (`REQ-131..136`,
`REQ-141..142`).

### 9.2 MAP state

MAP SHALL apply the same admitted membership, geometric placement, and exact
support-row selection to signal and declared linear companions
(`SCI-MAP-REQ-016`). A stored kernel is a realized response to template `tau`
only if its parent proves `k=H tau` and the output proves
`khat_out=A_out H tau=R_out tau` on identical rows (`REQ-017`). Otherwise it
is a tracer or typed unavailable.

The conditional covariance is `C_x,out=A_out N A_out^T` on the exact
authorized row domain, retaining consequential cross-pixel terms
(`SCI-MAP-REQ-019`). Its representation SHALL say persisted,
lineage-resolvable, summarized, or unavailable, with omitted correlations and
calibration, astrometric, response, and additive-bias nuisance terms explicit
(`REQ-022`). OD-003 leaves restricted use of a response-unavailable map open;
OD-004 leaves minimum covariance persistence open. A response-dependent or
covariance-dependent consumer fails closed when the needed state is absent.

## 10. Required failures and hard stops

The following conditions SHALL fail before live MAP aggregate mutation:

| Condition | Required disposition | Anchor |
|---|---|---|
| Malformed/missing RTC pair identity, mapping, or required authority | Atomic upstream failure; no partial branch | `SCI-RTC-REQ-049`, `REQ-056..057`, `REQ-084` |
| Pair/coordinate/grid diamond does not close | Reject candidate; no row-position inference or second ALIGN | `SCI-RTC-REQ-013`, `REQ-028..029`, `REQ-086`; `SCI-MAP-REQ-003`, `REQ-005` |
| Required origin/cause/influence/consumer policy absent | Decision unavailable and hard stop; only an authoritative decisive false may produce explicit ineligibility | `SCI-RTC-REQ-019..020`, `REQ-041`, `REQ-052`, `REQ-139..143`; `SCI-VAL-REQ-004--007`, `REQ-035--038` |
| PTC disabled/absent and no owner-approved replacement route | **No numerical MAP route** | `SCI-MAP-REQ-002`, `REQ-006`, `REQ-019..023`; this profile §6 |
| `G_{pi}` class/normalization/boundary/owner not bound | **No numerical MAP route** | `SCI-MAP-REQ-005`, OD-008 |
| Coordinate, signal, coefficient, exposure, or required companion invalid/nonfinite | Exclude at declared contribution scope before payload evaluation, or fail required product | `SCI-MAP-REQ-006..010`, `REQ-034..035`, `REQ-047` |
| Exact `coverage_cut` not admitted by effective policy | Fail before support rows or required-product mutation | `SCI-MAP-REQ-031..032`, OD-007 |
| Response absent but encoded as numeric/sentinel, or required response unavailable | Typed unavailable; dependent profile rejects | `SCI-MAP-REQ-008`, `REQ-017`, OD-003 |
| Required covariance/uncertainty state insufficient | Dependent profile rejects; missing correlations are not zero | `SCI-MAP-REQ-019..024`, OD-004 |
| Incompatible observation/coadd bundle or odd/different grid | Atomic rejection; all coadd state unchanged | `SCI-MAP-REQ-037..042` |
| Unrepresentable aggregate/index, missing required companion, or failed required publication | Fail at declared scope before mutation; no completion marker | `SCI-MAP-REQ-047..048` |

## 11. Output and downstream boundary

An admitted observation output is one immutable ordered bundle containing
signal, numerator, normalization identity, response state, conditional
uncertainty, all eight MAP facts, units, frame/WCS, shape/indexing, estimator,
lifecycle, parentage, exact product identity, and failures
(`SCI-MAP-REQ-036`, `REQ-043..050`). The normalized vector contains exactly
the effective-policy-authorized rows; unsupported storage is not zero sky
(`REQ-012`).

No consumer receives a bare signal plane as a complete scientific map. A
downstream consumer SHALL preserve raw parent and raw validity and SHALL NOT
promote a raw-invalid pixel (`SCI-MAP-REQ-049..050`).

### 11.1 Generic downstream-consumer envelope

NOI, FLT, BEAM, SRC, Pointing, OOF, and FRUIT contracts were not admitted and
were not inspected. The rows below therefore state only what a future
consumer-specific profile would minimally have to request from the immutable
bundle; they do **not** claim that any named package accepts, interprets, or can
produce these objects.

| Generic consumer | Minimum inherited state | Additional external/use authority still needed | Prohibited inference |
|---|---|---|---|
| NOI | Exact signal rows and parents; normalization identity; conditional covariance and omitted terms; selection/support/exposure; response; lifecycle and validity | Named noise estimand, population, empirical/model relation, correlation domain, stationarity assumptions, significance use, and exact VAL profile | `Q`, an inverse-square coefficient, hit count, or exposure is not precision or a noise model |
| FLT | Signal plus identical-row response/covariance/support/WCS and null-space state; immutable raw parent and validity | Exact filter operator, domain, boundary/padding rule, application count, response/covariance propagation, missing behavior, and output-validity profile | A filtered finite map is not automatically valid; filter application cannot leave response/covariance unchanged by silence |
| BEAM | Calibrated quantity/unit/sign/reference; nominal-beam identity; full WCS; realized response/template; passband/source-model/calibrator parents; covariance and validity | BEAM/flxscale owner, beam convention and validity interval, calibrator/source model, passband dependence, normalization, and claim-specific profile | `mJy/beam` alone is not a realized beam, literal peak, or point-source transfer function |
| SRC | Raw map bundle, source/field identity, frame/WCS, response/null space, covariance, support/exposure, and immutable parent | Source model, association/deblending rule, estimator, selection/completeness and uncertainty profile | A local maximum, field label, or finite pixel is not source identity or significance |
| Pointing / OOF | Raw bundle and validity; requested/realized pointing; source center; exact frame/WCS; response and covariance; observation identity | Mode owner, source/beam model, AltAz and time/EOP/refraction authority, fitting/parameter convention, and explicit MAP OD-006 registration where MAP arithmetic is reused | Ordinary-map arithmetic grants no pointing/OOF fitting, astrometric, response, or production authority |
| FRUIT | Immutable raw and prior-generation parents; exact masks/eligibility/support; response/covariance/null space; iteration/application count and failure state | Iterative source-model/update operator, convergence/stopping meaning, nonretroactivity rule, uncertainty/selection propagation, and profile for every generation | A later model or eligibility result cannot rewrite an earlier fit/support generation or erase the raw parent |

## 12. Approval blockers

This draft cannot become an executable registered ordinary-route profile until
the blockers that affect that route are closed:

1. **Frozen PTC conflicts:** F-001 and F-002 require XOD-001/XOD-002 owner
   dispositions; F-011/XOD-017 requires exhaustive disabled-product state.
2. **VAL source and policy:** stale bindings F-003/XOD-003, reserved PTC and
   MAP profiles F-004/XOD-004--005, incomplete independent-exposure registration
   F-005/XOD-006, and absent aggregate/coadd policy F-015/XOD-007 remain
   unavailable. Producers own facts, use owners own policies, the Registry
   binds, VAL evaluates, and consumers own numerical actions.
3. **Coordinate boundary:** F-006/XOD-008 and F-007/XOD-009 require the exact
   RTC→AST boundary and detector-geometry/field-rotation authority.
4. **Projection:** F-014/XOD-010 and MAP OD-008 must name authorized projection
   classes, `G_{pi}` normalization/boundary semantics, and the materializer.
   This is a universal numerical-deposition blocker.
5. **Coefficient and exposure:** F-023/XOD-014 must identify the exact
   MAP-facing coefficient; F-019/XOD-018 must identify the exposure carrier.
6. **Alternate route:** F-018/XOD-013 must explicitly prohibit direct CAL→MAP
   or authorize a separate complete route. CAL alone does not own that choice;
   MAP plus the route, coefficient, and policy owners are required.
7. **External producers:** F-017/XOD-015 requires exact tune/readout, TEL
   timing/field/pointing, detector/APT, BEAM/flxscale, atmosphere/passband
   inputs, source/provenance, frame/EOP/refraction, and calibrator/source-model
   authorities as applicable to the selected route.
8. **MAP support policy:** F-024/XOD-019 and MAP OD-007 require the exact
   `coverage_cut` value to be admitted by an owner-authorized effective
   policy; an unadmitted value blocks support-row construction.
9. **Response scope:** F-012/XOD-011 and MAP OD-003 block complete-response and
   response-dependent claims. OD-003 does not by itself block a future
   owner-approved response-independent consumer class.
10. **Uncertainty scope:** F-013/XOD-012 and MAP OD-004 block total-uncertainty,
    precision, significance, and stronger covariance-persistence claims. They
    do not turn an otherwise authorized finite signal into total uncertainty.
11. **Coadd/grid scope:** F-015/XOD-007 blocks policy-authorized coaddition;
    F-025/XOD-020 and MAP OD-009 separately block canonical crop/pad,
    incompatible-grid preparation, reprojection, or mosaic. OD-009 does not by
    itself block an otherwise compatible single-observation map.
12. **Final authority:** F-016/XOD-016 requires current bindings, completed
    owner reviews, and final source-manifest closure before an authoritative
    handoff can freeze.

MAP OD-006 additionally governs any future Pointing/OOF registration but is not
an ordinary Stokes-I observation-map prerequisite.

Until then, this document is a review aid and its only safe executable
conclusion is the hard stop stated in §1.

## 13. Evidence identity

Every clause above resolves against exact committed package paths at
`55efd8a54464636a24e621f6d1b60486d235b20e`. Full SHA-256 values, status,
revision, authority role, amendments, ledgers, imported boundaries, and
exclusions are recorded once in `SIX_PACKAGE_SOURCE_PACKET_REPORT.md`.
`SIX_PACKAGE_SOURCE_CROSSWALK.csv` records the producer/consumer relation for
each proposed interface or invariant. This profile does not supersede either
record and does not turn a missing or open source into authority.
