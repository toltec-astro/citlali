# SCI-FLT-INF operator, state, and product taxonomy

Taxonomy identity: `SCI-FLT-INF-TAXONOMY v0.1/r0.10`

Status: Stage A vocabulary updated through approved ODQ-009 and the ODQ-004/
ODQ-006/ODQ-009 author delegations; remaining details are not normative science

## Identity tuple

Every future inference-bearing method should bind at least

```text
(method_id, estimand, parent_id, parent_grouping, domain, grid/WCS,
 state_generation, state_sources, operator, approximation,
 response, units/beam, support/null, covariance, order,
 NOI_lifecycle, product_bundle, failure_policy)
```

Two outputs with different values in a scientifically consequential field are
not the same method realization even if they share code or filenames.

## Parent roles

| Role | Meaning | Required distinction |
| --- | --- | --- |
| `P_MAP_OBS` | one immutable normalized ordinary-MAP observation bundle | not a coadd and not a covariance-bearing precision map |
| `P_MAP_COADD` | one immutable normalized ordinary-MAP coadd bundle | binds exact contributing observation set and coadd authority |
| `P_JINC_OBS` | one immutable per-array JINC observation bundle | separate signed estimator, support, response, and covariance status |
| `P_FIXED_DERIVED` | an exact immutable SCI-FLT-FIXED successor | binds the exact fixed operator and order before inference |
| `P_INF_DERIVED` | an exact immutable matched-filtered map used by a later independently authorized operation | cannot be treated as its own raw parent |

Observation and coadd parents are separate identities. An estimator learned or
applied after coaddition is not presumed equivalent to combining observation-
level estimator products.

ODQ-003 admits `P_MAP_OBS` and `P_MAP_COADD` for v0.1 and no other parent role.
For `P_MAP_OBS`, learning/application is observation-local. For
`P_MAP_COADD`, learning/application is coadd-local and binds the exact
contributing-observation set and coadd generation. `P_JINC_OBS`,
`P_FIXED_DERIVED`, `P_INF_DERIVED`, and all other derived parents are deferred.
No cross-observation combination or commutation identity is admitted.

## Estimand families

### Owner-selected optimal matched-template amplitude estimator

ODQ-001 selects the amplitude of the exact supplied template as the estimand
of the historical full path. ODQ-005 defines that template as an immutable
scientifically declared template-response product representing parent-map
response per unit amplitude `A`. For a declared parent vector `m`, location-
indexed template `t_x`, and exact realized weighting operator `Q_x`, ODQ-006
selects the authoritative conditional reference form

```text
N(x) = <t_x, Q_x m_x>
D(x) = <t_x, Q_x t_x>
A_hat(x) = N(x) / D(x)
```

or a mathematically identical representation under the exact declared
discrete inner product, units, indexing, WCS, boundary, and support
conventions. `Q_x` is supplied by the eventual owner-selected ODQ-004 option;
ODQ-006 does not define it. Under the authorized zero-mean model
`m = A t_x + n`, exact template, fixed weighting, support, edge, validity, and
regularity assumptions, the normalization must satisfy `E[A_hat(x)] = A` for
a matching signal of amplitude `A`.

The template scaling fixes the amplitude convention, with
`unit(t) = unit(m) / unit(A)`. When `t_x` is the exact parent-bound point-source
response, the estimand is a matched point-source amplitude field. Another
explicitly supplied scientific template yields the amplitude field of that
specified shape. No peak, integral, flux-density, or beam convention follows
from a generic kernel name. `D` is a normalization coefficient. `D^{-1}` is a
conditional variance only if the exact covariance, linear model, fixed state,
domain, and regularity assumptions authorize that interpretation. A code field
called `weight` does not supply that authority.

ODQ-008 fixes the filtered signal unit as
`unit(A_hat)=unit(A)=unit(m)/unit(t)`. This is not automatically the parent
signal unit. Applicable parent WCS/frame, location indexing, array/band,
observation/coadd grouping, parentage, support/validity facts, and calibration
provenance persist. Parent signal meaning, nominal-beam interpretation, DC
response, integrated flux, surface brightness, extended-source fidelity, and
calibration covariance do not persist without exact matched-estimator
authority.

For each application, the template product binds exact source and immutable
identity, compatible parent role, units, grid/WCS/frame, centering/subpixel
phase, support/truncation/tails, array dependence, parent-beam relationship,
calibration, validity, and provenance. Gaussian/Airy construction may only
materialize this complete product. Target/source/NOI-learned templates and the
historical high-pass/delta case are not base-v0.1 template identities. When
`Q_x` is the admitted inverse covariance, the complete assumptions support the
optimal GLS matched-estimator claim. A weaker ODQ-004 weighting option must
weaken optimality and uncertainty claims accordingly; `optimal` is not an
unconditional achieved-performance claim.

This estimator is not ordinary convolution with `t_x`: convolution alone does
not apply the declared noise weighting and amplitude-unbiased normalization.

ODQ-002 fixes the published product role as a matched-filtered map: a filtered
version of the exact admitted parent map product that preserves its applicable
map-domain structure and semantics. The local estimator identity does not
create source detections, candidates, fitted-source or peak interpretations,
deblended objects, or catalog rows.

### Posterior/Wiener reconstruction

A linear Gaussian example would require an explicit signal prior covariance
`S`, noise covariance `N`, and measurement/response operator `H`, such as

```text
m_post = S H^T (H S H^T + N)^{-1} d
C_post = S - S H^T (H S H^T + N)^{-1} H S
```

or a mathematically equivalent declared form. The estimand is a posterior
field, not template amplitude. Prior-conditioned response, posterior
covariance, regularization, null space, and bias/coverage interpretation must
be explicit. This equation is illustrative scope vocabulary, not a proposed
TolTEC method. ODQ-001 explicitly excludes this interpretation for the
historical Citlali full path. Any genuine Wiener/posterior sky reconstruction
would require a separate future scientific contract.

### Data-selected transformation

When modes, support, background, template, covariance, hyperparameters, or
method are selected from the parent, represent the end-to-end map as

```text
state = Learn(parent, external_inputs)
output = Apply(parent; state)
```

The conditional apply operator may be linear or affine for fixed `state`,
while the end-to-end `Apply(parent; Learn(parent))` is generally nonlinear.
Both identities and the generation boundary are required.

## State classes

| State class | Definition | Consequence |
| --- | --- | --- |
| `DECLARED_FIXED` | supplied by exact external or parent-owned scientific authority and fixed before this method applies to the target parent | output is conditional on that exact state; an immutable parent-bound template is not learning by this method |
| `PARENT_LEARNED_FROZEN` | learned from the target real parent, then frozen for application | learning and application are distinct immutable generations; uncertainty is conditional unless relearning is included |
| `NOI_INFORMED_SUCCESSOR` | learned/selected/updated using a prior NOI product | prior UNC, learning, new state, science product, new GEN, and successor UNC remain distinct; prior UNC is dependent input, not validation |
| `MEMBER_RELEARNED` | the complete declared learning graph runs separately for every admitted NOI member | separate NOI-GEN method and member population; no mixing with fixed-state members |
| `UNAVAILABLE` | required state authority or realization absent | no numerical output for that method; no silent substitute |

`Learned` without one of these exact graphs is insufficient.

## Operator components

Candidate operator identities must decompose, where applicable, into:

1. parent admission and immutable snapshot;
2. edge/missing/background conditioning;
3. immutable template-response product or signal-prior state;
4. noise/covariance/spectral state;
5. fixed conditional apply operator;
6. normalization and any denominator approximation;
7. response and null-space evaluation;
8. product-domain support/validity decision;
9. NOI member graph; and
10. downstream coefficient calibration or standardized-product derivation.

Omitting a component is permitted only when it is explicitly not applicable,
not when it is hidden in an implementation step.

## Approximation and regularization roles

ODQ-006 makes the exact normalized reference operator scientific authority.
An exact numerical evaluation is conformant. FFT evaluation, interpolation,
iteration, or finite truncation remains a realization technique only when its
effect on normalization, matching-template amplitude response, support/null
behavior, and uncertainty lies within an owner-selected scientific conformance
envelope.

The future author must give the scientific and engineering views the same
bounded quantitative envelope alternatives. Every approximate realization
binds its reference operator, approximation identity/parameters, applicability
domain, bound, envelope result, and completion status. Until an envelope is
selected, approximate execution is unavailable.

A floor, pseudoinverse cutoff, omitted mode, clipping rule, or regularization
that defines `Q_x` or its null space is ODQ-004 scientific state. A change
beyond the selected envelope is a separate versioned method rather than an
implementation detail. Nonfinite `N`/`D`, nonpositive `D`, null templates,
unresolved convergence, or unmet bounds are null/unavailable/failed and never
scientific amplitude zero.

## Complete-support identity

ODQ-007 restricts base v0.1 to complete-support output locations. For each
location, the exact declared influence set includes every parent, template,
weighting, approximation, and boundary input on which `N`, `D`, or their
validity depends. That set may be a bounded local footprint or a global/
nonlocal domain; a finite stencil may not be invented for a nonlocal operator.

Every required element must be in the immutable parent domain, admitted for
the exact FLT use, finite and available where required, and valid under all
bound predicates. Failure at any required element makes the output location
unavailable. The base method has no partial-support/truncated estimator,
support renormalization, extension, interpolation, imputation, background
learning, taper learning, or signal-derived support selection.

Numerical padding or fill is outside the scientific vector and is conformant
only when conservative erosion establishes that no admitted output depends on
it. Parent-shaped storage does not change the scientific support. A future
adaptive edge/background operation has a distinct method and learned-state
identity; it is not a base realization option.

## Order identities

At minimum distinguish:

```text
T_INF(P_MAP_OBS)
T_INF(P_MAP_COADD)
T_INF(T_FIXED(P))
T_FIXED(T_INF(P))
T_INF(P; state learned from NOI(P))
```

No commutation is presumed. Any comparison or composition must bind exact
response, covariance, support, validity, and generation identity after the
full ordered chain.

For the selected v0.1 parent roles, only `T_INF(P_MAP_OBS)` and
`T_INF(P_MAP_COADD)` are admitted. The other order identities remain taxonomy
for deferred work and are not current package routes.

## Response and transfer roles

| Role | Minimum meaning |
| --- | --- |
| `R_MODE` | transfer/response to declared spatial modes for exact state and support |
| `R_TEMPLATE` | response to the exact admitted template, location, subpixel phase, and boundary state |
| `R_LEARNING` | full-procedure response when the state-learning graph is rerun |
| `R_FIXED_STATE` | response conditional on one frozen learned/declared state |

`R_FIXED_STATE` and `R_LEARNING` are not interchangeable. A convolved or
filtered kernel is evidence about a response only under its exact unit-source,
centering, support, normalization, and parent-beam convention.

ODQ-008 defines the exact fixed-state row response, for any declared
parent-domain perturbation `u`, as

```text
L_x u = <t_x, Q_x u_x> / <t_x, Q_x t_x>.
```

Thus `delta A_hat(x)=L_x delta m`, and the exact response to the declared
unit-amplitude template placed at `y` is

```text
R_t(x,y) = L_x t_y
           = <t_x, Q_x t_y> / <t_x, Q_x t_x>.
```

At every admitted matching location, `R_t(y,y)=1` under the exact fixed state,
complete support, phase, boundary, and validity assumptions. Off-diagonal
response may be asymmetric, nonstationary, anisotropic, position dependent,
or nonlocal. A uniformly processed template is not a universal response unless
translation invariance, identical weighting, support/validity, centering/
phase, boundary, and normalization are proved over the declared domain.

The parent nominal beam remains provenance. A matched point-source response
footprint may be called an effective matched-filter beam only with an explicit
response-derived definition; any beam area or solid angle requires a declared
coordinate measure, domain, and normalization. Point-source flux-density
meaning requires exact template amplitude plus CAL/BEAM lineage. Non-point
templates retain shape-amplitude terminology.

If `Q`, support, approximation state, selector state, or other consequential
state is re-estimated under a perturbation, response of the complete procedure
is `R_LEARNING`, not `L_x`; ODQ-010 retains that learning graph. ODQ-013
retains the persisted response representation. Parent/template calibration
dependence must be joint, with no presumed independence or cancellation;
missing calibration covariance is unavailable, not zero or `D`.

## Covariance and uncertainty roles

| Role | Meaning |
| --- | --- |
| `C_PARENT` | exact covariance/uncertainty authority of the admitted parent; unknown remains unknown |
| `C_COND` | uncertainty conditional on fixed declared/learned state |
| `C_FULLPROC` | uncertainty including declared learning/reselection/re-estimation |
| `C_POST` | posterior covariance for the separate deferred posterior family; not a selected-package role |
| `C_EMP` | exact NOI-defined empirical conditional product |
| `K_NORM` | estimator normalization coefficient; not covariance or precision by shape/name |
| `W_USE` | consumer-specific effective weight; requires exact use-specific authority |

Diagonal-like products do not imply independence. A reciprocal does not imply
precision. Unreported covariance is unavailable, not zero.

ODQ-009 makes the exact fixed-state identity, when the authoritative matching
parent covariance exists,

```text
C_COND = L C_PARENT L^T.
```

It binds the exact parent/grouping, domain, support, fixed state, response,
population, rank/null, regularization, approximation, omissions, calibration,
and lifecycle. Missing entries and cross blocks remain unavailable. Only when
ODQ-004 selects exact `Q=C_PARENT^-1` and every GLS premise holds is
`D(x)^-1` the marginal conditional variance; it is not full precision or an
independence claim. A frozen-NOI `C_EMP` ordinary product remains a conditional
randomization second moment rather than covariance or physical-noise variance.
Calibration and full-procedure uncertainty remain separate identities.

The future author must give both contract views the same bounded covariance-
representation option identities and consequences. Exact explicit,
structured, projected, lineage-resolvable, and unavailable forms may be
considered; none is selected here. Owner disposition precedes freeze or a
numerical covariance route.

ODQ-004 does not select one of these roles. It assigns the future author to
develop bounded noise/covariance, spectral-weighting, and parent-coefficient
options with common identities in both contract views, separately accountable
for observation and coadd parents. Historical use of a radially symmetrized
average map noise PSD is one candidate for examination only. Until an authored
option is owner-selected, `C_PARENT`, any spectral weighting operator, and any
precision/variance interpretation of `K_NORM` or a parent coefficient remain
unavailable.

## Support, null, and unavailable roles

Future products should distinguish:

- numerical computability;
- exact parent validity inherited without promotion;
- method support for the selected operator/state;
- response-qualified scientific admission;
- null or unresolved modes;
- regularized modes;
- edge/fill/taper influence;
- learned-state availability;
- approximation adequacy; and
- required-product completion.

A zero-denominator location is not assigned scientific value zero by default.
It must be typed as null, unavailable, outside support, or another
owner-approved state.

## Minimum atomic product bundle

For any selected future method, the smallest candidate atomic bundle is:

1. immutable parent reference and digest;
2. requested/effective/observation-resolved/realized method identity;
3. exact state artifact references and dependence graph;
4. output matched-filtered map and its applicable inherited map-domain
   spatial structure/semantics, with exact template-amplitude quantity and
   unit rather than automatic parent signal-unit inheritance;
5. normalization/denominator product if it carries scientific meaning;
6. exact fixed-state response identity, declared representation, beam/solid-
   angle interpretation if any, and full-procedure-response availability;
7. support, validity, null, edge, and missing-state products;
8. covariance/uncertainty identity and availability;
9. approximation, regularization, and realized stop/selection record;
10. NOI parity/generation identity when uncertainty is requested; and
11. atomic completion/failure record.

Diagnostics may be optional, but a required scientific role cannot be omitted
and then inferred from another plane.

## Failure classes

- `PARENT_UNAVAILABLE`: exact parent bundle or numerical route absent.
- `STATE_UNAVAILABLE`: required template/prior/covariance/learning authority
  absent.
- `STATE_LEARNING_FAILED`: declared learning graph did not complete.
- `OPERATOR_UNAVAILABLE`: exact apply/normalization cannot be evaluated.
- `APPROXIMATION_UNQUALIFIED`: truncation/convergence state lacks permitted
  error authority.
- `RESPONSE_UNAVAILABLE`: output may be diagnostic but cannot carry the
  intended scientific amplitude/field claim.
- `COVARIANCE_UNAVAILABLE`: uncertainty/precision/significance uses fail
  closed while permitted signal use may remain separate.
- `PARITY_UNAVAILABLE`: transformed NOI route lacks exact owner authority or
  member parity.
- `REQUIRED_PRODUCT_FAILED`: atomic bundle fails; partial survivors do not
  acquire authority.
- `ALTERNATIVE_METHOD_SELECTED`: only an explicitly authorized selector may
  produce this state; the realized product carries the alternative method's
  identity and never claims primary-method success.
