# SCI-FLT-INF operator, state, and product taxonomy

Taxonomy identity: `SCI-FLT-INF-TAXONOMY v0.1/r0.1`

Status: proposed Stage A vocabulary; not normative science

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
| `P_INF_DERIVED` | an exact immutable inference product used by a later fixed or source operation | cannot be treated as its own raw parent |

Observation and coadd parents are separate identities. An estimator learned or
applied after coaddition is not presumed equivalent to combining observation-
level estimator products.

## Estimand families

### Template-amplitude estimator

For a declared parent vector `m`, template `t`, and admitted covariance or
inverse-noise operator `C`, a reusable candidate is

```text
N = t^T C^{-1} m
D = t^T C^{-1} t
A = N / D
```

or a location-indexed/spatially varying analogue. This estimates a declared
template amplitude only under its model. `D` is a normalization coefficient.
`D^{-1}` is a conditional variance only if the exact covariance, linear model,
fixed state, domain, and regularity assumptions are authorized. A code field
called `weight` does not supply that authority.

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
TolTEC method.

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
| `DECLARED_FIXED` | supplied by exact external scientific authority before the target parent is used | output is conditional on that exact state |
| `PARENT_LEARNED_FROZEN` | learned from the target real parent, then frozen for application | learning and application are distinct immutable generations; uncertainty is conditional unless relearning is included |
| `NOI_INFORMED_SUCCESSOR` | learned/selected/updated using a prior NOI product | prior UNC, learning, new state, science product, new GEN, and successor UNC remain distinct; prior UNC is dependent input, not validation |
| `MEMBER_RELEARNED` | the complete declared learning graph runs separately for every admitted NOI member | separate NOI-GEN method and member population; no mixing with fixed-state members |
| `SOURCE_LEARNED` | learned from source fit/model/selection | requires exact SRC authority, selection effects, response, calibration, and covariance boundary |
| `UNAVAILABLE` | required state authority or realization absent | no numerical output for that method; no silent substitute |

`Learned` without one of these exact graphs is insufficient.

## Operator components

Candidate operator identities must decompose, where applicable, into:

1. parent admission and immutable snapshot;
2. edge/missing/background conditioning;
3. template or signal-prior state;
4. noise/covariance/spectral state;
5. fixed conditional apply operator;
6. normalization and any denominator approximation;
7. response and null-space evaluation;
8. product-domain support/validity decision;
9. NOI member graph; and
10. downstream coefficient calibration or standardized-product derivation.

Omitting a component is permitted only when it is explicitly not applicable,
not when it is hidden in an implementation step.

## Order identities

At minimum distinguish:

```text
T_INF(P_MAP_OBS)
T_INF(P_MAP_COADD)
T_INF(T_FIXED(P))
T_FIXED(T_INF(P))
T_SRC(T_INF(P))
T_INF(P; state learned from NOI(P))
```

No commutation is presumed. Any comparison or composition must bind exact
response, covariance, support, validity, and generation identity after the
full ordered chain.

## Response and transfer roles

| Role | Minimum meaning |
| --- | --- |
| `R_MODE` | transfer/response to declared spatial modes for exact state and support |
| `R_TEMPLATE` | response to the exact admitted template, location, subpixel phase, and boundary state |
| `R_SOURCE` | response to an exact source model including parent beam and calibration identity |
| `R_LEARNING` | full-procedure response when the state-learning graph is rerun |
| `R_FIXED_STATE` | response conditional on one frozen learned/declared state |

`R_FIXED_STATE` and `R_LEARNING` are not interchangeable. A convolved or
filtered kernel is evidence about a response only under its exact unit-source,
centering, support, normalization, and parent-beam convention.

## Covariance and uncertainty roles

| Role | Meaning |
| --- | --- |
| `C_PARENT` | exact covariance/uncertainty authority of the admitted parent; unknown remains unknown |
| `C_COND` | uncertainty conditional on fixed declared/learned state |
| `C_FULLPROC` | uncertainty including declared learning/reselection/re-estimation |
| `C_POST` | posterior covariance for a posterior estimand |
| `C_EMP` | exact NOI-defined empirical conditional product |
| `K_NORM` | estimator normalization coefficient; not covariance or precision by shape/name |
| `W_USE` | consumer-specific effective weight; requires exact use-specific authority |

Diagonal-like products do not imply independence. A reciprocal does not imply
precision. Unreported covariance is unavailable, not zero.

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
4. output signal/estimand product;
5. normalization/denominator product if it carries scientific meaning;
6. response/transfer identity and availability;
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
