# SCI-FLT v0.1 Initial Operator And Product Taxonomy

Status: sanitized Stage A proposal; not scientific authority

## Program Adherence And Prior-Work Recovery

This taxonomy is the scientist-readable result of prior-work recovery. It is
implementation-blind: it names scientific distinctions without exposing
current algorithms, configuration labels, test results, audits, repairs, or
validation. A future author may use it only after the owner approves the exact
packet.

## Classification Tests

A method belongs to the fixed deterministic family only when all coefficients,
offsets, template/kernel state, domain, boundary treatment, normalization,
support, and missing-data rules are fixed before application to the admitted
parent random field. A method is inference-bearing when any of these is chosen,
estimated, updated, or conditioned using:

- a noise or covariance model;
- a prior or regularization rule;
- the target data or a related learned state;
- a source model, source location, morphology, or response estimate; or
- an objective that changes the estimand from transformed map amplitude to a
  fitted/template amplitude.

An inference-bearing method can later publish a frozen realized operator. That
does not make the learning method deterministic; it creates a fixed-state
application phase with explicit learned-state lineage.

## Operator Families

### `FLT-DET-AFFINE`

A fixed affine map-domain transformation

\[
  y = A x + c,
\]

with exact content-bound `A`, `c`, parent domain, output domain, units,
support, validity, and response. Fixed convolution is a structured special
case. A low-pass transformation belongs here only when its transfer and all
state are fixed.

### `FLT-DET-CONV`

A fixed convolution-like transformation with a named discrete kernel,
centering, grid, padding/edge policy, missing-data policy, normalization, and
declared support footprint. Method subtypes may include smoothing, resolution
matching, or another explicit purpose; a common formula does not erase those
product identities.

### `FLT-INF-WIENER`

A Wiener-family map transformation with an explicit target estimand, signal
and noise model, prior/regularization, template, weights, learned-state source,
fixed/relearned lifecycle, transfer/response, bias conditions, and uncertainty
meaning. An unavailable model input cannot silently select a different method.

### `FLT-INF-MATCH`

A matched or generalized least-squares template-amplitude estimator. It names
the amplitude estimand, template normalization, background/nuisance model,
noise/covariance model, position/grid, support, estimator response, bias, and
uncertainty. It is not synonymous with convolving a map by a source-shaped
kernel.

### `FLT-INF-SOURCE`

A transformation or selection whose state depends on learned source position,
morphology, mask, model, or subtraction. The source model and learning state
must be explicit. This family must not absorb source-fitting or FRUIT ownership;
it only covers a filtering method the owner deliberately assigns to FLT.

### `FLT-INF-SPECTRAL-SELECT`

A map-domain spectral selection whose retained modes or thresholds are derived
from the input spectrum. Its target artifact/contaminant, selection statistic,
threshold rule, response, bias, and uncertainty must be explicit. It is
scientifically distinct from RTC temporal filtering.

## State Classes

Every method declares one of these state identities:

| State class | Meaning |
| --- | --- |
| `fixed_external` | Complete operator state supplied independently of the parent product and frozen before application. |
| `fixed_parent_bound` | Complete operator state frozen and content-bound to this parent or a named predecessor, but not relearned during application. Dependence and source imprint remain explicit. |
| `learn_then_freeze` | A named learning procedure produces an immutable operator-state generation; application uses only that generation. |
| `successor_update` | New information selects or updates state, producing a new transformation and science-product generation. |
| `per_member_relearned` | The method relearns for each admitted NOI member under its own scientific definition. It cannot mix with fixed-state members. |

The classification is part of method identity, not optional provenance.

## Product Roles

### `FLT-PARENT`

The immutable admitted MAP or JINC bundle, including parent estimand, units,
grid/frame, response identity, support/validity, covariance availability, and
content identity.

### `FLT-STATE`

The immutable applied transformation state: family/method, purpose, parameters,
kernel/template/prior/noise model, learned-state generation, order, input and
output domain, edge/padding/missing policy, normalization, response convention,
support footprint, units rule, lifecycle, and failure policy.

### `FLT-SIG`

The transformed scientific amplitude. It preserves or changes units only as
declared by `FLT-STATE`. It is not automatically a flux, fitted amplitude,
statistical significance, or uncertainty-normalized field.

### `FLT-RSP`

The exact response object associated with `FLT-SIG`. Depending on the method,
this may be an identically transformed unit-source kernel, a transfer function,
a response operator, or a declared honest absence. Peak response, signed
integral, aperture response, beam solid angle, and frequency transfer are
distinct quantities.

### `FLT-SUP` and `FLT-VALID`

Filter-specific numerical support and scientific validity. They remain
distinct from parent exposure/coverage, parent validity, finite storage,
nonzero weight, and confidence. Parent coverage is carried as a parent fact;
it is not silently rewritten as filter support or transformed validity.
Edge/fill/padding influence and missing data are explicit.

### `FLT-COV-FORMAL`

An optional covariance or second-moment object obtained by applying a fixed
declared operator to an available declared parent covariance model. A diagonal
variance plane is labeled as such and is not full covariance. When parent
covariance is unavailable, the propagated object is unavailable rather than
zero.

### `NOI-UNC[FLT-SIG]`

An SCI-NOI-owned conditional uncertainty/covariance attachment produced by
applying the exact `FLT-STATE` method to every compatible admitted
randomization. It is not an FLT-authored uncertainty and cannot define the
filter.

### `FLT-LINEAGE`

The immutable binding among parent, transformation state, response, support,
validity, learned-state generation, output, and any attached NOI product.

## Product-Identity Rules

1. Changing the parent, method family, estimand, template/kernel, prior/noise
   model, learned state, normalization, edge/missing rule, support/validity,
   response convention, or application order creates a different product
   identity.
2. A requested method name cannot substitute for the realized transformation
   identity.
3. A fallback to another transformation creates a different method/product or
   fails; it cannot retain the requested method identity.
4. Filtering an observation and filtering a coadd are different parent/product
   identities.
5. A downstream source fit or FRUIT iteration creates a downstream product,
   not a mutation of `FLT-SIG`.
6. Fixed-state and per-member-relearned NOI routes create different method and
   uncertainty identities.

## Initial Package Recommendation

Use `SCI-FLT` only as the Stage A tranche. Commission `SCI-FLT-DET` first for
fixed deterministic transformations. Hold inference-bearing work in the
`SCI-FLT-INF` tranche and commission separate Wiener, matched-estimator,
source-learned, or spectral-selection contracts whenever their identities do
not satisfy the same scientific questions. This recommendation awaits owner
decision FLT-ODQ-101.
