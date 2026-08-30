# SCI-FLT-FIXED v0.1 Linear Transformation Science Extract

Status: sanitized recovered candidate; not scientific authority

## Program Adherence And Prior-Work Recovery

This extract abstracts reusable implementation-independent mathematics from a
mixed historical Convolve record. It deliberately omits all source-code
inspection, audit findings, candidate verdicts, repair requirements, tests,
validation, historical product assertions, and old empirical-uncertainty
estimators. Current SCI-NOI Stage A controls the uncertainty boundary.

The future implementation-blind author must independently evaluate these
identities under the owner-approved scope. Citation here does not require
adoption.

## Typed Objects

Let:

- `x` be the admitted parent scientific amplitude on an explicit input domain;
- `C_x` be its available declared covariance, or honestly unavailable;
- `A` be a fixed linear operator;
- `y` be the transformed amplitude on an explicit output domain; and
- `r` be a named unit-source or other response object on a compatible domain.

All objects require units, frame/grid, indexing, support, missing/non-finite
policy, and content identity. “Fixed” means fixed before application to the
admitted parent random field, not merely unchanged inside one numerical call.

## Fixed Linear Transformation

Base v0.1 admits only

\[
  y = A x.
\]

The mean and covariance, when the required moments exist and `A` is fixed,
satisfy

\[
  \mathrm{E}[y] = A\,\mathrm{E}[x],
  \qquad
  C_y = A C_x A^{\mathsf T}.
\]

Affine transformations `y=Ax+c`, fixed offsets, background/template
subtraction, and additive correction are outside base v0.1. They require a
versioned successor that distinguishes signal output, perturbation operator,
response, covariance, additive-term parent/unit/support/uncertainty, absolute-
reference consequences, NOI treatment, lifecycle, and failure.

## Fixed Discrete Convolution

For a fixed kernel with coefficients `k_{ij}` mapping admitted input pixels
`j` to output pixel `i`,

\[
  y_i = \sum_j k_{ij} x_j.
\]

With a declared diagonal parent covariance
`C_x = \operatorname{diag}(V_j)`,

\[
  \operatorname{Var}(y_i) = \sum_j k_{ij}^2 V_j,
\]

and

\[
  \operatorname{Cov}(y_i,y_\ell)
    = \sum_j k_{ij} k_{\ell j} V_j.
\]

Thus convolution generally creates off-diagonal output covariance even when
the input covariance is diagonal. A propagated variance plane is not the full
covariance and cannot support independent-pixel multi-pixel inference unless
an additional approximation is named and justified.

When `C_x` is not available, these equations do not authorize inventing it
from a coefficient or weight label.

## Kernel Normalization And Estimand

A unit signed-sum kernel,

\[
  \sum_j k_{ij} = 1
\]

on complete translation-invariant support, preserves a constant field. It does
not by itself preserve:

- a point-source peak;
- integrated flux under finite or clipped support;
- a beam or effective-PSF solid angle;
- arbitrary aperture/background response; or
- the amplitude of a matched template estimator.

Signed kernels require support and normalization quantities appropriate to the
claim. Geometric support, signed sum, absolute (`L1`) support, squared (`L2`)
support, point-source response, and transfer are not interchangeable.

Local renormalization near missing samples or boundaries defines a different,
position-dependent operator. It may preserve a constant locally while changing
noise, source response, covariance, and product identity.

## Response Propagation

For a fixed linear transformation, an admitted response object `r` transforms
as

\[
  r_y = A r
\]

only when it is represented on a compatible domain and the exact same operator,
centering, edge/padding, missing-data, support, and normalization rules apply.

If `r` represents the parent response to a unit-amplitude source, then `r_y`
can describe the transformed response to that same source convention. A peak
ratio or weighted-aperture ratio can be scientifically meaningful only after
the source convention, weights, background treatment, valid domain,
normalization, and CAL boundary are explicit. It is a response correction, not
automatic proof of a flux estimator.

For arbitrary aperture/background weights `a`, the transformed response is the
linear functional

\[
  R_a = a^{\mathsf T} r_y.
\]

If `R_a` is finite, nonzero, and scientifically admitted, a response-corrected
amplitude candidate can take the form

\[
  \widehat{s}_a = \frac{a^{\mathsf T} y}{a^{\mathsf T} r_y}.
\]

This identity does not establish absolute calibration, optimality, unbiasedness
outside its declared source/background model, or covariance.

## Edge, Fill, Padding, And Validity

Numerical computability and scientific validity are distinct. A transform can
produce a finite number using padding or fill even when the output stencil
depends on values outside the admitted scientific parent domain.

Let `S_i` be the exact input footprint used by output pixel `i`. A conservative
interior-admission policy can require

\[
  S_i \subseteq V_x,
\]

where `V_x` is the admitted parent-valid domain. Under such a policy, outputs
whose footprints reach fill, padding, missing, or invalid input are not
scientifically valid. They may remain numerical diagnostics with explicit
status.

A taper multiplies operator coefficients. Its effect enters covariance through
the squared coefficients and enters response through the same realized
operator. A zero produced outside support is not zero uncertainty or valid
science.

Methods using partial support, local normalization, inpainting, reflection,
periodic padding, stochastic fill, or data-derived background fill require
their own exact operator and conditioning identities. They cannot inherit the
full-support convolution equations without qualification.

## Coaddition And Ordering

Let `L` be a coaddition/registration operator and `A_o` an observation-specific
filter. In general,

\[
  A_c L(x_1,\ldots,x_n)
  \neq
  L(A_1 x_1,\ldots,A_n x_n).
\]

Equality requires explicit compatible assumptions about operators, grids,
weights, support, edge treatment, normalization, response, and learned state.
Therefore filter-after-coadd and coadd-after-filter are distinct product
identities unless a contract establishes the exact relation.

## Deterministic Versus Inference-Bearing Boundary

A fixed convolution formula does not make the total method deterministic if
the kernel, mask, support, normalization, position, background, threshold, or
other coefficients were learned from the target data. The contract must name:

- the estimand;
- how state was selected or learned;
- which data influenced it;
- whether it is frozen for application;
- expected source imprint and bias; and
- whether uncertainty is conditional on the frozen state or includes learning
  variability.

Wiener and matched-template methods normally require separate derivation
because their objectives, priors/noise models, response, and uncertainty need
not match fixed convolution.

## SCI-NOI Attachment

For a fixed owner-defined transformation `T`, NOI may estimate uncertainty for
the exact transformed product only by applying the exact compatible `T` to
every admitted compatible randomization. Same name, approximate transfer,
relocation, commutation, or substitution is insufficient.

If `T` is relearned for each member, that is a distinct method with distinct
conditioning and source-imprint semantics. It cannot be mixed with members
produced using a frozen `T`.

## Honest Absence

If the parent covariance, response object, exact footprint, operator state,
normalization, or compatibility condition is unavailable, the corresponding
propagated result is unavailable. Missing information is not filled by a
weight label, a finite denominator, an empirical width from an unmatched
ensemble, or numerical zero.
