# SCI-FLT-INF implementation-informed dossier

Dossier identity: `SCI-FLT-INF-INTERNAL-DOSSIER v0.1/r0.5`

Status: quarantined Stage A manager evidence; forbidden to any future
implementation-blind scientific author

## Firewall

This dossier records what the exact base implementation, configuration,
schemas, and history appear to do so that scope recovery does not omit a
scientifically consequential family. It does not establish what any method
ought to mean, whether the implementation is conformant, or whether a product
is valid. Every statement below is an implementation observation or an
explicit inference from one.

## Requested and effective surface

The typed map-filter request exposes three named types: `wiener_filter`,
`convolve`, and `destripe`; four templates: `kernel`, `gaussian`, `airy`, and
`highpass`; `lowpass_only`; `normalize_errors`; edge-guard thresholds/fill/
taper; and denominator approximation controls. Requested state is copied
one-way into the mature numerical object. Runtime policy requires noise maps
for the full `wiener_filter` path unless `lowpass_only` is selected, and
requires a mapmaking kernel when `template_type=kernel`.

Configuration names are not scientific identities. In particular,
`wiener_filter + lowpass_only` follows the convolution code path rather than
the full noise-weighted path, and `destripe` is named in configuration while
its call site is disabled.

## Apparent full-path algebra

For one map, the full path constructs:

- `R(x)=sqrt(weight(x))`, except for a separately computed kernel response
  where uniform weight is forced;
- a normalized radial spectral field `V(q)` derived from the map's noise PSD;
- a template `t(x)` from the parent kernel, a configured Gaussian/Airy width,
  or a high-pass delta;
- a numerator equivalent in structure to applying
  `t^T R F^{-1} V^{-1} F R` locally to the input map; and
- a spatially varying denominator constructed from the same template,
  weights, and inverse spectral model, then truncated/approximated by
  convergence and tail controls.

The published signal is `numerator/denominator` where the denominator is
nonzero, and numerical zero otherwise. The published `weight` field becomes
the denominator. The kernel response is separately passed through the full
path under forced uniform weights.

Manager recovery inference: this shape is closer to a local normalized
template-amplitude or generalized matched estimator than to a posterior mean
sky-field reconstruction. ODQ-001 subsequently supplied independent owner
authority: the scientific identity is an optimal matched-template amplitude
estimator, not a posterior/Wiener sky reconstruction. The implementation
observation did not establish that identity and still establishes no
conformity. No independently specified signal prior, posterior distribution,
or posterior covariance was recovered for the active path.

The owner also distinguishes the estimator from ordinary source-shaped
convolution and requires unbiased normalization for a matching amplitude under
the eventual exact noise, support, edge, and validity assumptions. This
dossier supplies none of those remaining scientific definitions or proofs.

## Noise-spectrum state and fallback

The spectral state is not a verbatim external covariance object. The code:

- reads a one-dimensional noise PSD from the map buffer;
- finds a data-dependent break relative to the PSD maximum;
- flattens portions interpreted as low- and high-pass behavior;
- interpolates radially over the two-dimensional Fourier grid;
- clamps small/nonpositive values; and
- normalizes the resulting two-dimensional field to unit sum.

If PSD inputs are absent or invalid, the code logs a warning and substitutes a
constant spectrum, described in the log as a lowpass-only response. The
requested method identity is not changed. This is scientific method
substitution and must be fail-closed or represented by a separate explicit
selector/realized-method identity in any future authority.

ODQ-004 subsequently supplied owner direction without selecting a noise or
covariance model. A future implementation-blind author must develop bounded
options in both the scientific and engineering contract views. The owner also
identified a **radially symmetrized average map noise PSD** as a historical
candidate for scientific examination only. The shaped spectral state observed
above is quarantined implementation evidence: it is neither the definition of
that candidate nor an option, default, covariance authority, or conformity
claim, and none of its numerical mechanics may enter the author packet.

## Approximation, threshold, and null behavior

The denominator calculation may stop because of relative-update convergence,
a number-of-checks cap, a configured iteration cap, or a tail-fraction cap.
The realized stop reason and summary values exist in memory. Denominator
values below an internal limit are set to zero; output values at those
locations are also set to numerical zero. No recovered scientific authority
defines the approximation error, permitted residual, zero-denominator
interpretation, or whether zero represents a physical value, a null mode, or
unavailability.

## Adaptive edge/background conditioning

Before the filter proper, the active path may derive a science mask from the
parent weight and `coverage_cut`, refine it using a fraction of the provisional
core median coverage, dilate it by a beam-FWHM-scaled radius, and optionally
build a cosine taper. It derives a background level from the median finite
signal inside the science mask. Outside/toward the edge, it applies the window
around that background to signal, applies the window squared to weight, and
applies the window to the kernel.

This is input-derived, partly signal-derived conditioning. With hard support
it is affine conditional on the learned mask/background; because the median
and masks depend on the parent, the end-to-end map operation is not one fixed
linear operator across parents. It is therefore not merely the
full-footprint-only fixed method selected for SCI-FLT-FIXED.

The same learned window is applied to NOI members, but the real signal is
conditioned around a learned nonzero median whereas members are multiplied as
zero-centered fields. Parity, conditional meaning, admitted output region,
and fill influence require scientific authority. Historical FLT D001 approved
an eroded valid region for a former mixed package, but the current fixed
package deferred this method.

## NOI-member application

For a map index, template, spectral state, denominator, and edge window are
resolved from the real parent first. Each stored noise realization is then
passed through the convolution path or through only the full-path numerator
and divided by the already resolved denominator. This is a learned-once,
frozen-state application, not per-member relearning.

The code contains no active route that recomputes the PSD, template,
denominator, edge mask, or background independently for every member. Frozen
SCI-NOI therefore classifies current apparent behavior, if scientifically
admitted, as conditional on the real-parent learned state. A per-member route
would be a new NOI-GEN method and cannot share an ensemble with the fixed-state
route.

## Empirical coefficient calibration and derived products

When empirical noise products or `normalize_errors` request it, the pipeline
computes member mean and pointwise variance, forms a robust global median of
`weight_formal * variance` over an admitted region, and defines a scalar
coefficient scale as its reciprocal. It creates an empirically rescaled weight
plane and may replace the published weight with it. It also creates pixelwise
and point-source-labeled uncertainty/standardized products.

This operation is downstream of the filter proper, depends on the NOI
ensemble, changes a published coefficient product, and currently uses
historical product identities that frozen SCI-NOI does not automatically
authorize. It is a separate calibration/derived-product contract problem,
not a parameter of the map estimator.

## Data-thresholded spectral selection

The `destripe` routine Fourier-transforms the current map, finds the maximum
coefficient magnitude, zeroes coefficients below a configured fraction of
that maximum, and transforms back. The selection depends on the input map and
is nonlinear as a parent-to-output operation. Its sole call site is commented
out at the base commit. No exact threshold policy, response, support,
uncertainty, product, or failure authority was recovered.

## Downstream source-analysis observations

Templates can be configured analytically or derived from the parent mapmaking
kernel. Filtered source finding and Gaussian fitting occur downstream. No
active base-commit route was found that uses a fitted source to relearn the
filter template or spectral state. Therefore:

- configured or kernel-derived templates remain method state whose scientific
  identity is unresolved;
- the existence of downstream source-finding code does not change the filter's
  published product role; and
- ODQ-002 excludes detection, selection, peak interpretation, deblending,
  fitting, catalog construction, and source-learned state from the selected
  package without creating an SRC ownership boundary. Any future method is an
  independent scientific contract and is not inferred from current behavior.

## Parent and order observations

Filtering runs on observation maps when coaddition is disabled and on the
coadd map when coaddition is enabled. These are different parents. Adaptive
state learning, nonlinear selection, spatially varying normalization, and
uncertainty meaning generally do not commute with coaddition.

ODQ-003 independently authorizes both ordinary-MAP parent roles while keeping
them distinct: observation-local filtering of one immutable observation bundle
and coadd-local filtering of one immutable coadd bundle. It authorizes no
equivalence, commutation, or filter-owned cross-observation combination and
defers JINC plus derived-map parents. The implementation observation did not
establish that scientific decision or conformity to it.

The apparent order is:

```text
immutable raw map parent
  -> adaptive edge/background state resolution and conditioning
  -> template/noise/denominator state resolution
  -> map estimator/transformation
  -> fixed-state member application
  -> empirical noise/coefficient/standardized products when requested
  -> diagnostics and source finding/fitting
```

No active composition of the full path with a separate SCI-FLT-FIXED operator
was recovered. Any future `FIXED after INF` or `INF after FIXED` composition
must have a distinct ordered identity and response/covariance contract.

## Evidence limits

The sequential and OpenMP classes appear intended to implement the same
surface, but this study performed no parity audit. Historical validation and
production use do not establish an estimand, prior, covariance, response,
scientific support, or correctness. The dossier selects no default, tolerance,
fallback, threshold, or product label.
