# SCI-FLT v0.1 Internal Recovery Dossier

Date: `2026-08-30`

Audience: coordinator, scientific owner, and later conformity reviewers

Status: implementation-informed Stage A evidence; permanently excluded from
implementation-blind Stage B authorship

## Program Adherence And Prior-Work Recovery

This dossier implements the recovery requirement of the Scientific Contract
Library Program. It records what was found without treating current code,
configuration, tests, audits, repairs, or validation as scientific authority.
It must not be supplied to a fresh Stage B author.

## Executive Recovery Result

The implementation label “filter” currently covers scientifically different
operations:

- fixed-kernel convolution and a low-pass-only route that can be represented
  as a fixed deterministic map transformation if all state is frozen;
- full Wiener construction whose realized operator depends on noise, weights,
  template, denominator, and regularization state;
- template-sensitive operations whose intended estimand may be a smoothed
  map, a point-source response, or a matched amplitude;
- a data-thresholded destriping routine whose operator depends on the input
  Fourier amplitudes and whose execution is disabled; and
- RTC temporal filters and FRUIT feedback operations that belong to other
  scientific owners despite sharing filter-like vocabulary.

These cannot safely enter one generic contract. Final owner disposition names
the first strict-linear package `SCI-FLT-FIXED`, retains `SCI-FLT-INF` only as
a holding tranche, and requires further recovery and method splits wherever
estimand, prior, learned state, response, uncertainty, or lifecycle differs.

## Current Configuration And Lifecycle Inventory

### Requested state

The requested map-filter surface includes:

- enabled/disabled state;
- method labels `convolve`, `wiener_filter`, and `destripe`;
- template labels `kernel`, `gaussian`, `airy`, and `highpass`;
- kernel-tail labels `constant`, `zero`, and `cosine`;
- `lowpass_only` and empirical-error-normalization switches;
- per-array template FWHM;
- edge-guard threshold, dilation, and taper controls; and
- denominator tolerances, check iterations, and limits.

These labels are implementation requests. They do not, by themselves, state a
scientific estimand, exact operator identity, transfer/response convention,
uncertainty meaning, or validity rule.

### Effective state

A one-way adapter transfers the typed request into the mature filtering
object. The effective plan can disable map filtering when the selected
mapmaking route does not support it. It can also require source fitting as a
consequence of filtering or source-finding requests. Full Wiener currently
requires active noise-map generation; low-pass-only does not.

This separation of requested and effective state is useful, but the effective
record remains primarily configuration state. It does not bind every
scientifically consequential operator input.

### Observation-resolved and applied state

No complete observation-resolved scientific transformation object was found.
The applied operation is assembled from the current map buffer, map index,
array identity, template source, map weights, optional noise spectrum, kernel,
edge-guard state, and runtime policy. Several of those facts can vary between
observation maps and a coadd.

An eventual contract therefore needs a content-bound applied-transformation
identity, not only a method name and requested parameters.

### Realized state

Current realized provenance records counts such as filter contexts, filtered
maps, noise maps, and filtered source-fit attempts or successes separately for
observations and coadds. It does not constitute a complete content-bound
record of operator coefficients, learned state, parent identity, response,
support, edge policy, and uncertainty attachment.

The scientific lifecycle should distinguish at least:

`requested -> effective plan -> observation-resolved operator state -> applied
operator -> realized transformed product`.

For learned methods, learning inputs and the immutable learned-state generation
must be separately identified.

### Coverage and parent-fact state

The recovered filtered-product surface carries the frozen raw-parent digest
and raw map fact/coverage companions while separately changing signal, weight,
kernel, and filter-edge state. No independent scientifically defined
“filtered coverage” estimand was recovered. Parent exposure/coverage,
filter-stencil support, output validity, response support, inverse-weight-like
coefficients, and confidence must therefore remain different typed facts. A
filter may restrict output admission, but it cannot silently reinterpret or
rewrite the parent coverage fact.

## Execution And Product Ordering

### Observation path

For every map slot, the implementation freezes the raw map bundle as an
immutable parent before filter mutation. It then builds a template, transforms
the signal, transforms each available noise realization, may recalculate
empirical noise products, writes filtered products, and runs downstream
diagnostic/source operations. Pointing can fit the filtered observation map.

Scientific consequences still requiring authority include whether filtering
an observation is a supported scientific method, which parent bundle is
admitted, and whether any subsequent source fit estimates an amplitude in the
transformed or parent-product response convention.

### Coadd path

The code can form a raw coadd, freeze that raw coadd as the immutable parent,
and then filter the coadd. It can also filter individual observations in their
own output contexts. Those are different products. In general,

`filter(coadd(parents))` and `coadd(filter(each parent))`

need not be interchangeable because weights, support, edge treatment,
observation-specific operators, registration, normalization, and learned state
can differ. No commutation may be inferred from a shared method label.

### JINC

JINC supplies a distinct signed-estimator parent identity. Its frozen authority
does not define a general numerical filtering route, and current parent
response/covariance availability can be limited. A filtered JINC product is
therefore unavailable until both the exact JINC parent route and an applicable
FLT method are authorized.

### FRUIT

Configuration vocabulary permits FRUIT map inputs under raw or filtered
observation/coadd path labels. Separate FRUIT authority owns source-model
construction, subtraction/addition, learning, recurrence, stopping, restart,
selection, response, support, validity, lifecycle, and failure. Existing
convergence work explicitly evaluates the raw feedback product, not a
downstream filtered map.

Under approved SCI-NOI Stage A, uncertainty for a frozen FRUIT residual or
terminal product is conditional on that frozen state. NOI-informed continuation
creates a successor generation; per-member replay is a different method. FLT
does not authorize FRUIT use of a transformed product.

## Recovered Method Families

### Fixed convolution

When the template, coefficients, edge policy, normalization, parent support,
and missing-data policy are fully fixed independently of the input random
field, the current convolution lane is a deterministic linear or affine
transformation. The kernel is normalized to unit signed sum before the main
convolution path. A unit-sum convention preserves a constant field under
complete support; it does not by itself preserve point-source peak or
integrated-flux response.

The implementation also transforms the stored mapmaking kernel. Historical
owner direction required the signal and unit-source kernel response to use the
same realized operator, centering, and valid-region policy. Current code shape
is consistent with an intended paired transformation, but this dossier makes
no equality or conformity claim.

### Low-pass-only

`wiener_filter` with `lowpass_only` enters the same convolution branch as
`convolve` and does not require noise maps. The label does not establish what
frequencies are attenuated, the desired transfer function, the estimand, or
whether the result is a smoothing product or an approximation to another
method. It should be typed as a fixed deterministic method only if the owner
binds all operator state and scientific purpose.

### Template construction

Recovered template sources are:

- the current map kernel, centered using an absolute-peak location and a
  radial/tail construction in one path;
- a Gaussian parameterized by per-array FWHM;
- an Airy-like squared profile parameterized by per-array FWHM; and
- a `highpass` label whose current template construction does not establish a
  scientifically specified high-pass transfer.

Each template requires identity, units, sampling/grid, normalization,
centering, support, source/beam convention, and provenance. A template derived
from the parent product or an inferred source model may make the total method
data-dependent even if the later convolution is numerically fixed.

The standalone `gaussian_filter.h` helper is not bound into the active
execution lane and contains prior convolution work only. It is not a separate
available scientific method.

### Full Wiener

The full Wiener lane combines a template, map weights, a noise power spectrum,
frequency-domain normalization, denominator construction, and iterative
denominator truncation/tolerance rules. The realized operator is thus
conditioned on a noise model and other state. The state may be learned from or
selected using the same observation, a related ensemble, or a prior product;
those cases have different source-imprint, dependence, bias, and uncertainty
meanings.

The recovered code can replace an unusable noise spectrum with a low-pass-like
response. Scientifically, this is a method substitution and cannot be treated
as the requested Wiener method without explicit owner authority, product
identity, and failure/fallback policy.

Once a Wiener transformation is owner-frozen, SCI-NOI may apply that exact
fixed transformation under ODQ-110A. Relearning or updating it from NOI
information creates a successor transformation/science/GEN/UNC generation.
Learning independently for each realization is a distinct ODQ-104 method.

### Matched and source-sensitive methods

No distinct active method labeled `matched_filter` was recovered. Nevertheless,
current templates and downstream source fits create three concepts that must
not be conflated:

1. deterministic convolution of a map by a fixed template;
2. a response correction using an identically transformed unit-source
   response; and
3. a matched or generalized least-squares template-amplitude estimator using a
   noise/covariance model.

The third changes the estimand and is inference-bearing. If the template,
position, mask, background, or source model is learned from the target data,
that learned state and its source imprint must be explicit. Beammap and future
source-fitting contracts retain ownership of fitted-source interpretation;
FLT owns only an explicitly selected filtering estimator or transformation.

### Map-domain destripe

The recovered routine computes the input-map Fourier amplitudes, sets a
threshold as a fraction of the maximum magnitude, removes coefficients below
that data-derived threshold, and transforms back. The configured method value
exists, but the execution call is commented out. Its intended science,
threshold direction, transfer/response, support, output identity, and
uncertainty meaning are unavailable. Because the operator is selected from the
input map, it is not a fixed deterministic convolution.

This map-domain routine is also distinct from RTC's active temporal/filtering
and alt-az destriping responsibilities.

## Edge, Padding, Support, And Missing Data

The current edge guard derives a science/core mask from map weights and
coverage-related thresholds, dilates it by a radius tied to an initial FWHM,
and may apply a cosine taper. For the science map it fills exterior values
toward a same-map core median and tapers weights and kernels. The output signal,
weight, and kernel are tapered again.

Noise realizations are zero-centered and multiplied by the edge window rather
than receiving the signal map's affine median fill. Therefore an exact
signal/realization transformation parity claim is not established by the
current path. The old FLT-D001 decision proposed treating fill as a numerical
device and eroding scientific admission so no admitted stencil reaches fill.
That is a strong candidate for reaffirmation, but remains an owner decision in
this new package.

Zero output caused by missing support, an invalid denominator, or a numerical
fill cannot mean zero uncertainty or valid science. Edge/padding behavior,
missing and non-finite input policy, signed-kernel support, stencil footprint,
and output validity must be explicit method state.

## Units, Normalization, Transfer, And Response

Recovered code and historical material distinguish several facts that a
contract must type:

- stored transformed amplitude;
- map units and any unit change;
- kernel normalization such as unit signed sum;
- point-source peak response;
- signed integral and effective beam solid angle;
- spatial/frequency transfer or response as applicable;
- local support normalization versus one global operator; and
- response correction or amplitude estimation performed later by a consumer.

A map unit label does not establish absolute calibration, point-source flux,
aperture flux, or cross-band covariance. CAL retains absolute calibration,
passband/color correction, and calibration covariance. Beammap/source packages
retain fitted-source and effective-PSF interpretation where applicable.

## Uncertainty And Covariance Inventory

The deterministic fixed-operator mathematics permits formal propagation of a
declared input covariance. A diagonal input model yields a pointwise propagated
second moment through squared coefficients, but convolution creates
off-diagonal output covariance even when the input covariance is diagonal. A
variance plane is not a covariance object.

The current convolution path contains a diagonal-variance weight-propagation
calculation with support masking. The current Wiener path replaces map weights
with a denominator-like quantity. These are implementation facts, not proof
that the resulting planes are precision, uncertainty, inverse variance, or
statistical significance.

Approved SCI-NOI Stage A controls empirical uncertainty:

- FLT first defines and freezes the exact transformation and product identity;
- NOI applies exactly it to every compatible admitted randomization;
- NOI owns the resulting conditional second moment/covariance/standardization
  semantics and attachment;
- unknown covariance is not zero;
- fixed-state, successor-generation, and per-member-relearned methods remain
  distinct; and
- a partial surviving ensemble cannot silently replace the admitted ensemble.

The historical FLT-D003 placement of a robust global empirical scale inside
FLT is superseded by approved SCI-NOI Stage A and owner decision FLT-ODQ-106.
It cannot be imported unchanged.

## Parent, Product, And Consumer Inventory

| Relationship | Recovered state | Ownership consequence |
| --- | --- | --- |
| MAP parent | Raw observation/coadd bundle frozen before filtering | MAP owns parent; FLT owns only successor transformation/product. |
| JINC parent | Separate signed-estimator bundle; general numerical route unavailable | No ordinary-MAP analogy or silent route. |
| transformed signal | Map-domain successor amplitude | Needs new product identity; not automatic photometry. |
| transformed kernel/response | Kernel can be transformed alongside signal | Must bind unit-source convention, centering, support, normalization, and operator equality. |
| coverage/support/validity | Raw facts carried; filter-specific edge/support state also exists | Parent facts cannot silently become transformed validity. FLT owns transformed support/validity policy; VAL only evaluates owner policy. |
| formal weights | Implementation updates weights differently by method | Scientific meaning unavailable without method/covariance authority. |
| NOI products | Realizations transformed and empirical products may be recalculated | NOI owns uncertainty; exact FLT operator parity is prerequisite. |
| source finding/fitting | Runs downstream of filtering; Pointing may fit filtered observation maps | Source/mode package owns fitted estimand and use. |
| Beammap | May supply/use effective PSF or source-response information | Frozen SCI-BEAM owns Beammap product meaning; FLT may bind a versioned input only. |
| OOF | Configuration exposes filter controls and source fits | OOF mode interpretation is outside FLT. |
| FRUIT | Can reference raw/filtered map paths; owns iterative feedback state | FLT supplies only a named transformed product if FRUIT later admits it. |
| CAL | Supplies calibration/passband/covariance authority | FLT cannot manufacture missing calibration from units or response labels. |
| RTC | Owns temporal/timestream filters and related flags | Excluded from map-domain FLT. |

## Conflicts, Ambiguities, And Unavailable States

1. **Package identity:** one historical FLT record grouped Convolve uncertainty
   and response, while the current roadmap requires deterministic and
   inference-bearing separation.
2. **Convolve versus low-pass:** shared execution machinery does not establish
   a shared scientific purpose, transfer function, or product identity.
3. **Wiener fallback:** an unusable noise spectrum can yield a different
   low-pass response under the requested Wiener label.
4. **Filter parity for NOI:** signal and realization edge conditioning differ
   in recovered implementation behavior.
5. **Empirical-scale ownership:** historical FLT-D003 assigns one scale to FLT;
   approved SCI-NOI Stage A assigns empirical uncertainty inference to NOI.
6. **Response identity:** template normalization, unit-source convention,
   point-source peak response, aperture response, and frequency transfer are
   not one quantity.
7. **Coadd ordering:** filter-before-coadd and filter-after-coadd are not shown
   or authorized to commute.
8. **Matched estimand:** smoothing with a source-shaped template is not the
   same as a matched amplitude estimator.
9. **Learned-state source imprint:** Wiener PSD, weights, source templates,
   masks, and thresholds can depend on target data; fixed versus learned
   behavior is not fully content-bound in current provenance.
10. **Destripe:** configured but inactive, with unavailable scientific
    identity and response.
11. **Observation-resolved identity:** current requested/effective/realized
    records do not fully capture every applied operator and learned-state fact.
12. **Uncertainty/covariance:** formal diagonal propagation is not full
    covariance, and current denominator/weight labels do not establish
    precision.
13. **JINC route:** a filtered JINC product remains unavailable until both
    parent and filter routes are explicit.
14. **Downstream admission:** Beammap, Pointing, OOF, source fitting, and FRUIT
    have not generically admitted all transformed products.

## Historical Validation Material

Tests and validation records were examined only to determine that certain code
paths, product shapes, parent digests, or historical comparisons existed. No
successful or failed test was used to select scientific truth. This dossier
makes no statement that any current method conforms to the proposed taxonomy,
meets a numerical tolerance, preserves a response, propagates uncertainty
correctly, or is ready for scientific or production use.

## Recovery Conclusion

Stage A recovery and the later owner scope repair are sufficient to prepare the
sanitized exact 17-object SCI-FLT-FIXED candidate author set. The bounded scope
questions are resolved: base v0.1 is strict-linear, fixed low-pass is a
qualified convolution subtype, and full-footprint-only is the sole edge/
missing method. Stage B remains stopped until the scientific owner approves
the exact repaired bytes and explicitly launches fresh implementation-blind
authorship.
