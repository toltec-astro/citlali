# SCI-FLT-INF scientific-owner decision ledger

Ledger identity: `SCI-FLT-INF-ODQ v0.1/r0.5`

Status: proposed ordered owner walkthrough; ODQ-001 through ODQ-003 approved
and closed; ODQ-004 author-delegated; ODQ-005 through ODQ-013 open

## Decision discipline

Questions are ordered by scientific consequence. Later questions must not be
answered in a way that presumes an earlier answer. Each approved answer should
be recorded in a separate exact owner artifact before a package-local Stage A
packet is built. This holding study records the exact approved ODQ-001 through
ODQ-003 answers and the exact ODQ-004 author delegation but does not approve a
noise/covariance option or any proposed answer for ODQ-005 onward.

## `SCI-FLT-INF-ODQ-001` — estimand of the existing full path

Status: **approved and closed** by
[`SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md`](SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md).

The historical Citlali full path is scientifically an **optimal matched-
template amplitude estimator**. It estimates the supplied template's amplitude
as a function of map position using the declared noise model. With a
point-source-response kernel it yields a matched point-source amplitude field;
with another scientifically defined kernel it yields the amplitude field of
that specified template/shape. Exact normalization must return an unbiased
estimate of a matching signal's amplitude, subject to the stated noise,
support, edge, missing/nonfinite, validity, response, and method assumptions.

It is not a prior-bearing posterior/Wiener sky reconstruction. Historical
`Wiener filter` terminology may remain only where compatibility requires it.
Ordinary source-shaped convolution is a separate deterministic operation, not
the noise-weighted normalized matched estimator. Any future genuine
Wiener/posterior reconstruction requires its own scientific contract.

This decision selects the estimand, not a package name, numerical operator,
noise/covariance model, template instance, optimality proof, support rule,
response, uncertainty, product bundle, or Stage B launch.

## `SCI-FLT-INF-ODQ-002` — package ownership and split

Status: **approved and closed** by
[`SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md`](SCIENTIFIC_OWNER_ODQ_002_APPROVAL_2026-08-30.md).

The selected method belongs to a narrow map-domain filtering package. It owns
the matched-filter operation and publishes a matched-filtered version of the
exact admitted input map product or products, preserving their applicable
map-domain structure and semantics. The local mathematical identity remains
the ODQ-001 optimal matched-template amplitude estimator, including exact
noise weighting and amplitude-unbiased normalization under the declared
assumptions; the published product role is nevertheless a filtered map.

This package does not own or require source detection, candidate selection,
catalog construction, peak interpretation, deblending, source fitting, or
other source-analysis behavior. No source-estimation package or SRC ownership
boundary is introduced. A later independent source-analysis contract may
consume matched-filtered maps if separately authorized.

A genuine prior-bearing Wiener/posterior reconstruction remains a distinct
deferred method and must not enter this package. ODQ-002 selects ownership and
the top-level product role, not the final package name, parent, operator,
template, noise model, support, response, uncertainty, lifecycle, or Stage B
launch.

## `SCI-FLT-INF-ODQ-003` — admitted parent and grouping

Status: **approved and closed** by
[`SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md`](SCIENTIFIC_OWNER_ODQ_003_APPROVAL_2026-08-31.md).

V0.1 admits both one exact immutable normalized ordinary-MAP observation
bundle and one exact immutable normalized ordinary-MAP coadd bundle. They are
distinct parent/grouping and realized product identities. Observation-map
filtering is observation-local; coadd-map filtering is coadd-local and binds
the exact contributing-observation set and coadd generation.

No equivalence, commutation, or cross-observation combination is approved.
Filtering a coadd is not presumed equivalent to filtering its contributing
observations and combining the results. The package performs no independent
coaddition. JINC observation bundles, SCI-FLT-FIXED derivatives, and other
derived parents are deferred and unavailable. Frozen PTC coefficient and
numerical `coverage_cut` gates remain absolute.

This decision selects parent roles and grouping, not the noise/covariance
authority, operator, normalization, state, support, response, uncertainty,
product details, or Stage B launch.

## `SCI-FLT-INF-ODQ-004` — covariance/noise and parent-coefficient authority

Status: **author-delegated; no option selected** by
[`SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md`](SCIENTIFIC_OWNER_ODQ_004_AUTHOR_DELEGATION_2026-08-31.md).

A future implementation-blind author must develop a bounded option set in both
the Scientific Rationale and Contract and the Engineering Conformance
Specification. Both views must share exact option identities and consequences.
The owner must select or dispose of an authored option before freeze or any
numerical route is authorized.

The owner supplies one historical candidate for scientific examination:
Citlali has used a **radially symmetrized average map noise PSD**. This is not a
selected default, covariance authority, proof of stationarity/isotropy or
optimality, or permission to inherit historical mechanics. The author must
define the population/domain, averaging and radialization ordering, Fourier/
WCS and unit conventions, normalization, support/edge/window behavior,
missingness, rank/null space, approximation, and state dependence for any
option using those ideas.

Options must separately account for observation-map and coadd-map parents,
classify whether they supply covariance or only a weaker spectral weighting,
and state whether/how a parent coefficient participates. No coefficient may
be called precision or covariance from naming, unit, positivity, or historical
use. Typed unavailability is required when an option's object or assumptions
are absent.

## `SCI-FLT-INF-ODQ-005` — template and kernel identity

For the matched-filter method, specify the exact template source,
normalization, unit-source convention, parent-beam relation, grid/WCS,
centering/subpixel phase, support, truncation/tail, array dependence, and
calibration. Is template state declared fixed, learned from the parent, or
externally supplied?

Manager recommendation: do not combine kernel-derived, analytic Gaussian/
Airy, and high-pass cases in one base method unless an exact parameterized
family preserves one estimand and all response/unit semantics. Source fitting
and source-learned template state are outside this package.

## `SCI-FLT-INF-ODQ-006` — exact operator, approximation, and regularization

What exact mathematical operator or estimator is authoritative? Which
approximations are permitted, how is their achieved error bounded, and what
are the typed consequences of convergence, iteration/tail caps, floors,
clipping, singularity, or nonpositive normalization?

Manager recommendation: a numerical zero at an unresolved normalization is
unavailable/null, not a scientific amplitude zero. Each approximation or
regularization that changes the estimand or response is a versioned method.

## `SCI-FLT-INF-ODQ-007` — edge, missing, nonfinite, and learned support

Does v0.1 admit only full-footprint inputs, or a separate adaptive edge method?
If adaptive conditioning is admitted, define the learning facts, background/
fill, taper, influence footprint, eroded admitted region, response,
covariance, validity, member parity, and failure.

Manager recommendation: keep adaptive edge/background conditioning separate
from the base estimator. If later admitted, consider re-adopting historical
fill-as-numerical-device plus eroded-science-region policy under a new exact
owner record, after resolving member parity.

## `SCI-FLT-INF-ODQ-008` — response, units, beam, and output interpretation

What is the response of the exact selected estimator to declared modes,
templates, state learning, and boundaries? Which applicable parent map-domain
semantics are preserved, and what signal units or template-amplitude units,
beam/effective transfer, and calibration covariance apply to the filtered
map?

Manager recommendation: distinguish fixed-state response from full-procedure
response. Do not use a uniformly processed kernel as universal response for a
spatially varying estimator without an exact proof and domain.

## `SCI-FLT-INF-ODQ-009` — uncertainty and covariance products

Which uncertainty is required:

- conditional analytic covariance for fixed exact state;
- frozen-state empirical NOI product;
- full-procedure/relearned empirical product;
- projected/structured covariance; or
- explicitly unavailable?

What is the domain, rank/null space, regularization, dependence, calibration,
and permitted consumer use? Which normalization/denominator products remain
nonprecision?

Manager recommendation: v0.1 may publish signal with typed covariance
unavailable if that is scientifically useful and honest. Never infer
independent-pixel aperture uncertainty. Posterior covariance is outside the
selected package.

## `SCI-FLT-INF-ODQ-010` — learned-state and NOI generation graph

For every consequential state component—template/prior, covariance/PSD,
edge/background/support, approximation/selection, normalization, and method
choice—classify it as:

- declared fixed;
- learned once from the real parent and frozen;
- updated from prior NOI in a new immutable successor generation;
- relearned separately per admitted member;
- not applicable; or
- unavailable.

Manager recommendation: start with learned-once/frozen if that matches the
selected science, because it is the narrowest recoverable current lifecycle.
Commission a per-member method only with an exact complete rerun graph and a
separate estimand; never mix the member populations.

## `SCI-FLT-INF-ODQ-011` — method selection, fallback, and data-thresholded modes

Does v0.1 permit any automatic alternative selection? If so, define the
selector and realized-method product identity. What happens when PSD/template/
parent/learning/approximation is unavailable? Is data-thresholded spectral
selection/destriping in scope?

Manager recommendation: no silent fallback. Fail the requested method closed
unless a separately authorized selector realizes a separately named method.
Defer inactive data-thresholded destriping to its own Stage A package.

## `SCI-FLT-INF-ODQ-012` — NOI coefficient calibration and standardized products

May a NOI-derived global scalar create a new versioned normalization/
coefficient product? If yes, what exact parent coefficient and NOI estimand
does it calibrate, what region/statistic defines it, and what does it not mean?
Which standardized numerator/scale pair is authorized, and which significance,
detection, peak, or catalog claims remain outside?

Manager recommendation: route this as a separate FLT/NOI derived-product
contract. Preserve immutable formal/parent coefficients, create a successor
coefficient rather than mutation, and prohibit precision/significance labels
without exact authority.

## `SCI-FLT-INF-ODQ-013` — product bundle, lifecycle, VAL, and failure

For each selected method, approve the required/conditional/optional product
roles, atomic completion, disabled/unavailable/failed/superseded states,
requested/effective/resolved/realized identities, permitted consumers, and
owner-authored VAL profiles.

Manager recommendation: require an immutable parent reference, method/state
identity, output estimand, normalization, response, support/null/validity,
covariance availability, approximation/selection record, NOI generation, and
atomic failure record. Diagnostics may remain optional but cannot substitute
for a missing required scientific role.

## Decision dependency graph

```text
ODQ-001 estimand
  -> ODQ-002 map-domain package and filtered-map role
     -> ODQ-003 parent/grouping
        -> ODQ-004 covariance/noise
        -> ODQ-005 template/prior
           -> ODQ-006 exact operator/approximation
              -> ODQ-007 edge/support method
              -> ODQ-008 response/units/beam
              -> ODQ-009 uncertainty/covariance
                 -> ODQ-010 NOI generation graph
                    -> ODQ-011 selector/fallback/mode selection
                    -> ODQ-012 coefficient/STD derivatives
                       -> ODQ-013 product/VAL/lifecycle
```

ODQ-001 through ODQ-003 are closed, and ODQ-004 is author-delegated without an
option selection. ODQ-005 is the next owner gate. Stage B is blocked until all
remaining pre-author decisions have exact owner answers and an exclusive
implementation-blind author packet containing the ODQ-004 assignment. Freeze
and numerical authorization remain blocked until the owner disposes of the
authored ODQ-004 options.
