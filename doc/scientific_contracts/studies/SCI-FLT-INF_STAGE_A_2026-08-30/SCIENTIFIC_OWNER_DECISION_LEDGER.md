# SCI-FLT-INF scientific-owner decision ledger

Ledger identity: `SCI-FLT-INF-ODQ v0.1/r0.8`

Status: proposed ordered owner walkthrough; ODQ-001 through ODQ-003 approved
and closed; ODQ-004 author-delegated; ODQ-005 approved and closed; ODQ-006
approved and closed with quantitative conformance-envelope alternatives
author-delegated; ODQ-007 approved and closed; ODQ-008 through ODQ-013 open

## Decision discipline

Questions are ordered by scientific consequence. Later questions must not be
answered in a way that presumes an earlier answer. Each approved answer should
be recorded in a separate exact owner artifact before a package-local Stage A
packet is built. This holding study records the exact approved ODQ-001 through
ODQ-003 and ODQ-005 answers and the exact ODQ-004 author delegation but does
not approve a noise/covariance option. It also records the exact ODQ-006
reference-operator decision and quantitative author delegation but does not
approve a conformance-envelope option. It records the exact ODQ-007 complete-
support decision but does not approve any proposed answer for ODQ-008
onward.

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

Status: **approved and closed** by
[`SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md`](SCIENTIFIC_OWNER_ODQ_005_APPROVAL_2026-08-31.md).

Each base-v0.1 application uses one exact immutable, scientifically declared
template-response product representing expected parent-map response per unit
of the declared amplitude `A`. Its scaling defines the amplitude convention:
`unit(t) = unit(m) / unit(A)`. Its identity binds source authority, compatible
parent, amplitude/signal/template units, grid/WCS/frame, centering/subpixel
phase, support/truncation/tails, array dependence, parent-beam relationship,
calibration, validity, and provenance. No peak, integral, flux-density, beam,
or other amplitude convention is inferred from a generic kernel label.

Admitted sources are the exact point-source-response product bound to the
immutable parent bundle or another explicitly supplied scientific template-
response product. Gaussian/Airy construction is admissible only as a producer
of that same fully specified materialized product before application. Base
v0.1 excludes template learning/selection from the target parent, sources,
candidates, populations, or NOI members. The historical high-pass/delta case
requires a separate future scientific method.

Observation-parent and coadd-parent compatibility remain separately declared;
no equality or reuse is presumed. Template discretization and approximation
consequences pass to ODQ-006, while response, beam, calibration covariance,
uncertainty, edge, and NOI details remain with their later ordered decisions.

## `SCI-FLT-INF-ODQ-006` — exact operator, approximation, and regularization

Status: **approved and closed at the reference-operator and realization-policy
level; quantitative conformance-envelope alternatives author-delegated** by
[`SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md`](SCIENTIFIC_OWNER_ODQ_006_APPROVAL_2026-08-31.md).

Conditional on the exact realized ODQ-004 weighting object `Q_x`, ODQ-005
template `t_x`, ODQ-007 support, and declared discrete conventions, the
authoritative reference estimator is

```text
N(x) = <t_x, Q_x m_x>
D(x) = <t_x, Q_x t_x>
A_hat(x) = N(x) / D(x).
```

When `Q_x` is admitted inverse covariance, the complete assumptions support
the optimal GLS matched-estimator claim. A weaker ODQ-004 weighting object
requires correspondingly weaker optimality and uncertainty claims; ODQ-006
selects no noise model.

Exact evaluation is conformant. Approximate evaluation is permitted only
inside a scientifically selected envelope bounding effects on normalization,
matching-template amplitude response, support/null behavior, and uncertainty.
The future implementation-blind author must develop bounded quantitative
envelope alternatives with shared identities in both contract views. Owner
disposition is required before freeze or approximate execution.

Regularization defining `Q_x`, its modes, or null space is ODQ-004 scientific
state. Any approximation or other regularization changing the operator beyond
the selected envelope is a separate versioned method or unavailable. `N` and
`D` must be finite and `D` strictly positive on admitted support. Empty/invalid
support, weighting-null templates, singular/unresolved normalization,
nonfinite/nonpositive `D`, or an unmet convergence/error bound is typed null,
unavailable, or failed—never scientific amplitude zero. Iteration and tail
caps are not success without the selected bound.

## `SCI-FLT-INF-ODQ-007` — edge, missing, nonfinite, and learned support

Status: **approved and closed** by
[`SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-08-31.md`](SCIENTIFIC_OWNER_ODQ_007_APPROVAL_2026-08-31.md).

Base v0.1 admits only complete-support output locations. The complete declared
local or nonlocal influence support of the template, weighting operator,
ODQ-006 realization, and boundary convention must be in the exact parent
domain, admitted for this use, finite and available where required, and valid
under every required predicate. The entire stored map need not be globally
valid for a bounded local operator; support is decided per output location.

Base v0.1 performs no partial-support or truncated estimation, support
renormalization, boundary extension, interpolation/imputation/replacement,
background estimation/subtraction, learned taper, or signal-derived support
selection. Parent-authoritative support/validity facts may be consumed as
inputs. An affected output is unavailable, never zero or a successful partial
estimate.

Padding or fill is a numerical device only when conservative erosion over the
complete influence support establishes that no admitted output depends on an
invented value. A nonlocal operator cannot claim an unaffected interior from a
finite guard band without that proof. Adaptive edge/background conditioning is
deferred to a separately identified future method with complete learning,
response, covariance, NOI, validity, lifecycle, and failure authority.

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

ODQ-001 through ODQ-003 and ODQ-005 through ODQ-007 are closed, and ODQ-004 is
author-delegated without an option selection. ODQ-006 also delegates
quantitative conformance-envelope alternatives without selecting one. ODQ-008
is the next owner gate. Stage B is blocked until all remaining pre-author
decisions have exact owner answers and an exclusive implementation-blind
author packet containing the ODQ-004 assignment and ODQ-005 through ODQ-007
approvals. Freeze and numerical authorization remain blocked until the owner
disposes of the authored ODQ-004 and ODQ-006 option sets.
