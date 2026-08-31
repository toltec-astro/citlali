# SCI-FLT-INF scientific-owner decision ledger

Ledger identity: `SCI-FLT-INF-ODQ v0.1/r0.2`

Status: proposed ordered owner walkthrough; ODQ-001 approved and closed;
ODQ-002 through ODQ-013 open

## Decision discipline

Questions are ordered by scientific consequence. Later questions must not be
answered in a way that presumes an earlier answer. Each approved answer should
be recorded in a separate exact owner artifact before a package-local Stage A
packet is built. This holding study records the exact approved ODQ-001 answer
but does not approve any proposed answer for ODQ-002 onward.

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

For each estimand selected in ODQ-001, which package owns it?

- a map-domain FLT package;
- a source/template-amplitude estimator package at the FLT/SRC boundary;
- a posterior-reconstruction package separate from both; or
- deferred/unavailable.

Manager recommendation: do not approve `SCI-FLT-INF` as a combined package.
Use a narrow owner-selected identity for template-amplitude fields and put
selected-source/catalog estimates behind an explicit SRC boundary. Use a
separate package for a genuine posterior reconstruction.

## `SCI-FLT-INF-ODQ-003` — admitted parent and grouping

For each selected method, which exact parents are in v0.1?

- ordinary MAP observation bundle;
- ordinary MAP coadd bundle;
- JINC observation bundle;
- an exact SCI-FLT-FIXED derivative; or
- another immutable derived parent.

For each admitted parent, is learning/application observation-local or
coadd-local? Is any cross-observation combination authorized?

Manager recommendation: begin with at most one parent/grouping. Do not claim
observation/coadd equivalence or MAP/JINC portability. Frozen numerical-parent
gates remain absolute.

## `SCI-FLT-INF-ODQ-004` — covariance/noise and parent-coefficient authority

What exact object supplies the noise/covariance model required by the selected
estimand?

- exact parent covariance;
- an owner-authorized external/NOI covariance or spectral model;
- a relative spectral weight/preconditioner that does not support covariance
  claims; or
- unavailable.

Does any parent coefficient field enter, and if so what is its exact
nonprecision/precision meaning? How are stationarity, isotropy, radialization,
normalization, mode omissions, rank/null space, and dependence justified?

Manager recommendation: treat the current parent coefficient and shaped PSD
as scientifically unavailable until an exact authority identifies them. Do
not call the denominator inverse variance merely because a GLS form is desired.

## `SCI-FLT-INF-ODQ-005` — template, prior, and source-model identity

For a template-amplitude method, specify the exact template source,
normalization, unit-source convention, parent-beam relation, grid/WCS,
centering/subpixel phase, support, truncation/tail, array dependence, and
calibration. For a posterior method, specify the exact signal prior and
hyperparameters. Is state declared fixed, learned from the parent, learned
from a source model, or externally supplied?

Manager recommendation: do not combine kernel-derived, analytic Gaussian/
Airy, high-pass, and source-learned cases in one base method unless an exact
parameterized family preserves one estimand and all response/unit semantics.

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
templates, source models, state learning, and boundaries? Does the output
retain parent signal units, represent template amplitude, or represent a
posterior field? What beam/effective transfer and calibration covariance apply?

Manager recommendation: distinguish fixed-state response from full-procedure
response. Do not use a uniformly processed kernel as universal response for a
spatially varying estimator without an exact proof and domain.

## `SCI-FLT-INF-ODQ-009` — uncertainty and covariance products

Which uncertainty is required:

- conditional analytic covariance for fixed exact state;
- posterior covariance;
- frozen-state empirical NOI product;
- full-procedure/relearned empirical product;
- projected/structured covariance; or
- explicitly unavailable?

What is the domain, rank/null space, regularization, dependence, calibration,
and permitted consumer use? Which normalization/denominator products remain
nonprecision?

Manager recommendation: v0.1 may publish signal with typed covariance
unavailable if that is scientifically useful and honest. Never infer
independent-pixel aperture uncertainty.

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
Which standardized numerator/scale pair is authorized, and which significance
or source claims remain outside?

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
  -> ODQ-002 package split
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

ODQ-004 and ODQ-005 may be discussed in parallel after ODQ-003. ODQ-001 is
closed. Package-local Stage A is blocked until ODQ-002 is approved. Stage B is
blocked until all decisions needed by that one selected package have exact
owner answers and an exclusive implementation-blind author packet.
