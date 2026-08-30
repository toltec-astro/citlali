# SCI-FLT-INF scientific-owner decision ledger

Ledger identity: `SCI-FLT-INF-ODQ v0.1/r0.1`

Status: proposed ordered owner walkthrough; all questions open

## Decision discipline

Questions are ordered by scientific consequence. Later questions must not be
answered in a way that presumes an earlier answer. Each approved answer should
be recorded in a separate exact owner artifact before a package-local Stage A
packet is built. This holding study does not approve any proposed answer.

## `SCI-FLT-INF-ODQ-001` — estimand of the existing full path

What scientific quantity is the existing template/noise/weight-dependent full
map path intended to estimate?

1. **Normalized template-amplitude field**: at each admitted location, the
   amplitude of one exact template under one exact weighting/covariance model.
2. **Posterior/Wiener reconstructed sky field**: an explicitly prior-bearing
   posterior quantity with exact likelihood, prior, response, and posterior
   covariance.
3. **Both, as separate methods/products**: retain a template-amplitude method
   and commission a distinct posterior method.
4. **Neither**: retain the current path only as non-authoritative diagnostic or
   retire it from future scientific-contract scope.

Manager recommendation: select option 1 for recovery of the existing path,
because its apparent numerator/denominator structure is amplitude-estimator-
like, and reserve any genuine Wiener/posterior product for option 3 as a
separate future package. This is an implementation-informed recommendation,
not evidence sufficient to approve the science.

Consequence: no package name, operator derivation, response, uncertainty, or
product table can be finalized before this answer.

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

ODQ-004 and ODQ-005 may be discussed in parallel after ODQ-003. Package-local
Stage A is blocked until ODQ-001 and ODQ-002 are approved. Stage B is blocked
until all decisions needed by that one selected package have exact owner
answers and an exclusive implementation-blind author packet.
