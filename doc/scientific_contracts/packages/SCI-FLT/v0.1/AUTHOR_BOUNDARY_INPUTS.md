# SCI-FLT v0.1 Sanitized Boundary Inputs

Status: proposed future author input; scientific-owner approval required

## Program Adherence And Prior-Work Recovery

This file restates only the minimum applicable scientific boundaries recovered
from frozen or approved package records. It does not expose implementation,
configuration, audits, repairs, tests, validation, performance, or the current
SCI-NOI Stage B draft. It neither changes the cited authorities nor makes an
unavailable numerical route available.

## Parent Map Boundary

### Ordinary MAP parent

The parent is an immutable, content-bound MAP scientific product with its exact
estimand, normalization, grid/frame, map unit, response identity, support,
validity, required companions, coefficient meaning, lifecycle, and covariance
availability. FLT creates a successor product and cannot mutate the parent or
strengthen an unavailable/conditional MAP claim.

MAP weight or normalization-coefficient labels do not, without their own
authority, establish inverse variance, precision, uncertainty, statistical
significance, confidence, or validity. A finite stored value is not by itself
scientific admission.

Observation MAP products and raw coadds are different parents. Their
registration, membership, response, support, and covariance identity are part
of the parent boundary.

### JINC parent

JINC is a separate signed-estimator parent with its own normalization, response,
support, validity, and covariance semantics. Ordinary positive-coefficient MAP
rules cannot be imported by analogy. Where the exact JINC numerical parent,
response, or covariance is unavailable, the corresponding filtered route or
claim remains unavailable.

## Calibration And Response Boundary

CAL owns absolute calibration, passband/color correction, calibration
validity, and calibration covariance. A map unit, filter normalization, or
unit-source response cannot manufacture absent CAL authority.

A transformed response object must name what it responds to. Point-source peak
response, aperture response, signed integral, beam solid angle, spatial or
frequency transfer, and a fitted template amplitude are distinct. Beammap and
source-fitting authorities retain their own effective-PSF and fitted-source
interpretation.

## SCI-NOI Boundary

The following approved Stage A rules control:

1. The appropriate scientific process—in this tranche, the exact FLT method—
   owns the transformation purpose, algorithm/method identity, parameters and
   learned state, order, domain, support, edge and missing-data behavior,
   normalization, units, response, lifecycle, and failure policy.
2. NOI neither chooses nor defines the transformation.
3. To estimate uncertainty for an exact transformed scientific product, NOI
   applies exactly that owner-defined transformation to every compatible
   admitted randomization.
4. No commutation, relocation, substitution, or same-name equivalence may be
   inferred. Exact owner authority and transformation parity must be bound.
5. An owner-frozen Wiener transformation can enter this fixed-state route.
6. Learning, selecting, or updating a transformation with NOI information
   creates an immutable successor transformation, science, GEN, and UNC
   generation.
7. Learning separately for every randomization is a distinct method. Fixed-
   state and per-member-relearned members cannot be mixed.
8. Unknown covariance is not zero. A pointwise conditional second moment is not
   automatically covariance, precision, or statistical significance.

SCI-NOI owns ensemble design, realization identity, conditional uncertainty/
covariance inference, inverse conditional scale, standardized-signal meaning,
completion policy, persistence/sufficiency mode, and uncertainty attachment.
FLT owns only the exact transformation and its scientific output/response/
support/validity identity.

## Source, Beammap, Pointing, And OOF Boundary

A deterministic convolution by a source-shaped kernel produces a transformed
map. It does not by itself estimate source amplitude, flux, position,
morphology, or a pointing/focus correction.

- SCI-BEAM retains frozen Beammap effective-PSF, source-fit, calibration, and
  sensitivity ownership.
- A future source-fitting contract owns a fitted-source estimand, nuisance/
  background model, position/morphology, fit validity, and fit uncertainty.
- Pointing and OOF contracts own their mode-specific scientific interpretation
  even when they consume a transformed map or reuse an upstream gridding
  operator.

These consumers must name the exact admitted FLT product and response. FLT
does not authorize their use.

## FRUIT Boundary

FRUIT owns source-model construction, subtract/add behavior, learning,
recurrence, stopping, restart, selection, response, support, validity,
lifecycle, and failure. A fixed residual or terminal-product uncertainty is
conditional on the frozen FRUIT state. NOI-informed continuation creates a
successor generation, and per-member replay is a distinct method. Fixed-state
and replayed ensembles cannot be mixed.

FLT may define a transformed product that FRUIT could later admit, but it does
not choose the feedback product, define the FRUIT estimand, or establish an
iterative route.

## VAL Boundary

FLT owns any filter-specific support, validity, response-use, uncertainty-use,
or publication policy. VAL may register and evaluate an exact owner-approved
profile; it does not author that policy or make an unavailable product
available.

## Honest Absence

Unavailable parent response, covariance, calibration, operator state, learned
state, support, validity, or consumer admission remains explicitly unavailable.
It is not numerical zero, false precision, a default fallback, or evidence for
a same-name method.
