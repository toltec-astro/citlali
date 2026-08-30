# SCI-FLT-FIXED v0.1 Scientist Rationale

Document identity: `SCI-FLT-FIXED-SCIENTIST-RATIONALE v0.1/draft-r0.2`

Status: implementation-blind Stage B explanatory draft; scientific-owner review required

Scientific owner: Grant Wilson

Normative import: the complete
`SCI-FLT-FIXED-NORMATIVE-CORE v0.1/draft-r0.2`, source SHA-256
`{{NORMATIVE_CORE_SHA256}}`, is incorporated without modification. If this
rationale and that core differ, the core controls.

## 1. The scientific question

SCI-FLT-FIXED answers one deliberately narrow question: given one exact
admitted map-domain parent and one exact sampled convolution resolved before
application under a fixed FLT-owned plan, what scientific product results when
that convolution is applied once on the same grid?

The answer is the transformed parent-map amplitude

```text
A_Theta,J = J_full L_Theta,
y = A_Theta,J m.
```

The restriction `J_full` matters as much as the convolution itself. It is
resolved once before payload arithmetic from immutable parent membership and
the exact required dependency footprint. The contract therefore describes a
conditional linear transformation and the exact domain on which its result is
scientific.

This is not a general filtering contract. It does not combine deterministic
convolution with Wiener inference, matched or template-amplitude estimation,
source learning, data-derived mode selection, automatic method selection, or
per-realization relearning.

## 2. Why strict linearity is the base object

Strict linearity makes the estimand and its consequences explicit after parent
membership and `J_full` are frozen. Doubling payload values on that domain
doubles the transformed signal. An impulse exposes the exact sampled kernel.
A compatible response propagates through the same `A_Theta,J`. An available
covariance propagates by the same linear map on both sides.

This is not global linearity across parents with different support, validity,
or membership. Recomputing a selector after a perturbation changes the applied
object. Response perturbations, covariance draws, noise realizations, and NOI
members therefore reuse the frozen selector; a missing required member value
becomes unavailable rather than silently changing the row domain.

An additive offset would break this compact identity. It would need its own
parent, units, support, uncertainty, reference-mode, response, covariance, NOI,
lifecycle, and failure semantics. Rather than hide those scientific questions
inside a familiar operation, v0.1 admits no additive term.

The word fixed applies to the entire scientific state, not merely to a
numerical function call. Coefficients, parameters, support, normalization,
grid, edge rule, transfer qualification, and the realized selector are frozen
before the admitted payload or any companion realization is transformed.

## 3. Fixed convolution and the low-pass qualification

Convolution is the only numerically admitted base family in v0.1, not merely
one example of an arbitrary dense linear family. `L_Theta` is its exact matrix
representation. One FLT product applies that convolution once; a final kernel
may be constructed elsewhere, but FLT makes no intermediate-composition claim.

The exact
sampled coefficients, their grid offsets, center, phase, orientation, support,
and normalization define the transformation. A family label or continuous
ideal does not.

Low-pass is a further scientific claim about that exact sampled operator. It
requires an exact coordinate-domain method, a compatible affine angular metric
or another proof of angular transfer, frequency grid, DC gain, passband,
transition, stopband or attenuation, phase, anisotropy, finite-grid and edge
limitations, and complete parameter provenance. If any are absent, it can
still be honest to say fixed convolution, but not low-pass.

The interior sampled-kernel transfer is also not automatically one global
Fourier transfer for the finite row-restricted `A_Theta,J`. Edge restriction
and missing-row selection break the symmetry required by that shortcut unless
an exact theorem establishes otherwise.

This distinction prevents smoothing language from silently promising a
frequency response that has never been stated. It also keeps a source-shaped
kernel from being confused with a matched estimator: the former transforms a
map; the latter estimates a template amplitude and belongs to a different
scientific method.

## 4. Why the full-footprint rule is the sole v0.1 edge method

Near an edge or missing sample, several numerically convenient choices are
possible. Extending the boundary, wrapping periodically, truncating the
kernel, or renormalizing the surviving coefficients can all return finite
numbers. They do not realize the same operator.

Truncation changes local gain and response. Support renormalization makes the
operator position-dependent and changes noise and covariance. Reflection,
inpainting, padding, and replacement add a boundary or missing-data model.
Those can be scientifically defensible methods, but they require their own
identities and contracts.

SCI-FLT-FIXED v0.1 therefore admits only rows with a complete authorized
footprint. An edge row can remain in parent-shaped storage for alignment, but
its scientific state is unavailable with a cause. It is not zero. This keeps
numerical convenience separate from scientific admission.

The closure distinguishes geometric storage footprint `K_geom`, exact
nonzero-coefficient support `K_nonzero`, and scientifically required dependency
set `K_req`. Base v0.1 deliberately requires every stored geometric location,
including an exact-zero coefficient; zero is a representation fact, not a
floating threshold. This makes the admission rule reproducible and prevents a
numerical tolerance from silently changing scientific support.

Identity and zero need explicit special cases. Identity requires only the same
parent row and therefore preserves the exact admitted finite parent domain.
The zero operator still inherits an exact declared parent-support row domain;
an empty arithmetic support must not grant it every storage row. Numerical
zero never erases parent identity, support, lifecycle, or unavailable
companions.

## 5. What the transformed amplitude does and does not mean

The output is the transformed parent-map amplitude. Its units follow from the
parent units and operator coefficient units. For a MAP parent, the originating
nominal-beam identity and calibration lineage remain attached.

That does not make the result a new `mJy/filtered-beam` quantity. A unit-sum
kernel preserves a constant field on complete support, but it does not by
itself preserve a point-source peak, an aperture measurement, integrated flux,
effective beam solid angle, or extended-source fidelity. Unit peak, unit
angular integral, and unit L2 normalization answer different questions.

The contract therefore keeps stored transformed amplitude separate from
response-corrected amplitude, fitted or template amplitude, uncertainty,
standardized signal, and statistical significance.

## 6. Response, beam, transfer, and modes

Response must say what was perturbed. A fixed-state linear parent response, an
already realized parent-grid response, and a parent full-procedure finite
difference with FLT fixed can each receive the identical `A_Theta,J` exactly
once, but they remain different response families. Re-resolving FLT during a
procedure response leaves SCI-FLT-FIXED.

When no compatible basis, domain, or parent response exists, the requested
transformed response remains unavailable. The kernel cannot stand in for the
whole source response because the parent response, beam, centering, edge rule,
and finite domain all matter. The exact zero operator has a local zero
derivative, but that does not manufacture a complete sky-response identity or
erase separately typed systematic uncertainty.

The local transfer and mode records describe the exact finite operator where
those descriptions are scientifically defined. Nulling a local constant mode,
for example, does not prove a whole-chain sky-mode claim. Influence similarly
describes which parent rows and coefficients contribute; it is not physical
exposure. An explicit exposure-lineage record carries the parent exposure
identity or typed absence, states that FLT creates no physical exposure, and
prevents a convolved exposure plane from being relabeled as the filtered
signal's physical exposure.

## 7. Formal covariance and empirical uncertainty

If a compatible parent covariance is available, the exact deterministic
propagation is

```text
C_out = A_Theta,J C_parent transpose(A_Theta,J).
```

Even independent parent pixels generally become correlated after convolution
because neighboring outputs reuse input pixels. A variance plane contains only
marginals. It is not full covariance, and unknown cross terms are not zero.

The contract separates the parent stochastic authority from the output
representation. A complete matrix propagated from an explicitly independent-
diagonal parent model can be complete relative to that conditional model while
remaining silent about unknown real parent cross terms. Every record therefore
names its conditional model, representation, domain, rank, null space,
omissions, supported operations, and excluded uncertainty classes.

Empirical uncertainty is a different ownership boundary. SCI-NOI applies the
exact frozen FLT state to every compatible admitted randomization and infers
the conditional uncertainty attachment. Filtering a variance, weight,
standard deviation, standardized map, or significance field is not equivalent.
Relearning a filter or selector for each member would define a different
method and cannot be mixed into this fixed-state ensemble. A member that
cannot supply a frozen required footprint becomes unavailable on that row.

## 8. Parent identity and ordering

A filtered MAP observation, a filtered MAP coadd, and a filtered JINC
observation are different scientific products. Equal-looking arrays or WCS
values do not erase their different estimands, membership, normalization,
support, response, covariance, and lineage.

SCI-FLT-FIXED performs no coaddition and assumes no filter/coadd commutation.
Such equality can hold only under explicit compatible operators, grids,
weights, boundaries, normalization, support, and response facts. A future
bounded proof could establish one relation without making it universal.

The admitted MAP and JINC boundaries also preserve upstream unavailability.
At this launch state, the ordinary numerical MAP and TolTEC JINC routes remain
gated upstream. Defining their successor types does not create numerical
parents.

## 9. Lifecycle, atomicity, and honest absence

Requested, effective, resolved, applied, complete publication candidate,
publication decision, and realized are not synonyms. The
plan may disable the route. Required state may be unavailable. Application may
fail. A transformed array may exist before its required response, support,
validity, covariance-state, lineage, and failure records form an atomic bundle.

Publication policy evaluates the complete candidate, not a product that is
already realized. Disabled yields `not_produced`. Identity and zero operators
become real separately parented products only after the same candidate and
publication sequence. A zero signal is not an unavailable row, a disabled
route, zero total uncertainty, or infinite precision.

Honest absence is part of the product. A base transformed signal may carry an
explicit unavailable response or covariance state where the role permits it.
A response-qualified or covariance-qualified request cannot. Missing required
records never become placeholders.

The SCI-NOI product is not an atomic FLT role. FLT carries only a typed NOI
attachment-state relation. A separately parented NOI product may be attached
later without deciding, reopening, or mutating FLT completeness.

## 10. Use and ownership limits

SCI-FLT-FIXED owns its local transformation, product, response state, support,
validity, deterministic covariance state, lifecycle, and failure behavior. MAP
or JINC continues to own the parent estimand. CAL owns absolute calibration.
SCI-NOI owns empirical uncertainty inference. Beam, source, Pointing, OOF, and
FRUIT interpretations remain with their respective future or upstream owners.

Three policy domains prevent bundle, row, and publication questions from being
collapsed. Their draft records use VAL-congruent applicability, eligibility,
realization, missing/conflict, lifecycle, and consumer-action states. They are
not owner-approved Registry entries, and this draft claims no Registry
evaluation.

The transformed product is therefore not a generic downstream admission
ticket. A Beammap, Pointing, OOF, source-fit, catalog, NOI, or FRUIT consumer
must provide an exact use policy. SCI-VAL may bind and evaluate an approved
policy, but it does not invent producer facts or FLT policy.

## 11. Reading the falsifiable predictions

The core predictions turn the contract into observable distinctions:

- identity, zero, scaling, constant, impulse, signed, and zero-sum cases test
  the declared linear operator and normalization;
- footprint, missing-value, non-finite, and deferred-edge cases test the exact
  scientific row domain;
- response and covariance cases test composition without strengthening absent
  parent claims;
- WCS and parent-order cases test scientific identity rather than array
  similarity;
- NOI cases test exact fixed-state parity and rejection of relearning;
- lifecycle cases test disabled, unavailable, identity, zero, realized, and
  failed distinctions; and
- low-pass completeness and an independent sampled-transfer evaluation test
  whether a qualified transfer claim is actually supported;
- exact-zero coefficient and zero-operator support cases test required-
  dependency semantics independently of arithmetic nonzero support.

Passing these predictions would be evidence relevant to a later conformity or
validation activity. This rationale reports no such result.

## 12. Resolution ownership and numerical evidence

"Externally resolved" means that one exact FLT-owned plan is complete before
application. It does not mean an unnamed producer owns FLT policy or the
transformation. A final kernel can arrive from elsewhere while FLT continues
to own admission, the frozen selector, application, product identity,
publication, and failure semantics.

Exact coefficients define scientific identity, but finite arithmetic needs a
predeclared way to compare a candidate with an independent exact or high-
precision oracle. The numerical-policy draft covers absolute, relative, and
near-zero behavior, cancellation, conditioning, covariance, parallel
agreement, overflow, underflow, non-finite values, and simultaneous row-level
decisions. Freezing those rules before a result is seen prevents the tolerance
from adapting to a failure. The policy is an evidence protocol, not new filter
science or a validation result.

## 13. Stage B nonclaims

This rationale explains the draft scientific contract only. It makes no
implementation-conformity, algorithm-change, validation, calibration,
achieved-response, achieved-covariance, numerical-adequacy, performance,
readiness, scientific-freeze, production, or Unity claim.
