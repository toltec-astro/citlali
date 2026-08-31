# SCI-FLT-FIXED v0.1 Scientist Rationale

Document identity: `SCI-FLT-FIXED-SCIENTIST-RATIONALE v0.1/freeze-candidate`

Status: implementation-blind conditional scientific-owner freeze-candidate rationale; owner signature required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

Normative import: the complete
`SCI-FLT-FIXED-NORMATIVE-CORE v0.1/freeze-candidate`, source SHA-256
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

The parent facts and the numerical signal are not the same domain. The parent
fact domain `S_parent_fact` contains exact row identity and typed state. The
scientific function `m : D_m -> R` exists only where an available finite real
payload exists, and `y : S_out -> R` exists only on admitted output rows. A
stored missing, unavailable, or non-finite value is therefore a fact, not a
number that the scientific operator may evaluate.

The restriction `J_full` matters as much as the convolution itself. It is
resolved once before payload arithmetic from immutable parent membership,
typed facts establishing `D_m`, and the exact required dependency footprint.
Membership in `D_m` is established before `m_q` is evaluated. The contract
therefore describes a conditional linear transformation and the exact domain
on which its result is scientific.

This is not a general filtering contract. It does not combine deterministic
convolution with Wiener inference, matched or template-amplitude estimation,
source learning, data-derived mode selection, automatic method selection, or
per-realization relearning.

The complete scientific route is deliberately one-way:

```text
exact parent signal role
  -> exact parent-row admission
  -> frozen J_full
  -> one sampled convolution applied once
  -> immutable FLT bundle with FLT-NOI-COMPATIBILITY
  -> optional separate SCI-NOI child referencing FLT
```

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
grid, edge rule, and transfer qualification are frozen before application.
Parent amplitudes never tune those plan facts. The selector is then resolved
once from declared immutable `S_parent_fact`, row membership, exact row
admission, typed facts defining `D_m`, support, and required predicates.
Reading the declared typed row-state fact for an authorized required location
is structural screening, not evaluation of `m_q`, plan tuning, or convolution
arithmetic. Every compatible companion reuses that resolved selector.

## 3. Fixed convolution and the low-pass qualification

Convolution is the only numerically admitted base family in v0.1, not merely
one example of an arbitrary dense linear family. `L_Theta` is its exact matrix
representation. One FLT product applies that convolution once; a final kernel
may be constructed elsewhere, but FLT makes no intermediate-composition claim.

The exact
sampled coefficients, their grid offsets, center, phase, orientation, support,
and normalization define the transformation. A family label or continuous
ideal does not.

That scientific operator identity is distinct from its serialization. The
canonical offset-to-coefficient map, scientific domains, support, response,
covariance, and lifecycle define the scientific digest. `K_store`, dense or
sparse layout, field order, bytes, compression, and container define a
separate representation digest and generation. Changing only representation
may create a new representation artifact, but it leaves the scientific
operator, FLT product identity, and scientific generation unchanged.

The base signal, coefficients, and map operator are real-valued. Every
coefficient must be finite, real, unit-typed, canonically represented, and
content-bound. Missing, non-finite, complex, unrepresentable, or conflicting
coefficients stop plan resolution. A complex `H(nu)` may represent the
frequency response of this real operator; it does not turn the map signal or
covariance rule into a complex-valued method.

Low-pass is a further scientific claim about that exact sampled operator. Its
plan binds the transform sign and normalization; coordinate and frequency
units; origin, ordering, signed and Nyquist treatment; frequency grid;
response quantity; linear or decibel attenuation; band-region geometry;
phase branch; anisotropy; WCS metric; DC gain; limitations; and provenance.
If any fact is absent or a different transform convention is used, it can
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

The owner disposition separates scientific geometry `K_geom_science` from
nonauthoritative serialization `K_store`. For the ordinary method, convolution
arithmetic is summed over exactly `K_nonzero = K_req`, where exact nonzero is
decided from the canonical coefficient representation and never from a
tolerance. `K_geom_science` describes geometry; it is not the arithmetic
dependency set. An exact-zero offset contributes no arithmetic term, payload
dependency, influence, covariance, or row exclusion, and its parent payload is
never dereferenced. Dense, sparse, cropped, and zero-padded encodings of the
same canonical kernel therefore have their own representation identities while
leaving `S_out`, response, covariance, scientific operator identity, FLT
product identity, and scientific generation unchanged. A required zero-valued
offset needs a separately named scientific method independent of storage.

Identity and zero need explicit special cases. Identity requires only the same
parent row and therefore preserves the exact admitted finite parent-signal
domain. The zero operator has empty `K_nonzero_zero` and `K_req_zero`, while
its `S_out_zero` is independently the exact admitted finite parent-signal row
domain under its request and predicates. An empty arithmetic set cannot grant
it every storage row. Numerical zero never erases parent identity, support,
lifecycle, or unavailable companions.

For a requested nonzero convolution, an empty `S_out` has a different meaning.
The application records `applied_no_scientific_output_support` and constructs
a complete `no_output_support_candidate` containing the attempted-domain proof
and bound evidence but no realized `FLT-SIG` or atomic bundle. For the
publication use, request is requested, applicability is applicable,
eligibility is ineligible, realization is `not_produced`, and the cause is
`no_full_footprint_output_rows`. It is not not-requested, disabled, execution
failure, decision unavailable, identity, the zero operator, or a successfully
realized empty product.

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

A full-procedure difference is valid only when baseline and perturbed parent
products define one exact compatible difference on the frozen domain. A change
in membership, availability, WCS, quantity, or support makes affected response
rows unavailable while preserving `J_full` and the parent state-change record;
equal shape is not enough.

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
because neighboring outputs reuse input pixels. More generally,

```text
Var(y_i) = sum_j A_ij^2 Var(m_j)
           + 2 sum over j<k of A_ij A_ik Cov(m_j,m_k).
```

Marginal parent variances therefore do not establish independence and usually
do not determine filtered marginals. Complete covariance, an explicit
independent-diagonal model, marginal-only information, a structured or partial
model, and unavailable authority each permit different exact results. A
diagonal-contribution diagnostic is not variance or uncertainty.

There is one exact marginal-only edge case. If an output row has exactly one
nonzero coefficient `A_ij`, its conditional marginal is
`A_ij^2 Var(m_j)` without any independence assumption. A row mixing two or
more parent variables remains unavailable or explicitly partial when parent
cross terms are unknown. The one-row result authorizes no cross-row covariance;
the exact-zero row retains its separately typed zero parent-payload
contribution.

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

Requested, effective, resolved, applied, complete publication disposition
candidate, publication decision, and realized are not synonyms. The
plan may disable the route. Required state may be unavailable. Application may
fail. A transformed array may exist before its required response, support,
validity, covariance-state, lineage, and failure records form an atomic bundle.

Publication policy evaluates either exact candidate variant, not a product
that is already realized. A `product_candidate` contains the complete atomic
bundle. A `no_output_support_candidate` contains the exact attempted-domain
proof and application evidence but no realized `FLT-SIG` and is not an atomic
bundle. Disabled yields `not_produced`. Identity and zero operators become
real separately parented products only after the product-candidate and
publication sequence. A zero signal is not an unavailable row, a disabled
route, zero total uncertainty, or infinite precision.

Honest absence is part of the product. A base transformed signal may carry an
explicit unavailable response or covariance state where the role permits it.
A response-qualified or covariance-qualified request cannot. Missing required
records never become placeholders.

The SCI-NOI product is not an atomic FLT role. Immutable
`FLT-NOI-COMPATIBILITY` records only publication-time FLT identity,
fixed-state compatibility, request, and typed availability. It contains no
future NOI identity. A later NOI child references FLT; an optional separately
versioned reverse relation cannot decide, reopen, or mutate FLT completeness.
A recorded `not_requested_at_FLT_publication` state is only historical
provenance: it does not bar a later independent NOI request. The child owns its
own request, applicability, eligibility, realization, generation, and failure.

## 10. Use and ownership limits

SCI-FLT-FIXED owns its local transformation, product, response state, support,
validity, deterministic covariance state, lifecycle, and failure behavior. MAP
or JINC continues to own the parent estimand. CAL owns absolute calibration.
SCI-NOI owns empirical uncertainty inference. Beam, source, Pointing, OOF, and
FRUIT interpretations remain with their respective future or upstream owners.

Three policy domains prevent bundle, parent-row, and publication questions
from being collapsed. Bundle and parent-row profiles decide their named inputs;
FLT constructs `J_full` and `S_out`. Publication policy defines a disposition
and action. VAL may produce a decision artifact, but only the FLT publisher
acts and establishes realization and local validity. The profile records first
drafted at r0.3 remain unregistered and are not owner-approved Registry
entries; this conditional freeze candidate binds their amended exact bytes
without claiming a Registry evaluation or numerical route.

The transformed product is therefore not a generic downstream admission
ticket. A Beammap, Pointing, OOF, source-fit, catalog, NOI, or FRUIT consumer
must provide an exact use policy. SCI-VAL may bind and evaluate an approved
policy, but it does not invent producer facts or FLT policy.

The source-preflight route status is:

```text
Route                           freeze-candidate disposition
generic contract                conditional owner signature required
MAP observation parent          typed route; numerical parent unavailable
MAP coadd parent                typed route; numerical parent unavailable
JINC parent                     typed route; numerical parent unavailable
base signal                     defined for an exact admitted parent
low-pass qualification          conditional on complete exact convention
response-qualified product      conditional on exact compatible response
covariance-qualified product    conditional on exact compatible authority
NOI compatibility               immutable FLT compatibility only
late NOI request                allowed child route; no FLT mutation
real coefficient gate           invalid coefficient makes plan unavailable
empty nonzero output support     not produced with exact typed cause
profile registration            not registered and not Registry-evaluated
implementation assessment       not performed and no claim
```

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
  failed distinctions;
- low-pass completeness and an independent sampled-transfer evaluation test
  whether a qualified transfer claim is actually supported;
- exact-zero/storage invariance and zero-operator support cases test scientific
  dependency semantics independently of serialization;
- non-signal-role rejection tests exact parent-role binding;
- cross-term sensitivity tests covariance-authority honesty; and
- NOI nonmutation and full-procedure mismatch test immutable and domain
  boundaries;
- marginal-only one-sparse, two-sparse, and exact-zero fixtures test the exact
  covariance-authority edge; and
- typed numerical domains, representation invariance, zero-domain, empty-
  output-support, and invalid-coefficient fixtures test the final formal
  closures.

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

This rationale explains the conditional freeze-candidate scientific contract
only. It makes no
implementation-conformity, algorithm-change, validation, calibration,
achieved-response, achieved-covariance, numerical-adequacy, performance,
readiness, scientific-freeze, production, or Unity claim.
