# SCI-FLT-FIXED v0.1 Scientist Rationale

Document identity: `SCI-FLT-FIXED-SCIENTIST-RATIONALE v0.1/draft-r0.1`

Status: implementation-blind Stage B explanatory draft; scientific-owner review required

Normative import: the complete
`SCI-FLT-FIXED-NORMATIVE-CORE v0.1/draft-r0.1`, source SHA-256
`{{NORMATIVE_CORE_SHA256}}`, is incorporated without modification. If this
rationale and that core differ, the core controls.

## 1. The scientific question

SCI-FLT-FIXED answers one deliberately narrow question: given one exact
admitted map-domain parent and one complete fixed linear operator resolved
outside the application, what scientific product results when that operator is
applied on the same grid?

The answer is the transformed parent-map amplitude

```text
y = J_full L_Theta m.
```

The restriction `J_full` matters as much as the convolution itself. It keeps
only rows whose entire required input footprint is present, admitted, finite,
and predicate-passing. The contract therefore describes both a transformation
and the exact domain on which its result is scientific.

This is not a general filtering contract. It does not combine deterministic
convolution with Wiener inference, matched or template-amplitude estimation,
source learning, data-derived mode selection, automatic method selection, or
per-realization relearning.

## 2. Why strict linearity is the base object

Strict linearity makes the estimand and its consequences explicit. Doubling
the parent doubles the transformed signal. An impulse exposes the exact
sampled kernel. A compatible response propagates through the same operator.
An available covariance propagates by the same linear map on both sides.

An additive offset would break this compact identity. It would need its own
parent, units, support, uncertainty, reference-mode, response, covariance, NOI,
lifecycle, and failure semantics. Rather than hide those scientific questions
inside a familiar operation, v0.1 admits no additive term.

The word fixed applies to the entire scientific state, not merely to a
numerical function call. Coefficients, parameters, support, normalization,
grid, edge rule, and transfer qualification are all frozen before the admitted
parent random field or any NOI member is transformed.

## 3. Fixed convolution and the low-pass qualification

Convolution is a structured way to construct the finite operator. The exact
sampled coefficients, their grid offsets, center, phase, orientation, support,
and normalization define the transformation. A family label or continuous
ideal does not.

Low-pass is a further scientific claim about that exact sampled operator. It
requires a frequency domain and WCS metric, DC gain, passband, transition,
stopband or attenuation, phase, anisotropy, finite-grid and edge limitations,
and complete parameter provenance. If any of those are absent, it can still be
honest to say fixed convolution, but not low-pass.

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

When the parent supplies an exact compatible response, the filter-composed
response is obtained by applying the identical realized operator and row
selection. This is the response to the same parent source convention, not an
automatic new calibration or proof of source-flux fidelity.

When no compatible parent response exists, the transformed response remains
unavailable. The kernel cannot stand in for the whole source response because
the parent response, beam, centering, edge rule, and finite domain all matter.

The local transfer and mode records describe the exact finite operator where
those descriptions are scientifically defined. Nulling a local constant mode,
for example, does not prove a whole-chain sky-mode claim. Influence similarly
describes which parent rows and coefficients contribute; it is not physical
exposure.

## 7. Formal covariance and empirical uncertainty

If a compatible parent covariance is available, the exact deterministic
propagation is

```text
C_out = J_full L_Theta C_parent transpose(L_Theta) transpose(J_full).
```

Even independent parent pixels generally become correlated after convolution
because neighboring outputs reuse input pixels. A variance plane contains only
marginals. It is not full covariance, and unknown cross terms are not zero.

The contract names complete, structured, partial, marginal, and unavailable
covariance states so that an honest limited representation does not acquire a
stronger meaning through a label.

Empirical uncertainty is a different ownership boundary. SCI-NOI applies the
exact frozen FLT state to every compatible admitted randomization and infers
the conditional uncertainty attachment. Filtering a variance, weight,
standard deviation, standardized map, or significance field is not equivalent.
Relearning a filter for each member would define a different method and cannot
be mixed into this fixed-state ensemble.

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

Requested, effective, resolved, applied, and realized are not synonyms. The
plan may disable the route. Required state may be unavailable. Application may
fail. A transformed array may exist before its required response, support,
validity, covariance-state, lineage, and failure records form an atomic bundle.

The lifecycle preserves these distinctions. Disabled yields no product.
Identity and zero operators, when requested, resolved, and applied, yield real
separately parented products. A zero signal is not an unavailable row, a
disabled route, zero uncertainty, or infinite precision.

Honest absence is part of the product. A base transformed signal may carry an
explicit unavailable response or covariance state where the role permits it.
A response-qualified or covariance-qualified request cannot. Missing required
records never become placeholders.

## 10. Use and ownership limits

SCI-FLT-FIXED owns its local transformation, product, response state, support,
validity, deterministic covariance state, lifecycle, and failure behavior. MAP
or JINC continues to own the parent estimand. CAL owns absolute calibration.
SCI-NOI owns empirical uncertainty inference. Beam, source, Pointing, OOF, and
FRUIT interpretations remain with their respective future or upstream owners.

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
- low-pass completeness tests whether a qualified transfer claim is actually
  supported.

Passing these predictions would be evidence relevant to a later conformity or
validation activity. This rationale reports no such result.

## 12. Stage B nonclaims

This rationale explains the draft scientific contract only. It makes no
implementation-conformity, algorithm-change, validation, calibration,
achieved-response, achieved-covariance, numerical-adequacy, performance,
readiness, scientific-freeze, production, or Unity claim.
