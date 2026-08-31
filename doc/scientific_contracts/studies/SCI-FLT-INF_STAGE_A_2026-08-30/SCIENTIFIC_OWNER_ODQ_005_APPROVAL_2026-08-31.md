# SCI-FLT-INF-ODQ-005 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-005`

Date: `2026-08-31`

Scientific owner: Grant Wilson

Status: approved and closed

## Approved decision

The base v0.1 matched-filter method uses one exact immutable,
scientifically declared **template-response product** for each realized
application. The template is the expected parent-map response per unit of the
declared amplitude estimand `A`. In the local model

```text
m = A t + n,
```

the parent-map signal unit, amplitude unit, and template unit are related
explicitly as `unit(t) = unit(m) / unit(A)`. The template's scaling therefore
defines the amplitude convention. No peak, integral, flux-density, beam, or
other amplitude convention may be inferred from a generic `kernel` name.

The ODQ-001 normalization obligation remains exact: under the selected noise,
support, edge, validity, and operator assumptions, a parent signal equal to
`A t` must return an unbiased estimate of `A` wherever the method is
scientifically admitted.

## Admitted template sources

The immutable template-response product may be:

1. the exact point-source-response product bound to the admitted immutable
   parent bundle; or
2. another explicitly supplied scientific template-response product defining
   the amplitude of its stated shape.

An analytic Gaussian or Airy construction is admissible only as a producer of
that same exact template-response product. Its complete scientific parameters,
units, sampling, normalization, and provenance must be resolved and the
materialized template identity fixed before application. Analytic construction
does not create a weaker or unnamed template route.

Every admitted template product must bind:

- its source authority and immutable identity;
- its compatible parent role and exact parent reference where parent-bound;
- amplitude definition and units, parent signal units, and template units;
- grid, WCS/frame, pixel convention, centering, and subpixel phase;
- finite support, truncation rule, and omitted-tail treatment;
- array identity and any other population dependence;
- parent-beam/effective-response relationship and calibration state; and
- validity, missing/nonfinite, null, and unavailable behavior.

The observation-parent and coadd-parent routes remain distinct under ODQ-003.
A template may be used in both only when its exact scientific identity and
compatibility are declared for both; no equality, reuse, or response
equivalence is inferred.

## State and exclusions

For base v0.1, the template is fixed before the filter applies to the target
parent and is unchanged throughout that realized application. A point-source
response already owned by and frozen in the immutable parent bundle is a
parent-bound declared input, not template learning performed by this package.

Template estimation or selection from the target parent, a detected or fitted
source, a candidate population, or an NOI member is not admitted in base
v0.1. Any such route requires a separate learned-state method and exact
generation graph. This preserves the ODQ-002 exclusion of source detection,
selection, fitting, peak interpretation, deblending, and catalog behavior.

The historical high-pass/delta case is not admitted as a base-v0.1 matched-
template identity. Any future high-pass transformation requires a separately
authorized scientific method; it may not be introduced by labeling it a
template.

## Consequences

- `SCI-FLT-INF-ODQ-005` is approved and closed.
- Base v0.1 has one template-response product type with two admitted source
  classes, not unrelated kernel/Gaussian/Airy modes.
- The exact amplitude convention is carried by each template identity; there
  is no hidden universal peak or integral convention.
- Template discretization and any approximation must preserve the declared
  response or expose their bounded consequences under ODQ-006.
- Response, beam, calibration-covariance, uncertainty, edge, and NOI details
  remain governed by their later ordered decisions.
- ODQ-006 exact operator, approximation, and regularization is the next owner
  gate.

## Nonclaims

This approval does not select the ODQ-004 noise/covariance option, a numerical
template array, Gaussian/Airy parameters, an interpolation/discretization
algorithm, support radius, truncation tolerance, response approximation,
uncertainty, calibration covariance, edge method, NOI lifecycle, product
bundle, implementation conformity, validation, performance, readiness,
production, freeze, or Unity action. It changes no SCI-FLT-FIXED or frozen
SCI-NOI byte.
