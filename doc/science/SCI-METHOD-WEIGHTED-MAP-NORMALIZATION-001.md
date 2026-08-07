# SCI-METHOD-WEIGHTED-MAP-NORMALIZATION-001: Ordinary Positive-Coefficient Map Normalization

Status: `validated_bounded`

Scientific owner: Citlali scientific/product owner

First applicable contract: `sci-map-001-f010-v1` at application source
`af0c849ce59a5f80e5efc8db435bb6662863052f`

Supersedes: none

## Purpose

This method defines the ordinary weighted accumulation and normalization used
for naive array-grouped Stokes-I observation maps and for coadds of admitted
observation maps. It also explains why the same arithmetic does not, by itself,
turn the normalization coefficient into a statistical precision.

## Definition

At one output pixel, let the admitted contributions be indexed by (i). Each
contribution has signal (x_i), strictly positive finite coefficient (u_i),
and, when required, kernel value (k_i). In declared order Citlali accumulates

\[
Q = \sum_i u_i, \qquad
N = \sum_i u_i x_i, \qquad
K = \sum_i u_i k_i.
\]

Where the versioned normalization-support rule authorizes division and (Q)
is finite and positive, the normalized signal and kernel are

\[
\widehat{x} = \frac{N}{Q}, \qquad
\widehat{k} = \frac{K}{Q}.
\]

For an observation map, (x_i) and (k_i) are admitted detector-sample
values and (u_i) is the applicable detector coefficient. For a coadd,
(x_i) and (k_i) are normalized observation-map values and (u_i) is that
observation map's realized `weight_I` coefficient after observation
normalization and any declared global empirical coefficient rescaling.

The same admitted membership and normalization are used for declared
realization companions when they exist. This statement specifies their linear
map operation, not their statistical meaning; realization semantics belong to
the applicable NOI contract.

An explicitly invalid contribution is skipped before its numerical payload is
examined. A contribution declared valid must have finite signal, finite
positive coefficient, and finite required companions. Unexpected violations
or unrepresentable aggregates fail before partial live-product mutation.

## Properties

For fixed coefficients and admitted membership, the estimator is linear in the
signals. Because every admitted (u_i>0), it preserves a constant input and is
a convex weighted mean:

\[
x_i=c\ \forall i \Longrightarrow \widehat{x}=c,
\qquad
\min_i x_i \leq \widehat{x} \leq \max_i x_i.
\]

Normalizing the kernel with the same coefficients and membership carries the
realized response alongside the signal rather than assuming it separately.
The signal-centering operator is `L = I`: this method does not subtract a
mean, remove a null mode, or recenter a source.

The general variance of the weighted mean is

\[
\operatorname{Var}(\widehat{x}) =
\frac{1}{Q^2}
\sum_{i,j} u_i u_j\operatorname{Cov}(x_i,x_j).
\]

Only if the contributions are mutually uncorrelated and
(u_i=1/\operatorname{Var}(x_i)) does this reduce to

\[
\operatorname{Var}(\widehat{x}) = \frac{1}{Q}.
\]

The SCI-MAP-001 contract does not establish those coefficient-calibration and
covariance conditions. Therefore `weight_I` is a nonprecision gridding and
normalization coefficient by default even though its recorded unit is the
inverse square of the signal unit. Its numerical value cannot be used as
inverse variance, uncertainty, or statistical significance without the
separate `SCI-PTC-001` and covariance evidence.

## Interpretation And Limitations

The equations apply only after the method-specific admission and support rules
have selected contributions and authorized normalization. They do not define
those rules. In particular:

- `science_valid_I`, not (Q), is authoritative raw science validity;
- a finite normalized value does not imply science-policy support;
- the method does not define a JINC signed-contribution estimator;
- no GLS, covariance regularization, coadd uncertainty, or significance map is
  implied; and
- filtering or another downstream operator must retain raw validity and
  response identity separately from its own support and output validity.

Floating-point addition is order-dependent. The accepted implementation keeps
one detector/sample-ordered primitive for sequential and requested-parallel
work within a scan. Concurrent scan commits may differ only within the
registered `within-scan-exact-scan-farm-2gamma-n-sumabs-v1` bound; integer fact
planes remain exact.

## Citlali Use

Registered consumers are:

1. ordinary naive sample accumulation and normalization for array-grouped
   Stokes-I observation maps;
2. centered integer coaddition of atomically admitted observation-map
   packages; and
3. normalization of the matched kernel and any declared linear realization
   companions using the same membership.

The user-facing interpretation is in
[DOC-MAP-001 Phase A](../user/DOC-MAP-001_ORDINARY_NAIVE_MAPS.md). Other
pipeline stages using this unchanged method should link to this note rather
than reproduce the derivation.

## Validation

- [ADR 0009](../adr/0009-science-map-bundle-admission-and-validity.md)
- [Executable product contract](../../validation/product_contracts.json)
- [Focused contract tests](../../tests/test_science_map_contract.cpp)
- [Independent equation/truth tests](../../tests/test_science_map_truth_suite.cpp)
- [Production FITS tests](../../tests/test_science_map_fits_products.cpp)
- [Final application-integration decision](../../handoff/SCI-MAP-001_APPLICATION_INTEGRATION_DECISION_2026-08-05.md)

## Revision History

- 2026-08-07: Initial note, extracted without changing the accepted
  SCI-MAP-001 equations or scope.
