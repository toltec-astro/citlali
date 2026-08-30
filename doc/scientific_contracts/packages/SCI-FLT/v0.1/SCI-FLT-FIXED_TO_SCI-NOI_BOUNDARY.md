# SCI-FLT-FIXED To SCI-NOI Boundary

Boundary identity: `SCI-FLT-FIXED_TO_SCI-NOI v0.1/r0.1`

Status: sanitized Stage A boundary awaiting exact-byte owner approval; no
numerical transformed-uncertainty route is made available

Scientific owner: Grant Wilson

## Controlling Authority

This boundary instantiates approved SCI-NOI Stage A decisions ODQ-110A and
ODQ-110B for the strict-linear SCI-FLT-FIXED method. FLT owns the exact
transformation. NOI owns realization ensembles and empirical uncertainty.
NOI neither chooses nor defines the filter.

## Exact Fixed Transformation

For one immutable parent and one exact resolved plan,

\[
  y = J_{\rm full} L_\Theta m,
\]

where `L_Theta` is the fixed same-grid linear operator and `J_full` selects
the complete admitted full-footprint output-row domain. The method binds exact
owner/package/version/generation, parent, operator coefficients and parameters,
input/output grids and row domains, kernel and transfer state, normalization,
units, response, support, edge/missing/non-finite rule, lifecycle, causes,
failure, and provenance.

## Fixed-State NOI Parity

For every compatible admitted NOI member `M_b`,

\[
  M_b^{\rm FLT} = J_{\rm full} L_\Theta M_b.
\]

The real parent and every member use the identical operator, parameters, grid,
support, edge rule, row domain, and lifecycle generation. NOI infers
uncertainty from that exact transformed ensemble for that exact transformed
scientific product.

No commutation, relocation, approximate transfer, parameter inference,
same-name substitution, or method fallback is allowed. Failure to establish
exact compatibility/parity makes the transformed uncertainty route
unavailable.

## Forbidden Uncertainty Shortcuts

Transformed uncertainty is not obtained by filtering a variance, standard
deviation, precision, reciprocal, weight, standardized map, or significance
field. In general,

\[
  \operatorname{FLT}(\operatorname{NOI\text{-}STD}(m))
  \ne
  \operatorname{NOI\text{-}STD}(\operatorname{FLT}(m)).
\]

SCI-FLT-FIXED v0.1 has no additive term, so a zero-centered NOI member receives
only the exact linear transformation.

## Relearned And Inference-Bearing Exclusion

If a kernel, cutoff, support, threshold, edge state, or other parameter is
selected or re-resolved for a member, the result is a separately named
inference-bearing/relearned method under SCI-NOI ODQ-104. It cannot mix with
fixed-state members. Wiener, matched/template-amplitude, source-learned,
data-derived selection, automatic choice, and per-member relearning remain
outside SCI-FLT-FIXED.

## Product And Claim Boundary

NOI owns the conditional uncertainty/covariance/scale product and attachment;
FLT owns the transformed signal, exact operator, response, support, validity,
and lifecycle facts. The uncertainty applies only to the exact transformed
product and common admitted row domain. It establishes no physical-noise
equivalence, source cancellation, covariance completeness, Gaussian
significance, calibration, or scientific validity of the transformation.

This boundary supplies no numerical route and makes no implementation,
validation, performance, readiness, freeze, or production claim.
