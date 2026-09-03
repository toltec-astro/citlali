# SCI-POINT Gaussian Source Model And Parameter Conventions

Identity: `SCI-POINT_SYMBOLIC_GAUSSIAN_MODEL v0.1/r0.3`

For admitted tangent coordinate `u_q`, the generic zero-background model is

`g_q(A,mu,Sigma) = A exp[-1/2 (u_q-mu)^T Sigma^{-1} (u_q-mu)]`,

where `A` is the parent-unit amplitude, `mu=(mu_1,mu_2)` is the centroid in the
declared tangent basis, and `Sigma` is symmetric positive definite. No fitted
constant, gradient, local baseline, or other nuisance term is present.

The six scientific parameter roles are amplitude, two centroid components,
two principal-width roles, and orientation. The invariant representation obeys
major principal extent greater than or equal to minor principal extent, both
strictly positive. Orientation may be represented on a half-turn gauge only
after its exact origin and positive direction are supplied by the approved
compatibility method. At exact circularity, orientation state is
`undefined_by_model_symmetry`; an arbitrary number has no physical meaning.

The following remain unavailable pending `POINT-COMPATIBILITY-METHOD v0.1`:

- numerical parameter ordering;
- whether published widths are sigma, FWHM, or another convention;
- numerical angle origin, positive direction, period/gauge canonicalization;
- exact amplitude sign domain and parameter bounds; and
- active-bound interpretation and source-profile reference convention beyond
  the explicit source-reference boundary.

Therefore the generic contract may use `Sigma` symbolically but may not
publish legacy numerical width or angle fields as though their convention were
known.
