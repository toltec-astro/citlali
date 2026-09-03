# SCI-POINT Zero-Background, Residual, And Identifiability Table

Identity: `SCI-POINT_RESIDUAL_IDENTIFIABILITY v0.1/r0.3`

The base source model is exactly zero-background:

`m = g(theta_true) + b + n`,

where `b` is any deterministic parent component not represented by the model.
POINT neither fits nor silently subtracts a constant, gradient, local
baseline, or other nuisance term. `b` may bias amplitude, centroid, widths,
and orientation; a finite fit does not convert it into stochastic noise.

When a numerical fit becomes available, the product shall publish or make
exactly reconstructible the fitted model, residual, objective value, support,
admitted row count, rank/identifiability, convergence/termination, active
constraints, seed/fallback state, residual diagnostics, and parent additive-
reference/null state. A named-use owner may require a compatible parent
background/reference state; missing authority produces unavailability rather
than hidden repair.

| Component/condition | Required state or consequence |
| --- | --- |
| amplitude/centroid/width/orientation ordinarily resolved | `available_identifiable` |
| active parameter bound | `available_bound_censored`, never silently ordinary |
| weak curvature or near degeneracy | `weakly_identified` |
| exact circularity and orientation | `undefined_by_model_symmetry` |
| missing method, boundary, or required input | `unavailable` |
| attempted computation that cannot return a conforming component | `failed` |
| major/minor exchange | canonical ordering required; orientation transforms consistently |
| centroid on search or fit boundary | explicit boundary/constraint state; use eligibility separate |
| rank-deficient curvature | formal error unavailable; fit component state separately assessed |

Required role presence means a typed component role, not a misleading finite
number. The POINT-owned completeness policy must define which combinations of
these states permit `complete_publication_candidate`.
