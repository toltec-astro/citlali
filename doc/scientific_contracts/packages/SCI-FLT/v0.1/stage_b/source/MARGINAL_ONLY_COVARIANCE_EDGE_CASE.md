# SCI-FLT-FIXED v0.1 Marginal-Only Covariance Edge-Case Amendment

Record identity: `SCI-FLT-FIXED-MARGINAL-COVARIANCE-AMENDMENT v0.1/draft-r0.4`

Status: implementation-blind Stage B closure artifact; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

Marginal parent variances do not authorize independence. When one output row
has exactly one nonzero parent coefficient `A_ij`, marginal-only authority is
sufficient for that row's conditional marginal:

```text
Var(y_i) = A_ij^2 Var(m_j).
```

When the row mixes two or more parent variables, unknown cross terms make its
exact marginal unavailable or explicitly partial. A one-row result authorizes
neither cross-row covariance nor independence. The exact zero operator retains
its separately typed zero parent-payload covariance contribution.

The required fixture set covers one-sparse, two-sparse with unknown
covariance, and exact-zero rows without promoting any marginal result to full
covariance.
