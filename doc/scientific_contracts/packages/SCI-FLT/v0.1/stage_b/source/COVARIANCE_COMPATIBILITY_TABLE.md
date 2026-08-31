# SCI-FLT-FIXED v0.1 Covariance Compatibility Table

Record identity: `SCI-FLT-FIXED-COVARIANCE-COMPATIBILITY v0.1/draft-r0.4`

Status: implementation-blind Stage B closure draft; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

## Authority and representation compatibility

```text
Parent authority              Authorized FLT covariance result
complete covariance           exact A C A^T on the complete domain in any
                              mathematically exact declared representation
independent-diagonal model    full covariance relative to that exact model,
                              including induced off-diagonal terms
marginal variances only       exact conditional marginal for a row with
                              exactly one nonzero parent coefficient;
                              otherwise unavailable or explicitly partial
structured or partial model   only operations proved exact for that exact
                              representation and domain
unavailable                   unavailable, except separately stated local
                              zero-operator parent-payload facts
```

## Marginal identity

```text
Var(y_i) = sum_j A_ij^2 Var(m_j)
           + 2 sum over j<k of A_ij A_ik Cov(m_j,m_k).
```

Marginal parent variances alone do not imply independence and generally do not
determine output marginals. A separately named diagonal-contribution
diagnostic may report only its exact contribution and must not be called
variance, covariance, uncertainty, or precision.

For a row with exactly one nonzero coefficient `A_ij`, marginal-only authority
does determine the conditional marginal

```text
Var(y_i) = A_ij^2 Var(m_j).
```

For a row mixing two or more parent variables, unknown cross terms leave the
exact marginal unavailable or explicitly partial. One exact row marginal does
not authorize cross-row covariance or independence. The exact-zero row keeps
its separately typed zero parent-payload covariance contribution.

## Nonclaims

This table does not assert an available parent covariance or achieved output
covariance and makes no validation, adequacy, performance, or production claim.
