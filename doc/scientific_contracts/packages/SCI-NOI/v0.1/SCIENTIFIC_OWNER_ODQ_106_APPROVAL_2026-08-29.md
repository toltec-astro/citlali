# SCI-NOI v0.1 — Scientific-Owner ODQ-106 Approval

Decision identity: `SCI-NOI-ODQ-106`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

Status: approved covariance-representation and rank policy

## Exact Owner Decision

The owner approved the proposed ODQ-106 disposition: the ordinary initial
uncertainty representation is the ODQ-105B pointwise conditional second-moment
field; it is not covariance merely because it is pointwise or diagonal-like.
Additional covariance representations are optional, separately identified
NOI-UNC methods. Dense full covariance is never universally required.

## Initial Representation

The initial product remains

```text
V_hat_cond(p) = sum_b omega_b M_b(p)^2
```

on the exact common all-member domain approved by ODQ-105B. It is a pointwise
conditional randomization second-moment field in squared signal units. It does
not, by itself, claim marginal physical-noise variance, MAP covariance,
off-diagonal covariance, independence, invertibility, precision, or statistical
significance.

## Additional Authorized Representation Families

SCI-NOI may define separately identified methods that produce or preserve:

- a retained realization ensemble;
- one named fixed projection or projected uncertainty;
- stationary or kernel covariance;
- block, spectral, sparse, low-rank, or another exact structured covariance;
- a full covariance representation; or
- an explicit unavailable covariance state.

Each numerical covariance method must declare its exact target and estimator,
member population, domain and support, response reference, representation,
rank or rank limitations, null space or unresolved modes, regularization and
omissions, uncertainty/calibration state, and lifecycle. Storage shape or a
numerically square array does not select a covariance meaning.

A retained realization ensemble is a representation and possible input to an
authorized estimator; it is not automatically an estimated covariance.

## Missing And Unreported Covariance

The ordinary initial method retains the ODQ-105A/B all-members-successful and
common-all-member requirements. It shall not use a survivor subset, pairwise
member population, or generic missing-data estimator to rescue a failed member
or unavailable location.

An unreported off-diagonal entry, block, mode, or region means unknown or
unavailable. It shall not be interpreted as zero covariance or statistical
independence. A future separately authorized covariance method may define an
exact treatment for within-product/domain missingness, but it cannot override
the admitted-member failure rule or silently change the member population.

## Rank And Inverse Boundary

Every covariance representation reports exact rank when available, or a
bounded/unknown rank state; its domain, null space or unresolved modes; and any
regularization or approximation. Limited or rank-deficient products may remain
scientifically useful when honestly labeled.

No covariance representation implies invertibility. A numerical inverse does
not by itself establish statistical precision. Marginal inverse variance,
precision, generalized inverse, and consumer-effective weights remain governed
by ODQ-107.

## Non-Implications

This approval does not create a numerical covariance product, require dense
covariance storage, authorize a missing-member survivor estimator, approve an
inverse or weight, select an STD scale, or establish implementation conformity,
validation, calibration, performance, readiness, or production use.
