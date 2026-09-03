# SCI-NOI v0.1 — Scientific-Owner ODQ-105B Approval

Decision identity: `SCI-NOI-ODQ-105B`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

Status: approved initial conditional second-moment estimand

## Exact Owner Decision

The owner approved the proposed ODQ-105B disposition: use a zero-centered
conditional randomization second moment with exact finite-design weights,
common all-member domain, and no `B-1`, physical-noise, or independent-
observation interpretation.

## Sanitized Disposition

For one exact all-members-successful admitted GEN ensemble with realization
maps `M_b(p)`, the initial NOI-UNC estimand is the second moment under the
declared detector-sign assignment law, conditional on the immutable parent and
the declared frozen reduction state. On the exact common domain

```text
D_common = {p : every admitted realization supplies a valid finite value at p},
```

the estimator is

```text
V_hat_cond(p) = sum_b omega_b M_b(p)^2,
sum_b omega_b = 1,
```

where the exact nonnegative `omega_b` are dictated by the authored finite
design and realized admitted assignment set. Uniform weights may be used only
when the exact design establishes them; numerical equality cannot be assumed.

The center is the design-declared known zero. The finite ensemble mean is not
subtracted. Consequently, residual astronomical source imprint, structured
residuals, and other nonzero realization content contribute to this second
moment rather than being silently removed by empirical recentering.

There is no `B-1` correction because the method estimates a second moment about
a declared known center, not covariance about an empirically estimated mean.
There is also no universal `1/B` rule independent of the exact design weights.

## Domain And Information Meaning

The initial estimator is available only on `D_common`. A location lacking a
valid finite value from any admitted realization is unavailable for this method;
the method shall not silently use a different realization subset there.

The method records requested/resolved/completed/unique counts, complement and
dependence structure, exact design rank, use-specific effective information,
and the uncertainty of `V_hat_cond` or an explicit unavailable state. The
member count is not a number of independent astronomical observations,
exposure, or proof that parent-map noise decreases as `1/sqrt(B)`.

## Claim Boundary

`V_hat_cond` is an empirical conditional randomization second moment in squared
signal units. It is not, by this construction alone, repeated-observation
physical-noise variance, formal MAP covariance, a calibrated null, precision,
statistical significance, or detection probability. Any square-root scale,
off-diagonal covariance, projection, inverse, weight, or standardized-signal
product requires its own exact authorized method identity and transformation.

## Non-Implications

This approval does not make the estimator numerically available before the GEN
route, exact finite design/weights, all-member completion, common domain, and
use-specific adequacy gates are realized. It does not resolve broader
covariance representation, inverse/weight, STD pairing, persistence, filtering,
FRUIT, VAL Registry, conformity, validation, performance, readiness, or
production questions.
