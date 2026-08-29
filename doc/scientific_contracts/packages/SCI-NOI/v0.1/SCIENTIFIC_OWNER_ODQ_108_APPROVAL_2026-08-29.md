# SCI-NOI v0.1 — Scientific-Owner ODQ-108 Approval

Decision identity: `SCI-NOI-ODQ-108`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

Status: approved initial MAP standardized-signal method

## Exact Owner Decision

The owner approved the proposed ODQ-108 disposition: the first standardized-
signal method uses the exact immutable normalized real-observation MAP signal
as numerator and the square root of the compatible ODQ-105B conditional second
moment as its canonical empirical signal-unit scale. Its claim is only “MAP
signal standardized by the stated conditional randomization second-moment
scale.” JINC standardization remains a separate future method.

## Initial Method And Formula

The method identity is
`NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1`. For the exact immutable
normalized MAP signal `q_MAP(p)` and compatible `V_hat_cond(p)`, define

```text
D_STD = exact valid-domain intersection of q_MAP and V_hat_cond,
        restricted to finite strictly positive V_hat_cond,
sigma_cond(p) = sqrt(V_hat_cond(p)),
S_cond(p) = q_MAP(p) / sigma_cond(p).
```

The output product role is `empirical_scale_standardized_signal` with unit
`1`. The numerator is the real-observation MAP parent associated with the same
exact frozen MAP operator state through which the NOI realization ensemble was
generated. The method binds exact MAP estimator and generation, immutable
parent relation, response reference, signal unit/beam, WCS, support, validity,
and lifecycle, together with the compatible UNC method/generation and scale
transformation.

No interpolation, domain extension, parent substitution, response
substitution, or lifecycle-generation substitution is implicit. Zero,
negative, nonfinite, unavailable, outside-domain, or incompatible scale makes
the standardized product unavailable rather than zero or infinity.

## Canonical Scale Route

The canonical initial denominator is `sqrt(V_hat_cond)`. Although
`1/sqrt(W_hat_cond)` is algebraically equal on the exact valid reciprocal
domain, ODQ-107's inverse-scale product does not create a second implicit STD
method identity. Any alternate source/transform route must be separately named
and shown to preserve exact domain, lifecycle, and meaning.

## Dependence And Claim Boundary

The method records the dependence between the real-observation MAP numerator
and the NOI-derived empirical scale. Standardization does not create
independence, a pivotal statistic, or a calibrated null distribution.

`S_cond` means only “MAP signal standardized by the stated conditional
randomization second-moment scale.” It is not, by this construction alone, an
uncertainty estimate, Gaussian z-score, Student statistic, N-sigma detection,
statistical significance, false-alarm probability, detection probability,
completeness, purity, or catalog authority.

## JINC Separation

`jinc_map` remains eligible only for a future separately identified STD method
with a complete numerical JINC parent and an exact compatible JINC-specific
signal-unit uncertainty scale. It shall not inherit the MAP scale or method by
analogy.

## Non-Implications

This approval does not create a numerical MAP, UNC, scale, or STD product
before their existing parent, compatibility, finite-positive-domain, and
admission gates pass. It does not establish implementation conformity,
validation, calibration, performance, readiness, production use, detection,
or catalog authority.
