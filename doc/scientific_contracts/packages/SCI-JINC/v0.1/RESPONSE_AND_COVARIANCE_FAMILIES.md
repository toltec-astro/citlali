# SCI-JINC v0.1 — Response And Covariance Families

Status: recovered Stage A scientific reference; base-v0.1 products deferred by
SCI-JINC-ODQ-107; excluded from the implementation-blind author packet

Prepared: `2026-08-28`

ODQ-107 preserves the mathematics below for a future concrete scientific use
but authorizes no response, covariance, formal-weight or limitation-record
product in the fixed base-v0.1 bundle. This file is not a base-v0.1 author
input and must not be used to introduce conditional or optional product-role
machinery.

## Response Families

| Family | Exact domain and realization | Availability / prohibition |
| --- | --- | --- |
| Fixed-state JINC response | Hold exact admitted signal membership, PTC coefficient values, analytic parameters, point phase, square placement, edge crop, `C_p`, conditioning and output rows fixed. Apply `A_pi=I_ip omega_i kappa_ip/C_p` to the available realized PTC-grid response companion exactly once. | Available only with exact response parent/domain and all fixed JINC state. It is not a bare analytic JINC, measured PSF, generic beam, or achieved-fidelity result. |
| PTC full-procedure finite difference with JINC fixed | Re-run the exact PTC procedure under its authorized perturbation while holding the exact JINC operator and membership fixed, then difference the JINC outputs under the declared convention. | Available only when frozen PTC authority supplies the complete procedure-response family and state-change record. It does not re-resolve JINC admission, coefficient, parameters, phase, support or conditioning. |
| JINC re-resolved procedure response | Under a declared perturbation, re-resolve every authorized JINC-dependent admission, coefficient, parameter, phase, support, edge, conditioning and product state, and record state changes. | Unavailable until a separate exact procedure authority defines perturbation, re-resolution, discontinuities, domain and output. It must not be mislabeled fixed-state response. |
| Whole-chain RTC-to-CAL-to-PTC-to-JINC response | A separately authorized complete rerun of the named whole chain with all learning, selection, calibration, PTC and JINC states governed. | Unavailable in v0.1 unless a separate authority is added. Partial composition or multiplication of incomparable companions is not this family. |

The response membership is exactly the signal membership for the selected
family. A PTC-grid companion begins at JINC and receives the JINC operator
once; upstream response is never applied twice and no hidden subset is used.
A rejected or unavailable JINC pixel has a typed unavailable response with
cause, not a zero response. Edge-truncated response uses the actual retained
square membership and may be asymmetric.

## Covariance Families

Let `A_JINC` contain the fixed normalized rows `A_pi`. On every available
domain, the exact operator relation is

```text
C_JINC = A_JINC C_PTC A_JINC^T.
```

| Covariance view | Required bindings | Claim limit |
| --- | --- | --- |
| Exact transformed covariance | Exact available `C_PTC` block/domain, axes, units, support, lifecycle generation, JINC membership/operator and all unavailable blocks. | “Exact” means exact action of the declared operator on that declared upstream covariance, not complete physical uncertainty. |
| Independent-sample diagonal shortcut | Exact proof that `omega_i=Var(z_i)^-1` on the admitted population and mutually independent admitted sample noise; exact `Q_p` and `C_p`. Then `Var_formal(m_p)=Q_p/C_p^2`. | Conditional formal variance only. It omits overlap-induced off-diagonal covariance and all unlisted terms. |
| Independent-sample off-diagonal view | Same assumptions plus shared-member identity: `Cov(m_p,m_p')=sum I_ip I_ip' omega_i kappa_ip kappa_ip'/(C_p C_p')`. | Conditional covariance of that diagonal upstream model; edge truncation and actual overlap are retained. |
| Relative/dimensionless coefficient view | Dimensionless `omega_i` with its named statistic and normalization. | `Q_p/C_p^2` is dimensionless and is not signal variance; `C_p^2/Q_p` is not formal precision. |

Every published covariance or formal-weight role states unavailable or omitted
correlated detector/atmospheric terms, calibration uncertainty, upstream
response uncertainty, selection and nuisance uncertainty, coefficient and
kernel-parameter uncertainty, phase/support/edge uncertainty and re-resolution
effects. Unknown is never zero.

`C_p`, `Q_p`, `jinc_coefficient_squared_time`, contributor count and
inverse-square units are not automatically precision. Empirical noise,
empirical covariance/weight and significance remain SCI-NOI authority.

## Scientific Numerical Adequacy Versus Execution Choice

ODQ-109 requires explicit conditioning and total numerical error negligible
compared with the approximately `10^-3` relative fidelity relevant to the
instrument. It does not prescribe a summation algorithm, contributor-count
bound, cache layout, thread order, bitwise reproducibility or stronger
precision. Candidate procedures and evidence belong in the future Engineering
Conformance Specification and later assessment, not in the scientific
response/covariance authority.
