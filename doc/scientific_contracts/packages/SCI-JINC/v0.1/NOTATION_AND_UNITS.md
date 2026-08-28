# SCI-JINC v0.1 — Collision-Free Notation And Units

Status: final Stage A repair candidate; awaiting scientific-owner approval

Prepared: `2026-08-28`

## Canonical Stage A Symbols

| Symbol | Meaning | Sign/domain | Unit |
| --- | --- | --- | --- |
| `i` | Exact admitted PTC occurrence on stable RTC output sample `n` | identity | — |
| `p` | Target JINC map pixel in one exact WCS | identity | — |
| `I_ip` | Complete Boolean sample-pixel membership after JINC sample admission, exact same-processed-sample AST association, local geometry, finite support, square placement, and pixel-local gates | `0` or `1` | 1 |
| `z_i` | Exact transformed signal, `Z_i^PTC` | finite when admitted | signal unit `U` |
| `kappa_ip` | Signed dimensionless analytic JINC kernel coefficient at the selected point phase | finite; positive, zero, or negative | 1 |
| `omega_i` | Positive producer-supplied JINC-facing analysis coefficient | finite and `>0` after JINC-local admission | either 1 or `U^-2`, according to its exact family |
| `w_ip` | Complete effective signed pixel contribution coefficient, `kappa_ip omega_i` | finite; signed | same unit as `omega_i` |
| `s_a` | Explicit angular radial scale associated with stable TolTEC array `a` | finite and `>0` for a numerical route | angle |
| `r'_a` | Dimensionless radial coordinate, `r/s_a` | finite and `>=0` | 1 |
| `theta_a` | Ordered kernel realization `(s_a,a_a,b_a,c_a,(r_max)_a)` for stable TolTEC array `a` | every component finite and positive for a numerical route; exact scientifically authorized identity otherwise unavailable | `s_a`: angle; remaining components: 1 |
| `C_p` | Signed normalization, `sum I_ip w_ip` | finite; may be negative; never accepted at exact zero | same unit as `omega_i` |
| `Q_p` | Quadratic accumulator, `sum I_ip omega_i kappa_ip^2` | finite and `>0` for formal support | same unit as `omega_i` |
| `N_p` | Signed signal numerator, `sum I_ip w_ip z_i` | finite | `U` times the unit of `omega_i` |
| `m_p` | Normalized estimate, `N_p/C_p` | available only on admitted, conditioned support | `U` |
| `rho_p` | Cancellation statistic, `abs(C_p)/sum I_ip abs(w_ip)` | finite in `[0,1]` when denominator is positive | 1 |
| `f_s,i` | Effective processed-timestream sample frequency | finite and `>0` | `s^-1` |
| `T_p^(kappa^2)` | `jinc_coefficient_squared_time`, `sum I_ip kappa_ip^2/f_s,i` | finite and nonnegative | s |
| `A_pi` | Fixed-state normalized linear JINC operator row, `I_ip w_ip/C_p` | signed | 1 |

The symbol `c` is reserved, if ultimately approved, for one named component
of `theta_a`. It is never reused for the analytic coefficient, effective
pixel coefficient, normalization, or coefficient-squared time.

The generic parameter semantics do not supply TolTEC numerical values.
Parameters may be array-associated where scientifically appropriate.
Requested, effective, observation-resolved, and realized parameter-set
identities are distinct; absence of an authorized set makes the numerical
route unavailable without a hidden default.

`SCI-JINC:jinc_map_contribution@1` sample admission does not itself set
`I_ip=1`: sample-pixel support is a later JINC-owned decision. Outside support
and a contract-defined `kappa_ip=0` are ordinary no-contribution states, while
finite negative `kappa_ip` is normal. Every accumulator term for one
contribution uses the same admitted sample-pixel pair and the same
`kappa_ip` identity.

## Exact Retained Algebra

For every pixel with a positive membership population,

```text
N_p = sum_i I_ip omega_i kappa_ip z_i = sum_i I_ip w_ip z_i
C_p = sum_i I_ip omega_i kappa_ip     = sum_i I_ip w_ip
Q_p = sum_i I_ip omega_i kappa_ip^2
m_p = N_p / C_p
A_pi = I_ip omega_i kappa_ip / C_p
```

If and only if the exact coefficient family establishes
`omega_i=Var(z_i)^-1` for mutually independent admitted occurrences,

```text
Var_formal(m_p) = Q_p / C_p^2
W_formal,p      = C_p^2 / Q_p

Cov(m_p,m_p') =
  [sum_i I_ip I_ip' omega_i kappa_ip kappa_ip'] / (C_p C_p').
```

For a general available upstream covariance `C_PTC`, the exact operator
relation is

```text
C_JINC = A_JINC C_PTC A_JINC^T.
```

For an available processed source-template response `h_i^processed`, the
fixed-state response companion is

```text
R_p = [sum_i I_ip omega_i kappa_ip h_i^processed] / C_p.
```

It uses the exact signal membership and applies the realized PTC-grid
companion through JINC exactly once.

The temporal accounting product is

```text
T_p^(kappa^2) = sum_i I_ip kappa_ip^2 / f_s,i.
```

The squared object is the signed dimensionless analytic coefficient
`kappa_ip`; its square is nonnegative. Neither `omega_i` nor `w_ip` is squared
in this time product.

## Unit Table For Both Allowed Coefficient Classes

| Object | Dimensionless `omega_i` | Inverse-square `omega_i` |
| --- | --- | --- |
| `z_i` | `U` | `U` |
| `kappa_ip` | 1 | 1 |
| `omega_i`, `w_ip`, `C_p`, `Q_p` | 1 | `U^-2` |
| `N_p` | `U` | `U^-1` |
| `m_p=N_p/C_p` | `U` | `U` |
| `Q_p/C_p^2` | 1; not a signal variance | `U^2`; a formal variance only with the declared statistical assumptions |
| `C_p^2/Q_p` | 1; not formal precision | `U^-2`; a formal weight only with the declared statistical assumptions |
| `Cov(m_p,m_p')` from `C_PTC` | `U^2` when `C_PTC` has `U^2` | `U^2` |
| `T_p^(kappa^2)` | s | s |

Inverse-square units alone do not prove inverse variance, independence,
precision, or significance. If `omega_i` is dimensionless, the algebraic
`Q_p/C_p^2` and `C_p^2/Q_p` factors may be published only under a separately
named relative/statistical interpretation; they are not signal variance or
formal inverse variance.

## Scaling Predictions

- Constant input `z_i=z_0` gives `m_p=z_0` whenever `C_p` is accepted.
- One contributor with nonzero `kappa` gives `m_p=z_i`.
- Multiplying all `omega_i` by a common positive factor leaves `m_p`,
  `rho_p`, and `A_pi` unchanged and scales `N_p`, `C_p`, and `Q_p` together.
- Rescaling the signal unit by `z_i -> lambda z_i` and a true inverse-variance
  coefficient by `omega_i -> omega_i/lambda^2` yields
  `m_p -> lambda m_p` and `W_formal,p -> W_formal,p/lambda^2`.
- An analytic `kappa_ip=0` changes none of `N_p`, `C_p`, `Q_p`, or
  `T_p^(kappa^2)` while remaining distinguishable from outside support.
- A finite negative lobe retains its sign in `N_p` and `C_p`, enters `Q_p` and
  `T_p^(kappa^2)` quadratically, and may produce a finite negative `C_p`.
- Exact `C_p=0` or numerically unresolved cancellation is unavailable, never
  zero sky. A resolved finite negative `C_p` is admissible.
