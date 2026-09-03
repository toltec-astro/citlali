# SCI-NOI v0.1 r0.3 Shared Normative Module: Equations

Module identity: `SCI-NOI_NORMATIVE_EQUATIONS v0.1/r0.3`

Scientific owner: Grant Wilson

Status: implementation-blind Stage B draft authority; content-bound by
`SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.3`; not frozen.

## `SCI-NOI-EQ-001` - frozen contribution

```text
a_pi = G_pi gamma_i.
```

## `SCI-NOI-EQ-002` - unsigned normalization

```text
Q_p = sum_{i in C_p} a_pi.
```

## `SCI-NOI-EQ-003` - detector mass and active population

```text
beta_d = sum_p sum_{i in C_p, d(i)=d} a_pi,
D_h^+ = {d : beta_d > 0 and exact h(d)=h}.
```

## `SCI-NOI-EQ-004` - exact active-stratum condition

```text
Delta_h(epsilon_b)
  = abs(sum_{d in D_h^+} epsilon_bd beta_d)
    / sum_{d in D_h^+} beta_d
  <= tau_h,                 0 <= tau_h < 1.
```

The ratio is evaluated only for an active positive-total stratum. `tau_h` is
exact, plan-bound, and has no default.

## `SCI-NOI-EQ-005` - ordinary realization member

```text
M_b(p)
  = [sum_{i in C_p} a_pi epsilon_b,d(i) Z_i^PTC] / Q_p.
```

The sign modifies only `Z_i^PTC`, exactly once. All coefficients, membership,
normalization, WCS, projection, support, `coverage_cut`, response handling,
validity, and MAP-local gates remain frozen.

## `SCI-NOI-EQ-006` - independently governed ordinary MAP signal

```text
m_MAP(p) = [sum_{i in C_p} a_pi Z_i^PTC] / Q_p.
```

This is the corresponding all-`+1` arithmetic, not an NOI-generated product.
`m_MAP` exists only when the independently governed ordinary SCI-MAP route is
scientifically realized. Missing coefficient, numerical `coverage_cut`, MAP
admission, or other required parent makes `m_MAP` and dependent STD unavailable.

## `SCI-NOI-EQ-007` - fixed signal operator

```text
R_b^fixed = A_MAP,Pi D_epsilon_b H_PTC^fixed.
```

`D_epsilon_b` is the occurrence-domain diagonal expansion of the active
detector assignment over the exact admitted occurrence population. Required
response companions use identical occurrence membership and the identical
modifier exactly once; missing response is unavailable and never shrinks the
signal population.

## `SCI-NOI-EQ-008` - active-domain expectations

```text
E_D[epsilon_bd] = 0, for d in the exact active design population,
E_D[M_b(p) | fixed parent and operator] = 0,
  for p in the exact available frozen MAP row domain.
```

The relations require the approved complement-symmetric target law. No sign
is implied for nonactive detectors.

## `SCI-NOI-EQ-009` - conditional marginal second moment

```text
Vhat_cond(p) = sum_{b in A_UNC} [1/N_resolved] M_b(p)^2.
```

The known target center is zero; no ensemble-mean subtraction or `B-1` rule
applies. The equation uses the exact common all-member domain.

## `SCI-NOI-EQ-010` - unavailable proposed reciprocal

```text
rho_cond(p) = 1 / Vhat_cond(p).
```

This equation defines no active base product. It remains unavailable pending
owner disposition and exact profile/source binding.

## `SCI-NOI-EQ-011` - conditional scale

```text
sigma_cond(p) = sqrt(Vhat_cond(p)).
```

## `SCI-NOI-EQ-012` - standardized MAP signal

```text
zeta_cond(p) = m_MAP(p) / sigma_cond(p).
```

The output exists only on the exact compatible finite-positive domain, has
unit `1`, and carries only the claim in `SCI-NOI-REQ-031`.
