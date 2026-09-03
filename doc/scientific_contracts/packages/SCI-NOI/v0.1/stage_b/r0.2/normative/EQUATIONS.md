# SCI-NOI v0.1 r0.2 Shared Normative Module: Equations

Module identity: `SCI-NOI_NORMATIVE_EQUATIONS v0.1/r0.2`

Status: implementation-blind Stage B draft authority; content-bound by
`SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.2`; not owner-accepted or frozen.

## `SCI-NOI-EQ-001` - frozen pre-normalization contribution

```text
a_pi = G_pi gamma_i.
```

The factors, occurrence membership, coefficient generation, and QC state are
exact frozen parent facts.

## `SCI-NOI-EQ-002` - frozen unsigned normalization

```text
Q_p = sum_{i in C_p} a_pi.
```

`Q_p` is evaluated only on the exact frozen MAP output-row domain where its
parent gates are satisfied. It is not signed or relearned.

## `SCI-NOI-EQ-003` - detector coefficient mass

```text
beta_d = sum_p sum_{i in C_p, d(i)=d} a_pi.
```

The sum uses the exact frozen admitted occurrence/row population. `beta_d`
replaces the ambiguous r0.1 symbol `B_d`.

## `SCI-NOI-EQ-004` - candidate network imbalance

```text
Delta_h(epsilon_b)
  = abs(sum_{d in D_h^active} epsilon_bd beta_d)
    / sum_{d in D_h^active} beta_d.
```

This equation is an author-recommended ODQ-102D candidate, not accepted base
science. It shall not be evaluated when the denominator is zero. Its tolerance,
population, construction, and failure semantics remain unavailable pending
owner acceptance of the complete design candidate.

## `SCI-NOI-EQ-005` - canonical ordinary realization member

```text
M_b(p)
  = [sum_{i in C_p} a_pi epsilon_b,d(i) Z_i^PTC] / Q_p.
```

The sign modifies only `Z_i^PTC`. `G_pi`, `gamma_i`, `C_p`, `Q_p`, WCS,
projection, support, `coverage_cut`, response handling, and every MAP-local
gate remain frozen. The denominator is unsigned and is neither randomized nor
relearned. The sign is applied exactly once to each admitted occurrence.
Inline and materialized application are equivalent only under this equation
and identical parent/membership identity.

## `SCI-NOI-EQ-006` - corresponding real-observation MAP signal

```text
m_MAP(p) = [sum_{i in C_p} a_pi Z_i^PTC] / Q_p.
```

This is the corresponding all-`+1` operation and retains its own immutable
ordinary MAP product identity. `M_b` is an NOI realization, not ordinary MAP
science.

## `SCI-NOI-EQ-007` - fixed-state signal/response operator relation

```text
R_b^fixed = A_MAP,Pi D_epsilon_b H_PTC^fixed.
```

`H_PTC^fixed` maps exact admitted parent occurrences into the fixed PTC output
domain; `D_epsilon_b` applies the same detector assignment exactly once;
`A_MAP,Pi` applies the exact frozen MAP projection, membership, normalization,
support, response-reference, and publication operator to the declared output
domain `Pi`. A response companion, when required and available, receives the
identical occurrence membership and identical modifier exactly once under its
owner-declared response operator. Missing response is typed unavailable and
shall not create a hidden subset of signal membership.

## `SCI-NOI-EQ-008` - complement-symmetric target mean

```text
E_D[epsilon_bd] = 0,
E_D[M_b(p) | fixed parent and operator] = 0.
```

The second equality follows from the linear fixed-state member equation and
the approved complement-symmetric target law. It does not remove source or
structured residual terms from individual members, second moments, or tails.

## `SCI-NOI-EQ-009` - initial conditional second moment

```text
Vhat_cond(p) = sum_{b in A_UNC} omega_b M_b(p)^2,
sum_b omega_b = 1,
omega_b >= 0.
```

The finite design supplies the exact weights. The center is the known target
zero; the finite ensemble mean is not subtracted and no `B-1` rule applies.
The equation is evaluated only on one exact common all-member domain.

## `SCI-NOI-EQ-010` - proposed reciprocal successor

```text
rho_cond(p) = 1 / Vhat_cond(p).
```

This r0.2 successor spelling is unavailable pending owner acceptance and
profile/source rebinding. If accepted, its domain is only the exact finite
strictly positive parent domain. It is not inverse variance, precision,
validity, support, exposure, a consumer weight, or a PTC/MAP coefficient.

## `SCI-NOI-EQ-011` - canonical conditional scale

```text
sigma_cond(p) = sqrt(Vhat_cond(p)).
```

The scale exists only on the exact finite-positive compatible parent domain
and has the signal unit of `M_b` and `m_MAP`.

## `SCI-NOI-EQ-012` - canonical standardized MAP signal

```text
zeta_cond(p) = m_MAP(p) / sigma_cond(p).
```

The output exists only on the exact compatible finite-positive intersection,
has unit exactly `1`, and carries the claim in `SCI-NOI-REQ-031` only.
