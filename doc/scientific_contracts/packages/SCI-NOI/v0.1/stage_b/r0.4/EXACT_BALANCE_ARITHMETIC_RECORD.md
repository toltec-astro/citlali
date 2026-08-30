# SCI-NOI v0.1 r0.4 Exact Balance-Arithmetic Record

Scientific owner: Grant Wilson. Date: 2026-08-30. Status: author-selected
scientific design detail under the r0.4 owner directive; proposed final; not
frozen.

The selected representation is a canonical reduced arbitrary-precision
rational pair `(n,q)`: `q>0`, `gcd(|n|,q)=1`, and zero is only `(0,1)`.
Positive `a_pi` has `n>0`; `tau_h` satisfies `0<=n<q`. Canonical external
serialization is ASCII `n/q` with base-ten integers, no leading plus sign, no
leading zeros except zero, and no whitespace.

Each admitted coefficient is interpreted directly as this rational. `beta_d`,
imbalance numerators, and totals are accumulated exactly in stable scientific-
identity order with arbitrary-precision integer arithmetic and canonical
reduction. For nonnegative `L_h=n_L/q_L`, `T_h=n_T/q_T`, and
`tau_h=n_tau/q_tau`, admission is exactly
`n_L q_T q_tau <= n_tau n_T q_L`.

Missing canonical rational input makes design resolution unavailable.
Traversal-dependent floating reduction, undocumented epsilon, post-failure
tolerance relaxation, and best-failed candidate selection are prohibited. The
ECS binds equality/above/below boundary fixtures that floating conversion can
collapse. D-006, EQ-012, REQ-044, and PRED-020 carry this record.
