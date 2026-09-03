# SCI-NOI v0.1 r0.5 Exact Rational-Parent Binding

Scientific owner: Grant Wilson. Date: 2026-08-30. Status: author-selected
scientific design detail under the r0.5 owner directive; proposed final; not
frozen.

The selected representation is a canonical reduced arbitrary-precision
rational pair `(n,q)`: `q>0`, `gcd(|n|,q)=1`, and zero is only `(0,1)`.
Positive `a_pi` has `n>0`; `tau_h` satisfies `0<=n<q`. Canonical external
serialization is ASCII `n/q` with base-ten integers, no leading plus sign, no
leading zeros except zero, and no whitespace.

Each design contribution satisfies `decode(n_pi/q_pi)=a_pi^MAP` exactly and
binds the frozen MAP product/generation, exact `G_pi`, exact `gamma_i`
family/generation/payload, exact parent `a_pi`, representation source, and
source digest. `beta_d`,
imbalance numerators, and totals are accumulated exactly in stable scientific-
identity order with arbitrary-precision integer arithmetic and canonical
reduction. For nonnegative `L_h=n_L/q_L`, `T_h=n_T/q_T`, and
`tau_h=n_tau/q_tau`, admission is exactly
`n_L q_T q_tau <= n_tau n_T q_L`.

The rational is not rounded from display text, independently quantized,
re-estimated, converted through undocumented decimal precision, or taken from
another generation. It governs NOI balance/design identity only and never
replaces independently governed MAP numerical arithmetic. Missing lossless
canonical rational identity makes design resolution unavailable.
Traversal-dependent floating reduction, undocumented epsilon, post-failure
tolerance relaxation, and best-failed candidate selection are prohibited. The
ECS binds equality/above/below boundaries and a parent/display mismatch fixture.
D-006, EQ-012/016, REQ-044/050, and PRED-020/025 carry this record.
