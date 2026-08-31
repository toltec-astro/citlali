# SCI-FLT-FIXED v0.1 Real Finite Scalar and Coefficient Declaration

Record identity: `SCI-FLT-FIXED-REAL-FINITE-COEFFICIENT-DECLARATION v0.1/freeze-candidate`

Status: implementation-blind conditional freeze-candidate closure artifact; owner signature required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

Normative authority remains the shared normative core. The base contract is

```text
S_parent_fact = exact parent row-identity and fact domain;
D_m = rows in S_parent_fact with an available finite real signal payload;
m : D_m -> R;
y : S_out -> R;
k_Theta(r), L_Theta, and A_Theta,J are real-valued.
```

A typed missing, unavailable, or non-finite stored payload is not an element
of `m`. Membership in `D_m` is established before `m_q` is evaluated.

Every sampled coefficient is finite, real, unit-typed, present in one
canonical exact representation, and content-bound before application. A
missing, non-finite, complex, unrepresentable, or conflicting coefficient
makes plan resolution unavailable. Prospective numerical-comparison policy
cannot repair coefficient inadmissibility.

`H(nu)` may be complex as a frequency-domain representation of the real map
operator; it does not authorize complex `FLT-SIG`. The real base covariance
rule remains `C_out = A C_parent A^T`. A future complex-valued method requires
separate authority for units, conjugation, Hermitian covariance, and
`A C A^dagger` propagation.
