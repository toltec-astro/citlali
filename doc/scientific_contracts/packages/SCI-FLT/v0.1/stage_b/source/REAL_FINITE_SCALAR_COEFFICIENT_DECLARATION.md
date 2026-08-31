# SCI-FLT-FIXED v0.1 Real Finite Scalar and Coefficient Declaration

Record identity: `SCI-FLT-FIXED-REAL-FINITE-COEFFICIENT-DECLARATION v0.1/draft-r0.4`

Status: implementation-blind Stage B closure artifact; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

Normative authority remains the shared normative core. The base contract is

```text
m, y, k_Theta(r), L_Theta, and A_Theta,J are real-valued.
```

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
