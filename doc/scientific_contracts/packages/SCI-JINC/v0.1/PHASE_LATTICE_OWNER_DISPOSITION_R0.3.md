# SCI-JINC v0.1 Phase-Lattice And Center-Tie Owner Decision r0.3

Stable decision ID: `SCI-JINC-DEC-PHASE-CENTER-001`

Status: owner-approved and frozen as part of the conditional,
implementation-independent `SCI-JINC v0.1/r0.3` scientific authority

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

## Exact Authority

The exact final targeted r0.3 directive has SHA-256
`4878e1745e085b4e33d2e71f1190299d72f2cfd7b2215e36a9e8405a977bd207`.
Its Disposition A covered the phase lattice only and separately requested
owner confirmation of the positive-axis half-pixel center tie.

The phase-lattice statement is exactly:

> I approve disposition A

Its literal UTF-8 text without a trailing newline has SHA-256
`9a70cbc63c0c79a7db70ad9796481fb0fe3f1f4c2d7524820b54cec68b8b1620`.

The exact final owner-review and freeze-preflight directive has SHA-256
`958cffeac67c11e916527c0f78e9c80d648f68d5eec38a0607fc4af1511dddec`.
It prohibited inferring the separate center decision from Disposition A.

The separately supplied center-tie statement is exactly:

> I approve the positive-axis half-pixel center-tie convention
> \\(c=\lfloor u+\tfrac12\rfloor\\) as part of SCI-JINC v0.1.

The literal UTF-8 text on one line, without a trailing newline, has SHA-256
`3b79351f7661e2432a5426fba3a16e9710c2fae0b34fd9b8f60dd45bca837ecb`.
The mathematical identity is `c = floor(u + 1/2)`. The two owner statements
are separately authoritative and jointly bound by
`SCI-JINC-DEC-PHASE-CENTER-001`; neither is inferred from the other.

## Decision

- every positive integer `n_sub` is permitted;
- phase bins are left closed and right open;
- an exact interior-boundary phase selects the upper bin;
- for even `n_sub`, exact phase zero selects `+1/(2 n_sub)` on each
  applicable WCS axis;
- an exact positive half-pixel center tie selects the larger integer center;
- there is no zero-phase special case and no parity restriction.

For even `n_sub`, an exact integer-center occurrence with zero phase on both
axes and central offset has

`r_hat = Delta/(sqrt(2) n_sub)`

and coefficient

`kappa_a(Delta/(sqrt(2) n_sub s_a); a_a,b_a,c_a,(r_max)_a)`.

That coefficient is the exact analytic kernel at the stated nonzero
dimensionless radius. It is not replaced by an imposed `kappa_a(0)=1`.

## Affected Authority

The decision binds:

- notation for `c_i`, `phi_i`, `n_sub`, `q_i`, and `phi_hat_i`;
- the Point phase definition;
- equations `rounded-center`, `residual-phase`, `phase-bin`, `phase-index`,
  `phase-representative`, `phase-cache-index`, `square-wcs-metric`, and
  `discrete-radius`, including the unnumbered even-lattice central formula;
- `SCI-JINC-ASM-006`;
- `SCI-JINC-REQ-007`, `SCI-JINC-REQ-010`, `SCI-JINC-REQ-011`,
  `SCI-JINC-REQ-012`, and `SCI-JINC-REQ-044`; and
- `SCI-JINC-PRED-016` and `SCI-JINC-PRED-017`.

No requirement, prediction, assumption, or equation identifier is added,
removed, or renumbered by this decision.

## Compatibility And Supersession

This record supersedes the r0.3 draft's inference that the phase-only
statement also approved the center tie. It preserves the already authored
mathematical operator because the owner has now separately approved that exact
center rule. A realization is compatible with `SCI-JINC v0.1/r0.3` only when
it uses both rules exactly under this decision ID. Any alternate phase lattice,
tie direction, center equation, parity restriction, or zero-phase special case
is incompatible and requires a versioned SCI-JINC successor.

After freeze, this record and the authority bytes it binds are immutable. Any
later scientific correction shall be recorded only in a versioned successor.

This decision supplies no TolTEC numerical parameter, coefficient family,
numerical-adequacy profile, implementation fact, validation result, or
performance, readiness, production, or production-authorization claim.
