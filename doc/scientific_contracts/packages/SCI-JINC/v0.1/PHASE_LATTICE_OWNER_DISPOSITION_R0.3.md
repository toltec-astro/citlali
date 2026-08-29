# SCI-JINC v0.1 Phase-Lattice Owner Disposition r0.3

Status: owner disposition incorporated into an implementation-blind Stage B
author draft; final Stage B acceptance and freeze remain pending

Prepared: `2026-08-29`

## Authority

The final targeted owner-review directive has SHA-256
`4878e1745e085b4e33d2e71f1190299d72f2cfd7b2215e36a9e8405a977bd207`.
The scientific owner's exact response is `I approve disposition A`; the
literal response text has SHA-256
`9a70cbc63c0c79a7db70ad9796481fb0fe3f1f4c2d7524820b54cec68b8b1620`.

The directive asks the owner to choose A, B, or C and also to confirm or
separately disposition the positive-axis half-pixel center tie. The direct
approval of A is applied to the complete presented Disposition-A choice,
including its paired center-tie convention.

## Disposition

Disposition A is retained:

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

## Contract Effect

The shared notation, definitions, equations, assumptions, requirements, and
predictions carry this owner decision. `SCI-JINC-PRED-017` is amended to make
the exact even-lattice center behavior falsifiable. No requirement or
prediction identifier is added, removed, or renumbered.

This disposition supplies no TolTEC numerical parameter, coefficient family,
numerical-adequacy profile, implementation fact, validation result, or
performance/readiness/production claim.
