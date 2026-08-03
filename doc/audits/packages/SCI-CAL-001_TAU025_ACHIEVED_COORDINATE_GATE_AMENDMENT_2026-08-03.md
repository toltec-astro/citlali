# SCI-CAL-001 tau225 achieved-coordinate gate amendment — 2026-08-03

Status: owner-directed bounded numerical-gate correction; no AM evidence has been
accepted and no application action is authorized.

Decision ID: CAL-ATM-D007-ACHIEVED-COORDINATE-001

## Finding

The no-AM preflight of the approved TAU025 engineering-extension request stopped
solely because the display table for tau01625 ended in ...1410, whereas a
high-precision recomputation of the stated formula yielded ...140795.... The
absolute difference is approximately 2.04e-34.

This is neither an AM input discrepancy nor a physical model discrepancy. The
actual AM anchor is the request's seven-significant-digit target transmission
literal, which remains exact and unchanged. The achieved tau225 value is a
derived provenance annotation.

## Approved correction

Amend only gate 3 of the execution request:

1. The requested tau225 decimal, EL80 airmass constant, target-transmission
   literal, AM parser convention, and exact parsed-literal equality remain
   mandatory and byte-for-byte unchanged.
2. Recompute and serialize the achieved coordinate at high precision as
   provenance. It is not a second AM target and must not be used to select a
   scale or substitute for the target literal.
3. For the printed achieved-coordinate reference table, require an absolute
   difference no greater than 1e-12. This is a formula/provenance consistency
   check, over four orders of magnitude tighter than the approximately 1e-8
   resolution implied by the seven-significant-digit target literal; it is not
   a representation-fidelity or photometric-error criterion.
4. Record the exact recomputed decimal string and the absolute difference for
   every node. A difference above 1e-12, a changed target literal, or a
   failure of parsed-literal equality remains fail-closed.

The request may correct the tau01625 display value to its properly rounded form,
but it need not do so for the AM study to proceed under this comparison rule.
The cache remains absent and all other D007 gates, WARN-001, tuple inventory,
profile/passband identities, 5% held-out engineering screen, and
non-authorizations remain unchanged.

## Scope

This amendment authorizes only the CAL task to make the corresponding
documentation/provenance-gate correction, refresh digests, rerun no-AM
preflight, and continue to the already approved direct-AM study only if every
registered gate passes. It does not authorize candidate fitting or selection,
operator adoption, implementation, repair, Unity, re-audit, or production use.
