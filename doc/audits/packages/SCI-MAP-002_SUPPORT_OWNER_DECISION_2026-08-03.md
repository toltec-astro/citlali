# SCI-MAP-002 JINC support owner decision — 2026-08-03

Status: owner approved contract clarification; no implementation work
authorized

Package: `SCI-MAP-002`

Decision ID: `SCI-MAP-002-D003-SUPPORT-001`

Authority: project owner

## Decision

The realized JINC footprint is the approved scientific convention. For each
array, `r_max` has two deliberately coupled meanings:

1. it sets the first zero of the second JINC factor; and
2. it sets the half-width of the fully populated square cache used for
   deposition.

The support is therefore a square cache support, not a radial cutoff. Kernel
values in the square corners, including values at radii greater than
`r_max`, are part of the defined response. The name `r_max` must never be
described as a strict radial support maximum without an explicitly distinct
future parameter.

This decision preserves existing response and avoids an unapproved change to
JINC footprint, beam response, or array-parameter tuning. A future
implementation/provenance repair must serialize the square-support and
dual-use-`r_max` convention explicitly, preserve cropping at map edges, and
validate below/equal/above `r_max` plus corner and edge fixtures. It may not
introduce a radial predicate, alter JINC parameters, or begin a numerical
optimization campaign under this authority.

The remaining SCI-MAP-002-D003 decisions—subpixel response, conditioning,
parameter/coefficient admission, coverage/mask/kernel identity, and realized
provenance—remain open. No code change, Unity evidence, repair, re-audit, or
production-status change is authorized.
