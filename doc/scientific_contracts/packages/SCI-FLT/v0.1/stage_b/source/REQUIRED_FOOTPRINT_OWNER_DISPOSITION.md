# SCI-FLT-FIXED v0.1 Required-Footprint Owner Disposition

Record identity: `SCI-FLT-FIXED-REQUIRED-FOOTPRINT-DISPOSITION v0.1/freeze-candidate`

Status: historical r0.4 disposition repaired by the final conditional-freeze directive and bound into the freeze candidate

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

## Decision

The r0.4 owner directive's corrected disposition is adopted:

```text
K_geom_science = exact representation-invariant scientific geometry
K_store        = nonauthoritative storage/serialization footprint
K_nonzero      = canonical exactly nonzero coefficient offsets
K_req          = K_nonzero for ordinary FLT-FIXED-CONV
```

Exact zero is decided from the canonical coefficient representation, never a
floating tolerance. Dense, sparse, cropped, and zero-padded encodings of one
scientific kernel do not change scientific support, `S_out`, response,
covariance, scientific operator identity, product identity, or scientific
generation; their representation identities may differ. A required zero-
valued offset needs a separately named method and a scientific reason
independent of storage.

Ordinary arithmetic sums exactly over `K_nonzero = K_req`, not over
`K_geom_science`. An exact-zero coefficient creates no arithmetic term,
payload dependency, influence, covariance contribution, or row exclusion, and
its parent payload is not evaluated or dereferenced.

Identity retains `K_req = {0}`. The exact zero operator uses empty
`K_nonzero_zero` and `K_req_zero` and independently constructs `S_out_zero`
from the exact admitted finite parent-signal rows under its request and
predicates. It cannot acquire arbitrary storage rows through an empty-
footprint predicate.

A requested nonzero convolution with no complete admitted footprint is
`applied_no_scientific_output_support` and creates a complete
`no_output_support_candidate`; publication axes are requested, applicable,
ineligible, and not produced, with cause `no_full_footprint_output_rows`, while
its plan, operator, parent, causes, and application evidence remain bound.

## Supersession

This disposition supersedes the r0.2 rule that made stored geometric zeros
ordinary required dependencies and corrects any r0.3 equation that summed over
geometric support. It preserves the full-footprint-only edge method and every
edge, missing, and non-finite exclusion at exact required nonzero offsets.
