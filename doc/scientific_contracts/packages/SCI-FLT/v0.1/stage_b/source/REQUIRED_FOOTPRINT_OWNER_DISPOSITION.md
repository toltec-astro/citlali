# SCI-FLT-FIXED v0.1 Required-Footprint Owner Disposition

Record identity: `SCI-FLT-FIXED-REQUIRED-FOOTPRINT-DISPOSITION v0.1/draft-r0.3`

Status: explicit r0.3 scientific-owner disposition; proposed freeze input

Scientific owner: Grant Wilson

Stage B date: `2026-08-30`

## Decision

The r0.3 owner directive's preferred disposition is adopted:

```text
K_geom_science = exact representation-invariant scientific geometry
K_store        = nonauthoritative storage/serialization footprint
K_nonzero      = canonical exactly nonzero coefficient offsets
K_req          = K_nonzero for ordinary FLT-FIXED-CONV
```

Exact zero is decided from the canonical coefficient representation, never a
floating tolerance. Dense, sparse, cropped, and zero-padded encodings of one
scientific kernel do not change scientific support, `S_out`, response,
covariance, or product identity. A required zero-valued offset needs a
separately named method and a scientific reason independent of storage.

Identity retains `K_req = {0}`. The exact zero operator inherits one exact
admitted parent-support row domain and cannot acquire arbitrary storage rows
through an empty-footprint predicate.

## Supersession

This disposition supersedes only the r0.2 rule that made stored geometric
zeros ordinary required dependencies. It preserves the full-footprint-only
edge method and every edge, missing, and non-finite exclusion.
