# SCI-FLT-FIXED v0.1 WCS, Kernel, And Discretization Decision Table

Status: sanitized Stage A author candidate awaiting exact-byte owner approval

| Fact | Base v0.1 decision | Missing/conflict action |
| --- | --- | --- |
| Parent/output grid | Same-grid only; output preserves the exact parent WCS, frame, topology, metric, shape, and pixel indexing while scientific rows are restricted by `S_out` | Route unavailable; no approximate-WCS join |
| Reprojection/resampling | Excluded | Requires a versioned successor |
| Operator domain | Exact finite parent row domain to exact full-footprint output row domain | Route unavailable |
| Coordinate metric | Exact WCS-derived spatial metric named with units | Low-pass/physical-scale claim unavailable; operator unavailable if coefficients cannot be interpreted |
| Pixel area | Exact parent/output pixel-area convention bound to coefficient units and angular normalization | Affected normalization/beam claim unavailable |
| Kernel identity | Immutable family, version, parameter set, sampled coefficients, and content digest | Route unavailable; name is insufficient |
| Center | Exact discrete center in the parent grid convention | Route unavailable |
| Even/odd extent | Exact extent and deterministic tie convention | Route unavailable |
| Phase/subpixel state | Exact sampled phase; no implicit recentering or interpolation | Route unavailable |
| Orientation/handedness | Exact relation to WCS axes; anisotropy named | Route unavailable or anisotropy claim unavailable as applicable |
| Finite support | Exact offset set `K_Theta`; signed, absolute, and squared support summaries remain distinct | Route unavailable |
| Coefficient units | Derived from input/output quantity and declared normalization | Output-unit claim unavailable |
| Numerical representation | Exact coefficient representation and ordering sufficient to reconstruct `L_Theta` | Applied-operator identity unavailable |
| Implementation mechanism | FFT/direct/separable/cache/threading are non-scientific choices only if they realize identical declared `L_Theta` | Difference is a different realized operator or nonconformity question; no scientific equivalence inference |
| Periodic wrapping | Forbidden in v0.1 | Affected rows unavailable; no implicit FFT periodicity |

This table selects no numerical kernel, WCS, resolution, cutoff, or parameter
value. It defines the facts a future method must bind.
