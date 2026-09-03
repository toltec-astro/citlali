# SCI-POINT Search, Association, Fallback, And Sentinel Specification

Identity: `SCI-POINT_SEARCH_ASSOCIATION_ENVELOPE v0.1/r0.3`

The known-source method distinguishes authoritative expected position,
requested search center, effective search center, selected peak/seed, fitted
centroid, and final source-association state.

The mature procedure's branch-independent association method, central-search
domain, peak score and weight, candidate
population, interpolation, tie rule, parameter initialization, fallback
trigger/domain, fit-support motion, constrained resolution, multiple-minimum
rule, termination, retry, and sentinel mapping remain unavailable pending the
compatibility-method record. Requested, effective, resolved, applied, and
realized search/fallback states must remain distinct when that record becomes
available.

Every central-search, global-fallback, restart, retry, or other authorized
search branch produces a candidate seed, not source identity. Source
association must be established independently over a declared association
domain using one branch-independent method supplied by the future
compatibility record. Global fallback remains an internal known-source branch,
not blind detection.

For every attempted fit, retain branch identity, seed/peak identity, fitted
centroid, expected source identity, association domain and method, and
`POINT-SOURCE-ASSOCIATION-STATE` with exactly `established`, `unavailable`, or
`failed` plus cause. A central-window result, global maximum, finite fit, small
offset, large fitted amplitude, dynamic-range ratio, or formal standardization
cannot establish association.

Source-attributed displacement and amplitude require `established`
association. A numerical fit to an unassociated feature may remain a fit
diagnostic but cannot inherit the intended source identity.
