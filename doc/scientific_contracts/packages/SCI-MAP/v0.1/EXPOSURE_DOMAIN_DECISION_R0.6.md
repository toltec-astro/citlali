# SCI-MAP v0.1 r0.6 Exposure Domain And Placement Decision

Physical valid-original exposure is defined on unique stable original
occurrences, not RTC/PTC descendants. The key is observation, detector
occurrence/UID, native original occurrence, and ALIGN mapping generation.
Deduplication is within one observation/detector lineage and coadd union is
over observation-scoped keys.

Each positive `e_vo,u` original is placed exactly once by that original's own
authorized AST coordinate under `SCI-MAP:one_hot_containing_pixel@1`.
Retained exposure restricts originals to ancestors of admitted MAP outputs;
it does not relocate seconds to descendant coordinates. Full temporal
influence remains a separate causal/support graph. One original may influence
many descendants and pixels but contributes physical exposure to at most one
pixel. Overlapping filters, decimation, donor reuse, replacement, and
synthesis create no seconds. Invalid originals contribute zero valid-original
exposure. Out-of-grid and outer-upper-boundary originals are lost, never
wrapped or clamped.

Units are detector-seconds. Exposure is neither hits, duration, cadence,
sample count, normalization, coefficient, precision, nor response. ECS must
include one original influencing multiple descendants at different pixels and
prove one own-coordinate placement with unchanged total seconds.
