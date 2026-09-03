# SCI-MAP v0.1 r0.7 Exposure Identity, Coordinate, And Interpretation

Physical valid-original exposure is defined on unique stable original
occurrences, not RTC/PTC descendants. The key is observation, detector
occurrence/UID, stable original/native occurrence, stable ALIGN slot, and
ALIGN mapping generation.
Deduplication is within one observation/detector lineage and coadd union is
over observation-scoped keys.

Each positive `e_vo,u` original is placed exactly once under
`SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`: its own layered
AST ALIGN-grid direction/tangent/continuous-pixel parent in the exact target
MAP WCS, with observation, UID, native occurrence, stable ALIGN slot and
generation, geometry/pointing realization, frame, coordinate validity,
boundary state, parent, and version. A descendant RTC-output coordinate is
never substituted. Placement then uses
`SCI-MAP:one_hot_containing_pixel@1`.
Retained exposure restricts originals to ancestors of admitted MAP outputs;
it does not relocate seconds to descendant coordinates. Full temporal
influence remains a separate causal/support graph. One original may influence
many descendants and pixels but contributes physical exposure to at most one
pixel. Overlapping filters, decimation, donor reuse, replacement, and
synthesis create no seconds. Invalid originals contribute zero valid-original
exposure. Out-of-grid and outer-upper-boundary originals are lost, never
wrapped or clamped.

The normative products are
`upstream_eligible_original_footprint_exposure` and
`retained_original_footprint_exposure`. Units are detector-seconds. They map
unique original acquisition footprints and are neither complete temporal
support, effective map integration time, normalized-map influence, hits,
duration, cadence, sample count, normalization, coefficient, precision, nor
response. Causal influence and representative RTC occurrence are separate.

Required fixtures cover: one original influencing descendants at different
pixels; original and descendants in different pixels; missing original
coordinate; boundary loss; multiple maps sharing one original; distinct
observations with numerically identical original coordinates; and donor and
synthesized descendants.
