# SCI-AST to SCI-MAP Original-Footprint Coordinate Boundary

Boundary identity: `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`

Status: implementation-blind scientific source binding for SCI-MAP v0.1/r0.7;
no implementation conformity, validation, response fidelity, freeze,
readiness, numerical-route availability, or production claim

## Purpose and authority

This boundary binds MAP physical-original-footprint exposure to coordinate
roles already defined by frozen SCI-AST v0.1/r0.3. It adds no astrometric
algorithm, reconstructs no coordinate, and does not modify SCI-AST authority.
Its exact AST sources are packet-manifest SHA-256
`b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`,
Equation `align-parent`, Equations `tangent-parent`--`pixel-parent`, and
`SCI-AST-REQ-073`, `SCI-AST-REQ-080`, and `SCI-AST-REQ-081`.

## Bound object

For each stable original acquisition occurrence `u`, the exposure coordinate
record binds all of the following as one immutable identity:

- observation identity;
- detector occurrence and UID;
- stable original/native occurrence identity and stable ALIGN slot `(o,s)`;
- exact ALIGN mapping generation and `SCI-ALIGN_TO_SCI-AST v0.1/r0.1` parent;
- AST ALIGN-grid direction parent `Pi^(A,direction)_(ds)`, including exact
  ALIGN plan/grid/source relation, observing state, pointing correction,
  geometry/rotation realization, and coordinate frame;
- when pixel placement is requested, the successive AST tangent parent and
  continuous-pixel parent `Pi^(A,pixel)_(ds)`, extended by the exact target
  MAP WCS and no other scientific dependency;
- coordinate availability/validity and exact cause;
- the `SCI-MAP:one_hot_containing_pixel@1` boundary state, including finite
  in-grid placement or lower/upper/out-of-grid loss;
- this boundary identity/version, source versions, lifecycle generation, and
  exact parent product identity.

The target MAP WCS is the same immutable WCS/grid identity used by the MAP
product whose exposure is being formed. The continuous pixel is therefore the
original occurrence's own AST projection into that target WCS. Numerical
equality to a descendant coordinate does not establish identity.

## Prohibitions and failure

The coordinate shall not be taken from
`SCI-AST:rtc_output_grid_coordinates@1`, inferred from an RTC representative,
copied from any PTC descendant, or reconstructed from time, row, cardinality,
or numerical equality. The RTC-grid role remains the ordinary signal-route
coordinate; this boundary is only the original-footprint exposure role.

Unavailable or invalid direction, tangent, continuous-pixel, WCS, parent, or
boundary state makes the affected exposure role unavailable. Boundary loss
places no seconds. It does not relocate the original to a descendant pixel.

## Scientific interpretation

The coordinate places unique original acquisition footprints for
`upstream_eligible_original_footprint_exposure` and
`retained_original_footprint_exposure`. Those products are not complete
temporal support, effective map integration time, precision, or the causal
influence of the normalized map. Causal influence and representative RTC
occurrence remain separate published facts.

## Compatibility and supersession

Compatibility requires exact preservation of the object, layered AST parent,
target-WCS binding, stable-original identity, coordinate validity, boundary
semantics, and failure behavior above. Any change requires a new immutable
boundary version. Similar names, shapes, coordinate values, or descendant
roles are not compatible substitutes.
