# SCI-POINT v0.1 Author Predecessor Boundary Inputs

Identity: `SCI-POINT_PREDECESSOR_BOUNDARIES v0.1/r0.3`

Status: compact sanitized ownership input; no predecessor bytes imported

## Exact Ownership Imports

| Boundary | Imported meaning | POINT prohibition |
| --- | --- | --- |
| MAP/JINC | immutable map estimand, unit, normalization, WCS/grid/frame, support/validity, response, covariance state, lifecycle, and provenance | do not redefine, infer, or equate routes |
| FLT-FIXED/FLT-MATCHED | exact transformed signal, operator/template, normalization, response, edge/support, covariance, phase/origin, and parent lineage | do not call a transformed fit equivalent to an ordinary-map fit without separate authority |
| FRUIT | exact terminal map type plus method, iteration, generation, recurrence response/support/uncertainty, and restart lineage | do not invent a generic FRUIT map type or accept an intermediate iteration |
| AST | AltAz tangent-basis and WCS realization; later correction interpretation/application under producer-selected records/support | do not select, compose, interpolate, or apply a correction |
| pointing-support producer | aggregation, partial-member policy, measurement-to-correction sign, telescope-offset composition, selection, native support, and correction-record publication | do not publish an aggregate or correction record |
| BEAM | per-detector Beammap fits, detector PSF/sensitivity/APT and beam authority | do not infer intrinsic beam or take per-detector ownership |
| CAL/TolProj | authorized photometric reference/transfer and amplitude-use policy | do not claim universal flux or absolute calibration |
| NOI | empirical uncertainty and later uncertainty companions | do not infer empirical uncertainty from map weights or formal errors |
| VAL | registry and evaluator of exact named-use policies and source bindings | do not assign policy authorship or implicit composition to VAL |

## Coordinate And Product Conventions

- Pointing displacement uses one exact declared AltAz tangent basis and
  arcseconds. The boundary must name the ordered basis vectors, handedness,
  positive directions, time/state, map transformation, validity, uncertainty,
  and failure. Axis labels and memory/display/FITS order are not authority.
- Continuous fitted map coordinates must be transformed through the exact
  parent WCS/pixel metric; memory order is not an axis-sign convention.
- A fitted displacement is not an absolute spherical coordinate and is not an
  action to apply.
- Fitted shape uses the declared tangent-plane metric. The generic contract
  may use a positive-definite shape matrix, but the numerical published width
  convention and angle gauge remain unavailable until the compatibility-method
  record is owner-approved. Exact circularity makes angle undefined by model
  symmetry.
- Every product binds exact observation, array, parent, method, version,
  generation, and complete applicable ancestry.

## Availability Rule

This compact ownership input establishes no numerical predecessor
availability. The Stage B contract shall retain typed unavailability until the
exact predecessor product and compatibility state are bound.
