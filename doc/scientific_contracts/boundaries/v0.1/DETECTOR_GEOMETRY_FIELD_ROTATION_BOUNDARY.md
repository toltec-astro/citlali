# Detector Geometry And Field-Rotation Boundary

Boundary identity: `DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY v0.1/r0.1`

Status: owner-decision-complete candidate; exact artifact approval pending

Prepared: `2026-08-23`

Scientific owner: Grant Wilson

Boundary owners: measured-geometry/APT authority and SCI-AST boundary owner

## Purpose And Authority

This package-neutral boundary defines the static producer interface and
runtime binding rule by which measured relative detector geometry becomes an
observation-specific AST geometry realization. It supplies candidate
authority for audit finding `F-007` without changing SCI-BEAM fitting or
SCI-AST coordinate mathematics.

It composes, without superseding:

- SCI-BEAM v0.1/r0.3 relative-geometry, derotation, same-APT, covariance, and
  provenance authority, especially Equations for centroid-to-detector,
  affine rotation, and derotation and Requirements 036--038 and 044;
- SCI-AST v0.1/r0.3 geometry and ordered-composition authority, especially
  Requirements 023--034; and
- the exact ALIGN occurrence-time and aligned observing-state relation used
  by AST.

The general frozen interface does not embed every observation-specific APT
instance. Each reduction binds its actual immutable geometry and association
records at runtime.

## Ownership Separation

SCI-BEAM owns measured relative detector geometry, its measurement support,
representation, limitations, uncertainty state, and parent Beammap lineage.

An exact named TolAPT/TolProj association authority owns selection and binding
of the applicable APT realization to the observation and its Tune/readout and
detector-occurrence relations.

SCI-AST consumes those authorities and composes geometry with the exact
occurrence-local pointing and observing state. AST does not choose, refit,
recentre, or reinterpret measured geometry. SCI-CAL does not acquire geometry
ownership by consuming APT calibration fields.

## Portable And Parent-Beammap Coordinate Roles

The canonical portable detector coordinates are

```text
xi[d] = (x_t[d], y_t[d]).
```

They are horizon-referenced relative focal-plane coordinates. The target
observation's field rotation has not been pre-applied. Their target-observation
field-rotation application count is zero before AST composition.

Coordinates with the registered `rot` semantic, such as
`x_t_rot` and `y_t_rot`, are different derived quantities. Their rotation is
the realized rotation of their exact parent Beammap, not a reusable rotation
for another observation. They bind that Beammap's identity, time/elevation and
pointing support, rotation law and realized angle, basis, pivot, sign, units,
application count, and provenance.

For target observation `o` and parent Beammap `b`, the ordinary relation is

```text
theta[d,o] = AST(pointing[o], R[o], xi[d]),
```

not

```text
theta[d,o] = (x_t_rot[b,d], y_t_rot[b,d]).
```

An exact `rot` quantity may be used only within its own parent Beammap support.
It does not establish that rotation has been applied for another observation.
If only the parent-specific rotated quantity exists, recovery of portable
geometry requires a separately authorized exact inverse transform; no inverse
is inferred.

`rot` is a registered schema semantic, not an arbitrary filename-substring
heuristic. Unknown or ambiguous coordinate roles fail compatibility.

## Portable Geometry Versus Pointing Correction

Portable `x_t,y_t` describe relative detector layout in their declared
horizon tangent basis. A parent Beammap's global translation, centering error,
or pointing offset does not silently travel with that layout into a target
observation.

The geometry artifact declares its ordered basis, handedness, angular units,
origin, gauge, and conventional pivot. Any admitted instrument-relative
scale, rotation, skew, or shear term is explicit. Unresolved affine modes are
typed and are not set to identity or zero.

A Beammap-specific pointing correction has a separate owner, target and
support. Recentring or changing gauge requires an explicit equivalence
transform. A conventional origin or ensemble translation is not physical
boresight truth.

## Exact Producer And Runtime Binding

The selected geometry realization binds:

| Role | Required exact content |
| --- | --- |
| Geometry authority | SCI-BEAM product/version, exact APT artifact identity and content digest, parent Beammap identity, measurement/fit support, detector population, state, limitations, and provenance. |
| Observation association | Target observation, Tune/readout mapping, detector-occurrence relation, exact TolAPT/TolProj association or child-artifact transform, selection authority/version, validity, and compatibility. |
| Coordinate representation | Registered coordinate role, horizon basis, ordered axes, handedness, tangent-increment meaning, angular units, origin/gauge, conventional pivot, and whether the quantity is portable or parent-Beammap-specific. |
| Rotation state | Target-observation rotation not embodied in portable `x_t,y_t`; every parent-Beammap rotation embodied in a `rot` quantity; exact law, zero, sign, units, input/output frames, time/elevation support, pivot, and application count. |
| Affine state | Explicit admitted scale, skew, shear, translation, recentering, or other affine terms; unresolved modes; ordered composition; and equivalence-transform identity where applicable. |
| Uncertainty | Geometry covariance or typed unavailability, axes, units, detector ordering, pivot/gauge, included and omitted modes, cross terms, support, and transformation history. |
| Provenance | Exact producer, association, transform, parent, digest, lifecycle, compatibility/supersession, and application history sufficient to reconstruct the selected realization. |

Matching filenames, dates, arrays, row counts, detector numbers, or shapes do
not establish observation association or compatibility.

## Field-Rotation Application Rule

AST applies the target observation's selected, versioned field-rotation law
exactly once to portable `x_t,y_t`, using occurrence-local observing state from
the exact aligned parent. It does not use the rotation embodied by another
Beammap and does not infer rotation from visual orientation or approximate
agreement.

Every operation records its input/output frame, sign, zero point, units,
pivot, support, and application count. AST applies only operations absent from
the selected representation. A missing, ambiguous, or duplicate required
rotation fails coordinate realization.

## Gauge, Equivalence, And Same-Realization Rule

Bracketing pointing observations and their associated science observation use
the same immutable APT artifact and AST rotation convention unless a
separately authorized transform proves equivalence. Such a transform binds
both artifacts, detector relation, domain/range, basis, units, gauge/pivot,
ordered affine operations, uncertainty propagation, validity support, and
provenance.

No raw-coordinate equality test proves equivalence across gauges. A transform
that does not account for a material translation, scale, rotation, skew,
shear, detector remapping, covariance, or unresolved mode is incomplete.

## Availability And Failure

Missing or incompatible numerical geometry, target-observation association,
basis, units, detector relation, or required rotation state makes pointing
unrecoverable and invokes the observation-level halt in the RTC-to-AST
boundary. No ordinary or companion-ML observation product is published.

Numerically valid geometry with unavailable covariance may still realize a
coordinate. It carries `geometry_uncertainty_unavailable` and cannot support a
complete astrometric covariance, total pointing uncertainty, or precision
claim. Missing covariance is never replaced by zero. A supplied covariance is
transformed consistently with every basis, gauge, pivot, or affine change.

## Compatibility And Supersession

Compatibility requires this exact boundary identity and preservation of the
portable-versus-parent-Beammap coordinate distinction, exact association,
registered representation, application counts, gauge/pivot, affine state,
uncertainty availability, failure semantics, and provenance. A successor
shall name this revision and provide a complete semantic mapping.

This boundary does not establish a physical boresight or rotation pivot,
validate a geometry artifact, assess implementation conformity, or authorize
production use.

