# SCI-AST To SCI-JINC Coordinate Boundary

Boundary identity: `SCI-AST_TO_SCI-JINC v0.1/r0.1`

Status: final Stage A boundary candidate; awaiting scientific-owner approval

Prepared: `2026-08-28`

Scientific owner: Grant Wilson

## Exact Frozen Source

This boundary cites frozen SCI-AST v0.1/r0.3. Its source-manifest SHA-256 is
`b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`.
The exact admitted sources include:

- `src/common/definitions.tex`, SHA-256
  `d03d04fb35026091e32e6071e53cbb40087dda8429d59c5a3c43f8151c07e5c3`;
- `src/common/requirements.tex`, SHA-256
  `47b357dd79136fb3d019f45b1092a2efd88fbd1ed16ab038fc6bf51beaf06f01`;
- `src/scientific-rationale.tex`, SHA-256
  `b8835be124e57245e1f9da850fb3a94f8d9407af10b4f569a3ec6f35b93a8ea9`;
  and
- `ROLE_FACTORED_PARENTAGE_MAP.md`, SHA-256
  `cd181110cbbc6b4834bfd0ce1d150db79eb9c3946a9c8fd52676509ea5ae8bf2`.

Those frozen bytes establish the exact role
`SCI-AST:rtc_output_grid_coordinates@1`. This boundary does not alter AST or
create a stronger coordinate source.

## JINC Coordinate Input

For the same stable RTC output sample `n` that parents `z_i=Z_i^PTC`, JINC
consumes the AST RTC-output-grid **continuous pixel coordinate** in the exact
target JINC WCS. The complete role binds:

| Facet | Exact binding |
| --- | --- |
| Occurrence | Observation, detector occurrence/UID, stable RTC output `n`, representative stable ALIGN slot, and complete ALIGN/RTC parent chain. |
| RTC relation | Exact RTC product and plan, output grid, selected output time, phase/delay/time-shift convention, segment, decimation factor/phase/support, temporal response, and coordinate-correction state. |
| PTC relation | Exact PTC product/application generation whose transformed signal retains the same RTC `n` parent. PTC does not create or relabel the coordinate. |
| Direction and frame | AST-owned detector direction and requested tangent/WCS/pixel roles; exact output frame, epoch/equinox, time scale, site/model applicability, ordered axes, units, and circular topology. |
| Target and WCS | Named target/center authority; immutable target JINC WCS identity including projection, CRVAL, one-based CRPIX, finite nonsingular pixel matrix, axis order/sign/rotation/handedness, dimensions/bounds, units, frame/epoch, plan and version. |
| Continuous coordinate | Finite continuous FITS pixel coordinate with its exact tangent and WCS parents. It is not a nominal integer pixel. |
| Validity and boundary | Dedicated AST role validity/cause plus continuous in-bounds/out-of-bounds state. Coordinate validity, bounds, JINC square support and JINC final validity remain separate. |
| Geometry/pointing | Exact selected geometry/rotation, observing state, pointing correction and model realization already parented by AST. |
| Uncertainty | AST uncertainty/Jacobian availability and every typed unavailable term. The map-center Jacobian, when available, is an astrometric derivative, not JINC source response or covariance. |
| Provenance | Exact layered parent and requested/effective/observation-resolved/realized AST provenance. |

The AST WCS bundle is the scientific coordinate authority. FITS WCS is its
declared persisted representation, not an independently stronger authority.

## Join And Ownership Rules

- The join is by exact occurrence, stable RTC `n`, complete parents and
  compatible generations. It is never by time, row, shape, cardinality,
  detector label, or numerical coordinate equality.
- AST determines where the occurrence is. SCI-JINC owns center rounding,
  residual phase, phase bin, square-cache placement, kernel value, edge crop,
  normalization, conditioning, response, covariance and accumulation.
- AST may provide an optional nominal containing pixel only under its exact
  half-open rule, but SCI-JINC may not silently substitute that role for its
  still-unresolved rounded-center and phase convention.
- AST out-of-bounds state does not by itself decide whether an overlapping
  JINC square is admitted. `SCI-JINC-ODQ-110` must select the JINC edge rule.
- Missing or invalid RTC-parent facts make this dependent role unavailable.
  JINC does not borrow an ALIGN-grid coordinate, filter angles as signal, or
  reconstruct a coordinate from the PTC payload.

## Compatibility And Failure

Compatibility requires frozen AST v0.1/r0.3, the exact role name, same-`n`
parentage, the exact target JINC WCS and all typed validity/uncertainty causes.
A missing or conflicting required coordinate role prevents the affected JINC
bundle from realized success. A changed coordinate role, projection authority,
join identity or missing/conflict rule requires a versioned successor.
