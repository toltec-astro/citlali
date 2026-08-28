# SCI-AST To SCI-JINC Coordinate Boundary

Boundary identity: `SCI-AST_TO_SCI-JINC v0.1/r0.2`

Status: ODQ-103 Stage A successor boundary candidate; exact-byte owner
approval required

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

## Authoritative Coordinate And Sample Association

AST owns the coordinate realization associated with each detector sample,
including its frame, units, coordinate validity/support facts, and parent-
sample identity. JINC consumes the AST RTC-output-grid **continuous pixel
coordinate** in the exact target JINC WCS that corresponds to the same
processed sample realization entering the JINC estimator. The association
remains exact across alignment, filtering, decimation, or any other change of
sample realization. The complete role binds:

| Facet | Exact binding |
| --- | --- |
| Occurrence | Observation, detector occurrence/UID, stable RTC output `n`, representative stable ALIGN slot, and complete ALIGN/RTC parent chain. |
| RTC relation | Exact RTC product and plan, output grid, selected output time, phase/delay/time-shift convention, segment, decimation factor/phase/support, temporal response, and coordinate-correction state. |
| PTC relation | Exact PTC product/application generation whose transformed signal and the AST coordinate refer to the same processed sample realization and immutable upstream ancestry. PTC does not create or relabel the coordinate. |
| Direction and frame | AST-owned detector direction and requested tangent/WCS/pixel roles; exact output frame, epoch/equinox, time scale, site/model applicability, ordered axes, units, and circular topology. |
| Target and WCS | Named target/center authority; immutable target JINC WCS identity including projection, CRVAL, one-based CRPIX, finite nonsingular pixel matrix, axis order/sign/rotation/handedness, dimensions/bounds, units, frame/epoch, plan and version. |
| Continuous coordinate | Finite continuous FITS pixel coordinate with its exact tangent and WCS parents. It is not a nominal integer pixel. |
| Validity and boundary | Dedicated AST role validity/cause plus continuous in-bounds/out-of-bounds state. Coordinate validity, JINC sample admission, sample-pixel support, and JINC final validity remain separate. |
| Geometry/pointing | Exact selected geometry/rotation, observing state, pointing correction and model realization already parented by AST. |
| Uncertainty | AST uncertainty/Jacobian availability and every typed unavailable term. The map-center Jacobian, when available, is an astrometric derivative, not JINC source response or covariance. |
| Provenance | Exact layered parent and requested/effective/observation-resolved/realized AST provenance. |

The AST WCS bundle is the scientific coordinate authority. FITS WCS is its
declared persisted representation, not an independently stronger authority.

The canonical scientific boundary statement is:

> AST provides the authoritative coordinate realization and parent-sample
> association for the detector sample entering JINC. JINC does not reconstruct
> AST pointing or infer a replacement coordinate. JINC owns the use of that
> coordinate relative to its destination pixels, its radial and normalized
> geometry, finite support, signed coefficient, and admission for JINC map
> contribution. Producer facts and cause accumulation cross the boundary; a
> producer-owned JINC-usability decision does not.

## Scientific Association And Ownership Rules

- The signal and coordinate must identify the same processed sample
  realization under compatible immutable ancestry and generations. Row order,
  nearest-time or tolerance matching, detector ordering, shape/cardinality,
  numerical coordinate equality, and other inferred fallbacks are prohibited.
  The key, index, table join, object relation, or other data-model mechanism
  used to establish the association is an engineering choice.
- Missing, duplicate, or ambiguous sample-coordinate association makes the
  coordinate unavailable for the affected JINC contribution. It is not
  reclassified as outside support.
- AST determines the coordinate facts. SCI-JINC owns local offset geometry,
  radial coordinate, dimensionless radius, center rounding, residual phase,
  phase bin, finite square support, signed kernel coefficient, edge crop,
  normalization, conditioning, response, covariance and accumulation.
- AST does not decide JINC support, calculate or authorize a JINC coefficient,
  manufacture a general JINC-valid flag, or encode JINC kernel semantics.
- AST may provide an optional nominal containing pixel only under its exact
  half-open rule, but SCI-JINC may not silently substitute that role for its
  still-unresolved rounded-center and phase convention.
- AST out-of-bounds state does not by itself decide whether an overlapping
  JINC square is admitted. `SCI-JINC-ODQ-110` must select the JINC edge rule.
- Missing or invalid RTC-parent facts make this dependent role unavailable.
  JINC does not borrow an ALIGN-grid coordinate, filter angles as signal, or
  reconstruct a coordinate from the PTC payload.

## Admission, Support, Coefficient, And Causes

- JINC owns admission for the named scientific use
  `SCI-JINC:jinc_map_contribution@1`. It does not inherit an ordinary SCI-MAP
  admission result or validity mask, or an upstream producer's conclusion
  about JINC usability.
- JINC sample admission decides whether an upstream sample may be considered.
  JINC kernel support separately decides whether that admitted sample
  contributes to a particular destination pixel.
- Outside finite JINC support and a contract-defined zero coefficient are
  ordinary no-contribution results, not upstream validity failures or defect
  causes. A finite negative coefficient is scientifically normal and is not an
  admission failure, invalidity, or cause.
- All JINC accumulators for one contribution use the same admitted sample-
  pixel pair and the same coefficient identity. Their contract-defined
  algebra may differ, but their association, admission, coordinate, and
  coefficient realization may not.
- Producer facts and applicable causes cross the boundary. A producer-owned
  JINC-usability decision does not. JINC may add causes for genuine local
  failures such as unavailable or ambiguous association/coordinate,
  unavailable authorized parameters, non-finite geometry, inadmissible
  coefficient evaluation, or another explicit JINC precondition.
- No cause is created merely because JINC and ordinary MAP reach different
  admission conclusions.
- No new per-contribution provenance system is created. Causes and support use
  established contract mechanisms at the existing appropriate product/support
  granularity.

## Compatibility And Failure

Compatibility requires frozen AST v0.1/r0.3, the exact role name, exact same-
processed-sample association, compatible ancestry/generations, the exact
target JINC WCS, `SCI-JINC:jinc_map_contribution@1`, and all typed validity/
uncertainty causes. A missing, duplicate, ambiguous, or conflicting required
coordinate association prevents the affected contribution and required
dependent product from realized success. A changed coordinate role,
projection authority, scientific association, admission identity, ownership,
or missing/conflict rule requires a versioned successor.
