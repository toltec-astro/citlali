# WP-2 Timestream Boundary Candidate Change Map

Status: exact candidate approved and promoted to owner-approved boundary
authority on `2026-08-23`

Date: `2026-08-23`

Baseline audit commit: `55efd8a54464636a24e621f6d1b60486d235b20e`

Authoring baseline: SCI-PTC v0.1/r0.5 frozen authority at commit
`a18defe701bc824879a18cc6adafa6631fd22391`

Scientific owner: Grant Wilson

Candidate identity: commit
`967cf83452269038d13c7233ce40ba8e2c8a1790`

This candidate implements `WP2-OWNER-D001--D010`. Grant Wilson approved the
exact packet and authorized its status-only promotion on `2026-08-23`. The
promotion does not close an audit finding before the authorized clean-room
re-audit.

## Owner-Decision Realization

| Owner decision | Candidate realization | Principal artifact | Audit target |
| --- | --- | --- | --- |
| `WP2-OWNER-D001` | Attaches an RTC-grid coordinate to the phase-zero selected occurrence/time while preserving full temporal support; requires exact response semantics and recoverability without dense response serialization. | RTC-to-AST boundary: Stable Identity; Response Authority. | `F-006` |
| `WP2-OWNER-D002` | Uses one coordinate bundle for common conditioned-`x/r` grid identity while preserving separate numerical validity, cause, response, uncertainty, and publication states. | RTC-to-AST boundary: One Coordinate Grid. | `F-006` |
| `WP2-OWNER-D003` | Makes unrecoverable RTC-grid pointing an observation-level hard stop; only failure diagnostics and raw inputs remain, with no CAL/PTC/science/ML handoff. | RTC-to-AST boundary: Failure Semantics. | `F-006`, `F-007` |
| `WP2-OWNER-D004` | Keeps selected-point time distinct from filter delay; prohibits automatic scalar group-delay pointing shifts and requires any exact correction adapter to be separately authorized and applied once. | RTC-to-AST boundary: Delay And Coordinate-Correction Rule. | `F-006` |
| `WP2-OWNER-D005` | Separates SCI-BEAM measured-geometry authority, TolAPT/TolProj observation association, and AST coordinate realization. | Geometry boundary: Ownership Separation; Exact Producer And Runtime Binding. | `F-007` |
| `WP2-OWNER-D006` | Defines portable horizon-referenced `x_t,y_t` with no target-observation rotation applied; constrains registered `rot` quantities to their exact parent Beammap and requires target field rotation exactly once. | Geometry boundary: Portable And Parent-Beammap Roles; Field-Rotation Rule. | `F-007` |
| `WP2-OWNER-D007` | Keeps portable relative detector layout separate from Beammap-specific pointing/translation, with explicit basis, handedness, gauge, pivot, affine modes, and equivalence transforms. | Geometry boundary: Portable Geometry Versus Pointing; Gauge And Equivalence. | `F-007` |
| `WP2-OWNER-D008` | Allows numerical coordinates with typed unavailable geometry covariance while prohibiting total astrometric uncertainty/precision claims; missing numerical geometry still fails pointing. | Geometry boundary: Availability And Failure. | `F-007` |
| `WP2-OWNER-D009` | Leaves acquisition/exposure state on original occurrences, requires recoverable downstream support/response lineage, and forbids cadence/filter-width/replacement/PTC transformation from creating exposure. | Exposure boundary: Immutable Facts; Required Lineage; Stage Rules. | `F-019`, `TS-CLAR-002` |
| `WP2-OWNER-D010` | Limits WP-2 to physical-acquired and valid-original facts plus lineage; any use-qualified exposure receives a later distinct owner and definition. | Exposure boundary: No Generic Usable Exposure. | `F-019`, `TS-CLAR-002` |

## Artifact Inventory

| Artifact | Candidate role |
| --- | --- |
| `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md` | Exact RTC product/plan/grid/sample-to-AST coordinate parent and hard-failure contract. |
| `DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md` | Exact geometry producer, observation association, representation, rotation, gauge, uncertainty, and runtime-binding contract. |
| `TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY.md` | Immutable occurrence-level exposure and downstream lineage contract through PTC. |
| `SCIENTIFIC_OWNER_APPROVAL_2026-08-23.md` | Exact packet approval and status-only promotion authority. |
| `SOURCE_MANIFEST.md` | Exact approved byte identities and source-authority references. |

## Scope And Deferral

The candidate changes no frozen package equation or numerical operator. It
adds package-neutral composition authority. In particular it does not:

- reopen SCI-ALIGN, SCI-AST, SCI-RTC, SCI-BEAM, or SCI-PTC;
- freeze or supersede SCI-CAL;
- define generic usable, retained, projected, or coadded exposure;
- select a MAP projection, coefficient, support, response, coadd, or
  reprojection policy; or
- claim implementation conformity, validation, performance, or production
  readiness.

`F-006`, `F-007`, the timestream facet of `F-019`, and `TS-CLAR-002` remain
open until the approved, content-bound authority passes the authorized
clean-room re-audit.
