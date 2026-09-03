# SCI-POINT v0.1 Author Boundaries And Unavailable States

Identity: `SCI-POINT_BOUNDARIES_UNAVAILABLE v0.1/r0.3`

Status: sanitized author input candidate

## Required Boundary Distinctions

- Targeted Pointing inference is not blind detection or catalog fitting.
- Per-array POINT measurements are not a cross-array aggregate or telescope
  correction.
- Measured displacement is not correction sign/composition/application.
- MAP, JINC, FLT-FIXED, and FLT-MATCHED are distinct routes.
- FRUIT ancestry is not a separate map type.
- Parent WCS/response/support/covariance is not POINT-owned.
- Marginal formal errors are not joint covariance, astrometric uncertainty,
  empirical repeatability, calibration uncertainty, or significance.
- Effective fitted width is a useful QC metric but not automatically an
  intrinsic beam or SCI-BEAM product.
- A QC metric is not an automatic threshold/action or unique causal diagnosis.
- Named-use eligibility is not one universal good/bad flag.
- Per-array scientific atomicity is not file/table atomicity.
- Producer lifecycle is not component identifiability or named-use
  evaluation; diagnostic display is a prescribed action after the four
  request/applicability/eligibility/realization axes.

## Typed Unavailable States

| Identity | Unavailable quantity or claim | Release condition or disposition |
| --- | --- | --- |
| `SCI-POINT-UNAV-001` | POINT-owned cross-array aggregate | outside base v0.1; future successor decision required |
| `SCI-POINT-UNAV-002` | POINT-owned correction candidate or applied correction | outside base v0.1; pointing-support producer and AST retain ownership |
| `SCI-POINT-UNAV-003` | numerical MAP/JINC/FLT POINT route | exact predecessor authority, numerical product, required state, and POINT compatibility binding |
| `SCI-POINT-UNAV-004` | numerical compatibility fit and every fit-derived product | separately content-bound and owner-approved `POINT-COMPATIBILITY-METHOD`, exact numerical parent, and route compatibility binding |
| `SCI-POINT-UNAV-005` | POINT whole-observation success from partial arrays | outside base v0.1; downstream producer owns any partial-set admission |
| `SCI-POINT-UNAV-006` | full joint covariance or uncertainty calibration/coverage | separately authorized representation and evidence; fit may remain valid without it |
| `SCI-POINT-UNAV-007` | absolute flux or calibration accuracy | exact CAL/TolProj authorization and evidence; never POINT alone |
| `SCI-POINT-UNAV-008` | intrinsic beam inference from Pointing width | separate authorized method; SCI-BEAM boundary preserved |
| `SCI-POINT-UNAV-009` | statistical significance or detection probability | complete probabilistic method and validation; neither legacy nor formal ratio is enough |
| `SCI-POINT-UNAV-010` | applied correction for another observation | exact producer selection and AST application authority |
| `SCI-POINT-UNAV-011` | OOF or blank-field source result | future separate package authority |
| `SCI-POINT-UNAV-012` | coadd or intermediate-FRUIT POINT parent | outside base v0.1 |
| `SCI-POINT-UNAV-013` | universal cross-use eligibility or POINT aggregate profile | prohibited; named-use owners define separate policies and VAL only evaluates |
| `SCI-POINT-UNAV-014` | source-attributed displacement or amplitude | exact source-reference authority and established source association |
| `SCI-POINT-UNAV-015` | published AltAz displacement | exact AST/source-to-POINT tangent-basis boundary and valid transform |
| `SCI-POINT-UNAV-016` | marginal formal errors, formal standardization, and uncertainty-required uses | separately content-bound and owner-approved `POINT-FORMAL-ERROR-METHOD`; numerical fit may remain available |
| `SCI-POINT-UNAV-017` | fitted-amplitude/full-map-RMS diagnostic | exact RMS population, centering, weighting, support, finite, and zero-denominator rules |
| `SCI-POINT-UNAV-018` | full-procedure response or observational pointing bias/accuracy | separately authorized response or empirical authority; fixed-branch Jacobian is not a substitute |
| `SCI-POINT-UNAV-019` | full-map-RMS and `fitted_amplitude_over_full_map_rms`/`sig2noise` | separately content-bound and owner-approved `POINT-FULL-MAP-RMS-METHOD`; absence blocks only this diagnostic |
| `SCI-POINT-UNAV-020` | known/isolated/bright/approximately-centered applicability fact | exact fact authority/test and cause; no observing label or implementation default may substitute |
| `SCI-POINT-UNAV-021` | exact MAP/JINC/FLT numerical boundary instance | exact package/version/source digest, signal role, owner approval, compatibility/supersession, and complete boundary state |

Unavailable means the named quantity or claim is not established. It does not
require discarding an honest per-array fit or prohibit a later versioned
companion or successor product.
