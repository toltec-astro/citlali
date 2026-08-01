# SCI-ALIGN-001 coordinator and scientific-owner decision — 2026-08-01

Status: partial; `ALIGN-OD1` approved with compatibility guard; `OD2`--`OD8`
pending

Package: `SCI-ALIGN-001`

Governing audit report:
`SCI-ALIGN-001_SCIENTIFIC_CONTRACT_AUDIT.tex`, SHA-256
`6aaed0e6e16e4c37cd24d15b98346f84024ffd7920bd0524e7a170dbc728a393`

## ALIGN-OD1 — Common grid and clock model

Decision: approved with a compatibility-preserving validation guard.

- The detector/KIDs stream is the common-grid and reference-clock authority.
- Detector acquisition support is retained rather than trimmed to telescope or
  optional HWPR availability.
- Common slots use explicit detector cadence and phase, stable
  observation/slot identity, one shared round-half-up assignment operator, and
  a declared residual tolerance strictly below half a detector sample.
- The initial supported interface clock model permits only one
  observation-constant offset per interface. Clock drift requires a separately
  versioned model, evidence, and owner approval.
- Nonmonotonic or duplicate timestamps, slot collisions, incomparable epochs,
  and out-of-tolerance assignments fail closed.

### Compatibility condition

This decision is not authority for a wholesale retiming of historically
well-aligned TolTEC data. Existing Beammap source crossings, point-source
centroids, and recovered PSF widths show that present telescope/detector timing
is already close to the required physical solution. The repair must preserve
ordinary conforming behavior and change only paths that are ambiguous,
invalid, untraceable, or explicitly repaired.

Before the repair design is frozen, derive detector cadence, phase, and the
candidate slot-residual tolerance from the authoritative native timestamp and
header contract plus measured cadence/jitter. Return for owner review if this
would move ordinary valid samples to different slots or imply a material
timing shift.

The local and exact-repair-SHA validation must include:

- old/new native-row to common-slot identity comparison, with any changed row
  explained by a named defect or validity rule;
- old/new aligned telescope position and timestamp residuals over representative
  Pointing and Beammap observations;
- source-crossing timing and along-scan centroid comparison;
- fitted centroid and major/minor PSF-width comparison for all arrays; and
- demonstration that the successor produces no material degradation relative
  to established Beammap/Pointing repeatability or expected beam sizes.

Numerical source-crossing and PSF tolerances must be preregistered from the
existing empirical repeatability/fit uncertainty rather than selected after
viewing candidate results. Exact equality is not required where the current
path is one of the audited invalid-success cases.

## Pending decisions

- `ALIGN-OD2`: offset and header authority;
- `ALIGN-OD3`: telescope/HWPR field topology;
- `ALIGN-OD4`: gap bounds and action;
- `ALIGN-OD5`: scan policy and identity;
- `ALIGN-OD6`: synthesized eligibility;
- `ALIGN-OD7`: mapping/covariance/response; and
- `ALIGN-OD8`: HWPR separation and interim production.

No repair, Unity request, re-audit, or production change is authorized by this
partial record.
