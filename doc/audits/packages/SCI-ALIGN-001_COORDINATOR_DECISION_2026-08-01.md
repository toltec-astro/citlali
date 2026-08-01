# SCI-ALIGN-001 coordinator and scientific-owner decision — 2026-08-01

Status: partial; `ALIGN-OD1`--`ALIGN-OD3` approved; `OD4`--`OD8` pending

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

## ALIGN-OD2 — Offset and header authority

Decision: approved with an explicit default-zero compatibility policy.

- The detector clock is the reference interface. For interface `i`, the
  corrected coordinate is the checked epoch conversion plus
  `delta_(i->detector)`.
- Offsets are floating-point seconds. A positive offset is added to the native
  coordinate and therefore places it later on the detector-reference clock.
- An offset is applied exactly once before ordering checks, common-slot
  assignment, gap detection, scan construction, or interpolation. It is not
  rounded to an integer number of samples.
- Requested, effective, observation-resolved, and realized state records the
  value, source, sign, unit, reference interface, application stage,
  uncertainty or bound, and whether the correction was actually applied.
- An omitted authoring value may resolve at the typed configuration boundary
  to zero with source `schema_default_zero`. This preserves existing ordinary
  zero-offset reductions but does not represent the value as measured or
  header-derived.
- A nonzero offset requires authoritative comparable clock/epoch and native
  timestamp/header semantics. An ambiguous sign, reference, unit, epoch, or
  application stage fails closed.
- A requested HWPR offset must either be applied under this same contract or be
  rejected as unavailable; it may not be reported effective while ignored.

Before repair, inventory the exact detector, telescope, and HWPR timestamp and
header fields, their epochs, rollover behavior, units, and acquisition bounds.
Return for owner review if this evidence conflicts with the approved sign or
reference convention. Synthetic positive, negative, fractional, omitted, and
ambiguous-offset fixtures plus the OD1 Beammap/Pointing compatibility evidence
are mandatory.

## ALIGN-OD3 — Telescope and HWPR field topology

Decision: approved with mandatory owner review of the generated field registry
before repair design is frozen.

- Maintain a versioned per-variable registry naming scientific identity, unit,
  coordinate frame, topology, validity and missing-data rules, maximum
  interpolation span, and operator.
- Classify each field only as a continuous scalar, circular angle with an
  explicit period and wrap rule, declared step state with half-open hold
  support, or non-interpolable/exact-only.
- Never apply ordinary linear interpolation to timestamps, counters, flags,
  modes, Boolean values, or categorical state.
- Prohibit extrapolation and interpolation across invalid native rows or source
  gaps unless a separately approved bounded-gap rule explicitly classifies the
  resulting value as synthesized.
- Persist the actual per-variable units and semantics; telescope timestamps and
  state are not radians merely because they share the telescope-data container.

Before implementation, generate the complete detector/telescope/HWPR field
registry from authoritative NetCDF metadata and acquisition contracts. Return
that registry to the scientific owner for review rather than guessing any
period, frame, bounded-versus-circular topology, state transition convention,
or interpolation span. Ordinary valid samples away from wraps, state
transitions, invalid rows, and gaps should remain numerically unchanged.

## Pending decisions

- `ALIGN-OD4`: gap bounds and action;
- `ALIGN-OD5`: scan policy and identity;
- `ALIGN-OD6`: synthesized eligibility;
- `ALIGN-OD7`: mapping/covariance/response; and
- `ALIGN-OD8`: HWPR separation and interim production.

No repair, Unity request, re-audit, or production change is authorized by this
partial record.
