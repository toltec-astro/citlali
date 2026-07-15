# Polarimetry Capability And Config Contract

This document records the bounded Phase 2 disposition of
`timestream.polarimetry`. It does not claim that enabled polarimetry is ready
for scientific use and does not authorize changes to the dormant numerical
algorithms.

## Project Direction

Citlali is intended to become the center of TolTEC polarimetry reductions in a
future development project. The present refactor has no approved polarimetry
or HWPR scientific contract and no enabled reference dataset. Enabled
polarimetry is therefore a planned but unavailable capability for this release.

The production config boundary mechanically rejects
`timestream.polarimetry.enabled: true`. This is a temporary capability gate,
not removal of the configuration vocabulary or implementation path. Removing
the gate requires both:

1. an approved polarimetry/HWPR contract; and
2. an enabled end-to-end reference validation gate.

## Frozen Surface

The complete current low-level surface is:

- `timestream.polarimetry.enabled`;
- `timestream.polarimetry.grouping`; and
- `timestream.polarimetry.ignore_hwpr`.

There is no separate `calibration.ignore_hwpr` YAML input. The historical
legacy string in `Engine::calib` is an execution target populated by the
typed-to-legacy adapter.

The config-exposure policy classifies this surface as expert, not a normal
compact user control, while enabled execution remains unavailable.

`tools/config/polarimetry_config_paths.json` freezes these three paths and
their digest. `tools/config/audit_polarimetry_boundary.py` enforces the frozen
reader, capability plan, one-way adapter, serialization, required provenance,
and absence of the retired compatibility reader and reverse mirror.

## Authority Flow

The production direction is:

`merged YAML -> typed request -> capability resolution -> RTCProc/Calib adapter`

The immutable request retains all three supplied values. The effective plan is
identical for accepted disabled reductions. If enablement is requested, the
plan records rejection, forces the non-executed effective state to disabled,
adds the exact config path to diagnostics, and prevents the reduction from
starting. No runtime object can re-enable the request or mirror state back into
the typed config.

For accepted disabled reductions, the adapter preserves the established
runtime state: `run_polarization=false`, grouping `fg` unless configured
otherwise, one Stokes-I label, and the requested legacy HWPR spelling. This
retains ordinary point, OOF, Beammap, and science behavior.

## Provenance

Every successful CLI reduction requires atomic
`polarimetry_provenance.yaml` with schema
`citlali-polarimetry-provenance-v1`. It records:

- capability status, reason, and exit condition;
- requested and effective values;
- whether the request was accepted or disabled by the capability gate; and
- completed reduction, polarimetry execution, and HWPR-loading state.

While the capability is unavailable, a successful sidecar must record a
disabled request, disabled effective state, no polarized execution, and no
HWPR load. The reduction auditor rejects any inconsistent record.

## Deferred Scientific Work

The existing meanings implied by `auto`, `true`, and `false` are not approved
as the future HWPR contract. Detector grouping, Stokes construction, missing
HWP behavior, calibration, units, angle conventions, output schemas, and
acceptance tolerances all remain future scientific decisions. Existing dormant
code and low-level spellings are retained only to preserve an implementation
starting point.

The present refactor must not broaden or optimize enabled polarimetry. It may
only preserve the capability gate, disabled behavior, and future-facing
boundary until the exit conditions above are met.
