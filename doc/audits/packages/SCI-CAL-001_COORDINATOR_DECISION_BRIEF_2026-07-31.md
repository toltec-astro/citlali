# SCI-CAL-001 Coordinator Decision Brief

Date: 2026-07-31

Package: `SCI-CAL-001`

Governing application SHA: `9aae0e669384c5c0c0dda93debc194d6b8dac787`

Frozen independent-core commit: `1565836282d936b974aeb5b5a7d4554ce55b10bd`

Final audit commit: `27b0916e725696597c3ba84fb6a82bf6cf0ea356`

Final audit artifact SHA-256:
`957ed71d1432ad67fe582d6137fbe72c52e82a31f3199331a94ab7b39490d376`

## Owner disposition

All five decisions were resolved by the project owner on 2026-07-31. D001
through D004 adopt the recommended bounded scientific contract, including the
full cross-repository APT authority and identity chain. D005 selects option 2:
the assessed implementation is `fail_closed` pending an approved repair,
validation, and fresh re-audit.

The authoritative wording is the
[coordinator and scientific-owner decision](SCI-CAL-001_COORDINATOR_DECISION_2026-07-31.md).
The remaining sections preserve the pre-decision rationale and alternatives.

## Why a decision is required

The independent audit is complete, but the package is not scientifically or
operationally complete. It found four P0 implementation defects and six P1
contract, policy, evidence, or dependency gaps. The assessed implementation is
`nonconformant`, the contract is `proposed`, validation is `in_progress`, the
verdict is `amend`, and a fresh exact-repair-SHA re-audit will be required.

No repair, Unity request, re-audit, or production authorization has begun.
The five decisions below must be resolved before a bounded repair contract can
be frozen. Decisions D001--D004 define scientific meaning; D005 defines the
interim operational restriction.

## Decision summary

### CAL-D001 — Atmosphere and extinction convention

Owner: calibration/atmosphere scientist.

Decide:

- what `tau225` physically represents and which band-transmission polynomial
  is authoritative;
- the calibration reference pivot `X_ref` for detector coefficients;
- the airmass model and valid elevation/airmass domain;
- low-opacity behavior, including the current q0 discontinuity; and
- interpolation in opacity and time.

The contract then applies the atmospheric factor as
`exp[tau_band * (X(elevation) - X_ref)]`. `X_ref = 0` means a
top-of-atmosphere coefficient; `X_ref = 1` means a zenith-normalized
coefficient. A coefficient derived at another pivot must record or transform
that pivot without double correction.

Recommended bounded choice: select one explicit pivot based on the actual APT
coefficient provenance, retain a named airmass model only over a declared
domain, make the low-opacity model continuous, and fail closed outside valid
opacity/elevation/interpolation state. Do not infer the pivot from historical
output.

Restriction until decided: extinction-enabled use fails closed.

### CAL-D002 — APT coefficient semantics and detector identity

Owner: calibration/APT owner.

Decide the units, sign, gain-versus-reciprocal convention, normalization,
epoch/validity, and factor decomposition of `flxscale`, `responsivity`, `sens`,
and `fcf`. Also approve a detector UID join as the authority linking external
APT rows to raw TOD columns; row-position fallback is prohibited.

Recommended bounded choice: define one typed factor table in which every
quantity has exactly one meaning and unit, invert responsivity at most once at
a checked boundary, require exactly one valid UID match per selected detector,
and record the normalization population and source epoch. Preserve current
numbers only when they satisfy that declared convention.

Restriction until decided: absolute calibration and PTC weight interpretation
fail closed. Relative use still requires an observation-specific identity
proof.

### CAL-D003 — Unit, beam, template, and photometry policy

Owner: photometry/beam scientist.

Decide the supported matrix of:

- output units;
- per-detector versus effective beam identity;
- elliptical versus circular beam approximation;
- point-source, extended-source, and integrated-flux meaning;
- map-pixel transfer;
- Rayleigh--Jeans versus CMB, bandpass, and color convention;
- kernel/template normalization; and
- allowed downstream consumers.

Recommended bounded choice: make `mJy/beam` the only initially repaired
photometric unit, with a declared effective beam and point-source peak
normalization. Defer `MJy/sr`, `Jy/pixel`, temperature units, and integrated
photometry until each has explicit beam, bandpass/color, pixel/WCS, response,
and validation contracts. This keeps the first repair narrow without
prejudging later unit support.

Restriction until decided: non-default units and absolute or integrated
photometry fail closed.

### CAL-D004 — Uncertainty and covariance representation

Owner: calibration/uncertainty owner.

Decide how products distinguish and propagate:

- conditional sample noise;
- detector-relative calibration uncertainty;
- array/common absolute gain;
- opacity/extinction uncertainty;
- beam, bandpass, color, and template response uncertainty; and
- correlations across detector, time, array, observation, and downstream
  products.

Recommended bounded choice: retain conditional statistical variance or weight
as a separately labeled component and publish calibration systematics as
named nuisance parameters plus their covariance/correlation scope. Do not
materialize a dense sample covariance unless a consumer requires it; allow
downstream construction from the nuisance representation. Explicitly label
conditional weights as excluding calibration and response systematics.

Restriction until decided: full-covariance and statistical-significance claims
fail closed.

### CAL-D005 — Interim operational disposition

Owner: project owner/coordinator.

Choose one:

1. Accept the audit's proposed `restricted_use` policy: only
   extinction-disabled `mJy/beam` relative or diagnostic processing, and only
   after independently proving the exact raw-column/APT UID relation for the
   observation. Extinction-enabled science, non-default units, absolute or
   integrated photometry, full covariance/significance, Beammap APT promotion,
   and calibration-dependent feedback remain fail closed.
2. Narrow the package to `fail_closed` until repair and re-audit.
3. Replace the proposal with another explicit allowlist justified by an urgent
   operational assessment.

Coordinator recommendation: choose option 1 only if the detector/APT identity
proof can be made explicit and retained for each affected observation;
otherwise choose option 2. Do not treat historical accepted reductions as
proof of absolute calibration.

Current canonical state: `existing_use_only`. The audit's narrower
`restricted_use` state is recorded as proposed, not authorized, until this
decision is made.

## What follows after the decisions

Once D001--D005 are recorded, the coordinator may prepare—but not yet launch—a
bounded repair/re-audit handoff. The repair must start from a separately
selected application SHA, not from the audit or coordination branch. Local
gates precede any human-run exact-repair-SHA `SCI-CAL-001-UNITY-001` request;
a fresh re-auditor then assesses the repair, returned evidence, and all
handoff dispositions.
