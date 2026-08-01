# SCI-CAL-001 Coordinator and Scientific-Owner Decision

Date: 2026-07-31

Package: `SCI-CAL-001`

Decision authority: project owner acting as calibration, atmosphere,
photometry/beam, uncertainty, and operational owner

Governing application SHA: `9aae0e669384c5c0c0dda93debc194d6b8dac787`

Frozen independent-core commit: `1565836282d936b974aeb5b5a7d4554ce55b10bd`

Final audit commit: `27b0916e725696597c3ba84fb6a82bf6cf0ea356`

Final audit artifact SHA-256:
`957ed71d1432ad67fe582d6137fbe72c52e82a31f3199331a94ab7b39490d376`

## Decision scope

This record resolves `CAL-D001`--`CAL-D005` from the integrated independent
audit. It approves the successor scientific contract needed to design a
bounded repair. It does not approve the assessed implementation, choose a
repair-base SHA, authorize a repair branch, request Unity evidence, launch a
re-audit, or authorize production.

The independent core remains the mathematical basis except where the exact
decisions below specialize its allowed choices. The audit remains immutable
evidence of the assessed source and findings.

## CAL-D001 — Extinction identity and validity

Decision: approved.

- Beammap-derived `flxscale` is referenced to the top of the atmosphere;
  therefore its airmass pivot is `X_ref = 0`.
- The water-vapor-radiometer `tau225` is zenith optical depth.
- The applied sample correction is
  `exp[tau_band(t) * X(elevation(t))]`; a zenith-equivalent band opacity must
  not be applied without the sample airmass.
- Positive opacity must not enter a finite interval with an exactly unity
  correction. The low-opacity model must vary continuously between the
  zero-opacity anchor and the first nonzero atmospheric-model anchor.
- The existing airmass approximation may be retained initially only over a
  declared and tested elevation/airmass domain.
- A single WVR header value is observation-constant. If a time-resolved WVR
  series is supplied by an approved input contract, interpolation must be
  bracketed; silent extrapolation is prohibited.
- Negative or non-finite opacity, invalid or non-finite elevation/airmass,
  invalid transmission/log domain, missing interpolation support, or values
  outside an approved model domain fail closed.

The bounded repair design must state the exact low-opacity interpolation
operator and model-anchor provenance before implementation. Continuity,
monotonicity, the zero limit, both sides of every anchor, and representative
airmass grids are acceptance gates.

## CAL-D002 — APT factor semantics and identity chain

Decision: approved.

- `responsivity` is the relative detector-response quantity used for
  donor/target conversion in operations such as RTC despike replacement. It
  is distinct from absolute flux calibration.
- Beammap `sens` is calculated from detector PSDs over the configured
  sensitivity-frequency interval: each scan contributes the mean of
  `sqrt(PSD/2)` over the selected bins, the detector value is the median over
  scans, and Beammap multiplies the raw-unit result by `flxscale`. The stored
  calibrated unit is top-of-atmosphere `mJy/beam * sqrt(s)`; extinction and
  non-default unit transfer are not included.
- `flxscale`, `sens`, extinction, and target-unit transfer remain separately
  identified factors. The realized total signal multiplier is their explicit
  composition where applicable.
- `fcf` may remain only as a precisely labeled compatibility value with exact
  contents, units, and exclusions. It is not authoritative total calibration.

The APT authority and identity chain is:

1. Citlali Beammap produces the measured APT and Beammap calibration fields.
2. `toltec_beammap` owns downstream Beammap calibration analysis, diagnostics,
   and reviewed APT updates.
3. TolAPT treats measured and design APTs as immutable inputs, owns
   design-to-measured matching, and writes new provenance-bearing matched
   products.
4. TolProj owns library curation, cohort seed selection, invocation and use of
   matching, separate pointing-derived `flxscale` products, and binding the
   selected artifact to each observation.
5. Citlali consumes the exact selected artifact and proves the mapping from
   raw TOD column through observation-local detector identity and matched APT
   row to the distinct common/design UID.

Row position is never identity. The consumer must retain and validate local
detector identity, common/design UID, array, network, parent artifact digests,
matching and flux-calibration lineage, and the exact selected APT digest.
Missing, duplicate, conflicting, invalid, or unproven mappings fail closed.
Derived APTs are new immutable artifacts; they do not overwrite their parents.

## CAL-D003 — Initial unit, beam, and photometry policy

Decision: approved.

- The initial successor contract supports only top-of-atmosphere `mJy/beam`.
- Its initial scientific meaning is point-source peak normalization: an
  unresolved source of flux `S` mJy ideally recovers peak `S mJy/beam` under
  the declared response and valid support.
- The per-detector calibration beam/template identity inherited from the
  originating Beammap APT is preserved separately from the realized map or
  filtered-product response.
- Any kernel that retains `mJy/beam` must have an explicit normalization that
  preserves the declared point-source response.
- Elliptical beam parameters are retained when available. A circularized
  Gaussian is a labeled approximation, not an identity replacement.
- `MJy/sr`, `Jy/pixel`, temperature units, extended-source calibration, and
  integrated photometry remain fail closed pending their own beam-area,
  bandpass/color, WCS-pixel, response, uncertainty, and validation contracts.
- Any future temperature unit must distinguish Rayleigh--Jeans from
  thermodynamic CMB temperature; ambiguous `uK` is not an approved unit.

## CAL-D004 — Uncertainty and covariance representation

Decision: approved.

- Conditional statistical variance or inverse variance is stored and labeled
  separately from calibration and response systematics.
- Multiplicative calibration propagates conditional variance as `a^2 v` and
  conditional inverse variance as `w / a^2` on valid support.
- Calibration systematics are represented by named nuisance parameters with
  value, uncertainty, provenance, validity, and detector/array/observation/
  cohort/global correlation scope. A compact nuisance covariance is retained
  where terms correlate.
- Required terms include detector `flxscale`, common absolute calibrator
  scale, TolProj pointing-derived fluxscale correction, WVR/atmospheric model,
  Beammap `sens` estimation where it affects approximate weights, and
  beam/template response. Future unit conversions add their own bandpass,
  color, and pixel/WCS terms.
- Downstream consumers construct full covariance from the nuisance model when
  needed; a dense sample covariance is not required as the default persisted
  representation.
- Nonlinear propagation or samples are required when linearized extinction
  propagation is inadequate.
- Missing uncertainty is recorded as unavailable, never as zero. Total-
  uncertainty and statistical-significance claims fail closed when only
  conditional weight is available.

## CAL-D005 — Interim operational disposition

Decision: option 2, `fail_closed`.

No new CAL-dependent scientific processing is authorized from the assessed
implementation. Historical products remain available as historical and
regression evidence; they are not promoted as newly authorized calibrated
science and do not establish absolute calibration.

This fail-closed state remains until an owner-approved bounded repair passes
its local gates, exact-repair-SHA external evidence is returned and accepted,
and a fresh re-audit authorizes an explicit successor production disposition.

## Required next coordination step

The next permissible step is to prepare a bounded repair/re-audit handoff from
these decisions and select its exact application base. Repair must occur in a
fresh repair worktree, never on the audit or coordination branch. No repair or
external evidence request begins solely because this decision record exists.
