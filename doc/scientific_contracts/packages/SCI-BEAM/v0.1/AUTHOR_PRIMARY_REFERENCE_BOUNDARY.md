# SCI-BEAM v0.1 — Primary Reference Boundary

Status: owner-approved, sanitized author reference

Approval date: `2026-08-16`

This boundary admits only the exact instrument/context claims below. The
implementation-blind author must derive the SCI-BEAM mathematical contract from
first principles under the Scope Brief and may not import an estimator,
threshold, calibration convention, or production policy merely because another
instrument or current software uses it.

## PR-001 — TolTEC Optical Design Context

Source: S. Bryan et al., *Optical Design of the TolTEC Millimeter-wave Camera*,
2018, arXiv [`1807.00097`](https://arxiv.org/abs/1807.00097), version 1.

Admitted claims:

- TolTEC is a three-band millimeter-wave camera whose optical design couples
  the field of view to focal planes near 150, 220, and 280 GHz.
- Instrument optical design, detector properties, atmospheric emission, and
  passbands are relevant context for predicted resolution and sensitivity.

Not admitted:

- pre-commissioning predictions as realized per-detector beam truth;
- a numerical v0.1 beam-width acceptance range;
- a source model, likelihood, covariance, fit radius, QC threshold,
  calibration factor, or sensitivity convention; or
- any claim about current Citlali implementation or TolTEC production state.

## PR-002 — TolTEC Commissioning Beammap Scope

Source: J. Golec and the TolTEC Collaboration, *Early high-resolution
millimeter-wave maps from TolTEC*, 2024, DOI
[`10.1051/epjconf/202429300022`](https://doi.org/10.1051/epjconf/202429300022).

Admitted claims:

- Commissioning observations included bright-source scans used to characterize
  instrument beams and detector properties.
- Publicly described Citlali Beammap outputs include detector location,
  conversion from raw timestream units to flux density, and detector beam
  full width at half maximum in an array-properties product.

Not admitted:

- the correct detector beam/source estimator or parameterization;
- the correctness, precision, uncertainty, or production readiness of those
  outputs;
- the meaning or promotion of a calibration candidate;
- any numerical QC, convergence, prior, flag, or sensitivity policy; or
- current implementation, validation, or observational-performance evidence.

## Deliberate Reference Exclusion

No AzTEC, SPIRE, or other analogue-instrument methodology paper is admitted to
the author packet. Such literature can motivate later review, but v0.1 shall
not acquire another instrument's beam model, finite-source correction,
calibration scheme, or uncertainty policy by analogy.

The two primary sources provide context, not a reusable SCI-BEAM independent
core. The author is expected to derive the model, objective, response,
uncertainty, prior, convergence, QC, validity, and edge predictions without
opening current implementation or audit evidence. This file is content-bound
in [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).
