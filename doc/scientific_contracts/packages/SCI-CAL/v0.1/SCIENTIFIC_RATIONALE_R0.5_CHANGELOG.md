# SCI-CAL v0.1 Science-Team Rationale r0.5 Change Log

Date: `2026-08-20`

Revision r0.5 incorporates the scientific owner's dispositions of Q01--Q09
without changing contract version v0.1 or the stable 11-assumption,
50-requirement, and 30-edge inventories.

## Scientific changes

- Defines ordinary `xs` as dimensionless fitted KID `delta_f/f_res`, positive
  with absorbed optical power, with no CAL additive baseline operation.
- Fixes CAL before PTC; PTC owns DC and correlated common-mode removal.
- Makes frozen SCI-BEAM the `flxscale` producer authority and records the
  closest-accepted-Beammap default plus optional scientist-directed TolProj
  per-array photometric rescale.
- Records 272/214/150 GHz array reference frequencies, source-dependent
  spectra, and downstream ownership of target color correction.
- Binds the exact WVR/AM/passband atmosphere operator, support, interpolation,
  reference-spectrum surfaces, commit, and content digests.
- Replaces segment splitting and engineering no-output with one CAL-owned
  observation class, a 0.025 momentary-excursion tolerance, and the same
  operator for supported engineering samples without a science-quality claim.
- Places measurement-noise estimation downstream and records current
  systematic-uncertainty mechanisms as unavailable rather than zero.
- Replaces the abstract Q09 campaign with the actual Beammap/APT/library/
  TolProj closure and associated-pointing transfer workflow. Numerical targets
  are reporting benchmarks; final acceptance remains owner-owned.

## Preserved boundaries

No implementation is mapped or certified. No validation result or production
status is asserted. Source, child, delivery, CAL, PTC, and downstream-response
ownership remain distinct.
