# SCI-PTC-001 optional transfer-characterization plan — 2026-08-09

Record ID: `SCI-PTC-001-OPTIONAL-TRANSFER-CHARACTERIZATION-2026-08-09`

Status: optional plan prepared; not authorized, frozen, requested, or launched

Authority: `SCI-PTC-001-D003-OWNER-AMENDMENT-2026-08-09`

## Purpose and boundary

This plan defines a possible measured extension of the existing PTC kernel,
which is the collaboration's estimated map-center point-source response of the
instrument after the declared RTC/PTC/analysis chain. The optional work does
not replace, deny, or relabel that current estimate. It is separate from
ordinary PTC repair and operation. No ordinary product must run this
characterization, and no implementation, reduction, Unity, MAP, BEAM,
external-contact, or evidence action follows from this plan.

Before any future launch, the coordinator must select an exact application
SHA and evidence producer, apply the scientific-audit cost-control and scope
checkpoints, freeze the complete case register and stop conditions, and obtain
separate owner authorization.

## Conditioning identity

Each characterization cell must bind, without cross-cell substitution:

- exact TolTEC band;
- reduction and map mode;
- RTC and PTC requested, effective, and realized configuration;
- source, exclusion, validity, and analysis masks;
- detector population and selection state;
- iteration and pass identity;
- input, calibration, and applicable upstream response identity and status;
- application/evidence-producer SHA, configuration and input digests, and
  deterministic realization identity; and
- output response-family identity, parentage, validity domain, and completion
  state.

Longer- and shorter-wavelength bands and distinct map/reduction modes are
separate cells. The study must publish per-band/per-mode results and may not
collapse them into one universal scalar curve.

## Characterization grid

Within each admitted cell, injection/recovery should sample:

- map position sufficiently to characterize off-center or spatially varying
  response within the declared domain;
- spatial scale and, where the processing operator makes it relevant,
  temporal frequency;
- amplitude over a range sufficient to detect nonlinearity and saturation;
- point-source and extended-source morphology; and
- enough independent realizations to report uncertainty or realization
  scatter rather than only a point estimate.

The preregistration must define the sampling grid, estimand, recovery
operator, uncertainty/scatter summary, completeness rule, numerical
tolerances, and early-stop/failure conditions before execution. Unmeasured
regions remain outside the validity domain; interpolation or model extension
must be separately declared and validated.

## Minimum optional product

For each band/mode cell, publish:

- the measured response versus sampled map-position and spatial-scale
  coordinates and, where applicable, temporal-frequency coordinates;
- amplitude dependence or an admitted linearity domain;
- morphology class and response;
- uncertainty or realization scatter and realization count;
- exact conditioning/provenance identity;
- explicit validity-domain bounds and invalid/unavailable regions; and
- typed response status consistent with the D003 amendment.

The product must retain the ordinary stored kernel as the estimated map-center
point-source anchor with its exact parent/configuration/state provenance,
domain, and calibration/validation/uncertainty status. Broader cells extend
that estimate; they must not erase it or imply cross-band or cross-mode
equality.

## Admission and stop rules

A future execution requires a separately approved, digest-bound protocol under
`doc/audits/AUDIT_MANAGER_INSTRUCTIONS.md` and
`doc/audits/NUMERICAL_PROPORTIONALITY_AND_COST_CONTROL_POLICY.md`. It must stop
for owner/coordinator review on an identity mismatch, missing conditioning
fact, incomplete cell, preregistered validity failure, or scope/cost expansion.

This plan supplies no response measurement, F005 closure, repair authority,
production claim, or downstream launch.
