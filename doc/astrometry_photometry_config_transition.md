# Astrometry And Photometry Configuration Transition

## Scope

This domain contains two observation calibration inputs with distinct
scientific ownership:

- astrometry pointing offsets supplied as azimuth/elevation corrections in
  arcseconds, optionally sampled at two modified Julian dates; and
- Beammap per-array calibrator fluxes consumed by Citlali after TolProj selects
  the calibrator and estimates its flux.

Source identity is telescope-data state. Citlali does not infer calibrator
identity or flux.

## Current Authority

Beammap photometry and astrometry are both typed-authoritative. Each loader
constructs and validates one complete observation value before replacing any
Engine state. One-way adapters populate the numerical compatibility state:

- `AstrometryConfig` to `PointingOffsetState`; and
- `BeammapPhotometryConfig` to the mJy/beam runtime flux map.

Neither path merges with prior observation state. The astrometry loader no
longer exits the process or mirrors partially populated runtime vectors back
into typed configuration.

## Preserved Behavior

The astrometry input accepts named `az`/`alt` entries and the legacy positional
form. Each axis contains one constant value or two values for interpolation.
An omitted MJD pair or accepted non-positive sentinel is normalized to the
existing zero pair. A one-value offset remains constant; a two-value offset
with the zero pair is interpolated across the observation span. The numerical
interpolation and coordinate transforms are unchanged by this authority
migration.

## Remaining Gate

The domain remains partial until Citlali records every observation's immutable
request, effective interpolation policy, and realized installation/interpolation
state in required atomic provenance. Multi-observation OOF and Beammap gates
must prove that later observations cannot inherit earlier calibration state.

Scientific ownership must approve the explicit contract for positive MJD
endpoints, including ordering, full-observation coverage, and extrapolation.
Until then the existing implementation continues to require increasing
endpoints that bracket the observation and does not extrapolate.
