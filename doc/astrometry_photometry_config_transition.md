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

TolTECA owns selection of the pointing support supplied to Citlali. The approved
upstream policy is:

- with two bracketing pointing observations, interpolate the offsets across the
  supplied support points;
- with one pointing observation, apply its offsets constantly; and
- with no pointing observations, use the offsets explicitly entered in the
  reduction configuration.

Citlali currently receives the resulting values but not metadata that identifies
which of those three upstream cases produced them. It therefore records support
origin as `upstream-unspecified`; it must not infer a source from vector length.

## Preserved Behavior

The astrometry input accepts named `az`/`alt` entries and the legacy positional
form. Each axis contains one constant value or two values for interpolation.
An omitted MJD pair or accepted non-positive sentinel is normalized to the
existing zero pair. A one-value offset remains constant; a two-value offset
with the zero pair is interpolated across the observation span. The numerical
interpolation and coordinate transforms are unchanged by this authority
migration.

The project owner approved the OG Citlali positive-MJD contract: endpoints are
strictly increasing, must bracket the complete observation, and are not
extrapolated. Failures now propagate through typed exceptions instead of a
library-level process exit.

## Requested, Effective, And Realized Record

Citlali now records every observation's immutable request, effective application
mode, and realized installation/interpolation state. The required atomic
`astrometry_provenance.yaml` uses schema
`citlali-astrometry-provenance-v1` and includes:

- TolTECA calibration-selection and Citlali application authority;
- az/alt axes, arcsecond units, MJD support identity, and the preserved legacy
  algorithm identity;
- observation index and number;
- requested and effective offset values;
- constant, observation-span-linear, or explicit-MJD-linear resolution; and
- installation count, application count, and telescope sample count.

The reduction auditor can require this contract with
`--require-astrometry-provenance`. A static boundary audit prevents return of
reverse synchronization, process termination, missing lifecycle calls, or a
non-atomic optional write.

## Remaining Gate

The implementation is a local candidate. A point run must validate ordinary
single-observation behavior and the required sidecar. A multi-observation OOF
run must validate observation ordering, repeated execution, and stale-state
isolation. Beammap must then show that the adjacent photometry path remains
unchanged before the combined domain is marked complete.
