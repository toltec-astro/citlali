# Mapmaking Config Authority

This document fixes the finite Phase 2 contract for the `mapmaking.*` domain.

## Boundary

The frozen surface contains 22 paths in
`tools/config/mapmaking_legacy_paths.json`. Requested YAML is read once into
`citlali::config::MapmakingConfig`. Core activation and enum values, output
geometry, JINC settings, and maximum-likelihood settings all use typed readers.

`MapBuffer`, `JincMapmaker`, and the maximum-likelihood mapmaker do not read
YAML. They are compatibility execution targets populated by one-way adapters.
No adapter writes back into typed request state.

## State Layers

- `requested` preserves the merged TolTECA request, including `grouping: auto`.
- `effective` resolves reduction-dependent grouping without mutating the
  request. Automatic Beammap grouping becomes detector; other automatic
  grouping becomes array; unsupported explicit detector grouping falls back
  to array with the existing warning. When flux calibration is disabled, the
  effective map unit resolves to the TOD type while the requested `cunit`
  remains unchanged.
- `observations` contains one identified record per input in the final fruit-
  loop iteration. Each record carries its input index, obsnum, map count,
  effective pixel size, required logical map-write count, and successful output
  completion state.
- `coadd` is present only when coadd map output ran. It records the coadd map
  count, required logical map-write count, and successful output completion.
- `realized` records successful reduction completion, whether mapmaking ran,
  and completed observation/coadd cardinality. Cardinality is reset at every
  fruit-loop iteration, so the reduction-root sidecar describes the final
  iteration rather than an accumulation across intermediate products.

The current sidecar schema is `citlali-mapmaking-provenance-v2`, written as
`mapmaking_provenance.yaml` at the reduction root. Version 2 replaces the
single unavailable observation placeholder with the identified sequence and
coadd record above. Validation tooling remains backward-compatible with
historical version-1 sidecars. The sidecar is a required atomic output;
failure fails the reduction.

`required_map_write_count` counts logical map products required by active
output stages: one per calculated map for raw output, plus one per map when a
filtered output stage is active. It is intentionally independent of physical
FITS container packing and Beammap flag splitting. A record is marked complete
only after every required stage returns successfully.

## Validation

The local gate is:

```text
cmake --build build --target citlali_cli citlali_test citlali_safety_test -j 8
ctest --test-dir build --output-on-failure
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all
```

Real-run acceptance requires:

1. Point: strict complete-product comparison, exact merged config, valid
   sidecar, and zero unexpected errors. This primarily gates WCS construction.
2. Beammap: the same gate plus all detector products, exercising JINC and
   automatic detector grouping.
3. Science: the accepted scientific-equivalence profile, exercising JINC and
   automatic array grouping.

Do not call the domain complete until these gates pass with valid version-2
observation and product cardinality.
