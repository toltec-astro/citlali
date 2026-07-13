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
- `observation` is reserved for observation-derived map count, effective pixel
  size, and required write count. Unavailable values are explicit.
- `realized` records successful reduction completion and whether mapmaking ran.
  Product cardinalities remain unavailable until the next bounded slice.

The stable sidecar schema is `citlali-mapmaking-provenance-v1`, written as
`mapmaking_provenance.yaml` at the reduction root. It is a required atomic
output; failure fails the reduction.

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

Do not call the domain complete until these gates pass and realized
observation/product cardinality is either populated or explicitly accepted as
unavailable.
