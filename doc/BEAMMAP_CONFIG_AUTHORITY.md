# Beammap Config Authority

This document characterizes the next bounded Phase 2 authority domain. It does
not change Beammap execution or numerical behavior.

## Frozen Surface

`tools/config/beammap_legacy_paths.json` freezes the 74 leaves under
`beammap.*` in `data/config.yaml`. The surface covers iteration and phase
policy, reference-detector handling, RFI and scan-band masking, detector
weighting and TOD selection, Gaussian-fit support, split output, soft priors,
quality flagging, and sensitivity-band policy.

The current typed request has no known leaf gaps. `BeammapConfig` owns the
parsed values and the execution paths generally read that typed object. This
is materially further along than a legacy-authoritative starting point.

## Remaining Boundary

Configuration enters through one `Engine::get_beammap_config` boundary. Its
family readers construct typed values and `apply_beammap_typed_config` installs
one request snapshot. One compatibility adapter,
`sync_beammap_map_fitter`, copies `beammap.fitting.fit_radius_fwhm` into the
shared numerical fitter. The fitter remains the owner of fit workspaces and
realized fit results; it must not become a source of requested policy.

The domain has no dedicated effective execution plan or versioned Beammap
config provenance. This absence is explicit, not silently treated as
completion. The static audit requires the current adapter and missing
provenance state until an implementation checkpoint deliberately changes both.

## Target Contract

The target remains one-way:

```text
merged YAML -> immutable Beammap request -> effective Beammap plan
            -> narrow numerical adapters -> realized iteration/output record
```

The effective plan should record normalization currently performed while
loading, including phase-strategy correction, prior enablement when a path is
missing, split-flag normalization, and mode-dependent iteration behavior.
Realized state should describe attempted/completed iterations, detector-fit
cardinality, required output cardinality, and completion without duplicating
the post-processing fit record.

## Stop Rule

Do not redesign Gaussian fitting, prior matching, flagging, or detector-map
algorithms in this domain. First replace the shared fitting-radius policy read
with a typed effective input and add provenance around the established
execution. Any algorithmic change requires separate scientific ownership and
validation evidence.
