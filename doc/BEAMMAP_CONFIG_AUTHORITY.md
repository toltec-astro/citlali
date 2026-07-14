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

The domain now has a pure, non-wired `BeammapExecutionPlan` preparation
checkpoint. It preserves a requested snapshot and separately resolves the
current phase correction, missing-prior-path disablement, per-phase prior
inheritance, split-flag normalization, convergence availability, and
mapmaking-disabled iteration policy. Production does not yet construct or
consume this plan. The current typed object and fitting adapter therefore
remain the execution boundary.

There is still no versioned Beammap config provenance. This absence is
explicit, not silently treated as completion. The static audit requires the
current adapter, the unwired plan state, and missing provenance until a later
implementation checkpoint deliberately changes those claims.

## Preparation Checkpoint

The boundary audit mechanically expands the 59 reader roots and 59 serializer
roots over the fixed-length vector leaves. Both cover all 74 frozen paths with
no missing or extra roots. The serializer is a component serializer only; it
is not a published provenance schema.

Cold-boundary validation now rejects non-finite iteration tolerance, prior
SNR, flagging-vector, and sensitivity-band values. It also enforces the vector
cardinality already required by the existing readers and requires a nonempty
subdirectory for enabled detector-TOD output. These checks do not add new
scientific ranges or alter numerical algorithms.

## Target Contract

The target remains one-way:

```text
merged YAML -> immutable Beammap request -> effective Beammap plan
            -> narrow numerical adapters -> realized iteration/output record
```

The prepared effective plan records normalization currently performed while
loading, including phase-strategy correction, prior enablement when a path is
missing, split-flag normalization, and mode-dependent iteration behavior.
The next checkpoint must construct it from an immutable request before
switching one bounded consumer.
Realized state should describe attempted/completed iterations, detector-fit
cardinality, required output cardinality, and completion without duplicating
the post-processing fit record.

## Stop Rule

Do not redesign Gaussian fitting, prior matching, flagging, or detector-map
algorithms in this domain. First replace the shared fitting-radius policy read
with a typed effective input and add provenance around the established
execution. Any algorithmic change requires separate scientific ownership and
validation evidence.
