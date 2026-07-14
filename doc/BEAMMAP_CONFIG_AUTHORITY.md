# Beammap Config Authority

This document records the bounded Phase 2 Beammap authority migration. It does
not redesign Beammap execution or numerical behavior.

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
family readers now assemble one raw typed request without applying effective
policy. `BeammapExecutionPlan` preserves that immutable request and separately
resolves phase correction, missing-prior-path disablement, per-phase prior
inheritance, split-flag normalization, convergence availability, and
mapmaking-disabled iteration policy.

Production constructs the plan at that boundary. Mature Beammap consumers
temporarily receive a one-way copy of the effective snapshot through
`ReductionConfig::beammap`; this is a compatibility projection, not a second
authority. `sync_beammap_map_fitter` reads the plan's effective fitting policy
directly and copies only `fit_radius_fwhm` into the shared numerical fitter.
The fitter remains the owner of fit workspaces and realized fit results; it
must not become a source of requested policy.

There is still no versioned Beammap config provenance. This absence is
explicit, not silently treated as completion. The static audit requires plan
construction, ordered one-way compatibility installation, the current fitter
adapter, and missing provenance until a later implementation checkpoint
deliberately changes those claims.

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

The effective plan records normalization formerly performed while loading,
including phase-strategy correction, prior enablement when a path is missing,
split-flag normalization, and mode-dependent iteration behavior. Reader-side
mutation helpers are retired and the audit rejects their reintroduction.
Realized state should describe attempted/completed iterations, detector-fit
cardinality, required output cardinality, and completion without duplicating
the post-processing fit record.

## Stop Rule

Do not redesign Gaussian fitting, prior matching, flagging, or detector-map
algorithms in this domain. Next add realized lifecycle and provenance around
the established execution, then replace compatibility consumers only where an
explicit effective input clarifies ownership. Any algorithmic change requires
separate scientific ownership and validation evidence.
