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

The plan now also owns the cold observation and internal-iteration lifecycle.
It records observation identity and detector/map/scan counts; iteration phase,
active-map count, mapmaking-pass count, source-aware RTC rerun, fit completion,
and convergence; and terminal reason. Final completion cross-checks
observation identity and map count against mapmaking provenance and requires
exact agreement between completed Beammap iterations and post-processing fit
contexts. It does not copy map writes or fit attempt/valid counts from those
authoritative domains.

Successful Beammap reductions publish required atomic
`beammap_provenance.yaml` using schema
`citlali-beammap-provenance-v1`. Publication is allowed only after lifecycle,
mapmaking, post-processing, and observation-output completion; publication
failure propagates to the CLI. The static audit requires the complete ordered
lifecycle and exactly one completion/write path.

## Preparation Checkpoint

The boundary audit mechanically expands the 59 reader roots and 59 config
serializer roots over the fixed-length vector leaves. Both cover all 74 frozen
paths with no missing or extra roots. That 74-leaf component serializer now
feeds the versioned provenance envelope together with effective-resolution and
realized-state serializers.

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
Realized state describes attempted/completed iterations and completion without
duplicating mapmaking or post-processing authority. Observation output
completion is recorded only after the existing Beammap output calls return.
More detailed Beammap-specific optional-product cardinality remains a bounded
follow-up rather than a reason to copy established map and fit aggregates.

## Stop Rule

Do not redesign Gaussian fitting, prior matching, flagging, or detector-map
algorithms in this domain. Next validate this lifecycle/provenance checkpoint
with a matched Unity Beammap run, then add only the observation-resolved prior,
reference, and Beammap-specific product facts needed to close the documented
domain gates. Replace compatibility consumers only where an explicit effective
input clarifies ownership. Any algorithmic change requires separate scientific
ownership and validation evidence.
