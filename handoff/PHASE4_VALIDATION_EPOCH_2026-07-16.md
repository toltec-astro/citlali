# Phase 4 Validation Epoch Handoff - 2026-07-16

## Decision

Compilation-side Phase 4 work remains deferred until the TolTECA developer's
new C++ build and integration approach can be reviewed. This tranche changes no
CMake, dependency, install, CI-build, or cluster-build infrastructure.

Phase 4 proceeds through immutable validation snapshots. Future Citlali
development may intentionally and non-incrementally change algorithms,
defaults, schemas, and final products. Such a change creates a successor
validation epoch with a predecessor comparison and explicit scientific
rationale. It does not rewrite an accepted record or loosen an old profile.

## Active Epoch

`phase4-structural-closeout-2026-07-16` contains one active profile per
supported validation family:

- Point: `phase4-point-152389-v1`, baseline `redu66` at `2a974e0dd`.
- OOF: `phase4-oof-152385-152387-v1`, baseline `redu02` at `9ea6d7f01`.
- Science: `phase4-science-152390-152392-v1`, baseline `redu31` at
  `a7a35a00f`.
- Beammap: `phase4-beammap-148670-v1`, baseline `redu06` at `6dd0057f8`.

Point, OOF, and Beammap use zero-tolerance complete-product comparisons with
timestreams included and only the volatile profiling sidecar excluded. Science
uses `science_refactor_snapshot_v1.json`: product sets and integer diagnostics
are exact, and floating-point drift has the strict pre-existing `1e-8` map
bound. The historical 1.5% OG/refactor filtered-map allowance is not used for
successor comparisons against the accepted refactor snapshot.

## Entry Point

`tools/baseline/validate_reduction.py` performs three gates:

1. completion and required-provenance audit;
2. exact merged low-level YAML comparison;
3. profile-pinned product comparison.

It resolves the accepted baseline through `validation/accepted_runs.json` and
`validation/validation_profiles.json`. Use `--baseline` when the ledger's local
artifact path is unavailable on the current host.

## Verification

- The validation ledger contains 60 valid records.
- The profile registry validates with four active modes.
- All 74 baseline-tool unit tests pass.
- All four profiles pass an end-to-end self-check against their accepted
  snapshots, including complete Beammap products.
- The unified command also accepts point `redu66` against predecessor `redu65`
  and science `redu31` against predecessor `redu23` under the active strict
  profiles.
- The full config preflight passes: 96 tests, eight compatibility fixtures,
  100% compact-surface coverage, and all authority/boundary audits.

## Next Work

Continue compilation-independent Phase 4 work with controlled timing and
memory evidence plus explicit scientific-contract documentation. Do not begin
build-system modernization until the external TolTECA build direction is
available. A future candidate is validated by selecting its mode profile and
passing the downloaded `reduNN` directory to the unified entry point.

The next tranche added the controlled Beammap performance protocol and tooling
described in `doc/PHASE4_PERFORMANCE_PROTOCOL_2026-07-16.md`. Real same-node
warmup and alternating paired Unity runs are now a triggered diagnostic rather
than a mandatory closeout campaign. The template leaves peak-RSS policy unset
while still requiring and reporting the measurement; the analyzer marks an
otherwise complete campaign `pending_policy`. Build infrastructure remains
untouched and deferred. All 86 baseline-tool tests and full config preflight
pass; accepted Beammap `redu06` exercises the portable evidence extractor.

The live wrapper subsequently passed a Unity point pilot in `redu67` at
`7ca0be50c`. The retained and attached records are identical and complete;
external wall time was 131.08 seconds, Citlali time was 110.477 seconds, and
peak RSS was 908,316 KB. The active point profile accepted `redu67` against
immutable baseline `redu66` with no log, config, or product differences. This
qualifies wrapper integration.

The project owner then accepted the existing Beammap timing history as
proportionate operational evidence. Twelve accepted refactor checkpoints range
from 3,397.522 to 4,215.296 seconds, have a 3,594.693-second median, move in both
directions, and show no sustained regression. The latest adjacent change is
1.9%; a prior 13.0% total increase occurred while mapmaking was 1.3% faster and
the increase was concentrated in VAST-sensitive PTC and diagnostics I/O.
Because unrelated users dominate shared VAST traffic, serializing only Citlali
jobs would not create a controlled storage environment. No dedicated Beammap
campaign is required absent a sustained regression, unexplained stage
slowdown, memory failure, peak RSS near node capacity, or material hot-path
change. Use the wrapper for the next naturally required Beammap validation to
collect RSS and provenance without spending an extra reduction.
