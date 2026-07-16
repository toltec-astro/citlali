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

`tools/baseline/validate_reduction.py` performs four gates:

1. completion and required-provenance audit;
2. exact merged low-level YAML comparison;
3. profile-pinned scientific product-contract validation; and
4. profile-pinned numerical product comparison.

It resolves the accepted baseline through `validation/accepted_runs.json` and
`validation/validation_profiles.json`. Use `--baseline` when the ledger's local
artifact path is unavailable on the current host.

## Verification

- The validation ledger contains 60 valid records.
- The profile registry validates with four active modes.
- All 106 baseline-tool unit tests pass, including the science-change ledger
  integrity checks.
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

The owner also plans a post-refactor re-reduction of approximately 50
historical Beammap observations. That corpus will span the expected operational
variation and is the preferred broad performance census. Before it starts, add
a lightweight manifest/analyzer that reuses the evidence extractor and retains
observation/workload, config, binary, host, runtime, RSS, I/O, stage, and
outcome identity per run. Analyze performance against workload and preserve
same-observation pairings where possible; unlike observations are not repeated
timing trials. The census will establish a future-release baseline but is not a
Phase 4 closeout prerequisite.

The scientific product census is now machine-readable in
`validation/product_contracts.json`. It classifies every FITS, NetCDF, ECSV,
and CSV product in the four accepted snapshots: point 21/21, OOF 31/31,
science 28/28, and Beammap 13/13. Explicit output switches are evaluated from
the generated low-level YAML, so requested products must be present and
disabled products must be absent. Core companion diagnostics without an
independent switch remain required; profile timing and bounded learning CSVs
remain optional. The contracts record scientific identity, frames, axes,
units policy, indexing, missing-value policy, and fatal required-write policy.
Known incomplete NetCDF units/fill metadata and ECSV column-unit metadata are
documented debt, not silently invented semantics. See
`doc/PHASE4_SCIENTIFIC_PRODUCT_CONTRACT_2026-07-16.md`.

The intended-science-change manifest required by the external review is now
`validation/intended_science_changes.json`. Its scope begins after the explicit
refactor baseline `376e0022`; earlier cleaning and fruit-loop history is
inherited baseline behavior. The three traced imports are the RTC/PTC
determinism fix (`991428e70` to `ee429eca0`), Wiener optimizations (three source
to integration mappings), and PCA active-detector optimization (`b83c87507` to
`be7974636`). The validator checks full Git objects, baseline/integration
ancestry, four exact patch identities, accepted-run IDs, repository documents,
and product-family IDs. Later OG `ffc6b907` remains a validation comparator and
is not falsely listed as an import. Future intentional non-structural changes
must add an entry and successor validation epoch when products or expected
numerics change.

The durable scientific-conventions requirement is now addressed by
`doc/SCIENTIFIC_CONVENTIONS.md`. It records supported reduction intents,
array/network/detector/map/Stokes identity, sample and detector ordering,
map/astrometry frames, units, validity and missing-data rules, configuration
lifecycle states, required-product failure policy, active numerical gates, and
validation routing. Provisional polarimetry, R-channel, UID-lifetime,
network-mapping, coordinate, fallback, and schema-metadata questions remain
explicit owner decisions rather than inferred conventions. The census also
corrected a documentation defect in `validation/product_contracts.json`:
RTC/PTC `output_scan_index` is one-based as written and as stated by its NetCDF
metadata; NetCDF dimension positions remain zero-based. This is a contract-text
repair with no output or algorithm change.

The canonical architecture-map requirement is now addressed by
`doc/ARCHITECTURE.md`. It traces the supported `citlali_cli` entry through
`ReductionSession`, fresh reduction inputs and mode ownership, runtime setup,
geometry, iteration/observation orchestration, required output publication,
and structured result reporting. It also records the one-way config state
flow, explicit lifecycle owners, dependency rules for new work, the frozen
`Engine` compatibility boundary, and the header-versus-compiled-code policy.
Active, transitional, unbuilt legacy/experimental, and deferred paths are
separately classified so an external reviewer or future agent does not mistake
old main programs, empty `.cpp` placeholders, compact config, enabled
polarimetry, R execution, or deferred build work for supported architecture.
No production code or compilation infrastructure changed in this tranche.

The adopted broader-refactor checklist is now explicitly reconciled in
`doc/PHASE4_CLOSEOUT_CENSUS_2026-07-16.md`. Ten of its 15 criteria are closed
by implementation/evidence, two by an owner-approved scope decision or
performance proportionality exception, and three remain in the deliberately
deferred compilation package. Five focused ADRs live under `doc/adr/`; root
`CODEX.md` is a concise redirect instead of retaining invalid historical build
instructions; and `doc/RETAINED_DEBT.md` records role owners, reopening
triggers, and exit conditions. No criterion remains open outside the
compilation package. Stop
compilation-independent Phase 4 work rather than resuming structural
decomposition. The deferred header/build/CI criteria must be reviewed together
after the TolTECA integration model is known.
