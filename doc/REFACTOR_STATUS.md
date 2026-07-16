# Citlali Refactor Status

This is the living roadmap and completion ledger for the Citlali refactor.
Update it when a phase gate, governing decision, or validated snapshot changes.

## Governing Decision

On 2026-07-10 the project formally adopted the five-phase roadmap from the
[independent architecture review](../handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md).
The review verdict, **sound with material reservations**, is accepted.

The original [structural refactor plan](STRUCTURAL_REFACTOR_PLAN_2026-06-29.md)
remains the historical statement of intent. This document governs current
sequencing and exit criteria where the original plan differs.

The project will improve the existing tree incrementally. It will not restart
as a broad rewrite and will not rewrite the granular history of the validated
branch. The exact validated tree will remain available for forensic review.

## Current Snapshot

- Refactor baseline: `376e0022`.
- Production code inspected by the external review: `84670829`.
- Latest accepted point reduction: Phase 3 exit checkpoint `redu66`, produced
  by `2a974e0dd`, is exact against full-Wiener checkpoint `redu65` across all 19
  non-profile scientific products, including complete RTC/PTC timestreams,
  with zero changed or skipped records. Their 490-leaf configs are exact. Both
  profiles contain the same multiset of 78 stage/context records; only elapsed
  values and concurrent completion order differ. The run has 12 complete PTC
  chunks, zero logged issues, all required provenance valid, and a successful
  VAST-backed exclusive output-root acquisition. `redu64` remains the accepted
  mature library-exit checkpoint and `redu63` the first compiled boundary.
  Observation-resolved astrometry
  provenance `redu61`, disabled polarimetry capability provenance
  `redu60`, external KIDs/config-source provenance
  `redu59`, post-processing authority cleanup `redu58`,
  realized provenance `redu57`, typed source-fitting `redu56`, source-finding `redu55`, map-filter `redu54`,
  enabled-filtering `redu53`,
  unfiltered `redu51`, and bounded full-noise-output `redu49` remain the
  immediate post-processing, pointing, and noise-products control fixtures.
- Phase 3 full-Wiener point `redu65`, produced by `6dd0057f8`, is accepted
  against matched OG `redu10` at `ffc6b907`. Both use five noise realizations,
  a Gaussian template, and `lowpass_only: false`, and both execute all six
  expected Wiener core calls. The seven filtered products pass the strict
  scientific-tolerance gate with 148 compared records and no skips; the
  three-array pointing-fit table is exact. The refactor run has zero issues.
- Latest accepted OOF reduction: refactor `redu02` for observations
  152385-152387, produced by `9ea6d7f01`, is exact against accepted refactor
  `redu01`. The established OG `redu00` versus refactor relationship is
  unchanged. All 30 comparable products are
  present with no skipped records; pointing-table data and all per-observation
  ECSV/FITS dates are exact, and all scientific numeric differences pass the
  standard `2e-8 + 1e-10 * abs(reference)` tolerance. The only accepted
  differences are inactive RTC-despike config metadata recorded differently
  by the legacy and typed paths.
- Latest accepted science reduction: clean single-job four-iteration sequence
  `redu28` through final `redu31`, produced by `a7a35a00`. Its 502-leaf config
  is exact against accepted `redu23`; all 12 FITS and 15 NetCDF product sets are
  complete. All 84 map layers pass the science-equivalence gate with maximum
  relative RMS `7.09e-14`, all integer diagnostics are exact, and all 1,394
  NetCDF variables pass. The final run has zero logged issues and every required
  provenance record is valid. This supersedes `redu23` as the science fixture.
- Latest accepted Beammap reduction: Phase 3 checkpoint `redu06`, produced by
  `6dd0057f8`, is exact against accepted `redu05` across all 12 comparable
  products and 16,453 comparison records, including complete detector TOD,
  diagnostic NetCDF, detector-fit tables, and six split-map FITS products. Its
  529-leaf config is exact, all required provenance is valid, and the log has
  zero issues. It accepts the fruit-loop input/feedback and mature Wiener
  failure-contract tranches for Beammap.
- `redu23` and `redu24` completed all 12 PTC chunks with zero error-level log
  records and complete TOD/diagnostic products. Their common numeric products,
  FITS maps, and pointing tables are exact; only profiling timing differs.
- `redu21` and `redu22` had exact common numeric products with complete TOD
  comparison, but both contained 12 logged NetCDF errors.
- The same YAML exposed two provenance defects in `redu22`: an effective IIR
  default appeared for a disabled filter and an extinction sentinel changed.
  `redu25` validates the intended disabled-state provenance correction with
  exact scientific products.
- Local `citlali_cli`/test builds and full config preflight pass.
- CTest discovers and passes all 460 tests. All 96 config-boundary/preflight
  tests pass; the checked leaf contract covers 574 leaves and the generated
  startup schema covers 726 normalized YAML nodes.

These facts are characterization evidence, not a production-equivalence claim.

## Active Phase

**Phase 4 - Validation, performance, and reproducible build** is active as of
2026-07-16. Phase 3 library/session work is complete: local gates pass and
Unity point `redu66` accepts the output-root ownership repair and exact
scientific behavior at the first compiled boundary. Phase 2 config authority
and provenance remains complete at Unity point `redu62`.

Compilation-side Phase 4 work is explicitly deferred as of 2026-07-16 pending
review of the TolTECA developer's revised C++ build and integration approach.
Do not change Citlali CMake structure, presets, dependency management, CI build
lanes, install/export rules, or cluster build helpers until that direction is
understood. This is a sequencing decision, not acceptance of the current build
as the final reproducible-build solution. Phase 4 continues meanwhile through
strict validation, current baseline/ledger work, controlled performance
evidence, and scientific-contract documentation that is independent of the
eventual compilation strategy.

The first compilation-independent Phase 4 tranche establishes a versioned
validation epoch. Four named profiles pin the current point, OOF, science, and
Beammap snapshots to their required provenance, exact low-level configuration,
and mode-appropriate product comparator. Point `redu66` is the zero-tolerance
structural-closeout snapshot; clean science `redu31` is the current
scientific-tolerance snapshot. One profile-driven command now performs the run
audit, config comparison, and product comparison without duplicating the
existing scientific comparator logic. Accepted snapshots are immutable.
Future intentional algorithm, default, schema, or product changes create a
successor epoch with a predecessor comparison and explicit scientific
rationale instead of silently replacing a baseline or loosening its policy.

The controlled-performance evidence path is now specified without touching
deferred compilation infrastructure. A Unity-side wrapper records GNU Time
wall/RSS/I/O data together with Citlali log time, exact config leaves, bounded
input hashes, runtime policy, binary identity, serious log counts, and
profile-stage totals. An offline campaign analyzer requires same-node warmups,
at least three alternating measured Beammap pairs, matched config/input/runtime
policy, complete measurements, an explicit runtime budget, and required RSS
measurement; it reports paired ratios plus median and IQR. The checked-in
campaign is a diagnostic template rather than a mandatory Phase 4 run. If used,
the 5% wall-time ceiling applies and peak RSS remains required evidence with an
evidence-driven limit.

The GNU Time wrapper passed its first live Unity exercise in point `redu67` at
`7ca0be50c`. It captured matching retained and attached evidence, binary and
dependency identities, host/storage/runtime policy, 131.08 seconds external
wall time, 110.477 seconds Citlali time, and 908,316 KB peak RSS. The active
point profile accepted `redu67` against immutable baseline `redu66`: zero
logged issues, zero differences across 490 config leaves, and zero changes in
2,064 records from 19 products. This qualifies the wrapper integration but does
not constitute a Beammap performance conclusion.

The project owner accepted a proportionality exception to a dedicated Beammap
campaign on 2026-07-16. Twelve accepted refactor checkpoints range from
3,397.522 to 4,215.296 seconds with a median of 3,594.693 seconds, move in both
directions, and end with a 1.9% adjacent increase. A prior 13.0% total-time
increase coincided with 1.3% faster mapmaking and was concentrated in
VAST-sensitive PTC and diagnostics I/O. This history and repeated scientific
validation show no sustained regression signal. Serializing Citlali jobs would
not control unrelated VAST traffic, so eight dedicated hour-scale reductions
are not justified now.

Future naturally required Beammap validation should use the wrapper to collect
peak RSS and full provenance. A controlled campaign becomes mandatory only for
a sustained runtime regression, unexplained stage slowdown, memory failure,
peak RSS near node capacity, or a material hot-path change. Profiling overhead
is likewise investigated when a performance signal warrants adding an explicit
control. See the
[controlled performance protocol](PHASE4_PERFORMANCE_PROTOCOL_2026-07-16.md).

A planned post-refactor re-reduction of approximately 50 historical Beammap
observations will provide the broader operational performance census. Before
that work starts, add a lightweight corpus manifest and analyzer around the
existing evidence extractor. Each run should retain observation/workload,
config, binary, node, runtime, RSS, stage, I/O, and outcome identity. Analyze
the distribution against workload and preserve same-observation pairings where
available; do not treat unlike observations as repeated trials or reduce the
corpus to an unqualified average. This census is a future release baseline, not
a Phase 4 closeout prerequisite.

Phase 1 safety stabilization is complete for point, Beammap, science, and OOF.
OOF refactor `redu01` closes the multi-observation date-header gate and is the
accepted comparison against OG `redu00`. Do not reopen typed analysis-control
migration during Phase 4; validation and reproducibility are now the priority.

Operational config migration must proceed one authority domain at a time with
the one-way requested-to-effective-to-realized contract, focused tests, and the
existing mode gates. Compact-config production rollout and open-ended file
splitting remain out of scope.

The initial Phase 3 session boundary is implemented locally. A non-copyable
`citlali::session::ReductionSession` owns sequential run state and returns a
structured `ReductionResult` containing status, diagnostics, product roots,
and published provenance artifacts. Standard reduction loading and processor
selection now execute inside that session. The CLI remains the only layer that
prints result diagnostics and translates success to a process exit code.
Focused tests cover success, exception conversion, failure recovery, two
sequential runs, nested-run rejection, CLI policy separation, independent
header compilation, and multi-translation-unit linkage. Both local test
targets build, all 448 CTests pass, and full config preflight passes. This is
the facade checkpoint, not the Phase 3 exit gate: reachable library exits,
complete internal failure classification, remaining lifecycle ownership cuts,
and validation of the first `.cpp` boundary remain open. The
[bounded ownership plan](PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md) records the
sequence and stop rules.

The first failure-boundary and exit-census slice is also complete locally.
`ReductionSession` classifies canonical config, I/O, output, runtime, and
internal errors without terminating the process. Eight direct setup exits are
retired without touching numerical loops. An independent scan-context test
found and repaired a real include-order dependency on typed runtime policy.
The new conservative session audit follows 667 project-header dependencies
from the reusable entry and freezes a no-growth baseline of 94 direct library
exits across 22 files, with no CLI exits in the graph. The
[exit census](PHASE3_SESSION_EXIT_CENSUS_2026-07-15.md) defines the bounded
retirement order and separates low-risk setup/output work from mature
timestream and Wiener kernels. The first post-baseline cluster removes all six
TOD output-selection config exits. Invalid strings, empty or nonpositive chunk
lists, negative counts, and impossible selection modes now accumulate atomic,
path-aware config diagnostics. The adjacent effective row-selection boundary
now converts invalid effective modes, empty source-crossing selections, and
out-of-range chunks to canonical errors while preserving valid row assignment.
The current dependency-reachable count is 85. Its isolated test also
characterized the remaining ambient named-logger dependency in the legacy
`get_config_value` helper for later ownership work.

The first observation-input tranche centralizes the three duplicated KIDs
matrix validity checks used by direct, loaded, and gap-aligned RTC input.
Finite matrices retain the same path; NaN and infinite values now become
canonical I/O errors that a `ReductionSession` can report without terminating
the process. Three focused tests cover the contract, and the session audit is
down to 79 dependency-reachable library exits.

The observation/input setup census group is now complete. Detector-count and
cross-network sample-rate mismatches, invalid gap-alignment sample rates,
negative derived extinction, missing polarization calibration groups, invalid
IIR/Nyquist combinations, and Beammap fit-map shape mismatches all use explicit
canonical failure categories. Existing valid setup, metadata reads, and
numerical work are unchanged; the sample-rate path retains one metadata read
per network. Eight focused contract tests pass, and the session audit is down
to 71 dependency-reachable library exits.

Required FITS image and PHDU output-slot validation is now session-safe. Nine
map, Stokes, array, noise-map, and PHDU cardinality exits route through one
canonical required-output failure helper: the library logs the concrete slot
diagnostic and throws an output error, while only the CLI selects a process
exit code. Valid slot lookup and map writing are unchanged. Focused success
and failure tests pass, including every retired branch, and the session audit
is down to 62 dependency-reachable library exits.

The FITS/ECSV adapter tranche completes the output census group. CCfits'
nonstandard `FitsException` hierarchy is caught at operation boundaries and
classified as input I/O or required-output failure; ECSV input and atomic
publication use the same categories. Negative-path tests found and fixed the
distinction between `FitsError` and its sibling open/create exceptions. The
last apparent exit in this group was inside a fully commented, unused Gaussian
transfer-function prototype, which was removed as dead code. The audit reached
57 exits after this tranche.

The final three non-kernel mapmaking preconditions are also session-safe.
Unsupported polarization/grouping combinations, non-altaz Beammap requests,
and missing Wiener template FWHM values now throw canonical config errors;
successful policy and template setup are unchanged. The audit now reports 54
dependency-reachable library exits, all confined to mature RTC, PTC,
timestream, and Wiener implementations. Further retirement must proceed by
measured algorithm-boundary tranche with corresponding mode validation, not by
mechanical replacement.

Run-owned profiling migration is complete locally without changing production
timing records. `ReductionSession` owns and resets a non-copyable
`StageProfileCollector`; the explicit owner now crosses loading, processor
selection, reduction, iteration, observation, generic output, engine setup and
pipeline, Pointing ordered-output, and Beammap internal and specialized-output
boundaries. Output-directory configuration, every production timing scope,
and sidecar publication use that owner. The process-static collector and the
legacy implicit adapter are deleted, and the collector is not stored in
`Engine`.

Tests prove sequential-run reset behavior and verify representative reduction,
observation, and map-output records in the supplied collector. Both local build
targets pass, all 451 CTests pass, and full config preflight passes after the
atomic cutover. Unity point `redu63` confirms unchanged products and profile-
sidecar behavior. Its profile contains the same 76 stage/context records as
accepted `redu62`; only elapsed values and the natural completion order of
concurrent chunk writes differ.

The first concrete lifecycle cut after profiling removes a duplicate collector
reset from `run_reduction_pipeline`. Reset policy now belongs only to
`ReductionSession`, and a regression test proves that records created before
scientific-pipeline entry survive in the same run-owned collector. This is the
bounded stale-state repair required by Phase 3 step 4; no observation or scan
state was moved without a demonstrated hazard.

The first real compiled implementation boundary is accepted.
Timestream enum name tables and parse/format definitions now compile once in
`src/citlali/core/config/timestream_enums.cpp`; the public header retains enum
declarations and small predicates. The header shrank from 946 to 712 lines and
the new source is linked through `citlali`. One immediate before/after CLI
compile pair was 62.4 versus 63.7 seconds, so this slice demonstrates neither a
build-time win nor a material regression. All three local targets build, all
451 CTests pass, and full config preflight passes. Unity compile and point
`redu63` accept the boundary with zero product differences and no runtime
regression attributable to the extraction.

The first bounded mature-implementation exit tranche is accepted for its point
coverage.
Two PTC weighting exits now classify non-contiguous network grouping as input
I/O failure and impossible counters as an internal failure. RTC kernel setup
classifies mismatched kernel-image cardinality as invalid configuration. Valid
paths and numerical loops are unchanged; focused contracts cover each error
class. The dependency audit now reports 51 library exits and zero CLI exits.
Point `redu63` exercises the unchanged valid PTC weighting path exactly. The
next production tranche is fruit-loop map ingestion and requires matched
science and Beammap validation after its local checkpoint.

The fruit-loop map-ingestion tranche is locally complete. All 37 exits in
`TCProc::load_mb` now become canonical config or input-I/O failures at the
session boundary. Required file discovery, FITS header/schema, grouping and map
identity, WCS, and cardinality diagnostics retain their concrete context.
Optional `GROUPING` and `RADESYS` handling ignores only missing-key exceptions,
preventing real schema failures from being swallowed after the move to
exceptions. Valid loading and numerical processing are unchanged. All three
local targets build, all 453 CTests pass, full config preflight passes, and the
session audit is down to 14 library exits with zero CLI exits. Matched science
and Beammap fruit-loop validation is pending.

The three adjacent fruit-loop feedback exits are also retired locally behind a
header-isolated invariant boundary. Non-contiguous calibration grouping,
unknown detector-array identity, and out-of-range map indices now become
session-owned input-I/O failures before the affected map access. Interpolation
and map-to-TOD loops are unchanged. All three local targets build, all 454
CTests pass, and the session audit is down to 11 library exits, all in serial
or OpenMP Wiener filtering. This change shares the pending science and Beammap
fruit-loop acceptance runs.

The Wiener failure-boundary tranche is locally complete. Shared runtime
contracts cover serial and OpenMP template geometry, pixel spacing,
kernel/weight identity and shape, finite kernel peaks, and FFTW resource
creation. The OpenMP allocation path captures exceptions inside each worker,
synchronizes before worksharing, and rethrows only after leaving the parallel
region; partial FFTW resources are reset before failure. Valid filtering and
denominator arithmetic are unchanged. All three local targets build, all 455
CTests pass, full config preflight passes, and the conservative session audit
now reports zero dependency-reachable library or CLI exits. Standard point and
focused full-Wiener point are accepted. Fruit-loop science and fruit-loop
Beammap validation remain required before accepting these final mature
tranches.

The exit audit now also scans every implementation source under
`src/citlali/core`, closing a blind spot in the original header-reachability
census. The wider scan found and retired three invalid APT-table exits and one
invalid Lissajous chunk exit. Manual review confines the remaining textual
exits to successful CLI help/version handling and two legacy main programs that
CMake does not build. No supported non-CLI path retains explicit process
termination.

Unity point `redu64` accepts the standard point path at `6dd0057f8`. Its merged
configuration is byte-identical to `redu63`; the strict complete-product gate
opens every RTC/PTC array and reports 21 common products, 2,041 comparison
records, zero changed records, and zero skipped records. The audit reports 56
files, 22 stable comparable products, 12 PTC chunks, no logged issues, and all
required provenance valid. Total log time is 174.880 seconds versus 169.728
seconds and PTC chunk spacing differs by 0.5%, so no performance regression is
attributed. The queued science config enables fruit loops but retains
`wiener_filter.lowpass_only: true`, so it exercises convolution rather than
Wiener denominator construction. It remains the fruit-loop science gate. A
focused point run with noise maps enabled and `lowpass_only: false` supplies the
full-Wiener denominator gate. The fruit-loop science and Beammap runs remain the
mode-specific acceptance gates for map ingestion and feedback.

The full-Wiener gate is accepted on matched OG `redu10` and refactor `redu65`.
Their 490 low-level leaves differ only in the OG/refactor output directory and
the corresponding telescope-file path; the two telescope inputs have identical
SHA-256 hashes. Both runs execute six Wiener core calls with five noise maps and
`lowpass_only: false`. The strict filtered-product comparison reads seven
products and 148 records with zero changed or skipped records under the
established `2e-8 + 1e-10 * abs(reference)` profile. The pointing-fit table is
exact across all columns. Maximum signal and kernel absolute differences are
`6.34e-9` and `7.99e-10`. Refactor non-uniform denominator work totals 38.4
seconds versus 42.9 seconds for OG; the uncontrolled pair shows no performance
regression. The refactor run has no logged issues and valid required
provenance. OG's twelve `NetCDF: Not a valid ID` records are its known legacy
limitation and are not accepted as refactor behavior.

Beammap `redu06` accepts the remaining mature Phase 3 tranches for that mode.
Its low-level config is byte-identical to accepted `redu05`; both runs complete
198 PTC chunks and expose the same valid provenance and product inventory. The
strict zero-tolerance comparison reads every comparable FITS, NetCDF, and ECSV
product, including complete detector TOD, and reports 12 common products,
16,453 records, zero changes, and zero skips. Total log time is 4,215.296
seconds versus 4,136.440 seconds, a 1.9% uncontrolled difference with no
performance attribution.

The matched science attempt at `6dd0057f8` stopped during configuration before
creating a `reduNN` directory. TolTECA emitted the historical
`timestream.output.rtcdiag.enabled` leaf, which the new complete startup schema
did not recognize. Because that diagnostic prevented installation of the raw
execution adapter, the later kernel-template check reported the misleading
secondary error `wiener filter kernel template requires kernel`. Commit
`7ef43ef93` explicitly classifies the historical switch as an ignored
compatibility spelling: RTC diagnostics remain required and are always
written. The Wiener prerequisite now reads typed raw policy instead of mutable
`rtcproc` state, reducing the checked legacy access census from 44 to 43. The
exact failed science YAML now passes local configuration and reaches the raw
data boundary; all 456 CTests and full config preflight pass. A Unity science
rerun is required before closing the science fruit-loop gate.

The first repaired science rerun was invalidated by two Citlali jobs sharing
the same output root while `fruit_loops.save_all_iters=true`. One job advanced
to `redu26` and attempted to read `redu25/coadded/raw` while the other job was
still writing observation products into `redu25`; the resulting missing-map
diagnostic correctly exposed the incomplete input. This is an output-directory
ownership failure, not evidence of a numerical or fruit-loop ingestion change.
Production session execution now holds a nonblocking filesystem lease on the
configured output root from successful runtime setup through final provenance
publication. A competing Citlali process fails immediately with a required-
output diagnostic, while reductions using distinct output roots remain
independent. Focused tests cover contention, automatic release, independent
roots, and public-header linkage. The CLI build, all 460 CTests, and full config
preflight pass locally. Clean single-job science sequence `redu28` through
`redu31` then completed normally at pre-lease commit `a7a35a00`: every
iteration consumed its immediately preceding complete map directory, the final
run logged no issues, and exact-config scientific equivalence against accepted
`redu23` passed. This closes the fruit-loop map-input repair gate. The output-
root lease then passed its first Unity/VAST exercise in point `redu66`: the
parent log records successful exclusive acquisition, the run completed without
issues, and all non-timing products are exact against `redu65`. This closes the
Phase 3 output-ownership and compiled-boundary gates.

The runtime domain is the first operational Phase 2 migration. Requested,
effective, and realized runtime state are now separate in memory, and execution
consumes the effective thread and runtime policy. Remaining direct mutable
runtime reads are confined to config construction. The required, atomically
published `runtime_provenance.yaml` sidecar uses the stable
`citlali-runtime-provenance-v1` schema. Unity `redu27` validates the sidecar,
zero serious log issues, and exact pre-existing point products. The runtime
domain is complete; the next operational domain is timestream output selection
and chunking.

The timestream-output domain routes RTC/PTC output shape, outer-buffer
allocation, NetCDF serialization mode, metadata, selection, and scan-index
construction through typed configuration. The required, atomically published
per-observation `timestream_output_provenance.yaml` carries the versioned
requested/effective/realized output record. Unity `redu28` validates all 12
selected and realized RTC/PTC chunks, both registered TOD files, zero serious
log issues, and exact existing products. The former processor output-mode and
telescope chunking mirrors are removed; parser and writer boundaries receive
typed values explicitly. The local CLI/test build, all 229 tests, and full
config preflight pass. This domain is complete.

Work has started on the `raw-timestream` domain. Downsample enablement,
requested factor/frequency, anti-alias validation, and effective sample-rate
preflight now use typed raw-time-chunk configuration. Frequency-derived factors
are synchronized into the RTC downsampler only as an execution adapter. A
divergence test proves typed policy wins over stale processor mirrors. Typed
policy also controls FIR/notch/IIR setup, kernel-dependent allocation and
products, flux-unit selection, and extinction setup; processor objects retain
the corresponding numerical state. All 231 tests pass. Remaining RTC flagging,
source-protection, line-audit, and diagnostics boundaries are being migrated in
bounded clusters.

Raw source-protection activation now flows requested typed policy to realized
typed state and then to the RTC execution adapter. Learned-mask application,
FITS event-mask provenance, and RTC diagnostic impulsive-product shape consume
typed policy directly. The shared processed source-protection activation follows
the same direction. All 232 tests pass; line-audit and remaining diagnostic
configuration are the next raw-timestream clusters.

The line-audit cluster now uses typed policy for model-protected PTC audit
activation, model-subtraction requirements, notch-family selection, iteration
counts, frequency overrides, and dynamic edge-guard decisions. RTC diagnostic
sidecars, TOD headers, and chunk summaries serialize requested raw settings from
typed config. Existing RTC notch methods still consume the processor options
object as a numerical adapter, and realized edge-context/guard sample counts
remain processor state. The CLI build, all 232 tests, and full config preflight
pass.

The processed migration now has its first explicit one-way adapter:
`TimestreamFruitLoopsConfig` synchronizes the numerical `PTCProc` fields after
loading. A focused divergence test proves typed values overwrite adapter state.
This enables direct typed parsing to replace legacy parsing incrementally. All
236 tests and full preflight pass.

Direct typed parsing now owns the core fruit-loop lifecycle and model-selection
fields before the one-way processor adapter runs. This is a staged extraction:
expert fruit-loop numerical fields still arrive through the legacy parser and
typed mirror until their cohesive reader is moved. All eight real config
profiles, all 236 tests, and full preflight pass.

The direct fruit-loop reader now covers the complete typed fruit-loop surface,
including expert local-noise, adaptive-support, feedback, interpolation, and
post-addback controls. The legacy combined PTC parser remains temporarily for
other processed domains; fruit-loop execution state is overwritten only from
typed policy through the adapter. All 236 tests and full preflight pass.

Processed cleaning now has a complete one-way typed adapter covering all four
cleaner modes and correlation grouping. The local build caught and corrected
an `int` versus `Eigen::Index` boundary conversion before Unity. All 236 tests,
the CLI build, and full preflight pass.

The cleaning reader now directly owns core activation and mode-selection
policy before the one-way cleaner adapter runs. Expert mode parameters and
eigen-count padding remain in the compatibility parser for the next slices.
All 236 tests, all eight real config profiles, and full preflight pass.

Direct cleaning parsing now includes standard-PCA eigen-count normalization
and both current and legacy key aliases. Empty and short vectors receive the
same defaulting and padding behavior before the one-way adapter runs. All 236
tests, all eight real profiles, and full preflight pass.

Direct cleaning parsing now covers correlation grouping and null-model scalar
policy. Group-name canonicalization remains deliberately mirrored because it
still depends on cleaner-specific helpers. All 236 tests, all eight real config
profiles, and full preflight pass.

Direct cleaning parsing now covers Marchenko-Pastur and adaptive-selector
numerical policy, including adaptive frequency-band validation. The remaining
cleaning-parser dependency is cleaner-specific grouping-name canonicalization.
All 236 tests, all eight real profiles, and full preflight pass.

Raw input and metadata boundaries now use typed policy for duplicate-tone
frequency separation, RTC diagnostic FIR/source-bandwidth ratios, and whether
FITS/TOD tau metadata is calculated. The atmospheric calibration object remains
processor-owned numerical state. The CLI build, all 232 tests, and full config
preflight pass.

Learning collection and learned-mask/exclusion orchestration now reads typed
second-pass source-protection activation, radius, and score thresholds. This
removes another execution-facing dependency on `PTCProc` policy mirrors while
leaving its numerical implementation unchanged. All 235 tests and preflight
pass.

RTC diagnostic and RTC TOD diagnostic schema construction now receives typed
downsample and impulsive-capture policy explicitly. External raw product-shape
decisions no longer depend on `RTCProc` mirrors. Remaining raw-timestream work
is concentrated in numerical-method adapters internal to `RTCProc`; polarimetry
is tracked as a separate authority domain.

The first processed-timestream authority slice now routes fruit-loop
enablement, effective iteration count, retained-iteration output layout,
initial/previous model-map paths, and learning source-model availability
through typed fruit-loop config. Beammap and disabled-loop normalization is
recorded in typed effective state and copied into `PTCProc` only as an execution
adapter. All 232 tests and the full config preflight pass.

Processed-timestream orchestration now also uses typed fruit-loop policy for
model subtraction/add-back, source-subtracted weight retention, final noise-map
population, and beammap adaptive-gate setup. Processor state remains the home
of runtime model buffers and numerical kernels, but no longer decides whether
these operations are enabled. All 233 tests and the full config preflight pass.
Interpolation override selection and fruit-loop runtime-policy logging now use
the same typed authority; the processor retains only the realized interpolation
mode required by map-to-TOD execution. All 234 tests pass.

TOD, PTC-diagnostic, and FITS-map fruit-loop metadata now serializes typed
effective configuration, including array flux limits and pointing source-center
policy. Pointing warnings use the same authority. Runtime detector fit vectors
remain in `PTCProc`. The CLI build, all 235 tests, and full preflight pass.

Compact PTC-diagnostic `CONFIG.*` metadata now also reads typed cleaning,
weight-penalty, busy-row, and second-pass policy. This establishes a consistent
boundary: typed configuration is serialized as policy, while processor-owned
arrays remain realized diagnostics. All 235 tests and full preflight pass.

TOD NetCDF and map FITS cleaning metadata now uses typed processed-timestream
policy throughout. The only retained cleaner value at this output boundary is
the per-array removed-eigenmode count, which is a realized result rather than
configuration. The CLI build, all 235 tests, and full preflight pass.

Weighting metadata now uses typed raw and processed policy for scheme,
cutoffs, hybrid correction, and validation settings. The PTC diagnostic
sampling-window duration remains an explicit realized processor input pending
a typed representation. The CLI build, all 235 tests, and preflight pass.

Optional PTC TOD diagnostic block selection now reads typed processed policy.
Second-pass, correlation, busy-row, and adaptive-cleaner schema decisions no
longer depend on processor mirrors. The CLI build, all 235 tests, and full
preflight pass.

Processed effective-policy resolution is now being separated from YAML
parsing. Pure result types preserve requested values while recording cleaner
group canonicalization, weighting source-mask inheritance, validated-weighting
and busy-row dependency decisions, and disabled/beammap fruit-loop iteration
normalization. Cleaner-mode precedence and fruit-loop interpolation defaults,
overrides, and JINC fallback now use the same pattern; source-protection
activation has an explicit realized-state result. Existing mutating calls
remain thin compatibility adapters with unchanged warnings and processor
values. A non-wired `ProcessedTimestreamExecutionPlan` now provides separate
requested, effective, effective-resolution, and realized storage without
claiming complete output provenance. The CLI and test builds, all 243 tests,
all eight config profiles, and the frozen 171-path PTC boundary audit pass.
The boundary audit now also routes all 171 paths to their declared typed
reader, requires each leaf key in that source, and fails preflight on uncovered
paths or stale compatibility aliases. This mechanically satisfies the path-
coverage prerequisite for removing the legacy parser; the provenance and
cross-mode validation prerequisites remain open.
Focused adapter tests now assign and verify every field copied from typed
fruit-loop, cleaning, weighting, validation, correlation-penalty, busy-row,
and second-pass configuration into the processor compatibility targets. The
full C++ suite passes all 244 tests. The concrete `PTCProc` header remains a
contextual include rather than an isolated test dependency; that existing
header-boundary defect belongs to Phase 3 and was not expanded in this phase.
The non-wired execution plan now has an atomic repeated-run reset operation.
Disabled sections retain their requested parameter values while remaining
inactive, and reset clears all prior effective-resolution and realized state.
All 245 C++ tests pass. Current legacy reader objects are not reset piecemeal;
the contract became operational only with the complete Engine wiring described
below.
Pure YAML component serializers now cover the complete requested/effective
processed snapshot surface. The boundary audit enforces serialization of all
171 frozen legacy paths as well as typed-reader coverage. There is deliberately
no final provenance schema version, output filename, or writer yet; effective-
resolution and realized-state component serialization now also use explicit
availability records. Beammap `redu14` (`4b0126e7`) completed cleanly and
exactly reproduces accepted refactor `redu11` across all 5,234 detector maps.
It also passes the versioned OG scientific-equivalence profile with exact
detector identities, flags, and product sets. The matched beammap gate is
therefore closed. `Engine` now owns and initializes the processed execution
plan, processed runtime accessors select its effective snapshot, and cleaner,
weighting, source-protection, interpolation, iteration-policy, and completed-
iteration decisions populate explicit resolution or realized records. The
legacy parser remains only as the compatibility seed and no provenance file is
published yet. Unity point `redu34` (`86c47fa7`) passes the strict complete-
product gate against accepted `redu33`: its 489-leaf config is exact, all 13
scientific product families are present, and every RTC/PTC timestream and map
record is exact with zero skipped records. The Engine authority change is
accepted; the versioned provenance root and atomic writer are next.
The v1 processed provenance sidecar is now implemented at the CLI success
boundary. It writes the authoritative plan only after completed iterations,
uses the shared atomic YAML writer, and fails the reduction on uninitialized
state or filesystem failure. Local CLI/test builds, all 252 tests, all eight
config profiles, and the 171/171 boundary audit pass. Unity output validation
of the new required sidecar passes at `81020d46` point `redu35`. The sidecar
contains all five effective-resolution and all three realized-state records;
its schema and hash are recorded in the validation ledger. Against accepted
`redu34`, the merged 489-leaf config and all 13 scientific product families,
including complete RTC/PTC timestreams, are exact with zero skipped records.
Point, beammap, and science processed provenance are accepted. The documented
compatibility-parser removal gate is closed.
Parser-removal preparation now includes a complete default-snapshot parity
test using a real value-initialized `PTCProc`. Typed defaults and the legacy
compatibility snapshot are identical across every serialized processed field.
Together with 171/171 reader coverage and exhaustive one-way adapter tests,
this closes the deterministic omitted-default prerequisite without changing
production parsing. All 253 C++ tests pass.
The six PTC-to-typed mirror calls are now consolidated behind
`seed_processed_timestream_config_from_legacy(...)`. Production still performs
the same compatibility seeding, typed reads, resolution, and one-way adapter
steps, but the legacy parser exit is now one named boundary. Local CLI/test
builds, all 253 C++ tests, config preflight, and provenance-audit tests pass.
Unity beammap `redu15` at `50235fd6` closes the beammap processed-provenance
gate. Its 529-leaf config is exact against accepted `redu14`; all 12 comparable
FITS, NetCDF, and ECSV products are exact with no skipped records; the required
sidecar passes semantic audit; and wall time improved from 3576.607 to 3458.917
seconds. The final matched science pair is OG `ffc6b907` `redu27` and refactor
`50235fd6` `redu24`; the intermediate `reduNN` directories are retained
fruit-loop iterations, not independent runs. Their 502-leaf configs differ only
in input/output path strings. Science-equivalence profile v2 preserves the
`1e-8` raw-map bound and separately enforces the owner-approved 1.5% filtered-
map bound. All 63 raw layers remain within `2.33e-11`; the 21 Wiener-filtered
layers peak at 0.986%; product sets and integer diagnostics are exact; all
other numerical bounds pass. Refactor wall time is 2686.252 seconds versus
2754.146 seconds for OG. The science processed-provenance gate is accepted and
recorded in the validation ledger.
The processed authority migration is now operationally complete in production:
`Engine::get_ptc_config` starts from typed defaults, reads all 171 paths through
typed readers, resolves the effective plan, and populates `PTCProc` only through
one-way execution adapters. The legacy parser call, compatibility seed, and all
processed PTC-to-typed mirrors are removed. The retired-boundary audit rejects
their reintroduction while preserving 171/171 reader and serializer coverage.
Local CLI/test builds, all 252 C++ tests, all eight config profiles, and 13
focused Python tests pass.
Unity point `redu36` at `c22bc127` closes the production parser-retirement
gate. Its merged config is an exact 489-leaf match to accepted `redu35`; all 13
scientific product families, including every RTC/PTC array, are exact with zero
missing, extra, changed, or skipped records. Processed and runtime provenance
are byte-identical; timestream-output provenance differs only in the expected
`redu35`/`redu36` paths. The run completed without serious log issues in 53.277
seconds versus 60.159 seconds for the baseline. This acceptance is recorded in
the validation ledger. The frozen 171-path inventory now lives in the versioned
`processed_timestream_legacy_paths.json` manifest. Boundary-audit schema v5
validates its canonical ordering, declared count, and digest before checking
171/171 typed-reader and serializer coverage. The unreachable
`PTCProc::get_config` declaration and roughly 1,190-line body are deleted.
Local CLI/test builds, all 252 C++ tests, all eight config profiles, and eight
focused boundary-audit tests pass after deletion. The processed-timestream
authority migration and its legacy-parser cleanup are complete.
Raw-timestream characterization is the next bounded Phase 2 domain. The frozen
RTC boundary contains 169 raw paths plus two adjacent polarimetry paths,
originally 14 direct parser exits, one production parser call, and ten
legacy-to-typed mirror helpers. The authority inventory now labels raw execution as legacy-authoritative
instead of incorrectly claiming a typed-to-legacy adapter. The finite transition
contract is `doc/raw_timestream_config_transition.md`. No RTC execution behavior
has changed. The non-wired preparation checkpoint now has 169/169 direct typed-
reader and request-serializer coverage. A 40-record external RTC access census
classifies 22 executor operations, six observation-state accesses, seven
output/realized-state accesses, one raw policy read, and four separate-domain
polarimetry accesses, with zero unreviewed records. An unwired execution plan
separates requested, context-free effective, per-observation, and realized
state and resets observation state between runs. All 260 C++ tests, 21 focused
config-tool tests, and all eight config profiles pass. Production remains
legacy-authoritative. The complete unwired typed-to-RTC adapter now covers all
169 raw paths, with a real-`RTCProc` request round trip, disabled-sentinel
checks, and a separate observation-state overlay for sample rate, downsampling,
edge context, source protection, and extinction. The frozen audit enforces
169/169 adapter coverage. All 264 C++ tests, 22 focused config-tool tests, and
all eight config profiles pass. Pure observation resolution now covers native
and effective sample rate, derived downsample factor and anti-alias checks,
filter edge guard/context contributions, source-protection activation, and
extinction-model selection. Filter transient estimates and extinction-model
selection are shared by the typed resolver and legacy processors rather than
duplicated. Focused tests prove edge-guard parity for sum/max policies and
extinction parity across representative tau values. All 271 C++ tests and full
preflight pass. Constructing the typed plan as a non-authoritative production
shadow is the next gate before the authority flip. That context-free shadow is
now active: the Engine directly reads an isolated typed request, constructs the
raw execution plan, adapts into a temporary RTC policy object, and requires its
deterministic 169-path snapshot to equal the legacy parser/mirror snapshot.
Legacy `rtcproc` still drives execution. The frozen audit requires one typed
read before the parser and one comparison after all ten mirrors. The generated
default config, disabled expert semantics, and injected divergence behavior are
covered by focused tests. The per-observation shadow is now active at the
existing lifecycle boundaries: input preparation records and compares native
and effective sample rate, downsample factor, edge guard/context, and raw source
protection; observation setup records and compares extinction activation and
model. Legacy `rtcproc` remains the execution authority. A second observation
resets the first observation's state and realized counters. Frequency-derived
downsampling exposes a pre-existing ordering gap because legacy configures its
edge guard before deriving the factor; that single comparison is explicitly
marked deferred rather than changing numerical behavior. All other divergence
fails with field-level diagnostics. The external RTC census is frozen at 44
classified records with zero review-required entries. Local CLI/test builds,
all 277 C++ tests, 23 focused config-tool tests, all eight profiles, and full
preflight pass. Unity validation of this shadow checkpoint is pending; no raw
authority flip or parser/mirror retirement is permitted before that gate.
The versioned `citlali-raw-timestream-provenance-v1` schema was prepared but not
yet wired at this checkpoint. It serializes the complete requested/effective
config, context-free resolutions, explicit observation-field availability and
edge-guard deferral, an execution-completed marker, and realized counters. Its
atomic writer rejects uninitialized plans and propagates publication failures.
All 281 C++ tests and full preflight pass. Production publication remained
deferred so required-output placement and lifecycle completion could be reviewed
with the Unity shadow checkpoint rather than introduced without mode evidence.
The remaining 14 direct exits in `RTCProc::get_config` are removed. Legacy
cross-field checks now append exact invalid-key paths to the existing config
diagnostics and continue safely through malformed notch vector shapes; the CLI
validation boundary remains responsible for rejecting the reduction. Valid
configuration behavior is unchanged. The frozen raw boundary now requires zero
direct parser exits. Local builds, all 282 C++ tests, and full preflight pass.
Unity point `redu37` accepts the complete raw-shadow checkpoint at `cd8da24f`.
The run used the same 489-leaf merged config hash as accepted `redu36`, completed
all 12 PTC chunks with zero logged issues, and retained the exact 36-file/14
stable-product inventory. Strict comparison including complete RTC/PTC
timestreams found 13 common product families, zero missing or extra products,
zero changed records, and zero skipped records. Runtime and processed
provenance are byte-identical; output provenance differs only in expected
`redu36`/`redu37` paths. Logged runtime was 51.723 seconds versus 53.277 seconds
for `redu36`. This closes the Unity point gate for observation shadowing,
prepared raw provenance, propagated parser diagnostics, and yaml-cpp 0.7
compatibility. Beammap/science evidence remains required before raw authority
flip and parser/mirror retirement.
The accepted point shadow gate now permits required production raw provenance.
Each successfully completed observation atomically publishes
`raw_timestream_provenance.yaml` in its observation directory after required TOD
writers and observation products have completed. The observation lifecycle owns
the completed-scan count and expected required TOD-write count; flagged-sample
and dynamic-notch counts remain explicitly unavailable rather than being
guessed from mutable RTC state. Publication failure propagates and fails the
reduction, and the writer rejects observation, completion, or realized-count
state that is incomplete. Repeated-observation tests prove state reset and
independent sidecars, while a filesystem-failure test proves required-output
propagation. The run-audit tooling can require and semantically validate every
observation's sidecar, including science reductions. It pairs setup-time output
provenance with completion-time raw provenance, rejects missing observation
sidecars, cross-checks scan counts, and validates resolved sample-rate state.
Local CLI/test builds, all 287 C++ tests, 11 provenance-audit tests, and full
config preflight pass. Unity point `redu38` accepts the required raw provenance
at `6bbc12ce`. It has the identical merged config and stable 14-product inventory
as accepted `redu37`, zero serious log issues, and a valid observation sidecar
recording 12 completed scans and 48 required writes. Strict comparison opened
all RTC/PTC arrays across 13 common product families and found zero missing,
extra, changed, or skipped records. Logged runtime was 51.459 seconds versus
51.723 seconds for `redu37`. The point publication gate is closed and recorded
in the validation ledger. Beammap and science acceptance remain pending; raw
execution therefore remains legacy-authoritative.
A science cross-mode attempt at `5d403887` stopped before observation numerical
processing because the shadow compared typed physical downsample factor 1 with
legacy RTC's disabled value 0. Legacy initializes and reads that factor only
when downsampling is enabled, so inspecting it while disabled is outside the
legacy contract and can read inactive state. Observation parity now always
compares enablement and compares factor only when enabled; typed observation
state still records the physical identity factor 1 and unchanged sample rate.
A focused science-style test preserves enabled-factor divergence detection and
accepts the disabled legacy sentinel. Local CLI/test builds, all 288 C++ tests,
and full config preflight pass. The science and Beammap gates must be rerun.
The repaired `2d6f80a3` candidate closes both cross-mode publication gates.
Beammap `redu17` has one complete raw sidecar with 198 scans and 198 required
writes; all 12 complete product families are exact against accepted `redu15`,
with zero skipped records and runtime 3397.522 versus 3458.917 seconds. Science
final iteration `redu29` has two complete raw sidecars, each with 124 scans and
248 required writes; all 27 complete product families pass the strict gate
against accepted `redu24` with zero changed or skipped records. Its largest
absolute difference is `4.452e-10`, within established tolerance, and runtime
is 697.572 versus 705.784 seconds. Both runs have zero serious log issues and
log each published sidecar path. The validation ledger records both accepted
checkpoints. Point, Beammap, and science prerequisites are now satisfied for
the bounded raw execution-authority cutover; OOF reuses the accepted pointing
execution gate, and polarimetry remains outside this authority claim.
The bounded raw execution-authority cutover is now implemented locally. Direct
typed parsing initializes requested/effective plan state and the one-way
production `RTCProc` adapter. The legacy parser and ten mirrors remain only as
a temporary read-only oracle whose deterministic snapshot must match the
production RTC before execution. Focused tests prove stale processor state is
overwritten, disabled requested values remain intact, and divergence fails.
The CLI build, all 291 C++ tests, all eight real config profiles, the complete
169-path boundary audit, and the frozen 44-record execution-read census pass.
Unity point, Beammap, and science cutover validation is the next gate; parser
and mirror retirement is prohibited until it passes.
The first Unity point cutover attempt at `475bf8e22` reached map output but
failed because the production `RTCProc` no longer received the adjacent legacy
polarimetry initialization. For an unpolarized run, that parser side effect
creates the mandatory Stokes-I entry; without it, `stokes_params` was empty and
map indexing read invalid state. A narrow legacy-polarimetry runtime adapter now
copies only enablement, grouping, and Stokes labels from the temporary parser
object. Polarimetry remains outside the raw authority claim. A focused
regression test and the boundary audit require this transfer. The repaired
candidate builds locally, all 292 C++ tests and all eight profiles pass, and
full preflight has zero drift. Unity point cutover validation must be rerun.
The repaired point run `redu40` completes with zero serious issues, all required
provenance valid, and exact scientific products and complete timestream arrays
against accepted `redu38`. The strict gate nevertheless rejects two metadata
records: disabled `CONFIG.TODIIRHP.FREQ_HZ` changed from the established
processor-effective sentinel `0.0` to the preserved inactive request `0.1` in
the RTC and PTC NetCDF products. Raw provenance correctly retains the request
and explicit disabled resolution, so the fix is a pure FITS/NetCDF metadata
projection rather than a plan mutation or processor readback. Disabled IIR
metadata now resolves to frequency `0.0`, order `1`, and zero-phase `false`;
enabled values pass through. All 293 C++ tests and full preflight pass locally.
One final point rerun is required before starting the expensive Beammap and
science cutover gates.
The raw execution-authority cutover validation gate is closed. Point `redu42`
at `880869b3` passes the complete strict comparison against accepted `redu38`:
13 common product families, zero changed or skipped records, valid byte-stable
raw/processed/runtime provenance, zero serious issues, and runtime 54.412 versus
51.459 seconds. Beammap `redu18` at `398d5127` has exact numerical products and
all 5,234 detector results against `redu17`, zero skipped records, valid
byte-stable provenance, and zero serious issues. Its six accepted rtcdiag
metadata changes expose configured values beneath a disabled local-residual
section instead of legacy processor defaults. Science final iteration `redu33`
at `398d5127` passes against `redu29` with 27 common products, zero changed or
skipped records, maximum absolute difference `3.746e-10`, byte-stable
provenance, zero serious issues, and runtime 704.234 versus 697.572 seconds.
The validation ledger records all three accepted gates. OOF reuses the point
execution gate; polarimetry remains separate. The temporary 169-path raw parser
and ten oracle mirrors may now be retired as the next bounded change while
retaining the narrow adjacent polarimetry compatibility boundary.

The authorized raw-parser retirement is complete locally. The declaration and
roughly 1,080-line `RTCProc::get_config` implementation, all ten raw reverse
mirrors, and the context-free parity oracle are removed. The versioned
`raw_timestream_legacy_paths.json` manifest preserves the canonical 171-path
historical surface and digest. The boundary audit now rejects reintroduction of
the parser, a raw mirror, or the parity comparison while continuing to enforce
169/169 direct-reader, serializer, and typed-to-RTC adapter coverage. The two
adjacent polarimetry keys use a dedicated compatibility reader and one-way
runtime adapter; they do not repopulate raw typed state. A forward TOD output-
context helper formerly hidden in the mirror umbrella now has its own named
header. Fresh local CLI, primary-test, and safety-test builds pass all 285 C++
tests; 12 focused raw-boundary audit tests, the unchanged 44-record execution
census, all config profiles, full preflight, and the validation ledger pass.
Unity point `redu43` at `11afd6f6` closes the retirement gate against accepted
`redu42`. The merged 489-leaf config is exact, all 13 product families and
complete RTC/PTC arrays are exact with zero changed or skipped records, all
required provenance is valid, and raw, processed, and runtime sidecars are
byte-identical. Output provenance differs only in the expected reduction-number
file paths. The run has zero serious issues and completed in 54.182 seconds
versus 54.412 seconds. The validation ledger records the acceptance. The raw-
timestream authority migration, including legacy parser/oracle cleanup, is now
complete; polarimetry remains a separate compatibility domain.

The mapmaking authority migration has passed its first Unity mode gates. All
22 frozen `mapmaking.*` leaves now enter typed request state through
one boundary. `MapBuffer`, JINC, maximum-likelihood, observation-map, and
coadd-map configuration no longer parse YAML. One-way adapters construct the
legacy numerical mapmakers and WCS buffers from typed state. The immutable
execution plan preserves the requested grouping while exposing the resolved
effective grouping to downstream accessors; the transitional root request is
no longer mutated by map-count setup. Successful reductions must atomically
publish versioned `mapmaking_provenance.yaml`, and write failures propagate.
The effective plan also records the uncalibrated TOD-type unit substitution
without changing the requested `cunit`. Version-2 provenance now records one
identified observation per input in the final fruit-loop iteration, each
observation's map count, effective pixel size, required logical map-product
count, optional coadd cardinality, and completion state. Lifecycle counters
reset between fruit-loop iterations and advance only after required output
stages return successfully; CLI completion rejects incomplete or inconsistent
counts. The audit accepts historical version-1 sidecars but applies strict
cardinality semantics to version 2. The boundary preflight freezes the
22-path digest, enforces 22/22 reader
coverage, rejects retired parser symbols, and checks the production authority
sequence and provenance writer. Local CLI/test/safety builds, all 305 C++
tests, all eight config profiles, and the full preflight pass. A strict point
run is required first to validate the lifecycle wiring and new sidecar;
Beammap and science runs then validate their mode-specific output cardinality.
This Unity validation is the last mapmaking provenance sub-gate. Unity point
`redu44`, final science
iteration `redu03`, and Beammap `redu00` all embed `5c8f5eb4`; their merged
configs are exact against accepted `redu43`, `redu33`, and `redu18`
respectively. All three runs have zero serious log issues and valid mapmaking,
raw, processed, output, and runtime provenance. Point has 13 exact complete
product families including RTC/PTC TOD. Science has all 27 products with zero
skips and passes the scientific-equivalence profile; its largest map
RMS-relative difference is `5.87e-14`. Beammap has exact non-map products,
exact identity and flags for all 5,234 detectors, and zero RMS difference in
every accepted good/bad signal, weight, and kernel map. Point, science, and
Beammap runtimes are 55.341, 699.904, and 3483.362 seconds, respectively,
versus 54.182, 704.234, and 3580.078 seconds for their baselines.

Version-2 cardinality validation is accepted at `e8e42945`. Point
`redu45` is exact against `redu44`: its 489-leaf merged config is unchanged,
all 13 product families including complete RTC/PTC arrays compare exactly, the
strict audit reports zero issues, and runtime is 56.176 seconds versus 55.341
seconds. Final science iteration `redu07` is accepted against `redu03`: its
502-leaf merged config is unchanged, all 27 products are present with no
skips, the dedicated science-equivalence profile reports a maximum map RMS-
relative difference of `6.23e-14`, and runtime is 709.597 seconds versus
699.904 seconds. Both version-2 sidecars report complete, internally
consistent observation/coadd cardinality. Beammap `redu01` is exact against
`redu00`: its 529-leaf merged config is unchanged, all non-map ECSV/NetCDF
products compare exactly, all 5,234 detector identities and flags are exact,
and every accepted good/bad signal, weight, and kernel map has zero RMS
difference. Its strict audit reports zero issues, 198 completed PTC chunks,
and one completed 5,234-map observation with no coadd; runtime is 3449.262
seconds versus 3483.362 seconds. The validation ledger records all three
accepted runs. The mapmaking authority and provenance domain is complete.

The bounded coadd authority domain is implemented locally without changing
coaddition numerics. Its frozen one-path reader owns `coadd.enabled` and
preserves the requested value. `CoaddExecutionPlan` resolves effective
activation from the mapmaking plan without mutating that request. Successful
CLI reductions require atomic `coadd_provenance.yaml` using schema
`citlali-coadd-provenance-v1`; its realized map and required-write cardinality
is a one-way snapshot of the already validated mapmaking coadd lifecycle, and
the reduction audit rejects disagreement between the two sidecars. The legacy
coadd reader and reverse mutation helper are removed. Local CLI/test builds,
all 314 C++ tests, all 38 focused config tests, 24 reduction-audit tests, all
eight config profiles and full preflight pass. Unity point `redu46` at
`c2e053b3` closes the disabled-coadd gate against accepted `redu45`: all 489
config leaves and all 13 complete scientific product families, including RTC
and PTC timestream arrays, are exact with zero skipped records or serious log
issues. The new coadd sidecar records requested/effective disabled activation,
no execution or cardinality, and agrees with the unchanged mapmaking sidecar.
All prior provenance is byte-identical except the expected reduction-number
TOD paths. Runtime is 53.804 seconds versus 56.176 seconds. Final science
iteration `redu11` at `c2e053b3` closes the enabled-coadd gate against accepted
`redu07`: all 502 config leaves match, all 27 products are present with zero
skipped records or serious log issues, and the science-equivalence profile
accepts a maximum map RMS-relative difference of `7.65e-14`. Coadd provenance
records requested/effective enabled, successful execution, three maps, six
required logical writes, and completed outputs; every value agrees with
mapmaking provenance. Runtime is 719.154 seconds versus 709.597 seconds. The
33-record validation ledger passes. The coadd authority and provenance domain
is complete.

The bounded `noise-products` implementation checkpoint is complete.
The six frozen `noise_maps.*` inputs now have one direct typed reader, a
requested/effective/realized `NoiseExecutionPlan`, and a one-way adapter into
the mature observation/coadd map buffers. The existing deterministic Boost
MT19937 identity is now explicit and versioned as fixed internal seed `5489`;
no user-facing seed knob was added. Required atomic
`noise_products_provenance.yaml` records activation/count resolution, final-
iteration observation/coadd realization cardinality, empirical-product count,
realization-image count, and completion. The reduction auditor validates those
semantics and cross-checks scientific-map cardinality against mapmaking v2
provenance. The legacy noise readers and reverse request mutations are retired.
The CLI/test build, all 328 CTest cases, all eight config profiles, the frozen
six-path audit, 48 config-boundary tests, and full preflight pass. No noise-
generation or product algorithm changed.

Unity point `redu47` at `1faec7cc` closes the disabled-noise path against
accepted `redu46`: all 489 config leaves and all 13 complete product families
are exact, with no skipped records or serious log issues. Point `redu49`
closes the bounded full-output fixture with ten realizations per scientific
map, three empirical-product maps, and 30 realization-image writes. Its
realization, empirical-variance, and empirical-weight outputs agree with the
matching OG fixture at maximum RMS-relative differences of `7.65e-14`,
`8.84e-14`, and `6.42e-14`, respectively. The final science iteration
`redu15` closes the generation-only coadd path: six observation maps produce
60 realizations and three coadd maps produce 30, for exactly 90 total with no
optional empirical products or realization files. Its 502-leaf config is
exact against accepted `redu11`; all 27 scientific products are present with
no skips, and the science-equivalence profile accepts a maximum map RMS-
relative difference of `6.93e-14`. Against the matching OG science run, the
profile accepts the previously approved filtered-map differences with maximum
map RMS-relative difference `0.00986`. All three candidate runs have valid
version-1 noise provenance and zero serious log issues. The noise-products
authority and provenance domain is complete.

The bounded pointing implementation is locally complete. Its frozen five-key
surface now has a direct typed request reader, a separate effective execution
plan, and a one-way adapter for the three mature PTC source-center fields.
Effective fit activation preserves the request and depends only on availability
of normalized observation maps from mapmaking. Optional filtering and coaddition
occur downstream and do not disable raw pointing fits. Required atomic
`pointing_provenance.yaml` records the request, resolution decisions,
per-observation map/fit cardinality, and realized completion. The reduction
auditor validates those semantics and cross-checks observation identity and
map counts against mapmaking v2 provenance. The CLI/test builds, all 336 CTest
cases, the frozen boundary audit, all eight compact profiles, and full config
preflight pass. Gaussian fitting, Ceres use, source finding, and map numerics
are unchanged. Unity point validation remains the sole exit gate before this
domain is complete.

The first Unity candidate, point `redu50` at `98d2a5d2`, correctly exposed an
effective-policy error. Its 489-leaf config, maps, timestreams, diagnostics, and
all non-fit products are exact against disabled-noise `redu47`, and it has zero
serious log issues. However, the new plan incorrectly treated disabled map
filtering as making pointing fits unavailable. The resulting three-row pointing
table zeroed all 11 fitted columns instead of preserving the accepted fits. The
gate is failed. Pointing fit availability now follows mapmaking alone; the
semantic auditor rejects the invalid `redu50` sidecar, and focused tests cover
both filter-independent fitting and mapmaking-disabled fitting. Local builds,
all 336 CTests, 43 baseline-tool tests, 54 config tests, all eight profiles, and
full preflight pass. A corrected Unity point run remains required.

Corrected Unity point `redu51` at `a9d17fa1` closes the pointing gate. Its
489-leaf merged config is exact against accepted disabled-noise `redu47`; all
13 scientific product families, including every RTC/PTC timestream record and
all pointing-fit columns, are exact with zero changed or skipped records. The
candidate has zero serious log issues and valid pointing provenance recording
one observation, three scientific maps, three fit attempts, and three valid
fits. Runtime is 59.971 seconds versus 58.627 seconds. The validation ledger
records the accepted checkpoint. The pointing authority and provenance domain
is complete; post-processing is now the active bounded domain.

Post-processing characterization freezes 35 supported leaves: 24 under
`post_processing.*` and 11 under the historical top-level `wiener_filter.*`
prefix. The latter controls filter template construction and convergence and
therefore belongs to the same authority domain. The starting boundary is
intentionally mixed: the legacy Wiener parser still reads 21 leaves and
reverse-mirrors most of them into typed state, while direct typed readers cover
13 other leaves. The initial typed-request gaps,
`post_processing.source_fitting.model` and
`wiener_filter.kernel_template_tail_mode`, now have closed-enum representation
in a complete 35-leaf direct request reader. That reader now runs during
`Engine` config loading as a fail-fast, read-only shadow. Activation and
histogram always compare; detail fields compare only when the legacy path
loads them, so disabled requested values are preserved without false mismatch
reports. The legacy parser and reverse mirrors still drive execution. Focused
shadow tests cover inactive science policy, pointing fit values, active filter
values, and mismatch diagnostics. The CLI/test builds, all 342 CTests, 60
config tests, all eight compatibility profiles, and full preflight pass. See
`doc/POST_PROCESSING_CONFIG_AUTHORITY.md`.

Unity point `redu52` at `d9db1183`, the first enabled-filtering overlay,
reached both the raw and filtered
pointing-fit stages, then failed during lifecycle recording with `pointing fit
results already recorded`. This exposed a provenance-model defect rather than
a fitting or mapmaking failure: version 1 represented only one fit event per
observation even though filtered pointing output deliberately fits the maps a
second time. The execution plan now names the raw and filtered fit stages,
enforces exactly one result per expected stage, and records their cardinalities
separately in `citlali-pointing-provenance-v2`. The reduction auditor accepts
both historical v1 and current v2 sidecars and validates stage expectations
against filtering/coadd policy. Numerical fitting and product-writing order are
unchanged. Local `citlali_cli`/test builds, all 344 CTests, 45 provenance-tool
tests, 60 config tests, all eight compact profiles, and full preflight pass. The
same enabled-filtering point overlay must pass on Unity before post-processing
authority migration proceeds.

Unity point `redu53` at `c75f079b` closes that repair and enabled-filtering
gate. It completes in 59.772 seconds with zero serious log issues. Its v2
pointing sidecar records one observation, three raw and three filtered fit
attempts, all valid, and completed output. All 13 products shared with accepted
unfiltered refactor `redu51` are exact, proving the overlay and lifecycle repair
did not alter the raw path. Against matching OG point `redu09`, all eight
filtered products are present with no skipped or changed records under the
standard numerical gate; the three-row pointing-fit table and 195-row source
table are exact. Maximum filtered signal absolute difference is `2.97e-11`.
The 490-leaf merged configs differ only in their two expected output paths.
The validation ledger records the accepted checkpoint. Post-processing may now
advance from request shadowing to a separate effective execution plan.

The first post-processing authority checkpoint is complete locally. A
`PostProcessingExecutionPlan` now owns the immutable 35-leaf request, a
separate effective snapshot, explicit resolution reasons, and reset realized
state. Effective map filtering and source finding are suppressed only when
mapmaking is unavailable; pointing and Beammap source fitting remains required
whenever mapmaking is available, independent of optional filtering. The plan
is constructed once during config loading and the legacy state is still
compared against its request. Production filtering, finding, fitting, and
output consumers have not been switched yet, so this checkpoint changes no
numerical or output behavior. Focused plan and frozen-boundary tests, all 347
CTest cases, all eight compatibility profiles, and full config preflight pass.
The next bounded cutover at that checkpoint was the one-way typed map-filter
adapter, followed by source finding and source fitting; accepted `redu53` is
the validation baseline after a consumer cutover, not for plan construction
alone.

The map-filter consumer cutover is complete and accepted. The duplicate serial and
OpenMP Wiener YAML parsers and the reverse Wiener-to-typed mirror are removed.
A single one-way adapter copies the effective typed filter snapshot into the
mature numerical target while preserving conditional Gaussian/Airy FWHM
loading and arcsecond-to-radian conversion. Filter activation, runtime noise/
kernel dependency checks, required filtered-output policy, and map-diagnostic
edge-guard metadata now consume effective typed policy. The Wiener algorithms,
map arrays, and output ordering are unchanged. The frozen audit rejects parser,
reverse-mirror, output-policy, or adapter drift. Local CLI/test builds, all 347
CTest cases, 60 config tests, all eight compatibility profiles, and full
preflight pass. Unity point `redu54` at `a89e0ee5` reruns the unchanged
enabled-filtering overlay with zero serious log issues, all required provenance
valid, and the same 21-product inventory as `redu53`. Its 490-leaf merged
low-level config is byte-identical to `redu53`; all 2,041 compared records pass
the established tolerance with no skips, and all 639 non-PTC records compared
against matching OG `redu09` pass as well. The 16 non-bitwise records are
confined to three filtered a1400 products, have no finite-mask mismatch, and
have maximum absolute difference `8.73e-11`.

The source-finding consumer cutover is complete locally. Its duplicate YAML
parser and observation-to-coadd reverse mirror are removed. One adapter writes
`source_sigma`, the arcsecond-to-radian source window, and finder mode directly
from the effective typed plan to the observation map buffer and, when enabled,
the coadd map buffer. Source-finding execution and output activation now use
the same effective authority. The legacy shadow retains activation parity but
no longer compares details that legacy state does not own. Detection, fitting,
map arrays, source tables, and output order are unchanged. Both local targets
build, all 349 CTests and 61 config-boundary tests pass, all eight compatibility
profiles pass, and full preflight is clean. Unity point `redu55` at `aa593a2b`
closes this gate with zero serious log issues, all required provenance valid,
and bit-for-bit identity across all 2,041 records in the 21 common products
against `redu54`, including full RTC/PTC timestreams, 195 source rows, and both
pointing tables. The 490-leaf config is byte-identical to `redu54`; all 639
non-PTC records also pass against matching OG `redu09`. Source fitting is now
the active bounded consumer cutover.

The source-fitting consumer cutover is complete and accepted. The mixed
YAML-to-`mapFitter` parser is removed. A standalone
one-way adapter now projects the effective typed fitting request into the
mature fitter target, preserving arcsecond-to-pixel conversion, fit-angle
policy, two-element amplitude/FWHM vectors, and the historical rule that a
nonpositive limit factor retains the fitter's established default. The
Gaussian fitting implementation and its numerical inputs are otherwise
unchanged. Source-fitting details are no longer copied into or compared
against legacy config state; the temporary legacy shadow now covers only the
remaining activation and histogram values it actually owns. Both local
targets build, all 350 CTests and 62 config-boundary tests pass, all eight
compatibility profiles pass, and full preflight is clean. Unity point `redu56`
at `9f8ad50e` closes the gate with zero serious log issues, all required
provenance valid, the same 50-file inventory, byte-identical 490-leaf merged
config, and bit-for-bit identity across all 2,041 records in the 21 common
products against `redu55`, including the 195-row source table and complete
RTC/PTC timestreams. Realized post-processing state and required provenance are
now the active bounded work.

The realized post-processing implementation is complete and accepted for the
point workflow. Per-iteration state records observation and coadd filter
contexts and map counts; source-finding contexts, detected candidates, catalog
fit attempts/valid fits, and successfully written source-table rows; raw and
filtered pointing fit contexts; and Beammap fit contexts. These fitter families
remain separate by project-owner decision rather than being collapsed into one
ambiguous total. Completion rejects missing or inconsistent cardinality and is
cross-checked against completed mapmaking. Source finding without map filtering
is now a fail-fast configuration error because the supported execution path
operates only on filtered maps.

The CLI publishes required atomic `post_processing_provenance.yaml` using
`citlali-post-processing-provenance-v1` only after successful pipeline output
and realized-state completion; write or lifecycle failures fail the reduction.
The reduction auditor validates internal cardinality and activation semantics,
cross-checks filter map counts with mapmaking v2, and cross-checks raw/filtered
pointing fit totals with pointing v2. The frozen source-boundary audit requires
the lifecycle hooks, schema, atomic writer, and single CLI completion/write
calls. Local `citlali_cli` and `citlali_test` builds pass, all 357 CTests pass,
43 reduction-auditor tests pass, all eight compact profiles pass, and full
config preflight is clean. No filter, source-detection, Gaussian-fit, or map
numerical algorithm was changed.

Unity point `redu57` at `f8a4a596` closes the point gate. It has zero serious
log issues, a valid required sidecar with one observation filter/source/table
context, 195 source rows, and separate three-map raw/filtered pointing fit
contexts. Its 490-leaf merged config is byte-identical to `redu56`, and all
2,041 records in the 21 common products, including full RTC/PTC timestreams,
are exact. Science must still exercise coadd-only filtering/source routing and
Beammap must exercise iterative detector-fit cardinality. Those expensive mode
gates are intentionally batched until after the remaining activation-only
legacy shadow is retired locally; the domain is not complete until both pass.

The activation-only compatibility shadow is now retired locally. The complete
typed post-processing request is loaded once before mapmaking setup, owns the
histogram setting that map buffers consume through the existing one-way
adapter, and initializes the effective execution plan without a second YAML
activation pass. Disabling mapmaking no longer mutates requested filtering,
finding, or fitting policy; effective suppression remains the execution plan's
responsibility. The established no-map Beammap single-iteration optimization
is preserved separately. Both local targets build, all 355 CTests and 63
config-boundary tests pass, all eight compact profiles pass, and full preflight
is clean. This cleanup still requires a point run after Unity compilation; it
is not covered by the preceding `redu57` acceptance. Unity point `redu58` now
closes that gate with the same config and post-processing provenance hashes,
zero serious log issues, and exact identity across all 2,041 records in the 21
common products, including full RTC/PTC timestreams.

The next Beammap authority domain is characterized without changing runtime
behavior. A versioned manifest freezes all 74 `beammap.*` leaves; there are no
known typed-model gaps, and config literals remain confined to the declared
loading and validation boundary. One typed-to-legacy adapter copies only the
fit support radius into the mature `map_fitter`. Dedicated requested/effective/
realized Beammap provenance is explicitly missing. The six-test static audit
is part of the full preflight and will reject surface, reader-boundary,
authority, or adapter drift.

The final post-processing mode gates are accepted. Science final iteration
`redu19` at `342a021c` has zero serious log records and valid required
provenance. Its realized record contains no observation filter contexts and
exactly one coadd filter context with three filtered maps. Against accepted
science `redu15`, the low-level config is byte-identical and the strict full-
depth comparison finds 27 common products, no missing or extra products, no
skipped records, and no changed records outside the standard tolerance.

Beammap `redu02` at the same commit has zero serious log records and valid
required provenance. Its realized record contains exactly three detector-fit
contexts with 15,407 attempts and 15,407 valid fits. Against accepted Beammap
`redu01`, the low-level config is byte-identical and the strict full-depth
comparison, including complete detector TOD and split FITS maps, finds 12
common products with no missing, extra, skipped, or changed records. The
profiling sidecar differs only in elapsed timing and is excluded from the
scientific gate. Post-processing authority and provenance are complete.

Project-owner decision (2026-07-10): every output explicitly enabled in the
configuration is required. RTC TOD, PTC TOD, `rtcdiag`, and `ptcdiag` write
failures must fail the reduction. There are no best-effort enabled products.

Immediate work order:

1. Begin the bounded Beammap effective-plan and provenance migration using the
   accepted sequence in the Beammap authority review.
2. Ask only the owner questions needed by the first Beammap implementation
   cut; do not silently change phase, prior, split, reference, or source-flux
   behavior.
3. Preserve Gaussian fitting, prior matching, detector flagging, RTC/PTC,
   mapmaking, and all other mature numerical algorithms.
4. Keep compact-config rollout, polarimetry expansion, and Phase 3 compiled-
   boundary work paused.

### Parallel Review Synthesis - 2026-07-14

Three read-only reviews were completed and adopted as advisory detail under
this living roadmap:

- [Phase 2 completion census](../handoff/PHASE2_COMPLETION_CENSUS_2026-07-14.md)
- [Beammap authority design review](../handoff/BEAMMAP_AUTHORITY_DESIGN_REVIEW_2026-07-14.md)
- [compact configuration and TolTECA usability review](../handoff/CONFIG_USABILITY_TOLTECA_REVIEW_2026-07-14.md)

They agree with the active sequence and expose no reason to reopen the nine
completed authority domains. Phase 2 remains incomplete: Beammap and the
minimal KIDs external boundary are implementation-ready; polarimetry and atomic
astrometry/photometry still require scientific-policy decisions. Domain-level
completion must not be mistaken for the global Phase 2 exit gate.

After the post-processing gates close, the adopted shortest sequence is:

1. Complete the bounded Beammap effective-plan and provenance migration,
   preserving all mature numerical algorithms.
2. Complete atomic Beammap photometry observation configuration, including
   replacement rather than merging of per-observation calibrator flux. Keep
   source identity in telescope data and leave flux estimation to TolProj.
3. Record the minimal external KIDs schema/config identity and the durable
   ordered configuration-source manifest.
4. Mechanically disposition polarimetry as either supported and validated or
   rejected as an unavailable capability.
5. Run current matched point, OOF, Beammap, and science snapshots on the final
   Phase 2 candidate before beginning Phase 3.

The frozen 74-leaf `beammap.*` manifest is the correct Beammap policy boundary,
not a claim to contain every scientific input used by a Beammap reduction.
`beammap_source.fluxes` remains an adjacent photometry input. The
review identified a concrete stale-state risk there: a later observation can
inherit a per-array source flux omitted from its own input. The Beammap work
must therefore reference an atomically constructed observation photometry value;
it must not absorb that adjacent domain or preserve merge semantics.

For Phase 2, "reviewed overlay fixtures" means retained matched low-level mode
overlays plus durable ordered-source evidence. Compact-config production
deployment and its full hermetic TolTECA numbered-overlay acceptance suite are
explicitly deferred rollout blockers, not Phase 2 exit requirements. Current
`*_standard` compact profiles remain translation prototypes and must not be
presented as approved operational defaults. Normal compact controls must also
be audited in both directions: user-facing low-level paths must be reachable,
and ordinary compact fields must not write expert-only policy.

Open scientific and operational choices listed in the reviews will be asked
only when the next implementation depends on them. They must not be inferred
silently. In particular, Beammap source-flux failure behavior, phase/prior/
split/reference fallbacks, HWPR and polarimetry capability, astrometry frame
and time rules, supported KIDs types, and ownership of ordered TolTECA source
provenance remain owner decisions.

### Phase 1 Progress

- The 12 `NetCDF: Not a valid ID` errors in `redu21`/`redu22` were traced to
  the PTC TOD stream, one error per requested output scan. The schema omitted
  four second-pass rejection/source-protection variables that the append path
  wrote unconditionally. Signal, flags, weights, and earlier diagnostics had
  already been written before each exception, which explains why pairwise
  numeric comparison passed despite incomplete diagnostics.
- The PTC TOD schema now includes all four fields. A focused NetCDF schema test
  creates the file layout and checks their presence. Local `citlali_cli` build
  and `citlali::safety::ptc_tod_schema.includes_all_second_pass_summary_fields`
  pass. Unity reduction validation is pending.
- CTest is now enabled at the project boundary and the focused safety target is
  discoverable from the normal top-level build directory.
- Parsed enum failures now enter the authoritative invalid-key diagnostics
  instead of silently retaining their typed default. Legacy authoritative
  range parsing and typed validation reject NaN and infinity for ordinary
  numeric fields. The four documented line-frequency inheritance fields retain
  their explicit NaN sentinel but reject either infinity. Focused parser and
  finite-value tests pass locally.
- Required RTC TOD, PTC TOD, `rtcdiag`, and `ptcdiag` NetCDF failures now retain
  the failing path in an error diagnostic and propagate out of the reduction.
  Ordered writers cancel as one output domain, so a failure wakes workers
  waiting on the same or another product stream instead of deadlocking. Focused
  serialization, cancellation, and cross-stream cancellation tests pass.
- A real fixed-size NetCDF failure test now writes the first row, injects an
  out-of-range second write, verifies that a waiting third writer is cancelled
  and the partial product is explicit, confirms a nonzero CLI result, then
  recreates and completes the product with a fresh writer domain in the same
  process.
- The owner-thread failure state now lets Pointing, Lali, and Beammap rethrow
  required output failures after GrPPI worker drainage, so the normal CLI error
  boundary can report them without an exception escaping a worker thread.
- Disabled IIR and extinction mirrors now preserve legacy effective provenance:
  IIR uses frequency `0`/order `1`/zero-phase `false`, and extinction uses
  `N/A`. Enabled values are unchanged. Four focused mirror tests pass.
- Reduction audit comparison now treats any error-level log record as blocking;
  `redu22` correctly fails the audit with 12 errors while the clean `redu23` to
  `redu24` comparison passes.
- Reduction product comparison now has an explicit strict mode. It fails on
  product-set differences, skipped items, or changed records. A complete TOD
  comparison of `redu23` and `redu24` passes with zero changes/skips when the
  volatile profile sidecar is explicitly excluded; retaining that sidecar or
  the default large-array cap correctly fails the gate.
- The pre-existing `citlali_test` target was found to have substantial test
  infrastructure and source decay. It has now been decoupled from the obsolete
  Google Benchmark runner, modernized for typed config and explicit alignment
  and output-path ownership, and reactivated with all 201 declared legacy tests
  passing. The seven utility tests that had remained inside a block comment now
  exercise the current Tula APIs with assertions. Together with the 18 focused
  safety tests, CTest discovers and passes 219 tests with none skipped or
  disabled. The local CLI build and complete config preflight continue to pass.
- Enabled timestream products now carry mode- and config-derived expected write
  counts. Pointing, Lali, and Beammap verify RTC TOD, PTC TOD, `rtcdiag`, and
  `ptcdiag` cardinality after worker drainage and before map finalization, so a
  silently omitted required chunk fails even when no individual write throws.
- Main timestream scan generators now own their cursors per pipeline invocation
  instead of sharing function-local static counters. Focused tests prove exact
  enumeration and a clean scan-zero start after an earlier cursor is abandoned.
- `redu25` (`c2ec8ae5`) finished with zero serious log issues and the same
  complete 33-file/14 stable-product inventory as `redu24`. Scientific arrays,
  maps, and tables are exact. The only strict-comparison differences are the
  intended disabled-IIR effective-provenance changes in RTC/PTC metadata.
- Beammap detector-specific TOD now obeys the required-output policy. Config
  preflight rejects enabled output with no slots or non-detector map grouping;
  unavailable scans, PTC samples, or pointing fail at runtime instead of
  silently skipping the declared product.
- Enabled learning diagnostics now fail on open, write, flush, or close errors.
  Required Beammap PTC TOD metadata updates likewise fail when the file or
  `FRUITLOOPS_ITER` variable is unavailable.
- ECSV table output is now published atomically through a temporary file.
  Failure removes the temporary product and propagates instead of silently
  substituting a differently named ASCII table.
- `validation/accepted_runs.json` is the checked-in machine-readable validation
  ledger. Its first record captures the accepted `redu25` point checkpoint,
  including explicit unavailable provenance and the two intended metadata
  differences. A standard-library validator enforces its core consistency
  rules.
- `redu26` validates the full current Phase 1 checkpoint at `9ef7da8a`. It has
  zero serious log records, the same complete 33-file product inventory and
  merged-config hash as `redu25`, and zero changed or skipped records in the
  strict comparison including every TOD array. Total logged runtime was 59.25
  seconds versus 61.51 seconds for `redu25`; this is recorded as run variation,
  not a performance conclusion.
- Phase 2 preparation now has a checked authority inventory covering 13 config
  domains. It enforces the one-way requested-YAML to typed-config to legacy
  adapter contract and records a concrete exit gate for each domain. Seven
  domains remain materially mixed, four are typed-authoritative without an
  adapter, Beammap is typed-authoritative with one fitting adapter, and KIDs is
  an explicit external boundary. This checkpoint changes no runtime behavior;
  operational authority migration remains gated on the remaining Phase 1
  validation decisions.
- Phase 1 science validation at refactor `redu12` (`59c35e60`) completed both
  observations with 248 PTC chunks, zero logged issues, and the expected 25
  stable products. Against same-config refactor `redu10` (`9ef7da8a`), all 24
  compared FITS/NetCDF products have zero changed or skipped records. Against
  deterministic OG science `redu15`, all nine FITS products remain within the
  current tolerance, while 30 RTC/PTC diagnostic records differ under the
  generic pointwise comparator. Scientific-owner review accepted those
  differences on 2026-07-11: all integer diagnostics are exact, map RMS drift
  is at most `2.31e-11`, PTC weight RMS drift is `2.14e-12`, and the largest
  near-zero detector-median difference is `2.85e-5` absolute and `2.42e-4`
  fractional. The versioned `science-scientific-equivalence-v1` gate enforces
  the accepted bounds and the validation ledger records the checkpoint.
- The intervening science `redu11` failed after observation 0 when observation
  1 metadata loading raised an unqualified NetCDF `No such file or directory`.
  Its merged config was identical to the successful runs. Metadata-load
  failures now report observation index, name, and telescope filepath; all 220
  local tests pass. The successful `redu12` shows this was not a persistent
  numerical or lifecycle failure.
- Beammap refactor `redu10` (`f278bd32`) and `redu11` (`9ef7da8a`) use identical
  merged configs and are numerically repeatable: all six large split FITS
  products, both APT tables, RTC/PTC diagnostics, and the complete detector-TOD
  `signal`/`flags` arrays have zero changed records. The matched OG Beammap pair
  is also deterministic. Scientific-owner review accepted the bounded OG to
  refactor differences on 2026-07-11: detector identities and flags are exact;
  the worst good-detector signal and weight RMS-relative differences are
  0.625% and 0.308%; sensitivity differs by at most 0.255%; and positional and
  FWHM differences are sub-microarcsecond. The versioned
  `beammap-scientific-equivalence-v1` gate now enforces these limits and the
  validation ledger records the accepted checkpoint. Any future threshold
  breach is numerical creep and requires investigation rather than automatic
  tolerance relaxation.

The first Beammap authority preparation checkpoint is complete locally without
changing production execution. Mechanical boundary checks expand 59 typed
reader roots and 59 serializer roots to exact 74/74 frozen-path coverage. A
pure, production-unwired `BeammapExecutionPlan` preserves requested values and
separately characterizes current phase correction, prior inheritance and
missing-path disablement, split-flag normalization, convergence availability,
and mapmaking-disabled iteration policy. Cold-boundary validation now rejects
non-finite Beammap vector and scalar values and enforces reader-established
vector cardinality. The existing typed request and one-way fitting adapter
remain production authority, and dedicated Beammap provenance remains missing;
this is preparation for a later bounded consumer cutover, not a completed
migration claim.
The local CLI and test targets build, all 363 CTests pass, and full config
preflight passes 74 boundary tests, all eight compatibility profiles, and the
complete authority audit suite. Because the plan and serializer are explicitly
unwired, this checkpoint does not require a Unity reduction.

## Beammap Effective-Plan Boundary Activated

The next bounded checkpoint constructs `BeammapExecutionPlan` in production
from one raw 74-leaf request plus explicit-key presence. Policy correction no
longer mutates values inside the family YAML readers. The immutable request is
preserved while a separate effective snapshot records phase correction, prior
inheritance and missing-path disablement, split-flag normalization, convergence
availability, and mapmaking-disabled iteration behavior.

Existing mature Beammap algorithms temporarily consume a one-way copy of the
effective snapshot through `ReductionConfig::beammap`, preserving their current
inputs without creating reverse synchronization. The existing map-fitter
radius adapter is the first bounded consumer to read effective plan policy
directly. The boundary audit enforces the ordered read/resolve/install/adapt
sequence and rejects reintroduction of the retired reader-side mutation
helpers. Dedicated Beammap realized lifecycle and provenance remain missing,
and the component serializer remains unpublished.

Local verification is clean: `citlali_cli` and `citlali_test` build, all 364
CTest cases pass, and full config preflight passes 74 tests, all eight compact
compatibility profiles, 100% compact-surface coverage, and every authority
audit. This changes production configuration construction, so the eventual
Beammap provenance checkpoint requires a Unity compile and matched Beammap
reduction before the domain can be accepted. The next local work is realized
iteration/output state and required atomic provenance; do not spend a Beammap
run on this intermediate commit alone.

## Beammap Realized Lifecycle And Provenance Prepared

The next local checkpoint adds an explicit Beammap observation and internal-
iteration lifecycle around the established execution without changing its
numerical control flow. Each enabled-mapmaking observation records identity,
detector/map/scan counts, contiguous iteration indices and phases, active map
counts, one or two completed mapmaking passes, the source-aware RTC decision,
fit completion, newly/total converged maps, and maximum-iteration or all-maps-
converged termination. Disabled mapmaking records a successful zero-product
execution instead of manufacturing observations or fit contexts.

Completion requires every internal stage and observation output to finish. It
then cross-checks Beammap observation identity/map counts against the completed
mapmaking plan and requires the post-processing Beammap fit-context count to
equal the exact number of completed internal iterations. Map write counts and
fit attempt/valid aggregates remain owned by their existing plans rather than
being copied into Beammap state.

Successful Beammap reductions now require atomically published
`beammap_provenance.yaml` with schema `citlali-beammap-provenance-v1`. The file
contains the complete requested and effective 74-leaf snapshots, effective-
resolution reasons, observation/iteration lifecycle, and terminal realized
state. Incomplete lifecycle and publication failures propagate to the CLI.
The strengthened boundary audit requires all lifecycle hooks, exact 74/74
reader and config-serializer coverage, and one ordered CLI completion/write
path.

Local verification is clean: both build targets pass, all 372 CTests pass,
and full preflight passes 75 Python tests, all eight compatibility profiles,
100% compact-surface coverage, and every authority audit. The authority
inventory deliberately remains `partial` until a matched Unity Beammap run
accepts this sidecar and scientific products. Observation-resolved prior and
reference decisions, adjacent atomic `beammap_source.fluxes` state, and any
additional Beammap-specific optional-product cardinalities required by the
design review remain bounded follow-up work; this checkpoint does not claim
the Beammap domain complete.

Enabled detector-specific Beammap PTC TOD is now an explicit required
observation product in the realized plan. The record is updated only after the
existing atomic NetCDF writer returns and captures the output iteration plus
detector, slot, and maximum-sample dimensions. Observation completion requires
exactly one such write when `beammap.detector_tod_output.enabled=true`, rejects
duplicates, and requires zero writes when disabled. This implements the
project-wide enabled-output decision without selecting new scan slots or
changing the detector-TOD numerical content.

The CLI and test targets build, all 373 CTests pass, and full preflight remains
clean with 75 Python tests and all eight compatibility profiles. This is part
of the pending Beammap Unity validation candidate, not a separately accepted
domain gate. Prior/reference and split-output fallback policies remain
unchanged and unresolved owner decisions are not inferred.

## Beammap Lifecycle Gate Accepted

Unity Beammap `redu03` was produced by `v4.0.0-3486-gb530e838` from the same
low-level configuration as accepted `redu02` (SHA-256
`aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d`).
The run completed 198 PTC chunks in 3,609.307 seconds with zero error-,
critical-, or fatal-level log records. The required
`citlali-beammap-provenance-v1` sidecar records one 5,234-detector/map
observation, three contiguous completed Beammap iterations, one mapmaking pass
per iteration, the expected source-aware RTC rerun on iteration one,
maximum-iteration termination, and exactly one required detector-TOD write at
iteration two with shape 5,234 detectors by 20 slots and 788 maximum samples.

Against `redu02`, the merged configuration is byte-identical. The accepted
Beammap profile reports exact detector identity, flags, APT quantities, and
all good/bad signal, weight, and kernel maps. The strict full-depth comparison
excludes only volatile `citlali_profile.ecsv` timing, reads all 12 scientific
products including detector TOD and six split FITS files, and finds no missing,
extra, skipped, or changed records.

The standard reduction audit now recognizes and can require Beammap provenance.
It validates observation/iteration lifecycle, terminal state, convergence
accounting, detector-TOD cardinality and shape, and cross-checks observation
identity/map count against mapmaking plus iteration count against
post-processing fit contexts. This closes the pending lifecycle/provenance
validation checkpoint, but the Beammap authority domain remains partial until
observation-resolved prior/reference state and adjacent atomic
`beammap_source.*` handling are completed. No unresolved fallback policy is
inferred by this gate.

## Atomic Beammap Photometry State Accepted

The adjacent photometry safety cut removes the concrete
cross-observation source-flux hazard without changing successful numerical
behavior. `beammap_source.*` is parsed into a temporary observation value and
all required runtime-array fluxes are validated before any Engine state is
mutated. Successful installation replaces typed photometry and the legacy
mJy/beam map and clears the derived MJy/sr map; it never merges with an
earlier observation. Missing or invalid required flux retains the established
fatal reduction outcome, but now throws a typed invalid-config error instead
of calling `exit()` inside `Engine::get_photometry_config`.

Project-owner clarification (2026-07-15): source identity belongs to telescope
data and TolProj owns calibrator selection and flux estimation. Citlali must
not mirror source name or coordinates into this config domain. Beammap
provenance therefore advances to `citlali-beammap-provenance-v2` with
`telescope_data` named as the source-identity authority and only the installed
per-array flux/uncertainty recorded as Citlali photometry input. The reduction
audit accepts historical v1 sidecars and requires this ownership record for
v2.

Project-owner decision (2026-07-15): every runtime array requires a positive,
finite calibrator flux; missing or invalid required flux fails the reduction.
No fallback is permitted.

Unity Beammap `redu04` was produced by `v4.0.0-3489-g7e577c81` from the same
byte-identical low-level config as accepted `redu03` (SHA-256
`aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d`).
The run completed all 198 PTC chunks with zero error-level messages. Its valid
`citlali-beammap-provenance-v2` sidecar names telescope data and TolProj as the
respective source-identity and calibrator-flux authorities and records the
three required installed array fluxes. The strict full-depth comparison reads
all 12 scientific products, including detector TOD and six split FITS files,
and finds no missing, extra, skipped, or changed records. The dedicated
Beammap profile also reports exact detector identity, flags, APT quantities,
and good/bad signal, weight, and kernel maps.

The total log interval is 3,661.793 seconds versus 3,609.307 seconds for
`redu03` (+1.45%). The dominant mapmaking interval is 0.53% faster; the
variation is concentrated in PTC chunk and diagnostics timing. This is within
the provisional 3-5% runtime budget and does not indicate a provenance-path
regression. Peak RSS remains unmeasured.

Both local targets build; all 24 focused Beammap/photometry tests, all 377
CTests, and all 49 reduction-audit tests pass. Full config preflight passes 75
tests, all eight compatibility profiles, 100% compact coverage, and every
authority audit.

## External KIDs And Config-Source Provenance Prepared

The bounded external KIDs checkpoint preserves Kidscpp as the numerical
execution authority while recording the exact bridge identity Citlali uses.
All four solved TOD representations (`xs`, `rs`, `is`, and `qs`) are supported.
The requested fitter/solver values, effective values, selected TOD type,
TolTEC data schema, and Kidscpp build version are separate fields in the
required atomic `citlali-kids-external-provenance-v1` sidecar. Historical
`solver.extra_output` behavior remains disabled and is now recorded explicitly
instead of being controlled by a header-level global.

The same successful CLI boundary now requires
`citlali-config-source-manifest-v1`. It records the ordered files actually
passed to Citlali, collision-safe copies, byte sizes, SHA-256 digests, and the
canonical merged YAML snapshot. TolTECA remains the owner of numbered
`NN*.yaml` discovery and upstream merge provenance; the record explicitly says
that TolTECA's complete ordered authoring-source list is not currently passed
to Citlali. Citlali does not guess or duplicate that merge.

Local CLI and test builds, all 382 CTests, 52 reduction-audit tests, and the
full 78-test config preflight pass. Unity point `redu59` identifies
`d016e1a64`, has zero serious log records, and passes semantic and digest
audits for both new records. Its low-level config is byte-identical to accepted
`redu58`; the strict full-depth comparison reads all 21 scientific products,
including complete RTC/PTC timestreams, with zero changed, skipped, missing, or
extra records. The external KIDs and Citlali CLI config-source checkpoint is
accepted. Complete upstream `NN*.yaml` provenance remains a future TolTECA
interface responsibility rather than a Citlali reconstruction task.

## Polarimetry Capability Disposition Accepted

The project owner intends Citlali to become the center of polarimetry
reductions, but not in the present refactor and not without an enabled
validation dataset. Phase 2 therefore preserves polarimetry as a planned
capability while mechanically rejecting `timestream.polarimetry.enabled: true`
before reduction execution. The exit condition is an approved polarimetry/HWPR
scientific contract plus an enabled end-to-end reference gate.

The frozen three-leaf request now has one direct typed reader, one immutable
request/effective capability plan, and one forward adapter into `RTCProc` and
`Calib`. The temporary legacy compatibility reader and reverse mirror are
removed. There is no separate `calibration.ignore_hwpr` YAML input; that name
was stale inventory text referring to the legacy adapter target. Disabled
reductions retain Stokes-I initialization and the established default values.

Successful reductions now require atomic
`citlali-polarimetry-provenance-v1`, recording the capability disposition,
requested/effective policy, accepted resolution, and realized non-execution.
The dedicated static audit freezes the boundary, while the reduction auditor
semantically rejects enabled or executed polarimetry in a successful run.
Local CLI and test builds, all 386 CTests, 54 reduction-audit tests, and the
full 82-test config preflight pass.

Unity point `redu60` identifies `db22bca1f`, completes all 12 PTC chunks in a
67.032-second total log interval, and has zero error-, critical-, or fatal-level
records. Its required v1 sidecar records the planned-unavailable capability,
an accepted disabled request, a disabled effective plan, completed reduction,
and no polarimetry or HWPR execution. The low-level input is byte-identical to
accepted `redu59`; the strict zero-tolerance comparison reads all 21 stable
scientific products, including complete RTC/PTC timestreams, with no changed,
skipped, missing, or extra records. The disabled capability boundary is
accepted. Enabled polarimetry remains planned but unavailable until its
scientific/HWPR contract and enabled reference gate are approved.

## Observation-Resolved Astrometry Candidate

The astrometry calibration-item loader now constructs the complete typed
pointing-offset request before touching observation runtime state. Structural
and finite-value validation runs on that temporary value, and a single forward
adapter then replaces both the typed request and the legacy Eigen vectors.
Invalid input throws the normal typed invalid-config error; the loader no
longer calls `exit()` or builds typed policy by mirroring partially mutated
runtime state. Legacy named axes, positional axes, one/two-value shapes, and
non-positive MJD sentinel normalization are preserved. The interpolation
kernel and its existing no-extrapolation behavior are unchanged. The remaining
interpolation failures now propagate as typed exceptions rather than terminating
the process from library code; successful numerical behavior is unchanged.

The project owner approved the legacy application contract. TolTECA selects
pointing support: two bracketing pointing observations produce interpolated
offsets, one pointing produces constant offsets, and no pointing observations
leave the explicitly configured offsets in force. Citlali applies the supplied
values. Positive MJD endpoints must remain strictly increasing, bracket the
whole observation, and are never extrapolated. Citlali does not receive the
upstream support-selection metadata, so it records that origin as unspecified
rather than inferring whether a constant came from one pointing or direct
configuration.

An observation-indexed execution plan now retains each immutable request,
effective application mode (`constant`, `observation-span-linear`, or
`explicit-mjd-linear`), observation number, installation/application counts,
and telescope sample count. Successful CLI completion requires atomic
`citlali-astrometry-provenance-v1`. Its authority record names TolTECA for
calibration selection and Citlali for application. A semantic reduction audit
and a static config-boundary audit reject incomplete lifecycle, malformed
offsets, inconsistent modes, authority drift, reverse mirrors, process exits,
or a missing required write.

The CLI and test targets build; all 398 CTests, 60 reduction-audit tests, and the
full 84-test config preflight pass. The combined astrometry/photometry domain is
still marked partial until Unity validates the new required sidecar and
scientific equivalence. The next gate should include a point reduction, then a
multi-observation OOF reduction because that fixture exercises observation
identity and stale-state isolation most directly. Beammap should follow before
the combined domain is marked complete.

## Astrometry Point Gate Accepted

Unity point `redu61` was produced by `v4.0.0-3496-g9ea6d7f0` from the same
byte-identical low-level and canonical merged configuration as accepted
`redu60`. It completed all 12 PTC chunks in a 63.741-second total log interval
with zero error-, critical-, or fatal-level records. Every applicable required
provenance record passes semantic audit.

The new `citlali-astrometry-provenance-v1` sidecar records TolTECA as calibration-
selection authority and Citlali as application authority without claiming
unavailable support-origin metadata. Observation 152389 has one requested and
effective zero-valued az/alt correction, constant application mode, one atomic
installation, one application, and 7,697 telescope samples. The reduction is
complete.

The strict zero-tolerance comparison against `redu60` reads all 21 scientific
products and 2,041 records, including complete RTC/PTC timestreams, with zero
changed, skipped, missing, or extra records. The point checkpoint is accepted.
The combined astrometry/photometry domain remains partial until a multi-
observation OOF run validates observation identity and stale-state isolation,
followed by a Beammap run validating the adjacent accepted photometry contract.

## Astrometry Multi-Observation OOF Gate Accepted

Unity OOF `redu02` was produced by `v4.0.0-3496-g9ea6d7f0` from the same byte-
identical low-level configuration as accepted refactor `redu01`. It completed
all 18 PTC chunks for observations 152385-152387 in a 40.667-second total log
interval with zero error-, critical-, or fatal-level records. All applicable
required provenance records pass semantic audit.

The astrometry sidecar contains three contiguous observation identities. Each
was installed and applied twice, once during initial geometry and once during
the reduction iteration, with stable per-observation telescope sample counts.
This closes the multi-observation replacement and stale-state-isolation gate.
TolTECA supplied a constant zero-offset request for each observation, so this
fixture does not provide an end-to-end positive-MJD interpolation test; that
limitation is retained explicitly rather than overstating the evidence.

The strict zero-tolerance comparison against accepted refactor `redu01` reads
all 30 configured products and 1,941 records with zero changed, skipped,
missing, or extra records. Direct comparison against OG `redu00` reproduces the
same nine previously accepted inactive RTC-despike metadata differences; all
scientific numeric differences remain within the standard OOF tolerance. The
OOF checkpoint is accepted. Beammap remains the combined astrometry/photometry
gate, and science remains required for the final Phase 2 snapshot matrix.

## Astrometry Science Interpolation Gate Accepted

Unity science `redu20` through `redu23` was produced by
`v4.0.0-3496-g9ea6d7f0` from the same byte-identical low-level configuration as
accepted `redu16` through `redu19`. Final `redu23` completed 248 PTC chunks in a
711.330-second total log interval with zero error-, critical-, or fatal-level
records. Every science-applicable required provenance record passes semantic
audit.

The astrometry sidecar records observations 152390 and 152392 with distinct,
strictly increasing positive-MJD support pairs and `explicit-mjd-linear`
effective mode. Each observation was installed and applied five times, once
during initial geometry and once in each of four fruit-loop iterations, with
stable telescope sample counts of 151,535 and 151,941. Successful completion
also proves that each support pair bracketed its complete telescope timestream;
the unchanged application kernel forbids extrapolation.

Every retained fruit-loop iteration passes the standard strict science gate:
`redu16`-`redu19` versus `redu20`-`redu23` each has 27 common products and 1,478
comparison records, with zero missing, extra, skipped, or out-of-tolerance
records at `2e-8 + 1e-10 * abs(reference)`. A zero-tolerance probe sees only the
expected tiny OMP run-to-run drift. The science and explicit-MJD interpolation
checkpoint is accepted. Beammap is the final mode gate for the combined
astrometry/photometry authority domain.

## Astrometry And Photometry Beammap Gate Accepted

Unity Beammap `redu05` was produced by `v4.0.0-3496-g9ea6d7f0` from the same
byte-identical low-level configuration as accepted `redu04`. It completed all
198 PTC chunks with zero error-, critical-, or fatal-level records. Every
Beammap-applicable required provenance record passes semantic audit.

The version-two Beammap provenance is identical to `redu04`: one 5,234-map
observation, three completed iterations, 15,407 valid detector fits, exact
telescope-data source identity and TolProj flux authority, three atomically
installed array fluxes, and one required 5,234-detector by 20-slot TOD write.
The added astrometry record captures one constant zero-offset application over
383,699 telescope samples without changing that accepted photometry contract.

The zero-tolerance full-depth comparison reads all 12 products and 16,453
records, including complete detector TOD and six split FITS cubes, with zero
changed, skipped, missing, or extra records. The dedicated Beammap scientific-
equivalence profile reports exact detector identities, flags, APT quantities,
and signal/weight/kernel maps for all 4,980 good and 254 bad detectors.

The 4,136.440-second total interval is 13.0% slower than `redu04`, but the
dominant mapmaking interval is 1.3% faster. The increase is concentrated in PTC
and diagnostics I/O before mapmaking, outside the astrometry change. Record the
variation without attributing it or treating one uncontrolled Unity comparison
as a performance conclusion; controlled performance/RSS certification remains
Phase 4 work.

The combined astrometry/photometry authority domain is complete. All 13 domains
in the original operational migration matrix have complete migration and
provenance disposition. The global F.1 leaf census and document/ledger
reconciliation remain before changing the active phase to Phase 3.

## F.1 Leaf Census Checkpoint

The owner approved the generated low-level Citlali YAML as Citlali's immutable
configuration/provenance boundary. TolTECA owns discovery, ordering, and merge
semantics for upstream `NN*.yaml` authoring files and must eventually record
that upstream provenance; Citlali records exact source bytes and ordered paths
from the generated low-level input onward. This is an explicit boundary
decision, not an inference that Citlali received unavailable source metadata.

The checked F.1 leaf contract resolves the union of `data/config.yaml` and the
four retained point, OOF, Beammap, and science low-level fixtures. It records
573 unique leaves, including 572 executable leaves and one explicitly ignored
deprecated leaf. Every record has a machine-readable authority, typed or
external owner, unit, allowed value-domain class, mode applicability, lifecycle
classification, resolution stage, and validation source. The preflight fails
on an uncovered leaf or drift from the resolved manifest.

This census exposed two real closeout omissions hidden by the earlier broad
subsystem grouping: 28 `timestream.learning` leaves executed from a legacy
options object populated in parallel with the typed request, and 14
`interface_sync_offset` leaves executed from an untyped mutable map with
permissive duplicate handling. They are now explicit `learning` and
`interface-sync` authority domains. Both are locally migrated through immutable
typed request, one-way adapter, validation, and versioned provenance. No
scientific algorithm or reduction behavior changed in either migration.

The learning omission is now locally migrated. All 28 leaves parse directly
into immutable `TimestreamLearningConfig`; one one-way adapter constructs the
unchanged `ReductionLearningState::Options` numerical input. The processed-
timestream requested/effective snapshots and versioned provenance now include
the complete learning policy. A frozen 28-path audit rejects reverse mirrors,
reader drift, incomplete adapter coverage, or missing serialization. Local CLI
and test builds plus focused reader/adapter tests pass. Because the standard
point fixture enables learning, an exact point Unity gate is the remaining
condition before marking this closeout domain complete.

The interface-sync omission is also locally migrated. All 14 TolTEC/HWPR
offsets parse atomically into immutable typed request state. Duplicate,
unknown, malformed, and non-finite entries are fatal; omitted interfaces retain
the established zero-second default with an explicit warning. One adapter
populates the unchanged alignment map. Raw-timestream provenance version 2
records requested and effective offsets with seconds as the explicit unit. A
frozen 14-path audit rejects reader, adapter, or provenance drift.

The F.1 startup gate is now operational rather than documentary. A generated
allowlist covers every normalized node in the checked 573-leaf contract and
the retained default configuration. Unknown nodes, including unknown empty
containers, enter fatal config diagnostics before execution. The `inputs`
subtree is deliberately excluded because its schema is owned by TolTECA; all
other low-level nodes are Citlali-owned. Typed validation errors now enter the
same fatal diagnostics instead of being logged as advisory mirror warnings.
The existing observation-scoped astrometry and photometry gates remain atomic.

The detailed [Phase 2 F.1 closeout](../handoff/PHASE2_F1_CLOSEOUT_2026-07-15.md)
maps every adopted checklist item to code, audit, and reduction evidence. Local
`citlali_cli` and test builds, all 410 CTests, all 96 config tests, eight compact
compatibility fixtures, 100% compact-surface coverage, and every boundary audit
pass. Unity point `redu62` closes the final gate as recorded below.

## Phase 2 Final Point Gate Accepted

Unity point `redu62` identifies `v4.0.0-3503-g9a3901e9` and the expected commit
`9a3901e91`. Its generated low-level input is byte-identical to accepted
`redu61`. It completed all 12 PTC chunks with zero error-, critical-, or
fatal-level records. Every required provenance sidecar passes semantic audit.

Processed-timestream provenance contains the complete 28-leaf requested and
effective learning policy exercised by the standard point fixture. Raw-
timestream provenance v2 contains all 13 TolTEC interface offsets plus HWPR in
requested and effective state, with unit seconds and exact equality. The
configuration-source manifest and canonical merged input are valid.

The strict zero-tolerance full-depth comparison reads all 21 stable products
and 2,041 records, including every RTC/PTC array. It reports zero changed,
skipped, missing, or extra records. The final F.1 gate is accepted; all 15
authority domains now have complete disposition and Phase 2 is complete.

The run took 176.435 seconds versus 63.671 seconds for `redu61`. The difference
is isolated to filesystem-facing stages: observation file setup increased from
1.758 to 28.723 seconds, raw/filtered output from 6.136 to 52.767 seconds, and
the 48 chunk-write calls averaged 4.172 rather than 2.482 seconds. Map
filtering, diagnostics, fitting, and other computational stages remained near
their prior timings. Treat this as an uncontrolled Unity/VAST I/O observation,
not a Phase 2 code-performance regression. A same-SHA rerun may characterize
the storage variance but is not required for scientific acceptance.

## Five-Phase Roadmap

### Phase 1 - Safety Stabilization

Repair output and run-success contracts, config parsing and finite-value
validation, output schema/cardinality checks, and ordered-writer cancellation.
Add injected failure and repeated-run tests without rewriting mature numerical
algorithms.

Exit gates:

- An injected required write failure returns a nonzero CLI status.
- Ordered output cannot deadlock after failure, and partial products have an
  explicit diagnosed disposition.
- A subsequent reduction in the same process starts with clean state.
- Invalid enums, NaN, and infinity fail with actionable config paths.
- The current point run has zero unexpected error-level messages and passes a
  strict complete-TOD and metadata comparison.

### Phase 2 - Config Authority And Provenance

Build the one-way flow from immutable requested config to effective execution
plan to realized observation metadata, with a temporary one-way legacy adapter.
Fix disabled-option provenance, atomic observation config, stale beammap flux
state, and typed/legacy parity checks. Validate real TolTECA overlay behavior
before compact config becomes operational.

Exit gates are the complete current-config definition of done in section F.1 of
the external review, including one authority per migrated field, no fallback to
raw YAML in migrated execution paths, correct provenance, and reviewed overlay
fixtures for each supported reduction mode.

### Phase 3 - Library, Session, And First Compiled Boundary

Introduce a minimal non-CLI reduction session/result boundary, remove reachable
library exits, give run/observation/scan state explicit owners, and freeze
`Engine` as a compatibility adapter. Add header-isolation and multi-translation-
unit checks, repair ODR hazards, and move one measured, coherent declaration and
validation tranche into `.cpp` files.

Exit gates:

- CLI policy is outside the library boundary.
- Sequential reductions in one process are clean and supported.
- Lifecycle state is reset by ownership rather than scattered cleanup.
- The first compiled boundary reduces dependency exposure without a material
  build or runtime regression.
- Further extraction has a named ownership or contract benefit; textual
  subdivision alone is not sufficient.

### Phase 4 - Validation, Performance, And Reproducible Build

Make strict comparison and active tests pinned CI gates. Add hermetic fixtures,
version/dependency provenance, current matched mode baselines, and controlled
performance diagnostics when triggered. Continue collecting timing and
peak-memory evidence during naturally required Beammap validation. Establish
polarimetry support or an explicit capability policy before release claims.

Exit gates are the broader structural definition of done in section F.2 of the
external review, with the project-owner performance proportionality exception:
strict scientific equivalence, zero unexpected errors, reproducible builds,
operational performance evidence with triggered controlled diagnostics, and
documented scientific conventions.

### Phase 5 - Integration And Closeout

Consolidate canonical architecture and scientific-convention documentation,
the validation ledger, and the intended-science-change manifest. Mark or remove
legacy/stub paths, tag the forensic refactor branch, and integrate the exact
validated tree. Add install/export support only if external library consumption
is an accepted project goal.

Core RTC/PTC algorithm cleanup, broad compact-config rollout, and R execution
are follow-up projects unless their prerequisites are explicitly brought into
this roadmap.

## Stop And Defer Rules

- Stop splitting files when a split has no clear owner, contract, test seam, or
  dependency benefit.
- Do not broadly rewrite RTC/PTC, JINC, or Wiener-filter numerical kernels in
  this refactor.
- Do not make compact config authoritative before TolTECA overlay acceptance.
- Do not implement R execution before a measured-channel data contract exists.
- Do not add concurrent reductions as a requirement unless the project owner
  explicitly needs them; sequential same-process reentrancy is required.
- After the refactor, replace the flat fruit-loop `reduNN` iteration sequence
  with one atomically claimed run directory containing explicit nested
  iteration identities, for example `redu01/iterations/iter00` through
  `iterNN`. Treat `redu01` as the stable identity of one user-invoked reduction,
  not as an iteration number. Add a run manifest that records a stable execution
  ID, each child iteration ID, the selected final iteration, Citlali version and
  git revision, and the effective-config digest. Preserve TolTECA-facing final-
  product compatibility during migration. This is the preferred long-term
  replacement for coarse output-root exclusion, but it is not part of the
  bounded Phase 3 repair.
- Do not squash or rewrite the only validated branch history.

## Decisions Requiring Scientific Ownership

Ask the project owner when implementation first depends on an answer. Do not
silently choose among these:

- Which output products are required versus optional in each reduction mode.
- How disabled filters and extinction states appear in requested, effective,
  and realized provenance.
- The future scientific meaning of hardware-polarization controls and the
  contract required to make enabled polarimetry a supported capability.
- Allowed calibration or analysis fallbacks and their required diagnostics.
- Canonical detector/network/array identities, coordinate frames, units,
  missing-value sentinels, and table schemas.
- OOF scientific intent and the acceptance tolerances for each mode.
- Whether any future caller needs concurrent reductions in one process.
- The measured-channel contract and missing-data policy for future R analysis.
- Whether Citlali must be installable and consumable as an external library.

## Durable Evidence

`validation/accepted_runs.json` is the machine-readable validation ledger. New
accepted checkpoints must record commit, binary version, mode, input/config
identity, comparator version, tolerances, error count, timing, available memory
evidence, and disposition. Run
`tools/baseline/validate_validation_ledger.py` after editing it.
`validation/validation_profiles.json` identifies the active immutable
validation epoch and one profile per supported reduction family; validate it
with `tools/baseline/validation_profiles.py --list`. Continue to update this
document and the dated handoff note at phase gates and material validation
checkpoints.
