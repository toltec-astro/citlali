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
- Latest inspected point reduction: `redu43`, produced by `11afd6f6`.
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
- CTest discovers and passes all 297 tests with none skipped or disabled; all
  32 config-boundary/preflight tests pass.

These facts are characterization evidence, not a production-equivalence claim.

## Active Phase

**Phase 2 - Config authority and provenance** is active as of 2026-07-11.

Phase 1 safety stabilization is complete for the currently available point,
Beammap, and science validation modes. OOF validation is explicitly deferred
by the project owner until its workflow is available, expected during the week
after 2026-07-11; because OOF closely follows pointing, that deferral does not
block Phase 2. It remains a required validation before final integration.

Operational config migration must proceed one authority domain at a time with
the one-way requested-to-effective-to-realized contract, focused tests, and the
existing mode gates. Compact-config production rollout and open-ended file
splitting remain out of scope.

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

The mapmaking authority migration is locally complete and awaiting Unity mode
gates. All 22 frozen `mapmaking.*` leaves now enter typed request state through
one boundary. `MapBuffer`, JINC, maximum-likelihood, observation-map, and
coadd-map configuration no longer parse YAML. One-way adapters construct the
legacy numerical mapmakers and WCS buffers from typed state. The immutable
execution plan preserves the requested grouping while exposing the resolved
effective grouping to downstream accessors; the transitional root request is
no longer mutated by map-count setup. Successful reductions must atomically
publish versioned `mapmaking_provenance.yaml`, and write failures propagate.
The effective plan also records the uncalibrated TOD-type unit substitution
without changing the requested `cunit`. The boundary preflight freezes the
22-path digest, enforces 22/22 reader
coverage, rejects retired parser symbols, and checks the production authority
sequence and provenance writer. Local CLI/test/safety builds, all 297 C++
tests, all eight config profiles, and the full preflight pass. A strict point
run is required first to validate WCS construction and the new sidecar;
Beammap and science runs then validate the JINC adapter. Realized per-
observation and product cardinality remain explicitly unavailable and are the
last mapmaking provenance sub-gate.

Project-owner decision (2026-07-10): every output explicitly enabled in the
configuration is required. RTC TOD, PTC TOD, `rtcdiag`, and `ptcdiag` write
failures must fail the reduction. There are no best-effort enabled products.

Immediate work order:

1. Build on Unity and run the standard point gate for the mapmaking authority
   cutover, including strict products, zero errors, and the new sidecar.
2. Run matched Beammap and science gates to exercise typed JINC parsing and
   one-way adaptation; OOF remains owner-deferred.
3. Populate and validate realized mapmaking observation/product cardinality
   without moving numerical algorithms.
4. Reassess Phase 2 domain priority using the authority inventory. Do not
   broaden polarimetry without the pending scientific-policy decisions.
5. Keep compact-config rollout and Phase 3 compiled-boundary work paused until
   the active domain gates close.

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
beammap timing and peak-memory evidence. Establish polarimetry support or an
explicit capability policy before release claims.

Exit gates are the broader structural definition of done in section F.2 of the
external review: strict scientific equivalence, zero unexpected errors,
reproducible builds, measured performance, and documented scientific
conventions.

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
- Do not squash or rewrite the only validated branch history.

## Decisions Requiring Scientific Ownership

Ask the project owner when implementation first depends on an answer. Do not
silently choose among these:

- Which output products are required versus optional in each reduction mode.
- Which TOD types are supported and how unknown values must fail.
- How disabled filters and extinction states appear in requested, effective,
  and realized provenance.
- The exact meaning of hardware-polarization ignore/enable controls and whether
  polarimetry is a supported release capability.
- Allowed calibration or analysis fallbacks and their required diagnostics.
- Canonical detector/network/array identities, coordinate frames, units,
  missing-value sentinels, and table schemas.
- Beammap source-flux fallback and reset behavior.
- OOF scientific intent and the acceptance tolerances for each mode.
- Whether any future caller needs concurrent reductions in one process.
- The measured-channel contract and missing-data policy for future R analysis.
- Whether Citlali must be installable and consumable as an external library.

## Durable Evidence

`validation/accepted_runs.json` is the machine-readable validation ledger. New
accepted checkpoints must record commit, binary version, mode, input/config
identity, comparator version, tolerances, error count, timing, available memory
evidence, and disposition. Run
`tools/baseline/validate_validation_ledger.py` after editing it. Continue to
update this document and the dated handoff note at phase gates and material
validation checkpoints.
