# Independent External Architecture Review — Citlali Refactor

**Review date:** 2026-07-10

**Reviewed branch:** `codex/structural-refactor`

**Reviewed tree:** `7417a2ad` (`84670829` plus the review brief; the production code is the `84670829` snapshot)

**Baseline:** `376e0022`

**Verdict:** **Sound with material reservations**

## Review scope and evidence standard

This is an architecture, scientific-trustworthiness, validation, and engineering-process review, not a line-by-line defect audit. I read the external brief and the governing refactor/config/R-channel material, traced the active control path and state model, examined representative config, output, error, concurrency, test, build, and validation code, inspected the branch history, and independently ran the available local gates. I also inspected the new Unity point product for the exact code snapshot under review.

Evidence is described as one of the following:

- **Observed:** directly present in source, build metadata, Git, or a reduction artifact.
- **Conclusion:** supported by multiple observations.
- **Inference:** a likely consequence that has not been reproduced as a failure.
- **Unknown:** requires a controlled experiment or scientific-owner decision.

The most relevant independent checks were:

| Check | Result |
| --- | --- |
| `cmake --build build --target citlali_cli -j 8` | Passed in the existing local build tree. |
| `tools/refactor/refactor_inventory.py` | Passed; 616 headers, 145 direct exits, 137 high-risk header exits, and seven commented CMake source entries. |
| `tools/config/run_config_preflight.py --require-all` | Passed 8/8 cases; 265 actionable leaves covered, 17 profile-owned, zero gaps. |
| Local CTest discovery | Zero tests: the inspected build has `CITLALI_BUILD_TESTS=OFF`. |
| Branch scale from `376e0022` | 2,888 commits, 696 changed files, 74,299 insertions, and 21,557 deletions. |
| Current point run | `redu22`, Citlali `v4.0.0-3304-g84670829`, 61.051 s, 33 files, 12 PTC chunks, and 12 logged errors. |
| `redu21` versus `redu22` low-level config | 489 leaves on each side and zero differences. |
| Full `redu21` versus `redu22` numeric product comparison | Both are refactor runs: `71635a7e` versus `84670829`, six commits apart. With TOD enabled and the array-size cap disabled, all common science, FITS, diagnostic, RTC TOD, and PTC TOD numeric arrays were exact; no arrays were skipped. Three numeric records changed: two IIR-frequency provenance values and profile elapsed time. |

This is positive evidence that the six-commit typed-output/provenance batch did not alter common point numeric products. It is not a current original-versus-refactor equivalence test: both runs contain the same 12 NetCDF errors, so pairwise equality also cannot prove either product set complete. It does not negate the release-blocking error and provenance findings below.

---

## A. Executive assessment

### Overall judgment

The refactor is **directionally sound** and the current tree is a viable foundation for completion. A broad rewrite is neither necessary nor justified. The CLI and high-level reduction flow are markedly easier to understand, the typed scientific vocabulary is a real improvement, and the project has acquired useful comparison and profiling tools.

The repository is **approaching**, but has not yet reached, a modern and professionally maintainable state. The architecture currently has a clear conceptual pipeline but not yet a comparably clear physical module, ownership, failure, or reproducibility model. Much of the original coupling has been redistributed into hundreds of contextual headers operating on the same public mutable `Engine` aggregate. That is useful extraction work, but it is not yet an enforceable library architecture.

### Greatest strength

The greatest strength is that the operational lifecycle is now named and inspectable. A 54-line CLI delegates to reduction, iteration, observation, preflight, mode, RTC/PTC, mapmaking, and output stages that can increasingly be tested as orchestration. This makes the scientific workflow visible without requiring an immediate rewrite of mature hot-loop algorithms.

### Greatest risk

The greatest risk is a mismatch between **reported success and scientifically reproducible execution**:

- required-looking NetCDF writes catch and log failures without propagating them;
- the audit classifies a run with 12 error messages as successful;
- typed config can influence execution while its validation is advisory and a parse failure can silently preserve a valid default;
- requested, effective, and realized config values are not distinguished, and the current validated snapshot already changed output provenance for unchanged YAML.

These are contract and state-authority problems. They are architectural, not cosmetic.

### Pause recommendation

Pause **further typed analysis-control migration, compact-config production rollout, and any claim that this branch is production-ready** until the two Critical findings in section E are fixed and protected by strict gates. Do not pause narrowly scoped work that fixes those findings, activates tests, clarifies config authority, creates safe session ownership, or establishes a first measured `.cpp` boundary. Do not begin a broad RTC/PTC/JINC/Wiener rewrite now.

For integration, review the branch as an exact, validated snapshot with subsystem summaries and follow-up changes, not by asking reviewers to consume 2,888 microcommits. Preserve the granular branch or a tag for forensics. Avoid a risky history rewrite of the only validated tree; if repository policy requires a condensed integration commit, verify that its tree hash and built artifacts match the tagged snapshot.

---

## B. Current architecture

### B.1 Runtime control and data flow

The actual active flow is:

```text
TolTECA numbered YAML overlays (external operational boundary)
    -> generated low-level citlali_o*.yaml
    -> Citlali CLI/config merge and runtime setup
    -> variant<TimeOrderedDataProc<Lali | Pointing | Beammap>>
       (selected mode engine is owned by value)
    -> initial geometry pass
    -> fruit-loop reduction iteration
       -> observation loop
          -> observation input/calibration preflight
          -> mode-specific setup and scan generator/farm
          -> KIDs solve / raw time chunk (RTC)
          -> processed time chunk (PTC)
          -> learning and diagnostic capture
          -> map accumulation
          -> ordered TOD/diagnostic output
       -> observation and coadd finalization
       -> filtering, fitting/source products, learning/index finalization
    -> CLI completion or exception reporting
```

The top-level path is visible in:

- `src/citlali/cli/main.cpp`
- `include/citlali/core/cli/standard_reduction_execution.h`
- `include/citlali/core/cli/reduction_execution.h`
- `include/citlali/core/pipeline/reduction_pipeline.h`
- `include/citlali/core/pipeline/reduction_iteration_loop.h`
- `include/citlali/core/pipeline/reduction_observation_loop.h`

This control-flow description is a **direct observation**. The judgment that it is clearer than baseline is a **conclusion** supported by the baseline/current file shapes and the extracted stage tests.

### B.2 Major components as they actually exist

| Component | Actual responsibility and current condition |
| --- | --- |
| `src/citlali/cli`, `include/.../cli` | Argument/config startup, runtime selection, top-level exception reporting, and the current full-reduction library-like entry. The executable entry is thin, but the reusable entry still lives in the `citlali::cli` header namespace rather than a stable library API. |
| `citlali::pipeline` | High-level orchestration, config readers/mirrors, context/accessor helpers, output policy/writers, profiling, and many mode-independent implementation fragments. It is both an orchestration layer and a large policy/implementation namespace. |
| `citlali::config` | Typed enums, aggregates, validators, activation policy, and unit-suffixed values. This is the cleanest lower-level dependency island. It is not yet the sole authority for acceptance or execution. |
| `Engine` and mode types | `Engine` publicly aggregates calibration, telescope, I/O, RTC/PTC, mapmaking, buffers, output, config, observation, iteration, and progress state. `Lali`, `Pointing`, and `Beammap` publicly inherit it. It remains the ambient mutable state model. |
| RTC/PTC and mapmaking | Mature, performance-sensitive processing and scientific algorithms plus substantial configuration, diagnostics, and output behavior. The largest headers intentionally remain mostly legacy-shaped. |
| Calibration/telescope/raw-observation models | Scientific tables, telescope series, raw input metadata, and related ownership. Some ownership is explicit, but many calibration/table interfaces remain string-keyed and publicly mutable. |
| Product writers | FITS, NetCDF, ECSV/CSV, diagnostics, statistics, TOD, and map products. The code is split among pipeline and engine-detail contextual fragments and RTC/PTC methods; required-versus-optional output semantics are not explicit. |
| Tools | Reduction audit, product/config comparison, inventory, profiling, compact-config translation/preflight, and analysis-flow documentation. Useful as triage and evidence collection; several are not yet strict acceptance gates. |
| Tests/CI | A large config/pipeline scaffold with useful pure/fake-stage coverage, but the current build runs no tests and CI does not reliably build the excluded test executable. |

### B.3 Physical dependency and ownership map

The important distinction is between conceptual stages and C++ module boundaries:

```text
                         one large executable translation unit
                                      |
                                      v
        CLI templates -> pipeline orchestration/config/output fragments
                                      |
                          implicit template field access
                                      v
             Lali / Pointing / Beammap -> public Engine aggregate
                                      |
              +-----------------------+-----------------------+
              |                       |                       |
              v                       v                       v
       RTC/PTC processors       mapmaking/buffers       calib/telescope/I/O

Actual reverse/layer leaks:
  Engine headers ---------------> pipeline policy/helpers
  RTC/PTC -----------------------> engine/io.h synchronization and models
  mapmaking ---------------------> engine/config.h

Clean lower-level island:
  config enums/models/validation (no engine or pipeline dependency found)
```

Observed details:

- `Engine` publicly inherits three state aggregates in `include/citlali/core/engine/engine.h:123-125`; `ReductionComponents` in `component_state.h:22-45` publicly inherits observation, timestream, and mapmaking component bags.
- `TimeOrderedDataProc` owns the selected mode engine by value (`include/citlali/core/engine/todproc.h:126-140`), which is a sound ownership decision.
- The static library compiles only five substantive `.cpp` files, while the CLI source pulls in nearly the full template/header graph (`CMakeLists.txt:66-83,106-114`). Seven prospective `.cpp` entries remain commented out.
- The current CLI dependency file references 599 Citlali headers, including 525 under engine/pipeline.
- An include-occurrence scan found 233 engine-to-pipeline include occurrences. Pipeline-to-engine has little explicit inclusion but extensive implicit coupling through templated direct field access.
- `pipeline` contains 348 headers and 23,282 lines; `engine/detail` contains 171 headers and 15,878 lines. Of these, 283 headers explicitly say they must be included inside another namespace or after a class declaration. Those files are contextual fragments, not independently enforceable modules.

### B.4 Legacy and transitional architecture

The transition has four important dualities:

1. **Typed config and legacy runtime fields.** Some values are only mirrored; some typed values already drive channels, filenames, output policy, map grouping, and provenance; many algorithms still consume legacy fields.
2. **Named stage contexts and ambient `Engine` state.** Small contexts exist, but most scientific and runtime state remains public and is mutated through the engine.
3. **Library intent and CLI implementation.** The executable is thin, yet `PipelineRunner`, `ReductionSession`, and `ReductionResult` remain design-document concepts; the real top-level entry is a CLI template.
4. **Active and obsolete source.** Unbuilt files such as `src/citlali/main_old.cpp`, `mpi_main.cpp`, `kids_main.cpp`, and `lali_main.cpp` remain beside active source without a durable legacy designation; empty source stubs look like active future boundaries.

These are acceptable as temporary migration states only if their authority, removal condition, and owner are documented. Several currently lack that finite exit condition.

---

## C. Alignment between stated intent and implementation

| Stated intent | Status | Independent assessment |
| --- | --- | --- |
| Thin CLI and reusable orchestration | **Largely realized** | The CLI is 54 lines and the scientific lifecycle has named stages. The stable non-CLI result/failure API is not yet present. |
| Make the flow understandable | **Realized at the conceptual level** | The reduction/iteration/observation sequence is much easier to trace and fake-stage tests protect ordering. |
| Complete typed internal config | **Partially realized** | The vocabulary is broad and useful, but it contains duplicate facts, does not cover all method-specific settings, is not the authoritative validation gate, and already controls selected behavior. |
| Preserve low-level YAML and TolTECA overlays | **Partially evidenced** | Static compact translation passes 8/8 fixtures. Real numbered-overlay execution through TolTECA, list/null semantics, multiple steps, and clean-CI reproducibility are not proven. |
| Separate config, state, algorithms, diagnostics, and output | **Partially realized** | Names and files are separated; ownership and dependencies often are not. Pipeline and engine-detail fragments still mutate the same broad aggregate, and RTC/PTC combine algorithms, config, diagnostics, and I/O. |
| Replace library exits with typed failures | **Not materially realized** | The inventory finds 145 direct exits, 137 in high-risk headers. Typed error scaffolding is not the production-wide contract, and output exceptions are swallowed. |
| Move appropriate implementation to `.cpp` | **Not materially realized** | Only five real library sources are compiled; the seven proposed additions remain commented/stub-like. Header hygiene currently hides at least one ODR defect. |
| Preserve science behavior | **Strong for the latest six-commit point delta; incomplete overall** | Common point numeric products, including TOD when fully compared, are exact between two successive refactor snapshots. Both share the same errors. The documented deterministic original-versus-refactor point comparison is 2,882 commits behind current and did not claim full TOD coverage; current beammap/science/OOF/polarimetry evidence is also insufficient. |
| Meet 3–5% runtime and stable memory budget | **Not demonstrated** | The two latest point timings are uncontrolled and internally mixed. Historical beammap candidates are 7.2–8.1% slower than the cited baseline; paired repeat data and peak RSS are absent. |
| Establish a dependable test/CI pyramid | **Partially implemented, not operational** | Useful scaffold tests exist, but the inspected build exposes zero tests and CI does not reliably build the excluded test target or run new validation tools. |
| Preserve a path for optional measured R data | **Directionally preserved, execution deferred** | The scaffold recognizes an auxiliary channel, but its measured-versus-synthetic type/shape/unit/calibration contract is not yet enforceable. Deferral is appropriate. |

The largest divergence is that **file extraction advanced much farther than enforceable state, failure, and compilation boundaries**. The largest unresolved design question is the exact state transition from immutable request to normalized/effective plan to observation-specific realized metadata.

---

## D. Architectural strengths

1. **The high-level lifecycle now reflects the scientific process.** Reduction, iteration, observation, RTC/PTC, map accumulation, coadd, filtering/fitting, and finalization are named stages. That makes ordering and mode-specific behavior reviewable without disturbing hot algorithms.

2. **Orchestration is becoming testable independently of the full application.** The fake-stage tests in `tests/test_config_scaffold.cpp:3322-3571` exercise ordering and preflight short-circuit behavior. This is the right kind of seam for a scientific pipeline.

3. **`citlali::config` is a useful dependency island.** Central enums, unit-bearing field names, path-aware validation reports, one-based chunk naming checks, and explicit activation vocabulary reduce scattered string policy and improve future AI reasoning.

4. **The refactor deliberately avoided a wholesale algorithm rewrite.** Keeping broad RTC/PTC/JINC/Wiener redesign out of the structural work is prudent. The branch nevertheless includes intentional imports of prior determinism, cleaning, and Wiener behavior changes; this review did not independently re-establish each import’s scientific evidence. Those changes require an explicit intended-science-change ledger so they are not mistaken for structural regressions or equivalence. Scientific code earns modernization through demonstrated correctness and performance, not fashion.

5. **Mode ownership by value is clear.** `TimeOrderedDataProc` owns its chosen engine; FITS handles use `unique_ptr` in relevant paths. These are concrete improvements over ambiguous external lifetime management.

6. **The compact-config strategy preserves the expert surface.** Translating a smaller user model to the existing low-level schema, while allowing later expert overlays, is appropriate for TolTECA operations. The preflight inventory is honest about what it proves.

7. **Validation tooling is materially better than at baseline.** Audits, product comparisons, deterministic manifests, config comparisons, and profiling make discrepancies visible. The current review was able to identify an exact requested/effective provenance difference because those tools and artifacts exist.

8. **The handoffs acknowledge uncertainty and incomplete work.** The repository does not falsely claim that typed config, science validation, `.cpp` migration, or R execution are complete. That candor is valuable durable context, even though the notes now need consolidation.

9. **Validation has already found a real identity/index defect.** The beammap map/detector index versus TolTEC array-id correction is precisely the kind of latent scientific ownership error that characterization runs should expose. It supports retaining strong product-level gates during further type work.

---

## E. Findings by priority

### Summary

| ID | Priority | Finding | Blocks |
| --- | --- | --- | --- |
| C1 | Critical | Output and run-success contracts permit logged write failures to complete successfully. | Production-ready claim and broader feature/control migration. |
| C2 | Critical | Config parsing/validation can silently substitute a valid typed default or admit non-finite scientific values. | Further typed execution migration. |
| I1 | Important | Typed/legacy dual state conflates requested, effective, and realized config; current provenance already diverges. | Config-phase completion. |
| I2 | Important | Decomposition is predominantly textual; `Engine` remains ambient state and the compiled module boundary is almost unchanged. | Broader refactor completion. |
| I3 | Important | Run/observation/scan lifecycle ownership is incomplete and execution is not safely reentrant. | Credible library API and repeated-run safety. |
| I4 | Important | The validation matrix and comparison tools do not support production-equivalence claims across modes. | Analysis-control migration and release. |
| I5 | Important | Scientific identities, units, table schemas, and valid-state invariants remain too implicit at subsystem boundaries. | Broader refactor completion; targeted types before relevant migrations. |
| I6 | Important | Tests, CI, dependency versions, version provenance, and build instructions are not reproducible gates. | Production-ready claim. |
| I7 | Important | The 3–5% performance and memory policy is not demonstrated for beammap. | Performance-sensitive integration/release. |
| I8 | Important, phase-gated | Compact config needs a real TolTECA overlay acceptance test and stronger source provenance before rollout. | Compact-config rollout, not current low-level operation. |
| I9 | Important | Active/legacy source and durable validation/science-change evidence are not reliably designated. | Broader completion and snapshot integration. |
| A1 | Advisable | The 2,888-commit history needs a practical review/integration strategy. | Reviewability, not runtime safety. |
| A2 | Advisable | The auxiliary R scaffold does not yet enforce a measured-channel contract. | Future R implementation only. |

### C1 — Output and run-success contracts permit logged write failures to complete successfully

**Observation and evidence.** The current `redu22` run for `84670829` contains 12 `NetCDF: Not a valid ID` messages at error severity during a 12-PTC-chunk run and later logs normal completion. The preceding `redu21` refactor run contains the same 12 errors. The audit reports `Issue counts: {'error': 12}`, yet `compare_reduction_audits.py` defines serious issues as only `fatal`, `critical`, and `traceback` and reports both reductions OK (`tools/baseline/compare_reduction_audits.py:29-45`).

The behavior is structurally enabled by the code:

- RTC diagnostic and TOD append functions catch `netCDF::exceptions::NcException`, log it, and return normally (`include/citlali/core/timestream/rtc/rtcproc.h:6395-6416,6419-6473`).
- PTC append functions use the same pattern (`include/citlali/core/timestream/ptc/ptcproc.h:4755-5293` and the diagnostic append path beginning at line 5296).
- The pointing output caller then advances the ordered writer after the void append call (`include/citlali/core/engine/detail/pointing_timestream_output_impl.h:89-103`). Lali and beammap use the same wait/write/advance protocol.
- `run_reduction_observation` returns false only for preflight; after invoking a void runtime pipeline it returns true (`include/citlali/core/pipeline/reduction_observation.h:24-42`). `run_cli_reduction_processor` logs completion and returns `EXIT_SUCCESS` on that boolean (`include/citlali/core/cli/reduction_execution.h:103-115`).
- `OrderedWriter` separates `wait_turn()` and `advance()` and has no cancellation or failure state (`include/citlali/core/pipeline/ordered_writer.h:10-24`). A different exception between those calls can leave later farm workers blocked indefinitely. That deadlock is an **inference**, not a reproduced failure.
- The wider contract still includes 41 direct process exits in active RTC/PTC and engine-detail paths (representative locations include `rtcproc.h:969-1421`, `ptcproc.h:969,4027,4745`, `engine/detail/photometry_config_impl.h:24`, and `todproc_alignment_impl.h:39,186,250`). A `ReductionResult` wrapper cannot catch `std::exit`, so repeated-run recovery is not credible while any supported non-CLI path can terminate the process.

**Concrete consequence.** A required product can be partial, stale, or incomplete while the CLI and audit declare success. The full current product comparison happened to find exact numeric arrays, so this review does **not** assert that `redu22` science arrays are corrupt. Because both sides have the same errors, the equality is equivalence under a shared failure, not evidence of completeness. A pairwise comparator cannot detect a systematic omission without an independent expected schema/cardinality/product oracle.

**Why architectural.** This crosses the processor, output coordinator, concurrency primitive, library result, CLI exit, and validation layers. It is the system-wide definition of success, not a local logging choice.

**Smallest corrective direction.** Define required and optional product classes. Make every required append return/throw a structured failure that reaches a non-CLI `ReductionResult`; have the CLI alone translate that result to an exit code. Replace manual wait/write/advance with a cancellable RAII turn or queue that records the first failure and wakes all waiters. At finalization verify expected schema, chunk, row, and product counts against a mode/config-derived oracle. Make an unallowlisted error fail the reduction audit. Add an injected mid-stream NetCDF failure test with multiple workers and a partially created output.

**Scope and confidence.** **Present refactor; release-blocking. High confidence.** Fix the concrete NetCDF path first, then convert the remaining exits incrementally by boundary rather than in one destabilizing patch. Architectural completion requires zero reachable process termination in every supported non-CLI path; only mechanically unreachable, explicitly allowlisted legacy code may remain. Retain `citlali::error::Error` or a result type as one canonical library contract; remove the parallel global `DataIOError`/bool/exit mixture incrementally.

### C2 — Config parsing/validation can silently substitute a default or admit non-finite values

**Observation and evidence.** There are two independent acceptance defects:

1. `read_parsed_mirrored_config_value` assigns the typed target only when its parser succeeds. On failure it leaves the target unchanged and does not itself append an invalid diagnostic (`include/citlali/core/pipeline/config_parse_tracking.h:92-109`). The timestream-type reader supplies no accepted-value list (`include/citlali/core/pipeline/timestream_config_read.h:34-43`), so an unknown string can leave the valid default `TodType::xs` (`include/citlali/core/config/timestream_config.h:581-584`). This typed enum selects the actual KIDs array (`include/citlali/core/pipeline/kids_tod_channel.h:12-26` and the mode scan setup paths), so this is not limited to metadata.
2. Both legacy and typed range helpers rely only on `<`, `<=`, and `>` (`include/citlali/core/engine/config.h:22-54`; `include/citlali/core/config/config_error.h:107-129`). IEEE NaN makes those comparisons false and passes. `check_optional_minimum` skips every non-finite value, including positive and negative infinity (`config_error.h:131-137`). A field such as map pixel size is converted to radians after a legacy minimum check (`src/citlali/core/mapmaking/map.cpp:84-100`), allowing a non-finite geometry input to escape the intended domain check.

The central helper fix will not cover every field by itself. Beammap flag thresholds, for example, are read directly as `std::vector<double>` values (`include/citlali/core/pipeline/beammap_config_fitting_flagging.h:82-128`), while typed validation currently checks only `max_prior_d2` and not vector-element finiteness, range, or lower/upper consistency (`include/citlali/core/config/beammap_config_validation.h:98-101`).

**Concrete consequence.** A misspelled channel can execute the `xs` path without a fatal diagnostic. NaN/Inf can enter geometry or other scientific calculations and create plausible-looking invalid products or delayed failures. No current reduction demonstrates that outcome; the acceptance path itself is directly observed.

**Why architectural.** Typed config is the planned scientific policy boundary. If it cannot distinguish invalid input from a legitimate default, every downstream typed consumer is unsafe regardless of its local implementation.

**Smallest corrective direction.** Make parser failure append a fatal path-aware diagnostic; derive accepted spellings and aliases from the same enum table used by the parser. Require finite floating-point values by default before range checks. Represent optionality with `std::optional` or an explicit domain state rather than a non-finite sentinel. Audit the full schema—including container elements and direct `get_typed` readers—for finite-by-default, range, and cross-field coverage. Add real-reader tests for every enum and representative `.nan`, `.inf`, `-.inf`, overflow, zero, disabled, and automatic cases.

**Scope and confidence.** **Present config phase; blocks any further typed execution/control migration. High confidence.** Existing typed consumers should be audited after the central fix, not patched individually.

### I1 — Typed/legacy dual state conflates requested, effective, and realized config

**Observation and evidence.** Typed validation currently logs warnings that “legacy config parsing remains authoritative” (`include/citlali/core/pipeline/reduction_config_validation_logging.h:21-47`), while typed values already choose TOD channels, output policy and filenames, map grouping, beammap behavior, and provenance. The function named `validate_typed_config_mirrors` validates the typed object internally; it does not compare it to the legacy runtime.

The exact current-snapshot validation supplies a concrete divergence:

- `redu21` and `redu22` low-level YAML each contain 489 leaves with zero differences.
- Full numeric comparison, including all RTC/PTC TOD arrays, found exact common numeric arrays between the two refactor runs; both runs nevertheless contain the same output errors.
- All three FITS primary headers and both TOD NetCDF products changed `CONFIG.TODIIRHP.FREQ_HZ` from `0.0` to `0.1`.
- The same provenance changed `CONFIG.EXTINCTION.EXTMODEL` from `N/A` to an empty string.

For disabled IIR filtering, legacy setup explicitly sets the effective frequency to zero and reads `freq_Hz` only if enabled (`include/citlali/core/timestream/rtc/rtcproc.h:1380-1407`). The typed object defaults to `0.1` (`include/citlali/core/config/timestream_config.h:126-130`), and mirroring returns after setting `enabled=false` without replacing that default (`include/citlali/core/pipeline/timestream_config_mirror_raw_filters.h:77-88`). The new FITS/NetCDF writers emit the typed value (`include/citlali/core/pipeline/phdu_reduction_weight_tod_config.h:87-107`; `reduction_config_weight_runtime_netcdf.h:76-100`). Extinction has the analogous distinction between typed empty default and legacy effective `N/A` (`timestream_config.h:258-270`; `engine/detail/observation_setup_impl.h:43-45`).

Other observed examples reinforce the same structural issue. Automatic map grouping overwrites `mapmaking_config.grouping` in place with the effective value (`include/citlali/core/engine/detail/todproc_map_count_impl.h:10-29`), while downsampling and calibration decisions can be resolved only after observation data is known. `PolarimetryHwprPolicy::require` serializes to legacy `"false"` (`include/citlali/core/config/timestream_enums.h:134-139`), but execution only treats `"true"` as forced ignore and disables HWPR when data are absent rather than enforcing a requirement (`include/citlali/core/pipeline/hwpr_policy.h:14-27`; `hwpr_loading.h:15-33`). The typed name therefore promises stronger behavior than the legacy control path supplies.

**Concrete consequence.** Output can describe the request, a typed default, or an effective runtime value without saying which. Reconstructing a reduction or comparing old/new products becomes ambiguous even when the numerical science is unchanged. In other cases the same duality can change execution.

**Why architectural.** Config is not a bag of equivalent copies. It has a lifecycle and provenance semantics that must be explicit across parsing, normalization, observation resolution, execution, and serialization.

**Smallest corrective direction.** Establish a one-way transition:

```text
immutable RequestedReductionConfig
    -> normalized/resolved EffectiveReductionPlan
    -> per-run/per-observation RealizedRunMetadata
    -> temporary one-way legacy processor adapter
```

Preserve collision-safe exact source bytes, merge order/role, and content hashes—not paths/hashes alone—then record every fallback/coercion and reason. Serialize the generated low-level request, canonical typed request, effective plan, and realized state with unambiguous labels. During migration, compare the adapter immediately after context-free resolution and again after observation resolution. Values subsequently derived or mutated by processors belong in realized metadata rather than being compared indefinitely with the effective plan. Do not add bidirectional synchronization.

**Scope and confidence.** **Present config phase. High confidence.** The observed metadata changes may be semantically reasonable if deliberately labeled; the defect is that the current format does not make that distinction.

### I2 — Decomposition is predominantly textual, while `Engine` remains ambient state

**Observation and evidence.** The reduction flow has good named boundaries, but the physical architecture remains a single large header graph:

- `pipeline`: 348 headers, 23,282 lines; 106 headers are at most 30 lines and 202 are at most 60.
- `engine/detail`: 171 headers, 15,878 lines; 33 are at most 30 lines and 65 are at most 60.
- 283 headers explicitly require inclusion after a class declaration or inside another namespace.
- `engine.h` fell from 9,674 to 298 lines, but now has 111 direct includes; the CLI object depends on 599 Citlali headers.
- `Engine` publicly inherits `ReductionComponents`, `BeammapFluxState`, and `EngineRuntimeState` (`include/citlali/core/engine/engine.h:123-125`). Those bases expose calibration, telescope, I/O, RTC/PTC, all mapmakers/buffers, config, output, observation, logging, and progress state (`component_state.h:22-45`; `runtime_state.h:9-13`). Mode classes publicly inherit `Engine` and expose broad mutable interfaces.
- The CMake static library still compiles only five substantive sources (`CMakeLists.txt:66-83`). The planned `PipelineRunner`, `ReductionSession`, and `ReductionResult` are absent from production code; the full reusable entry remains under `citlali::cli`.
- Header-only compilation currently masks an ODR problem: `include/citlali/core/engine/kidsproc.h:23` defines the non-inline global `bool extra_output = 0`.

**Concrete consequence.** A small header edit rebuilds the large CLI translation unit; contextual fragments cannot be checked independently; dependencies are enforced by convention; and any pipeline helper can mutate almost any engine field. Navigation cost and AI selection risk have replaced part of the original monolith cost.

**Why architectural.** A module boundary must restrict dependencies and state access, not only provide a filename. The current shape makes conceptual stage names clearer but does not yet make invalid dependencies or ownership violations hard to express.

**Smallest corrective direction.** Stop all line-count-driven splitting. Keep the small top-level stage headers that express real lifecycle boundaries. Consolidate product-family fragments (for example the many `mapdiag`/`rtcdiag`/`ptcdiag`/PHDU writers) and freeze `Engine` as a compatibility adapter: no new public cross-cutting state. First add public-header self-compilation and a two-translation-unit link test, then fix `extra_output` and similar ODR hazards.

The safest first meaningful compiled tranche is cold, concrete config code: move non-template enum parse/string tables from the 946-line `timestream_enums.h` and concrete validator definitions from the 1,068-line `timestream_config_validation.h` behind declarations. Keep tiny performance-relevant predicates inline. A second tranche can move non-template profiling/output-name and product-creation coordination by coherent product family. Measure header closure, clean/incremental build time, binary behavior, and runtime after each tranche. Do not simply uncomment empty source stubs.

**Scope and confidence.** **Present broader refactor, after C1/C2 stabilization. High confidence.** RTC/PTC kernels, JINC, Wiener filtering, and a sweeping Beammap move should remain later work.

### I3 — Run, observation, and scan lifecycle ownership is incomplete

**Observation and evidence.** All three real mode scan generators use a function-static scan cursor that resets only after normal exhaustion:

- `include/citlali/core/engine/detail/lali_setup_pipeline_impl.h:29-67`
- `include/citlali/core/engine/detail/pointing_pipeline_impl.h:23-61`
- `include/citlali/core/engine/detail/beammap_timestream_pipeline_impl.h:21-54`

The cursor is shared across instances of a template specialization and can remain stale after an exception. The stage profiler is a process singleton (`include/citlali/core/pipeline/stage_profile.h:133-147`) and is reset per top-level reduction. The current `ReductionObservationContext` owns raw observation metadata and an index, but sample rate, calibration, output layout, maps, iteration state, and scan state remain ambient in `Engine`.

Observation-time config also crosses lifecycle boundaries. Astrometry/photometry are loaded during observation preparation (`include/citlali/core/pipeline/observation_calibration_config.h:9-51`), after the startup aggregate typed validation gate, without a new fatal observation-scoped gate. Existing domain checks are also incomplete: beammap source validation verifies RA/Dec finiteness but not range, frame, or wrap rules (`include/citlali/core/config/beammap_config_validation.h:118-128`), while astrometry validation checks vector lengths but not element finiteness, MJD ordering, or observation coverage (`include/citlali/core/config/calibration_config_validation.h:8-45`). `Engine::get_photometry_config` resets the typed source object but populates persistent legacy `source_flux_mJy_beam` without clearing it (`include/citlali/core/engine/detail/photometry_config_impl.h:9-24`; `include/citlali/core/engine/beammap_flux_state.h:6-10`). A later observation with a missing array flux can therefore inherit earlier state; that consequence is a strong **inference** from the observed merge behavior and needs a two-observation regression test.

**Concrete consequence.** Sequential use after failure is unsafe, concurrent reductions are not supported by the state model, and observation-specific scientific state may leak across observations. A library caller cannot reason locally about what a run owns or resets.

**Why architectural.** Lifecycle is part of scientific meaning: a calibration or flux valid for one observation is not generic engine state. Reentrancy and reset behavior cannot be reliably repaired with comments around globals.

**Smallest corrective direction.** Require sequential reentrancy even if concurrent reductions remain unsupported. Move the scan cursor, ordered writers, profile collector, run result, and run/iteration/observation/scan state into an explicit `ReductionSession` assembled incrementally. Build and scientifically validate each observation plan atomically—including finite elements, coordinate rules, temporal ordering/coverage, and required calibration—then replace, not merge, observation-scoped maps. Add tests for two sequential reductions, two beammap observations with a missing second flux, and recovery after injected failure. Document whether concurrent sessions are intentionally unsupported before attempting parallel-session support.

**Scope and confidence.** **Present broader refactor. High confidence** for the static/global state; **medium-high** for the stale-flux consequence until reproduced.

### I4 — The validation matrix and comparator do not support production-equivalence claims

**Observation and evidence.** The evidence by mode is uneven:

| Mode/gate | Current independent assessment |
| --- | --- |
| Pointing | Strongest frequent gate, but the current pair is refactor-to-refactor: `redu21` at `71635a7e` and `redu22` six commits later at `84670829`. All common numeric products, including complete TOD when forced, match exactly, but both have the same 12 output errors. This proves the recent delta did not change common numeric arrays, not current original-versus-refactor equivalence or completeness. The checked-in deterministic original comparison is at `a47705f2`, 2,882 commits behind current, and does not claim full TOD coverage. |
| Beammap | Useful historical product/repeatability evidence at `f278bd32`, but that code is 120 commits behind `84670829`; 157 files changed, including 123 engine/detail/pipeline-sensitive paths. No current-snapshot beammap gate protects the latest typed output/config work. |
| Science | The available baseline/candidate pair is 1,136 commits behind current and uses non-equivalent config/product selections. It is not a clean acceptance pair for the present branch. Independent `EXTNAME` matching confirms missing weight/noise HDUs, missing filtered-coadd point-source HDUs, and large differences in common coadd products, but generic positional comparator values after the first missing HDU are not trustworthy because different products can be zipped together. |
| OOF | Compact-config fixtures exist, but no current refactor-versus-baseline runtime/product gate was found; the comparator has no OOF mode. |
| Enabled polarimetry | No current enabled end-to-end baseline. Disabled ordinary pointing does not validate HWPR/polarized execution, yet the enabled path remains reachable. Until a reference gate exists, preflight should mechanically reject enabled polarimetry rather than merely document it as unsupported. |
| Unit/integration tests | Useful fake/config tests exist, but the inspected build discovers zero tests. There is no substantive injected writer-failure or small real product-schema integration suite. |

The product comparator says it is a triage tool, not a strict gate. By default it excludes timestream products, caps arrays at ten million elements, compares only a small FITS header allowlist, zips HDUs positionally, skips nonnumeric NetCDF variables and attributes, and returns zero regardless of changes (`tools/baseline/compare_reduction_products.py:40-62,151-157,251-350,615-660`). After an HDU is missing or inserted, positional zipping can report cross-product numerical differences; schema differences must therefore be resolved by stable identity before interpreting subsequent values. The normal point command also missed both provenance changes; a default size cap can skip large arrays even when `--include-timestream` is supplied. Generic `atol=2e-8` and `rtol=1e-10` are not justified per scientific product.

**Concrete consequence.** The repository can support “the latest six-commit refactor delta preserved common point numeric arrays,” but not “the current refactor is equivalent to original Citlali across pointing, OOF, beammap, and science.” A validation report can look green while skipping exactly the product or metadata affected by a change.

**Why architectural.** The validation matrix defines the safe change surface for an AI-maintained scientific system. It must follow behavior and product contracts, not merely the most convenient dataset.

**Smallest corrective direction.** Add a strict comparator mode that exits nonzero, matches FITS HDUs by stable identity, compares required WCS/scientific/config headers, NetCDF attributes and strings, and all required arrays in chunks; use an explicit volatile-field allowlist. Record product-specific tolerances approved by scientific owners. Require all TOD arrays for any timestream/config/provenance change. Use this change-to-gate matrix:

| Changed behavior | Minimum required gate |
| --- | --- |
| Pure parser/normalization/policy | Unit/reader boundary tests, invalid/boundary cases, static preflight, and real overlay fixture when overlay semantics are involved. |
| Output/provenance only | Strict current point including all TOD, full config/WCS metadata, strings/attributes, and zero unexpected log errors; add a mode run for mode-specific metadata. |
| RTC/PTC filtering, calibration, weighting, channel selection | Focused algorithm/fixture tests plus full TOD point; science and/or beammap gate according to the consumer. |
| Beammap grouping, priors, fitting, flags, detector products | Current matched-config beammap products, detector flags/tables, performance, and RSS. |
| Coadd/noise/filter/mapmaking | Current matched-config science products with owner-approved metrics/tolerances and performance. |
| OOF policy or execution | Current OOF runtime/product baseline. |
| HWPR/polarimetry | Enabled polarized baseline; until it exists, a fatal capability/preflight rejection. Disabled point and documentation alone are insufficient. |
| Concurrency/output/failure | Deterministic injected-failure integration tests plus an error-free current point run. |

**Scope and confidence.** **Present refactor and release gate. High confidence.** Exact equality need not be universal, but every tolerance must have a product-specific scientific reason.

### I5 — Scientific identities, units, schemas, and valid-state invariants remain too implicit

**Observation and evidence.** The typed config improves units and enums, but core runtime data still exposes ambiguous representations:

- `engine::Calib` stores APT columns as `std::map<std::string, Eigen::VectorXd>` and keeps units/descriptions separately (`include/citlali/core/engine/calib.h:14-60` and later members). Direct `operator[]` access can create an empty missing column; approximately 251 direct `.apt["..."]` uses remain.
- `Telescope` similarly exposes string-keyed series and many raw primitive fields (`include/citlali/core/engine/telescope.h`).
- `MapIndexState` uses raw `Eigen::Index`/integer vectors for maps-to-arrays, arrays-to-maps, and maps-to-Stokes (`include/citlali/core/pipeline/map_index_state.h:7-12`).
- Beammap flag thresholds are stored as positional vectors (`include/citlali/core/config/beammap_config.h:164-173`) and assigned sequentially while iterating `array_name_map` (`include/citlali/core/pipeline/beammap_config_fitting_flagging.h:139-155`). Their scientific meaning therefore depends on an implicit container-order contract.
- `MapBuffer` exposes dimensions, pixel size, WCS arrays, products, grouping, and unit strings publicly; several primitive members do not have declaration-site defaults (`include/citlali/core/mapmaking/map.h:51-98`).
- The validated branch history contains a concrete array-identity versus map/detector-index bug. The current corrected path resolves FWHM by `array_id` (`include/citlali/core/engine/detail/map_image_output_impl.h:64-72`).
- Typed models still contain parallel representations such as string and enum pixel-axis concepts, while some JINC/maximum-likelihood method details remain legacy-only.

**Concrete consequence.** A future contributor or AI agent can pass an array ID where a map index is expected, assume zero- versus one-based scan numbering, use a value in the wrong frame/unit, or silently materialize a missing detector-table column. These errors can compile and produce scientifically plausible output.

**Why architectural.** Identity, unit, dimensionality, coordinate frame, and missing-data policy are domain invariants. They belong at subsystem boundaries, not in historical knowledge or variable-name convention alone.

**Smallest corrective direction.** Add checked cold-boundary views and IDs—such as `DetectorTableView`, `ArrayId`, `ArrayName`, `DetectorUid`, `MapIndex`, and `ScanIndex1Based`—where values cross config/calibration/map/output subsystems. Key per-array thresholds by `ArrayName`/`ArrayId`, or explicitly validate and version a stable ordering convention at the parse boundary. Validate required columns, shapes, units, ordering, and referential integrity once, then resolve raw Eigen indices/references for hot loops. Replace magic sentinels such as negative reference-detector values with optionals. Do not wrap every integer inside measured inner loops unless profiling justifies it.

**Scope and confidence.** **Present broader refactor at touched boundaries; full table redesign may be follow-up. High confidence.** Document the complete conventions before migrating more analysis control.

### I6 — Tests, CI, dependencies, and build/version provenance are not reproducible gates

**Observation and evidence.** Several independent issues combine:

- `tests/CMakeLists.txt:5` declares `citlali_test EXCLUDE_FROM_ALL`; only the custom `check` target depends on it (`tests/CMakeLists.txt:21-27`). The current build cache has tests off and CTest reports zero tests.
- `.github/workflows/cmake-single-platform.yml:5-44` targets only `v4.x`, builds the default target, then invokes CTest without explicitly building the excluded test. It uses `actions/checkout@v3` and mutable `turtlebrowser/get-conan@main`; it does not run config preflight, validator self-tests, strict product validation, warnings, sanitizers, or static analysis.
- `CMakeLists.txt:9-16` fetches kidscpp using floating `GIT_TAG "v1.x"`; downstream/local helpers also rely on mutable branches and locally applied patches. No checked-in lock/manifest records exact dependency SHAs and patch state.
- Git-version headers are generated at configure time (`CMakeLists.txt:135-137`), so a no-reconfigure workflow can build newer code with stale embedded provenance. The inspected local executable reports an older dirty revision than the source tree.
- Top-level CMake has no Citlali install/export rules for the library target, executable, or public headers. The tree therefore does not yet produce a consumable installed library/CLI despite the stated library goal.
- Root build instructions and `CODEX.md` are stale or contradictory. `CODEX.md` says the toolchain is unavailable and local build/test should not be attempted, while `tools/macos/README.md` documents the working Apple Silicon route. Tracked user presets assume machine-specific paths.

**Concrete consequence.** A clean agent or CI runner cannot reliably reconstruct the reviewed binary, dependency graph, test result, or version string from the repository alone. Local success depends on workstation state and external fixture paths.

**Why architectural.** Reproducibility is part of scientific provenance and of the development interface. It determines whether the same source/config can be independently trusted.

**Smallest corrective direction.** Pin dependency commits/releases and record any patches declaratively. Provide a clean preset/toolchain route for each supported platform. Generate version information as a build dependency in a small translation unit so it stays current without invalidating the full PCH. Make CI explicitly build `check`/`citlali_test`, run CTest and hermetic config/validator tests, and exercise GCC/Clang Release plus at least one Debug/sanitizer lane. Update one canonical root instruction file and remove/redirect contradictory notes.

**Scope and confidence.** **Present refactor before production-ready status. High confidence.** Install/export work should follow stabilization of the non-CLI API, but the refactor cannot claim its “maintainable library and CLI” goal complete until a clean external smoke client can consume the installed result.

### I7 — The performance and memory policy is not demonstrated for beammap

**Observation and evidence.** The documented budget is approximately 3–5% wall-time regression with no material peak-memory increase. Available beammap logs show:

| Run | Wall time | Difference from cited baseline |
| --- | ---: | ---: |
| Original `beammap/citlali/reduced/redu01` | 3297.186 s | baseline |
| Refactor `redu07` | 3535.634 s | +7.2% |
| Refactor `redu08` | 3564.335 s | +8.1% |
| Refactor `redu09` | 3564.328 s | +8.1% |

These beammap candidates are all at `f278bd32`, 120 commits behind the reviewed code. The repeat candidates are consistent, but there is only one cited baseline, the environment/storage/cache schedule was not controlled, and peak RSS was not recorded. The result therefore proves neither current-tree causation nor current-tree compliance; it leaves the budget unresolved. The latest point pair is also not performance evidence: total log time changed from 61.659 to 61.051 s, while PTC first-to-last-chunk time changed from 39.459 to 41.356 s (+4.8%), with one uncontrolled, erroring run per revision.

Stage profiling is useful, but it currently writes an ECSV row per profiled scope through a process-global collector (`include/citlali/core/pipeline/stage_profile.h:23-79,106-147`). Its cost should be measured or made opt-in; this review does not attribute the beammap difference to profiling.

**Concrete consequence.** Performance regressions can accumulate under refactor labels, and an hour-scale mode makes later attribution expensive. Memory behavior is unknown.

**Why architectural.** Hot-path and diagnostic boundaries are core constraints of this HPC pipeline. They determine which abstractions and instrumentation are acceptable.

**Smallest corrective direction.** On the same node and storage, use the same data/config/build type/dependency SHAs/thread policy and an explicit cache/warm-up policy. Alternate baseline/candidate order, run at least one warm-up and three—preferably five—measured pairs, and report median, dispersion/IQR, per-stage timing, CPU/thread settings, I/O volume, and peak RSS. Meet the budget or obtain a documented scientific/engineering exception before adding more runtime-path change.

**Scope and confidence.** **Present release/performance gate. High confidence** that the budget is unproven; **low confidence** about the cause of the observed difference.

### I8 — Compact config needs real TolTECA overlay acceptance and source provenance

**Observation and evidence.** The 8/8 preflight result proves deterministic translation of representative fixtures into equivalent low-level leaves. It does not execute the operational path in which TolTECA recursively combines numbered profile, normal, target, and expert YAMLs. The fixtures principally model a single selected low-level section and rely on workstation-local inputs. List replacement, null/deletion behavior, aliases, unknown keys, multiple `reduce.steps`, and exact precedence are not covered end to end. Original config copies also lack a canonical ordered merge manifest, hashes, and a final merged snapshot; basenames can collide.

**Concrete consequence.** A compact config may pass static policy comparison yet produce a different operational `citlali_o*.yaml` under real overlay ordering, and a finished reduction may not retain enough input evidence to reconstruct the merge.

**Why architectural.** TolTECA is the production config boundary; its merge semantics and provenance are part of Citlali’s external contract even if implemented in another repository.

**Smallest corrective direction.** Retain the current preflight. Add hermetic acceptance fixtures that invoke the real TolTECA merge/generation path on complete numbered directories for pointing, OOF, beammap, and science, then compare canonical generated low-level YAML. Retain the exact compact/user and expert overlay bytes collision-safely, with merge order, role/precedence, hashes, and tool version; also retain the generated merged low-level request, canonical typed request, effective plan, and realized decisions. Coordinate template/catalog changes in `tolproj` rather than duplicating the merge engine here.

**Scope and confidence.** **Important and mandatory before compact-config production rollout; not a blocker for continued legacy low-level input. High confidence.**

### I9 — Active/legacy source and durable validation/science-change evidence are not reliably designated

**Observation and evidence.** Unbuilt `main_old.cpp` (1,123 lines), `mpi_main.cpp` (703), `kids_main.cpp` (406), and `lali_main.cpp` (230) remain beside active code. Empty/one-line `.cpp` stubs are listed as if they were imminent modules. Many validation reports live in `/private/tmp` and products live only under workstation paths. The review brief also acknowledges imported determinism, cleaning, and Wiener improvements from prior development; this review did not independently trace each source commit, expected scientific difference, or validation record.

**Concrete consequence.** Future agents can select obsolete entry points, an accepted result cannot be independently located or reconstructed from the repository record, and intended science changes can be mistaken for refactor regressions—or vice versa—during snapshot integration.

**Why architectural.** Active-source designation, migration status, intended behavior, and durable validation evidence are navigation and trust boundaries for an AI-maintained scientific codebase.

**Smallest corrective direction.** Mark/delete confirmed obsolete entry points or move them into an explicitly documented attic outside the active build tree. Remove placeholder sources until a real boundary is implemented. Check in a compact machine-readable validation ledger per accepted SHA/mode. Add an intended-science-change ledger recording source commit, rationale/behavior, affected modes/products, expected differences, and independent evidence for every non-structural import included in the integration snapshot.

**Scope and confidence.** **Present integration/documentation phase and required before snapshot integration. High confidence.**

### A1 — The 2,888-commit history needs a practical review/integration strategy

**Observation and evidence.** The branch has 2,888 no-merge commits; 1,800 change one file, the median commit changes one file and 11 lines, and only a small fraction touch tests or build/CI. These boundaries reflect the historical Unity-gated workflow more than reviewable subsystem decisions.

**Concrete consequence.** Reviewers cannot use the commit sequence as an efficient architectural narrative, and replaying/reworking it risks changing the only repeatedly exercised tree.

**Why architectural.** Integration strategy affects traceability and future change isolation, although it does not alter runtime correctness by itself.

**Smallest corrective direction.** Preserve and tag the granular branch for forensics. Review and integrate the exact validated tree as a snapshot with subsystem summaries and tree-hash verification; make subsequent work coherent subsystem PRs. Do not rewrite thousands of commits merely for aesthetics.

**Scope and confidence.** **Advisable integration/process improvement. High confidence.**

### A2 — The auxiliary R scaffold does not yet enforce a measured-channel contract

**Observation and evidence.** `AuxiliaryMeasuredStream` has public `channel`, `name`, `source_type`, unit, calibration, transfer, map, and diagnostic fields that can disagree. Its `TimestreamChannel` enum includes `synthetic_kernel` even though the type represents measured streams (`include/citlali/core/timestream/auxiliary_stream.h:14-45`). No RTC/PTC operator currently processes these streams.

**Concrete consequence.** Implementing R now could conflate a measured quadrature channel with synthetic kernels or apply primary transfer/calibration/flags without an explicit shape and science contract.

**Why architectural.** Auxiliary measured data need first-class identity and transfer semantics across TCData, RTC/PTC, learning, output, and calibration. A generic matrix plus independent labels does not enforce them.

**Smallest corrective direction.** Keep execution deferred. Before implementation, separate measured and synthetic channel types and specify row/column alignment, sampling, native units, calibration, flag propagation, linear-transfer applicability, missing-data behavior, and product provenance. Then add one end-to-end disabled/no-cost contract and one enabled reference dataset.

**Scope and confidence.** **Later R-specific phase. High confidence.** The current default-off scaffold does not block this refactor.

---

## F. Refactor completion criteria

The refactor needs a finite finish line. “All large headers made small” and “all legacy code modernized” are not useful completion conditions.

### F.1 Required definition of done for the current config phase

- [ ] **One authoritative validation contract with two gates.** A startup gate rejects every parser failure, unknown enum, non-finite required scalar/container element, range/domain violation, missing required key, inconsistent duplicate fact, and unknown/unconsumed key inside the Citlali-owned low-level schema. A second observation-scoped gate runs after calibration/observation resolution and before any scientific execution. Outer TolTECA-owned keys are outside Citlali’s rejection scope.
- [ ] **Explicit state classification.** Every executable low-level leaf has a typed owner, unit, allowed domain, mode applicability, and classification as requested, normalized/effective, observation-resolved, or realized.
- [ ] **Immutable request.** The accepted requested config is not overwritten by fallback, automatic grouping, calibration availability, or observation resolution.
- [ ] **One-way transition.** Normalization produces an effective plan; observation data produces an observation/realized plan; one narrow adapter populates legacy processors. No bidirectional mirror synchronization is introduced.
- [ ] **Parity is measured at the right phase.** For every migrated field, tests compare the adapter immediately after context-free resolution and again after observation resolution. Later processor-derived values are captured as realized metadata rather than treated as permanent parity with the effective plan. “Validate typed mirrors” must mean comparison as well as internal validation.
- [ ] **Observation config is atomic and scientifically complete.** Astrometry, photometry, source flux, calibration-dependent choices, coordinate/frame/range rules, finite vector elements, MJD ordering/coverage, and their diagnostics are built and validated as a complete per-observation value before mutating processor state. A second observation cannot inherit the first observation’s missing values.
- [ ] **No duplicate mutable facts.** String/enum aliases, duplicated enable flags, and requested/effective values are not stored as independently writable representations. Compatibility spellings live at the boundary.
- [ ] **No processor-owned YAML reads remain.** The config phase is not complete until every executable low-level leaf—including method-specific JINC/maximum-likelihood settings—is boundary-parsed and RTC, PTC, mapmaker, and mode code own no YAML parsing.
- [ ] **Versioned provenance.** Outputs or their durable run manifest preserve collision-safe exact source bytes plus ordered path/role/precedence/hashes, canonical merged low-level YAML, canonical typed request, effective plan, realized decisions, calibration sources, and schema/tool versions with unambiguous labels.
- [ ] **Behavior-appropriate validation.** Current point always compares complete TOD and metadata for timestream/output changes; beammap/science/OOF/polarimetry gates are required before their execution fields become typed-authoritative.
- [ ] **Zero unexplained errors.** Accepted reductions have no unallowlisted error log records, and output completeness is checked independently of the “done” marker.

Compact-config deployment is not required to close the typed-config ownership phase. If deployment remains deferred, record it as such. Before any production rollout, the additional I8 gate is mandatory: hermetic TolTECA numbered-overlay fixtures must cover pointing, OOF, beammap, and science, including expert overrides, list replacement, null/deletion rules, aliases, unknown keys, and multiple steps.

### F.2 Required definition of done for the broader structural refactor

- [ ] A non-CLI library entry accepts an explicit request/session input and returns a structured `ReductionResult` containing success/failure, products, diagnostics, and realized provenance. Only the CLI chooses process exit codes.
- [ ] No supported non-CLI path contains reachable process termination. Any remaining exit is mechanically unreachable, explicitly allowlisted legacy code with a removal owner; tests prove supported failures return through the library contract.
- [ ] Required output failures propagate; ordered output cancellation cannot deadlock; failure-injection tests prove cleanup, wake-up, partial-product policy, and repeated-run recovery.
- [ ] Run, iteration, observation, scan, and product-writer mutable state have explicit owners. Two sequential reductions in one process pass, including a run after injected failure.
- [ ] `Engine` is frozen as a compatibility adapter: no new public cross-cutting state, and new pipeline stages receive narrow contexts/plans rather than arbitrary mutable engine access.
- [ ] Public headers compile independently and a multi-translation-unit link test passes. Contextual implementation fragments are private, bounded, and grouped by coherent subsystem.
- [ ] At least one meaningful, measured cold subsystem is implemented in `.cpp`; the CLI no longer has to compile nearly the entire implementation graph. Header closure and clean/incremental build time improve without runtime or product change.
- [ ] Active, legacy, generated, and transitional files are mechanically or prominently distinguishable. Empty stubs and obsolete entry points no longer look active.
- [ ] Checked cold-boundary invariants cover detector-table schemas, shapes, array/detector/map identity, units, coordinate frames, index bases, and missing-data semantics for every subsystem touched by this refactor.
- [ ] A clean pinned-dependency build runs the real unit/integration suite and hermetic config tools in CI. The embedded version and dependency manifest identify the built tree without requiring reconfigure folklore.
- [ ] The stable non-CLI target has install/export rules and a clean external smoke client can consume the installed library, headers, and CLI. If external library consumption is intentionally not a goal, the project states that scope explicitly instead of claiming a consumable library.
- [ ] Current matched-config validation exists for pointing, beammap, science, and OOF. Enabled polarimetry either has an enabled reference gate or is mechanically rejected at capability/preflight until it does; documentation-only deferral is insufficient. Strict tools report all skipped/changed/error records and fail appropriately.
- [ ] Controlled performance measurements meet the 3–5% policy or carry an approved exception, and peak RSS plus profiler overhead are recorded.
- [ ] Every intentional non-structural/science import has a ledger entry with source commit, expected behavior/product differences, affected modes, and independent validation evidence.
- [ ] The durable documents in section H are current and identify any deliberately retained debt with an owner and exit condition.

### F.3 Desirable follow-up improvements

- Continue converting legacy calibration/telescope string maps into checked typed views as relevant subsystems change.
- Consolidate duplicated Lali/Pointing scan skeletons only after contract tests show a stable common sequence.
- Add warnings-as-errors for Citlali-owned code, focused static analysis, and periodic sanitizers after third-party warning boundaries are controlled.
- Move additional cold product writers and policy code into `.cpp` by coherent family, measuring each step.

### F.4 Work to stop or explicitly defer

**Stop now:**

- Splitting files solely to reduce line count.
- Migrating another analysis-control field to typed execution without the C2 fix, parity evidence, and a mode-appropriate gate.
- Treating a normal “done” marker or zero tool exit from the current comparators as proof of a valid run.
- Adding new public mutable state to `Engine` or new session data in function statics/singletons.
- Describing typed output as “runtime/effective” unless it comes from the effective/realized plan.
- Accumulating new workstation-only `/private/tmp` validation evidence without a durable ledger entry.

**Defer deliberately:**

- Full R/quadrature-channel execution.
- Broad RTC/PTC/clean/JINC/Wiener algorithm redesign or stylistic modernization.
- A wholesale conversion of all 145 exits in one patch; convert by boundary with failure tests.
- Concurrent multi-reduction support unless it is an actual operational requirement; require sequential reentrancy now.
- Compact-config production rollout and TolTECA catalog/template replacement until real overlay acceptance passes.
- A complete ABI-stable plugin framework, dependency-injection container, service registry, or other enterprise machinery without a demonstrated need.
- Rewriting thousands of historical commits for aesthetic reviewability. Preserve the forensic branch and integrate a verified tree.

---

## G. Architectural rules for this codebase

These rules should be recorded verbatim or with equivalent precision for future contributors and AI agents:

1. **YAML ends at the boundary.** Parse TolTECA/Citlali YAML once; processors, mapmakers, output algorithms, and hot loops must not read YAML.
2. **Request, effective plan, and realized metadata are different values.** The request is immutable; every automatic choice or fallback has an explicit result and reason; output labels which state it records.
3. **A migrated config fact has one authority.** During transition, one typed plan populates legacy state through one adapter and parity is asserted; never synchronize typed and legacy state in both directions.
4. **Library code never logs-and-forgets a required failure and never terminates the process.** Required failures reach `ReductionResult`; only the CLI selects an exit code. Optional-product failure policy must be explicit.
5. **Reduction state has a lifecycle owner.** No new function-static, process singleton, or unrelated global may hold run-, iteration-, observation-, scan-, or writer-specific mutable state.
6. **`Engine` is a compatibility adapter, not the destination architecture.** Add no new public cross-cutting fields; new stages take the narrowest explicit context/plan/result that expresses their contract.
7. **Scientific identity is not a raw interchangeable integer at subsystem boundaries.** Validate and distinguish array ID, array index, detector UID, detector index, map index, Stokes index, and one-based scan/chunk identity before entering hot loops.
8. **Every scientific interface states units, frame, shape, index base, and missing-data policy.** Non-finite values are invalid unless the domain explicitly models them; use optionals/domain states rather than magic sentinels.
9. **A public header compiles independently.** Contextual fragments are private implementation details, grouped by subsystem, and must not create ODR reliance on a single translation unit.
10. **Hot paths remain boring and measured.** No new filesystem access, YAML parsing, string-map lookup, logging, heap allocation, or virtual dispatch inside per-sample/per-detector/per-pixel loops without benchmark evidence and a scientific need.
11. **Validation follows the touched behavior.** Timestream changes compare complete TOD; beammap/science/OOF/polarimetry changes run their mode; provenance changes compare full relevant metadata; all accepted runs have zero unexpected errors.
12. **R is optional measured data, never a synthetic kernel by convenience.** Its channel identity, sample alignment, units, calibration, transfer, flags, and output provenance must be explicit before execution is added.

---

## H. Minimum durable documentation for future AI agents

The repository has abundant dated handoffs, but agents need a small canonical set. Do not copy the same facts into every document.

### H.1 Root `AGENTS.md`

Create a concise root file that contains only operational and non-negotiable guidance:

- supported build presets/toolchains and the exact fast/full commands;
- test, config-preflight, strict validation, and performance commands;
- which datasets are local versus durable and how to locate the validation ledger;
- the architectural rules in section G or a direct link to their canonical location;
- hot-path restrictions and change-to-validation routing;
- active branch/source locations and prominent “do not edit/execute” legacy paths;
- a reminder that only the CLI exits and that output errors cannot be ignored.

It should link to the architecture and scientific-conventions documents rather than repeat them. Replace or redirect the contradictory `CODEX.md`; do not maintain competing agent instructions.

### H.2 `doc/ARCHITECTURE.md`

Maintain one current, undated overview containing:

- the actual component/dependency map and allowed dependency direction;
- runtime data/control flow from TolTECA input through products;
- ownership/lifetimes for request, session, iteration, observation, scan, processors, maps, and writers;
- the stable library/CLI failure boundary;
- active extension points and which paths are compatibility adapters;
- rules for header versus `.cpp` implementation and hot versus cold code.

Generated HTML diagrams may derive from it, but the Markdown source is canonical.

### H.3 `doc/SCIENTIFIC_CONVENTIONS.md`

This should be owned jointly by engineering and the scientific lead and specify:

- array/detector/network/map/Stokes/scan identities and index bases;
- matrix dimensions and detector/sample ordering at RTC/PTC/map boundaries;
- units and coordinate frames, WCS/epoch/wrapping conventions, and conversions;
- calibration state, validity, missing data, sentinels, and non-finite policy;
- requested/effective/realized provenance semantics;
- determinism expectations and product-specific numerical acceptance metrics/tolerances;
- enabled/disabled/automatic/fallback meaning for scientifically consequential policy.

This is the primary defense against plausible but scientifically wrong AI changes.

### H.4 Focused architecture decision records

Create a small `doc/adr/` set only for consequential, hard-to-reconstruct choices. The initial ADRs should cover:

1. immutable requested config → effective plan → realized metadata → one-way legacy adapter;
2. structured reduction result, required/optional product failure, and CLI-only process exit;
3. `Engine` as a frozen compatibility adapter and the intended `ReductionSession` lifecycle;
4. first compiled-source boundary and header-fragment policy;
5. optional measured R-channel contract and why execution is deferred.

Each ADR should state context, decision, consequences, and supersession status. Implementation details that change frequently belong in architecture/status docs, not ADRs.

### H.5 `doc/REFACTOR_STATUS.md`

Replace the need to reconstruct progress across many handoffs with one short living status document:

- completed, transitional, and deferred boundaries;
- explicit authority for each dual representation and its removal condition;
- current Critical/Important debt with owner/exit test;
- the finite completion checklist from section F;
- the exact validated snapshot and a link to evidence.

Dated handoffs remain historical records, not the current source of truth.

### H.6 Machine-readable validation ledger

Add a compact checked-in ledger or manifest directory keyed by source SHA and mode. Each accepted run should record:

- source tree and dependency/toolchain SHAs, dirty status, build flags, host/runtime policy;
- content-addressed exact config/overlay sources with order, role, precedence, and hashes, plus canonical merged/requested/effective/realized config hashes;
- dataset identity, command, expected mode, and product inventory;
- strict comparator version/options, volatile allowlist, tolerances, skipped data, and verdict;
- log severity counts, wall/stage time, peak RSS, and thread policy;
- durable URI/path for larger artifacts.

Maintain a linked intended-science-change manifest for non-structural imports, recording the source commit, rationale, affected modes/products, expected numerical or schema change, and validation evidence. This prevents a structural snapshot comparison from silently treating intended algorithm changes as equivalence.

Do not check large scientific products into Git. Check the evidence needed to find and interpret them.

---

## I. Questions requiring domain-expert judgment

1. **Which products are required for each mode?** A failed RTC TOD, PTC TOD, diagnostic sidecar, fit table, or map may have different operational severity. The answer determines `ReductionResult`, cleanup, audit, and retry policy.

2. **What should disabled-filter provenance mean?** For `enabled=false, freq_Hz=0.1`, should output retain the requested 0.1, report effective 0.0, omit the effective frequency, or record both? This must be decided once for all disabled/automatic fields.

3. **Which requested, canonical, effective, and realized states must be retained for reproducibility?** This determines the config/provenance schema and whether a historical reduction can be reconstructed.

4. **Which KIDs TOD types are operationally supported?** `xs`, `rs`, `is`, and `qs` exist and typed selection reaches execution. Their units, calibration, validity, and allowed modes must be explicit; unsupported values should fail rather than fall back.

5. **What exactly should `ignore_hwpr: false` mean?** The typed model currently calls it `require`, but execution silently disables HWPR when data are absent. Decide whether the domain means require HWPR data, use it when available, or merely do not force-ignore it; then either enforce the requirement or rename the typed policy.

6. **Which fallbacks are scientifically acceptable and which are fatal?** Examples include missing beammap priors/fluxes, phase coercion, detector grouping fallback, unavailable calibration, incompatible product requests, and source-protection activation. Silent policy changes undermine reproducibility.

7. **What is the stable TolTEC identity contract?** Clarify hardware array IDs versus dense array indices/names, detector UID stability, network ordering, and whether these may change between observations. This governs typed boundary IDs and product joins.

8. **What are the authoritative coordinate conventions?** Specify RA/Dec frame, epoch, wrapping/range, tangent-plane sign, Alt/Az conventions, astrometry extrapolation, pixel-axis interpretation, and how they appear in WCS/provenance.

9. **Which zeros, negatives, NaNs, nulls, and “auto” tokens have domain meaning?** The config layer needs explicit optional/disabled/automatic types; it cannot safely infer semantics from historical sentinel values.

10. **How should beammap source flux behave across observations?** Is a missing per-array flux fatal, may it be inherited from a named catalog, or can processing proceed uncalibrated? It must never inherit incidental prior engine state.

11. **Is OOF a first-class reduction intent or a pointing execution strategy?** The answer affects typed config, mode routing, fixtures, product contracts, and validation ownership.

12. **What acceptance metrics and tolerances apply per product/mode?** Define exact-required fields, absolute/relative/ULP limits, fit/flag stability, integrated-flux/shape metrics, and acceptable nondeterminism for point, beammap, science, OOF, and enabled polarimetry.

13. **Must multiple reductions run concurrently in one process?** Sequential reentrancy is necessary; concurrent-session support would additionally constrain logger, dependency, memory, FFT, and NetCDF state and should not be inferred without a requirement.

14. **What is the scientific contract for measured R?** Specify alignment with X, units, calibration, linear operations that must be shared, nonlinear operations that must remain independent, flags, learning use, and whether R may ever make science maps.

---

## J. Recommended next actions

The next work should be five bounded phases. Each phase has an explicit exit gate; do not begin the next semantic migration merely because the code compiles.

### 1. Safety stabilization — immediate

Fix C1 and C2 without broad refactoring: fatal enum/non-finite validation; propagated required NetCDF failure; cancellable ordered writers; mode/config-derived schema/cardinality checks; and audits that fail on unexpected errors. Establish the minimum strict comparison behavior needed to identify products/HDUs stably, compare complete point TOD/metadata, and fail on skipped or changed required items. Add injected parse, non-finite, output-failure, cancellation, and repeated-run tests.

**Exit gate:** a deliberately failed write produces nonzero CLI status, no deadlock, a diagnosed partial-product disposition, and a subsequent in-process run succeeds; current point reruns with zero unexpected errors and strict complete-TOD/product comparison.

### 2. Config authority and provenance — current config phase

Implement immutable requested config, effective plan, observation/realized metadata, and one one-way legacy adapter. Correct current IIR/extinction/HWPR semantics with labels approved by the scientific owner. Make observation config atomic and domain-complete, fix stale beammap flux state, remove duplicate typed facts, and add phase-specific parity assertions. Add real TolTECA overlay fixtures in this phase only if compact rollout remains near-term; otherwise keep I8 as an explicit rollout blocker. Before changing shared session/output structure in phase 3, establish matched current pre-change characterization for every supported mode that phase 3 will touch.

**Exit gate:** the section F.1 checklist passes; every migrated field has request/effective/realized tests and mode-appropriate product/provenance evidence; current matched pre-change baselines exist for every supported mode affected by the next shared structural phase. No further analysis-control field migrates ahead of this gate.

### 3. Library/session and first compiled boundary — structural completion

Introduce the minimal non-CLI `ReductionSession`/`ReductionResult` around existing processors; convert every reachable process exit in supported non-CLI paths to that failure contract; move scan/profile/writer lifecycle out of statics; freeze `Engine`; add header self-compilation and multi-TU link tests; fix ODR hazards. Move the concrete enum parsing/validation tranche into `.cpp`, then one cold product/output tranche if measurements justify it. Consolidate contextual microheaders by coherent product family.

**Exit gate:** no supported non-CLI path can terminate the process; two sequential reductions and recovery after failure pass; public headers/self-link tests pass; the library result carries errors/products/provenance; header closure and rebuild time measurably improve; and strict before/after gates pass for pointing, beammap, science, OOF, and any other supported mode touched by the shared lifecycle/output changes, with no material runtime change.

### 4. Validation, performance, and reproducible build — release candidate

Turn the comparator into a complete strict acceptance mode, activate the real test suite in clean pinned CI, make config fixtures hermetic, fix build version/dependency provenance, and create current matched-config point, beammap, science, and OOF baselines. Add an enabled polarimetry reference gate or a fatal capability/preflight rejection until its dataset exists. Run the controlled paired beammap timing/RSS protocol.

**Exit gate:** all supported-mode strict gates pass at the same candidate SHA; no required items are silently skipped; performance/RSS meets policy or has an explicit approved exception; a clean agent can reproduce the build and fast tests from checked-in instructions.

### 5. Integration and closeout — no new architecture phase

Create the canonical documents, validation ledger, and intended-science-change manifest; add install/export plus an external smoke client for the now-stable library/CLI; mark/remove legacy and stub sources; preserve/tag the forensic branch; and integrate the exact validated tree as a snapshot with a short subsystem review summary. Put RTC/PTC core cleanup, further compiled-source moves, compact rollout, and R execution into separate prioritized follow-ups.

**Exit gate:** the section F.2 required checklist is closed or every explicit exception is owner-approved and recorded; an independent reviewer can identify the active architecture, exact evidence, supported modes, and remaining debt without reading the historical handoff chain.

### Final answer to the primary question

An external reviewer needs to know that this is **not a failed refactor and not yet a finished one**. Its conceptual architecture and tooling are substantially better, the latest six-commit typed-output batch preserved common point numeric arrays, and avoiding a new wholesale hot-path redesign is sound. The branch does include intentional prior science/algorithm imports that need their own evidence ledger, and current original-versus-refactor equivalence is not established. The remaining risk is concentrated and correctable: success/failure semantics, config authority/provenance, session ownership, enforceable compilation boundaries, strict cross-mode validation, reproducible builds, and controlled performance evidence. Resolve those in the order above and the branch should become a credible modern foundation without a wholesale rewrite.
