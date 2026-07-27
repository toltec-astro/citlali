# Citlali Retained Debt Register

## Status And Use

This is the canonical register of architectural and scientific limitations
deliberately retained at the current refactor boundary. Each item has a role
owner, a trigger that justifies reopening it, and an observable exit condition.

Retained debt is not permission for opportunistic cleanup. Work starts when
the named trigger exists and proceeds as a bounded project with validation
proportionate to its scientific and operational effect. Current phase
sequencing remains authoritative in [`REFACTOR_STATUS.md`](REFACTOR_STATUS.md).

Owners are roles rather than individual people so the register remains useful
when personnel change.

## Register

| ID | Retained debt and current disposition | Owner | Reopen trigger | Exit condition |
| --- | --- | --- | --- | --- |
| D01 | `Engine` remains a broad active compatibility aggregate; existing pipeline/header code still accesses it widely. It is frozen for growth. | Citlali engineering | A concrete feature, defect, stale-state hazard, or testability problem needs a narrower owner | The affected state has one explicit lifecycle owner and narrow request/context/result; no replacement state bag is added; focused and affected-mode gates pass |
| D02 | `citlali::pipeline` combines orchestration, policy, output, provenance, and compatibility helpers. | Citlali engineering | A bounded subsystem change cannot be understood or tested behind the current namespace/interface | The extracted family has a coherent responsibility and dependency direction, independent contract tests, and no textual-only fragmentation |
| D03 | The physical implementation is header-dominant; 171 contextual `engine/detail` fragments remain under the public include tree; not all historical headers have an isolation compile test. | Citlali engineering with TolTECA build owner | TolTECA's revised C++ integration/build direction is available | Supported public/private target boundaries are defined; the intended header matrix passes; contextual fragments are private or explicitly bounded; clean/incremental measurements are recorded |
| D04 | The first `.cpp` extraction reduced header size but did not demonstrate a compile-time improvement; broader cold boundaries are paused. | Citlali engineering with TolTECA build owner | The intended build topology and representative compile measurement are known | At least one coherent cold boundary materially reduces exposed dependency/compile cost without product or runtime regression |
| D05 | Dependency pins, CI build lanes, embedded version regeneration, and cluster build reproducibility are not closed. | Citlali engineering with TolTECA build owner | The revised TolTECA integration model is ready for review | One checked-in, pinned supported lane performs a clean build and runs CTest, config preflight, baseline self-tests, and version/dependency identity checks from canonical instructions |
| D06 | Some NetCDF products lack complete unit/fill metadata and some ECSV units live only in table metadata. Current flags and conventions remain authoritative. | Citlali engineering and scientific owner | A consumer, schema revision, or new product requires stronger machine-readable semantics | A versioned successor schema states units, missing/fill behavior, migration compatibility, and passes product-contract plus affected-mode validation |
| D07 | Enabled polarimetry is planned but mechanically rejected and has no supported reference dataset or product contract. | Scientific owner and Citlali engineering | Approved polarimetry/HWPR algorithm, dataset, calibration, and product semantics exist | Enabled preflight, execution, products, provenance, and a reference validation profile pass; until then rejection remains mandatory |
| D08 | The optional measured R channel is structure-only; its scientific and processing semantics are not approved. | Scientific owner and Citlali engineering | A measured-channel project has an approved dataset and scientific questions | ADR 0005 is superseded by an explicit shape/unit/alignment/calibration/transfer/flag/product contract with disabled/no-cost and enabled reference gates |
| D09 | **Closed 2026-07-23:** `phase4.1-v2.1` provides four human-facing numbered-YAML mode kits, vendored behind TolPROJ's opt-in `--refactor` path while preserving legacy behavior. Hermetic overlay and exact-policy gates pass, and fresh native Unity projects completed point, OOF, Beammap, and science smoke reductions with requested products and no unexpected errors. | TolTECA/config owner and Citlali engineering | Closed | Four ready-to-copy mode kits and hermetic real TolTECA overlay tests cover ordering, precedence, expert overrides, list replacement, null/deletion, aliases, unknown keys, and multiple reduction steps; accepted low-level equivalence and mode smoke gates pass |
| D10 | Saved fruit-loop iterations still use a flat reduction-directory sequence. Output-root exclusion prevents concurrent publication collisions, and ADR 0006 now provides state-complete cross-job continuation with absolute iteration identity, but one invocation still lacks a stable nested run identity and whole-science-config digest. | TolTECA and Citlali engineering | Concurrent saved-iteration work or stronger run identity is scheduled | One atomically claimed run root owns explicit nested iteration IDs and a manifest with execution ID, selected final iteration, version, and effective-config digest while preserving the ADR 0006 restart contract and final-product compatibility |
| D11 | The static library target is an internal composition/test boundary, not a supported installed external API or ABI. | Project owner | A real external client and support commitment are identified | Install/export, API/version policy, and a clean external smoke client are designed against that caller and tested on a supported environment |
| D12 | Concurrent reductions in one process are not supported; sequential recovery is supported. | Project owner and Citlali engineering | A real caller requires concurrent sessions and can state resource/isolation needs | Logger, FFTW, dependency, memory, output, signal, and mutable-state isolation are explicit and stress-tested without scientific or performance regression |
| D13 | Beammap peak RSS and profiler overhead do not have a dedicated controlled campaign; current monitoring uses an approved proportionality exception on shared VAST. | Citlali engineering and project owner | Sustained runtime regression, unexplained stage slowdown, memory failure, RSS near node capacity, material hot-path change, or a naturally required controlled run | Same-node evidence records matched workload/config/runtime, wall/stages, peak RSS, and profiling policy; triggered budgets pass or receive a new explicit disposition |
| D14 | **Collection-ready 2026-07-23:** the manifest, analyzer, and protocol exist, but the auxiliary approximately 50-observation Beammap corpus has not yet been re-reduced into a future-release performance census. | Project owner and Citlali engineering | Post-refactor historical Beammap re-reduction begins | A filled checked manifest records workload, config, binary, node/storage, runtime, RSS, I/O, stages, output volume, outcome, and explicit same-observation pairings; every expected run is complete and accepted; analysis avoids treating unlike observations as repeats |
| D15 | **Complete population evidence recorded 2026-07-27:** fruit-loop execution has activation validation and a hard iteration bound, but no implemented scientific convergence statistic; a valid active feedback request therefore runs to `max_iters`. All 108 quality-stratified observations and 3,240 array maps now pass the collection audit. An offline V0 candidate uses morphology-aware 3% amplitude, separate PSF/centroid/map/weight/support/background guards, no evaluation before iteration 6, and two consecutive all-array passes. It resolves 57/108; 23 failures are measurement-limited and 28 retain measurable but unresolved trajectories. The historical pointing-table `sig2noise` dynamic-range value is excluded, planet disks are convolved with each realized kernel, and no production policy is approved. | Scientific owner and Citlali engineering | **Triggered:** the full NGC4449 run demonstrated material cost from iteration policy | Continue the 28 trajectory cases, resolve the censored-PSF measurement class, approve separate scientific tolerances and provenance fields, and demonstrate early stopping without changing accepted final science beyond the successor profile |
| D16 | **Closed 2026-07-24:** TolPROJ derives refactor Slurm CPUs from the preserved runtime config, synchronizes explicit overrides, and validates before submission. Citlali independently discovers Slurm/affinity/hardware availability, caps oversubscribed thread plans with one warning, continues, and records requested/available/effective/realized state in runtime provenance V2. At `d339053cc`, the matching Unity smoke retained six requested/effective threads, while the intentional direct submission capped 12 requested threads to six affinity-available threads with exactly one resource warning. Both completed all 12 PTC chunks with valid V2 provenance and exact non-profile products against the predecessor. | TolTECA runtime owner and Citlali engineering | Closed | Local gates, the matching Unity smoke, and the intentional direct-submit mismatch all pass |

## Governing Stop Rules

- D01-D04 do not justify more file splitting without a concrete owner,
  dependency, test, or measured build benefit.
- D03-D05 and D11 remain paused until the TolTECA build/integration direction
  is reviewed as one package.
- D06-D08 and D10 require scientific or upstream ownership before
  implementation. D09 is closed by the bounded Phase 4.1 work.
- D12 is not inferred from the existence of a reusable sequential session.
- D13-D14 collect evidence when their triggers occur; they do not require
  speculative hour-scale reductions during Phase 4 closeout.

When an item closes, retain its row with the closing commit/evidence and mark
it closed, or supersede this register through a documented phase decision. Do
not silently delete the historical limitation.
