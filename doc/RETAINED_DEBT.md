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
| D09 | Compact config is translation tooling, not production authoring authority. | TolTECA/config owner and Citlali engineering | Production template/catalog migration is requested | Hermetic real TolTECA overlay tests cover ordering, precedence, expert overrides, list replacement, null/deletion, aliases, unknown keys, and multiple reduction steps for all supported modes |
| D10 | Saved fruit-loop iterations use a flat reduction-directory sequence and concurrent jobs can collide without output-root exclusion. | TolTECA and Citlali engineering | Concurrent saved-iteration work or stronger run identity is scheduled | One atomically claimed run root owns explicit nested iteration IDs and a manifest with execution ID, selected final iteration, version, and effective-config digest while preserving final-product compatibility |
| D11 | The static library target is an internal composition/test boundary, not a supported installed external API or ABI. | Project owner | A real external client and support commitment are identified | Install/export, API/version policy, and a clean external smoke client are designed against that caller and tested on a supported environment |
| D12 | Concurrent reductions in one process are not supported; sequential recovery is supported. | Project owner and Citlali engineering | A real caller requires concurrent sessions and can state resource/isolation needs | Logger, FFTW, dependency, memory, output, signal, and mutable-state isolation are explicit and stress-tested without scientific or performance regression |
| D13 | Beammap peak RSS and profiler overhead do not have a dedicated controlled campaign; current monitoring uses an approved proportionality exception on shared VAST. | Citlali engineering and project owner | Sustained runtime regression, unexplained stage slowdown, memory failure, RSS near node capacity, material hot-path change, or a naturally required controlled run | Same-node evidence records matched workload/config/runtime, wall/stages, peak RSS, and profiling policy; triggered budgets pass or receive a new explicit disposition |
| D14 | The auxiliary approximately 50-observation Beammap corpus has not yet been re-reduced into a future-release performance census. | Project owner and Citlali engineering | Post-refactor historical Beammap re-reduction begins | A checked manifest records workload, config, binary, node, runtime, RSS, I/O, stages, outcome, and same-observation pairings; analysis avoids treating unlike observations as repeats |

## Governing Stop Rules

- D01-D04 do not justify more file splitting without a concrete owner,
  dependency, test, or measured build benefit.
- D03-D05 and D11 remain paused until the TolTECA build/integration direction
  is reviewed as one package.
- D06-D10 require scientific or upstream ownership before implementation.
- D12 is not inferred from the existence of a reusable sequential session.
- D13-D14 collect evidence when their triggers occur; they do not require
  speculative hour-scale reductions during Phase 4 closeout.

When an item closes, retain its row with the closing commit/evidence and mark
it closed, or supersede this register through a documented phase decision. Do
not silently delete the historical limitation.
