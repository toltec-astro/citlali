# WP-7 Timestream Successor Implementation Baseline

Status: accepted implementation baseline; implementation not started

Version: 1

Recorded: 2026-08-26

Work name: **Citlali WP-7 Timestream Successor Implementation**
Work classification: **bounded subsystem succession**

## Status and authority

This document is the living implementation baseline for the WP-7 successor.
It records how to construct and migrate the new timestream execution spine. It
does not restate, extend, or reopen the scientific contract.

Authority applies in this order:

1. the closed WP-7.1 scientific authorities and scenarios bound by
   [`../validation/wp7_timestream_successor_authority.json`](../validation/wp7_timestream_successor_authority.json);
2. [ADR 0014](adr/0014-wp7-timestream-successor.md), which selects bounded
   subsystem succession and its migration boundary;
3. this living implementation baseline;
4. slice-specific implementation plans, benchmarks, and handoff evidence; and
5. legacy code, comments, and tests, which are evidence only when they agree
   with the higher authorities.

The contract is closed for its approved scope. Implementation work tests
conformity and may not broaden intentionally unavailable response, covariance,
stronger-tier, generic-exposure, or external-consumer capabilities.

## Goal

Build one conforming CPU reference route with explicit products and owners:

```text
native paired x/r
  -> ALIGN paired occurrence and required AST coordinates
  -> RTC conditioned raw-coordinate product
       +-> explicit RTC-only terminal completion
       `-> CAL calibrated ordinary x with retained raw-r parentage
            -> VAL decisions for each exact PTC use
            -> PTC configured-rank group-local product
                 -> existing downstream application boundary
```

The first success criterion is scientific and implementation conformity, not
maximum portability or peak hardware performance. Performance, memory, and
determinism are nevertheless measured from the first executable slice so the
correct reference design does not accumulate avoidable hot-path debt.

## Scope

The successor owns the reachable WP-7 route through:

- native paired-`x/r` production and admission;
- ALIGN occurrence mapping and the AST roles required on ALIGN and RTC grids;
- RTC evidence, immutable planning, apply state, causes, stable segments,
  filtering, transfer, phase-zero selection, and logical completion;
- the explicit RTC-only terminal route;
- CAL factor, WVR, atmosphere/passband, support, quality, identity, and
  once-only application;
- VAL cause-preserving named-use evaluation;
- PTC group construction, fit, load, application, response-companion, output
  support, rank, and full-rank failure behavior; and
- atomic handoff to established publication or downstream application
  boundaries.

MAP, post-MAP processing, FruitLoops, Beammap algorithms, generalized
uncertainty/covariance, stronger response tiers, polarimetry, and new external
consumers are outside this baseline except for narrowly required integration
tests.

## Retain, implement, and adapt

| Disposition | Content |
| --- | --- |
| **Retain** | CLI and session boundaries; typed configuration and validation; raw I/O; exact native identity and APT relation; reconstructed per-network time; exact-time pointing carriers; gap/run partitioning; phase-zero decimation and support; group-local cohort construction; required-output failure propagation; atomic publication; bounded provenance; test and performance harnesses |
| **Implement anew** | paired `x/r` product; member-local and pair state; immutable RTC plan/apply model; distinct CAL stage and frozen numerical operator; use-specific VAL evaluator; contract PTC operator; RTC-only terminal route; origin and exposure lineage required by WP-7 |
| **Adapt behind new contracts** | mature despiking, filtering, notch, transfer, FFT, decimation, and applicable linear-algebra kernels whose exact semantics conform |
| **Do not carry forward as architecture** | broad `Engine` ownership; single-stream carrier; calibration inside RTC; one generic exclusion mask for all PTC uses; silent fallback between legacy and successor semantics; ambient mutable processor state |

Reusing a numerical body does not imply reusing its orchestration, state
ownership, flags, support policy, or stage placement.

## Product and lifetime model

The implementation separates three categories.

### Scientific product storage

The scientific carrier owns the minimum state that must survive a package
boundary. Its logical content includes:

- `x` and `r` numerical planes with common occurrence identity;
- coordinate-local payload availability and validity;
- typed `x`-local, `r`-local, and pair-wide causes;
- representative origin and replacement state;
- stable detector, network, time, run, segment, and occurrence relations;
- exact support and parent references;
- acquisition and valid-original exposure lineage where required; and
- response, quality, and downstream-use references required by the contract.

Detector identity belongs on the detector axis and time identity belongs on
the time axis. They are not repeated in every sample cell. Cause storage uses
typed fixed-width masks or another equally compact representation, but cause
accumulation must remain distinct from scientific admission. A VAL decision is
not encoded as a producer cause.

The baseline uses aligned contiguous structure-of-arrays storage. The first
implementation selects one explicit physical layout after a focused prototype
of the RTC and PTC hot kernels. Detector-major, time-contiguous storage is the
initial candidate, not a frozen decision. A full CAL-to-PTC repack or tiled
layout is added only after end-to-end evidence shows a net benefit.

### Scratch workspaces

Explicit execution-lifetime workspaces own filter scratch, FFT arrays and
plans, PTC matrices, decompositions, coefficient vectors, and other reusable
temporaries. They resize at cold boundaries and are reused within their
declared worker, run, or observation lifetime.

`std::pmr`, a general allocation framework, and a global workspace cache are
not baseline requirements. They require allocation-profile evidence and a
bounded owner before introduction.

### Numerical views

Kernels receive small Citlali-owned typed views that state value type,
constness, shape, indexing, and supported layout. A C++20-compatible `mdspan`
implementation or a small private view may implement those types after a
toolchain probe. `std::mdspan` itself waits for an application C++23 decision.
A standard-library facility is an implementation choice and does not become
the scientific API.

Hot kernels should normally know their unit-stride dimension. A completely
general runtime-stride abstraction is not imposed on every loop.

## Chunk and stream invariants

Engineering chunks are storage and scheduling partitions only. They do not
redefine:

- native or aligned occurrence identity;
- RTC plan evidence or accepted level shifts;
- stable-segment boundaries and filter reset state;
- phase-zero representative selection;
- complete logical RTC support;
- CAL or PTC parentage; or
- acquisition, valid-original, or independent-exposure identity.

The same declared input domain processed under different allowed engineering
chunk partitions must satisfy the contract's chunk-boundary-invariance rule.
Chunk-owned allocation is permitted; chunk-relative scientific semantics are
not.

## Execution baseline

The initial execution backend is OpenMP with one controlled parallel layer.
The public scientific interfaces do not encode an OpenMP schedule or work-unit
choice.

The first implementation shall use:

- ordinary ordered stages with stage-local data-parallel loops;
- no nested OpenMP regions;
- Eigen, BLAS, and FFTW single-threaded inside an active outer parallel
  region;
- explicit effective and realized thread policy;
- stable partition and merge order where floating-point accumulation depends
  on order; and
- no allocation, virtual dispatch, logging, YAML access, or string policy in
  established sample loops.

GRPPI is not part of the successor execution spine. Whether separate OpenMP
regions or one persistent team is faster is a measured private implementation
choice.

## Numerical-library baseline

### Eigen

Eigen remains the default linear-algebra implementation. Implement the correct
PTC operation in Eigen first. Distinguish large covariance or matrix products
from the many bounded time-local normal solves. An external BLAS/LAPACK
provider is evaluated only for a measured large operation; it is not a new
baseline dependency and does not introduce nested threading.

Factorization reuse keyed by admission masks is not assumed. Instrument mask
diversity, identical-mask run length, factorization time, solve time, and all
matrix dependencies before proposing a cache.

### FFTW

FFTW remains the transform implementation. Each RTC worker or execution
workspace owns the small explicit set of compatible plans it needs. Plans are
created and destroyed at cold lifetime boundaries and reused for compatible
detectors and chunks.

Shared-plan new-array execution, per-worker plans, batched transforms, and
persistent wisdom are compared only when transform shape and segment
structure make the comparison representative. A process-global generalized
plan cache is not part of the baseline.

## Technology decision matrix

| Category | Meaning | Current contents |
| --- | --- | --- |
| **Baseline** | Required in the first conforming implementation | current application C++20 mode; aligned contiguous structure-of-arrays storage; Citlali typed views; explicit workspaces; one OpenMP layer; Eigen; FFTW; focused and end-to-end measurements |
| **Permitted** | May be used without changing the architecture when locally justified | a C++20-compatible `mdspan` implementation behind Citlali views; fixed-width typed cause masks; exact deployment CPU target; explicit aligned allocation |
| **Measure** | Prototype against a named workload before adoption | physical layout; repack/tile strategy; persistent OpenMP team; BLAS/LAPACK provider; FFTW sharing or batching; factorization reuse; ThinLTO |
| **Deferred** | Do not implement until an observed trigger exists | Highway or other explicit SIMD; Kokkos/GPU; PGO; bit-packed primary validity; `std::pmr`; general task graph; multiple numerical backends |
| **Rejected for the initial route** | Inconsistent with the accepted baseline | new implementation language; general tensor framework; mixed-precision scientific arithmetic; `-ffast-math`; multiple competing task runtimes; silent legacy fallback |

Moving an item between categories requires a dated baseline revision with the
measurement, affected boundary, validation result, and owner.

C++23 adoption is governed as an application/toolchain compatibility decision.
It is not required to implement or optimize this route.

## Determinism and numerical policy

For a fixed build, input, configuration, and declared execution policy, output
must not depend on worker arrival order or an unspecified schedule. Each
contract scenario or validation profile states whether acceptance requires
bitwise identity, exact discrete identity and support with floating-point
tolerance, or another named numerical policy.

Every performance comparison also checks scientific values, identities,
causes, supports, rank/failure results, and required products. A faster result
that changes scientific meaning or becomes schedule-dependent is not an
accepted optimization.

## Performance evidence

Performance evidence has two levels.

### Focused kernels

Register representative cases for:

- paired carrier construction and cause propagation;
- RTC plan resolution and apply;
- filtering and one FFT configuration;
- CAL WVR/atmosphere evaluation and multiplication;
- VAL decision evaluation;
- PTC basis/loading construction;
- one time-local normal solve and full-rank failure;
- any proposed layout conversion; and
- required RTC-only finalization.

Record time, processed samples and detectors, bytes, allocation counts where
available, and relevant hardware counters. Microbenchmarks guide local work;
they do not authorize an application change alone.

### Representative routes

Measure at least the RTC-only route and the ordinary RTC-to-PTC route across:

- clean and heavily excluded support;
- representative network, detector, sample, segment, and rank sizes;
- more than one engineering chunk partition;
- serial and supported OpenMP thread counts; and
- required output disabled and enabled where output cost is relevant.

Point, science, and Beammap application reductions provide later integration
evidence. Their mapmaking and I/O time must remain separated from successor
timestream stage time.

Retain wall time, CPU time, peak RSS, throughput, stage totals, input/config
identity, compiler/dependency identity, effective/realized threads, success,
serious-log counts, and scientific comparison results.

## Implementation slices and gates

Slices are dependency-ordered and remain independently reviewable.

### S0: executable seam and measurement scaffold

- define explicit route selection and disabled/no-cost behavior;
- establish Citlali-owned views, workspace conventions, and performance
  fixtures; and
- prove the existing legacy route is unchanged.

### S1: paired ingress and RTC-only identity witness

- construct exact native `x/r` atomically;
- carry independent member validity, pair causes, identity, support, origin,
  and exposure facts;
- implement the exact identity/pass-through RTC witness; and
- finalize the explicit RTC-only route without entering CAL, PTC, or MAP.

### S2: complete RTC plan and apply

- add immutable evidence, resolved plan, apply, and realized record;
- implement paired pathology, transition guards, stable segments, reset, and
  replacement semantics; and
- port only conforming filter, notch, transfer, and decimation kernels.

### S3: distinct CAL

- ensure RTC remains in raw detector coordinates;
- implement exact APT factor and WVR/atmosphere/passband authority;
- preserve sample-local support and observation-wide quality separately; and
- retain raw-`r` parentage without applying unauthorized optical calibration
  to `r`.

### S4: VAL and contract PTC

- implement immutable registry/source binding and the five named PTC uses;
- construct exact fit, loading, application, response, and output supports;
- implement configured positive rank, centering, mask-aware time-local solve,
  frozen tolerance, and full-rank guard; and
- keep post-fit diagnostics advisory unless separately authorized.

### S5: publication and application integration

- publish required bounded identities, causes, plans, decisions, and products
  atomically;
- connect the ordinary product to the existing downstream boundary without
  granting MAP authority; and
- prove failure leaves no false completion record.

### S6: activation candidate

- run the locked WP-7 scenario suite on the production path;
- demonstrate order, parallel, and chunk isolation;
- complete representative performance and memory comparisons;
- run affected local gates and owner-run operational validation; and
- create the required successor validation epoch and science-change records
  for intentionally changed products.

No slice weakens a contract or skips a required scenario because the legacy
implementation cannot satisfy it. The legacy route remains the rollback path
until a separate activation decision. Legacy retirement is a later bounded
cleanup after accepted operational evidence.

## Change protocol

For every change:

1. name the contract propositions and scenarios;
2. identify the product and lifecycle owner;
3. record the before state and smallest useful benchmark;
4. implement without unrelated modernization;
5. run focused conformance and determinism tests;
6. measure affected hot paths and memory when material;
7. run affected integration and product gates; and
8. update this baseline only when a decision category or governing boundary
   changes.

Conversation history is not authority. New decisions enter the ADR, this
baseline, a versioned executable contract, or a dated handoff as appropriate.

## Evidence and navigation

- [ADR 0014](adr/0014-wp7-timestream-successor.md)
- [WP-7 authority bindings](../validation/wp7_timestream_successor_authority.json)
- [Citlali architecture](ARCHITECTURE.md)
- [Current status](REFACTOR_STATUS.md)
- [Technique and performance review protocol](PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md)
- [Active performance protocol](PHASE4_PERFORMANCE_PROTOCOL_2026-07-16.md)
