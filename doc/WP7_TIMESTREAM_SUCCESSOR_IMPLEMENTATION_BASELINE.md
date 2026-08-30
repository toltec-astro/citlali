# WP-7 Timestream Successor Implementation Baseline

Status: accepted design and implementation-prompt basis; implementation in progress

Version: 2

Recorded: 2026-08-29

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

Scientific authority and implementation evidence are deliberately distinct.
The WP-7 scientific contracts, owner decisions, validated scientific
semantics, and accepted audit results are authoritative constraints. Language
mode, physical layout, typed-view form, workspace organization, parallel
runtime, numerical libraries, and the earlier S0-S6 decomposition are prior
design work and evidence. Retain them only when current workload and benchmark
evidence still support them.

## Goal

Build one conforming CPU reference route with explicit products and owners:

```text
network-native paired x/r -> network-timed ALIGN relation
  -> network-keyed RTC conditioned raw-coordinate product
       +-> explicit RTC-only terminal completion
       `-> CAL / ordinary AST / MAP-JINC / network-level PTC-PCA

network-keyed product
  -> explicitly requested ALIGN common-analysis-grid relation
  -> array-wide PTC-PCA or an authorized cross-network RTC method
```

Ordinary RTC reconstructs and preserves each participant network's
occurrence/time identity. It does not consume, invoke, or publish a
cross-network common-analysis-grid association. ALIGN owns that separate
derived view and its strict-half admission rule when an actual consumer's
mathematics couples simultaneous measurements from more than one network.
Processing multiple networks or multiple detectors within one network is not
by itself such a requirement.

The first success criterion is scientific and implementation conformity, not
maximum portability or peak hardware performance. Performance, memory, and
determinism are nevertheless measured from the first executable slice so the
correct reference design does not accumulate avoidable hot-path debt.

## Scope

The successor owns the reachable WP-7 route through:

- native paired-`x/r` production and admission;
- reconstructed network timing for ordinary ALIGN, RTC, CAL, AST, MAP/JINC,
  and network-level PTC/PCA;
- an explicit ALIGN common-analysis-grid relation only for a named synchronous
  cross-network consumer, without making that relation a blanket ingress
  prerequisite;
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

- paired `x` and `r` numerical planes sharing detector and native-occurrence
  identity within each KIDs solver result;
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

Use compact axes and contiguous numerical storage as the default data-oriented
shape. Select the physical layout from the access pattern of each complete
route, not from the earlier baseline alone. Structure-of-arrays,
detector-major/time-contiguous storage, a bounded repack, or a tiled layout are
all implementation choices; retain or introduce one only when focused and
representative end-to-end evidence shows a net benefit after movement and
allocation costs.

### Scratch workspaces

When demonstrably useful, explicit execution-lifetime workspaces own filter
scratch, FFT arrays and plans, PTC matrices, decompositions, coefficient
vectors, and other reusable temporaries. They resize at cold boundaries and
are reused within their declared worker, run, or observation lifetime. A
workspace is not required when a direct bounded value or view is clearer and
equally efficient.

`std::pmr`, a general allocation framework, and a global workspace cache are
not baseline requirements. They require allocation-profile evidence and a
bounded owner before introduction.

### Numerical views

Kernels receive small typed views or stable handles that state value type,
constness, shape, indexing, and supported layout. With the successor direction
of at least C++23, `std::mdspan` is a candidate where multidimensional access
actually benefits from it; a simpler `std::span` or bounded private view is
preferred when it communicates the data more directly. A standard-library
facility is an implementation choice and does not become the scientific API.

Hot kernels should normally know their unit-stride dimension. A completely
general runtime-stride abstraction is not imposed on every loop.

## Chunk and stream invariants

Engineering chunks are storage and scheduling partitions only. They do not
redefine:

- native occurrence identity, or aligned occurrence identity after an
  explicit ALIGN projection has actually been requested;
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

## Execution direction

Parallelism follows the measured work decomposition. Public scientific
interfaces do not encode an OpenMP schedule, work-unit choice, or any other
threading runtime. Begin with the simplest serial or stage-local implementation
that provides a correct reference; compare plausible parallel policies on
representative complete-route workloads before selecting one.

Every selected implementation records effective and realized thread policy,
uses stable partition and merge order where floating-point accumulation
depends on order, prevents accidental nested library thread teams, and keeps
allocation, virtual dispatch, logging, YAML access, and string policy out of
established sample loops. OpenMP remains available because it is mature and
already integrated, but it is neither sacred nor a fixed architecture.

## Numerical-library baseline

### Linear algebra

Eigen is the incumbent and is a sensible first reference for many bounded
operations, but it is retained operation by operation. Distinguish large
covariance or matrix products from the many bounded time-local normal solves.
Compare an external BLAS/LAPACK provider, a smaller direct implementation, or
another mature library when the representative operation justifies it, and
include conversion, allocation, and threading costs in the comparison.

Factorization reuse keyed by admission masks is not assumed. Instrument mask
diversity, identical-mask run length, factorization time, solve time, and all
matrix dependencies before proposing a cache.

### Transforms

FFTW is the incumbent transform implementation and remains the reference while
it is the best measured fit. If retained, each RTC worker or execution
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
| **Direction** | Expected successor foundation | at least C++23; explicit ownership and lifetime; compact axes; contiguous storage where advantageous; lightweight views/handles; focused and representative end-to-end measurements |
| **Retain when justified** | Mature incumbent choices, not design authority | structure-of-arrays layouts; reusable workspaces; OpenMP; Eigen; FFTW |
| **Measure** | Prototype against a named workload before adoption or retention | physical layout; repack/tile strategy; parallel work decomposition; persistent teams; BLAS/LAPACK provider; FFT implementation and sharing/batching; factorization reuse; ThinLTO |
| **Permitted** | May be used when locally justified | `std::span`; `std::mdspan`; fixed-width typed cause masks; exact deployment CPU target; explicit aligned allocation |
| **Deferred** | Do not implement until an observed trigger exists | Highway or other explicit SIMD; Kokkos/GPU; PGO; bit-packed primary validity; `std::pmr`; general task graph; multiple numerical backends |
| **Rejected for the initial route** | Inconsistent with the accepted baseline | new implementation language; general tensor framework; mixed-precision scientific arithmetic; `-ffast-math`; multiple competing task runtimes; silent legacy fallback |

Moving an item between categories requires a dated baseline revision with the
measurement, affected boundary, validation result, and owner.

C++23 migration is coordinated with the application/toolchain build lane. A
bounded increment need not invent a C++23 abstraction when its direct data
model is already clear, but new successor architecture must not assume C++20
is the long-term language ceiling.

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

## Implementation increments and gates

The earlier S0-S6 labels remain dependency and planning history, not frozen
implementation architecture. The accepted first vertical increment is one
scientific unit with three coherent review boundaries:

1. paired ingress and native product semantics;
2. explicit identity RTC learn-consider-apply; and
3. RTC-only route and in-memory publication integration.

Representative real-data execution and fresh independent conformance review
gate the completed vertical increment; they are not separate scientific
milestones. Later work may regroup the historical slices when complete-route
design and measurements show a clearer boundary.

### S0: executable seam and measurement scaffold

- define explicit route selection and disabled/no-cost behavior;
- establish Citlali-owned views, workspace conventions, and performance
  fixtures; and
- prove the existing legacy route is unchanged.

### S1: paired ingress and RTC-only identity witness

- construct exact native `x/r` atomically;
- carry independent member validity, pair causes, identity, support, origin,
  and exposure facts;
- retain each network's native occurrence axis without invoking a common-grid
  projection or AST;
- implement the exact identity/pass-through RTC witness; and
- finalize the explicit RTC-only route without entering CAL, PTC, or MAP.

### S2: complete RTC plan and apply

- add immutable evidence, resolved plan, apply, and realized record;
- retain network-keyed occurrence, time, gap, and support axes through every
  ordinary operation, creating a new per-network output relation when sampling
  changes;
- for the first nonidentity sampling method, consume immutable AST-valid
  science-scan motion, admit occurrences at `v >= 1 arcsec/s`, and isolate
  bounded runs so slow or invalid support cannot influence retained outputs;
- derive immutable filter/factor plans per scan, TolTEC array, and exact input
  cadence from the authoritative circular diffraction-limited beam and the
  approved product-level passband, phase, alias, sampling, support, and edge
  limits; select the largest conforming factor and use `M=1` without sampling
  change, but with the new admission dispositions, when none above one passes
  and the input cadence remains adequate; otherwise produce no admitted
  ordinary astronomical product with
  `input_cadence_inadequate_for_science_band`;
- preserve per-network outputs when arrays realize different filters, factors,
  or cadences; this does not request a common analysis grid;
- implement paired pathology, transition guards, stable segments, reset, and
  replacement semantics; and
- port only conforming filter, notch, transfer, and decimation kernels.

The scan/array structure, corrected
[`wp7-rtc-scan-array-numerical-policy-v2`](WP7_RTC_SCAN_ARRAY_FILTER_BANK_OWNER_AUTHORITY_2026-08-30.md)
and bounded
[`wp7-ast-scan-motion-v1`](WP7_AST_SCAN_MOTION_OWNER_DECISION_PACKET_2026-08-30.md)
authority are accepted. The locally gated AST implementation still requires
representative conformance and exact-SHA review before nonidentity RTC,
followed by representative cleaned-noise PSD envelopes,
immutable pre-certified filter-bank artifacts, native-rate versus filtered
naive/JINC and OOF/fruitloops certification, and the bounded implementation
gates. Production performs a bank lookup and no runtime filter
synthesis, optimization, or detector-PSD estimation. Narrow sub-input-Nyquist
lines remain owned by line detection/mitigation. Legacy frequency/FWHM
constants, the historical `32 Hz` filter, and v1 Kaiser factor/tap estimates are
evidence, not defaults. See ADRs 0016 and 0017 and the 2026-08-29 RTC scan/array
owner authority.

### S3: distinct CAL

- ensure RTC remains in raw detector coordinates;
- implement exact APT factor and WVR/atmosphere/passband authority;
- preserve sample-local support and observation-wide quality separately; and
- retain raw-`r` parentage without applying unauthorized optical calibration
  to `r`.

### S4: VAL and contract PTC

- keep network-level PTC/PCA on each network's time axis;
- request an explicit ALIGN common-analysis-grid relation before array-wide
  PTC/PCA or another named operator whose estimates mathematically couple
  simultaneous measurements from more than one network;
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
