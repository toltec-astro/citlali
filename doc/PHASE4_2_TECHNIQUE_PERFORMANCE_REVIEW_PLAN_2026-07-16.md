# Phase 4.2 Technique And Performance Review Plan - 2026-07-16

## Goal

Review every active Citlali subsystem and answer two questions:

1. Is this an appropriate scientific and numerical technique for its stated
   purpose?
2. Is its implementation efficient enough for the real workloads and four
   supported clusters?

This is a comprehensive engineering and scientific census, not a presumption
that old code is wrong or that modern-looking code is better. The output is an
evidence-ranked decision for every active subsystem and a bounded remediation
backlog.

## Review Coverage

The compile graph and canonical architecture map define the active scope. The
review proceeds by coherent subsystem rather than by arbitrary file size:

1. CLI, session, operation selection, and failure reporting;
2. config parsing, plans, adapters, validation, and provenance;
3. raw input, KIDs conversion, calibration, telescope, and astrometry;
4. RTC chunking, flagging, despiking, filtering, extinction, and calibration;
5. PTC cleaning, weighting, diagnostics, and processed outputs;
6. map geometry, naive/JINC mapmaking, kernels, and weight accumulation;
7. fruit-loop feedback and iteration control;
8. pointing and OOF mode orchestration and fitting;
9. Beammap iteration, priors, fitting, flagging, and products;
10. source finding/fitting, coadd, filtering, Wiener processing, and noise
    products;
11. FITS, NetCDF, ECSV/CSV, manifests, compression/chunking, and publication;
12. concurrency, memory ownership, logging, profiling, and external-library
    use; and
13. tests, validation contracts, and operational diagnostics.

Every active component maps to exactly one primary review unit. Shared helpers
are reviewed with the subsystem that owns their scientific contract; genuinely
cross-cutting helpers are reviewed in unit 12.

## Fixed Review Record

Each unit produces a short checked record with these fields:

- purpose, callers, inputs, outputs, and scientific identity;
- current algorithm and why it is used;
- assumptions, units, frames, indexing, missing/non-finite behavior, and edge
  cases;
- comparison with OG behavior, documented scientific intent, and relevant
  primary literature or established library practice;
- numerical stability and failure behavior;
- asymptotic cost and dominant real workload dimensions;
- allocations, copies, Eigen expression behavior, cache locality, and data
  layout;
- parallel boundary, synchronization, determinism, and thread scaling;
- filesystem access, NetCDF/FITS chunking/compression, and output volume;
- existing stage profiles, benchmarks, and validation evidence;
- alternatives considered and their scientific/operational tradeoffs; and
- conclusion, confidence, owner, and next action.

Scientific appropriateness is not inferred from coding style. When the answer
depends on instrument behavior or analysis intent, the record asks the project
scientific owner directly and preserves that decision. Technical literature
claims use primary sources where available.

## Evidence Labels

Every concern receives one of these labels:

- **Observed:** demonstrated by a product, failure, profile, benchmark, or
  reproducible test.
- **Derived:** follows from algorithmic complexity, data shape, or an explicit
  contract, but is not yet measured on a real workload.
- **Suspected:** plausible from inspection and useful only as a request for
  measurement.
- **Owner decision:** a scientific or operational qualification supplied by
  the responsible person.

Do not implement a performance optimization from a suspected finding alone.
First add the smallest measurement that can confirm or reject it.

## Dispositions And Priority

Each reviewed element ends with one disposition:

- **Retain:** appropriate and proportionate; no action.
- **Clarify:** implementation is acceptable but its contract or rationale is
  not sufficiently explicit.
- **Measure:** a performance or numerical concern needs evidence.
- **Improve:** a bounded implementation change has a stated benefit and gate.
- **Scientific decision required:** alternatives change analysis meaning.
- **Retire:** inactive, duplicated, or superseded behavior should be removed.

Findings are prioritized independently of file age or size:

| Priority | Meaning |
| --- | --- |
| P0 | Correctness, data corruption, invalid success, or unsafe failure |
| P1 | Material scientific ambiguity, reproducibility risk, or severe operational failure |
| P2 | Measured performance/resource bottleneck or high-value maintainability barrier |
| P3 | Useful cleanup with no current correctness, scientific, or measured performance impact |

P0 and P1 findings are resolved or explicitly blocked before Phase 5. P2 work
is admitted when measurements show material benefit relative to implementation
and validation cost. P3 remains a follow-up unless it naturally disappears in
an accepted higher-priority change.

## Change Protocol

The census is performed before broad remediation. Each accepted change is a
separate bounded project with:

1. named behavior and owner;
2. before evidence;
3. a focused test or benchmark;
4. implementation without unrelated modernization;
5. after evidence and affected-mode validation; and
6. an intended-science-change entry and successor validation epoch when
   scientific outputs intentionally change.

OG equality remains useful for unchanged behavior. It is not the acceptance
criterion for an intentional scientific improvement. In that case, compare
against the accepted refactor snapshot, demonstrate that the intended products
changed as expected, and prove unaffected contracts remain intact.

## Performance Method

Static inspection identifies where to measure; it does not determine what is
slow. Use the existing stage profiler and run evidence first. Add microbenchmarks
only for isolated kernels whose workload can be represented faithfully. Use
real point, science, or Beammap runs for I/O, memory, parallel scaling, and
whole-stage conclusions.

Optimization review focuses on total user cost:

- wall time and scaling;
- peak and transient memory;
- filesystem traffic and product volume;
- repeated calibration/config work;
- avoidable copies and allocations in hot paths;
- deterministic behavior; and
- complexity imposed on future scientific changes.

A faster implementation is not accepted if it weakens numerical behavior,
failure reporting, determinism, or product meaning without an explicit owner
decision.

## Exit Gate

Phase 4.2 is complete when:

- every active component is assigned to a completed review record;
- no unowned P0 or P1 finding remains;
- suspected performance concerns have either been measured or explicitly
  deferred with a trigger;
- the dominant observed runtime and memory contributors have evidence-backed
  dispositions;
- accepted scientific changes have successor evidence rather than relaxed old
  baselines;
- the resulting P2/P3 backlog is prioritized and finite; and
- an external reviewer can see why major techniques were retained, changed,
  or deferred without repeating the entire code reading.

Completion does not require rewriting every subsystem or making every routine
maximally fast. It requires an informed, recorded decision at the level where
scientific and performance risk actually lives.
