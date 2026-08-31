# ADR 0017: WP-7 timestream successor implementation

> Canonical numbering note: this decision first appeared as divergent ADR
> 0014 at `doc/adr/0014-wp7-timestream-successor.md`, introduced by
> `46824f7deeec56af786c877af7c53ae7cd55bef9`. It entered canonical application
> ancestry as ADR 0017.
> The historical number remains a provenance locator only.

Status: accepted 2026-08-26; canonical program active; application
implementation held pending governance reconciliation

Decision owners: Citlali project owner, scientific owner, and engineering

## Context

The closed WP-7.1 scientific-contract packet defines one ordinary native
paired-readout route through ALIGN, AST, RTC, CAL, PTC, and VAL, together with
an explicit RTC-only terminal route. The independent successor audit found the
bounded contract architecture ready at `TS-A`, `TS-S`, and `TS-C` and found no
scientific contradiction or regression within that scope.

The subsequent read-only implementation review found that the current
production timestream route does not conform to that contract. Its primary
gaps are structural rather than local: production carries one selected stream
instead of a paired `x/r` occurrence, executes calibration inside RTC, lacks
the approved CAL and VAL operators, uses a noncontract PTC application, and
has no honest RTC-only terminal route. The review classified four findings as
BLOCKER and seven as MAJOR.

The current application nevertheless contains substantial validated
infrastructure worth retaining: CLI and session boundaries, typed
configuration, native identities, APT relations, reconstructed network time,
pointing carriers, gap-bounded RTC support, phase-zero decimation, group-local
PTC cohorts, output publication, provenance, failure propagation, and mature
numerical kernels.

Patching all WP-7 semantics into the broad legacy `Engine`, `RTCProc`, and
`PTCProc` orchestration would create overlapping scientific models and expand
a compatibility aggregate that is already frozen for growth. Replacing the
entire application would discard validated operational behavior and repeat
unrelated integration work.

## Decision

The work is named the **Citlali WP-7 Timestream Successor Implementation**,
shortened to **the WP-7 successor**. It is classified as a **bounded subsystem
succession**: a new implementation of the scientific execution spine inside
the existing Citlali application, not a whole-application rewrite and not a
behavior-preserving refactor of the legacy timestream processors.

The successor owns two explicit routes from the same native paired product:

```text
native paired x/r
  +-> native-axis identity RTC
  |    `-> explicit RTC-only terminal completion
  `-> ALIGN common-slot projection and required AST roles
       `-> complete RTC
            -> CAL
            -> PTC with VAL-owned named-use decisions
            -> existing downstream application boundary
```

The first identity RTC witness consumes reconstructed native network timing,
not common-slot admission. It retains each network's exact native occurrence
axis and must not invoke or publish a cross-network projection. ALIGN remains
the sole owner of common-slot association and its strict half-cadence rule;
that projection becomes a prerequisite only for an operation whose scientific
operator requires cross-network simultaneity, such as PTC PCA or a future
cross-detector RTC learner.

The successor implementation shall:

1. remain in the Citlali repository and reuse the established CLI, session,
   configuration, input, output, provenance, and validation infrastructure;
2. introduce bounded typed products, plans, workspaces, and stage interfaces
   for the approved route rather than adding cross-cutting state to `Engine`;
3. use one-way adapters only at the native ingress, legacy-comparison, and
   downstream publication or application boundaries;
4. preserve the current production route as a comparison and rollback path
   until the successor passes its conformance, determinism, performance,
   product, and operational gates;
5. keep route selection explicit and fail closed rather than falling back
   silently between legacy and successor scientific semantics;
6. port mature numerical kernels only after their exact input, output,
   support, response, and state behavior is shown to satisfy the approved
   contract; and
7. treat activation and later retirement of the legacy route as separate
   decisions supported by accepted evidence.

Logical stage boundaries do not require a full materialized copy at every
boundary. Scientific products, temporary workspaces, and non-owning numerical
views have distinct lifetimes. Engineering chunks must not redefine
occurrence identity, retained support, filter state, or exposure lineage.

## Implementation-technology direction

The scientific contracts, owner decisions, validated semantics, and accepted
audit results are fixed authority. The implementation choices recorded by the
earlier WP-7 baseline are design evidence, not immutable architecture.

The successor direction is at least C++23. Adopt the language/toolchain change
through the application build lane, while allowing a bounded increment that
uses no C++23-only facility to remain source-compatible with the current build
during migration. Use clean data-oriented ownership, compact axes, contiguous
numerical storage where it benefits the workload, and explicit immutable
views or stable handles for referenced producer facts. Do not duplicate heavy
identity, support, pointing, or provenance per detector-sample cell.

Physical layout, typed-view form, workspace lifetime, allocation strategy,
parallel decomposition, and numerical-library selection remain measured
implementation decisions. Reusable workspaces are retained when they reduce
allocation and clarify ownership. Eigen, FFTW, OpenMP, BLAS/LAPACK, explicit
SIMD, or another mature implementation is selected only where representative
kernel and end-to-end evidence shows it remains the best fit. Public scientific
interfaces do not encode a particular threading runtime or library.

Mixed-precision scientific arithmetic, `-ffast-math`, a new implementation
language, a general tensor framework, and multiple competing task runtimes
remain outside the accepted initial direction absent a separately approved
scientific and operational case.

The live program and gate sequence is
[`../WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md`](../WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md).
The earlier
[`implementation baseline`](../WP7_TIMESTREAM_SUCCESSOR_IMPLEMENTATION_BASELINE.md)
is preserved as divergent design and replay evidence rather than current
sequencing authority.

## Compatibility and migration

The legacy route remains authoritative for existing accepted application
behavior until an explicitly selected successor candidate passes its gates.
The successor may run in isolated tests, diagnostic comparison, or an explicit
opt-in route without changing legacy defaults.

Each migration slice must have:

- named WP-7 requirements and scenarios;
- a narrow owner and product boundary;
- focused production-path tests;
- deterministic behavior under its declared execution policy;
- representative time and memory evidence when it changes a hot path; and
- no unexpected error-level messages in successful affected-mode validation.

An intentional algorithm, default, schema, or scientific-product change must
enter the intended-science-change ledger and a successor validation epoch. A
performance improvement is not accepted if it weakens scientific meaning,
failure behavior, reproducibility, or required publication.

## Relationship to ADR 0005

ADR 0005 deferred simultaneous measured `x/r` execution until an approved
scientific contract existed. The closed WP-7.1 packet now supplies that
authority for the exact bounded route named above. This ADR therefore
supersedes ADR 0005's execution deferral **only for the approved WP-7 route**.

ADR 0005 remains controlling for any stronger or different use of measured
`r`, including independent `r` science products, `r`-derived cleaning of `x`,
unapproved optical calibration of `r`, null mapping, polarimetry, or another
consumer not authorized by the WP-7 packet. No such capability is inferred by
this decision.

## Consequences

- The successor can use clean scientific boundaries without reopening the
  entire application architecture.
- The project temporarily carries two timestream implementations, but only
  one route is authoritative for a given explicit request and validation
  epoch.
- Some mature numerical bodies may be retained while their legacy
  orchestration and ambient state are not.
- Correctness and contract closure precede speculative hardware portability.
- Performance is designed and measured from the first slice but optimization
  complexity is purchased only with evidence.
- Release composition remains independent; numerical successor work and build
  or release integration are not combined in one change.

## Rejected alternatives

- **Continue as an ordinary in-place refactor:** understates the new product
  model and would entangle contract and legacy state.
- **Rewrite all of Citlali:** discards validated application and operational
  infrastructure unrelated to the WP-7 gaps.
- **Patch only the eleven implementation findings:** local patches do not by
  themselves create one coherent paired product and stage-ownership model.
- **Adopt an accelerator framework before the CPU reference route:** solves an
  unselected hardware problem before implementation conformity exists.
- **Replace or retain a numerical library without workload evidence:** a
  modernization label or prior use alone does not justify the dependency
  choice.
- **Activate the successor incrementally through silent fallbacks:** permits
  one request to mix incompatible scientific authorities.

## Supersession

Review this decision only if the approved WP-7 scientific authority changes,
a measured operational constraint makes the bounded CPU route infeasible, or
a complete successor has passed its gates and the migration/legacy-retirement
policy needs a new durable decision. A materially different scientific route
or whole-application replacement requires a new ADR rather than editing this
one into a different decision.

## Evidence

- [`../../validation/wp7_timestream_successor_authority.json`](../../validation/wp7_timestream_successor_authority.json)
- [`../WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md`](../WP7_TIMESTREAM_SUCCESSOR_PROGRAM.md)
- [`../WP7_TIMESTREAM_SUCCESSOR_IMPLEMENTATION_BASELINE.md`](../WP7_TIMESTREAM_SUCCESSOR_IMPLEMENTATION_BASELINE.md)
- [`0005-defer-measured-r-channel-execution.md`](0005-defer-measured-r-channel-execution.md)
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md)
- [`../PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md`](../PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md)
