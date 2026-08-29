# ADR 0016: Stage-boundary native runtime state

Status: accepted owner-directed correctness and resource repair; local
implementation complete, Unity science acceptance pending

## Context

ADR 0013 correctly rejected an exhaustive detector-by-sample history as a
canonical product, but it retained the same narrative in memory. The Stage 7
implementation consequently held a structured binding for each detector
sample, a second prepared cell representation, a per-cell update batch, a
processed projection-cell representation, and a node-based revision/value
ledger with per-sample history. These objects duplicated numerical matrices
and compact masks that already owned the scientific state.

Unity NGC4449 job `63757963` processed observations 152390 and 152392 with 16
workers and reached 100.93 GiB MaxRSS before stopping at the separately
identified Wiener-template policy gate. The map and noise-realization cubes
reported only about 0.04 GB each. The project owner judged the retained
per-cell operation narrative contrary to the governing reproducibility
principle: retain enough authority to reproduce a process, not a description
of every deterministic operation.

## Decision

Native runtime state is retained at its natural computational boundary:

- established dense numerical matrices remain the value owners;
- dense or compact masks retain exact sample validity and exclusion state;
- row/network anchors retain native identity and support without repeating
  strings, slot vectors, or identities for every detector cell;
- detector-scope causes remain detector-scope;
- one bounded scan operation gate issues and commits a stage identity only
  after the complete numerical result, shapes, partition, flags, and finite
  policy validate; and
- canonical provenance retains authoritative inputs, configuration, build,
  policy, bounded cause summaries, product identities, and validation state
  as defined by ADR 0013.

The runtime must not retain a per-detector-sample operation ledger, revision
vector, node-based value map, update object, invalidity string, or projection
record merely to narrate a deterministic transformation. A bounded opt-in
debug trace may reconstruct selected transitions from the authoritative
matrices and natural-scope metadata; it does not justify an exhaustive backing
ledger.

This decision does not alter RTC, PTC, mapmaking, JINC, Wiener, weighting, or
fruit-loop numerical algorithms. It changes ownership and retention only.

## Required invariants

1. Gather and scatter preserve exact native identity, detector partitions,
   input values for excluded cells, append-only flags, and numerical results.
2. A malformed, stale, foreign, duplicate, incomplete, or nonfinite candidate
   fails before any live stage commit.
3. Operation state is bounded by scan/stage count and is independent of the
   detector-by-sample cardinality.
4. Row metadata scales with output rows and participating networks, not with
   output rows times detectors.
5. Canonical provenance and opt-in trace contracts remain bounded and retain
   their existing validation rules.
6. An exact repaired executable must complete the existing NGC4449 successor
   science gate on Unity with Slurm MaxRSS recorded before this resource repair
   receives operational acceptance.

## Compatibility and migration

The historical `NativeSampleLedger`, rectangular coincidence cohort, and
their test-only scatter path are removed rather than retained as an alternate
API. Their scientific invariants are exercised through the production dense
PTC adapter and projection tests. Historical v2 provenance remains readable
and can reconstruct its explicit 0-to-1 transition records on demand; it is
not a runtime-state authority.

ADR 0013 remains authoritative for persisted provenance, but its statement
that runtime revision ledgers remain in memory is superseded by this decision.
No accepted-run or intended-science-change entry is created by the structural
repair alone.

## Consequences

- Peak memory no longer includes multiple structured objects or allocator
  nodes per detector sample.
- Reproducibility continues to derive from raw inputs, compact-v2 APT,
  effective configuration, software/build identity, execution policy,
  bounded provenance, and verified products.
- Unity validation is now an acceptance measurement of the repaired design,
  not a prerequisite experiment to establish that the superseded design was
  inappropriate.
