# ADR 0006: Fruit-Loop Restart Checkpoint

- **Status:** Accepted
- **Recorded:** 2026-07-22
- **Decision owners:** Citlali project owner, scientific owner, and engineering

## Context

Long full-science reductions need to extend an earlier fruit-loop sequence
without repeating completed iterations. Loading only the last map is not an
exact continuation once reduction learning is active: the next iteration also
depends on the accumulated effective sample-mask interval union, effective
detector penalties, and the absolute iteration number that selects the
learning/apply phase. Diagnostic event vectors are bounded QA history and are
not operational state.

Treating a prior map path as a restart would therefore create plausible output
while silently discarding learned state. Resetting the iteration number would
also repeat phase-dependent policy under a different scientific identity.

## Decision

An enabled fruit-loop iteration publishes the required atomic NetCDF artifact
`citlali_restart_checkpoint.nc` in its completed `reduNN` directory. Version 1
contains:

- schema and creator version;
- completed and next zero-based absolute iteration numbers;
- fruit-loop map type and ordered observation identities;
- a canonical snapshot of the complete learning policy;
- the effective sample-mask state as canonical disjoint intervals; and
- the effective detector-penalty state by scientific identity.

The explicit YAML leaf `timestream.fruit_loops.restart_path` names the completed
`reduNN` directory. It is mutually exclusive with the ordinary initial-map
`path`. Loading fails before iteration execution if the checkpoint is missing
or malformed, if its map type, observation order, or learning policy differs,
or if its stored operational state violates the schema. `max_iters` remains an
absolute exclusive stop, so it must be greater than the checkpoint's
`next_iteration`.

The first resumed iteration loads its input map from `restart_path`, restores
the operational learning state, and uses the recorded absolute next iteration.
Later iterations use the normal preceding-map path in the new output root. A
new output layout is always prepared for that first resumed iteration,
including when `save_all_iters` is false.

The checkpoint deliberately does not carry bounded diagnostic event history or
its dropped-record counters. Those records explain a particular process run;
they do not determine subsequent flags or detector weights. A state-complete
restart is numerically exact only when the remaining scientific configuration
and inputs are unchanged. Version 1 enforces the restart-critical identities
listed above; operators remain responsible for keeping other science settings
unchanged, and the copied YAML/provenance records make that comparison
auditable.

## Consequences

- A map-only continuation remains available through `path`, but it is not
  represented or described as exact learning continuation.
- Checkpoint write failure is a required-output failure and prevents successful
  completion of that iteration.
- Restart provenance records the requested source and the resolved checkpoint,
  creator version, iteration pair, and restored state cardinalities.
- Absolute iteration numbering prevents a restarted job from relearning as
  iteration zero or changing phase solely because it crossed a job boundary.
- The operational checkpoint scales with compacted learned state, not with the
  bounded diagnostic event log.
- A real Unity split-run comparison remains required before this becomes an
  accepted NGC4449 science checkpoint.

## Rejected Alternatives

- **Raise or serialize the diagnostic record cap:** event history is not the
  effective model and would preserve the wrong scaling behavior.
- **Load only the final map:** loses operational learning state.
- **Reset iteration numbering in the new job:** changes phase-dependent
  learning policy and provenance identity.
- **Infer restart from an existing `path`:** makes map seeding and exact
  continuation indistinguishable to operators and auditors.
- **Keep checkpoints optional:** a missing state artifact could silently turn
  a requested exact continuation into a map-only run.

## Supersession

A successor schema is required to change state identity or compatibility
rules. It should add a canonical digest of all non-runtime effective science
configuration if Citlali gains one stable cross-domain serialization. The
long-term nested run/iteration layout tracked as retained debt D10 may change
artifact location, but must preserve the explicit checkpoint contract and
TolTECA-facing final-product compatibility.

## Evidence

- `include/citlali/core/pipeline/reduction_restart_checkpoint.h`
- `src/citlali/core/pipeline/reduction_restart_checkpoint.cpp`
- `tests/test_learning_and_fruit_contracts.cpp`
- [`../SCIENTIFIC_CONVENTIONS.md`](../SCIENTIFIC_CONVENTIONS.md)
