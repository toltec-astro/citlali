# SCI-FRUIT v0.1 — D19 Iteration-Boundary State Candidate r0.1

Status: **owner-review proposal only; no restart repair or algorithm change is
authorized**

Date: `2026-09-02`

Source checkpoint: `38e477a84`

## The Decision In Plain Language

An uninterrupted FRUIT run currently remembers which suspicious map pixels it
should inspect in detail on the next iteration. A restarted run forgets that
choice. It can therefore make a different detector-exclusion decision one
iteration later.

The recommended repair is to make the already-resolved **next pixels to
inspect** part of the iteration-boundary state. This is small and bounded. It
does not require saving every diagnostic record from every earlier iteration.

There is one important correction to the phrase "save candidate detector
IDs." The detector responsible for a suspicious pixel is not known at the
boundary. It is discovered by detailed contributor tracing on the next pass.
Saving or applying a detector ID early would change the current method. The
state that actually has to cross the boundary is the set of map pixels to be
traced next.

## What The Pointing Replay Established

The iteration-4 split-run replay for pointing 152389 showed:

- continuous and restarted iteration 5 products are bitwise equal;
- the continuous process carries iteration-4 map-pixel outliers in memory and
  resolves 11 pixels for targeted contributor tracing during iteration 5;
- the restarted process has no restored outlier history and resolves no such
  targets;
- the continuous process consequently learns a scan-local exclusion for UID
  1489 during iteration 5, while the restarted process does not; and
- that penalty becomes applicable in iteration 6, where the `a1100` products
  diverge.

This is evidence about the present implementation, not authority for a new
scientific method. The complete result and hashes are in the
[six-iteration development record](../../../../../../validation/fruit_loop_point_152389_injected_convergence_development_2026-09-02/README.md).

## The Total State At A Completed Iteration

For exact continuation after completed iteration \(k\), define the total
boundary state provisionally as

\[
S_k = \left(F_k, M_k, P_k, W_k, T_{k+1}, I_k\right).
\]

In ordinary language:

| Item | Meaning | Present status |
| --- | --- | --- |
| \(F_k\) | accepted feedback model represented by the selected complete iteration-\(k\) route product | carried as the required prior map product |
| \(M_k\) | effective learned sample masks | stored by checkpoint v2 |
| \(P_k\) | effective detector/network penalties | stored by checkpoint v2 |
| \(W_k\) | accumulated and finalized PTC weight-validation state | stored by checkpoint v2 |
| \(T_{k+1}\) | bounded, resolved map pixels that iteration \(k+1\) must trace for contributors | **missing from checkpoint v2** |
| \(I_k\) | method, policy, ordered observations, route, grid/WCS/support, code, generation, and completed/next-iteration identities needed to interpret the state | partly enforced now; broader identity debt remains tracked separately |

This proposal closes only the newly demonstrated \(T_{k+1}\) omission. It
does not claim to close the broader run/configuration identity work in D10.
Any future stopping or adaptation history that affects later output would also
belong in \(S_k\); no such stopping rule is approved now.

## Recommended Learn–Resolve–Apply Contract

The proposed boundary order is:

```text
iteration k
  learn map-pixel outlier evidence
  resolve the exact bounded target set T[k+1]
  finish F[k], masks, penalties, and weight state
  write one causally complete checkpoint S[k]

iteration k+1
  restore S[k], if this is a restart
  apply exactly T[k+1] during contributor tracing
  learn any resulting detector penalty P[k+1]
  learn current outlier evidence
  resolve replacement target set T[k+2]
```

For the present code, each target is a pixel coordinate
`(map_index, row, col)` scoped by observation and producer/stage. The only
current target consumer is `mapdiag:raw_obs`. The resolved state also needs:

- the source iteration from which the target evidence was selected;
- the absolute iteration for which the targets are intended;
- observation, stage, route, and map-grid identity;
- the target-selection policy/version and configured maximum; and
- an explicit state distinguishing `not applicable`, `resolved empty`, and
  `required but unavailable`.

`resolved empty` means the method honestly chose no targets. `required but
unavailable` makes exact restart fail closed; it must not silently mean no
targets.

The target set is replaced at every completed boundary. Earlier outlier event
history may be retained as QA evidence, compacted, or expired under its own
retention policy. It is not required for restart once the exact next target set
has been resolved and content-bound.

## Exact-Trajectory Invariant

For fixed input data, executable, effective configuration, route, grid, and
complete boundary state \(S_k\), the required scientific invariant is:

> every required product, learned operational state, and later decision from
> iteration \(k+1\) onward is independent of whether iterations through \(k\)
> occurred in the same process or were restored from the checkpoint.

The primary acceptance standard remains bitwise equality. If any product
cannot be bitwise deterministic for an independently established reason, an
explicit numerical tolerance and proof that it cannot alter a discrete later
decision would require separate approval. No such exception is proposed here.

## Current Causal-State Audit

At source checkpoint `38e477a84`, a repository-local consumer audit gives the
following classification:

| State | Later numerical consumer | Boundary treatment |
| --- | --- | --- |
| effective sample masks | yes | checkpointed operational state |
| effective detector penalties | yes | checkpointed operational state |
| PTC weight-validation accumulators/final result | yes | checkpointed operational state |
| `map_pixel_outliers` | **yes**: selects next targeted contributor pixels | resolve into \(T_{k+1}\); do not depend on diagnostic history after the boundary |
| learned-mask and detector-penalty event vectors | no separate consumer; effects are in effective state | diagnostic history only |
| high-weight-detector, source-protection, busy-network, and learned-mask-application summaries | no later numerical consumer found | diagnostic history only |

This audit is bounded to the named source checkpoint. Any later code that makes
a diagnostic record causal must update the boundary contract before an exact
restart claim.

## Owner Choices

### A. Store the resolved next-target state — recommended

Resolve \(T_{k+1}\) once, make both uninterrupted and restarted execution use
that same resolved state, and serialize it in a successor checkpoint schema.
Demonstrate that resolving at the explicit boundary preserves the current
uninterrupted selection exactly.

This most directly separates operational state from diagnostic history and
keeps storage bounded by the configured target maximum.

### B. Store a compact causal outlier frontier

Store only the latest per-observation/per-stage outlier records needed to
reconstruct \(T_{k+1}\), then run the existing resolver after restart. This
can preserve behavior but remains coupled to scoring, ordering, tie handling,
coordinate validation, and diagnostic-record representation. It is acceptable
only if reconstruction of the exact same \(T_{k+1}\) is proved and tested.

This is bounded across iterations, but generally larger and less explicit
than choice A.

### C. Declare exact restart unavailable for this feature combination

When targeted contributor tracing can influence detector exclusion, reject an
exact-restart request rather than continue with incomplete state. Continuous
development runs may still be used, with the limitation recorded. A run that
disables the feature is a different effective method/policy and is not an
exact continuation of a run that enabled it.

This is scientifically honest but retains D19 and prevents restart-dependent
qualification for this path.

Persisting all accumulated diagnostic history is not proposed. Applying a
detector penalty before contributor tracing discovers it is also not proposed,
because that would change the method.

## Validation Required After A Repair Is Approved

Approval of A or B would authorize a separate bounded engineering repair. That
repair would have to pass all of the following before D19 can close:

1. a focused unit test exercises real target resolution, checkpoint
   serialization, restoration, and application rather than directly inserting
   only the already-effective penalties;
2. uninterrupted and split executions produce the same resolved target state,
   effective masks, effective penalties, and PTC weight state at every compared
   boundary;
3. a real pointing control compares at least three post-checkpoint iterations,
   so the target decision, its delayed penalty effect, and the following
   successor state are all observed;
4. all required signal, kernel, and weight products match bitwise for every
   array at each compared iteration;
5. malformed, incompatible, and missing required target state fail before
   iteration execution; and
6. the checkpoint version changes and older checkpoints are rejected or
   explicitly classified as incompatible for this enabled path.

The existing point-152389 development pair is sufficient for the first real
repair check. It is not a qualification population.

## What May Continue Now

The restart finding does **not** stop uninterrupted empirical development on
the already authorized development-only data. It blocks only:

- qualification evidence assembled from a trajectory that crosses this
  incomplete restart boundary;
- an exact-restart claim for the affected feature combination; and
- a restart-dependent stopping or reproducibility claim.

It does not authorize additional data access, a new recurrence, a learning or
detector-selection change, a stopping rule, Gate-D, qualification, Unity work,
or production use.

## Evidence And Existing Authority

- [Stage A restart taxonomy](../RESTART_CHECKPOINT_AND_LIFECYCLE_TAXONOMY.md)
  defines a checkpoint as complete causal state rather than diagnostic history.
- [Stage A iterative DAG](../ITERATIVE_DAG_AND_STATE_OWNERSHIP.md) applies the
  causal-completeness test to every later scientific output or decision.
- [`FRUIT-GAP-011`](../CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md) already
  required diagnostic history to be reclassified if later logic consumes it.
- [ADR 0006](../../../../../adr/0006-fruit-loop-restart-checkpoint.md) records
  the accepted exact-restart target and the measured checkpoint-v2 failure.
- [`ReductionLearningState`](../../../../../../include/citlali/core/engine/learning.h)
  currently labels all event vectors diagnostic.
- [Target resolution](../../../../../../include/citlali/core/engine/detail/learning_targets_impl.h)
  consumes prior `map_pixel_outliers` and selects bounded pixel targets.
- [Detector-penalty learning](../../../../../../include/citlali/core/pipeline/mapdiag_workspace_learning_emit.h)
  uses contributor tracing to create operational exclusion state.
- [Checkpoint v2](../../../../../../src/citlali/core/pipeline/reduction_restart_checkpoint.cpp)
  stores effective masks, effective penalties, and PTC weight-validation state
  but not the target-selection dependency.

These implementation and validation records are evidence about recovered
behavior. They are not independent scientific authority and do not approve
the recommendation.

## Requested Owner Decision

Please choose A, B, or C. The recommendation is **A**: checkpoint the bounded,
resolved next-iteration target state and validate exact current behavior across
three post-checkpoint iterations. No code should change until that state
semantics choice is approved.
