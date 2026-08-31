# SCI-FRUIT v0.1 — Restart, Checkpoint, And Lifecycle Taxonomy

Status: Stage A owner-review candidate

## Distinct Entry Modes

| Entry mode | Scientific identity | State admitted | Required consequence |
| --- | --- | --- | --- |
| New unseeded generation | New FRUIT generation at absolute iteration zero | Exact immutable parents and effective plan; no prior feedback model unless the contract defines a null/zero initialization | Cannot claim continuity with an earlier reduction |
| Map-only seed | New FRUIT generation/branch initialized from an explicitly admitted seed product | Seed signal plus its exact identity, response, support, validity, and route; no inherited operational learning state | Must not append iteration numbers, diagnostics, or convergence history to the seed's prior sequence as exact continuation |
| Exact checkpoint restart | Continuation of the same FRUIT generation/branch | Complete causal operational state, exact prior products, policy and ordered-parent identity | Completed `N` resumes at `N+1`; incompatible or incomplete state fails closed |
| Changed-policy continuation | Successor generation/branch, if scientifically authorized | Exact predecessor plus an explicit transition record | Must not be represented as exact restart or mutate predecessor products |
| NOI-informed continuation | New immutable FRUIT/science/GEN/UNC successor generation | Exact prior UNC as dependent input plus new learned FRUIT state | Prior uncertainty is not independent validation and is not overwritten |
| Per-realization replay | Separate NOI ODQ-104-style method | Exact realization-specific member graph and FRUIT state | Cannot pool with fixed-state members absent a separately authorized mixture estimand |

## Lifecycle Vocabulary

| Term | Definition for owner review | Not equivalent to |
| --- | --- | --- |
| `requested` | User/scientific intent before applicability and compatibility resolution | effective or realized |
| `effective` | Validated plan after defaults are forbidden or explicitly resolved and all selected policies are named | observation-resolved or successful |
| `observation_resolved` | Exact applicable products, grouping, arrays/networks, WCS/grid, and per-observation state | process-global request |
| `learning_input` | Data/products allowed to determine a later operator, selection, mask, penalty, or model | apply state |
| `learned_state` | Output of an exact learning method with parent and generation identity | authority for the method or empirical truth |
| `apply_state` | Exact frozen state applied to produce one product | a mutable accumulator or per-member relearning |
| `realized` | What operation/product actually occurred, with required outputs and failures resolved | requested or merely applicable |
| `iteration` | One absolute execution of the selected FRUIT recurrence within a generation/branch | generation, pass inferred from filename, or convergence |
| `generation` | Immutable lineage sharing declared parent/policy/state-transition semantics | iteration |
| `branch` | Successor lineage created by map seed, policy/state transition, or owner-authorized alternative | exact continuation |
| `terminal` | Exact iteration/product selected by an approved stopping/selection rule | last file written or maximum reached by coincidence |
| `checkpoint` | Complete causal state sufficient to continue the same generation exactly | map seed, QA archive, or diagnostic history |
| `diagnostic_history` | Causally inert record used for review/validation | operational state, unless the stop/update rule consumes it |

## Exact-Restart Compatibility Classes

The future contract must type every change as one of:

- **equal-required**: exact equality is necessary for continuation;
- **compatible-by-proof**: a declared equivalence/compatibility rule proves
  no future scientific output or identity can change;
- **successor-only**: change is permitted only by a new generation/branch;
- **forbidden/unavailable**: no authorized transition exists.

Candidate equal-required fields include method/version, recurrence family,
parent and ordered observation identity, route/grouping, model and selection
state, forward projector, PTC/RTC/map policies and their causal learned state,
support/validity policy, stop/terminal policy, and completed/next absolute
iteration. This list is not final until the owner closes the scientific DAG.

## Checkpoint Product Classes

| Class | Examples | Exact-restart treatment |
| --- | --- | --- |
| Scientific causal state | feedback model, selection/support state, response-affecting learned state, accumulated validation/weight state, stop-rule history when consumed | Required and content-bound |
| Exact identity/provenance | parents, ordered observations, policy versions, generation/branch/iteration, units/WCS/calibration | Required and compatibility checked |
| Required external product | prior iteration map/model, upstream coefficient/response products | Required by exact identity or embedded content |
| Diagnostic-only record | plots, report tables, event history not consumed by later logic | May be excluded only with an explicit causal-inertness claim |
| QA/validation evidence | comparison metrics, acceptance reports | Never substitutes for operational state |

## Failure Semantics To Author

Missing required checkpoint state, incompatible policy/parent identity, a
nonexistent next iteration under an absolute limit, unreadable required
products, or ambiguous generation/branch identity must fail the requested exact
restart. Implementations must not silently downgrade it to a map-only seed,
restart from zero, substitute another route, or discard learned state.
