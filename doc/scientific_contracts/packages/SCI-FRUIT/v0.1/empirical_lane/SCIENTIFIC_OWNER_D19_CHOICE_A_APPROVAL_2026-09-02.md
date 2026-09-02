# SCI-FRUIT v0.1 — D19 Choice-A Owner Approval

Scientific owner: Grant Wilson

Decision date: `2026-09-02`

Decision ID: `SCI-FRUIT-EL-D19-ITERATION-BOUNDARY-STATE-A-2026-09-02`

Status: **Choice A approved; bounded checkpoint repair validated; D19 closed**

## Exact Approval

After reporting that the proposal commit had been pushed, the scientific owner
stated:

> pushed, I agree with choice A

Remote state was not independently queried and is not scientific authority.

## Approved Object

Choice A is the option titled **Store the resolved next-target state** in the
following exact candidate:

| Object | Source commit | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `EL_D19_ITERATION_BOUNDARY_STATE_CANDIDATE_R0.1.md` | `5da2524e6` | 11994 | `d27b917bfa7d2249a45fd5351aad7710b44b27d7c00ea6fad613f28798f886ef` |

The candidate remains the content-bound decision object. This approval record
does not rewrite its bytes.

## Authorized Effect

This decision authorizes the separate bounded repair defined by Choice A:

- resolve the exact, policy-bounded map-pixel target set for the next
  iteration at the completed-iteration boundary;
- make uninterrupted and restarted execution consume that same resolved
  state;
- serialize the resolved state in a successor checkpoint schema without
  retaining unbounded map-pixel-outlier history;
- distinguish a valid empty resolution from missing required state and fail
  closed on the latter; and
- perform the unit, malformed-state, and real three-post-checkpoint-iteration
  validation required by the approved candidate.

The detector identity remains an output of contributor tracing on the next
iteration. It is not moved earlier or stored as a substitute for the approved
pixel-target state.

## Preserved Non-Effects

This approval does not change or select a FRUIT recurrence, learning policy,
target-ranking rule, detector-selection rule, stopping rule, scientific
profile, qualification population, or production policy. It does not close
D19 by itself. Exact-restart, restart-dependent qualification, and
restart-dependent stopping claims remain unavailable for the affected feature
combination until the complete validation specified by the approved candidate
passes.

## Validation Disposition

Repair commit `2b59ad642` implements the authorized checkpoint-v3 state.
Evidence commit `dd342448b` records the required point-152389 replay. Restarted
absolute iterations 5, 6, and 7 are bitwise equal to the uninterrupted
trajectory in all 27 required map planes and in every checkpoint variable.
The initial restart restores the 12 resolved targets from iteration 4, and the
subsequent resolved targets and detector penalties remain identical. D19 is
therefore closed for this enabled path. The approval's preserved non-effects
remain in force.

See
[`../../../../../../validation/fruit_loop_point_152389_restart_v3_choice_a_2026-09-02/README.md`](../../../../../../validation/fruit_loop_point_152389_restart_v3_choice_a_2026-09-02/README.md).
