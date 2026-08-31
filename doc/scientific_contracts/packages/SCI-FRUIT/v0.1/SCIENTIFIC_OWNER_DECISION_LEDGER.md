# SCI-FRUIT v0.1 — Scientific-Owner Decision Ledger

Status: Stage A owner-review queue; all scientific questions open

The questions are ordered by consequence. Later questions must not be answered
in a way that silently fixes an earlier one.

| ID | Owner question | Candidate dispositions to review | Consequence/stop gate | Status |
| --- | --- | --- | --- | --- |
| `SCI-FRUIT-ODQ-001` | **What exact scientific object is iterated, and what is the recurrence/update law?** | (A) cumulative feedback model plus separately persisted residual increments and explicit update `U`; (B) replacement by a new full-model estimate each iteration; (C) another owner-defined estimand/law. State whether sample-domain add-back is normative, derived, or excluded. | Determines DAG, response, state, restart, route admissibility, and what “iteration” means. No route decision should precede it. | **open — first review question** |
| `SCI-FRUIT-ODQ-002` | Which of ordinary MAP, JINC, FLT-FIXED, and future FLT-MATCHED may be candidate parents, for which observation/coadd groupings? | Admit individually, defer, or exclude; preserve numerical unavailable gates and no fallback. | Selects route-specific author boundaries; does not itself construct a feedback model. | open |
| `SCI-FRUIT-ODQ-003` | How does each admitted parent become the feedback model, including selection, synthesis, support, and calibration? | Owner-select exact model-construction family; delegate bounded alternatives to author only after target estimand is fixed; decide internal selection versus later source-package boundary. | Prevents “map equals model” and source-catalog leakage. | open |
| `SCI-FRUIT-ODQ-004` | What forward projector and subtraction/add-back order is scientifically intended? | Define exact sample-domain operator and its relation to parent response/grid/units; select residual-increment or full-map path if not closed by ODQ-001. | Required for response, null space, failure, and implementation conformance. | open |
| `SCI-FRUIT-ODQ-005` | Which policy/state is fixed, learned, applied, carried, reset, or relearned across observations and iterations? | Decide model selection, RTC/PTC, masks, detector penalties, weights, response state, and learning cadence; identify adjacent-owner state. | Defines generation graph and causal checkpoint set. | open |
| `SCI-FRUIT-ODQ-006` | What response and bias targets must be published? | Separate fixed-state response, complete-procedure response, attenuation/bias, null space, and route-specific validity; allow typed unavailability where justified. | Blocks calibration, correction, and downstream response claims. | open |
| `SCI-FRUIT-ODQ-007` | What support, selection, edge, missing/non-finite, validity, and failure rules apply? | Complete-support-only, partial-support with exact rule, or route-specific policies; define unavailable versus failed versus absent. | Required before numerical admission and stopping metrics. | open |
| `SCI-FRUIT-ODQ-008` | Which diagnostics, if any, define convergence, stopping, and terminal selection? | Define separate amplitude/morphology/centroid/map/support/noise metrics; persistence/hysteresis; source classes; measurement-limited handling; hard maximum semantics; disagreement rule. | No terminal product or downstream Pointing/OOF relation until closed. | open |
| `SCI-FRUIT-ODQ-009` | What causal state and compatibility policy make restart exact? | Approve causal completeness test; classify equal-required, compatible-by-proof, successor-only, and forbidden changes; decide stop-history causality. | Exact restart unavailable until closed; map-only seed always distinct. | open |
| `SCI-FRUIT-ODQ-010` | Which uncertainty/covariance targets and NOI lifecycle methods are in base v0.1? | Fixed-state conditional; complete-procedure/successor; partial or full per-member replay; typed unavailable. Do not pool without mixture estimand. | Determines GEN/UNC graph, response claims, cost, and checkpoint/replay state. | open |
| `SCI-FRUIT-ODQ-011` | Which per-iteration/terminal/state/diagnostic products are required and what downstream claims fail closed? | Select exact product bundle, response/covariance disclosures, VAL profile needs, required-output failure, and terminal consumer interface. | Needed for crosswalk and future engineering conformance. | open |
| `SCI-FRUIT-ODQ-012` | Is the sanitized exact author packet complete and may Stage B launch? | Approve exact scope, source manifest, owner records, supersession order, exclusions, model/reasoning instructions, and author stop rule; or return for repair. | Only explicit owner approval here can authorize implementation-blind authorship. | blocked on prior decisions |

## Recorded Non-Scientific Owner Decision

| ID | Decision | Effect | Non-effect |
| --- | --- | --- | --- |
| `SCI-FRUIT-SEQ-2026-08-31` | Launch recovery-first SCI-FRUIT after the single-pass MAP/JINC/NOI/filtering line and before source-fitting and Pointing/OOF | Changes roadmap sequencing and authorizes Stage A recovery on the dedicated branch | Does not approve this packet, any parent, estimator, recurrence, Stage B work, algorithm change, validation, or production action |

## Walkthrough Rule

Review one consequential question at a time. Record each answer in a separate
dated owner-decision artifact, update this ledger without rewriting the earlier
record, and propagate only the consequences actually authorized.
