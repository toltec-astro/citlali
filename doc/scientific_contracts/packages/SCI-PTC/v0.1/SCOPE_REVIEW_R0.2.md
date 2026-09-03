# SCI-PTC v0.1 — Stage A Scope Review R0.2

Status: manager disposition of the final bounded scientific review and owner
resolution supplied `2026-08-19`; exact packet approval pending

## Outcome

The review finds the revised scope scientifically mature and recommends only
bounded amendments before Stage B. Those amendments are incorporated without
changing the package boundary or launching scientific authorship.

The scientific owner resolved `PTC-OWNER-Q001`: in the first implementation/
base v0.1, `r` analysis is diagnostic-only. It is inert or advisory with
respect to calibrated `x` and may not provide subtraction modes or alter `x`
fit membership, subtraction, output retention, or coefficients. Any stronger
use requires a successor owner decision.

## Bounded Amendments

| Review item | Disposition | Revised authority |
| --- | --- | --- |
| Parent of a support-changing post-fit refit | adopted | Every complete refit begins from the same immutable admitted SCI-CAL parent; cleaned output is not the numerical parent; D009 |
| Explicit sequential residual estimator | adopted with boundary | Allowed only as a declared stage of one complete hierarchical estimator with exact order, cumulative removed subspace, response, covariance, and parentage; D009 |
| Concrete post-fit detector diagnostics | adopted | Residual, loading, influence, stability, source-response, and `x/r` diagnostic families now require population/model reference, normalization, support, uncertainty, and policy role; D009 |
| Detector pathology versus legitimate signal/state | adopted | Assessment must distinguish sky, source-model/mask failure, calibration, focal-plane position, and expected sensitivity variation; numeric thresholds remain owner-controlled |
| Mode-selection policy | adopted | D012 now requires the least aggressive candidate satisfying every required predicate; a failed predicate cannot be compensated by a scalar score |
| Candidate ordering and ties | adopted | Ordering and deterministic ties are explicit; nonnested candidates are compared through complete removed subspace and response |
| `r`-derived control of `x` | resolved unavailable | Diagnostic `r` is inert or advisory in base v0.1 and cannot alter `x`; Q001 resolved |
| Physical-origin label in grouping | corrected | “Array-wide optical/common component” is replaced by physical-origin-neutral “array-wide common component” |
| Scope of PTC full-procedure response | adopted | PTC reruns from its immutable CAL parent; whole-chain RTC-to-CAL-to-PTC injection remains separately owned cross-package work; D013 |

## Frozen-Core Conflict Preflight

The binding supersession cover was checked and strengthened to override any
older core clause inconsistent with:

- point-source-equivalent signal terminology;
- removed-subspace, additive-reference, gauge, and null-space state;
- per-cause stage-specific flag/support semantics;
- coefficient-family taxonomy;
- diagnostic-only conditioned `r`;
- immutable-parent post-fit refinement;
- within-array hierarchical grouping;
- conjunctive least-aggressive mode selection; and
- fixed-state versus PTC-full-procedure response companions.

The cover explicitly forbids using predecessor text to reintroduce
point-source-peak meaning, zero-filled missing data, universal flag actions,
cross-array or cross-channel authority, cleaned-output refit parentage,
compensating scalar rank scores, or whole-chain response ownership.

## Remaining Gate

`PTC-OWNER-Q001` is no longer a blocker. Stage B remains held only until the
scientific owner explicitly approves the amended `PTC-SCOPE-D001--D017` bytes
and the recomputed exact author packet. That approval authorizes
implementation-blind authorship, not implementation conformity, validation,
scientific freeze, or production use.
