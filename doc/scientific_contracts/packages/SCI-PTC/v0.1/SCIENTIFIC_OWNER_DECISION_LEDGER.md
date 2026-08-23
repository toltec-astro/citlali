# SCI-PTC v0.1 — Scientific Owner Decision Ledger

Status: frozen v0.1/r0.5 scientific-authority summary; implementation
conformity not yet assessed under this contract

| Decision group | Status | Required owner action | Blocked work |
| --- | --- | --- | --- |
| `PTC-SCOPE-D001--D006` | approved `2026-08-19` | No further Stage A action; calibrated route, signal convention, availability, ownership, route behavior, and exact packet accepted | None at Stage A |
| `PTC-SCOPE-D007--D013` | approved `2026-08-19` | No further Stage A action; support, diagnostic `r`, refit, estimator, grouping, selection, and response rules accepted | None at Stage A |
| `PTC-SCOPE-D014--D017` | approved `2026-08-19` | No further Stage A action; estimand/null space, coefficient taxonomy, map-center diagnostic, and centering/scaling accepted | None at Stage A |
| `PTC-OWNER-Q001` | resolved `2026-08-19` | First implementation/base v0.1 uses diagnostic-only `r`; it is inert or advisory and may not control calibrated-`x` membership, subtraction, output, or coefficients | Cross-channel `x <- r` subtraction and `r`-controlled `x` decisions are excluded; successor authority is required |
| `PTC-OWNER-Q002` | resolved `2026-08-20` | Base-v0.1 PCA/SVD projects every admitted application input or fixed-state companion through the frozen realized subspace and metric with linear coefficient recomputation; each realized family declares its exact acting space | Frozen numerical component subtraction is not the base PCA/SVD response rule; it is available only as a separately named affine family with identity derivative |
| `WP1-OWNER-D001--D009` | approved `2026-08-23` | The bounded successor repairs transformed-signal identity, total named-use truth, notation, disabled routing, grouping, positive rank, single immutable-parent fit, and fixed-state kernel authority | No unrelated PTC, MAP, implementation, validation, or production authority is implied |
| `PTC-OWNER-Q003` | resolved `2026-08-23` | Equation 8 uses the group-local detector-right mask-aware coefficient solve only when its finite time-local normal matrix has numerical rank `k_req,g` under frozen tolerance `tau_g` | A deficient data/kernel group-time is unavailable; no partial-rank, interpolation, zero-substitution, or cross-group fallback is authorized |
| `PTC-FREEZE-D002` | approved and frozen `2026-08-23` | The complete verified r0.5 candidate is promoted as the active scientific authority, superseding r0.4 while preserving r0.4 provenance | No audit finding closes before re-audit; implementation conformity, validation, performance, qualification, readiness, and MAP authority remain unestablished |

Approved historical `SCI-PTC-001-D001--D006` are recorded in
[`DECISION_LOG.md`](DECISION_LOG.md) and incorporated through the binding
supersession cover.

The single detailed Stage B ledger is the **Scientific Owner Decision Ledger**
section of [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md). It contains
15 entries: five decided, three open, four known-but-not-supplied, and three
deferred. This manager-facing file deliberately does not copy those rows;
keeping one detailed authority avoids decision drift.

No unresolved entry blocks review of the structural contract. Each blocks only
the specifically named automatic policy, numerical product, response,
coefficient use, covariance claim, or later evidence-layer claim.

The bounded r0.5 successor preserves the r0.4 cause-union, named-use
composition, ownership split, and nonrestoring-centering authority while
applying `WP1-OWNER-D001--D009`, the r0.5 centering resolution, and Q003. It
does not select any open numerical threshold, default estimator, MAP-facing
coefficient, full-procedure response domain, or stronger covariance model.

The active scientific-owner freeze is recorded in
[`SCIENTIFIC_OWNER_FREEZE_R0.5.md`](SCIENTIFIC_OWNER_FREEZE_R0.5.md). It
preserves all 15 detailed-ledger states: five decided, three open, four
known-but-not-supplied, and three deferred. The superseded r0.4 freeze remains
preserved by its own record and immutable Git history. Resolving any remaining
item requires explicit owner authority and a versioned successor or formally
reopened revision; the freeze itself does not supply that authority.
