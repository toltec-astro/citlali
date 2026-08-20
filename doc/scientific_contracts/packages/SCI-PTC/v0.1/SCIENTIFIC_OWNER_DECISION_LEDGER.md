# SCI-PTC v0.1 — Scientific Owner Decision Ledger

Status: frozen v0.1/r0.4 scientific-authority summary; implementation
conformity not yet assessed under this contract

| Decision group | Status | Required owner action | Blocked work |
| --- | --- | --- | --- |
| `PTC-SCOPE-D001--D006` | approved `2026-08-19` | No further Stage A action; calibrated route, signal convention, availability, ownership, route behavior, and exact packet accepted | None at Stage A |
| `PTC-SCOPE-D007--D013` | approved `2026-08-19` | No further Stage A action; support, diagnostic `r`, refit, estimator, grouping, selection, and response rules accepted | None at Stage A |
| `PTC-SCOPE-D014--D017` | approved `2026-08-19` | No further Stage A action; estimand/null space, coefficient taxonomy, map-center diagnostic, and centering/scaling accepted | None at Stage A |
| `PTC-OWNER-Q001` | resolved `2026-08-19` | First implementation/base v0.1 uses diagnostic-only `r`; it is inert or advisory and may not control calibrated-`x` membership, subtraction, output, or coefficients | Cross-channel `x <- r` subtraction and `r`-controlled `x` decisions are excluded; successor authority is required |
| `PTC-OWNER-Q002` | resolved `2026-08-20` | Base-v0.1 PCA/SVD projects every admitted application input or fixed-state companion through the frozen realized subspace and metric with linear coefficient recomputation; each realized family declares its exact acting space | Frozen numerical component subtraction is not the base PCA/SVD response rule; it is available only as a separately named affine family with identity derivative |

Approved historical `SCI-PTC-001-D001--D006` are recorded in
[`DECISION_LOG.md`](DECISION_LOG.md) and incorporated through the binding
supersession cover.

The single detailed Stage B ledger is the **Scientific Owner Decision Ledger**
section of [`AUTHOR_DRAFT_DECISIONS.md`](AUTHOR_DRAFT_DECISIONS.md). It contains
14 entries: two decided, six open, four known-but-not-supplied, and two
deferred. This manager-facing file deliberately does not copy those rows;
keeping one detailed authority avoids decision drift.

No unresolved entry blocks review of the structural contract. Each blocks only
the specifically named automatic policy, numerical product, response,
coefficient use, covariance claim, or later evidence-layer claim.

The bounded r0.4 revision introduces no new unresolved owner question. It
records the owner-directed cause-union and named-use support-composition rule,
the producer/PTC/downstream/VAL ownership split, nonrestoring detector
centering, and complete rationale locators as author dispositions. These
clarifications preserve the already approved estimator and Q002 projection
decisions and do not select any open numerical threshold or default.

The scientific-owner freeze recorded in
[`SCIENTIFIC_OWNER_FREEZE_R0.4.md`](SCIENTIFIC_OWNER_FREEZE_R0.4.md) preserves
all 14 detailed-ledger states: two decided, six open, four
known-but-not-supplied, and two deferred. Resolving any remaining item requires
explicit owner authority and a versioned successor or formally reopened
revision; the freeze itself does not supply that authority.
