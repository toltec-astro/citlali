# SCI-NOI v0.1 — Scientific-Owner Decision Ledger

Status: repaired Stage A owner-review candidate

Scientific owner: Grant Wilson

Updated: `2026-08-29`

Implementation defaults, historical products, successful tests, or accepted
reductions cannot resolve an open item. The future author-facing decision
authority is the single sanitized
[`SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`](SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md)
artifact after exact-byte approval. This manager ledger is not an author input.

## Decided Package Architecture

| ID | State | Approved substance |
| --- | --- | --- |
| `SCI-NOI-ODQ-001` | decided | One package contains realization generation, empirical uncertainty inference, and derived signal standardization |
| `SCI-NOI-ODQ-002` | decided | GEN, UNC, and STD have hard typed boundaries; no automatic realization chain exists |
| `SCI-NOI-ODQ-003` | decided | Standardized signal is not uncertainty and needs exact signal/scale parent identities |
| `SCI-NOI-ODQ-004` | decided | Without separate authority, STD means only “standardized by the stated empirical scale” |
| `SCI-NOI-ODQ-005` | decided | Fixed-state and relearned GEN are different methods and cannot share one ensemble identity |
| `SCI-NOI-ODQ-006` | decided | Validity, PTC/MAP coefficients, assignments, member QC, variance/covariance, empirical inverses/weights, and standardized signal remain separately typed |
| `SCI-NOI-ODQ-007` | decided | An empirical NOI inverse/weight is not a MAP-facing PTC coefficient |
| `SCI-NOI-ODQ-008` | decided | MAP/JINC parents remain immutable; NOI results attach as versioned companions |
| `SCI-NOI-ODQ-009` | decided | Dense full covariance is not universally required; honest unavailable is permitted |
| `SCI-NOI-ODQ-010` | decided | Persistence is plan-controlled; exact regeneration or sufficient statistics may replace stored members with explicit limitations |

## Recovered Constraints

| ID | State | Recovered substance and limit |
| --- | --- | --- |
| `SCI-NOI-HIST-001` | recovered approved policy | Compact deterministic assignment identity may replace dense signs when exactly reconstructible; no RNG/default selected |
| `SCI-NOI-HIST-002` | recovered approved policy | Enabled GEN requires positive effective/completed membership and disabled GEN is explicit zero-member/no-work; no adequate count selected |
| `SCI-NOI-HIST-003` | recovered approved policy | Historical source-imprinted products are conditional diagnostics, not physical-noise authority |
| `SCI-NOI-HIST-004` | recovered approved policy | Distinct S/N-like values require distinct identities; invalid denominators are unavailable, not zero |
| `SCI-NOI-HIST-005` | recovered approved policy | Package-level semantic provenance is admissible only with stable exact joins and restrictions |

## Owner Walkthrough Required

| ID | Topic | Recommended disposition location | State |
| --- | --- | --- | --- |
| `SCI-NOI-ODQ-101` | Ordinary fixed-state/relearned conditioning family | Decision artifact ODQ-101 | **decided — fixed-state conditional-sign ordinary; relearned separate; never mixed** |
| `SCI-NOI-ODQ-102A` | Exact ordinary parent/insertion route | Decision artifact ODQ-102A | decided — NOI modifier at PTC-to-MAP boundary; inline MAP application permitted; output is NOI realization map; numerical gates remain unavailable |
| `SCI-NOI-ODQ-102B` | Initial coherence-unit family | Decision artifact ODQ-102B | decided — one observation-scoped detector/channel assignment applies to every admitted sample throughout that observation |
| `SCI-NOI-ODQ-102C` | Ordinary sign and balance family | Decision artifact ODQ-102C | decided — network-stratified coefficient-balanced randomized signs; complement-symmetric marginal `1/2`; no detector-count balance or cross-network balancing |
| `SCI-NOI-ODQ-102D` | Exact balanced finite-design mechanics | Decision artifact ODQ-102D | decided — exact selection/rationale delegated to the implementation-blind scientific-contract author; tolerance-conditioned construction is nonbinding guidance; no advance acceptance or numerical availability |
| `SCI-NOI-ODQ-103` | Source-imprint/cancellation target and claim | Decision artifact ODQ-103 | **decided — randomization intends source suppression but does not by construction establish source-free maps; exact Stage-B terminology delegated** |
| `SCI-NOI-ODQ-104` | Consequential fixed/relearned state classification per method | Decision artifact ODQ-104 | **decided by explicit owner approval — every consequential adjacent state classified; relearned stages and resulting changed state named; no exhaustive implementation-provenance requirement** |
| `SCI-NOI-ODQ-105A` | Enabled/disabled and partial-completion behavior | Decision artifact ODQ-105A | **decided — rejected design candidates are not members/failures; any admitted-member failure fails the ensemble closed for every UNC use; no survivor ensemble** |
| `SCI-NOI-ODQ-105B` | Initial UNC target, center, estimator, correction, domain, and effective information | Decision artifact ODQ-105B | **decided — zero-centered design-weighted conditional randomization second moment on common all-member domain; no empirical recentering, `B-1`, or physical-noise interpretation** |
| `SCI-NOI-ODQ-106` | Covariance representations, domain, rank, null space, and unavailable policy | Decision artifact ODQ-106 | **decided — initial pointwise conditional second moment is not covariance; additional retained/projected/structured/full covariance methods are optional and separately identified; unknown covariance is not zero; exact rank/domain/null/regularization disclosure required; no inverse implication** |
| `SCI-NOI-ODQ-107` | Marginal inverse variance, precision, and consumer-effective weights | Decision artifact ODQ-107 | **decided — authorize finite-positive reciprocal `inverse_conditional_second_moment_scale`; not inverse variance/precision; unavailable rather than zero outside domain; regularization separate; marginal inverse variance, precision, and consumer weights remain separately typed; no PTC/MAP promotion** |
| `SCI-NOI-ODQ-108` | STD numerator, scale transformation, compatibility, identity, and claim | Decision artifact ODQ-108 | **decided — initial exact immutable normalized MAP numerator divided by canonical `sqrt(V_hat_cond)` on compatible finite-positive intersection; unit `1`; conditional-scale-standardized claim only; dependence disclosed; no significance; JINC separate/unavailable** |
| `SCI-NOI-ODQ-109` | Persistence/reconstruction modes and audit limitations | Decision artifact ODQ-109 | **decided — plan-selected persisted, compact-regeneration, or streaming-sufficient-statistic modes; no universal default or silent fallback; exact reproducibility/sufficiency/limitations recorded; required persistence fails closed; ODQ-105A preserved** |
| `SCI-NOI-ODQ-110A` | External transformation ownership and exact realization parity | Decision artifact ODQ-110A | **decided — the appropriate scientific process, not NOI, chooses and defines the transformation; NOI binds and applies exactly that transformation to every admitted compatible randomization for uncertainty of the exact transformed product; transformed routes remain unavailable until content-bound** |
| `SCI-NOI-ODQ-110B` | Wiener fixed-state, feedback, and per-realization scope | Decision artifact ODQ-110B | **decided — exact owner-defined Wiener fixed before randomization follows ODQ-110A; NOI-derived learning/update begins a new owner-defined successor product/GEN/UNC generation; per-member learning is separate ODQ-104 method; prior uncertainty is immutable and not independent validation; all numerical routes remain gated** |
| `SCI-NOI-ODQ-110C` | FRUIT scope | Decision artifact ODQ-110C | open |
| `SCI-NOI-ODQ-111` | VAL profile identities and exact consumer actions | Decision artifact ODQ-111 | open |

## Exact Stage A Gate

| ID | State | Decision required | Blocked action |
| --- | --- | --- | --- |
| `SCI-NOI-STAGE-A-Q001` | open | Approve the exact repaired Scope and packet; approve each granular decision or leave it explicitly open with dependent methods unavailable; approve profile bytes; complete exact source/Registry bindings and manifest hashes | Launch of a fresh implementation-blind Stage B author |

The parent boundaries also retain independent numerical unavailability: MAP and
pre-MAP routes lack the exact PTC MAP-facing coefficient and owner-admitted
numerical `coverage_cut`; JINC lacks required coefficient, TolTEC parameter,
and applicable numerical-adequacy authority. These are not repaired by Stage A
approval.

No implementation conformity, empirical calibration, physical-noise validity,
significance, achieved performance, readiness, or production question is
resolved here.
