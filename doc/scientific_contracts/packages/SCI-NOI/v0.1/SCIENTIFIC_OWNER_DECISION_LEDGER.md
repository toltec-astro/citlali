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
| `SCI-NOI-ODQ-102D` | Exact balanced finite-design mechanics | Decision artifact ODQ-102D | **open — next walkthrough question** |
| `SCI-NOI-ODQ-103` | Source-imprint/cancellation target and claim | Decision artifact row 103 | open |
| `SCI-NOI-ODQ-104` | Complete fixed/relearned state classification per route | Decision artifact ODQ-104 | decided as route-specific consequence of ODQ-101; exact route remains open |
| `SCI-NOI-ODQ-105A` | Enabled/disabled and partial-completion behavior | Decision artifact ODQ-105A | open |
| `SCI-NOI-ODQ-105B` | Initial UNC target, center, estimator, correction, missingness, and effective information | Decision artifact ODQ-105B | open |
| `SCI-NOI-ODQ-106` | Covariance representations, domain, rank, null space, and unavailable policy | Decision artifact row 106 | open |
| `SCI-NOI-ODQ-107` | Marginal inverse variance, precision, and consumer-effective weights | Decision artifact row 107 | open |
| `SCI-NOI-ODQ-108` | STD numerator, scale transformation, compatibility, identity, and claim | Decision artifact row 108 | open |
| `SCI-NOI-ODQ-109` | Persistence/reconstruction and incomplete-design behavior | Decision artifact row 109 | open |
| `SCI-NOI-ODQ-110A` | Held-fixed deterministic FLT scope | Decision artifact ODQ-110A | open |
| `SCI-NOI-ODQ-110B` | Wiener scope | Decision artifact ODQ-110B | open |
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
