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
| `SCI-NOI-ODQ-101` | Ordinary fixed-state/relearned availability | Decision artifact row 101 | **open — first walkthrough question** |
| `SCI-NOI-ODQ-102` | Initial coherence/randomization methods and assignment law | Decision artifact row 102 | open |
| `SCI-NOI-ODQ-103` | Source-imprint/cancellation target and claim | Decision artifact row 103 | open |
| `SCI-NOI-ODQ-104` | Exact fixed state and any rerun graph | Decision artifact row 104 | open |
| `SCI-NOI-ODQ-105` | Initial UNC target, center, estimator, correction, missingness, and effective information | Decision artifact row 105 | open |
| `SCI-NOI-ODQ-106` | Covariance representations, domain, rank, null space, and unavailable policy | Decision artifact row 106 | open |
| `SCI-NOI-ODQ-107` | Marginal inverse variance, precision, and consumer-effective weights | Decision artifact row 107 | open |
| `SCI-NOI-ODQ-108` | STD numerator, scale transformation, compatibility, identity, and claim | Decision artifact row 108 | open |
| `SCI-NOI-ODQ-109` | Persistence/reconstruction and incomplete-design behavior | Decision artifact row 109 | open |
| `SCI-NOI-ODQ-110` | Deterministic FLT, Wiener, and FRUIT scope | Decision artifact row 110 | open |

## Exact Stage A Gate

| ID | State | Decision required | Blocked action |
| --- | --- | --- | --- |
| `SCI-NOI-STAGE-A-Q001` | open | Approve every ODQ-101--110 disposition and exact repaired Scope, cover, conventions, taxonomy, three parent boundaries, four profile drafts, FLT/FRUIT record, change log, firewall, and final manifest hashes | Launch of a fresh implementation-blind Stage B author |

The parent boundaries also retain independent numerical unavailability: MAP and
pre-MAP routes lack the exact PTC MAP-facing coefficient and owner-admitted
numerical `coverage_cut`; JINC lacks required coefficient, TolTEC parameter,
and applicable numerical-adequacy authority. These are not repaired by Stage A
approval.

No implementation conformity, empirical calibration, physical-noise validity,
significance, achieved performance, readiness, or production question is
resolved here.
