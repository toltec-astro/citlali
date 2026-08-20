# SCI-PTC v0.1 — Scientific Decision Log

Status: Scientific authority frozen at v0.1/r0.4; implementation conformity
not yet assessed under this contract

Scientific owner: Grant Wilson

Date: `2026-08-19`

## Approved Package-Scope Decisions

| Decision | Status | Disposition |
| --- | --- | --- |
| `PTC-SCOPE-D001` | approved `2026-08-19` | V0.1 ordered boundary is `SCI-RTC -> SCI-CAL -> SCI-PTC -> optional SCI-MAP`, including complete calibration uncertainty and response lineage. This supersedes the abbreviated historical RTC-to-PTC stage chain without changing route-specific disabled semantics. |
| `PTC-SCOPE-D002` | approved `2026-08-19` | The primary branch accepts and emits only the admitted SCI-CAL top-of-atmosphere, point-source-equivalent quantity in mJy per fixed nominal beam. PTC preserves unit/convention but not point-source peak, absolute level, extended-source response, or beam shape. Raw-`Delta f/f` Beammap PTC remains separate follow-up. |
| `PTC-SCOPE-D003` | approved `2026-08-19` | PTC cannot repair or strengthen unavailable/invalid RTC or CAL authority. The affected calibrated PTC science output remains unavailable. |
| `PTC-SCOPE-D004` | approved `2026-08-19` | PTC owns the correlated-mode fit/subtraction, source-protection supports, and additive/null-space state; correlation does not identify physical origin. RTC/separate authority owns temporal notches, and FRUIT owns recurrence/add-back. |
| `PTC-SCOPE-D005` | approved `2026-08-19` | PTC may run without MAP for requested transformed TOD. Disabled PTC realizes no PTC or MAP product on the PTC-dependent v0.1 route. Any direct CAL-to-MAP route requires separate authority. |
| `PTC-SCOPE-D006` | approved `2026-08-19` | The Stage B packet contains only the Scope Brief, frozen core plus inseparable cover, sanitized conventions/ownership, and bounded method-reference summary. Raw implementation, audit, repair, test, validation, and full-paper access remain excluded. |
| `PTC-SCOPE-D007` | approved `2026-08-19`; narrowed by owner `2026-08-20` | Causes accumulate without erasure and do not themselves prescribe action. For every PTC-owned use, PTC composes its base predicate and all applicable cause/fact predicates conjunctively; any exclusion controls, and an unknown required predicate with no known exclusion yields `decision_unavailable`. PTC preserves facts for downstream uses, whose owners define their own admission rules. Every composite declares inputs, Boolean rule, missing-state behavior, scope, use, owner, and policy/version. Zero-fill PCA and universal flag actions remain unauthorized. |
| `PTC-SCOPE-D008` | approved `2026-08-19` | PTC may consume a separately conditioned, compatible `r` diagnostic parent for `r`-only PCA. Under resolved `PTC-OWNER-Q001`, that analysis is inert or advisory and may not alter calibrated-`x` membership, subtraction, output, or coefficients. Raw-`r` processing, `r`-derived `x` subtraction, and unconstrained joint `x/r` PCA are excluded from base v0.1. |
| `PTC-SCOPE-D009` | approved `2026-08-19` | PTC supports a finite fit-diagnose-classify-refit process. A fit-support change refits one complete selected model from the same immutable admitted CAL parent and applies the final model once; a cleaned output is not the refit parent. Output-only or coefficient-only decisions do not alter the fit. Sequential residual fitting is allowed only as an explicit ordered stage of one complete hierarchical estimator with cumulative subspace, response, covariance, and parentage. Post-fit diagnostics have declared population/model reference, normalization, support, uncertainty, and policy role; thresholds remain owner-controlled. |
| `PTC-SCOPE-D010` | approved `2026-08-19` | Base estimator families are robust group common modes, explicit fixed-template regression, masked/weighted PCA/SVD, and `r`-only diagnostic PCA. Cross-channel `r` templates are gated; joint sky/noise and correlated-noise ML mapmaking are adjacent/successor authorities. |
| `PTC-SCOPE-D011` | approved `2026-08-19` | Base fitting is hierarchical within one array, with explicit array-wide, network/electronics, and optional local/focal-plane components. Data-derived groups are learned state; cross-array fitting requires separate authority. |
| `PTC-SCOPE-D012` | approved `2026-08-19` | Rank/component selection chooses the least aggressive member of a finite candidate set for which every required residual-contamination, astronomical-transfer, conditioning, support, stability, and QC predicate passes. Failed predicates cannot be compensated by a scalar score. Candidate ordering and deterministic ties are declared; nonnested candidates are compared by complete removed subspace and response. No universal rank or singular-value threshold is authority. |
| `PTC-SCOPE-D013` | approved `2026-08-19` | Requested response companions follow the exact fixed realized signal operator and support. Fixed-state conditional response and full data-dependent PTC response are distinct products. PTC full-procedure response begins from the immutable admitted CAL parent; whole-chain RTC-to-CAL-to-PTC injection is separate cross-package work. |
| `PTC-SCOPE-D014` | approved `2026-08-19` | PTC publishes its estimand, fitted correlated model, removed subspace, additive reference, null space, and permitted astronomical attenuation; retaining signal units is not response preservation. |
| `PTC-SCOPE-D015` | approved `2026-08-19` | Fitted loadings, centering/scaling parameters, diagnostic coefficients, and downstream analysis/gridding coefficients are distinct families. Only an explicitly named analysis/gridding family may be MAP-facing. |
| `PTC-SCOPE-D016` | approved `2026-08-19` | PTC owns sample-domain response. `estimated_map_center_point_source_response` is an optional functional of an exact source template, propagated response, and named reference MAP operator—not the general PTC response or MAP authority. |
| `PTC-SCOPE-D017` | approved `2026-08-19`; physically resolved by owner `2026-08-20` | Every centering/scaling transform declares axis, population, support, estimator, masks, units, gauge, failure behavior, and null space. Internal scaling is inverted before ordinary output. The learned additive detector location `lambda` is discarded rather than restored because detector `x` has no scientifically meaningful optical DC response. Frozen-state response holds `lambda` fixed; full-procedure response re-estimates and again discards it. |

## Resolved Scientific Owner Decision

| Decision | Status | Question and consequence |
| --- | --- | --- |
| `PTC-OWNER-Q001` | resolved `2026-08-19` | In the first implementation/base v0.1, `r` analysis is diagnostic-only and inert or advisory relative to calibrated `x`. It may not supply subtraction modes or alter `x` membership, subtraction, output, or coefficients. Stronger use requires a successor owner decision; unconstrained joint `x/r` PCA remains deferred. |

## Scientific-Owner Freeze

| Decision | Status | Decision and consequence |
| --- | --- | --- |
| `PTC-FREEZE-D001` | approved and frozen `2026-08-20` | The scientific owner states exactly, “Freeze SCI-PTC v0.1/r0.4.” This establishes the exact r0.4 package as scientific authority while retaining all recorded detailed-ledger states and leaving implementation conformity, validation, performance, qualification, and production readiness unassessed. |

## Preserved Approved Historical Decisions

| Decision | Status | Preserved scientific authority |
| --- | --- | --- |
| `SCI-PTC-001-D001` | approved `2026-08-08`, partly superseded for v0.1 order | Disabled PTC is a terminal upstream mode: no mean subtraction, cleaning, PTC coefficients/products, or MAP. PTC may run without MAP. The old `RTC -> optional PTC` shorthand is replaced by `PTC-SCOPE-D001` for this calibrated route. |
| `SCI-PTC-001-D002` | approved `2026-08-08` | `fit_invalid`, `postfit_output_reject`, and `weight_only` are distinct. Only fit-invalid causes require refit or fitted-product invalidation; exact representation is engineering-owned. |
| `SCI-PTC-001-D003` | approved `2026-08-09` | The stored PTC kernel is `estimated_map_center_point_source_response`, with typed status and exact band/mode/state/upstream binding. It does not imply universal off-center, extended-source, cross-band, or cross-mode response. |
| `SCI-PTC-001-D004` | approved `2026-08-09` | Existing detector-weight families remain scalar analysis/gridding coefficients with exact identity, unit, normalization, lifecycle, and factors. Stronger precision/significance/independence claims require proof; covariance may be unavailable. |
| `SCI-PTC-001-D005` | approved replacement `2026-08-09` | The in-memory PTC-transformed TOD is the authoritative PTC-to-MAP intermediate. Provenance burden follows declared consumption; persisted artifacts have explicit roles and honest completeness; exhaustive replay is not universal. |
| `SCI-PTC-001-D006` | approved `2026-08-09` | Fit arithmetic uses eligible finite inputs; surrogates shift signal and eligibility together; insufficient support is unavailable/rejected; fallbacks are typed; material randomness is reproducible; selection uncertainty may be unavailable. |

## Decision Discipline

Stage A approval authorized implementation-blind scientific authorship from
the exact packet. The later reviews established candidate coherence, and
`PTC-FREEZE-D001` supplies the scientific-owner freeze. No step here approves
an implementation, authorizes a repair or numerical redesign, runs validation,
or changes production status.

If approved, `PTC-SCOPE-D001` is also a cross-package input to SCI-CAL owner
question Q02: it selects CAL-before-PTC ordering for this route without
silently resolving SCI-CAL's other baseline/affine and noncommutation
questions.
