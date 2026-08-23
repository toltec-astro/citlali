# Citlali Scientific Contract Package Index

This index is governed by the
[Scientific Contract Library Program](README.md). Every new package begins by
linking to that charter and completing the prior-work recovery procedure.
The [prior-work discovery registry](PRIOR_WORK_REGISTRY.md) seeds that search
without replacing package-specific review.
The [pilot process review](PILOT_PROCESS_REVIEW_2026-08-16.md) records the
permanent CAL/MAP lessons and Stage A/Stage B gate used by later packages.
The [consolidation ledger](CONSOLIDATION_LEDGER_2026-08-22.md) records the
canonical branch, package provenance, retained revision history, and known
recovery gaps before the next program-wide audit.

The package-neutral
[WP-2 timestream boundary authority](boundaries/v0.1/README.md) composes the
approved RTC-to-AST sample-grid, detector-geometry/field-rotation, and
occurrence-level exposure-lineage decisions. Exact artifact approval was
recorded on `2026-08-23`; clean-room re-audit remains pending and MAP work
remains deferred.

| Package | Scope status | Contract status | Owner decision needed | Next action |
| --- | --- | --- | --- | --- |
| [SCI-CAL — detector calibration, extinction, and signal transfer](packages/SCI-CAL/v0.1/README.md) | Owner-approved scientific contract v0.1 scope | **Scientific authority frozen** at v0.1/r0.5-r0.4 on `2026-08-23`; implementation conformity and achieved performance unassessed | No scientific decision remains open in Q01--Q09; named numerical uncertainty products and achieved-performance acceptance remain unavailable until their evidence exists | Preserve the frozen packet; execute the owner-approved validation workflow separately and report the evidence honestly |
| [SCI-MAP — ordinary mapmaking and observation coaddition](packages/SCI-MAP/v0.1/README.md) | Owner-approved scientific contract v0.1 scope | Science-team rationale r0.3 house version frozen; CI-001 resolved; 52 requirement IDs and 25 prediction IDs retained; scientific authority not frozen | Resolve `SCI-MAP-OD-001--009`; OD-007 now concerns numerical domain and failure policy only | No further stylistic round; revise only for owner decisions, normative change, new evidence, or inconsistency |
| [SCI-BEAM — Beammap effective PSF, calibration, sensitivity, and APT](packages/SCI-BEAM/v0.1/README.md) | Owner-approved v0.1 boundary; absolute boresight/pivot excluded | Scientific authority frozen at r0.3; 46 requirements and 24 predictions; implementation conformity unassessed | Resolve three stable decision groups containing nine atomic sensitivity, adequacy, wing, pivot, accuracy, and kernel questions | No further editorial round; revise only for a resolved decision, normative change, evidentiary-status change, or genuine inconsistency |
| [SCI-ALIGN — detector-reference alignment](packages/SCI-ALIGN/v0.1/README.md) | Approved scope and exact r0.3 authority retained; original standalone Stage A files were not recoverable and the gap is explicit | **Scientific authority frozen** at v0.1/r0.3 on `2026-08-22`; 55 requirements and 26 predictions; implementation conformity not assessed | `SCI-ALIGN-ODQ-101--105` and `110` remain open; `109` is deferred; each blocks only its named field or claim | Preserve the frozen packet; resolve typed owner questions or launch a separately governed implementation-conformity audit |
| [SCI-AST — astrometric coordinate realization](packages/SCI-AST/v0.1/README.md) | Approved scope and exact r0.3 authority retained; original standalone Stage A files were not recoverable and the gap is explicit | **Scientific authority frozen** at v0.1/r0.3 on `2026-08-22`; 90 requirements and 50 predictions; implementation conformity not assessed | `AST-OWNER-Q001--004`, `006`, and `007` remain open; Q005 is deferred; Q008 is closed | Preserve the frozen packet; resolve typed owner questions or launch a separately governed implementation-conformity audit |
| [SCI-RTC — raw-timestream conditioning and temporal response](packages/SCI-RTC/v0.1/README.md) | Owner-approved Stage A scope and packet; bounded r0.10--r0.12 reopening decisions retained | **Scientific authority frozen** at v0.1/r0.12 on `2026-08-21`; implementation conformity not assessed | Preserve the 63 open, one conditional, 34 resolved, and five deferred ledger states; no hidden defaults | Preserve the frozen packet; separately govern any conformity, validation, or successor-authority work |
| [SCI-PTC — correlated-mode cleaning and detector coefficients](packages/SCI-PTC/v0.1/README.md) | Owner-approved Stage A scope/packet; Q001--Q003 and `WP1-OWNER-D001--D009` resolve diagnostic `r`, frozen-subspace projection, transformed-signal/truth-rule defects, disabled routing, configured grouping, positive rank, centering, immutable-parent fit, and the time-local full-rank guard | Scientific authority frozen at v0.1/r0.5 on `2026-08-23`: 99 requirements, 60 predictions, exact 159-row crosswalk, canonical 13-page rationale and 26-page engineering view; implementation conformity not assessed | Preserve 10 unresolved entries: three open, four known-but-not-supplied, and three deferred roles remain unavailable; no audit finding is closed before clean-room re-audit | Stabilize CAL and the required timestream boundaries, bind final CAL/PTC sources into VAL, then run the timestream-only clean-room re-audit; MAP remains deferred |
| [SCI-VAL — sample/detector validity, flags, and map eligibility](packages/SCI-VAL/v0.1/README.md) | Owner-approved r0.3: VAL Core evaluates; the Registry binds actual policy ownership; `SCI-VAL:independent_exposure@1`, distinct aggregate profiles, exact conflict precedence, four response/uncertainty roles, homogeneous atomic compatibility, non-circular generations, and continuing source bindings are explicit | Surgical Stage B r0.3 manager-reviewed: 49 requirements, 24 predictions, standalone 8-page rationale and complete 20-page engineering view; scientific authority not frozen | No general SCI-VAL science question open; QB001/QB003 serialization is engineering-deferred and QB006 combine/equivalence detail is profile-local | Explicit scientific-owner review and freeze disposition; then author exact package-owned profiles separately; no conformity, validation, or readiness work yet |
| NOI — noise realizations and empirical uncertainty | Inventoried | Not started | None yet | Hold for tranche planning |
| FLT — Convolve, Wiener, and lowpass filtering | Prior Convolve material identified | Not started | None yet | Recover reusable science without importing audit/repair findings |
| SRC/MODE — source fitting, Pointing, and OOF | Inventoried | Not started | None yet | Hold for tranche planning |
| FRUIT — fruit-loop feedback, learning, and restart | Inventoried | Not started | None yet | Hold until single-pass contracts are established |

Enabled polarimetry and measured R-channel execution remain outside the active
inventory until their scientific execution boundaries are approved.
