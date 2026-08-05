# SCI-MAP-001 final narrow independent re-audit — 2026-08-05

## Audit identity and boundary

This is the final contract-first re-audit of the narrow SCI-MAP-001
bookkeeping repair. It is an audit and disposition record, not a repair.

- Verified clean starting HEAD and `codex/repair-sci-map-001` tip:
  `af0c849ce59a5f80e5efc8db435bb6662863052f`.
- Candidate parent:
  `f84b9fd7d7364f9d35317fc6c15b55d2a30e89f7`.
- Second-cycle independent audit:
  `fc26e24e6543d1102f9fcc9bf4e849369b39dd04`.
- Project-owner amendment:
  `6409a36d324072c9b29145c620d01a0686275870`.
- Amendment SHA-256 at both the authority commit and candidate:
  `52be19700b73659ba1847012d4cb0766407399cda5899570acb79bf5b45221f3`.
- Audit-only branch: `codex/final-reaudit-sci-map-001`, created from the
  verified clean candidate.

No application code, test, configuration, original audit/amendment artifact,
canonical ledger, coordination snapshot, or external corpus was modified by
this audit. No Unity access, repair, campaign, task launch, delegation,
integration, or push was performed. The owner-accepted seven-case corpus was
not inspected, copied, or rerun.

## Contract fixed before implementation inspection

The following stage-aware cardinality policy was stated before inspecting the
candidate diff:

- observation empirical maps and realization writes use observation output
  stages;
- coadd empirical maps and realization writes use coadd output stages;
- with coaddition and filtering enabled, three maps, two realizations, one
  observation, and one coadd must record exactly 9 empirical product maps and
  18 realization writes; and
- filtering-disabled and non-coadd states remain unchanged.

## Narrow independent inspection

`af0c849` is a direct child of `f84b9fd7`. Its diff contains four files:

- candidate state text in `doc/INTEGRATION_LEDGER.md` and
  `doc/REFACTOR_STATUS.md`;
- the bookkeeping function in
  `include/citlali/core/pipeline/noise_execution_plan.h`; and
- one regression case in `tests/test_config_scaffold.cpp`.

The sole application-source change replaces one global filtered-stage
multiplier with already-established observation and coadd stage counts. The
completion function only populates `NoiseRealizedState`; its call occurs after
mapmaking and coadd completion, and the existing serializer carries the two
computed counts directly into the noise provenance sidecar.

The three policy states are:

| State | Observation stages | Coadd stages | Empirical maps | Realization writes |
| --- | ---: | ---: | ---: | ---: |
| non-coadd, filtered | 2 | 0 | 6 | 12 |
| coadd, unfiltered | 1 | 1 | 6 | 12 |
| coadd, filtered | 1 | 2 | 9 | 18 |

For the repaired state, observation products are `3 x 1 = 3` and coadd
products are `3 x 2 = 6`. Observation realization writes are
`(3 x 2) x 1 = 6` and coadd realization writes are `(3 x 2) x 2 = 12`.
The totals are therefore exactly 9 and 18. The focused selection passes all
three states.

No numerical primitive, random-number generation, mapmaking, coadd, WCS,
threshold, or output-routing source changed between `f84b9fd7` and
`af0c849`. Accepted finite-domain results and arithmetic order are therefore
unchanged by this repair. The focused truth, TSan, and production writer suites
independently remain green.

## Finding adjudication

| Finding | Final proposal | Basis |
| --- | --- | --- |
| F001 | **closed; not reopened** | The final diff does not touch the accepted sequential/requested-parallel or concurrent realization path; TSan remains 9/9. |
| F002 | **closed; not reopened** | Validity, support, and companion policies are outside the final diff and remain covered. |
| F003 | **closed; not reopened** | Admission, identity, centering, and atomicity paths are unchanged. |
| F004 | **close after final re-audit** | The second-cycle persistence repair is unchanged, and the sole remaining coadd-plus-filter provenance defect now records the exact 9 products and 18 writes. |
| F005 | **confirm second-cycle closure** | Aggregate/index safety code is unchanged; its focused failure and atomicity tests pass. |
| F006 | **closed; not reopened** | Nonprecision coefficient semantics and the absence of unauthorized precision claims are unchanged. |
| F007 | **confirm second-cycle closure** | WCS and threshold paths are unchanged; all 22 production FITS tests pass the owner-amended contract. |
| F008 | **closed; not reopened** | Lossless one-way realized provenance remains intact; the corrected counts flow through the existing serializer. |
| F009 | **closed; not reopened** | Centered integer embedding, `L=I`, strict admission, and historical finite arithmetic are unchanged. |
| F010 | **confirm second-cycle closure** | The eight facts, masks, aliases, sidecar authority, aggregate safety, and amended FITS-card relations remain covered without arithmetic changes. |
| F011 | **close after final re-audit** | The previously absent and failing coadd-plus-filter state now has an exact regression and passes together with every proportionate complete local gate. |
| F012 | **closed only in bounded owner-accepted scope** | Accept only the amendment's exact-`ed28dafb` execution, completion, product/inventory, visible observation/coadd, and SEQ/OMP claims. |
| F013 | **remain open and conditioned** | MAP evidence closes none of ALIGN, CAL, AST, PTC, or VAL; production remains `existing_use_only`. |

F012 retains these named external-evidence limitations:

- no independent raw manifest or sample ledger;
- no scan-farm pre-normalization planes or commit-order trace;
- no wrapper/Slurm, environment, collection, or retrieval chain; and
- no same-case S-X observation-realization files in the historical corpus.

Those absences neither reopen F012 inside its owner-accepted scope nor support
claims about unobserved internal behavior. They are not rerun requests.

## Independent gates at exact `af0c849`

| Gate | Result |
| --- | --- |
| clean entry and exact HEAD/repair-tip identity | pass |
| candidate parent and amendment current/authority digest | pass |
| six required Release build targets | pass |
| CLI identity | `v4.0.0-3632-gaf0c849ce` |
| exact stage-cardinality selection | 3/3 pass |
| complete CTest | 594 registered; 593/593 enabled pass; zero failures; one pre-existing disabled lifecycle test |
| focused science-map truth/provenance | 31/31 pass |
| focused ThreadSanitizer | 9/9 pass; no sanitizer report |
| production FITS products | 22/22 pass |
| baseline-tool unit tests | 147/147 pass |
| full config preflight | 127/127 unit tests; four mode kits; eight compatibility cases; 100% compact coverage; all boundary audits pass |
| validation ledger | pass; 60 records |
| validation profile registry | pass; four active and eight preparing profiles |
| science-change ledger | pass; three changes and five integration commits |
| candidate diff check and final pre-artifact worktree check | pass |

There was no required-data skip and no unexpected error- or critical-level
application record in an accepted gate.

## Final disposition and coordinator handoff

The bounded MAP scientific-contract repair/re-audit chain is complete. F004
and F011 may close, the second-cycle F005/F007/F010 closures are confirmed,
and no prior scoped closure is reopened. The package is ready for coordinator
ledger integration at exact candidate `af0c849`.

This is not upstream production eligibility. F013 and its SCI-ALIGN-001,
SCI-CAL-001, SCI-AST-001, SCI-PTC-001, and SCI-VAL-001 dependencies remain
open, and production remains `existing_use_only`. Only the coordinator may
update canonical state or authorize application integration or production
expansion.

No further owner choice is required for this MAP disposition.
