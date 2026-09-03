# SCI-RTC v0.1/r0.8 Consistency Report

Date: `2026-08-20`

Status: implementation-blind author self-check of the binding Decision 9
revision. This is not an implementation-conformity, validation, science-impact,
readiness, or production assessment.

## Decision coverage

- The scientific rationale states `x(t) = s(t) + b_i` on stable plateaus and
  excludes gain/responsivity-change modeling from v0.1.
- DEF-035--038 and EQ-033--034 define additive plateaus, a finite transition
  interval in physical time, timing-vector-derived sample support, stable
  plateau offset estimation, and the response-changing successor boundary.
- REQ-094--102 and REQ-106--107 require compact production state, explicit
  unmodeled transition flags, exclusion from both plateau fits, distinct
  propagated influence, optional plan-selected additive correction, no
  invented offset under insufficient support, retained segment boundaries,
  and no fitted gain-change model.
- PRED-055, PRED-059, PRED-070, and PRED-071 falsify cadence-dependent widths,
  transition repair, ambiguous or unselected correction, gain fitting, and
  invented offsets.
- OWNER-075 and RTC-SCI-D009 record the owner decision; OWNER-059--065 retain
  the still-open numerical estimator, threshold, support, validation, and
  fallback choices.
- `CROSSWALK.md` and `RATIONALE_TO_CONTRACT_CROSSWALK_R0.8.md` route every
  Decision 9 element to the rationale and shared formal authority.

## Mechanical and PDF checks

- `src/verify_contract.py`: pass; inventories remain 38 definitions, 37
  equation tags, 12 assumptions, 108 requirements, and 71 predictions.
- Owner ledger: 75 sequential entries: 63 open, one conditional, six resolved,
  and five deferred.
- Both Tectonic builds completed with no TeX warning, overfull, underfull,
  undefined-reference, or error message.
- PDF metadata reports revision r0.8, US Letter, unencrypted, no forms, and no
  JavaScript.
- Poppler rendered all 15 scientific-rationale pages and all 42 engineering
  pages at 150 dpi. All 57 pages were inspected; no clipping, overlap, equation
  overflow, table overflow, missing glyph, or malformed page was found.

The package remains a scientific-owner review candidate and is not frozen.
