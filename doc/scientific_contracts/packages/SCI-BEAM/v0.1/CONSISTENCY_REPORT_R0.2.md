# SCI-BEAM v0.1 r0.2 — Consistency Report

Status: complete for the r0.2 authorship artifacts; draft not frozen

Review date: `2026-08-17`

Scope: first-principles documents only. No implementation, existing APT
contract/file, audit, repair, test, or production reduction was inspected.

## Directive checks

| Check | Disposition | Evidence location |
| --- | --- | --- |
| Every input/output has a unit and reference plane | PASS at source review | formal notation, inputs, equations, APT schema |
| `x` and `xs` mean raw `Delta f/f` | PASS | rationale Executive/Sec. 1; REQ-002--003 |
| Fitted amplitude remains in `Delta f/f` | PASS | model equation; REQ-013 |
| Source prediction is TOA mJy per fixed nominal beam | PASS | rationale Sec. 2; REQ-007 |
| No extra `H(0)` factor enters calibration | PASS | flxscale equation; REQ-039; PRED-008 |
| Source atmosphere is handled once in BEAM | PASS | rationale Sec. 5; REQ-039--040 |
| Target atmosphere remains outside BEAM | PASS | rationale Sec. 5; assumptions A14 |
| `flxscale` is BEAM-owned and APT-published | PASS | definitions; REQ-039--040, REQ-044 |
| `sens` is BEAM-owned and NEFD-like | PASS | rationale Sec. 6; REQ-041--042 |
| `responsivity` has no canonical role | PASS | rationale Sec. 8; REQ-043; PRED-022 |
| PSF is an effective Beammap core, not intrinsic/complete | PASS | rationale Secs. 1, 7; REQ-022--023 |
| Full rotation/off-diagonal tensor is fitted | PASS | tensor equation; REQ-010--011 |
| Complete model Jacobian and joint covariance required | PASS | formal equations; REQ-019--021 |
| Raw and horizon detector coordinates remain distinct | PASS | rationale Sec. 4; REQ-036 |
| Coordinate origin is arbitrary and recorded | PASS | rationale Sec. 4; REQ-037 |
| Physical pivot is not falsely claimed known | PASS | rationale Sec. 4; REQ-037 |
| Same-APT pointing-transfer rule is explicit | PASS | rationale Sec. 4; REQ-038 |
| Independent component validity replaces all-or-nothing result | PASS | rationale Executive/Sec. 8; REQ-029--031 |
| Beammap APT is mandatory | PASS | rationale Sec. 8; REQ-044 |
| APT schema is implementation-independent | PASS | formal APT section; REQ-044 |
| Array-level stacking remains optional | PASS | rationale Sec. 7; REQ-045 |
| No production thresholds invented | PASS | rationale Secs. 7, 9; REQ-032 |
| Evidence layers remain separate | PASS | rationale Sec. 9; REQ-046 |

## Cross-artifact checks

- Formal contract is the only artifact carrying the complete normative
  inventory and exact convergence machinery.
- Science-team rationale contains no requirement or prediction IDs and does
  not import the common formal modules.
- Crosswalk contains exactly one row per requirement and prediction.
- Decision ledger contains only nine genuinely unresolved scientific choices.
- Change log maps every r0.1 requirement and prediction group to r0.2.
- Cross-document follow-up records required adjacent-authority changes without
  modifying those packages.

## Closure checks

| Check | Result |
| --- | --- |
| LaTeX compilation | PASS — both canonical sources compiled without warnings, overfull/underfull boxes, unresolved references, or errors |
| Formal PDF identifier coverage | PASS — all 46 sequential requirements and all 24 sequential predictions occur in the rendered formal contract |
| Rationale genre separation | PASS — no formal requirement/prediction IDs and no imports of the six shared normative modules |
| Crosswalk coverage | PASS — exactly 70 rows, one per stable requirement or prediction ID |
| Decision ledger | PASS — exactly nine open items, `SCI-BEAM-ODQ-101--109` |
| Length | PASS — rationale 9 physical pages (title plus 8 numbered main-text pages); formal contract 17 physical pages (title plus 16 numbered pages) |
| Rendered visual QA | PASS — Poppler rendered all 26 pages at 144 dpi; every page was inspected for clipping, overlap, table/equation breakage, bad glyphs, and header/footer defects |
| PDF properties | PASS — US Letter, unencrypted, no forms, no JavaScript, and r0.2/v0.1 titles and document metadata present |
| Approved packet integrity | PASS — all three content hashes remain byte-for-byte equal to the approved manifest |
| Repository mechanics | PASS — `verify_layout.py` and `git diff --check` complete cleanly |

## Final artifact hashes

- Science-team rationale SHA-256:
  `a0a2fde8587d5e05c0fb46196e1cd2fdf0214cbe2a11cc8bcaccdd20f3e175e6`
- Formal Scientific/Engineering Contract SHA-256:
  `0a2a94d510ca73fa7a53a8e8af8025ab78f1810769dda9067bf115b36e4a1773`

This report closes document consistency and rendering only. It does not accept
or freeze the scientific authority and does not establish implementation
conformance, representation/response fidelity, observational performance,
science-impact qualification, or production readiness.
