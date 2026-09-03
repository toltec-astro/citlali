# SCI-BEAM v0.1 r0.3 — Final Consistency and Freeze Report

Status: PASS; scientific authority frozen

Review date: 2026-08-17

Scope: implementation-independent documents only. No implementation, current
APT storage, audit, repair, test, validation evidence, reduction, or production
behavior was inspected.

## Final-review corrections

| Check | Result |
| --- | --- |
| Positive sensitivity | PASS — sens uses abs(flxscale) and is finite and strictly positive whenever available; signed flxscale remains preserved |
| Covariance stages | PASS — map-fit Jacobian/covariance is distinct from derived flxscale/sens propagation, with material cross-stage dependence retained |
| Centroid sign/frame | PASS — the fitted-source-centroid to detector-coordinate transformation is explicit and cannot be inferred from implementation behavior |
| Rotation support | PASS — effective detector rotation derives from parent-sample contribution support propagated through realized pixel fit support |
| Pointing transfer | PASS — bracketing pointing and associated science observations require the same immutable APT artifact and AST convention unless an authorized transform proves equivalence |
| Soft prior/convergence | PASS — the bounded scientist-facing explanation is restored without duplicating formal convergence machinery |
| Stable decision identity | PASS — SCI-BEAM-OD-001--003 appear in the rationale, formal contract, crosswalk, and external ledger and map to SCI-BEAM-ODQ-101--109 |
| Source calibration unit | PASS — the required source flux is stated directly in TOA mJy per fixed nominal beam; defensive unit/double-factor argumentation is absent |
| Rationale contents | PASS — compact single-page list; no second contents page |

## Mechanical and rendered-artifact checks

| Check | Result |
| --- | --- |
| Stable formal inventory | PASS — 46 sequential requirements and 24 sequential predictions |
| Crosswalk | PASS — exactly 70 rows, one per requirement or prediction |
| Genre separation | PASS — rationale contains no requirement/prediction inventory and imports no shared formal module |
| Shared authority | PASS — formal contract imports all six shared modules exactly once |
| LaTeX compilation | PASS — both documents compile without warnings, overfull/underfull boxes, unresolved references, or errors |
| Length | PASS — rationale 9 physical pages (title plus 8 numbered pages); formal contract 17 physical pages |
| Visual QA | PASS — all 26 Poppler-rendered pages inspected at 144 dpi; no clipping, overlap, broken table/equation, bad glyph, header/footer, or pagination defect |
| PDF properties | PASS — US Letter, unencrypted, no forms, no JavaScript, and v0.1/r0.3 metadata |
| Approved packet integrity | PASS — all three approved author-packet SHA-256 values remain exact |
| Repository mechanics | PASS — final checker, package-layout verifier, and git diff check complete cleanly |

## Final artifact hashes

- Science-team rationale SHA-256:
  cd3143784ca977bc9f8714c256ff9249f578cf71a61cd7e21784542b91a07b06
- Formal Scientific/Engineering Contract SHA-256:
  7e40afd78ec0c38941b3ac0fe800efe0792674b4a4f5dbadddf83983800a1604

## Freeze disposition

SCI-BEAM v0.1/r0.3 is the frozen implementation-independent scientific
authority. The three open decision groups and nine atomic questions block only
the claims and numerical policies named in their ledger rows; they do not
authorize implementation defaults.

Freezing does not establish implementation conformance, representation or
response fidelity, observational performance, science-impact qualification,
validation, or production readiness. There is no further editorial round.
Future revision requires one of the contract-library's four governed triggers.
