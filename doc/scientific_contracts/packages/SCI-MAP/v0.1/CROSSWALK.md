# SCI-MAP v0.1 Contract Crosswalk

Document revision: `r0.1`

Status: author-draft traceability aid; it does not assert implementation,
validation, or production status.

## Authority key

- **SB**: owner-approved `SCOPE_BRIEF.md`, SHA-256
  `e2a9eb51edb5956191813b4cdbd23866e875d52cdf89cd8b6c272988b4f26674`.
- **SS**: owner-approved `AUTHOR_SUPERSESSION_COVER.md`, SHA-256
  `8ea283525f18199d9760c3f672d145d71f0db87b320ab2e88a5c6635ef3d4aa0`.
- **FC**: exact admitted frozen core
  `c28f18ed089657dae278caba2d6d6d65c7ec72f4:doc/audits/packages/SCI-MAP-001_INDEPENDENT_CORE.tex`,
  SHA-256
  `13dd5922bd492e381afcc3b015284216dde1ccc2199ece3d070ee577c7324381`.
- **CO**: owner-approved `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`, SHA-256
  `2d478cb6c5e897308d19614b8b01663318744971850c67459f84c7ddcd57c5c9`.

The canonical requirement and prediction text lives only in
`src/SCI-MAP-v0.1_SHARED_AUTHORITY_r0.1.tex`. Both PDFs render that same source.
Section names below are navigation aids, not a second normative wording.

Seven owner decisions remain `OPEN` in
`SCIENTIFIC_OWNER_DECISION_LEDGER.md`. In particular, SCI-MAP-OD-007 governs
the currently unspecified numeric domain, units or dimensionless status, and
failure disposition of `c=coverage_cut`. A crosswalk gate naming OD-007 never
supplies a range: pending an owner answer, the exact state must be explicitly
admitted by the owner-authorized effective support policy or fail closed.

## Requirement crosswalk

| Requirement | Admitted authority | Scientific-rationale location | Engineering location | Linked predictions | Open owner/dependency gate |
|---|---|---|---|---|---|
| SCI-MAP-REQ-001 | SB metadata/status | Cover; App. C | Sec. 1; Sec. 6 | -- | Owner freeze still required |
| SCI-MAP-REQ-002 | SB 2, 3, 6, 8; SS 1, 9; CO Capability | Sec. 1 | Sec. 4 canonical clause | PRED-024 | CAL meaning remains upstream |
| SCI-MAP-REQ-003 | SB 3, 6; SS 10; CO Identity | Sec. 7 | Sec. 4 canonical clause | PRED-014 | Occurrence binding supplied upstream |
| SCI-MAP-REQ-004 | SB 3, 5; CO Validity/Producers | Sec. 1, 5 | Sec. 4 canonical clause | PRED-011 | CAL/VAL facts remain upstream |
| SCI-MAP-REQ-005 | SB 2, 3, 5, 6; CO Shape/Frames | Sec. 1, 3, 7 | Sec. 4 canonical clause | PRED-005, PRED-009, PRED-011 | ALIGN/AST facts remain upstream |
| SCI-MAP-REQ-006 | SB 2, 3, 6; SS 1, 2; CO Units | Sec. 1 | Sec. 4 canonical clause | PRED-010 | PTC coefficient status remains upstream |
| SCI-MAP-REQ-007 | SB 3, 4; FC Hits/exposure; CO Units | Sec. 5 | Sec. 4 canonical clause | PRED-008 | Exposure accounting supplied upstream |
| SCI-MAP-REQ-008 | SB 2--4, 10; SS 8; CO Validity | Sec. 2, 8 | Sec. 4 canonical clause | PRED-013 | OD-003 |
| SCI-MAP-REQ-009 | SB 3, 4, 6; FC lifecycle; CO State | Sec. 7 | Sec. 4 canonical clause | PRED-014 | None |
| SCI-MAP-REQ-010 | SB 3, 6; FC ordinary gridder/validity | Sec. 1 | Sec. 3--4 | PRED-010, PRED-011 | VAL policy remains upstream |
| SCI-MAP-REQ-011 | SB 6.1; SS 1; FC ordinary gridder | Sec. 1 | Sec. 3--4 | PRED-003, PRED-004 | None |
| SCI-MAP-REQ-012 | SB 4, 6; FC ordinary gridder/validity | Sec. 1--2, 5 | Sec. 3--4 | PRED-003, PRED-025 | OD-007 when the effective support policy uses `c` |
| SCI-MAP-REQ-013 | SB 1, 4, 6; SS 2; FC formal covariance | Sec. 1, 4 | Sec. 4 | PRED-004, PRED-022 | PTC statistical status remains upstream |
| SCI-MAP-REQ-014 | SB 6.3, 7E; FC response/input classes | Sec. 2--3 | Sec. 4 | PRED-001, PRED-002, PRED-008 | Stronger response evidence not supplied |
| SCI-MAP-REQ-015 | SB 7A--B; FC WLS relation/covariance | Sec. 3--4 | Sec. 4 | PRED-004 | Projection meaning supplied upstream |
| SCI-MAP-REQ-016 | SB 4, 6.4; SS 8; FC kernel/noise/coadd | Sec. 1--2, 6, 8 | Sec. 4 | PRED-020 | Exact support-row identity comes from the effective policy |
| SCI-MAP-REQ-017 | SB 2--4, 7A; FC kernel-response identity | Sec. 2 | Sec. 4 | PRED-005, PRED-006, PRED-013 | OD-003 |
| SCI-MAP-REQ-018 | SB 6.7; SS 3; FC centering | Sec. 2, 6 | Sec. 4 | PRED-001, PRED-015 | Upstream null modes remain dependencies |
| SCI-MAP-REQ-019 | SB 4, 7B; FC full covariance | Sec. 2, 4 | Sec. 4 | PRED-004, PRED-019 | PTC covariance remains upstream; covariance domain is exact authorized rows |
| SCI-MAP-REQ-020 | SB 6.2, 7B; SS 2; FC formal covariance | Sec. 4 | Sec. 4 | PRED-004, PRED-019 | Coefficient calibration/correlation evidence absent |
| SCI-MAP-REQ-021 | SB 4, 7B; FC formal diagonal weight | Sec. 4 | Sec. 4 | PRED-004, PRED-022 | OD-004 for persistence form |
| SCI-MAP-REQ-022 | SB 4, 7B, 10; FC conditioned inputs/systematics | Sec. 4 | Sec. 4 | PRED-019 | OD-004; CAL/AST/PTC nuisance facts upstream |
| SCI-MAP-REQ-023 | SB 7B; FC conditioning rule | Sec. 2, 4 | Sec. 4 | PRED-020 | Joint model/resampling outside MAP fixed-state claim |
| SCI-MAP-REQ-024 | SB 7B, 8; CO Statistical labels | Sec. 4, 8 | Sec. 4 | PRED-022 | NOI empirical calibration remains upstream |
| SCI-MAP-REQ-025 | SB 4, 6.9; SS 6; CO Validity | Sec. 5 | Sec. 4 | PRED-010, PRED-011, PRED-017 | None |
| SCI-MAP-REQ-026 | SB 4; FC hits; CO Units | Sec. 5 | Sec. 4 | PRED-004, PRED-010 | Fractional counting convention must be declared |
| SCI-MAP-REQ-027 | SB 4; FC weighted hits | Sec. 5 | Sec. 4 | PRED-010, PRED-011 | None |
| SCI-MAP-REQ-028 | SB 4; SS 6, 8; FC coadd companions | Sec. 5--6 | Sec. 4 | PRED-017 | None |
| SCI-MAP-REQ-029 | SB 4; CO Units/Validity | Sec. 5 | Sec. 4 | PRED-008, PRED-010 | Exposure population supplied upstream |
| SCI-MAP-REQ-030 | SB 4, 6.4; SS 8; FC coverage | Sec. 5 | Sec. 4 | PRED-008, PRED-010 | None |
| SCI-MAP-REQ-031 | SB 6.10, 10; SS 7 | Sec. 5 | Sec. 4 | PRED-012 | OD-001, OD-002, OD-007 |
| SCI-MAP-REQ-032 | SB 6.10; SS 7; CO Validity | Sec. 5 | Sec. 4 | PRED-012 | OD-001, OD-002, OD-007 |
| SCI-MAP-REQ-033 | SB 4, 6.8--9; SS 5--6; CO Validity | Sec. 5, 8 | Sec. 4 | PRED-011, PRED-023, PRED-025 | OD-003 determines required-response set; OD-007 gates `c` admission |
| SCI-MAP-REQ-034 | SB 6.8; SS 5; FC validity; CO Missing data | Sec. 8 | Sec. 3--4 | PRED-011, PRED-025 | VAL cause policy remains upstream |
| SCI-MAP-REQ-035 | SB 7C; FC validity | Sec. 5, 8 | Sec. 4 | PRED-010, PRED-011 | Low-positive policy is the adopted support rule; OD-007 gates `c` admission |
| SCI-MAP-REQ-036 | SB 4, 6.5; SS 5--8; FC downstream bundle | Sec. 6--8 | Sec. 3--4 | PRED-013, PRED-014 | OD-003, OD-004 |
| SCI-MAP-REQ-037 | SB 6.5, 6.12; SS 4; FC coadd admission | Sec. 6, 8 | Sec. 3--4 | PRED-014, PRED-017 | None |
| SCI-MAP-REQ-038 | SB 6.6, 6.11; SS 4; CO Shape/Frames | Sec. 6--7 | Sec. 3--4 | PRED-018 | ALIGN/AST/WCS authority upstream |
| SCI-MAP-REQ-039 | SB 1--2, 6.1; SS 1, 4; FC coadd | Sec. 6 | Sec. 4 | PRED-015, PRED-016 | OD-007 when the effective coadd support policy uses `c` |
| SCI-MAP-REQ-040 | SB 4, 6.4--5; SS 8; FC coadd companions | Sec. 5--6 | Sec. 4 | PRED-015, PRED-017, PRED-018 | OD-003 for unavailable response; exact row set comes from effective policy |
| SCI-MAP-REQ-041 | SB 4, 7B; SS 2; FC coadd covariance | Sec. 4, 6 | Sec. 4 | PRED-019 | Cross-observation covariance status upstream; domain is exact authorized rows |
| SCI-MAP-REQ-042 | SB 2, 7B, 8; SS 1; FC correlated coadd | Sec. 6 | Sec. 4 | PRED-019, PRED-024 | Correlated GLS excluded from v0.1 |
| SCI-MAP-REQ-043 | SB 3--4, 6.13; SS 10; CO Identity | Sec. 7 | Sec. 4 | PRED-014 | Occurrence binding supplied upstream |
| SCI-MAP-REQ-044 | SB 3, 6.14; SS 9; CO Frames/Units | Sec. 1, 7 | Sec. 4 | PRED-005, PRED-024 | OD-006 for Point/OOF registered reuse |
| SCI-MAP-REQ-045 | SB 6.11, 6.13; CO Shape/Frames | Sec. 6--7 | Sec. 4 | PRED-018 | ALIGN/AST serialization reference needed for fidelity |
| SCI-MAP-REQ-046 | SB 4, 7D; FC lifecycle; CO State | Sec. 7 | Sec. 4 | PRED-014, PRED-020 | Effective support policy supplies exact row-set identity |
| SCI-MAP-REQ-047 | SB 4, 6.12, 7C; CO Missing data | Sec. 8 | Sec. 3--4 | PRED-014, PRED-025 | Exact product scope comes from effective plan |
| SCI-MAP-REQ-048 | SB 4, 6.12, 10; CO Missing data | Sec. 8, 10 | Sec. 4 | PRED-014, PRED-025 | OD-005 |
| SCI-MAP-REQ-049 | SB 4, 6.12; SS 8; CO Consumers | Sec. 8 | Sec. 4 | PRED-023 | None |
| SCI-MAP-REQ-050 | SB 1, 4--5, 7D; FC downstream contract; CO Consumers | Sec. 8 | Sec. 4 | PRED-013, PRED-022, PRED-023 | OD-003, OD-004, OD-006 |
| SCI-MAP-REQ-051 | SB 5, 6.4, 7E; FC noise-operator test | Sec. 8 | Sec. 4 | PRED-020, PRED-021 | NOI distribution remains upstream; OD-007 gates any `c`-bearing fixed policy |
| SCI-MAP-REQ-052 | SB 2, 5, 8; SS 11--13; CO Adjacent packages | Sec. 1, 9--10 | Sec. 2, 6 | PRED-005--009, PRED-022--024 | OD-006; all named adjacent-package science excluded |

## Prediction-to-requirement crosswalk

| Prediction | Governing requirement(s) | Primary evidence class |
|---|---|---|
| SCI-MAP-PRED-001 | REQ-014, REQ-018 | Analytic constant input |
| SCI-MAP-PRED-002 | REQ-014, REQ-019 | Analytic unequal coefficients |
| SCI-MAP-PRED-003 | REQ-011, REQ-012 | One-pixel hand calculation plus exact support-row oracle |
| SCI-MAP-PRED-004 | REQ-015, REQ-019--021, REQ-026 | Fractional projection matrix fixture |
| SCI-MAP-PRED-005 | REQ-005, REQ-017, REQ-044, REQ-052 | Support-row-restricted delta injection and WCS check |
| SCI-MAP-PRED-006 | REQ-017, REQ-052 | Point-template injection |
| SCI-MAP-PRED-007 | REQ-014, REQ-017, REQ-052 | Extended/gradient/Fourier injections |
| SCI-MAP-PRED-008 | REQ-007, REQ-014, REQ-029--030 | Variable-coverage analytic fixture |
| SCI-MAP-PRED-009 | REQ-005, REQ-014, REQ-052 | Finite-edge injection |
| SCI-MAP-PRED-010 | REQ-006, REQ-025--027, REQ-029--030, REQ-035 | Coefficient-state truth table |
| SCI-MAP-PRED-011 | REQ-004--005, REQ-010, REQ-025, REQ-033--035 | Eligibility/non-finite fixture |
| SCI-MAP-PRED-012 | REQ-031--032 | Integer-index, threshold, exact-`c` admission, and fail-closed oracle |
| SCI-MAP-PRED-013 | REQ-008, REQ-017, REQ-033, REQ-036, REQ-050 | Typed unavailable response fixture |
| SCI-MAP-PRED-014 | REQ-003, REQ-009, REQ-036--037, REQ-043, REQ-046--048 | Atomic bundle rejection |
| SCI-MAP-PRED-015 | REQ-018, REQ-039--040 | One-observation coadd with exact observation/coadd row sets |
| SCI-MAP-PRED-016 | REQ-039 | Unequal-coefficient coadd on authorized rows |
| SCI-MAP-PRED-017 | REQ-025, REQ-028, REQ-037, REQ-040 | Missing/invalid observation and absence of unsupported substitute rows |
| SCI-MAP-PRED-018 | REQ-038, REQ-040, REQ-045 | Centered-integer placement |
| SCI-MAP-PRED-019 | REQ-019--022, REQ-041--042 | Exact-row cross-observation covariance fixture |
| SCI-MAP-PRED-020 | REQ-016, REQ-023, REQ-046, REQ-051 | Fixed row-selected operator realization |
| SCI-MAP-PRED-021 | REQ-051--052 | Sequential/parallel comparison |
| SCI-MAP-PRED-022 | REQ-013, REQ-021, REQ-024, REQ-050, REQ-052 | Prohibited-label challenge |
| SCI-MAP-PRED-023 | REQ-033, REQ-049--050, REQ-052 | Raw-validity promotion challenge |
| SCI-MAP-PRED-024 | REQ-002, REQ-042, REQ-044, REQ-052 | Method-boundary challenge |
| SCI-MAP-PRED-025 | REQ-012, REQ-033--034, REQ-047--048 | Aggregate/index/publication failure injection |

## Completeness invariant

- Canonical requirements: **52**, `SCI-MAP-REQ-001` through
  `SCI-MAP-REQ-052`, with no gaps.
- Canonical predictions: **25**, `SCI-MAP-PRED-001` through
  `SCI-MAP-PRED-025`, with no gaps.
- Every requirement appears once in the requirement table above.
- Every prediction appears once in the prediction table above and is linked to
  at least one requirement.
- Open owner decisions: **7**, `SCI-MAP-OD-001` through
  `SCI-MAP-OD-007`, with no gaps. The scientist-facing PDF carries a register
  generated from the ledger and a compact requirement/crosswalk summary.
- The engineering PDF renders the same canonical requirement and prediction
  declarations as the scientist-facing PDF; it adds no independent science.
