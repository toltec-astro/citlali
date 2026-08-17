# SCI-BEAM v0.1 Stage B exact crosswalk

Status: implementation-blind author draft with manager-added approved-decision
traceability, document revision `r0.1`.

The author drafted the row-level owner-decision column using only decision
content expressed in the approved packet. The information firewall correctly
withheld the decision log and owner ledger, so the author recorded the missing
ID mapping as `SCI-BEAM-ODQ-001`. After the author froze `r0.1`, the contract
manager supplied the exact mapping below from the already approved
[`DECISION_LOG.md`](DECISION_LOG.md). This is traceability metadata only; it
does not alter or newly approve the scientific content.

## Exact approved-decision mapping

| Decision | Exact approved disposition | Principal crosswalk coverage |
| --- | --- | --- |
| `BEAM-SCOPE-D001` | V0.1 begins from an elliptical two-dimensional beam convolved with the declared calibrator brightness model and explicitly bounded background terms. Model identity is typed; claims outside the family are unavailable. | `REQ-008--012`, `REQ-016`, `PRED-001--012` |
| `BEAM-SCOPE-D002` | Calibrator brightness and beam remain separate forward-model components. Point-source treatment is an explicit limiting case, never a silent assumption. | `REQ-005`, `REQ-008`, `REQ-010--011`, `REQ-018`, `PRED-001--002`, `PRED-007--008`, `PRED-010`, `PRED-023` |
| `BEAM-SCOPE-D003` | The author derives a general likelihood/covariance statement and may define an explicitly conditional diagonal approximation. Every reported covariance names omitted correlation and nuisance terms. | `REQ-014--019`, `REQ-033`, `PRED-019`, `PRED-023` |
| `BEAM-SCOPE-D004` | Soft priors may initialize and bound candidate gating only. They are not exact UID or position truth, cannot impose unconditional veto, must record influence, and require blind fallback. | `REQ-021--025`, `PRED-013--017` |
| `BEAM-SCOPE-D005` | Internal iteration is an observation-local locator/measurement estimator. Convergence uses declared parameter, candidate-set, and valid-detector stability, with non-converged and maximum-iteration terminal states. | `REQ-026--028`, `REQ-042`, `PRED-020--022` |
| `BEAM-SCOPE-D006` | V0.1 defines diagnostics and state semantics. Numerical production thresholds remain owner-controlled unless justified by separately approved evidence. | `REQ-029--034`, `PRED-009--012`, `PRED-021` |
| `BEAM-SCOPE-D007` | BEAM may publish only a typed detector-calibration candidate derived from a declared source model and fitted amplitude/response, with uncertainty and lineage. SCI-CAL owns promotion. | `REQ-035--036`, `PRED-023` |
| `BEAM-SCOPE-D008` | Sensitivity is downstream/conditional in v0.1 unless a later owner decision supplies an exact noise, time, atmosphere, calibration, and bandwidth convention. | `REQ-037` |
| `BEAM-SCOPE-D009` | The atomic result includes one fit/result/QC identity per attempted detector plus observation provenance. Maps and optional TOD are parent/diagnostic companions, not alternate validity authorities. | `REQ-001--003`, `REQ-006--007`, `REQ-029`, `REQ-038--044`, `PRED-016--017`, `PRED-022`, `PRED-024` |
| `BEAM-SCOPE-D010` | The author packet admits bounded context from Bryan et al. 2018 and Golec and the TolTEC Collaboration 2024. No analogue-instrument methodology paper is admitted. | `REQ-045`; rationale bounded-reference section |
| `BEAM-SCOPE-D011` | CAL and MAP enter only through short content-bound conditional interface summaries. Their implementation/audit history and unresolved authority are not imported. | `REQ-020`, `REQ-046`, `PRED-018`, `PRED-024` |
| `BEAM-SCOPE-D012` | Citlali owns reduction inference and products; TolAPT owns soft-prior and matched/reference APT production; `toltec_beammap` owns downstream analysis/calibration use. No artifact silently supersedes another repository's authority. | `REQ-040--042`; dependency column throughout |

Locations use `SR` for `src/scientific-rationale.tex` and `EC` for `src/engineering-conformance.tex`. All normative rows originate in the named shared module; both PDFs import that module verbatim.

| ID | Shared source | Scientific rationale location | Engineering location | Owner-approved packet decision | Dependency |
| --- | --- | --- | --- | --- | --- |
| SCI-BEAM-REQ-001 | `common/requirements.tex`, row 001 | SR Sec. 1, 8; App. A row 001 | EC Sec. 1-2; shared row 001 | Scope Sec. 2/4: immutable identity and provenance | Observation/source identity producers |
| SCI-BEAM-REQ-002 | `common/requirements.tex`, row 002 | SR Sec. 1, 8; App. A row 002 | EC Sec. 1-2; shared row 002 | Convention: explicit occurrence-scoped stable detector identity | Metadata producer |
| SCI-BEAM-REQ-003 | `common/requirements.tex`, row 003 | SR Sec. 1, 8; App. A row 003 | EC Sec. 1-2; shared row 003 | Scope Sec. 6.4: one-way requested/effective/realized state | Policy producer |
| SCI-BEAM-REQ-004 | `common/requirements.tex`, row 004 | SR Sec. 1, 3; App. A row 004 | EC shared notation and row 004 | Scope Sec. 6.1: declared AltAz tangent plane and WCS authority | ALIGN/AST, SCI-MAP |
| SCI-BEAM-REQ-005 | `common/requirements.tex`, row 005 | SR opening, Sec. 1-2; App. A row 005 | EC shared row 005 | Scope Sec. 2/3: declared calibrator model and authority | TolProj/TolTECA photometry |
| SCI-BEAM-REQ-006 | `common/requirements.tex`, row 006 | SR opening, Sec. 1/3; App. A row 006 | EC shared row 006 | Scope Sec. 2/3: complete conditioned signal/map bundle | RTC/PTC, VAL, SCI-MAP, SCI-CAL |
| SCI-BEAM-REQ-007 | `common/requirements.tex`, row 007 | SR Sec. 3; App. A row 007 | EC shared definitions and row 007 | Scope Sec. 6.8: invalid excluded before payload | VAL and upstream producers |
| SCI-BEAM-REQ-008 | `common/requirements.tex`, row 008 | SR Sec. 2; Eq. 2; App. A row 008 | EC shared equations and row 008 | Sanitized capability boundary: finite-source elliptical beam plus bounded background | Source model, coordinates |
| SCI-BEAM-REQ-009 | `common/requirements.tex`, row 009 | SR Sec. 2; notation; App. A row 009 | EC shared notation/equations and row 009 | Sanitized frames convention requires one nondegenerate ellipse convention | WCS identity |
| SCI-BEAM-REQ-010 | `common/requirements.tex`, row 010 | SR Sec. 2; shared template equation; App. A row 010 | EC shared equations and row 010 | Scope Sec. 4.1-2: exact estimand and reference-origin normalization | Source model |
| SCI-BEAM-REQ-011 | `common/requirements.tex`, row 011 | SR Sec. 2/7; App. A row 011 | EC shared definitions and row 011 | Scope Sec. 6.7: reference-origin amplitude/peak/flux/response/calibration/sensitivity distinct | SCI-CAL |
| SCI-BEAM-REQ-012 | `common/requirements.tex`, row 012 | SR Sec. 2; App. A row 012 | EC shared definitions and row 012 | Scope asks explicitly bounded background terms | Effective model policy |
| SCI-BEAM-REQ-013 | `common/requirements.tex`, row 013 | SR Sec. 3; App. A row 013 | EC shared row 013 | Scope Sec. 4.3/8: support and edge cases explicit | VAL, SCI-MAP |
| SCI-BEAM-REQ-014 | `common/requirements.tex`, row 014 | SR Sec. 3; Eq. 3; App. A row 014 | EC shared equations and row 014 | Scope Sec. 4.3: likelihood/covariance assumptions explicit | Covariance producer |
| SCI-BEAM-REQ-015 | `common/requirements.tex`, row 015 | SR Sec. 3; Eq. 2; App. A row 015 | EC shared equations and row 015 | Scope Sec. 4.3: residual meaning required | Realized model/support |
| SCI-BEAM-REQ-016 | `common/requirements.tex`, row 016 | SR Sec. 3/5; App. A row 016 | EC shared definitions and row 016 | Scope Sec. 2/4: unavailable conditional claims explicit | Support/source model |
| SCI-BEAM-REQ-017 | `common/requirements.tex`, row 017 | SR Sec. 3/5; Eq. 4; App. A row 017 | EC shared equations and row 017 | Scope Sec. 4.5: covariance or explicit unavailable state | Likelihood/covariance producer |
| SCI-BEAM-REQ-018 | `common/requirements.tex`, row 018 | SR Sec. 5; Eq. 5; App. A row 018 | EC shared equations and row 018 | Scope Sec. 7B: nuisance propagation or unavailable | Upstream nuisance producers |
| SCI-BEAM-REQ-019 | `common/requirements.tex`, row 019 | SR Sec. 3/5; App. A row 019 | EC shared assumptions and row 019 | Sanitized units: coefficient is not automatically precision | Covariance producer |
| SCI-BEAM-REQ-020 | `common/requirements.tex`, row 020 | SR Sec. 1/5; App. A row 020 | EC shared assumptions and row 020 | Ownership: incomplete kernel cannot become complete realized beam | RTC/PTC, SCI-MAP |
| SCI-BEAM-REQ-021 | `common/requirements.tex`, row 021 | SR Sec. 4; App. A row 021 | EC shared assumptions and row 021 | Sanitized soft-prior boundary: compatibility fields explicit | TolAPT |
| SCI-BEAM-REQ-022 | `common/requirements.tex`, row 022 | SR Sec. 4; App. A row 022 | EC shared assumptions and row 022 | Scope Sec. 6.5: soft initialization/gating, not truth or veto | TolAPT |
| SCI-BEAM-REQ-023 | `common/requirements.tex`, row 023 | SR Sec. 4; App. A row 023 | EC shared assumptions and row 023 | Scope Sec. 6.5: blind fallback required | Locator policy |
| SCI-BEAM-REQ-024 | `common/requirements.tex`, row 024 | SR Sec. 4; App. A row 024 | EC shared row 024 | Sanitized prior boundary: strong observed source may defeat wrong prior | Measurement objective |
| SCI-BEAM-REQ-025 | `common/requirements.tex`, row 025 | SR Sec. 4; App. A row 025 | EC shared row 025 | Sanitized prior boundary: influence and route recorded | TolAPT, locator |
| SCI-BEAM-REQ-026 | `common/requirements.tex`, row 026 | SR Sec. 4; App. A row 026 | EC shared row 026 | Scope Sec. 2/6.10: candidate, availability, support, and valid-detector changes remain in the observation-local trace | Iteration policy |
| SCI-BEAM-REQ-027 | `common/requirements.tex`, row 027 | SR Sec. 4; shared convergence equation; App. A row 027 | EC shared equations and row 027 | Scope Sec. 4.10/7C: per-parameter metrics and each convergence-state component explicit | Iteration policy |
| SCI-BEAM-REQ-028 | `common/requirements.tex`, row 028 | SR Sec. 4; App. A row 028 | EC shared row 028 | Scope Sec. 8: candidate/support/availability/valid-detector instability and max iteration remain distinct | Iteration policy |
| SCI-BEAM-REQ-029 | `common/requirements.tex`, row 029 | SR Sec. 1/6; App. A row 029 | EC shared definitions and row 029 | Scope Sec. 2/4.4: every attempted detector and explicit states | Detector roster, VAL |
| SCI-BEAM-REQ-030 | `common/requirements.tex`, row 030 | SR Sec. 2/6; App. A row 030 | EC shared row 030 | Sanitized validity: finite/positive/converged alone insufficient | Realized model |
| SCI-BEAM-REQ-031 | `common/requirements.tex`, row 031 | SR Sec. 6; App. A row 031 | EC shared definitions and row 031 | Scope Sec. 7D: validity conditions vs review cues | QC policy owner |
| SCI-BEAM-REQ-032 | `common/requirements.tex`, row 032 | SR Sec. 4/6; App. A row 032 | EC shared assumptions and row 032 | Reference boundary forbids numerical production thresholds | Scientific/production owner |
| SCI-BEAM-REQ-033 | `common/requirements.tex`, row 033 | SR Sec. 5/6; App. A row 033 | EC shared notation and row 033 | Sanitized statistical labels require declared estimator/calibration | Covariance producer |
| SCI-BEAM-REQ-034 | `common/requirements.tex`, row 034 | SR Sec. 3/6; App. A row 034 | EC shared definitions and row 034 | Scope Sec. 7D: cause states shall not collapse | QC policy |
| SCI-BEAM-REQ-035 | `common/requirements.tex`, row 035 | SR Sec. 6; Eq. 6; App. A row 035 | EC shared equations and row 035 | Scope Sec. 2/5: BEAM may publish typed candidate | Source model, SCI-CAL |
| SCI-BEAM-REQ-036 | `common/requirements.tex`, row 036 | SR Sec. 6; App. A row 036 | EC shared definitions and row 036 | Scope Sec. 5: SCI-CAL owns promotion | SCI-CAL |
| SCI-BEAM-REQ-037 | `common/requirements.tex`, row 037 | SR Sec. 6; App. A row 037 | EC shared assumptions and row 037 | Sanitized capability boundary: no v0.1 sensitivity authority | Downstream sensitivity owner |
| SCI-BEAM-REQ-038 | `common/requirements.tex`, row 038 | SR Sec. 6/8; App. A row 038 | EC shared definitions and row 038 | Scope Sec. 6.9: atomic required bundle | Publication boundary |
| SCI-BEAM-REQ-039 | `common/requirements.tex`, row 039 | SR Sec. 6; App. A row 039 | EC shared definitions and row 039 | Scope Sec. 6.9: optional cannot replace required companion | Publication boundary |
| SCI-BEAM-REQ-040 | `common/requirements.tex`, row 040 | SR Sec. 6; App. A row 040 | EC shared definitions and row 040 | Scope Sec. 4.9/5: parentage and consumer limitations | TolAPT, toltec_beammap |
| SCI-BEAM-REQ-041 | `common/requirements.tex`, row 041 | SR Sec. 6; App. A row 041 | EC shared definitions and row 041 | Scope Sec. 4.12/5: compatibility; no retrospective redefinition | TolAPT, downstream consumers |
| SCI-BEAM-REQ-042 | `common/requirements.tex`, row 042 | SR Sec. 4/6; App. A row 042 | EC shared assumptions and row 042 | Scope Sec. 6.4: later observations cannot inherit state | Observation lifecycle owner |
| SCI-BEAM-REQ-043 | `common/requirements.tex`, row 043 | SR Sec. 1/6; App. A row 043 | EC shared definitions and row 043 | Scope Sec. 6.2 and edge cases: stable identity over row/slot | Metadata producer |
| SCI-BEAM-REQ-044 | `common/requirements.tex`, row 044 | SR Sec. 1/6; App. A row 044 | EC shared definitions and row 044 | Scope Sec. 3/8: validity and authority are semantic | All producers/consumers |
| SCI-BEAM-REQ-045 | `common/requirements.tex`, row 045 | SR Sec. 9; App. A row 045 | EC Sec. 1-2 and shared row 045 | Manifest requires five evidence/authority layers separated | Validation and production owners |
| SCI-BEAM-REQ-046 | `common/requirements.tex`, row 046 | SR Sec. 1/5/6; App. A row 046 | EC shared assumptions and row 046 | Scope Sec. 1: dependent claims remain unavailable/conditional | All upstream authorities |
| SCI-BEAM-PRED-001 | `common/edge_cases.tex`, row 001 | SR App. B row 001 | EC shared prediction row 001 | Scope Sec. 8: noiseless circular recovery under declared reference-origin normalization | Model/operator evidence |
| SCI-BEAM-PRED-002 | `common/edge_cases.tex`, row 002 | SR App. B row 002 | EC shared prediction row 002 | Scope Sec. 8: noiseless elliptical recovery of reference centroid and amplitude | Model/operator evidence |
| SCI-BEAM-PRED-003 | `common/edge_cases.tex`, row 003 | SR App. B row 003 | EC shared prediction row 003 | Scope Sec. 8: axis swap | Ellipse convention |
| SCI-BEAM-PRED-004 | `common/edge_cases.tex`, row 004 | SR App. B row 004 | EC shared prediction row 004 | Scope Sec. 8: orientation periodicity | WCS/ellipse convention |
| SCI-BEAM-PRED-005 | `common/edge_cases.tex`, row 005 | SR App. B row 005 | EC shared prediction row 005 | Scope Sec. 8: constant background | Support/background policy |
| SCI-BEAM-PRED-006 | `common/edge_cases.tex`, row 006 | SR App. B row 006 | EC shared prediction row 006 | Scope Sec. 8: admissible gradient/background | Support/background policy |
| SCI-BEAM-PRED-007 | `common/edge_cases.tex`, row 007 | SR App. B row 007 | EC shared prediction row 007 | Scope Sec. 8: finite size tends to zero with reference origin preserved | Source model |
| SCI-BEAM-PRED-008 | `common/edge_cases.tex`, row 008 | SR App. B row 008 | EC shared prediction row 008 | Scope Sec. 8: finite size approaches beam scale | Source nuisance covariance |
| SCI-BEAM-PRED-009 | `common/edge_cases.tex`, row 009 | SR App. B row 009 | EC shared prediction row 009 | Scope Sec. 8: cropped/edge source | Support/WCS |
| SCI-BEAM-PRED-010 | `common/edge_cases.tex`, row 010 | SR App. B row 010 | EC shared prediction row 010 | Scope Sec. 8: masked core and reference-origin-amplitude identifiability | VAL/support |
| SCI-BEAM-PRED-011 | `common/edge_cases.tex`, row 011 | SR App. B row 011 | EC shared prediction row 011 | Scope Sec. 8: disconnected support | VAL/support |
| SCI-BEAM-PRED-012 | `common/edge_cases.tex`, row 012 | SR App. B row 012 | EC shared prediction row 012 | Scope Sec. 8: zero support | VAL/support |
| SCI-BEAM-PRED-013 | `common/edge_cases.tex`, row 013 | SR App. B row 013 | EC shared prediction row 013 | Soft-prior boundary: prior is initialization/gating only | TolAPT, locator |
| SCI-BEAM-PRED-014 | `common/edge_cases.tex`, row 014 | SR App. B row 014 | EC shared prediction row 014 | Scope Sec. 8 and Sec. 6.5: blind fallback | TolAPT, locator |
| SCI-BEAM-PRED-015 | `common/edge_cases.tex`, row 015 | SR App. B row 015 | EC shared prediction row 015 | Soft-prior boundary: strong observed source may defeat wrong prior | Locator/measurement objective |
| SCI-BEAM-PRED-016 | `common/edge_cases.tex`, row 016 | SR App. B row 016 | EC shared prediction row 016 | Scope Sec. 8: duplicate/permuted rows | Stable binding producer |
| SCI-BEAM-PRED-017 | `common/edge_cases.tex`, row 017 | SR App. B row 017 | EC shared prediction row 017 | Scope Sec. 8: repeated slot labels | Stable binding producer |
| SCI-BEAM-PRED-018 | `common/edge_cases.tex`, row 018 | SR App. B row 018 | EC shared prediction row 018 | Scope Sec. 8: incomplete response/kernel | RTC/PTC, SCI-MAP |
| SCI-BEAM-PRED-019 | `common/edge_cases.tex`, row 019 | SR App. B row 019 | EC shared prediction row 019 | Scope Sec. 8: correlated versus diagonal covariance | Covariance producer |
| SCI-BEAM-PRED-020 | `common/edge_cases.tex`, row 020 | SR App. B row 020 | EC shared prediction row 020 | Scope Sec. 8: alternating selected-candidate identity | Iteration policy |
| SCI-BEAM-PRED-021 | `common/edge_cases.tex`, row 021 | SR App. B row 021 | EC shared prediction row 021 | Scope Sec. 8: modulo-$\pi$/unavailable-angle handling, component instability, and max iteration | Iteration policy |
| SCI-BEAM-PRED-022 | `common/edge_cases.tex`, row 022 | SR App. B row 022 | EC shared prediction row 022 | Scope Sec. 8 and Sec. 6.4: order independence | Observation lifecycle |
| SCI-BEAM-PRED-023 | `common/edge_cases.tex`, row 023 | SR App. B row 023 | EC shared prediction row 023 | Scope Sec. 8: candidate uncertainty | Source model, SCI-CAL |
| SCI-BEAM-PRED-024 | `common/edge_cases.tex`, row 024 | SR App. B row 024 | EC shared prediction row 024 | Scope Sec. 8: equal payload/different authority | All producers/consumers |

Count: 46 requirements and 24 predictions; every identifier appears exactly once as a crosswalk row.
