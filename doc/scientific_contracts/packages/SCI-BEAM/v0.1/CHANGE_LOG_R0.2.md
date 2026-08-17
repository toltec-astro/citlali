# SCI-BEAM v0.1 — Change Log r0.1 to r0.2

Status: complete substantive draft change map

Revision date: `2026-08-17`

The scientific owner directed a substantive correction while v0.1 remained
unfrozen. Requirement and prediction IDs are retained as draft-lineage keys,
but their r0.2 meaning supersedes r0.1; no r0.1 clause is implementation or
scientific authority. The original author packet and author record remain
unchanged historical provenance.

## Document-form changes

- Replaced the combined “Scientific Rationale and Contract” with a separate
  science-team rationale and normative Formal Scientific/Engineering Contract.
- Removed the complete requirements/predictions and exact convergence
  machinery from the rationale; retained them in the formal shared core.
- Added scientist-facing explanations of the effective core, nominal beam,
  geometry, `flxscale`, `sens`, APT, adequacy, and validation.
- Added three explanatory figures, independent-state and uncertainty tables,
  an r0.2 owner ledger, this change map, cross-package follow-up list, and
  consistency report.

## Requirement disposition

| ID | r0.1 subject | r0.2 disposition |
| --- | --- | --- |
| REQ-001 | result identity | Retained and expanded to parent map/document state. |
| REQ-002 | detector binding | Replaced by canonical raw-signal definition; stable binding moves into REQ-001/003/044. |
| REQ-003 | requested/effective/realized state | Retained through formal logical-state records and REQ-027. |
| REQ-004 | AltAz offsets | Replaced by metrically orthonormal WCS/Jacobian requirement. |
| REQ-005 | source identity | Retained and expanded in REQ-006/007. |
| REQ-006 | generic signal/map input | Replaced by standardized detector Beammap in raw `Delta f/f` and complete response lineage. |
| REQ-007 | upstream validity | Retained as REQ-008. |
| REQ-008 | generic finite-source fit | Replaced by canonical map-domain effective-core model in REQ-009. |
| REQ-009 | ellipse convention | Retained and strengthened in REQ-010/011. |
| REQ-010 | reference normalization | Retained as REQ-012 and separated from nominal-beam calibration. |
| REQ-011 | amplitude meaning | Retained as observed-plane `Delta f/f` in REQ-013. |
| REQ-012 | background | Retained as REQ-014. |
| REQ-013 | support | Retained and expanded as REQ-015. |
| REQ-014 | objective/covariance | Retained as REQ-016. |
| REQ-015 | residual | Retained and made a required dense product in REQ-017/023. |
| REQ-016 | identifiability | Retained and expanded as REQ-018. |
| REQ-017 | fit covariance | Replaced by full model-Jacobian and joint-covariance requirements REQ-019/020. |
| REQ-018 | nuisance propagation | Retained and strengthened in REQ-020. |
| REQ-019 | conditional covariance | Retained as REQ-021. |
| REQ-020 | response interpretation | Replaced by explicit effective-core/complete-PSF states REQ-022/023. |
| REQ-021 | prior metadata | Retained in REQ-024/026. |
| REQ-022 | soft prior role | Retained in REQ-024. |
| REQ-023 | blind fallback | Retained in REQ-025. |
| REQ-024 | common-objective candidate | Retained in REQ-025. |
| REQ-025 | prior influence | Retained in REQ-026. |
| REQ-026 | estimator trace | Retained as REQ-027. |
| REQ-027 | convergence conjunction | Retained as REQ-028 and formal equation. |
| REQ-028 | non-converged causes | Retained in independent state and exact convergence machinery. |
| REQ-029 | one terminal detector state | Replaced by independent per-quantity state in REQ-029. |
| REQ-030 | parameter admissibility | Retained and tensor-strengthened in REQ-030. |
| REQ-031 | QC separation | Expanded into use-specific state in REQ-031. |
| REQ-032 | policy thresholds | Retained and expanded in REQ-032. |
| REQ-033 | S/N meaning | Absorbed into adequacy/sensitivity policy and evidence-layer separation. |
| REQ-034 | cause separation | Expanded into quantitative model-adequacy requirement REQ-033. |
| REQ-035 | calibration candidate | Replaced by BEAM-owned fixed-nominal-beam `flxscale` in REQ-039. |
| REQ-036 | SCI-CAL promotion | Superseded: promotion boundary is inside BEAM; SCI-CAL is downstream consumer. |
| REQ-037 | sensitivity unavailable | Superseded: BEAM owns NEFD-like `sens` in REQ-041/042. |
| REQ-038 | atomic publication | Retained through mandatory APT/bundle in REQ-044. |
| REQ-039 | required companion failure | Retained through independent state plus atomic mandatory product semantics. |
| REQ-040 | optional views | Replaced by required dense companions and optional stacks in REQ-023/045. |
| REQ-041 | compatibility | Retained in shared definition and same-APT rule REQ-038. |
| REQ-042 | observation-local initialization | Retained in logical state and estimator trace. |
| REQ-043 | row/slot binding | Retained under stable APT/detector identity in REQ-001/003/044. |
| REQ-044 | authority-distinct payloads | Retained through independent state and compatibility. |
| REQ-045 | evidence layers | Retained and expanded as REQ-046. |
| REQ-046 | unavailable upstream claims | Retained through independent causal state in REQ-029/046. |

## Prediction disposition

| r0.1 predictions | r0.2 disposition |
| --- | --- |
| PRED-001--008 | Recast around raw-map input, full tensor, WCS metric, finite-source limit, and fixed-nominal-beam/no-extra-`H(0)` calibration. |
| PRED-009--012 | Recast around source atmosphere, independent `flxscale`/`sens` availability, finite support, and hidden wings. |
| PRED-013--015 | Recast around background/wing degeneracy, Jacobian perturbation, and rotation/circular covariance. |
| PRED-016--019 | Recast around broadening PSD, detector-specific derotation, common-origin gauge, conventional pivot, and same-APT transfer. |
| PRED-020--021 | Retain soft-prior and convergence principles with the corrected estimator. |
| PRED-022 | Replaced by the falsifiable independence of deprecated `responsivity`. |
| PRED-023--024 | Recast around independent quantity states, mandatory APT, hidden response, and science-impact qualification. |

## Scientific corrections

- V0.1 now fits standardized per-detector maps, not a generic map/timestream
  union.
- `x`/`xs` are explicitly raw `Delta f/f`; calibrated maps are excluded.
- The fitted tensor is explicitly the effective Beammap PSF core.
- Calibration uses a fixed nominal-beam TOA reference-origin amplitude; no
  extra finite-source factor enters `flxscale`.
- BEAM owns source atmosphere, accepted source-APT `flxscale`, and scan-domain
  NEFD-like `sens`.
- Complete WCS metric, model Jacobian, joint covariance, broadening tensor,
  detector-specific derotation, origin gauge, pivot limitation, and same-APT
  pointing transfer are normative.
- `responsivity` is deprecated, the APT is mandatory, and validity is
  independent per quantity/use.
