# SCI-BEAM v0.1 — Scientific-Owner Decision Ledger

Status: all scope decisions open; Stage B blocked pending owner review

Owner: Grant Wilson

Opened: `2026-08-16`

This ledger contains the choices that would materially change the scientific
author's task. The author may analyze alternatives and identify consequences,
but may not silently choose an owner policy.

| ID | Owner decision required | Proposed/default disposition for review | Status |
| --- | --- | --- | --- |
| `BEAM-SCOPE-Q001` | What source/beam model family is in v0.1? | Begin from an elliptical 2-D beam convolved with the declared calibrator brightness model plus explicitly bounded background terms; require typed model identity and allow unavailable claims outside that family. | OPEN |
| `BEAM-SCOPE-Q002` | How does finite calibrator angular extent enter? | The source model and beam model remain separate and are forward-convolved; point-source treatment is a declared limiting case, not a silent assumption. | OPEN |
| `BEAM-SCOPE-Q003` | What objective and uncertainty authority are required? | Author derives a general likelihood/covariance statement and an explicitly conditional diagonal approximation; reported covariance must name omitted correlation and nuisance terms. | OPEN |
| `BEAM-SCOPE-Q004` | What roles may a soft prior play? | Initialization and bounded candidate gating only in v0.1; no exact UID truth or unconditional veto; prior influence recorded and blind fallback required. | OPEN |
| `BEAM-SCOPE-Q005` | What constitutes BEAM iteration and convergence? | Observation-local locator/measurement estimator only; convergence requires a declared conjunction of parameter, candidate-set, and valid-detector stability, with maximum-iteration and non-converged terminal states. | OPEN |
| `BEAM-SCOPE-Q006` | Which QC thresholds belong in the scientific contract? | Define dimensionless/physical diagnostics and state semantics in v0.1; keep numerical production thresholds owner-controlled unless justified by approved evidence. | OPEN |
| `BEAM-SCOPE-Q007` | May BEAM publish a detector calibration factor? | Permit only a typed candidate derived from declared source-model flux and fitted amplitude/response, with full uncertainty and lineage; SCI-CAL separately owns promotion. | OPEN |
| `BEAM-SCOPE-Q008` | Is detector sensitivity a BEAM-owned result? | Treat sensitivity as downstream/conditional unless the owner supplies a precise noise, time, atmosphere, calibration, and bandwidth convention for v0.1. | OPEN |
| `BEAM-SCOPE-Q009` | What is the atomic product bundle? | One required fit/result/QC identity per attempted detector plus observation-level provenance; maps and optional TOD are parent/diagnostic companions, not alternate validity authorities. | OPEN |
| `BEAM-SCOPE-Q010` | Which external primary references enter the author packet? | Approve Bryan et al. 2018 and Golec and the TolTEC Collaboration 2024 for TolTEC context, plus at most one owner-selected detector-beam/calibration methodology paper. | OPEN |
| `BEAM-SCOPE-Q011` | May CAL and MAP interface extracts enter the packet? | Allow only short content-bound extracts that state conditional producer boundaries and open authority; do not supply their full implementation or audit history. | OPEN |
| `BEAM-SCOPE-Q012` | Is the proposed repository boundary correct? | Citlali owns reduction inference/products; TolAPT owns soft-prior and matched/reference APT production; `toltec_beammap` owns downstream analysis/calibration use. No repository silently supersedes another's artifact. | OPEN |

## Gate Rule

The owner may approve the proposed dispositions, replace them, or defer a
question with an explicit claim limitation. Stage B may start only when Q001,
Q002, Q004, Q007, Q009, Q010, Q011, and Q012 are resolved sufficiently to
define one author task. Q003, Q005, Q006, and Q008 may remain open only if the
Scope Brief explicitly assigns the author to derive alternatives without
selecting production policy.
