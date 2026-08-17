# SCI-BEAM v0.1 — Scientific-Owner Decision Ledger

Status: scope decisions approved; Stage B author may proceed

Owner: Grant Wilson

Opened: `2026-08-16`

This ledger contains the choices that would materially change the scientific
author's task. The author may analyze alternatives and identify consequences,
but may not silently choose an owner policy.

| ID | Decision | Approved disposition | Status |
| --- | --- | --- | --- |
| `BEAM-SCOPE-Q001` | What source/beam model family is in v0.1? | Begin from an elliptical 2-D beam convolved with the declared calibrator brightness model plus explicitly bounded background terms; require typed model identity and allow unavailable claims outside that family. | APPROVED as `BEAM-SCOPE-D001` |
| `BEAM-SCOPE-Q002` | How does finite calibrator angular extent enter? | The source model and beam model remain separate and are forward-convolved; point-source treatment is a declared limiting case, not a silent assumption. | APPROVED as `BEAM-SCOPE-D002` |
| `BEAM-SCOPE-Q003` | What objective and uncertainty authority are required? | Author derives a general likelihood/covariance statement and an explicitly conditional diagonal approximation; reported covariance must name omitted correlation and nuisance terms. | APPROVED as `BEAM-SCOPE-D003` |
| `BEAM-SCOPE-Q004` | What roles may a soft prior play? | Initialization and bounded candidate gating only in v0.1; no exact UID truth or unconditional veto; prior influence recorded and blind fallback required. | APPROVED as `BEAM-SCOPE-D004` |
| `BEAM-SCOPE-Q005` | What constitutes BEAM iteration and convergence? | Observation-local locator/measurement estimator only; convergence requires a declared conjunction of parameter, candidate-set, and valid-detector stability, with maximum-iteration and non-converged terminal states. | APPROVED as `BEAM-SCOPE-D005` |
| `BEAM-SCOPE-Q006` | Which QC thresholds belong in the scientific contract? | Define dimensionless/physical diagnostics and state semantics in v0.1; keep numerical production thresholds owner-controlled unless justified by approved evidence. | APPROVED as `BEAM-SCOPE-D006` |
| `BEAM-SCOPE-Q007` | May BEAM publish a detector calibration factor? | Permit only a typed candidate derived from declared source-model flux and fitted amplitude/response, with full uncertainty and lineage; SCI-CAL separately owns promotion. | APPROVED as `BEAM-SCOPE-D007` |
| `BEAM-SCOPE-Q008` | Is detector sensitivity a BEAM-owned result? | Treat sensitivity as downstream/conditional unless the owner supplies a precise noise, time, atmosphere, calibration, and bandwidth convention for v0.1. | APPROVED as `BEAM-SCOPE-D008` |
| `BEAM-SCOPE-Q009` | What is the atomic product bundle? | One required fit/result/QC identity per attempted detector plus observation-level provenance; maps and optional TOD are parent/diagnostic companions, not alternate validity authorities. | APPROVED as `BEAM-SCOPE-D009` |
| `BEAM-SCOPE-Q010` | Which external primary references enter the author packet? | Admit Bryan et al. 2018 and Golec and the TolTEC Collaboration 2024 for bounded TolTEC context. Admit no analogue-instrument methodology paper. | APPROVED as `BEAM-SCOPE-D010` |
| `BEAM-SCOPE-Q011` | May CAL and MAP interface extracts enter the packet? | Allow only short content-bound extracts that state conditional producer boundaries and open authority; do not supply their full implementation or audit history. | APPROVED as `BEAM-SCOPE-D011` |
| `BEAM-SCOPE-Q012` | Is the proposed repository boundary correct? | Citlali owns reduction inference/products; TolAPT owns soft-prior and matched/reference APT production; `toltec_beammap` owns downstream analysis/calibration use. No repository silently supersedes another's artifact. | APPROVED as `BEAM-SCOPE-D012` |

## Gate Disposition

Grant Wilson approved all twelve dispositions on `2026-08-16`. Stage B
authorship is authorized subject to the exact packet and firewall in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md). Contract acceptance,
implementation conformity, validation, and production promotion remain
separate future gates.
