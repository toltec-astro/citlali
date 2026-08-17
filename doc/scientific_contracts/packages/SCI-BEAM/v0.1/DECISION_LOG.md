# SCI-BEAM v0.1 — Decision Log

Status: Stage A process and scientific scope decisions approved

Scientific owner: Grant Wilson

Date: `2026-08-16`

## Approved Process Decisions

| Decision | Approved substance |
| --- | --- |
| `BEAM-PROCESS-D001` | SCI-BEAM is the next package after the CAL/MAP pilot review. |
| `BEAM-PROCESS-D002` | The permanent CAL/MAP workflow and anti-repetition procedure govern SCI-BEAM and later packages. |
| `BEAM-PROCESS-D003` | SCI-BEAM Stage A may proceed, but it remains separate from active ALIGN/AST work and may not make physical timing or absolute-placement claims. |
| `BEAM-PROCESS-D004` | Stage B requires a fresh implementation-blind scientific author after, not before, scientific-owner approval of the sanitized Scope Brief and exact packet. |

These process decisions do not approve the proposed scientific model or author
references.

## Approved Scientific Scope Decisions

The scientific owner approved the following dispositions on `2026-08-16`:

| Decision | Approved substance |
| --- | --- |
| `BEAM-SCOPE-D001` | V0.1 begins from an elliptical two-dimensional beam convolved with the declared calibrator brightness model and explicitly bounded background terms. Model identity is typed; claims outside the family are unavailable. |
| `BEAM-SCOPE-D002` | Calibrator brightness and beam remain separate forward-model components. Point-source treatment is an explicit limiting case, never a silent assumption. |
| `BEAM-SCOPE-D003` | The author derives a general likelihood/covariance statement and may define an explicitly conditional diagonal approximation. Every reported covariance names omitted correlation and nuisance terms. |
| `BEAM-SCOPE-D004` | Soft priors may initialize and bound candidate gating only. They are not exact UID or position truth, cannot impose unconditional veto, must record influence, and require blind fallback. |
| `BEAM-SCOPE-D005` | Internal iteration is an observation-local locator/measurement estimator. Convergence uses declared parameter, candidate-set, and valid-detector stability, with non-converged and maximum-iteration terminal states. |
| `BEAM-SCOPE-D006` | V0.1 defines diagnostics and state semantics. Numerical production thresholds remain owner-controlled unless justified by separately approved evidence. |
| `BEAM-SCOPE-D007` | BEAM may publish only a typed detector-calibration candidate derived from a declared source model and fitted amplitude/response, with uncertainty and lineage. SCI-CAL owns promotion. |
| `BEAM-SCOPE-D008` | Sensitivity is downstream/conditional in v0.1 unless a later owner decision supplies an exact noise, time, atmosphere, calibration, and bandwidth convention. |
| `BEAM-SCOPE-D009` | The atomic result includes one fit/result/QC identity per attempted detector plus observation provenance. Maps and optional TOD are parent/diagnostic companions, not alternate validity authorities. |
| `BEAM-SCOPE-D010` | The author packet admits bounded context from Bryan et al. 2018 and Golec and the TolTEC Collaboration 2024. No analogue-instrument methodology paper is admitted. |
| `BEAM-SCOPE-D011` | CAL and MAP enter only through short content-bound conditional interface summaries. Their implementation/audit history and unresolved authority are not imported. |
| `BEAM-SCOPE-D012` | Citlali owns reduction inference and products; TolAPT owns soft-prior and matched/reference APT production; `toltec_beammap` owns downstream analysis/calibration use. No artifact silently supersedes another repository's authority. |

The exact packet and firewall are content-bound in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

Implementation conformity, validation, and production promotion are outside
this log.
