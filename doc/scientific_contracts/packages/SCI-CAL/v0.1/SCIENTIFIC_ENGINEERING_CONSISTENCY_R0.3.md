# SCI-CAL v0.1 Rationale r0.3 / Engineering r0.2 Consistency Report

Status: superseded by the 2026-08-20 fresh implementation-blind review and the
r0.4/r0.3 bounded repair; scientific authority not frozen

Original review: `2026-08-16`

Corrected review: `2026-08-20`

The original r0.3 report established equation and inventory consistency but
treated stable-ID coverage as sufficient semantic closure. A fresh read-only
comparison found that the engineering view did not carry the rationale's full
Q01--Q09 authority consequences, producer--transformer--consumer boundary,
source-factor generating record, transfer limitations, broadband compatibility,
or pipeline-order limitation. Engineering conformance r0.2 corrects those gaps
without changing the science rationale or resolving an owner decision.

| Consistency axis | Corrected result |
| --- | --- |
| Open authority | Every Q01--Q09 state, owner, closure evidence, and blocked claim/output is now an explicit engineering record and disposition gate. Candidate metadata and realized behavior cannot close a decision. |
| Ownership | Beammap/source-APT production owns source calibration and factor meaning; TolProj owns association and approved child transformation; TolTECA delivery preserves the selected artifact; SCI-CAL consumes the selected child factor and target atmosphere once; MAP/FLT owns realized response. |
| Factor derivation | The selected child factor resolves a producer-owned generating record containing calibrator model/epoch, source atmosphere, estimator, beam/template normalization, pointing, spectral convention, direction, and uncertainty/covariance status. Q03 remains open. |
| Transfer | The child record carries the approved transformation, validity domain, and retained systematics. Transfer outside an approved domain remains unavailable under Q04. |
| Broadband meaning | A photometric-convention state connects source factor, target atmosphere, and reported output. Equal passband identity alone is not compatibility evidence; Q05 limitations remain explicit. |
| Ordering | The local CAL affine boundary and global pipeline-order state are distinct. Local operator equality is not promoted to a unique end-to-end response while Q02 is open. |
| Atmosphere | Orientation-neutral interpolation, reciprocal order, support, and invariants remain unchanged. Binding the missing operator record resolves Q06 only. |
| Units and output plane | The intended result remains top-of-atmosphere, point-source-equivalent, beam-peak-normalized amplitude. Literal peak meaning still depends on realized response; complete monochromatic/cross-array meaning remains unavailable under Q05. |
| Uncertainty and claims | Conditional, nuisance, total, representation-fidelity, observational-performance, and production claims remain separate. Q08 and Q09 consequences are unchanged. |
| Version axes | Contract version remains v0.1; science rationale remains r0.3; the repaired engineering document is r0.2. Stable canonical filenames continue to identify the active views. |

## Mechanical invariants

- 11 sequential assumption IDs remain `SCI-CAL-ASM-001`--`011`.
- 50 sequential requirement IDs remain `SCI-CAL-REQ-001`--`050`.
- 30 sequential edge IDs remain `SCI-CAL-EDGE-001`--`030`.
- No open decision is marked resolved, and no numerical science or
  implementation behavior is introduced.

## Historical remaining gate

At this report's stage, the corrected pair still required the program's fresh
implementation-blind consistency review and explicit scientific-owner freeze
disposition. The r0.4/r0.3 successor subsequently passed that review; owner
freeze remains pending. Neither repair is implementation conformity,
atmosphere fidelity, observational validation, or production readiness.
