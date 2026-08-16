# SCI-CAL v0.2 Unresolved Scientific Owner Decisions

Status: owner review required; none of these questions is silently resolved
by the scientist-facing rewrite

Date: `2026-08-16`

| ID | Required decision or evidence | Consequence while open |
| --- | --- | --- |
| `SCI-CAL-OWNER-Q01` | Bind the physical definition, unit, sign, preprocessing, normalization, total-intensity scope, and linear loading/tune/signal domain of ordinary `xs`. | The input observable and the physical units of `flxscale` remain incomplete. |
| `SCI-CAL-OWNER-Q02` | State the upstream baseline/affine convention and governed ordering relative to common-mode removal, PCA, temporal filtering, weighting, and mapmaking. | Noncommuting operations and downstream transfer functions cannot be interpreted uniquely. |
| `SCI-CAL-OWNER-Q03` | Bind the generating contract for `flxscale`: calibrator model/epoch, amplitude estimator, source atmosphere, pointing, beam/template normalization, spectral convention, factor direction, and uncertainty. | The selected execution factor is defined, but its full scientific derivation is not. |
| `SCI-CAL-OWNER-Q04` | Define calibrator-to-target transfer across time, tune state, loading, focus, pointing, and detector state; decide whether transfer is embodied in the child `flxscale`, removed by `xs`, negligible over a declared domain, or a retained systematic. | Universal transferability of one Beammap calibration is not established. |
| `SCI-CAL-OWNER-Q05` | Define per-array reference frequency/wavelength, reference spectrum, calibrator treatment, color-correction scope, passband weighting, detector/network variation, and the relationship to atmosphere weighting. | The complete monochromatic meaning of reported mJy is unresolved. |
| `SCI-CAL-OWNER-Q06` | Provide the exact atmosphere operator: arrays, nodes, ordinates, orientation, units, interpolation and seam rules, support, passband/spectral/photon-or-energy weighting, generating model, and content identity. Also classify whether it is absent project-wide or merely omitted from the author packet. | Numerical atmosphere evaluation and calibrated numerical output remain unauthorized. |
| `SCI-CAL-OWNER-Q07` | State the scientific or operational rationale for the `tau225` boundaries at 0.15 and 0.25, who declares coherent segments, whether splitting is allowed, and what evidence permits changing the limits. | The policy remains binding but scientifically unexplained. |
| `SCI-CAL-OWNER-Q08` | Identify which conditional-noise and calibration-systematic terms are numerically available, propagated, and present in science products. | Total calibrated uncertainty and significance claims remain unavailable. |
| `SCI-CAL-OWNER-Q09` | Approve exact evidence thresholds, population, support, covariance, and decision rule for atmosphere fidelity, relative repeatability, and absolute recovery. | Structural completeness cannot become an achieved science-qualified claim. |

The former `SCI-CAL-OWNER-Q001` is subsumed by `SCI-CAL-OWNER-Q06` without
weakening any of its required fields.
