# SCI-BEAM v0.1 — Scientific-Owner Decision Ledger

Status: r0.2 open scientific decisions only

Scientific owner: Grant Wilson

Updated: `2026-08-17`

The owner decisions settled in the r0.2 directive are recorded in
[`DECISION_LOG.md`](DECISION_LOG.md) as `BEAM-SCOPE-D013--D030` and are not
reopened here. This ledger contains only choices that remain genuinely
unresolved. No open item is replaced by an implementation default.

| ID | Owning authority | State | Evidence or decision required | Exact blocked claim or output | Resolution authority | Resolution date | Affected documents |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `SCI-BEAM-ODQ-101` | SCI-BEAM sensitivity estimator | open | Exact scan-level noise statistic, baseline treatment, frequency range or bandwidth, robust/spectral estimator, correlated-noise treatment, and perturbation validation | Numerically authorized `sens` estimator | Grant Wilson or named successor sensitivity authority | — | Rationale; formal contract; validation plan |
| `SCI-BEAM-ODQ-102` | SCI-BEAM support policy | open | Source-exclusion radius, trajectory-distance convention, partial-scan handling, minimum off-source scans, and admission policy | Production off-source scan membership and numerical `sens` | Grant Wilson | — | Formal contract; validation plan |
| `SCI-BEAM-ODQ-103` | SCI-BEAM sensitivity estimator | open | Exposure/time normalization, bandwidth convention, atmosphere timing, and repeatability/uncertainty statistic | Exact `mJy beam_nom^-1 sqrt(s)` normalization and uncertainty | Grant Wilson or named successor sensitivity authority | — | Rationale; formal contract; validation plan |
| `SCI-BEAM-ODQ-104` | SCI-BEAM model-adequacy policy | open | Quantitative residual, support-dependence, asymmetry, excess-response, multi-lobe, wing-completeness, and amplitude-bias diagnostics plus eventual policy thresholds | Production adequacy and `flxscale` robustness disposition | Grant Wilson | — | Rationale; formal contract; validation plan |
| `SCI-BEAM-ODQ-105` | Downstream science-impact authority, presently unnamed | open | Kernel-impact study establishing separate accuracy needs for health, amplitude, `flxscale`, `sens`, Gaussian kernels, empirical kernels, and wing claims | Required PSF accuracy and downstream science qualification | Grant Wilson and named downstream science owner | — | Validation plan; future kernel contract |
| `SCI-BEAM-ODQ-106` | SCI-BEAM observation design | open | Hidden-response study over map depth, radial extent, noise, backgrounds, and stacking | Quantitative wing-response completeness and complete-PSF claim | Grant Wilson | — | Rationale; validation plan; observation policy |
| `SCI-BEAM-ODQ-107` | ALIGN/AST geometry authority | open | Conventional-pivot perturbation study over elevation, support, detector angle, and pointing/science separation | Acceptable residual common and differential geometry error | Grant Wilson or named ALIGN/AST owner | — | Rationale; AST successor contract; validation plan |
| `SCI-BEAM-ODQ-108` | Future focal-plane/boresight registration authority, unnamed | open | Physical boresight and rotation-pivot measurement model, evidence, uncertainty, reference epoch, and transfer domain | Absolute array position, physical pivot, and telescope pointing interpretation | Grant Wilson or named future owner | — | AST successor; pointing contract; BEAM interface |
| `SCI-BEAM-ODQ-109` | Downstream kernel authority, presently unnamed | open | Science-impact criteria for Gaussian, regularized empirical, or unavailable kernel; normalization and response requirements | Any downstream kernel-use qualification | Grant Wilson and named kernel owner | — | Rationale; validation plan; future kernel contract |

Implementation conformance, current file/column names, historical values, and
production behavior are audit questions, not scientific-owner decisions in
this ledger.
