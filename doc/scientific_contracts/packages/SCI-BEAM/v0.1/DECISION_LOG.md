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

## Approved r0.2 Scientific-Owner Decisions

The directive dated `2026-08-17` is recorded in
[`SCIENTIFIC_OWNER_DIRECTIVE_R0.2.md`](SCIENTIFIC_OWNER_DIRECTIVE_R0.2.md).
It supersedes conflicting `r0.1` draft language without changing contract
version `v0.1` or the original author-packet hashes.

| Decision | Approved substance |
| --- | --- |
| `BEAM-SCOPE-D013` | SCI-BEAM owns complete Beammap analysis and the complete scientific Beammap APT. |
| `BEAM-SCOPE-D014` | `x_d=Delta f_d/f_d` and `xs` name the raw uncalibrated detector observable; the standardized per-detector Beammap remains in `Delta f/f`. |
| `BEAM-SCOPE-D015` | V0.1 fits standardized per-detector Beammap maps; timestream fitting is future validation or higher-fidelity work. |
| `BEAM-SCOPE-D016` | Calibration uses a fixed nominal-beam top-of-atmosphere reference-origin source amplitude; finite-source normalization is embodied and no extra `H(0)` enters the ratio. |
| `BEAM-SCOPE-D017` | The fitted tensor is the observation-local effective PSF core; intrinsic and complete-PSF interpretations require stronger evidence. |
| `BEAM-SCOPE-D018` | Fit the complete positive-definite 2-D shape tensor and retain conditional FWHM, orientation, and Gaussian-area meanings. |
| `BEAM-SCOPE-D019` | Evaluate the fit in a metrically orthonormal angular tangent plane established by the WCS transformation/Jacobian. |
| `BEAM-SCOPE-D020` | Resolve the full model Jacobian and joint covariance with material cross terms and invariant circular-limit handling. |
| `BEAM-SCOPE-D021` | Derive `Sigma_broad=Sigma_eff-Sigma_nom` only for compatible conventions and require positive-semidefinite support for literal Gaussian broadening. |
| `BEAM-SCOPE-D022` | Distinguish raw fitted and horizon-derotated detector coordinates; derive per-detector effective rotation from realized fit support and transform the PSF tensor consistently. |
| `BEAM-SCOPE-D023` | Treat common APT origin as a gauge, physical pivot as unestablished, and require exact same-APT/same-AST pointing transfer unless equivalence is separately proved. |
| `BEAM-SCOPE-D024` | BEAM owns source-atmosphere treatment and publishes TOA nominal-beam `flxscale`; SCI-CAL later applies target atmosphere once. |
| `BEAM-SCOPE-D025` | BEAM publishes NEFD-like TOA nominal-beam `sens` from robust off-source scan statistics; exact estimator policy remains open. |
| `BEAM-SCOPE-D026` | `responsivity` is deprecated compatibility metadata and is not canonical scientific content. |
| `BEAM-SCOPE-D027` | Availability and validity are independent per quantity; every attempted detector and unavailable field remains represented with causes. |
| `BEAM-SCOPE-D028` | Empirical maps and adequacy/wing diagnostics are required; array/network stacked PSFs are optional diagnostics. |
| `BEAM-SCOPE-D029` | Science-impact, recovery, hidden-response, kernel-impact, geometry, and rotation studies determine accuracy and downstream qualification; no validation result is presumed. |
| `BEAM-SCOPE-D030` | The Beammap APT is mandatory and scientifically defined independently of current implementation or storage schema. |

## Approved r0.3 Final-Review Decisions

The scientific owner approved the final bounded review on `2026-08-17` and
directed the authority to freeze after these corrections and consistency QA.

| Decision | Approved substance |
| --- | --- |
| `BEAM-SCOPE-D031` | An available NEFD-like `sens` is finite and strictly positive: `sens=abs(flxscale) n_off`; the physical sign remains on `flxscale`. |
| `BEAM-SCOPE-D032` | Map-fit Jacobian/covariance and derived calibration/sensitivity propagation are distinct stages with material dependence retained between them. |
| `BEAM-SCOPE-D033` | The centroid-to-detector sign/frame transformation is explicit, and effective derotation is derived from parent-sample contribution support propagated through the pixel fit support. |
| `BEAM-SCOPE-D034` | Bracketing pointing and associated science observations use the same immutable APT artifact and AST convention unless an authorized transform proves equivalence. |
| `BEAM-SCOPE-D035` | The rationale retains one concise soft-prior/convergence explanation and uses stable document-facing decision groups `SCI-BEAM-OD-001--003`. |
| `BEAM-SCOPE-D036` | The source flux is simply required in TOA mJy per fixed nominal beam at the declared reference origin; argumentative mJy-versus-mJy-per-beam and duplicate-factor language is removed. |
