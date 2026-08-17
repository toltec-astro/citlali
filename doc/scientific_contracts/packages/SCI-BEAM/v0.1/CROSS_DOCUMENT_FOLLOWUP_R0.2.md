# SCI-BEAM v0.1 r0.2 — Required Cross-Document Follow-Up

Status: future governance work; no other package edited by this revision

Date: `2026-08-17`

The following later amendments are required to align adjacent scientific
authorities with SCI-BEAM r0.2. This list does not claim current implementation
behavior and does not itself amend the named package.

| Follow-up | Required alignment | Present action |
| --- | --- | --- |
| SCI-CAL raw input | Define `x_d=Delta f_d/f_d` and `xs` as the raw uncalibrated detector observable, including reference frequency, sign, zero/baseline, conditioning, and valid domain. | Deferred; SCI-CAL unchanged. |
| SCI-CAL factor lineage | Identify BEAM as producer/acceptance owner of source-APT `flxscale`, TolProj as approved association/child transformer, and SCI-CAL as selected-factor plus target-atmosphere consumer. | Deferred; SCI-CAL unchanged. |
| SCI-CAL and shared schemas | Remove active scientific semantics from `responsivity`; distinguish it as deprecated compatibility metadata. | Deferred. |
| SCI-CAL sensitivity boundary | Identify BEAM as producer/scientific owner of source-APT `sens`; preserve that it is neither calibration nor total uncertainty. | Deferred. |
| SCI-MAP signal vocabulary | Distinguish raw `xs`, calibrated detector signal, standardized detector Beammap, and mapped calibrated signal; preserve response lineage. | Deferred; SCI-MAP unchanged. |
| ALIGN/AST | Align raw/horizon coordinates, full WCS metric, detector-specific field rotation, conventional pivot, arbitrary origin gauge, PSF-tensor rotation, and same-APT pointing transfer. | Deferred to ALIGN/AST authority. |
| Pointing transfer | Require bracketing pointings and science observations to use the same immutable APT realization and AST convention unless a governed transform proves equivalence. | Deferred to pointing/AST packages. |
| Weighting authority | Define admission, normalization, covariance assumptions, and claim limits for approximate coefficients proportional to `1/sens^2`. | Deferred to the future weighting/noise authority. |
| TolProj | Align required source APT, target association, approved child transformation, retained ancestry, and no reinterpretation of BEAM quantities. | Deferred to TolProj governance. |
| Kernel/filter authority | Establish science-impact criteria for Gaussian, empirical, or unavailable kernels and distinguish effective core from complete response. | Deferred to downstream kernel/FLT authority. |
| Scientific-contract program | Clarify that the rationale is scientist-facing while the formal contract carries the complete normative inventory and exact machinery. | Deferred program-level amendment. |

Each follow-up requires its owning scientific authority and normal review gate.
No change may be inferred solely because SCI-BEAM now exposes an interface.
