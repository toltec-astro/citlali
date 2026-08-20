# SCI-CAL Scientific Owner Decision Ledger

Status: active companion to the SCI-CAL v0.1 rationale r0.3; no open item is
resolved by inference

Date: 2026-08-16

State vocabulary: open, decided, deferred, superseded.

| Decision ID | Owning scientific authority | Status | Evidence or decision required | Blocked claim or product | Resolution authority | Resolution date | Affected documents |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SCI-CAL-OWNER-Q01 | Ordinary-xs signal producer and signal-definition authority; named owner not yet recorded | open | Physical observable, unit, sign, prior normalization and additive processing, stream scope, and valid loading/tune/signal domain | Complete physical meaning and units of the input and flxscale; universal multiplicative interpretation | — | — | Science rationale; engineering contract; validation plan |
| SCI-CAL-OWNER-Q02 | Upstream conditioning owners plus the SCI-CAL and MAP/FLT boundary owners; exact decision owner not yet recorded | open | Affine/baseline convention and exact order relative to common-mode removal, PCA, temporal filtering, weighting, and mapmaking | Unique interpretation of noncommuting operations and realized downstream transfer functions | — | — | Science rationale; engineering contract; MAP/FLT contracts; validation plan |
| SCI-CAL-OWNER-Q03 | Beammap calibration and source-APT production scientific authority | open | Calibrator model and epoch, source-atmosphere treatment, amplitude estimator, beam/template fit, pointing treatment, spectral convention, factor direction, and uncertainty | Full scientific derivation and meaning of source-APT flxscale | — | — | Science rationale; Beammap/source-APT contract; engineering contract |
| SCI-CAL-OWNER-Q04 | Beammap/source-APT producer authority and TolProj child-APT transformation authority | open | Transfer validity across time, tune, loading, focus, pointing, and detector state; approved child transformations and retained systematics | Universal transferability of source calibration to a target observation | — | — | Science rationale; Beammap/source-APT contract; TolProj contract; validation plan |
| SCI-CAL-OWNER-Q05 | Instrument photometric/passband and calibrator-model scientific authorities; named decision owner not yet recorded | open | Reference frequency or wavelength, reference spectrum, calibrator/color treatment, passband weighting and variation, and atmosphere-weighting relationship | Complete monochromatic meaning of reported mJy and cross-array spectral comparison | — | — | Science rationale; engineering contract; atmosphere contract; validation plan |
| SCI-CAL-OWNER-Q06 | Atmosphere-operator, opacity-source, and passband scientific authorities; named decision owner not yet recorded | open | Exact nodes, ordinates, orientation, units, interpolation/seams, support, weighting, generating model, and content identity; classify project-wide versus reviewed-material absence | Contract-supported numerical calibration, correction curves, and approximation-error claims | — | — | Science rationale; engineering contract; atmosphere authority; validation plan |
| SCI-CAL-OWNER-Q07 | SCI-CAL scientific owner with the upstream coherent-segment declaration authority | open | Rationale for tau225 boundaries, segment owner, split policy, and evidence required to change limits | Scientific explanation and change authority for the adopted opacity policy | — | — | Science rationale; engineering contract; segment-policy authority |
| SCI-CAL-OWNER-Q08 | Producers of conditional-noise and systematic-uncertainty products, coordinated by SCI-CAL/MAP/NOI owners | open | Which conditional and systematic terms are numerically available, propagated, correlated, and present in products | Total calibrated uncertainty, total significance, and completeness claims | — | — | Science rationale; engineering contract; MAP/NOI contracts; validation plan |
| SCI-CAL-OWNER-Q09 | SCI-CAL scientific owner and designated validation authority | open | Preregistered evidence thresholds, populations, supports, covariance models, and decision rules for fidelity, repeatability, and absolute recovery | Any achieved validated-science-use, observational-performance, or production-readiness claim | — | — | Science rationale; engineering contract; validation plan; production-readiness record |

The earlier SCI-CAL-OWNER-Q001 atmosphere-content question is superseded by
SCI-CAL-OWNER-Q06, which retains all of its required fields and adds an
explicit ownership and absence-classification decision.

This ledger records responsibility without manufacturing authority. When an
item is resolved, the state changes only with a cited owner decision or
approved scientific source and its date. Table 4 of rationale r0.3 was checked
against this ledger on 2026-08-16 for decision ID, issue, closure condition,
and blocked consequence.

Engineering conformance r0.2 was checked against this ledger on 2026-08-20.
It carries Q01--Q09 as explicit admission gates with claim-specific
unavailable consequences. The alignment repair does not resolve, defer, or
supersede any ledger item.
