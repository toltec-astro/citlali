# Closed uncertainty-role grammar - r0.4

Complete sole current Stage B review-candidate normative core: SCI-FRUIT-NORMATIVE-CORE v0.1/r0.4. Canonical source-inventory SHA-256: `3dc912d9f6a58d9f92561ad10b9442485c5b47232e5327389d2fb3b89c296e92`. Owner approval, freeze, Registry registration and activation are not established. Numerical methods/routes remain unavailable_pending_separate_owner_approval.

Prefix SCI-FRUIT/UNCERTAINTY/ plus exactly one of the thirteen terminal tokens below declares the product identity. No wildcard, Registry entry or activation is inferred. Exact core source: [definitions](src/common/definitions.tex).

| Core-declared token | Historical role split or retained |
| --- | --- |
| `MEASURED-PARENT-COVARIANCE` | measured-parent covariance |
| `CANDIDATE-MODEL` | grouped candidate/accepted/applied model/prior covariance |
| `CANDIDATE-SELECTION` | grouped model/selection dependence |
| `ACCEPTED-MODEL-COVARIANCE` | grouped model covariance |
| `ACCEPTANCE-TO-APPLICATION` | grouped model transformation dependence |
| `APPLIED-MODEL-COVARIANCE` | grouped model covariance |
| `PARENT-APPLIED-CROSS-COVARIANCE` | parent-model cross covariance |
| `OPERATOR-STATE` | operator/state uncertainty |
| `SUPPORT-THRESHOLD-SELECTION` | support/threshold selection |
| `STOPPING-TERMINAL-SELECTION` | stopping/terminal selection |
| `EXTERNAL-PRIOR` | grouped prior covariance |
| `EMPIRICAL-REPEATABILITY` | empirical repeatability |
| `NOI-PRODUCT-REFERENCE` | NOI uncertainty |

The NOI row is a FRUIT immutable reference to an exact NOI-owned product/source/law/estimator identity. Its external binding is currently `unavailable_external_binding`. MODEL-MISMATCH, MODEL-PATH-TRANSFER, MODEL-INDUCED-OUTPUT-SHIFT and MODEL-INDUCED-BIAS are separately addressable products outside this family. Deterministic discrepancy/shift is not covariance; a computable shift does not establish bias. One container is allowed only if every distinction is lossless.
