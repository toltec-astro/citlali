# SCI-MAP v0.2/r0.3 SCI-NOI terminology and ownership amendment

Status: **CANDIDATE review extraction**. Normative wording is in
`../src/common/definitions.tex` and REQ-024/REQ-050 in the shared authority.

SCI-NOI owns its exact realization-generation, empirical-uncertainty,
conditional-scale, and standardization methods and products. Any
physical-noise, total-uncertainty, covariance-completeness, significance,
probability, false-rate, completeness, purity, or catalogue interpretation
requires a separately approved exact method and remains unavailable by
default.

The following identities and claims remain distinct:

| Object | Authorized meaning in SCI-MAP v0.2/r0.3 | Prohibited promotion |
| --- | --- | --- |
| `Q` | MAP normalization accumulated from the exact admitted coefficients. | Exposure, hit count, empirical weight, physical-noise precision, covariance completeness, or significance. |
| Formal conditional precision | A formal result only under the stated coefficient, independence, projection, and conditioning premises. | Empirical uncertainty, total uncertainty, or achieved noise model. |
| Complete conditional covariance | Exact declared covariance propagated on exact domains when all required terms exist. | Unconditional covariance or empirical/physical-noise completeness. |
| NOI `conditional_detector_sign_randomization_marginal_second_moment` | The exact NOI-owned conditional marginal second-moment product under its own frozen method and conditions. | Generic empirical noise, complete covariance, total MAP uncertainty, or significance. |
| NOI standardized signal | The exact standardized signal authorized by its NOI method. | Calibrated significance, detection probability, false-alarm probability, completeness, purity, or catalogue authority. |
| Future calibrated-significance method | Unavailable unless a separately approved exact method, product identity, assumptions, and claim exist. | Inference from a formal weight, standardized signal, or historical label. |

MAP may consume a versioned NOI product only under the exact claim that the
NOI method authorizes. MAP does not rename a formal or standardized object to
obtain a stronger interpretation.
