# SCI-RTC v0.1/r0.6 cross-package follow-up

| Owner/package | Follow-up |
| --- | --- |
| TUNE/readout mapping | Supply the admitted general $\mathcal T_{d,\zeta}(I,Q)$ authority, domain, uncertainty, and any local Jacobian/affine representation. |
| ALIGN | Preserve exact paired admission and raw aligned parent identity used by representative occurrence and original-pair shift learning. |
| PTC or successor | Own any future numerical atmospheric/common-mode removal estimator, including null modes, response, covariance, source protection, and validation. |
| AST/source templates | Supply exact source/template identity and support for bright-source leakage and source-versus-shift discrimination. |
| SCI-BEAM/Pointing/OOF | Consume conditioned $x$ with the exact raw $r$ parent and role-specific response/plateau support; do not infer a conditioned $r$ branch. |
| SCI-CAL | Continue to consume conditioned $x$ only; target atmosphere and absolute calibration remain downstream. |
| VAL | Define consumer-specific use of causal $r$ selectors, role-specific plateau support, and pre/post-shift response-change causes. |
| MAP/FLT | Bind the exact conditioned-$x$ response and support; diagnostic atmosphere evidence does not imply numerical sky cleaning. |
| Readout-health products | Define which leakage, plateau, and shift diagnostics are required and whether any separately conditioned $r$ product is requested. |

These are routed authority and evidence needs, not approvals or implementation
instructions.
