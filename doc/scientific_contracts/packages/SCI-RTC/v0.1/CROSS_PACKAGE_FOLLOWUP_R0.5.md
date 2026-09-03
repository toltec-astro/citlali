# SCI-RTC v0.1/r0.5 cross-package follow-up

This register routes consequences without importing another package's
scientific authority into RTC.

| Owner/package | Follow-up |
| --- | --- |
| Tune/readout authority | Define and version the IQ-to-$x/r$ transform, reference, normalization, sign, validity/linearity domain, epoch, uncertainty, and covariance interface. |
| SCI-CAL | Confirm the conditioned-$x$-only handoff and represent target-atmosphere variation across RTC temporal support, including exact composition or an approved noncommutation bound. |
| SCI-BEAM | Bind Beammap factors and source-response inference to the complete paired RTC plan; do not use newly derived factors circularly inside the same Beammap RTC lineage. |
| AST/ALIGN | Preserve exact paired occurrence identity, cadence/phase, synthesis origin, coordinate binding, and support; missing a pair member is not ordinary optional metadata. |
| PTC/VAL/MAP/FLT | Declare each consumer's policy for nonrepresentative synthesis/replacement influence and bind the exact paired RTC parent; do not infer $r$ correction or calibration authority. |
| Validation program | Preregister mapping, paired validity, leakage, level-shift, atmosphere/filter composition, covariance, and role-plan studies against PRED-047--063. |

These are routed questions and evidence needs, not approvals or implementation
instructions.
