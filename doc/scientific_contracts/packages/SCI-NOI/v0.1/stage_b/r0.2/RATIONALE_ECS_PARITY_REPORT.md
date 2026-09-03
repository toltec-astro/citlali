# Rationale and ECS Parity Report

Both `SCIENTIFIC_RATIONALE.md` and
`ENGINEERING_CONFORMANCE_SPECIFICATION.md` bind the same exact authority:

- identity: `SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.2`
- binding-file SHA-256:
  `5c6f954b457a546abcf74a1dc6dae190f2b22ea43c14edf34d2e4d2a8a704268`
- ordered modules: notation, definitions, equations, assumptions,
  requirements, predictions.

| Check | Rationale | ECS | Result |
| --- | --- | --- | --- |
| Independently authored normative equation | None | None | Parity |
| Independently authored requirement | None | None | Parity |
| Independently authored prediction | None | None | Parity |
| ODQ-102D mechanics | Explained as pending | Evidence not evaluable until accepted | Parity |
| Base product scope | Pending | Evidence not evaluable until accepted | Parity |
| Reciprocal successor | Pending | Evidence not evaluable until accepted/rebound | Parity |
| r0.2 profile action bytes | Successor required | Evaluation unavailable until successor | Parity |
| Claim ceiling | No implementation/performance/readiness claims | No current conformance result | Parity |

The PDFs are generated from these sources, with the normative-core PDF
appending the exact six modules and the ECS PDF appending the exact traceability
rows. Byte identity is checked by `verify_stage_b_r0_2.py`.
