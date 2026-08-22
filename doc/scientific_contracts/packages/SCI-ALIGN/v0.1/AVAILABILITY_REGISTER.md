# SCI-ALIGN Stage B Targeted r0.3 Availability Register

Status: candidate-authority availability ledger; unresolved means typed
unavailable, not zero or inferred

Prepared: `2026-08-22`

| Availability ID | Unavailable claim | Governing cause | Scoped consequence |
| --- | --- | --- | --- |
| `SCI-ALIGN-UNAV-001` | Physical/absolute event time or offset validity beyond the conditional assigned relation | `SCI-ALIGN-ODQ-101` | Affected interface only; `t^ref` remains conditional on producer authority. |
| `SCI-ALIGN-UNAV-002` | Coordinate or field-rotation result requiring an unresolved observing-state field | `ODQ-102` | Dependent field or coordinate claim only. |
| `SCI-ALIGN-UNAV-003` | Physical scan segmentation or transition meaning from unresolved state/`Hold` semantics | `ODQ-103` | Processing windows remain available as distinct identities; physical-scan claim is blocked. |
| `SCI-ALIGN-UNAV-004` | Detector continuity surrogate, gap threshold, or long-gap action | `ODQ-104` | Conditional `EQ-008` is not authorized or enabled. |
| `SCI-ALIGN-UNAV-005` | Quantitative response, expanded mapping, covariance, model, mapping, or selection-uncertainty payload beyond the selected tier | `ODQ-105` | Dependent AST operation or quantitative claim only; unavailable is never numerical zero. |
| `SCI-ALIGN-UNAV-006` | Exact HWPR field requiring an unresolved registry item | `ODQ-109` | Affected optional HWPR field only. |
| `SCI-ALIGN-UNAV-007` | Polarization demodulation, efficiency, calibration, response, or Stokes reconstruction | `ODQ-109` plus a future polarimetry authority | Outside SCI-ALIGN v0.1 ordinary nonpolarimetric path; raw `x` is not Stokes I. |
| `SCI-ALIGN-UNAV-008` | Pointing correction or coordinate requiring an unresolved correction-record semantic | `ODQ-110` | Dependent correction or coordinate only. |
| `SCI-ALIGN-UNAV-009` | Time-varying drift/state-dependent correction | Bounded v0.1 constant-offset model | Affected interface/domain; no fitted correction. |
| `SCI-ALIGN-UNAV-010` | Cross-Tune or cross-readout-revision interpolation/synthesis | Absent separately named authority | Both affected paired outputs unavailable across the boundary. |
| `SCI-ALIGN-UNAV-011` | AST compatibility inferred from shape, field names, local row, or a substituted record | Exact `SCI-ALIGN_TO_SCI-AST v0.1/r0.1` rule | Incompatible absent declared profile/revision and semantic mapping. Stable slot `s` is not reconstructed from `j`. |
| `SCI-ALIGN-UNAV-012` | Implementation behavior/conformity, observational performance, validation, scientific approval, freeze, readiness, or production authorization | Stage B targeted implementation-blind boundary | Entire package assessment remains unassessed. |

Decided owner questions `SCI-ALIGN-ODQ-106`-`108` remain closed exactly as
recorded in `OWNER_DECISION_REGISTER.md`; the targeted revision does not reopen
them. No new scientific owner question was introduced.
