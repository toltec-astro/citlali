# SCI-ALIGN Stage B Targeted r0.2 Formal-ID Change Map

Status: semantic amendment map; stable identifiers preserved

Prepared: `2026-08-22`

## Equations

| Stable ID | Targeted amendment | Scientific meaning preserved or clarified |
| --- | --- | --- |
| `SCI-ALIGN-EQ-001` | `i_ref`, `t^ref`, and `delta_(i->ref)` replace reference-overloaded `r` notation. | Checked conversion, positive-add offset, and exactly-once application unchanged. |
| `EQ-002`-`003` | Stable slot `s`, grid `S`, times `t_s`, and identity `(o,s)` replace ALIGN use of `n`. | Grid and round-half-up assignment unchanged; `j` demoted to explicit view-local row. |
| `EQ-004`-`005` | Targets indexed by `s`; circular interval changed to `[-P/2,P/2)`; generic field endpoints are `v_a,v_b`, not KID-reserved `x_a,x_b`. | Scalar and circular mathematics unchanged; antipodal interpolation explicitly unavailable without unwrap authority. |
| `EQ-008` | Internal gap-position index changed from `j` to `ell`. | Avoids overloading local storage row; conditional-only surrogate meaning unchanged. |
| `EQ-009`-`010` | Processing windows/selectors expressed over stable slots `s`. | Window membership cannot become external identity or replace physical scans. |
| `EQ-007` | Generic stacked detector/telescope/HWPR inputs use `boldsymbol v_D,v_T,v_H` and `boldsymbol v`. | Block mapping unchanged; `x/r` remain exclusive paired KID coordinates. |
| `EQ-011`, `EQ-013`-`014` | Generic response and covariance inputs use `boldsymbol v`, `C_v`, and `v_a,v_b`. | Response, affine covariance propagation, scalar variance, and all stable identities are mathematically unchanged. |
| `EQ-018`-`020` | Exposure/local support indexed by `s`; product state renamed `mathcal S_ALIGN`. | Exposure, origin, local validity, and no-global-eligibility semantics unchanged. |

Equations `006`-`007` and `011`-`017` retain their prior mathematical content.
All 20 equation IDs remain exact and sequential.

## Requirements

| Stable ID | Targeted amendment |
| --- | --- |
| `SCI-ALIGN-REQ-003` | Makes `(observation,s)` stable identity, `j` local row only, downstream reconstruction prohibited, and `n` RTC-reserved. |
| `REQ-018` | Fixes circular shortest signed difference to `[-P/2,P/2)` and requires explicit unwrap authority for an antipodal interpolation. |
| `REQ-023` | Replaces “Stokes-I detector support” with ordinary nonpolarimetric detector-signal support. |
| `REQ-051` | States ordinary nonpolarimetric scope, raw KID `x/r`, and no demodulation, polarization calibration/response, or Stokes reconstruction. |
| `REQ-008`-`010`, `012`-`015`, `027`, `030`, `033`, `042`-`047`, `053`-`054` | Meaning retained; notation/source/provenance interpretation now follows the revised shared modules and boundary. |

All 55 requirement IDs remain exact and sequential; none was removed,
reassigned, renumbered, or semantically changed by the horizontal-audit
generic-operand repair.

## Predictions

| Stable ID | Targeted amendment |
| --- | --- |
| `SCI-ALIGN-PRED-001` | Uses `delta_(T->ref)` with `i_ref=D`; the numerical fixture and sign falsifier are unchanged. |
| `PRED-005` | Names the exact `[-P/2,P/2)` convention and retains antipodal unavailability. |
| `PRED-019` | Names common stable slot `s` in the paired-source fixture. |
| Fail-closed edge prose after `PRED-026` | Replaces Stokes-I wording with ordinary nonpolarimetric path wording. |
| `PRED-009`, `010`, `014`, `021`, `023`, `024`, `026` | Fixture meaning retained; stable-slot, local-row, boundary, and product-state interpretation follows r0.2 notation. |

All 26 prediction IDs remain exact and sequential. The generic-operand repair
does not amend prediction mathematics or content. No prediction is reported as
executed or passed; they remain future implementation-independent comparison
targets.
