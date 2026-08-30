# SCI-NOI Stage B r0.1 to r0.2 Change Map

## Stable identifier migration

- Every `NOI-REQ-NNN` r0.1 identifier maps one-to-one to
  `SCI-NOI-REQ-NNN` for `001` through `037`.
- `SCI-NOI-REQ-038` through `SCI-NOI-REQ-042` are new: immutable-parent,
  no-role-promotion, immutable-profile, shared-module-binding, and claim-ceiling
  requirements.
- Every `NOI-PRED-NNN` r0.1 identifier maps one-to-one to
  `SCI-NOI-PRED-NNN` for `001` through `015`.
- `SCI-NOI-PRED-016` and `SCI-NOI-PRED-017` are new: singleton pending-state
  and bounded dimensionless-STD predictions.

## Canonical notation migration

| r0.1 spelling | r0.2 spelling | Reason |
| --- | --- | --- |
| `B_d` | `beta_d` | Avoid member-count collision. |
| `s_bd` | `epsilon_bd` | Reserve clear assignment notation; no source-symbol collision. |
| bare `S` design | `mathcal_D` | Distinguish full design identity from a matrix or standardized product. |
| `q_MAP` | `m_MAP` | Canonical real MAP numerator. |
| `V_hat_cond` | `Vhat_cond` | Single canonical conditional-second-moment spelling. |
| inverse-scale role | proposed `rho_cond` reciprocal role | Remove false scale/precision implication; owner acceptance pending. |
| `S_cond` | `zeta_cond` | Avoid collision with design/source notation. |
| generic/member count `B` | six `N_*` counts | Separate requested, resolved, completed, admitted, sign-unique, and orbit-unique counts. |
| one rank | `r_sign`, `r_map` | Separate sign-design and named projected information. |

## Scientific repairs

| r0.1 issue | r0.2 repair |
| --- | --- |
| ODQ-102D candidate mechanics appeared normative without a stable pending-owner question. | Complete candidate isolated under `SCI-NOI-OWNER-Q-102D-01`; all detailed mechanics unavailable until accepted. |
| Product scope not given a stable owner disposition. | `SCI-NOI-OWNER-Q-SCOPE-01`; recommended one observation/array scope remains unavailable. |
| Singleton prediction overreached the owner state. | Removed as binding; `SCI-NOI-PRED-016` states no prediction until accepted design treatment. |
| Retry could be confused with scientific replacement. | D-006 and `SCI-NOI-REQ-019` separate idempotent attempt identity from scientific identity. |
| Ordinary operator placement/response parity insufficiently centralized. | `SCI-NOI-EQ-001..007` and the operator/response report provide one canonical representation. |
| Reciprocal called an inverse scale. | Proposed narrow reciprocal successor and owner question; unavailable meanwhile. |
| Normative text duplicated across documents. | Six ordered byte-identical modules; rationale and ECS bind them and do not re-author science. |
| Profile binding did not state effect of changed bytes. | r0.18 profiles remain immutable; affected r0.2 evaluation unavailable until successor binding. |
