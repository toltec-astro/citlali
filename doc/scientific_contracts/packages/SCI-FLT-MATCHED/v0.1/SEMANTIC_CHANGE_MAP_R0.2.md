# SCI-FLT-MATCHED v0.1 — r0.1 to r0.2 Semantic-Change Map

Status: proposed Stage B review map; no numerical or authored option selected

## Stable requirement identities

The r0.2 draft preserves `REQ-001` through `REQ-039`. The following IDs receive
targeted semantic repair; all unlisted IDs retain their r0.1 substance.

| ID | r0.1 issue | r0.2 repair |
| --- | --- | --- |
| `REQ-005` | Required materialization | Requires one scientific template-response object; exact materialized, structured, or lineage representation is permitted. |
| `REQ-007` | Colliding notation and ambiguous measure | Binds collision-free `p/q/W_p` notation and complete coordinate-basis bilinear weighting. |
| `REQ-008` | Measure-weighted bracket plus inverse-covariance ambiguity | Defines the exact coordinate-basis `n_p/d_p` operator with all measure factors inside `W_p`. |
| `REQ-009` | Made scientific reality depend on tolerance | Requires `d_p` exactly real and positive; finite-precision checks move to the engineering profile. |
| `REQ-010` | Collapsed several support concepts | Requires complete `S_apply`; other support families are separately typed. |
| `REQ-011` | Described only fixed-state response | Separates fixed-state, full-procedure, realized, and reference response. |
| `REQ-012` | Singular GLS competitor class incomplete | Adds exact local restriction, estimable projector, identifiability, and competitor class. |
| `REQ-015` | Opaque `DeclareOrLearnOnce` | Replaces it with immutable Learn--Resolve--Apply identities and generations. |
| `REQ-020` | Could attach reference covariance to a different realized field | Requires covariance of the operator that actually produced the field and retains reference covariance separately. |
| `REQ-026` | Defined disabled as not requested and had only five states | Binds the expanded lifecycle vocabulary and keeps `not_requested` distinct from `disabled`. |
| `REQ-030` | Said the method did no fitting | States fixed-template, fixed-anchor, one-parameter linear amplitude estimation while preserving all source-analysis exclusions. |
| `REQ-033` | Treated numerical ceilings as science | Requires exact science plus one preregistered engineering numerical-conformance profile; a scientifically different operator gets distinct identity. |
| `REQ-035` | Used validation ambiguously | Defines SCI-VAL as named-use profile evaluation on four axes, separate from observational/scientific validation. |
| `REQ-038` | Did not require complete boundary/route records | Adds exact producer boundaries, route status, and reconstruction/query closure. |

New requirements `REQ-040` onward close obligations that had no stable r0.1
identity. Existing IDs are not renumbered or reused.

## Stable prediction identities

`PRED-001` through `PRED-018` remain stable. `PRED-008` is corrected: its
sampling population is the exact declared parent stochastic law with
conditional covariance `C_parent`; NOI compatibility alone is insufficient.
`PRED-010` is repaired to compare a realized operator with its exact reference
under the preregistered numerical-conformance profile rather than under an
owner-selected scientific threshold. New predictions cover overlapping
templates, deterministic background, off-diagonal response, mismatch,
fixed-versus-full-procedure response, and realized covariance.

## Equation identities

| r0.1 form | r0.2 form | Meaning |
| --- | --- | --- |
| `N(x)=<t_x,Q_xm_x>_x` | `n_p=t_p^T W_p m_p` | Complete coordinate-basis bilinear action; no unexplained extra measure. |
| `D(x)=<t_x,Q_xt_x>_x` | `d_p=t_p^T W_p t_p` | Exact real positive normalization. |
| `m_x` implicit restriction | `m_p=E_pm`, `t_p=E_pt_p^full`, `C_p=E_pC_parent E_p^T` | Restriction precedes local inversion. |
| shared-null Moore--Penrose wording | `P_p`, positive-definite `C_{p,E}`, `W_p=P_p^T C_{p,E}^{-1}P_p` | Constrained local GLS with explicit estimable subspace. |
| single-template expectation only | `E[ahat_p|g]=sum_r R_fixed(p,r)A_r+L_{p,g}b` | General-sky response and nuisance leakage. |
| one response symbol | `R_fixed`, `R_FP`, `R_realized`, `R_reference` | Fixed-state, learned-procedure, realized, and exact-reference meanings remain distinct. |
| one conditional covariance symbol | `C_realized` and `C_reference` | A product carries covariance of the operator that produced it. |

## Authored-option repair

- `AO-001-C` retains its stable identity but is renamed field-power spectral
  weighting and loses every implied noise/covariance/isotropy/optimality claim.
- `AO-002-A` through `AO-002-C` are no longer privileged scientific numerical
  thresholds. They distinguish exact-operator numerical conformance, a
  separately authorized scientifically different operator, and typed
  unavailability.
- `AO-003` separates scientific covariance scope from exact representation;
  representation-only changes do not change covariance identity.
- `AO-004` makes exact state and query reproducibility invariant across its
  representation alternatives.
- `AO-005` makes response domain/query/validity/consumer scope invariant before
  choosing an exact representation.
- `AO-006` makes role semantics and dependency normative; its alternatives are
  lossless record layouts and include a separate response-use verdict.

This map is part of the r0.2 owner-review packet. It does not freeze any repair
or select any route.
