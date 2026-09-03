# SCI-FLT-MATCHED v0.1 r0.2 Notation Crosswalk

Status: normative notation repair for owner review

| r0.1 symbol/phrase | r0.2 symbol/phrase | Disposition |
| --- | --- | --- |
| `x`, application location | `p`, matched-template output anchor | Replaced. `x` remains reserved to paired KID readout authority. |
| `z`, parent-map sample | `q`, parent-map row/pixel | Replaced. |
| `y`, second template location | `s`, separately indexed template anchor | Replaced. `r` is not used because it remains reserved with `x` to paired KID readout authority. |
| `m`, `m_x` | `m`, `m_p = E_p m` | Restricted parent is now explicit. |
| `t_x(z)` | `t_pq`, `t_p = E_p t_p^full` | Anchor, parent row, full object, and restriction are explicit. |
| `Q_x` | `W_p` | Replaced. Bare `Q_p` remains authoritative in MAP/JINC. `W_p` is the complete coordinate-basis bilinear weight. |
| `<a,b>_x = sum a* b mu` | ordinary matrix action with all measure factors inside `W_p` | The double-measure/inverse-covariance ambiguity is removed. |
| `N(x)` | `n_p = t_p^T W_p m_p` | Renamed and algebraically closed. |
| `D(x)` | `d_p = t_p^T W_p t_p` | Renamed; exactly real and positive in science. |
| `A`, `Ahat(x)` | `A`, `ahat_p` | Physical/model amplitude and estimated amplitude remain distinct. |
| `L_x` | `L_p,g` | Anchor and frozen state generation are explicit. |
| `R_t(x,y)` | `R_fixed(p,s)` | Fixed-state response is explicitly typed. |
| implicit data-dependent response | `R_FP(p,s)` | New separately authorized full-procedure response; unavailable without rerun authority. |
| `L-tilde`, `R-tilde`, `C-tilde` as approximation symbols | `L-tilde`, `R_realized`, `C_realized` | Actual consequences attach to the operator that produced the field. |
| `R_t`, `C_cond` as comparators | `R_reference`, `C_reference` | Exact comparators are not mislabeled as actual. |
| one influence support | `S_apply`, `S_template`, `S_state`, `S_response`, `S_store` | Application, query, learning/provenance, and storage roles are separated. |
| `DeclareOrLearnOnce` | `Learn -> Resolve -> Apply` | Candidate derivation, authorized selection, and frozen use are distinct. |
| `complete` | `complete_publication_candidate` | Does not imply publication, SCI-VAL pass, observational validation, or readiness. |

Under a coordinate reparameterization `m'=S m`, the r0.2 convention uses
`C'=S C S^T`, `t'=S t`, and `W'=S^-T W S^-1`. This preserves the two scalar
contractions without an additional measure matrix.
