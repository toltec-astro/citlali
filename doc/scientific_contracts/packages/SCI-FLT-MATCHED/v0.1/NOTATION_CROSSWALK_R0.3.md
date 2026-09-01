# SCI-FLT-MATCHED v0.1 r0.3 Notation Crosswalk

Status: owner-authorized independent-review repair; no scientific or
engineering route selected

| Earlier symbol or ambiguity | r0.3 exact symbol or rule | Disposition |
| --- | --- | --- |
| `x`, application location | `p`, matched-template output anchor | Replaced. `x` remains reserved to paired KID readout authority. |
| `z`, parent-map sample | `q`, parent-map row/pixel | Replaced. |
| `y`, second template location | `s`, separately indexed template anchor | Replaced. `r` remains available for an explicitly named covariance role. |
| implicit local support | `D_loc(p)` and exact extraction `E_p` | The construction domain is declared before local inversion or weight construction. |
| one influence support | `S_apply`, `S_template`, `S_state`, `S_response`, `S_store` | Final payload support is the exact nonzero row pattern and cannot define the earlier construction domain. |
| `m`, `m_x` | `m`, `m_p = E_p m` | Full parent and local-domain restriction are distinct. |
| `t_x(z)` | `t_pq`, `t_p = E_p t_p^full` | Anchor, parent row, full template, and restriction are explicit. |
| bare `Q_x` | `W_p` | `W_p` is the complete coordinate-basis bilinear weight; bare `Q_p` remains MAP/JINC authority. |
| extra measure bracket | ordinary matrix action with every measure factor inside `W_p` | Double-counting of measure and inverse covariance is excluded. |
| `N(x)`, `D(x)` | `n_p = t_p^T W_p m_p`, `d_p = t_p^T W_p t_p` | The numerator and exact positive normalization are algebraically closed. |
| `A`, `Ahat(x)` | `A`, `ahat_p` | Physical/model amplitude and its estimate remain distinct. |
| `L_x` | `L_p,g` | Exact scientific row and immutable state generation are explicit. |
| numerical operator presumed linear | `F_g`, the actual fixed-state producing map | `F_g` may branch or fail and is not presumed exactly linear. |
| unconditional realized matrix | operational `R_realized` and `C_realized^(U1)` from `F_g` | Matrix forms using `L-tilde_g` are allowed only after fixed-state linearity is established on the declared domain. |
| one response | `R_fixed`, `R_FP`, `R_realized`, `R_reference` | Exact-science, full-procedure, operational-realized, and reference comparators are typed separately. |
| one covariance meaning | named U1, U2, or separately authorized joint role | U1 conditional stochastic variation and U2 template/CAL/BEAM variation cannot be combined by implication. |
| empirical sampling as science | exact covariance identity plus a preregistered engineering sampling protocol | Sampling convergence is evidence, not the scientific definition. |
| informal compatible NOI | frozen pre-Apply predicate `K_NOI(z,y)` | NOI owns population/membership meaning; FLT owns transformation compatibility. |
| unbounded future FRUIT reconstruction | finite `Q_FLT^0.1` | Extra future queries require a successor FLT envelope or generation. |
| unindexed radial bins/regularization | `B_pb`, `Pbar_pb`, `lambda_p,b(k)` | Anchor and bin dependence are explicit; half-open ties and final-boundary rules are exact. |
| `DeclareOrLearnOnce` | `Learn -> Resolve -> Apply` | Candidate derivation, authorized selection, and frozen use are distinct. |
| `complete` | `complete publication candidate` | This does not imply publication, validation, readiness, or production. |

Under an invertible coordinate reparameterization `m'=S m`, r0.3 retains
`C'=S C S^T`, `t'=S t`, and `W'=S^-T W S^-1`. These preserve the two scalar
contractions without an additional measure matrix.
