# Numerical application-domain amendment — r0.5

Status: normative closure amendment; no implementation claim

The parent row-identity/fact domain 'S_parent_fact' is distinct from the
numerical payload domain
'D_m = {q in S_parent_fact : an admitted finite real parent signal payload exists}'.
For each anchor, 'D_loc(p)' is the predeclared construction domain. After
constructing 't_p', 'W_p', and 'd_p', define

'ell_p* = W_p t_p / d_p', 'c_p = E_p^T ell_p*', and
'S_apply(p) = {q : c_pq != 0}'.

Apply evaluates only 'sum_(q in S_apply(p)) c_pq m_q'. Every active coordinate
must be in 'D_m'. A coordinate in 'D_loc(p)' with exact-zero final coefficient
may remain necessary to construct state or normalization but needs no signal
payload dereference. Exact zero is canonical scientific equality, never a
numerical threshold. 'm_p = E_p m' is shorthand only when a complete numerical
local vector exists or an explicit completion is proved irrelevant outside
'S_apply(p)'.

This amendment is represented normatively in REQ-007, REQ-008, REQ-010,
REQ-042, PRED-004, and appended PRED-025.
