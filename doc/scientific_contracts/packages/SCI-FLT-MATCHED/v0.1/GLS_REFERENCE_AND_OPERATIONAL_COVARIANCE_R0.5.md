# GLS reference variance and operational covariance — r0.5

Status: normative role-separation amendment

'v_GLS,reference(p) = d_p^-1' is a theorem-supported marginal reference
variance only for AO-001-A under every exact local constrained-GLS premise.
It is distinct from:

- 'C_U1,reference = P_C L C_parent|h_pre L^T P_C^T'; and
- 'C_U1,realized = Cov[P_C F_g(m) | h_pre]'.

| Combination | Reference marginal variance | Operational covariance |
|---|---|---|
| AO-001-A + AO-003-C | may be available | unavailable |
| AO-001-A + complete/projected AO-003 | may be available | separately available when its premises hold |
| AO-001-C or other non-GLS route | unavailable; 'd_p' is normalization | separately available or unavailable |
| missing or population-mismatched parent covariance | unavailable | unavailable |

AO-003-C forbids operational covariance, covariance-based standardization,
significance, draws, independence, and covariance-dependent uses. It does not
erase independently established 'v_GLS,reference'.
