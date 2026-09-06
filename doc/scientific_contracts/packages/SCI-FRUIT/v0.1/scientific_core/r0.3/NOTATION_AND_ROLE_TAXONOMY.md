# SCI-FRUIT — notation, types and scientific roles r0.3

Status: owner-directed Stage A candidate; exact packet approval pending.
Authority: SCI-FRUIT-OD-STAGE-A-TYPE-METHOD-REPAIR-2026-09-06, sections 1, 6, 15.
All symbols in this file belong to SCI-FRUIT and bind one exact method,
application scope, iteration and generation. They do not select a numerical
representation, estimator, unit, frame or sky prior.

## Type and notation crosswalk

| Semantic name | Scoped notation | Type and meaning |
| --- | --- | --- |
| Immutable measured base parent | `fruit.base_parent` | Exact producer product and ancestry; present in every iteration lineage |
| Astronomical target | `fruit.target` | Declared physical quantity, domain, units, frame and response reference; not a measured map by identity |
| Removal-input space | Y_k | Method-dependent input domain at the removal point |
| Accepted-model space | M_k | Method-dependent model domain; not assumed equal to Y_k or Z_k |
| Result space | Z_k | Method-dependent iteration-result domain |
| Exact iteration input | y_k^in in Y_k | Product selected by the method's iteration-input rule |
| Candidate model | `fruit.model_candidate_k` | Proposed external or internally constructed object, with origin and selection state |
| Accepted model | m_k in M_k | Model accepted for feedback; the mathematical model argument of the registered maps |
| Applied model state | `fruit.model_applied_k` | Exact binding of m_k to realized maps, application counts, controls, support and contributions; distinct from acceptance |
| Removal map | Pi_k: M_k -> Y_k | Declared model projection/removal contribution map |
| Residual | rho_k in Y_k | y_k^in - Pi_k(m_k), only on a method-defined compatible domain |
| Residual-processing procedure | F_k: Y_k -> Z_k | Complete declared procedure, with all fixed, recomputed and relearned state |
| Processed residual | `fruit.processed_residual_k` = F_k(rho_k) | Result of the residual path, not the rejoined iteration result |
| Rejoin map | B_k: M_k -> Z_k | Declared model contribution at the exact rejoin point |
| Rejoin contribution | `fruit.model_rejoin_k` = B_k(m_k) | Model-path contribution; no automatic true-sky or unit-response claim |
| Iteration result | z_k in Z_k | Complete declared composition of residual and model paths |
| Update contribution | `fruit.update_k` | Separately typed causal increment, model change, diagnostic difference or unavailable role; never inferred from z_k |
| Cumulative model | `fruit.cumulative_model_k` | Present only if the method assigns cumulative meaning; distinct from an increment |
| State consumed / produced | S_k / S_(k+1) | Scientific state before / after iteration k, including all consequential dependencies |
| Fixed residual linear map | A_k: Y_k -> Z_k | Available only when F_k is established as the exact fixed linear map |
| Model effect map | D_k = B_k - A_k Pi_k | Pointwise difference map M_k -> Z_k; its definition alone does not establish linearity |
| Conditioning | Omega_k^FRUIT | Exact parents, operators, support, states, generations and assumptions held fixed for a claim |
| Parent/model response | R_(z<-y,k)^FRUIT,fixed / R_(z<-m,k)^FRUIT,fixed | Separate response roles; conditions are in the response table |
| Covariance blocks | C_yy^FRUIT, C_mm^FRUIT, C_ym^FRUIT, C_my^FRUIT, C_z^FRUIT | Joint conditional covariance blocks with explicit axes, units, support and Omega_k^FRUIT |

Accepted and applied model records are not interchangeable. The equations use
the exact accepted model argument m_k; transformations, restrictions or changes
used in application must be part of the declared maps and applied-state record.
If a method uses another model operand, it must identify that operand and its
relation to acceptance. There is no silent replacement of m_k.

## Scientific type declarations

Each space states quantity, units, frame, grouping, index/axis meaning, support,
missing/non-finite policy and generation. The ordinary subtraction and sum
require compatible additive structures in Y_k and Z_k. A linear or covariance
claim additionally names the relevant vector spaces or coordinates and its
validity domain. A mapping arrow or matrix-like notation does not prove that
the mapping is linear. Partial-domain arithmetic is governed by the support
table; undefined paths are not silently extended by zero.

An external model retains its external parent and prior. An internally learned
model retains its learning parents, population, selection, response, uncertainty
and state generations. Either may carry only the claim `feedback_model_state`
under this core. Feedback acceptance does not establish model truth, a
detection, source catalogue, posterior or deconvolved sky, unbiased source
estimate, Pointing or OOF result.

## Reserved notation and frozen-source cover

Do not use x or r for new generic FRUIT signal, residual or model quantities;
they remain reserved for paired KID readout coordinates. Do not borrow bare
Q_p, N_p, C_p or unqualified NOI sign/member symbols. Use the semantic names
above or explicitly FRUIT-qualified symbols.

The byte-exact [PTC excerpt](PTC_APPLICATION_REFERENCE.tex) contains inherited
PTC-local notation, including x_(g,t) for its centered row. Preserve the quote;
in FRUIT explanatory prose call that object `PTC.centered_row`, its loading
matrix `PTC.loading_matrix`, and its application coefficients
`PTC.application_coefficients`. This crosswalk does not rename paired KID x/r
or edit frozen PTC equations. F_k here is the FRUIT residual-processing map;
S_k is the FRUIT state, so F_k must not also name carried feedback state.
