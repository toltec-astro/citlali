# SCI-FRUIT — notation, types and scientific roles r0.4

Status: owner-directed Stage A candidate; exact packet approval pending.
Authority: SCI-FRUIT-OD-STAGE-A-MICRO-REPAIR-2026-09-07, sections 1–4, 7;
the inherited core boundaries remain in force.
All symbols in this file belong to SCI-FRUIT and bind one exact method,
application scope, iteration and generation. They do not select a numerical
representation, estimator, unit, frame or sky prior.

## Type and notation crosswalk

| Semantic name | Scoped notation | Type and meaning |
| --- | --- | --- |
| Immutable measured base parent | `fruit.base_parent` | Exact producer product and ancestry; present in every iteration lineage |
| Astronomical target | `fruit.target` | Declared physical quantity, domain, units, frame and response reference; not a measured map by identity |
| Removal-input space | Y_k | Method-dependent input domain at the removal point |
| Candidate-model space | M_k^candidate | Exact method-dependent candidate domain |
| Accepted-model space | M_k^accepted | Exact method-dependent accepted domain; equality with other model domains is not assumed |
| Applied-model space | M_k = M_k^applied | Exact domain of the applied operand used by Pi_k and B_k; not assumed equal to Y_k, Z_k or the other model domains |
| Result space | Z_k | Method-dependent iteration-result domain |
| Exact iteration input | y_k^in in Y_k | Product selected by the method's iteration-input rule |
| Candidate model | m_k^candidate in M_k^candidate; `fruit.model_candidate_k` | Proposed external or internally constructed object, with origin and selection state |
| Accepted model | m_k^accepted in M_k^accepted; `fruit.model_accepted_k` | Exact model accepted for feedback, before any acceptance-to-application transformation |
| Applied model | m_k^applied in M_k; `fruit.model_applied_k` | Exact model operand used by the removal and rejoin maps; a distinct scientific object |
| Applied model state | `fruit.applied_model_state_k` | Binding of m_k^applied to realized maps, application counts, controls, support and contributions, including its relation to m_k^accepted |
| Removal map | Pi_k: M_k -> Y_k | Declared model projection/removal contribution map |
| Residual | rho_k in Y_k | y_k^in - Pi_k(m_k^applied), only on a method-defined compatible domain |
| Residual-processing procedure | F_k: Y_k -> Z_k | Complete declared procedure, with all fixed, recomputed and relearned state |
| Processed residual | `fruit.processed_residual_k` = F_k(rho_k) | Result of the residual path, not the rejoined iteration result |
| Rejoin map | B_k: M_k -> Z_k | Declared model contribution at the exact rejoin point |
| Rejoin contribution | `fruit.model_rejoin_k` = B_k(m_k^applied) | Model-path contribution; no automatic true-sky or unit-response claim |
| Iteration result | z_k in Z_k | Complete declared composition of residual and model paths |
| Update contribution | `fruit.update_k` | Separately typed causal increment, model change, diagnostic difference or unavailable role; never inferred from z_k |
| Cumulative model | `fruit.cumulative_model_k` | Present only if the method assigns cumulative meaning; distinct from an increment |
| State consumed / produced | S_k / S_(k+1) | Scientific state before / after iteration k, including all consequential dependencies |
| Fixed residual linear map | A_k: Y_k -> Z_k | Available only when F_k is established as the exact fixed linear map |
| Linear model-path operator | D_k = B_k - A_k o Pi_k: M_k -> Z_k | Defined only when A_k, Pi_k and B_k are all exact fixed linear maps on their declared spaces/support/state/conditioning; o denotes composition |
| Conditioning | Omega_k^FRUIT | Exact parents, operators, support, states, generations and assumptions held fixed for a claim |
| Measured-input/model-input response | R_(z<-y,k)^FRUIT,fixed / R_(z<-m^applied,k)^FRUIT,fixed | Separate response roles; the model query perturbs the applied operand, not the accepted model by implication |
| Covariance blocks | C_yy^FRUIT, C_mm^FRUIT, C_ym^FRUIT, C_my^FRUIT, C_z^FRUIT | Joint conditional covariance of (y_k^in, m_k^applied); m-labelled axes mean applied model only, with explicit units, support and Omega_k^FRUIT |

Candidate, accepted and applied model objects remain distinct even if a method
establishes equal numerical values. The scientific composition uses
m_k^applied. Equality m_k^applied = m_k^accepted requires an exact method/state
equality on compatible domains. Normalization, clipping, sign conversion,
support restriction, projection, unit conversion or any other transformation
between acceptance and application must be explicit, with its state and
generation. It is not hidden inside an alias for the accepted model.

If only F_k=A_k is fixed and linear, retain the map-valued expression
A_k y_k^in - A_k[Pi_k(m_k^applied)] + B_k(m_k^applied). Do not define D_k or
use a linear model response/covariance by implication when Pi_k or B_k is
nonlinear. The response table states the complete linear specialization.

## Scientific type declarations

Each space states quantity, units, frame, grouping, index/axis meaning, support,
missing/non-finite policy and generation. The ordinary subtraction and sum
require compatible additive structures in Y_k and Z_k. A linear or covariance
claim additionally names the relevant vector spaces or coordinates and its
validity domain. A mapping arrow or matrix-like notation does not prove that
the mapping is linear. Partial-domain arithmetic is governed by the support
table; undefined paths are not silently extended by zero.

Each selected parent route supplies its exact quantity, unit, frame, response,
beam/template convention, calibration, support, validity, lifecycle and
provenance. This packet selects no parent route or universal input unit.
Required domain conversions are method bindings; equal shape or unit labels
do not establish compatibility.

Method identity names the approved rules; realized coefficient values, learned
states and candidate/accepted/applied models have distinct state, iteration
and application generations under those rules. These identities are not aliases.

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
