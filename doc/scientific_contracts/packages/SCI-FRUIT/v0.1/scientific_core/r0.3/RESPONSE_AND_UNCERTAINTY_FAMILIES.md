# SCI-FRUIT — typed composition, response and uncertainty r0.3

Status: owner-directed Stage A candidate. Authority: repair direction
SCI-FRUIT-OD-STAGE-A-TYPE-METHOD-REPAIR-2026-09-06, sections 1, 8–9, 16.
Notation and domains are defined in [the type crosswalk](NOTATION_AND_ROLE_TAXONOMY.md).
These conditional formulas select no numerical FRUIT method or covariance.

## Typed remove/process/rejoin relation

For the declared ordinary additive structure, on compatible method-defined
domains and supports,

\[
\rho_k = y_k^{\rm in}-\Pi_k(m_k),\qquad
z_k = F_k(\rho_k)+B_k(m_k).
\]

The negative removal and positive rejoin signs are method identity. Another
sign, ordering, affine term, nonlinear operation, recomputed coefficient,
relearned state or support rule requires its own exact method identity.
Pi_k and B_k have different
codomains; they are not assumed equal or inverse. The model need not share a
domain with the measured signal. F_k is the complete residual-processing
procedure; it need not be linear. Removal/rejoin support need not agree, and
model truth or unit response is not assumed.

Only when F_k is established as the **exact fixed linear operator** A_k on
the declared input/result spaces does distributivity give

\[
z_k=A_k\!\left(y_k^{\rm in}-\Pi_k(m_k)\right)+B_k(m_k)
   =A_k y_k^{\rm in}+D_k(m_k),\qquad
D_k:=B_k-A_k\Pi_k.
\]

Here D_k is initially a pointwise map difference. If Pi_k and B_k are not
known linear, the identity alone does not make D_k a linear response matrix.
The fixed **joint linear** case below requires D_k itself to be a fixed linear
map as well; fixed linear Pi_k and B_k are sufficient. This is a mathematical
condition on the requested relation, not a choice of a numerical projector.

For affine, nonlinear, thresholded, coefficient-recomputing or relearned
procedures retain the full F_k expression and operation/state graph. A fixed
basis is insufficient to substitute F_k=A_k. Coefficient recomputation alone
neither proves nor disproves exact linearity; any claimed linear specialization
must establish it for the complete operator, centering, support and all
material state. In particular an affine offset cannot be silently discarded.

## Separate response families

Each record binds its query variable/source, input and output domains/units,
conditioning, operation counts, state/support generations, temporal extent,
upstream rerun boundary, approximation and unavailable causes. Temporal extent
and upstream scope are both material; similar-looking response arrays do not
identify the same family.

| Family | What is varied and included | What it does not establish |
| --- | --- | --- |
| RF-01 fixed one-step | One iteration with exact operators/support/state fixed; parent-input and accepted-model perturbations are separate queries | Full learned procedure, recursive or selected-terminal response |
| RF-02 one-step full procedure | One complete declared iteration including any model, coefficients, support, cleaner, branch and state changes caused by the query | An unchanged-branch derivative at a crossing or an unmeasured whole-chain response |
| RF-03 recursive multi-iteration | Declared input rule and state/model recursion across specified iterations, including dependence carried through S_k | Independent repetition of RF-01 or a terminal selector omitted from the query |
| RF-04 selected terminal | Exact predeclared stop/terminal-selection procedure, inspected iterations, selected bundle and selection changes | Response of one convenient fixed iteration presented as terminal response |
| RF-05 FRUIT full procedure with upstream parent held fixed | Exact FRUIT learning, recursion and requested termination/selection extent under the declared FRUIT-side query, holding the exact realized upstream parent and upstream producer state fixed | A response that varies that held-fixed parent or reruns upstream data-dependent producers |
| RF-06 whole chain | Every included upstream data-dependent operation plus the complete specified FRUIT procedure is rerun under an exact whole-chain query | Authority for an omitted/unavailable upstream operation or a generic whole-chain result from RF-05 |

In the exact fixed joint linear case,

\[
R_{z\leftarrow y,k}^{\rm FRUIT,fixed}=A_k,\qquad
R_{z\leftarrow m,k}^{\rm FRUIT,fixed}=D_k=B_k-A_k\Pi_k.
\]

The first holds the model fixed; the second holds the iteration input fixed.
They have different domains and scientific meanings. If only F_k is known
linear, the parent-input response may still be A_k under its fixed conditions,
but the model response requires the derivative of the actual D_k(m_k) where
it exists; no unavailable derivative is replaced by D_k by notation.

When model, coefficients, support, cleaner, stopping state or selection depends
on data, full-procedure response includes that dependence. Threshold crossings,
branch changes, support changes and terminal-selection changes may make the
derivative nonexistent. Record unavailable or a typed discontinuity/state
transition, with its exact scope; do not substitute the fixed-branch response.
Model bypass may reinsert a model component into a mode removed from the
measured path without observationally recovering that lost mode.

## Complete joint conditional covariance

Assume the exact fixed joint linear relation z_k=A_k y_k^in+D_k m_k in
compatible real vector representations, finite joint second moments, and
declared conditioning Omega_k^FRUIT fixing the relevant operators/support.
Define the FRUIT-qualified blocks by the joint conditional covariance of
(y_k^in,m_k). Then

\[
\begin{aligned}
C_z^{\rm FRUIT}={}&A_k C_{yy}^{\rm FRUIT} A_k^{\mathsf T}
 +D_k C_{mm}^{\rm FRUIT} D_k^{\mathsf T}\\
 &+A_k C_{ym}^{\rm FRUIT} D_k^{\mathsf T}
 +D_k C_{my}^{\rm FRUIT} A_k^{\mathsf T}.
\end{aligned}
\]

The blocks carry their exact quantity axes, units, domains and conditioning;
for a valid real joint covariance C_my is the transpose of C_ym. No model-parent
cross term is dropped without exact independence/conditioning authority that
establishes its absence. A model learned from this parent is not independent
by implication.

If m_k is held deterministic under the declared conditioning, its conditional
variance and cross blocks vanish by that conditioning, leaving the conditional
A_k C_yy A_k^T expression. This omits model-learning variation and supplies no
unconditional independence or covariance-completeness claim. It cannot erase
unknown model or cross uncertainty from another requested uncertainty target.
Nonlinear or random-operator procedures require their declared uncertainty
method; a Jacobian approximation is a separately typed approximation, not
the exact identity above.

## Uncertainty roles that remain separate

| Role | Required distinction |
| --- | --- |
| Measured-parent covariance | Producer-owned joint quantity/domain and dependence |
| Model/prior covariance | External or internally learned model origin, assumptions and omitted mismatch |
| Parent-model cross covariance | Joint dependence; absence requires authority, not convenience |
| Operator/state uncertainty | Variation excluded when those states are held fixed |
| Support/threshold selection | Changes of membership, branches and selection law |
| Stopping/terminal selection | Variation and dependence induced by the exact selector |
| Model mismatch and bias | Systematic target/model discrepancy; not a covariance estimate by identity |
| Empirical repeatability | Evidence about a declared repetition population; not interchangeable with any analytic covariance |
| NOI uncertainty | Exact approved ensemble/transform conditioning and member graph, with its own omissions |

Iterations sharing one parent are not independent observations. No uncertainty
falls as 1/sqrt(number of iterations) by implication.

A fixed-state NOI route applies the identical fixed FRUIT state and operator
graph, with exact application parity, to every compatible admitted realization.
Relearning model, support, cleaner, coefficients, stopping or terminal selection
per realization defines a separate NOI/FRUIT method and ensemble. Fixed and
relearned members cannot share that ensemble or be pooled by this core. An
NOI-informed later FRUIT state is a dependent successor generation, not
independent validation. The inherited NOI boundary supplies no numerical
FRUIT uncertainty route; no mixture authority is introduced here.
