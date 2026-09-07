# SCI-FRUIT — typed composition, response and uncertainty r0.4

Status: owner-directed Stage A candidate. Authority: repair direction
SCI-FRUIT-OD-STAGE-A-MICRO-REPAIR-2026-09-07, sections 1–3, 5–8.
Notation and domains are defined in [the type crosswalk](NOTATION_AND_ROLE_TAXONOMY.md).
These conditional formulas select no numerical FRUIT method or covariance.

## Typed remove/process/rejoin relation

For the declared ordinary additive structure, bind Y_k, M_k=M_k^applied and
Z_k, with Pi_k: M_k -> Y_k, F_k: Y_k -> Z_k and B_k: M_k -> Z_k.
Every composition binds domains, codomains, units, support, state and
conditioning. On compatible domains and supports,

\[
\rho_k = y_k^{\rm in}-\Pi_k(m_k^{\rm applied}),\qquad
z_k = F_k(\rho_k)+B_k(m_k^{\rm applied}).
\]

Candidate, accepted and applied model objects remain distinct. Equality
m_k^applied = m_k^accepted requires an exact method/state equality. Every
normalization, clipping, sign conversion, support restriction, projection,
unit conversion or other intervening transformation must be explicit.

Changing a recomputation/learning rule, operator family, role order, sign,
bypass graph, support rule, recurrence, parent route, grouping, stopping,
terminal selection or failure behavior changes method identity. Different
coefficient values or learned/model states under an unchanged rule instead
have new state/iteration/application generations. Pi_k and B_k have different
codomains; they are not assumed equal or inverse. The model need not share a
domain with the measured signal. F_k is the complete residual-processing
procedure; it need not be linear. Removal/rejoin support need not agree, and
model truth or unit response is not assumed.

Only when F_k is established as the **exact fixed linear operator** A_k on
the declared input/result spaces does distributivity give

\[
z_k=A_k y_k^{\rm in}
    -A_k\!\left[\Pi_k(m_k^{\rm applied})\right]
    +B_k(m_k^{\rm applied}).
\]

This is a map-valued expression. If Pi_k or B_k is nonlinear, retain it and
do not infer a linear model response or covariance formula. Only when A_k,
Pi_k and B_k are all declared exact fixed linear maps on the stated spaces,
units, support, state and conditioning define

\[
D_k:=B_k-A_k\circ\Pi_k,\qquad
z_k=A_k y_k^{\rm in}+D_k m_k^{\rm applied}.
\]

These are the complete linear premises used below. No linearity or numerical
projector follows merely from mapping signatures or a matrix-like symbol.

For affine, nonlinear, thresholded, coefficient-recomputing or relearned
procedures retain the full F_k expression and operation/state graph. A fixed
basis is insufficient to substitute F_k=A_k. Coefficient recomputation alone
neither proves nor disproves exact linearity; any claimed linear specialization
must establish it for the complete operator, centering, support and all
material state. In particular an affine offset cannot be silently discarded.

## Seven response roles

There are exactly seven roles: the six requested distinct roles plus the
retained FRUIT-full-procedure role with the upstream parent held fixed.
Each record must bind what is perturbed, what is held fixed, what is rerun or
relearned, baseline and perturbed domains, branch/support/selection comparability,
response codomain, and unavailable/discontinuity behavior. It also binds units,
operation counts, temporal extent, state generations, conditioning and any
approximation. The table supplies role boundaries, not numerical queries.

| Role | Perturbed and held-fixed objects | Rerun or relearned scope | Domains, codomain, comparability and unavailable behavior |
| --- | --- | --- | --- |
| RF-01 fixed one-step measured-input response | Perturb exact y_k^in; hold m_k^applied, maps, support and declared conditioning state fixed | Apply the same one-step composition; no model learning or acceptance-to-application change | Baseline/perturbed input in declared compatible Y_k; output response in Z_k. Fixed branch/support/selection is required; incompatible queries or nonexistent derivatives are unavailable, with transitions typed |
| RF-02 fixed one-step model-input response | Perturb m_k^applied; hold y_k^in, maps, support and declared conditioning state fixed | Apply the same removal/rejoin maps and residual procedure; do not substitute an accepted-model perturbation | Baseline/perturbed applied model in compatible M_k; response in Z_k. Fixed branch/support/selection is required; incompatible queries or nonexistent derivatives are unavailable, with transitions typed |
| RF-03 one-step FRUIT full-procedure response | Perturb declared FRUIT input/model/control; name all fixed upstream and seed facts | Rerun one complete iteration, including consequential candidate/acceptance/application transformations, coefficients, support, cleaner and state learning | Bind baseline/perturbed query spaces and iteration-result response codomain, including any comparison map; branch/support/selection changes are included, and nonexistent derivatives become unavailable or typed transitions |
| RF-04 recursive multi-iteration response | Perturb declared initial input/model/control; name fixed base and initial-state facts | Rerun the exact input rule and all consequential transitions, learning and application changes across the specified iterations | Bind baseline/perturbed query domains, iteration indices and sequence/result response codomain; require declared cross-iteration and branch/support/selection comparisons or mark response unavailable/transition |
| RF-05 selected-terminal response | Perturb the declared procedure input/control; name fixed parents and initial conditions | Rerun required iterations, convergence/stopping and the predeclared terminal selector, including all inspected iterations | Bind baseline/perturbed query domains and selected-result response codomain; selected iteration/product changes require explicit comparability, otherwise unavailable or typed discontinuity/selection transition |
| RF-06 whole-chain response | Perturb the exact upstream/source query; name all held-fixed external facts | Rerun every included upstream data-dependent operation and the declared FRUIT procedure, learning, stopping and selection | Bind upstream baseline/perturbed domains and requested final response codomain; every included branch/support/selection comparison and upstream authority is required, otherwise unavailable or typed transition |
| RF-07 FRUIT full-procedure response with upstream parent held fixed | Perturb only the declared FRUIT-side model/input/control that remains variable; hold the exact realized upstream parent and its producer state fixed | Rerun consequential FRUIT learning, recursion and requested stopping/selection extent; no upstream producer rerun | Bind baseline/perturbed FRUIT query spaces and requested result response codomain; include branch/support/selection changes. Varying the held-fixed parent is outside this role; incompatible comparisons or nonexistent derivatives are unavailable/typed transitions |

For RF-01, measured-input names the iteration input path; it does not assert
that y_k^in is the original measured parent or a new independent measurement.
Fixed in RF-01/RF-02 refers to the declared conditioned map and its unchanged
application rule. Coefficients that this rule recomputes from the perturbed
operand must still be recomputed; freezing their numerical values instead
changes the procedure. Record new coefficient/application states without
changing the unchanged method's identity.

Every query identifies the baseline and perturbed method/state bindings.
A control change that changes a method-defining rule requires a new method
identity and an explicit comparison of methods; it cannot masquerade as a
state perturbation under one unchanged method.

For every role, a missing compatible response codomain or comparison between
baseline and perturbed domains makes that response unavailable. No fixed-branch
derivative may replace a nonexistent full-procedure derivative. An online
adaptive filter remains a separately authored estimator; none is introduced
by these response names.

Under the complete fixed linear premises for A_k, Pi_k and B_k,

\[
R_{z\leftarrow y,k}^{\rm FRUIT,fixed}=A_k,\qquad
R_{z\leftarrow m^{\rm applied},k}^{\rm FRUIT,fixed}
 =D_k=B_k-A_k\circ\Pi_k.
\]

The first holds the applied model fixed; the second holds the iteration input fixed.
They have different domains and scientific meanings. If only F_k is known
linear, the parent-input response may still be A_k under its fixed conditions,
but a nonlinear model path requires the derivative of the complete map-valued
expression where it exists. It is not represented by D_k. A perturbation of
m_k^candidate or m_k^accepted additionally includes the declared selection or
acceptance-to-application transformation in the corresponding full procedure.

When model, coefficients, support, cleaner, stopping state or selection depends
on data, full-procedure response includes that dependence. Threshold crossings,
branch changes, support changes and terminal-selection changes may make the
derivative nonexistent. Record unavailable or a typed discontinuity/state
transition, with its exact scope; do not substitute the fixed-branch response.
Model bypass may reinsert a model component into a mode removed from the
measured path without observationally recovering that lost mode.

## Complete joint conditional covariance

Assume A_k, Pi_k and B_k are all exact fixed linear maps, hence the relation
z_k=A_k y_k^in+D_k m_k^applied, in compatible real vector representations
with declared domains/codomains, units and support, finite joint second moments,
and conditioning Omega_k^FRUIT fixing the relevant operators and state.
Define the FRUIT-qualified blocks by the joint conditional covariance of
(y_k^in,m_k^applied). In these block labels, m denotes the applied operand
only; it does not alias candidate or accepted models. Then

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

Candidate or accepted model covariance cannot be substituted for applied-model
covariance without the exact acceptance-to-application transformation and its
dependence. Those transformations are included when the uncertainty target
is the full procedure.

If m_k^applied is held deterministic under the declared conditioning, its conditional
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
| Model/prior covariance | Separate candidate/accepted/applied roles, origin and assumptions; the four-term relation uses applied-model covariance after all declared transformations |
| Parent-model cross covariance | Joint measured-input/applied-model dependence for the relation above; absence requires authority, not convenience |
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
Changing from fixed-state application to a rule that relearns model, support,
cleaner, coefficients, stopping or terminal selection per realization defines
a separate NOI/FRUIT method and ensemble. Members generated under that one
unchanged relearning rule have distinct member/state/application generations,
not a new method identity for each member. Fixed and
relearned members cannot share that ensemble or be pooled by this core. An
NOI-informed later FRUIT state is a dependent successor generation, not
independent validation. The inherited NOI boundary supplies no numerical
FRUIT uncertainty route; no mixture authority is introduced here.

Truthful response and uncertainty status is a universal bundle requirement.
An unavailable numerical companion can accompany only a narrower claim
permitted by both the core and exact method; neither may be silently omitted
or replaced by zero, identity or a claim of successful bypass.
