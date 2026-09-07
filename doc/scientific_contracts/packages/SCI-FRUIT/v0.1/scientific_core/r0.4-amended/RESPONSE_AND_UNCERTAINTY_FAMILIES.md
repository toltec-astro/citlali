# SCI-FRUIT — typed composition, response and uncertainty r0.4 (amended)

Status: owner-directed Stage A candidate. Authority: repair direction
SCI-FRUIT-OD-STAGE-A-FINAL-AMENDMENT-2026-09-07, sections 1–3, 5–6.
Notation and domains are defined in [the type crosswalk](NOTATION_AND_ROLE_TAXONOMY.md).
These conditional formulas select no numerical FRUIT method or covariance.

## Typed remove/process/rejoin relation

For the declared ordinary additive structure, bind Y_k, M_k=M_k^applied and
Z_k, with Pi_k: M_k -> Y_k, F_k: Y_k -> Z_k and B_k: M_k -> Z_k.
Y_k and Z_k are exact additive quantity spaces, or explicitly identified
affine/quotient spaces with one declared reference and gauge for each operation
and well-defined arithmetic as specified in the type crosswalk. Each removal
and rejoin binds scientific quantity, unit and numerical scale, coordinate
frame/domain, grid/sampling, support, additive origin/reference or gauge,
null-space state, calibration convention and response convention.
Pi_k(m_k^applied) must be subtractable from y_k^in under those exact facts;
F_k(rho_k) and B_k(m_k^applied) must be addable in one compatible Z_k quantity
space. Every composition also binds its state and conditioning. Only then,

\[
\rho_k = y_k^{\rm in}-\Pi_k(m_k^{\rm applied}),\qquad
z_k = F_k(\rho_k)+B_k(m_k^{\rm applied}).
\]

M_k^candidate, M_k^accepted and M_k^applied and their objects remain distinct.
Their transformations, if any, are explicit; no equal domain, support, unit,
normalization, sign, representation, response, uncertainty or parentage is
assumed. Equality among any two is an exact method/state fact. Equality
m_k^applied = m_k^accepted requires an exact method/state equality. Every
normalization, clipping, sign conversion, support restriction, projection,
unit conversion or other intervening transformation must be explicit.

Changing a recomputation/learning rule, operator family, role order, sign,
bypass graph, support rule, recurrence, parent route, grouping, stopping,
terminal selection or failure behavior changes method identity. Different
coefficient values or learned/model states under an unchanged rule instead
have new state/iteration/application generations. Pi_k maps into Y_k and B_k
into Z_k; neither equality of those spaces nor equality or inverse relations
between the maps is assumed. The model need not share a
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

Use all response-role identities and typed statuses required by the applicable
method and claim. A required status may truthfully be unavailable; the catalog
does not imply universal numerical response availability. Its seven roles
have distinct perturbation boundaries and iteration/selection scopes below.
RF-07 is a composed outer FRUIT-only role, not an alias of RF-03.

Every role binds perturbation source and domain; fixed versus rerun state;
recurrence and terminal-selection scope; baseline/perturbed comparability;
branch/support/state changes; codomain; derivative or finite-difference
convention; and unavailable/discontinuity behavior. Bind direction/basis,
limit or finite step and normalization when relevant, units, operation counts,
generations and conditioning. Baseline and perturbed reference/gauge/null-space
states require exact compatible comparison; a changed reference is not silently
a signal perturbation. No numerical query or finite-difference step is selected.

| Role | Perturbed and held-fixed objects | Rerun or relearned scope | Domains, codomain, comparability and unavailable behavior |
| --- | --- | --- | --- |
| RF-01 fixed one-step measured-input response | Perturb exact y_k^in in Y_k; hold applied model and all declared state, maps and support fixed | Evaluate the unchanged one-step composition only; no learning, recurrence, stopping or terminal selection | Response maps declared Y_k perturbations to Z_k differences under the stated derivative/finite-difference convention. Reference/gauge/null-space and branch/support/state compatibility are required, otherwise unavailable or a typed discontinuity |
| RF-02 fixed one-step applied-model response | Perturb m_k^applied in M_k^applied; hold measured iteration input and all declared state, maps and support fixed | Evaluate the unchanged one-step composition only; no acceptance/model learning, recurrence or terminal selection | Response maps declared applied-model perturbations to Z_k differences under the stated convention; fixed reference/gauge/null-space and branch/support/state comparison is required, otherwise unavailable or a typed discontinuity |
| RF-03 one-iteration FRUIT full-procedure response | Perturb the declared one-iteration input on its exact domain; hold prior FRUIT state S_k and upstream producer procedure/state fixed | Rerun model construction, selection, state resolution, support, coefficient realization and application for iteration k only; no earlier recurrence or terminal selector | Bind baseline/perturbed input domain and iteration-result Z_k response codomain and convention. Include within-iteration branch/support/state changes; incompatible reference/gauge/null-space comparisons or nonexistent derivatives are unavailable/typed transitions |
| RF-04 fixed-sequence recursive response | Perturb the declared recursion input; hold method rules, initial state, upstream producer procedure/state and predeclared iteration sequence or absolute endpoint index fixed | Propagate input/model/state dependence through that exact sequence of iterations; terminal selection and data-dependent sequence length are outside this role | Bind recursion-input domain, exact sequence/index, result/sequence codomain and convention. Include branch/support/state changes within fixed extent; incompatible comparisons, unavailable sequence results or nonexistent derivatives retain typed failure/discontinuity |
| RF-05 selected-terminal response | Perturb the declared selection/procedure input; identify all fixed initial and upstream facts | Include consequential iterations plus convergence, stopping, resource-limit and terminal-selection behavior, with complete candidate iteration population and selection causes | Bind input and selected-result codomain, derivative/finite-difference convention and terminal-record identity. Include branch/support/state/selection changes and reference/gauge/null-space comparability; otherwise unavailable or a typed discontinuity/selection transition |
| RF-06 whole-chain response | Perturb the exact upstream/source query on its declared domain; name held-fixed external facts | Rerun every included data-dependent upstream producer and the declared FRUIT procedure, including its recurrence and terminal-selection scope | Bind upstream query and final response domains and convention. Every included producer, branch/support/state/selection comparison and reference/gauge/null-space consequence must be available, otherwise retain typed unavailability/discontinuity |
| RF-07 complete FRUIT-only parent-domain response | Perturb the realized parent-domain payload; hold upstream producer operation, learned state and generation fixed, not the numerical query payload | Compose the declared parent-domain entry with the complete FRUIT procedure, including recurrence and terminal selection as applicable; no upstream producer rerun | Bind parent-query domain, final FRUIT response codomain and convention, initial FRUIT conditions, RF-03/RF-04/RF-05 component references, branch/support/state/selection comparisons and reference/gauge/null-space consequences; unavailable dependencies or nonexistent derivatives retain typed unavailability/discontinuity |

RF-03 supplies the local one-iteration relation with prior S_k fixed. RF-04
propagates those relations through a predeclared fixed sequence/index. RF-05
includes convergence, stopping, resource limits and terminal selection rather
than fixing the selected endpoint. RF-07 binds the outer realized-parent-domain
perturbation to the complete FRUIT-only composition: it references RF-03
iteration components and RF-04 fixed-extent or RF-05 selected-terminal scope
as applicable. It does not rerun an upstream producer; RF-06 does.

The role and composition references are part of response identity. Do not
publish the RF-07 composition as an independent alias of one component. If a
method's one-iteration special case reduces to a component relation, record
that exact specialization and reuse its bound response/status. The catalog
distinguishes the query boundaries even when a special case has equal values.
Any undefined numerical comparison remains unavailable.

Perturbed parent payloads are separately identified response-query objects
with immutable baseline-parent ancestry. Holding upstream producer generation
fixed describes the producer state used by the query; it does not claim a new
upstream realization or mutate the original measured parent.

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
If an upstream absolute or other additive mode is unavailable, the rejoined
mode remains `model_or_prior_supported`; it is not `observationally_measured`,
`recovered_from_data`, `acquired_exposure` or `unit_response_truth`.
The response record must retain that null-space and reference/gauge consequence.

## Complete joint conditional covariance

Assume A_k, Pi_k and B_k are all exact fixed linear maps, hence the relation
z_k=A_k y_k^in+D_k m_k^applied, in compatible real vector representations
with declared domains/codomains, units and support, finite joint second moments,
and conditioning Omega_k^FRUIT fixing the relevant operators and state.
The additive reference/gauge and null-space state are part of this conditioning
and of the exact coordinates/quotient space in which covariance is defined.
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

For each uncertainty role below, bind additive-reference/gauge assumptions,
unavailable modes, model/prior support in those modes and any omitted reference
or mode variation. A fixed gauge or singular covariance does not establish
that an unavailable absolute mode was measured with zero uncertainty. No
parent/model cross term or unavailable-mode consequence may be discarded by
choosing a reference. Selected-terminal uncertainty also binds the complete
selection record, candidate population, stop/resource facts and causes.

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
