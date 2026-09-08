# Concrete ordinary-MAP feedback method proposal

Status: proposed rules, not numerical approval. Proposed instance identity:
`FRUIT-FEEDBACK-METHOD/ordinary-map-fixed-bootstrap@r0.1`.
This supersedes the earlier discussion name only as the proposed method name.
Scope D001–D005 is approved; M01–M09 below are new concrete method choices for
owner disposition. Q01–Q06 in the [question record](DISPOSITIONS_AND_OWNER_QUESTIONS.md)
identify actual missing decisions/inputs. No value below is inherited from
historical configuration.

## M01. Request and immutable identities

The request binds one original observation, a nonempty explicit set of required
array identities, one fixed MAP WCS/grid per array, the exact upstream
RTC/CAL plan and source/calibration/beam lineage, and one immutable PTC segment
partition. Use one array-wide PTC group per segment; cross-array fitting and
coaddition are excluded. Arrays are separately identified, not statistically
independent. Upstream producers are rerun from the same original measured
parent each iteration with the same requested controls. FRUIT supplies no
upstream masks, detector exclusions or learned corrections.

At each crossing bind stable detector occurrence/UID and RTC sample n, segment,
array, exact producer/application generation, quantity/scale, support/validity,
frame/WCS and reference/gauge. Repeated computation gets a new realized record
linked to the original ancestry. It must preserve the declared bootstrap
occurrence/geometry/reference domain under explicit compatibility; row or
numerical equality is insufficient. Domain or required producer-state drift
fails the affected array request, without a new grid, resized group or fallback.

The calibrated pre-removal product is specifically the admitted SCI-CAL x
product after RTC/CAL, in top-of-atmosphere point-source-equivalent mJy per
originating fixed nominal beam. It is neither bare RTC nor previous PTC output.
Q03 must supply the exact positive-rank PTC plan; Q04 the MAP support scalar;
Q06 the later concrete request bindings. Those required inputs have no defaults.

## M02. Bootstrap, model and causal selector

Absolute iteration k=0 is the unseeded bootstrap. Rerun the original upstream
parent, fit ordinary PTC once per declared group/segment on the admitted CAL
product, apply it once, and request a complete unfiltered normalized ordinary
MAP array bundle. There is no predecessor or externally supplied model.

The bootstrap is a declared narrower no-feedback composition. Candidate,
accepted and applied model roles, removal and rejoin roles remain separately
recorded as not_applicable with reason `bootstrap_no_predecessor`; no model
operator is invoked. The PTC input remains the exact admitted CAL product.
The feedback-residual role is not_applicable, while the PTC output and
MAP input are the actual producer products. This uses the frozen core's
explicit narrower-composition/absent-role provision, not a zero substitute or
a waiver of universal role records. A missing predecessor at k>=1 is a failure,
never another bootstrap. This bootstrap rule requires method approval in Q01.

A successful bootstrap array bundle Q_(0,a) is the first eligible feedback
parent. Define S_a as exactly its MAP-valid, science-policy-supported normalized
output rows under the declared per-array effective support policy. That policy
publishes those science rows; mere retained storage or normalization-only
support is insufficient. Require S_a nonempty. Freeze this row identity and the
bootstrap MAP contributing occurrences I_MAP,a for the finite request.
I_MAP,a includes the occurrences used to form the full normalization/support
population; T_a below restricts its output rows to S_a. Contributions to rows
outside S_a do not become feedback-model values or published science rows.
All required arrays must complete bootstrap before any advances to k=1.

At k>=1, the candidate model is a typed view of Q_(k-1,a)'s signal on exactly
S_a. Accept every finite value on S_a, of either sign. Selection uses only this
preceding completed bundle, its immutable support/validity, and required
identity/reference compatibility, all available before removal. There is no
flux threshold, S/N, RMS, noise/kernel companion, peak fraction, source finding,
clipping, damping or stochastic selection. It protects noise and artifacts on
that support as well as sources; acceptance is feedback permission only.

The accepted model is an immutable identity copy of the candidate values and
support. The applied model is a separately identified immutable identity copy
of the accepted object. Record both identity transformations, their exact
quantity/reference/grid/domain and their shared-parent dependence. No hidden
renormalization or sign/unit conversion occurs. Any required non-finite value
or incompatibility fails the array before applying a model; it is not discarded
from the fixed model support. The next completed map replaces the preceding
map wholesale on S_a. It is not an increment or a sum of preceding residuals.

## M03. Proposed projection, rejoin and normalized-MAP round trip

Propose a piecewise-constant model-to-sample projection using the exact same
AST signal-coordinate association, MAP WCS and half-open containing-pixel rule
as frozen MAP. Do not use nearest-center rounding, interpolation, extrapolation,
clamping, filtering or a historical map sampler. On a sample whose containing
pixel p lies in S_a, the removal contribution is the applied value m_(k,a,p)
with numerical scale one and no additive offset. A bounded correction-domain
embedding must authorize that arithmetic; matching units alone does not.

The same declared value is restored after the residual PTC operation on the
same required model-path sample identities. Removal uses subtraction; rejoin
uses addition. Proposed equality of these numerical projections is an explicit
rule on their common declared support, not a generic Pi=B assertion across
different spaces. The residual-only branch outside S_a passes the original
CAL sample into PTC and has no model rejoin; its path records say absent by
method, not a fabricated zero-valued model. Such samples may still participate
in the fixed PTC application; they are not promoted to final MAP rows outside
S_a. Out-of-grid or coordinate-unavailable samples have no invented pixel.
Their PTC support and cause remain governed separately by the PTC plan.

Let T_a be the exact normalized MAP operator with fixed I_MAP,a, coefficients,
placement, normalization and row restriction to S_a. Every sample contributing
to a final row in S_a receives its containing applied-model pixel at rejoin. Therefore,
conditionally on finite compatible arithmetic and complete restoration,

\[
(T_a P_a^+m)_p=m_p,\qquad
\widehat s_{k,a,p}=m_{k,a,p}+(T_a r_k^{\rm clean})_p,
\quad p\in S_a.
\]

This is the frozen MAP normalization specialized to the proposed projection,
not approval of that projector or a new generic FRUIT equation. P_a^+ is the
rejoin projection; normalized T_a is not MAP's unnormalized placement G. The
identity is restricted to S_a and the applied model. It neither establishes
unit physical-sky response nor restores unmeasured modes. Post-rejoin signal
processing beyond the declared MAP operation is prohibited in this candidate.
PTC's fixed centering makes its full application affine; do not drop its offset
or call the whole procedure fixed linear. The owner must approve the exact
additive/reference embedding in Q02; otherwise both arithmetic and this
specialization remain unavailable.

## M04. Residual-operation graph and state

For k>=1 the proposed graph is:

    original measured parent -> RTC -> CAL -> model subtraction
      -> frozen-bootstrap PTC application -> model rejoin
      -> coefficient compatibility/QC -> ordinary MAP -> completion

No numerical residual PTC route is currently admitted: BC-IN in the
[boundary proposal](PTC_MAP_BOUNDARIES.md) must be separately authorized.
Fit PTC only at k=0 on CAL. Carry the complete resolved Theta_(0,a,g), including
its centering location, fit/loading/application supports, metric, subspace,
positive rank, normal-matrix/rank tolerance and failure rules. At k>=1 evaluate
the exact frozen PTC application on the FRUIT residual; recompute its internal
modal application coordinates on that operand. Do not relearn the basis,
recenter on the residual, change rank/support, perform a support-changing refit,
or subtract a frozen numerical removed component. Do not restore PTC's
published discarded centering reference at rejoin. Restoring the feedback
model does not restore that reference or measured optical DC.

| Operation/component | State category and schedule | Path/count and population |
| --- | --- | --- |
| RTC/CAL products, AST coordinates, calibration, segment plan | externally supplied each k under fixed requested controls and exact compatibility | Rerun once from original ancestry; before model subtraction; no FRUIT learning enters them. |
| PTC requested grouping/rank/metric/support plan | externally supplied; Q03 exact binding required | One array-wide group per immutable segment; no automatic candidate selection. |
| PTC fit/centering/subspace at bootstrap | relearned category, first and only learning event | One ordinary fit on admitted CAL per group at k=0; no model path then. |
| Complete resolved PTC map at k>=1 | fixed numerical operator at that conditioning scope; affine state retained | One residual application per group; internal application coordinates reevaluated, not a new defining map. |
| Applied model value and copy relations | externally supplied from the preceding completed array bundle to this iteration | One deterministic candidate/acceptance/application construction before removal; dependent, not external sky truth. |
| Selector support, projection and rejoin maps | fixed numerical operators after bootstrap compatibility | Same S_a/geometry; once on each applicable model path. No amplitude selection. |
| PTC action on the model path | bypassed | Removed before the named residual PTC application; rejoins immediately afterward. RTC/CAL and MAP are not bypassed. |
| MAP coefficients and placement/normalization | fixed numerical operators after the separately approved bootstrap family and support binding | Exact retained coefficient values; T_a fixed only while every required membership/QC/finite guard passes. |
| Per-iteration coefficient compatibility/QC | fixed rule evaluated against current immutable records | New QC/evaluation record per generation; does not reestimate coefficient values. |
| Support-changing learning, post-rejoin flagging/weight estimation, extra filtering | not_applicable by this method | Zero applications; not optional historical defaults. |
| Missing required permission/profile/state | unavailable | Prevent the affected invocation/completion; no fallback stage is called. |

No component uses the core category “fixed structural state with recomputed
operator-defining coefficients” in this first candidate. A method that activates
that category, residual basis learning or a post-rejoin coefficient estimator
needs a new method identity. Internal PTC application-coordinate evaluation is
not an eighth state category. Numerical response conditioning follows the exact
frozen map, not the word coefficient in an implementation.

## M05. Concrete coefficient and QC proposal

Request a separately PTC-owned family provisionally named
`PTC-ANALYSIS/UNIFORM-OCCURRENCE@proposal-r0.1`: dimensionless gamma_i=1 on its
explicitly eligible bootstrap PTC occurrence population I_Gamma,0,a. It is an
analysis/gridding coefficient, not inverse variance, sensitivity, precision or
an uncertainty estimate. Its constant definition has no fit statistic or
empirical normalization. This is an explicit proposed family/value requiring
Q02 approval and controlled PTC/source bindings; it is never a unity fallback.

I_Gamma,0,a consists of exact bootstrap PTC output-retained occurrences meeting
the proposed family identity, finite transformed-signal, valid occurrence and
compatible product/unit/group constraints. It is fixed before first MAP
formation. MAP still separately applies its full frozen contribution/admission
rules; I_MAP,a is the resulting subset. Re-estimation on the residual or rejoined
signal is not requested. Retain the exact bootstrap coefficient object and
value generation for all later k, with a new explicit compatibility reference
to each residual PTC output and rejoined product and a fresh QC decision. Do not
copy an earlier eligibility result as the current decision.

| Record | Signal/population and generation | Method disposition |
| --- | --- | --- |
| Bootstrap gamma_0 | Exact Z_PTC,0 on I_Gamma,0,a, its output/retention generation and family request | Constant value definition; no residual-noise estimator. Missing family permission produces no coefficient. |
| Bootstrap QC_0 | Same Z_PTC,0, occurrence IDs, producer retention, finite-payload and declared family checks | Explicit PTC-owned request/applicability/eligibility/realization; no MAP-use inference. |
| Residual-derived gamma_k | Would use residual output R_PTC,k and its generation | not_requested; distinct future estimator, never an alias for gamma_0. |
| Post-rejoin-recomputed gamma_k | Would use rejoined Z_FRUIT,k and its generation | not_requested; distinct future estimator. |
| Retained gamma_(0->k) | Exact unchanged gamma_0 value generation plus explicit compatibility to current parents | Proposed retained use, conditional on BC-OUT permission and all identities/supports matching. |
| QC_k and MAP admission_k | Current residual/rejoined product identities, unchanged gamma_0, current producer retention/causes, finite compatible payload and exact same occurrence domain | New evaluations under separately approved profiles; any required nonpass fails fixed-population application, not silently shrinking I_MAP,a. |

The proposed QC declares finite/compatible output and family predicates; it
authors no physical-noise or science-fitness judgment. MAP owns its separate
finite-positive coefficient classification and contribution decision. Q02 must
supply owner-approved family/QC/source-binding objects before numerical use.

## M06. Support, flags and failure scope

Retain eight distinct support records: original measured-parent and producer
support carried by CAL; each model stage's
S_a; projection/removal sample support; PTC support by group/operation; rejoin
support; final MAP S_a and I_MAP,a; bootstrap learning influence; and each
response query's declared support/status. Equality facts are recorded where
this candidate requires them; a common stored mask does not replace the roles.

After bootstrap, freeze PTC required input/application occurrence domains,
I_MAP,a, S_a, WCS/association and coefficient values. Preserve all upstream
flags/causes without rewriting them; do not generate adaptive FRUIT flags.
Current valid/finite/identity/QC checks must pass for every required occurrence
and output row. No exclusion, membership growth, changed support threshold,
borrowed row, lower-rank solve or renormalization rescues a failure.

Model-defined plus residual-supported occurrences take both paths. Occurrences
outside S_a may take the explicitly permitted residual-only branch. Model-only
final contributions are prohibited. A missing required path value is unavailable,
not zero. A failed required PTC group-time, coefficient/QC record or model
projection fails the affected array iteration and required array run; inherited
ordinary out-of-support/no-contribution at bootstrap is not itself failure.
An empty required array science domain fails bootstrap. Inactive/not-requested
arrays stay not_requested; no placeholder or spontaneous smaller request.

At each k the observation-level request advances only after all requested array
iterations complete. Any required array failure makes that observation request
failed and prevents starting k+1. Already completed array/iteration products
remain immutable with their actual completion states; no rollback or relabeling
is required. Pending/not-produced work retains the stage reached. Required
publication failure also fails its owning array and larger request.

## M07. Extent, terminal selection, persistence and resources

Use an explicitly supplied finite integer L>=2: L counts all required iterations,
including bootstrap k=0; feedback runs at k=1,...,L-1. L is not selected here.
The same declared extent applies to all required arrays. There is no data-driven
stop, convergence test or best-looking-map search. After successful completion
of every required array at every required k, select Q_(L-1,a) for each array and
its exact completed FRUIT bundle. The observation terminal record binds that
full set. Earlier k are retained candidates excluded by the declared extent
rule, not ranked by map appearance. A failed required run has no successful
observation terminal product; earlier completed arrays/maps remain usable only
under their actual separate status and any separately requested use.

Record all nine frozen terminal fields. No cross-array/statistical averaging or
cross-method conversion is part of selection. Bind common per-array quantity,
scale, WCS, reference convention, S_a and response/uncertainty meaning across k;
exact changed state/generations and parents remain visible. No response-based
ranking is used. If required comparability fails, terminal selection fails.

Retain immutable per-array complete bundles for every completed k, bootstrap
Theta/gamma/support, source and current compatibility/QC records, exact applied
models/path reconstruction rules, failure attempts and terminal record. Do not
require a particular file format. This finite request claims no restart or
continuation capability; its recorded causal state obligations are not waived,
but saved-state presence does not prove continuation sufficiency.

Before any execution Q06 must bind wall-time and memory limits and accounting.
Propose whole-request elapsed time from first upstream processing through
required terminal publication, plus per-stage elapsed time and peak aggregate
resident memory of the owned process tree, including workers and model/map
storage. State platform/counter identity and unavailable measurements. Exhaustion
marks the required run failed/resource-exhausted, never converged. No limits,
executable, observation, measurement or experiment are supplied here.

## M08. Response, uncertainty and claims

The first requested product is the complete central iteration/terminal bundle
with truthful response and uncertainty states; no numerical response or NOI
ensemble product is requested by this paper deliverable. Preserve all seven
core response roles, all thirteen uncertainty roles and separate mismatch,
transfer, shift and bias meanings. Numerical instances cannot suppress required
role records or hard claim dependencies.

RF-01/02 hold complete numerical maps/defining state fixed and reevaluate PTC
application coordinates; retain fixed affine centering. RF-03/04/05 preserve
single-iteration, sequence and terminal-conditioning distinctions; bootstrap
learning and later held state have explicit scopes. RF-06 needs exact upstream
response/rerun authority; RF-07 starts at its declared fixed upstream parent
domain. Use the frozen catalog without new families or inferred derivatives.
Until exact maps, query/perturbation bindings and realized evidence exist,
response records say unavailable with the exact missing binding, not identity.
A support/finite/rank transition is not silently treated as a derivative.

Uncertainty is unavailable where its exact law/input is absent. A future
physical-parent law must retain shared model/parent and cross-array dependence;
a fixed-realized-model law is a different target. Uniform gamma does not make
covariance diagonal or variance unity. No NOI law or member transformation is
selected, and no companion is needed by this selector. Preserve null/reference
and model-supported-mode disclosures even in a complete central product.
No source recovery, pointing accuracy, uncertainty coverage, convergence,
readiness or policy recommendation follows from this definition.

## M09. Twenty-one-class disposition

| Class | Concrete disposition / remaining blocker |
| --- | --- |
| 01 Authority/lineage | Exact proposed method name above; approved scope control is separate. Q01 method approval and Q02 boundary/family authorities remain required. |
| 02 Route/grouping | M01 and BC01–BC07: CAL→FRUIT residual→fixed-bootstrap PTC→FRUIT rejoin→ordinary MAP; one array-wide group per immutable segment; separately identified arrays. Current numerical route blocked. |
| 03 Input rule | M01: original measured-parent RTC/CAL rerun every k; exact compatible pre-removal CAL product, never prior residual. Q03/Q06 bind producer realization. |
| 04 Target/model | M02: preceding complete MAP values on fixed science S_a; candidate/accepted/applied roles with explicit identity-copy relations, feedback use only. |
| 05 Construction/selection/application | M02–M03: support/validity/finite-only causal selector, both signs, no companions; piecewise-constant projections proposed. Q01/Q02 govern exact method and additive compatibility. |
| 06 Accumulation | M02: complete-map replacement, no increment summation. |
| 07 Typed composition | M03–M04 and boundary table: explicit domains, signs, affine PTC and named rejoin; numerical identity embedding requires BC-IN/BC-OUT permission. |
| 08 Additive/unavailable modes | M03/BC02/BC06: scale one/no offset proposed only under exact correction-domain and reference compatibility. Missing measured modes remain model-supported/unavailable, not restored truth. |
| 09 Operation graph | M04: upstream rerun, removal, one fixed PTC application, rejoin, compatibility/QC, MAP, completion; no other residual or post-rejoin signal operation. |
| 10 State by operation | M04 table: bootstrap fit then held defining state, internal modal coordinate reevaluation; no adaptive learning or operator-defining coefficient recomputation. |
| 11 Support/missing | M06: eight records; fixed post-bootstrap required populations and row identities; explicit residual-only branch, no model-only final contribution or fallback. |
| 12 Update/transition | M02/M06: next complete map plus exact carried Theta_0/gamma_0/support; all-array barrier and failure rule. |
| 13 Response | M08: seven catalog roles retained with fixed/rerun conditioning; numerical queries unavailable without exact bindings, never fabricated identity. |
| 14 Uncertainty | M08: all roles/disclosures; no numerical NOI request or independent-noise inference; hard dependencies preserved. |
| 15 Continuation | M07: not requested/claimed for this finite method request; causal state retained, sufficiency remains unestablished. |
| 16 Stochastic state | FRUIT selection/update/terminal rules are deterministic; no randomization, seed or ensemble. Any upstream stochastic fit must be declared in Q03 or it is incompatible with this proposed deterministic plan. |
| 17 Convergence | M07: no convergence proposition requested; record unassessed and finite extent, not convergence. |
| 18 Resources | M07: whole-request elapsed/stage time and aggregate process-tree memory with exhaustion failure; exact limits/counters require Q06. |
| 19 Terminal selection | M07: after success only, final required completed k=L-1 for every requested array; no earlier-product rescue. |
| 20 Persistence | M07: immutable complete products and causal state/reconstruction, all attempts; no storage format or continuation proof selected. |
| 21 Completion/failure/publication | M02/M06/M07: narrower no-feedback bootstrap with explicit absent-role records; universal bundle at every k; array-local failure plus required-request failure; no fabricated later lifecycle stages or downstream fitness. |

Q05 is a possible later response/uncertainty/pointing request, explicitly
deferred and not a blocker for paper definition or a narrower central bundle
once its actual method/upstream gates are met. Q06 is an execution gate, not
permission to design an experiment in this pass.
