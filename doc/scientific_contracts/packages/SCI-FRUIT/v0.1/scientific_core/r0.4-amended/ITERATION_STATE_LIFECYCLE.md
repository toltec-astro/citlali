# SCI-FRUIT — iteration, operation state and lifecycle r0.4 (amended)

Status: owner-directed Stage A candidate. Authority: repair direction
SCI-FRUIT-OD-STAGE-A-FINAL-AMENDMENT-2026-09-07, sections 1, 3–5;
inherited lifecycle and continuation obligations remain in force.

## State and identity

S_0 is the initial seed state. The generic iteration transition is

\[
(S_k,y_k^{\rm in})\longrightarrow
(m_k^{\rm candidate},m_k^{\rm accepted},m_k^{\rm applied},
 \rho_k,z_k,S_{k+1}).
\]

Candidate, accepted and applied models are separate objects in the
[declared spaces](NOTATION_AND_ROLE_TAXONOMY.md). The composition consumes
m_k^applied. Equality to m_k^accepted requires an exact method/state equality;
every intervening transformation is explicit. A role that does not occur has
a core-permitted typed not_applicable reason; a missing required fact has an
unavailable cause. The method defines construction and the transition, so
this convention chooses no update law. After completion of the iteration with
absolute zero-based index N, continuation begins at absolute index N+1 from
S_(N+1). N is an index, not a count of completed iterations.

Every iteration binds at least `lineage_id`, `branch_id`, absolute zero-based
iteration index, method-specific identity and generation, parent generation,
state generation and exact application generation. Method identity/generation
is not an alias for realized state, iteration or application generation.
Its exact input rule always preserves the immutable base parent in ancestry.
Use `application_scope_resolved`, not a universal observation-scoped lifecycle.
Use distinct roles `learned_candidate`, `resolved_model_state`,
`applied_model_state` and `realized_iteration_state`. Not every model is fitted
or internally learned; inapplicable roles are explicitly typed.

| Lifecycle role | Identity and meaning |
| --- | --- |
| Seed | Initializes S_0 in a lineage with its exact source/model parents and method declarations |
| Continuation | After completion of the iteration with absolute zero-based index N, begins at absolute index N+1 from continuation-sufficient S_(N+1), under the same scientific lineage, branch and compatible method/profile controls |
| Restart | Creates a new lineage with explicit ancestry to any reused parent/state; cannot masquerade as continuation. Its initialization and counters follow the method record, not an inferred preservation of earlier iteration identity |
| Relearned successor | A newly identified state/application generation with learned inputs and dependence under the same unchanged method; a learning-rule change instead requires a new method identity, and a simultaneous restart creates a new lineage |
| Branch | A distinct branch identity with explicit parent state and departure; cannot relabel an earlier realized iteration |
| Retry | A distinct attempt of a declared action; method-defined state/randomness/failure accounting determines compatibility. It is not a new acquired observation or implicit state reset |
| Completed iteration | Immutable record of k, input/state, produced result/state and local completion/failure facts |
| Selected terminal | Separate immutable selection record referencing an immutable completed iteration and its bundle |
| Publication | Separate publication state; it does not confer convergence, consumer fitness or scientific validation |

The owner's current FRUIT vocabulary supersedes the earlier packet's use of
“exact restart” for same-lineage continuation. Zero-based absolute iteration
identity is retained for continuation. This scoped terminology change does
not edit an upstream producer's lifecycle authority.

Changing the coefficient-recomputation rule, learning rule, operator family,
role order, sign, bypass graph, support rule, recurrence, parent route, grouping,
stopping, terminal selection or failure behavior changes method identity.
Different coefficient values or candidate/accepted/applied models produced by
an unchanged rule change realized state, iteration and application generations.
Do not relabel one as the other. Relearning scheduled by an unchanged method
may be part of exact continuation; it does not itself change that method.

## Exactly one category per operation and state component

| Category | Required binding and consequence |
| --- | --- |
| `fixed numerical operator` | Complete exact numerical map and fixed state; linearity, domain and support are separate propositions |
| `fixed structural state with recomputed coefficients` | Exact fixed structure plus coefficient-recomputation rule, dependencies and application state; do not classify the whole stage as a frozen numerical subtraction |
| `externally supplied` | Exact external owner, parent/version/generation, state and applicable compatibility; external is not independent by implication |
| `relearned` | Exact learning rule, parents/population, dependence, generation and failure; include changes in full-procedure claims |
| `bypassed` | Exact operation absent from the model path, with removal/rejoin locations and counts on both paths; residual-path state remains separately classified |
| `not_applicable` | Exact absent role and reason under the method; not an unknown value |
| `unavailable` | Missing or incompatible required fact, cause, domain and blocked operation/claim |

Factor a mixed stage into its named operations and consequential components;
do not give one component two categories to conceal a mixture. Record category,
owner, parent/generation, input dependencies, operation order, application
count, path and failure for each entry. A generic fixed or bypass flag is
insufficient. Exact classifications at different claim-conditioning scopes
are separate records, not contradictory labels on one record.

For PTC explicitly distinguish fixed loading subspace with recomputed
application coefficients, frozen numerical-component subtraction, and fully
relearned PTC state. They are different procedures with different response
and uncertainty. The frozen excerpt remains its own exact PTC authority;
FRUIT defines composition and state binding, not another cleaner.
An unchanged fixed-subspace/recomputed-coefficient method retains its identity
while publishing exact new coefficient/application state for each realization
or iteration.

## Exact continuation record

After completion of the iteration with absolute zero-based index N, the
continuation record binds the immutable base parent, S_(N+1),
separate candidate/accepted/applied model identities and states, any cumulative
model, learned and operator state, supports/masks, update
and selection history, convergence history, counters/resource state,
stochastic generator/key/counter state where applicable, method/parameter
versions, branch/generations, additive origin/reference or gauge, null-space
state and every upstream/model parent. Include every
fact capable of changing a future required result; a historical field list
or terminal signal array alone is not sufficient by implication.

Continuation is exact only when application under the same future controls
yields the same scientific states and results as uninterrupted execution
under the declared numerical profile. This is a conditional scientific
criterion, not a replay executed by Stage A. Materialized, compact and
lineage-reconstructible representations are permitted only when the exact
required state-query vocabulary and reconstruction contract are preserved.
A representation-only change does not change scientific state identity.

## Termination propositions

Keep `iteration_completed`, `resource_limit_reached`, `converged`,
`stopped_by_policy`, `failed`, `selected_terminal` and `consumer_fit` distinct.
Convergence binds its proposition, quantity/domain, metric/norm,
scale/denominator, tolerance, direction, persistence or consecutive-iteration
rule, missing/non-finite behavior, generation and state. No value is chosen.
Maximum or completion is not convergence; convergence is not correctness,
unit response, unbiasedness or downstream fitness.

Terminal selection uses one exact predeclared rule. Looking across several
realized iterations makes that selection part of full-procedure response and
uncertainty. Selection references a completed immutable iteration without
mutating or relabeling it. A failed or unavailable required selection fact
does not silently choose the last or most favorable iteration.

The terminal-selection record contains or types unavailable every row below.
Each status retains its exact identity, scope, generation and cause.

| Terminal-selection field | Required binding |
| --- | --- |
| Selected completed-iteration identity | Exact immutable selected iteration and bundle reference |
| Complete candidate iteration population considered | Every considered iteration identity and inclusion/exclusion state under the rule, with population identity; not only the selected or favorable members |
| Exact selection rule and generation | Predeclared method-owned rule, parameters, generation and tie/failure behavior |
| Convergence, stopping and resource-limit facts | The separate realized propositions and resource accounting used by selection |
| Selection causes | Why selection or failure occurred, including unavailable facts and affected scope |
| Selected-terminal response | Exact RF-05 identity and typed status, with any RF-07 composition reference and additive-reference/gauge/null-space consequences |
| Selected-terminal uncertainty | Exact uncertainty identity, conditioning, reference/gauge/null-space consequences, omitted sources and typed status |
| Continuation or terminality state | Exact continuation-state reference or terminality proposition/status; no implied ability to continue from the selected array |
| Downstream named-use status references | Exact consumer-profile/status references or truthful unavailable states; no inferred consumer fitness |

This record does not mutate, relabel or replace the selected iteration bundle.
A missing selected-iteration identity or another hard selection dependency
blocks the selected_terminal claim. Truthfully unavailable response,
uncertainty or downstream-use statuses permit only the narrower claim allowed
by the core and exact method; they do not erase the completed iteration.

## Completed iteration and terminal bundles

Every completed iteration bundle contains the following roles, with their
exact identities, values or permitted typed statuses: immutable base
parent; exact iteration input; method/lineage/branch/iteration identity; target
and model identities; candidate, accepted and applied model states; projected
removal contribution; residual; processed residual; rejoin contribution;
iteration result; update/successor state; path support and influence;
information origin; response states; uncertainty states; validity and causes;
continuation state; completion/failure state; and provenance. Candidate,
accepted and applied model entries bind m_k^candidate, m_k^accepted and
m_k^applied separately in M_k^candidate, M_k^accepted and M_k^applied,
including every transformation among them and exact facts establishing any
equality. Equal domain, support, unit, normalization, sign, representation,
response, uncertainty or parentage is not assumed.

The bundle also binds the removal and rejoin additive spaces and, for each
operand pair separately, compatible quantity/unit/numerical scale, frame/domain,
grid/sampling, support, origin/reference or gauge, null-space state, calibration
and response conventions, and exact conversions. Compatibility within each
pair does not imply that Y_k and Z_k are equal. It preserves unavailable
upstream modes and their information origin through response and uncertainty.
Rejoined content in an
unavailable additive mode remains model_or_prior_supported; it is not a new
measurement or recovered observational mode.
Path numerical components may be materialized or exactly reconstructible.

The conditional core defines universal minimum bundle structure and hard claim
dependencies. A numerical method may add required roles and declare which
optional companions are necessary for a named result or use. It may not waive
the universal requirements for immutable parent identity; exact iteration
input; accepted and applied model identities; removal, residual-processing and
rejoin roles; iteration result; path support and information origin; lifecycle
and causes; state and generation; provenance; and truthful response and
uncertainty status. A typed status preserves a required role; it does not omit it.

These hard dependencies apply to every method:

- A claimed numerical result requires the exact input, applied operand and
  removal/processing/rejoin composition needed on its claimed support to be
  available or exactly reconstructible. A missing required contribution cannot
  be treated as zero, identity or successful bypass.
  Exact additive-reference/gauge/null-space compatibility under the
  [quantity-space contract](NOTATION_AND_ROLE_TAXONOMY.md) is required for each
  subtraction and rejoin; an unavailable upstream mode cannot acquire
  observational authority through a model contribution.
- Exact continuation requires all future-consequential state and compatibility
  under the declared profile. A terminal array alone cannot satisfy that claim.
- A numerical response or uncertainty claim requires its declared premises,
  inputs, domains and conditioning. An unavailable companion may coexist with
  a complete narrower iteration only where both the core and exact method
  permit that narrower claim; its status and causes remain explicit.
- Terminal selection requires a separate immutable reference to a completed
  iteration satisfying the core and method's completion requirements.

An absent role requires a core-permitted not_applicable reason and the method's
explicit narrower composition. Unavailable does not mean absent. Missing
science never permits omission of a universal role or an invented result.

A terminal product consists of the separate immutable selection record plus
its selected iteration bundle. FRUIT ancestry creates no fifth generic
numerical parent class: a consumer receives the exact underlying MAP, JINC,
FLT-FIXED, FLT-MATCHED, detector-time or other method-defined product role
plus complete terminal-FRUIT lineage. Naming a role does not admit it.
