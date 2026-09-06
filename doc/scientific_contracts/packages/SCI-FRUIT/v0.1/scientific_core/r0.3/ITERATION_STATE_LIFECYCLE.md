# SCI-FRUIT — iteration, operation state and lifecycle r0.3

Status: owner-directed Stage A candidate. Authority: repair direction
SCI-FRUIT-OD-STAGE-A-TYPE-METHOD-REPAIR-2026-09-06, sections 2–5, 10–14.

## State and identity

S_0 is the initial seed state. Iteration k consumes S_k and y_k^in and produces
the accepted/applied model m_k, result z_k and successor S_(k+1). The method
record defines how those objects are constructed; this convention chooses no
update law. After completed iteration N, exact continuation starts from
S_(N+1) and the next absolute zero-based index N+1.

Every iteration binds at least `lineage_id`, `branch_id`, absolute zero-based
iteration index, method generation, parent generation and state generation.
Its exact input rule always preserves the immutable base parent in ancestry.
Use `application_scope_resolved`, not a universal observation-scoped lifecycle.
Use distinct roles `learned_candidate`, `resolved_model_state`,
`applied_model_state` and `realized_iteration_state`. Not every model is fitted
or internally learned; inapplicable roles are explicitly typed.

| Lifecycle role | Identity and meaning |
| --- | --- |
| Seed | Initializes S_0 in a lineage with its exact source/model parents and method declarations |
| Continuation | Consumes continuation-sufficient S_(N+1) after completed N under the same scientific lineage, branch and compatible method/profile controls |
| Restart | Creates a new lineage with explicit ancestry to any reused parent/state; cannot masquerade as continuation. Its initialization and counters follow the method record, not an inferred preservation of earlier iteration identity |
| Relearned successor | A newly identified state/method generation with all learned inputs and dependence; any simultaneous restart also creates a new lineage |
| Branch | A distinct branch identity with explicit parent state and departure; cannot relabel an earlier realized iteration |
| Retry | A distinct attempt of a declared action; method-defined state/randomness/failure accounting determines compatibility. It is not a new acquired observation or implicit state reset |
| Completed iteration | Immutable record of k, input/state, produced result/state and local completion/failure facts |
| Selected terminal | Separate immutable selection record referencing an immutable completed iteration and its bundle |
| Publication | Separate publication state; it does not confer convergence, consumer fitness or scientific validation |

The owner's current FRUIT vocabulary supersedes the earlier packet's use of
“exact restart” for same-lineage continuation. Zero-based absolute iteration
identity is retained for continuation. This scoped terminology change does
not edit an upstream producer's lifecycle authority.

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

## Exact continuation record

After iteration N the record binds the immutable base parent, S_(N+1),
accepted/cumulative model, learned and operator state, supports/masks, update
and selection history, convergence history, counters/resource state,
stochastic generator/key/counter state where applicable, method/parameter
versions, branch/generations and every upstream/model parent. Include every
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

## Completed iteration and terminal bundles

Every completed iteration bundle contains or types unavailable: immutable base
parent; exact iteration input; method/lineage/branch/iteration identity; target
and model identities; candidate, accepted and applied model states; projected
removal contribution; residual; processed residual; rejoin contribution;
iteration result; update/successor state; path support and influence;
information origin; response states; uncertainty states; validity and causes;
continuation state; completion/failure state; and provenance. The method
states which unavailable roles block completion or a particular claim.
Path numerical components may be materialized or exactly reconstructible.

A terminal product consists of the separate immutable selection record plus
its selected iteration bundle. FRUIT ancestry creates no fifth generic
numerical parent class: a consumer receives the exact underlying MAP, JINC,
FLT-FIXED, FLT-MATCHED, detector-time or other method-defined product role
plus complete terminal-FRUIT lineage. Naming a role does not admit it.
