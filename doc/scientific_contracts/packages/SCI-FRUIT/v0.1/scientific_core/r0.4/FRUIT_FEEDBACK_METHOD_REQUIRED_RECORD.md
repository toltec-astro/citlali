# FRUIT-FEEDBACK-METHOD v0.1 — required-record template

Document revision: r0.4. Owner direction: SCI-FRUIT-OD-STAGE-A-MICRO-REPAIR-2026-09-07.
Authority state: `unavailable_pending_separate_owner_approval`.
`FRUIT-FEEDBACK-METHOD v0.1` is the required-record authority family, not an
approved numerical method. Every numerical method requires its own exact
identity `FRUIT-FEEDBACK-METHOD/<method-name>@<revision>` and Grant Wilson's
separate scientific approval. Creating this family record approves no
method-specific identity. This is a scientific record obligation, not a
storage schema or an executable profile.

## Complete method binding

Every row is required. An optional scientific role must carry an explicit
`not_applicable` with its reason where the core permits that absent role; an
unknown required fact is unavailable, not a default. The universal minimum
bundle and hard claim dependencies in the [lifecycle table](ITERATION_STATE_LIFECYCLE.md)
cannot be waived by a method. Actual scientific values remain unselected.

| Binding | Required scientific content |
| --- | --- |
| Authority and lineage | Method-specific identity/revision under the family, exact owner decision and authority state; method generation distinct from state/iteration/application generations; immutable base parent and producer lineage |
| Parent route and grouping | One exact numerically available route supplying quantity, unit, frame, response, beam/template convention, calibration, support, validity, lifecycle and provenance; one declared grouping with membership, axes and scope; all required domain conversions |
| Iteration-input rule | Immutable original parent, previous z_(k-1), reconstruction from base parent/current cumulative model, or another exact product; every conversion and its authority |
| Target and model | Quantity, units, domain/frame and response meaning of target and separately m_k^candidate, m_k^accepted and m_k^applied; distinct external versus internally learned origin and prior |
| Candidate/selection/acceptance/application | Exact construction, causal selection/acceptance and acceptance-to-application rule; allowed inputs at action time; every normalization, clipping, sign conversion, support restriction, projection, unit conversion or other transformation; equality of accepted and applied objects only by exact method/state binding |
| Model accumulation meaning | Cumulative model versus increment versus diagnostic difference; exact relation to the separate accepted/applied models and successor state |
| Typed composition | Y_k, M_k^candidate, M_k^accepted, M_k=M_k^applied, Z_k; Pi_k, F_k, B_k and domains/codomains, units, support, state and conditioning; applied model as operand; sign convention and exact removal/rejoin locations; each linearity claim explicit |
| Operation graph | Ordered named operations, exact bypass set, all application counts and parent/state bindings; no generic bypass flag or duplicated response application |
| State by operation | Exactly one category per named operation and consequential state component from the operation-state table; every fixed, recomputed, externally supplied or relearned fact, with learning schedule and triggering conditions |
| Support and missing data | All path supports, learning influence, response-query support, information-origin classes, unequal-support rules and failure scope |
| Update and transition | Exact rule consuming S_k and y_k^in and producing separate m_k^candidate, m_k^accepted, m_k^applied, rho_k, z_k and S_(k+1), with core-permitted typed not_applicable/unavailable roles; no default recurrence |
| Response | Exact role from the seven-role catalog; perturbed and held-fixed objects, rerun/relearning boundary, baseline and perturbed domains, branch/support/selection comparability, response codomain, unavailable/discontinuity behavior |
| Uncertainty | Exact joint conditioning, covariance or NOI method; applied-model covariance and measured-input/applied-model cross terms for the fully linear relation; candidate/accepted-to-applied transformations, omitted sources, ensemble and state generations |
| Continuation sufficiency | Complete continuation-state query vocabulary; same-future-controls equality criterion under the declared numerical profile; branch/generation compatibility |
| Stochastic state | Exact generator/algorithm, keys/counters/state, populations and sampling controls where applicable; otherwise justified not_applicable |
| Convergence proposition | Exact proposition, quantity/domain, metric or norm, scale/denominator, tolerance, direction, persistence/consecutive rule, missing/non-finite behavior, generation and state |
| Resource limits | Exact accounting, counters, stopping consequences and exhaustion state; no default cap or inference of convergence |
| Terminal selection | One exact predeclared rule and its data access, inspected iterations, ties, unavailable/failure behavior, response and uncertainty consequences |
| Persistence/reconstruction | Materialized, compact or lineage-reconstructible representation; exact state queries and path components preserved; retention/reconstruction and failure contract |
| Completion, failure and publication | Universal core minimum bundle and hard claim dependencies, additional method-required roles and named-use optional companions; permitted narrower claims with truthful unavailable status; exact failure actions/scope, immutable terminal selection and publication status; no waiver of core requirements |

## Method rules and realized state are distinct

A change to parent route, coefficient-recomputation rule, learning/model-
construction rule, operator family, role order, sign, bypass graph, support
rule, recurrence, grouping, stopping rule, terminal-selection rule or failure
behavior creates a new method identity. Methods differing in these facts shall
not share one identity.

Different coefficient values, learned states, accepted models or applied models
produced under one unchanged method create distinct state, iteration and
application generations. Fixed-subspace application with recomputed coefficients
retains one method identity and publishes exact coefficient/application state
for each realization or iteration. A rule change is not a state change, and
the next state produced by an unchanged approved rule is not a new method.

## Grouping and input are method identity

A method declares one complete grouping: observation-local, coadd-level,
multi-observation, per-array, joint-array or another explicitly named grouping.
If several axes are needed, name their complete composition as one method
grouping; do not silently borrow incompatible defaults. Shared parent, model,
calibration, response and reference terms retain their dependence.

Returning to the immutable original parent and repeatedly processing the
previous result are different methods even with the same model update. If
Z_(k-1) and Y_k differ, a declared compatible conversion is required before
z_(k-1) can supply y_k^in. The immutable base parent stays in every lineage.
The word iteration provides none of these choices. FRUIT of a coadd is not
the coadd of independently processed FRUIT observations by implication;
joint and independent model learning are also different procedures.

## Numerical gate and core authorship

Without one complete, exact owner-approved method-specific record there is **no
numerical FRUIT route or result**. Completing a template syntactically is not
approval, numerical availability, conformance or qualification. Missing
upstream numerical authority remains a separate blocker even for an approved
method record.

The conditional core may be authored and later scientifically frozen while
this authority and every numerical route, model construction, stopping rule,
response/covariance and downstream profile remain unavailable. It must derive
conditional consequences and type unavailable claims without supplying a
familiar algorithm or plausible default. Numerical development, qualification
and any policy recommendation retain their separate owner decisions and
independent-pointing evidence obligations.
