# SCI-FLT-MATCHED v0.1 — Required Authored Option Sets

Status: sanitized author assignment; no option selected

The Scientific Rationale and Contract and Engineering Conformance
Specification shall each contain the same bounded options, exact IDs,
assumptions, consequences, observables or quantitative bounds, failure rules,
and validation consequences. The author may refine the option count, but must
retain these family IDs and cover every consequence below. Options may be
recommended but never silently selected.

## `SCI-FLT-MATCHED-AO-001` — Weighting And Noise Authority

Provide bounded alternatives for the exact `Q_x` object for observation and
coadd parents. Each alternative shall define whether it is inverse covariance,
a structured covariance-derived approximation, or weaker spectral weighting;
its population/domain, estimator, averaging, radialization, Fourier/WCS and
unit conventions, window/support/edge rules, rank/null space, regularization,
state dependence, parent-coefficient role, normalization and optimality
consequences, response, uncertainty, NOI parity, products, and typed
unavailability.

At least one alternative shall scientifically define and assess a radially
symmetrized average map noise PSD, the historically supplied candidate. No
historical mechanics or default may be inferred, and radial symmetry does not
by itself prove stationarity, isotropy, covariance authority, or optimality.

## `SCI-FLT-MATCHED-AO-002` — Quantitative Approximation Envelope

Provide bounded alternatives for judging approximate realization against the
exact `N/D` operator. Each shall quantitatively bound normalization error,
matching-template amplitude-response error, support/null disagreement, and
uncertainty consequences over the declared validity domain. State sampling or
test domains, tolerances, aggregation, unresolved modes, failure rules, and
when a changed realization becomes a new method. Iteration or tail caps alone
are not convergence evidence.

## `SCI-FLT-MATCHED-AO-003` — Conditional Covariance Representation

Provide the smallest bounded alternatives for representing authorized
`C_cond=L C_parent L^T`: exact explicit, structured, projected,
lineage-resolvable/on-demand, or unavailable forms as scientifically
warranted. Each shall state estimator and population, domain/support/response,
rank/null space, approximation/regularization, omitted correlations,
calibration terms, admissible consumers, reconstruction fidelity, lifecycle,
and failure. Unknown entries remain unknown, not zero or independent.

## `SCI-FLT-MATCHED-AO-004` — Immutable State Persistence

Provide bounded alternatives for persisting or exactly reconstructing the
declared/learned-once frozen state: full materialization, structured compact
state, or exact lineage reconstruction where defensible. Each shall bind
algorithm/version, inputs and learning population, numerical identity and
tolerances, environment-dependent facts, reconstruction test, audit/reanalysis
limits, required persistence, NOI parity, successor generation, and failure.

## `SCI-FLT-MATCHED-AO-005` — Response, Product, And FLT→FRUIT Interface

Provide bounded alternatives for response materialization and lineage-
resolvable reconstruction consistent with the minimal-product decision and
the first-class FLT→FRUIT requirement. Each shall preserve the exact operator,
filtered quantity, template, response, valid support, units/calibration,
method/state, uncertainty availability, and provenance without requiring
undocumented implementation state. State storage cost, exactness, supported
queries/consumers, reconstruction checks, lifecycle, and failure. No option
may define FRUIT behavior or weaken the scientific response.

## `SCI-FLT-MATCHED-AO-006` — VAL Profile Granularity

Provide bounded alternatives for the smallest immutable FLT-owned profile
set needed to distinguish parent admission, state admission, public signal
publication, conditional covariance use, optional NOI companion use, and
FLT→FRUIT handoff. Each shall identify producer facts, named-use policy,
consumer action, unavailable/failed behavior, versioning, aggregate rules,
and whether combining roles would conceal a scientifically distinct failure.
VAL remains evaluator/registry only.
