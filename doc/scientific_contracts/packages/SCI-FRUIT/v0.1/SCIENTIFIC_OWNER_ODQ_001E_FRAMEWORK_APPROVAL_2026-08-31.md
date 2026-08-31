# SCI-FRUIT v0.1 — Scientific-Owner ODQ-001E Framework Approval

Date: `2026-08-31`

Status: **approved Stage A scientific-comparison framework; exact
parameterization and recurrence remain open**

Decision ID: `SCI-FRUIT-OD-001E-FRAMEWORK-2026-08-31`

## Approved Decision

The scientific owner approves the comparative-quality approach in
[`COMPARATIVE_QUALITY_OBJECTIVE_GATE.md`](COMPARATIVE_QUALITY_OBJECTIVE_GATE.md).
Candidate FRUIT recurrences shall be assessed by a controlled, paired
comparison with an exact, versioned historical Citlali benchmark that remains a
scientific control rather than scientific authority.

The approved comparison and acceptance framework is:

1. preserve a vector of scientific-quality results rather than an unapproved
   weighted scalar score;
2. include angular-scale recovery, per-mode flux recovery, atmosphere/other
   residual leakage, flux convergence, noise/false structure, and honest
   response/uncertainty disclosure;
3. define protected scientific dimensions and non-inferiority tolerances;
4. require validity, response/uncertainty honesty, exact-restart, and
   failure-disclosure gates;
5. require a material scientific improvement over the historical benchmark in
   at least one owner-prioritized domain to justify an incompatible new
   recurrence; and
6. report computational and operational performance separately. It may not
   compensate for scientific degradation unless the owner later approves that
   explicit trade.

Per-iteration speed is diagnostic. When recurrence or stopping behavior
differs, the governing efficiency comparison is end-to-end resource use and
elapsed time to reach the same declared scientific-quality target, including a
typed outcome when the target is not reached.

## Consequence

`SCI-FRUIT-ODQ-001E` is decided for its framework. Exact parameterization is
separated into open `SCI-FRUIT-ODQ-001F`, which must bind the benchmark profile,
signal and nuisance families, metric definitions, uncertainties, protected
dimensions, tolerances, material-improvement threshold, performance protocol,
and failure/unavailable rules before candidate recurrences can be ranked.

## Explicit Non-Effects

This approval does not select a recurrence, close ODQ-001, define a numerical
threshold, approve a benchmark run, admit a parent route, authorize Stage B,
change an algorithm, implement a metric, validate a method, or authorize
production or Unity activity. Historical anecdotes remain motivation only.
