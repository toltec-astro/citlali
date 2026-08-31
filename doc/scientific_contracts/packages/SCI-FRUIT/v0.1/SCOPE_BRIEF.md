# SCI-FRUIT v0.1 — Iterative Feedback, State, And Termination Scope Brief

Status: **Stage A owner-review candidate; not owner-approved**

Scientific owner: Grant Wilson

Version/date: `v0.1-stage-a-r0.1`, `2026-08-31`

Approved source identifier: **none**

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: pending scientific-owner review
- Existing material adopted or cited: absolute iteration/restart identity,
  requested/effective/realized state separation, frozen MAP/JINC/PTC and
  FLT-FIXED boundaries, and approved NOI FRUIT-generation boundary
- Existing material abstracted, deferred, or excluded: historical FRUIT
  metric taxonomy abstracted; empirical thresholds and calibration deferred;
  code, configuration, audits, repairs, tests, validation, production history,
  and provisional FLT-MATCHED authority excluded
- Genuinely new scientific work: feedback-model estimand, route admission,
  recurrence, projection, state/generation, support, stopping, response,
  uncertainty, checkpoint completeness, and terminal-product decisions
- Owner-approved author references: none yet
- Author-packet exclusions: all implementation-informed and provisional
  material named above

Confirm that this opening was reviewed before launching scientific authorship:
**no**.

## 1. Package Name And Scientific Purpose

SCI-FRUIT defines an iterative scientific procedure in which an explicitly
selected model of recoverable sky signal is forward projected, removed for
residual processing, and incorporated into a new immutable iteration state.
The contract must say what the model estimates, how it is constructed and
updated, what state can affect the future, when a sequence may terminate, and
what response and uncertainty claims belong to the resulting products.

## 2. Scientific Boundary

The boundary begins with exact immutable upstream reduction products and a
requested FRUIT plan. It includes feedback-model construction/selection,
forward projection, iteration-state update, recurrence, termination, restart,
and FRUIT-owned product identity. It ends with exact immutable terminal and/or
per-iteration products plus state, response, uncertainty, support, validity,
failure, and provenance disclosures selected by the contract.

RTC owns raw-timestream calibration/conditioning within its authority; PTC owns
the cleaning and coefficient policies it defines; MAP and JINC own their map
estimands; FLT packages own their transformations; NOI owns its declared
generation/uncertainty methods; VAL owns registry/evaluation mechanics; source
fitting, Pointing, and OOF packages own their later estimands. SCI-FRUIT may
bind or consume exact products from these packages but may not redefine them.

## 3. Legitimate Inputs

Any admitted input must have exact product/method/version/generation identity,
grouping, parent lineage, units and calibration convention, WCS/grid/frame,
response/transfer and null-space meaning, support and boundary behavior,
validity/missing/non-finite policy, covariance/uncertainty status, lifecycle,
and compatibility with the selected feedback-model and forward-projection
method.

Candidate route families are ordinary MAP, JINC, conditionally frozen
FLT-FIXED, and provisional FLT-MATCHED. They are distinct and individually
gated. Listing them does not admit them. A map-only seed and an exact checkpoint
restart are separate inputs with different generation meaning.

## 4. Required Outputs

The future contract must decide which of the following are required:

- per-iteration feedback-model and/or residual scientific products;
- the exact model increment and accumulated model, if both exist;
- a terminal scientific product and the rule selecting its iteration;
- exact iteration, generation, parent, grouping, and branch identity;
- response/transfer, null-space, support, validity, and failure disclosures;
- fixed-state or fuller-procedure covariance/uncertainty products, or typed
  unavailability;
- operational checkpoint state sufficient for exact continuation; and
- non-authoritative diagnostics that cannot affect future outputs.

No current file or product name is presumed required.

## 5. Upstream And Downstream Responsibilities

Upstream packages retain ownership of their estimator, normalization, units,
response, covariance, support, validity, and numerical admission. SCI-FRUIT
owns the scientific choice to admit a route, construct a feedback model from
it, project that model, update state, and terminate. Downstream source-fitting,
Pointing, OOF, and catalog consumers must use exact terminal/iteration and
response identity; SCI-FRUIT does not perform their inference or calibration.

## 6. Externally Imposed Conventions

- Iteration IDs are zero-based and absolute across exact restart.
- A completed iteration `N` resumes at `N+1`; an absolute exclusive iteration
  limit must exceed that next iteration to continue.
- Map-only seeding creates a new lineage/generation and is not exact restart.
- Requested, effective, observation-resolved, and realized state are distinct.
- Formal significance, formal standardization, empirical point-source S/N, and
  legacy dynamic range are not interchangeable.
- Morphology metrics use the exact realized kernel/response; planet and
  unresolved-source interpretation are not silently merged.

## 7. Questions The Contract Must Answer

1. What is the normative feedback-model estimand, and is the recurrence
   additive accumulation, replacement, residual correction, or another law?
2. Which candidate parent routes are admitted, for which groupings and
   generations, and how is a parent converted into a feedback model?
3. What exact forward operator maps the model into timestream sample space, and
   how do subtraction/add-back and PTC/RTC operations compose with it?
4. Which policy is fixed, which state is learned, which state is applied, and
   which state may be relearned at each iteration or observation?
5. What are the response, attenuation/bias, covariance/uncertainty, null-space,
   support, missing/non-finite, selection, and failure meanings?
6. Which diagnostics describe amplitude, morphology, centroid, map change,
   support/learning, and noise health, and which—if any—enter a stopping rule?
7. What selects the terminal iteration and product when criteria disagree or
   become measurement-limited?
8. What complete causal state must an exact checkpoint restore, and what
   compatibility changes require rejection or a new generation?
9. Which NOI method targets fixed state, complete procedure, or per-member
   replay, and what dependence prevents pooling?
10. Which products are required, and which downstream claims fail closed when
    response, covariance, numerical parents, or validation are unavailable?

## 8. Non-Goals

This package does not implement, repair, optimize, validate, calibrate, run, or
authorize FRUIT. It does not change MAP, JINC, PTC, NOI, FLT, VAL, RTC, CAL,
ALIGN, AST, BEAM, source-fitting, Pointing, or OOF authority. It does not approve
the provisional matched-filter study, define a source catalog, or establish a
universal stopping threshold, photometric correction, performance level, or
production policy.

## 9. Allowed References

No implementation-blind author references are approved. A proposed future set
is recorded in [`PROPOSED_SANITIZED_AUTHOR_INPUTS.md`](PROPOSED_SANITIZED_AUTHOR_INPUTS.md)
for explicit owner review and exact-byte binding.

## 10. Owner Decisions And Remaining Ambiguities

The only owner decision recorded in this Stage A revision is the sequencing
decision to launch FRUIT now, before source-fitting and Pointing/OOF. All
scientific decisions remain open and are ordered in
[`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).

## 11. Independence Statement

This brief defines the problem and boundaries without prescribing current
Citlali behavior as the answer. It contains no implementation names, defaults,
empirical trajectory values, or validation claims. It is not an author packet,
and Stage B remains unauthorized.
