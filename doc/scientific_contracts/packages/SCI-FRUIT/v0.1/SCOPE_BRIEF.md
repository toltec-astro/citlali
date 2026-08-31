# SCI-FRUIT v0.1 — Iterative Feedback, State, And Termination Scope Brief

Status: **Stage A owner-review candidate; not owner-approved**

Scientific owner: Grant Wilson

Version/date: `v0.1-stage-a-r0.5`, `2026-08-31`

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
- Genuinely new scientific work: accepted feedback-state identity, recurrence
  choice relative to the recovered historical baseline, route admission,
  projection/bypass semantics, state/generation, update-contribution status,
  bounded persistence, support, stopping, response, uncertainty, checkpoint
  completeness, and terminal-product decisions
- Owner-approved author references: none yet
- Author-packet exclusions: all implementation-informed and provisional
  material named above

Confirm that this opening was reviewed before launching scientific authorship:
**no**.

## 1. Package Name And Scientific Purpose

SCI-FRUIT defines an iterative scientific procedure in which an explicitly
selected model of recoverable sky signal is forward projected, removed for
residual processing, bypasses named residual-only operators, rejoins at an
exactly defined point, and contributes to a new versioned iteration state.
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

- per-iteration accepted feedback-model and/or residual scientific products;
- any update contribution required by the chosen transition, with its status
  as causal state, diagnostic, equivalence witness, or scientific product;
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

1. What is the versioned accepted feedback state `F_k` passed to the next
   iteration, including its route, estimand, support, response, grid, and
   lineage?
2. Will v0.1 preserve the recovered historical recurrence, adopt a proved and
   validated equivalent reformulation, or intentionally adopt a new recurrence?
   Until that choice, use only `F_{k+1}=U_k(F_k,R_{k+1})`.
3. Relative to which exact historical benchmark profile and using which
   scientific-quality and computational-performance vectors will a candidate
   recurrence be judged? Scientific quality must distinguish angular-scale
   recovery, per-mode flux recovery, nuisance leakage, flux convergence, noise,
   response, and uncertainty rather than collapse them by default.
4. Does a per-iteration update contribution exist, what scientific or
   diagnostic status does it have, and is it causal for later output?
5. What bounded persistence, compaction, reconstruction, lineage, and
   reproducibility rules apply? Stable identity must not imply unbounded
   intermediate retention.
6. Which candidate parent routes are admitted, for which groupings and
   generations, and how is a parent converted into a feedback model?
7. What exact forward operator maps the model into timestream sample space;
   which operators see only the residual; which does the accepted model bypass;
   where does it rejoin; and what response does it have in the next map?
8. Which policy is fixed, which state is learned, which state is applied, and
   which state may be relearned at each iteration or observation?
9. What are the response, attenuation/bias, covariance/uncertainty, null-space,
   support, missing/non-finite, selection, and failure meanings?
10. Which diagnostics describe amplitude, morphology, centroid, map change,
   support/learning, and noise health, and which—if any—enter a stopping rule?
11. What selects the terminal iteration and product when criteria disagree or
   become measurement-limited?
12. What complete causal state must an exact checkpoint restore, and what
   compatibility changes require rejection or a new generation?
13. Which NOI method targets fixed state, complete procedure, or per-member
   replay, and what dependence prevents pooling?
14. Which products are required, and which downstream claims fail closed when
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

The owner has also directed Stage A to recover the exact historical recurrence,
use it as the v0.1 reference baseline, separate the four recurrence decisions,
and re-present ODQ-001 with preserve/equivalent/new choices. After reviewing
that analysis, the owner provisionally favored Choice 3 as the
method-development category while retaining the historical recurrence as the
mandatory compatibility reference and scientific control. This does not
approve a candidate law or close ODQ-001. The scientific estimand, meaning of
"better," exact transition, expected differences, compatibility consequences,
and validation obligations remain open. The owner has now bounded "better" as
a controlled comparison with exact historical Citlali using distinct
scientific-quality and computational-performance metrics. Angular-scale
recovery, per-mode flux recovery, atmospheric/other residual leakage, and flux
convergence are required scientific domains; exact definitions, thresholds,
and benchmark profile remain open. The owner has approved the
governing constrained multi-objective framework: protected scientific
non-inferiority plus material improvement in an owner-prioritized scientific
domain, with computational performance separate absent later explicit trade
approval. Exact benchmark and metric parameterization remain open under
ODQ-001F and are ordered in
[`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).

## 11. Independence Statement

This brief defines the problem and boundaries without prescribing current
Citlali behavior as the answer. It contains no implementation names, defaults,
empirical trajectory values, or validation claims. It is not an author packet,
and Stage B remains unauthorized.
