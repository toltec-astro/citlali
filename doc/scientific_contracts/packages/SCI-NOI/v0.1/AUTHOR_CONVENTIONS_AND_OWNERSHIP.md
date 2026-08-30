# SCI-NOI v0.1 — Author Conventions And Ownership

Status: proposed sanitized Stage B author input; exact bytes await owner approval

This extract contains only the upstream/downstream conventions needed for
implementation-blind authorship. It is not an implementation map.

## Parent And Operator Authority

- RTC owns conditioning operations, learned state, response, causal influence,
  validity, and lineage.
- CAL owns signal quantity/unit meaning, calibration/extinction state,
  response/uncertainty, quality, and lineage.
- PTC owns `Z_i^PTC`, retention, segment, cleaning realization, analysis
  coefficient identity, response/covariance state, and application generation.
- AST owns occurrence coordinates, time identity, exact WCS/frame joins, and
  coordinate validity.
- MAP owns the ordinary map/coadd estimator, support, base-product validity,
  nonprecision coefficient use, response/covariance disclosure, and immutable
  bundle identity.
- JINC owns its signed estimator, complete five-role observation bundle, and
  typed numerical-unavailability state. No ordinary MAP rule applies to JINC
  by analogy.
- NOI owns ensemble design, realization identity, empirical uncertainty
  inference, and derived standardization. Rerunning an adjacent operator does
  not transfer ownership of that operator to NOI.
- GEN owns member completion and terminal-state truth. NOI owns the policy for
  admission to a named UNC use. SCI-VAL Registry binds exact NOI-owned profiles
  and VAL Core evaluates them; VAL authors neither producer facts nor policy.

NOI preserves parent meanings and causes. Its work may begin after MAP/JINC
scientific freeze while a numerical GEN route begins at an earlier immutable
parent. Every method therefore names both its earliest immutable parent and
its insertion point. No exact realized numerical parent is created by a route
description.

ODQ-102A selects the ordinary fixed-state route at the exact PTC-to-MAP
numerical boundary. NOI owns the assignment modifier, ensemble design,
realization identity, and resulting NOI realization map. MAP may apply that
modifier inline within its frozen ordinary accumulation arithmetic; MAP owns
only conformance of that application. Materializing a randomized timestream is
not required, and use of MAP arithmetic does not create an ordinary MAP science
product.

## Stable Identity And Unit Conventions

- Array identity is `a1100`, `a1400`, or `a2000`, never a container position.
- Observation, detector/channel occurrence and UID, stable RTC output sample
  `n`, scan/subscan/segment, sample, realization, map/group/Stokes, WCS/frame,
  support, operator state, and product/application generation are separate.
- In-memory indices are zero-based; persisted FITS/WCS pixel coordinates are
  one-based. WCS, not memory layout, defines persisted spatial meaning.
- Requested, effective, observation-resolved, applied, and realized states are
  distinct one-way lifecycle stages.
- Missing, unavailable, unsupported, invalid, incomplete, and failed are
  explicit states, not undocumented zero, NaN, infinity, or empty-container
  sentinels.
- A required publication or identity-join failure prevents realized success at
  its declared scope.
- Assignments and correlation objects are dimensionless. A realization has its
  parent signal unit. Variance/covariance has squared signal unit.
  Inverse-variance/precision has inverse-squared signal unit. Standardized
  signal has unit exactly `1` and is dimensionless. Equal units never establish
  equal scientific role or statistical significance.
- Bare NOI family symbols `G`, `U`, and `Z` are prohibited. Use `NOI-GEN`,
  `NOI-UNC`, and `NOI-STD`. `Z_i^PTC` is reserved for the PTC transformed
  sample; MAP projection notation remains MAP-owned.
- Enabled polarimetry and measured-R execution remain outside active authority.

## Fixed Versus Relearned State

Every GEN method classifies every scientifically consequential adjacent
reduction state as exactly one of:

- immutable/fixed from an exact parent;
- rerun/relearned by an exact named procedure;
- not applicable; or
- unavailable/unknown with its consequence.

The method names the applicable adjacent stages; examples can include RTC,
CAL, PTC, AST, MAP/JINC, filtering, source-model, coadd, mask/support, and
consumer-selection state, but this is not an exhaustive implementation-
provenance checklist.

For fixed-state generation, `realization_b = O_Theta0(R_b(parent))`. For a
relearned method, `Theta_b = LearnResolve_b(R_b(parent))` and
`realization_b = O_Theta_b(R_b(parent))`. “Relearned” is never generic. Two
methods that rerun different stages, training subsets, masks, weights,
convergence rules, or perturbations have different identities. Fixed-state and
relearned members cannot be pooled without a separately authorized mixture
estimand. A relearned method identifies the consequential stages it reruns or
relearns and the resulting state that may differ from the real-observation
reduction; it need not trace scientifically irrelevant implementation details.

## Same-Operator And Externally Owned Transformation Boundary

For a fixed-state realization to use the same MAP, coadd, or externally owned
deterministic transformation as its signal parent, exact membership,
projection, grouping,
coefficients, normalization, support-authorized row selection, boundary/edge
treatment, response handling, and publication domain are fixed or identically
prescribed. Re-estimation or selection from a realization creates another
method and changes the inference target.

Under ODQ-110A, NOI never chooses or defines the transformation. The
appropriate upstream/downstream scientific process owns its purpose,
algorithm, operator, parameters/learned state, operation order, domain,
support/edge/missing-data behavior, normalization, units, response/transfer
meaning, validity, lifecycle, and failure semantics. NOI binds that exact
authority and applies exactly that transformation to every admitted compatible
randomization when estimating uncertainty for the exact transformed scientific
product. It may not infer, tune, substitute, relocate, simplify, or silently
omit the transformation. Missing or conflicting parity makes the route
unavailable.

Under ODQ-110B, a Wiener transformation learned, selected, or updated by its
owning process using `UNC_k` begins a new immutable generation:

```text
UNC_k -> TRANSFORM_OWNER:LearnSelect_(k+1) -> T_(k+1)
science_parent -> T_(k+1) -> transformed_science_(k+1)
GEN_base_(k+1) -> T_(k+1) -> GEN_transformed_(k+1) -> UNC_(k+1)
```

Neither `T_(k+1)` nor `UNC_(k+1)` mutates or validates `UNC_k`
retroactively. `UNC_k` is a declared dependent input, not independent evidence
for the successor. A Wiener transformation learned once by its owner and frozen
before realization application follows ODQ-110A. A separately learned
transformation per member is a distinct ODQ-104 method.

ODQ-110C applies the same ownership and generation discipline to FRUIT. FRUIT
owns its source model, subtraction/add-back, learned state, recurrence,
iteration, stopping, restart, selection, response, support, validity,
lifecycle, and failure. An exact fixed residual or terminal transform supports
only uncertainty conditional on frozen FRUIT state under ODQ-110A parity. An
NOI-informed continuation begins a new immutable FRUIT/science/GEN/UNC
generation; partial or complete per-member replay is a distinct ODQ-104
relearned method. Fixed and replayed members cannot mix without an authorized
mixture estimand.

## Validity, Support, And Profiles

Sample validity, parent named-use eligibility, GEN input admission,
GEN-owned realization-member completion/QC, UNC member admission, UNC ensemble
admission, estimator validity, covariance-domain support, and STD admission are
separate. Assignment cannot clear a cause, make a missing occurrence valid, or
create support. A finite payload, completed member, generic flag, or pass for
another use has no universal veto or rescue effect. Rejected design candidates
are not members or failures; once admitted, failure of any member fails the
ensemble for all UNC use and completed survivors cannot be admitted as a
partial ensemble.

Every NOI-owned profile retains named request, applicability, eligibility, and
realization fields. Only the exact profile's projection authorizes its exact
consumer action. Missing or conflicting required facts fail closed at the
named action without rewriting parent validity.

## Uncertainty And Weight Meanings

The following roles are not interchangeable:

- second moment;
- conditional randomization variance/covariance;
- repeated physical-noise variance/covariance;
- calibrated empirical variance/covariance;
- marginal inverse variance;
- full or regularized precision;
- consumer-effective weight;
- PTC/MAP analysis or gridding coefficient;
- positive signal-unit standardization scale; and
- standardized signal.

An empirical NOI inverse or weight remains a companion result. Any future
mapmaking, coadd, filter-fitting, or adaptive use requires an explicit
consuming method and cannot retroactively modify an immutable parent.

## Adjacent Scientific-Process Ownership

- The appropriate upstream/downstream scientific process owns each
  deterministic or inference-bearing transformation, including transfer,
  response, support, edges, covariance transformation, and local validity.
- SRC/MODE/BEAM own source, Pointing, OOF, and Beammap estimators and their
  consumer-specific qualification.
- FRUIT owns source subtraction/add-back, feedback, learning, recurrence,
  convergence, restart, and adaptive selection.

NOI may apply an exact content-bound owner-defined transformation to admitted
randomizations or provide declared companions to consumers. This does not
transfer transformation authority to NOI or validate a consumer merely because
the consumer used an NOI product. Fixed, successor, and relearned Wiener/FRUIT
routes remain unavailable without their exact owner and NOI boundaries.

## Claim Separation

Algebraic contract correctness, implementation conformity, representation
fidelity, numerical validation, observational/statistical calibration,
achieved performance, readiness, and production authorization are separate.
No Stage B scientific derivation may infer the latter claims from historical
execution evidence.
