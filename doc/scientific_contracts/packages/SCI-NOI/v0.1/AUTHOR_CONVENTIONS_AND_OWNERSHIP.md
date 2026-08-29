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
- SCI-VAL Registry binds exact NOI-owned profiles and VAL Core evaluates them.
  Neither authors NOI method, target, adequacy, or publication policy.

NOI preserves parent meanings and causes. Its work may begin after MAP/JINC
scientific freeze while a numerical GEN route begins at an earlier immutable
parent. Every method therefore names both its earliest immutable parent and
its insertion point. No exact realized numerical parent is created by a route
description.

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
  signal is dimensionless. Equal units never establish equal scientific role.
- Bare NOI family symbols `G`, `U`, and `Z` are prohibited. Use `NOI-GEN`,
  `NOI-UNC`, and `NOI-STD`. `Z_i^PTC` is reserved for the PTC transformed
  sample; MAP projection notation remains MAP-owned.
- Enabled polarimetry and measured-R execution remain outside active authority.

## Fixed Versus Relearned State

Every GEN method enumerates RTC, CAL, PTC, AST, MAP/JINC, filtering,
source-model, coadd, mask/support, and consumer-selection state as exactly one
of:

- immutable/fixed from an exact parent;
- rerun/relearned by an exact named procedure;
- not applicable; or
- unavailable/unknown with its consequence.

For fixed-state generation, `realization_b = O_Theta0(R_b(parent))`. For a
relearned method, `Theta_b = LearnResolve_b(R_b(parent))` and
`realization_b = O_Theta_b(R_b(parent))`. “Relearned” is never generic. Two
methods that rerun different stages, training subsets, masks, weights,
convergence rules, or perturbations have different identities. Fixed-state and
relearned members cannot be pooled without a separately authorized mixture
estimand.

## Same-Operator Boundary

For a fixed-state realization to use the same MAP, coadd, or deterministic-FLT
operator as its signal parent, exact membership, projection, grouping,
coefficients, normalization, support-authorized row selection, boundary/edge
treatment, response handling, and publication domain are fixed or identically
prescribed. Re-estimation or selection from a realization creates another
method and changes the inference target.

If `UNC_k` selects or constructs a later filter, the immutable successor graph
is `UNC_k -> FLT_(k+1) -> GEN_(k+1) -> UNC_(k+1)`. Neither the later filter nor
later uncertainty mutates or validates `UNC_k` retroactively.

## Validity, Support, And Profiles

Sample validity, parent named-use eligibility, GEN input admission,
realization-member completion/QC, UNC ensemble admission, estimator validity,
covariance-domain support, and STD admission are separate. Assignment cannot
clear a cause, make a missing occurrence valid, or create support. A finite
payload, completed member, generic flag, or pass for another use has no
universal veto or rescue effect.

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

## Downstream Ownership

- FLT owns deterministic and inference-bearing filter transfer, response,
  support, edges, covariance transformation, and local validity.
- SRC/MODE/BEAM own source, Pointing, OOF, and Beammap estimators and their
  consumer-specific qualification.
- FRUIT owns source subtraction/add-back, feedback, learning, recurrence,
  convergence, restart, and adaptive selection.

NOI may use a content-bound deterministic fixed FLT operator or provide
declared companions to consumers. It does not validate a consumer merely
because the consumer used them. Re-estimated Wiener and FRUIT routes remain
unavailable without their exact inference/feedback boundaries.

## Claim Separation

Algebraic contract correctness, implementation conformity, representation
fidelity, numerical validation, observational/statistical calibration,
achieved performance, readiness, and production authorization are separate.
No Stage B scientific derivation may infer the latter claims from historical
execution evidence.
