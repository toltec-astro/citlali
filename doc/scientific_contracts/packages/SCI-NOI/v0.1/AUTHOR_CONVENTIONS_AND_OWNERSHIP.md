# SCI-NOI v0.1 — Author Conventions And Ownership

Status: proposed sanitized Stage B author input; not owner-approved

This extract contains only the upstream/downstream conventions needed for
implementation-blind authorship. It is not an implementation map.

## Parent Authority

- RTC owns its conditioning operations, learned state, response, causal
  influence, validity, and lineage.
- CAL owns signal quantity/unit meaning, calibration/extinction state,
  response/uncertainty, quality, and lineage.
- PTC owns the transformed sample, retention, segment, cleaning realization,
  analysis coefficient identity, response/covariance state, and application
  generation.
- AST owns occurrence coordinates, time identity, exact WCS/frame joins, and
  coordinate validity.
- VAL registers and evaluates exact owner-authored policies. It does not
  author NOI method, target, adequacy, or publication policy.
- MAP owns the ordinary map/coadd estimator, support, base-product validity,
  nonprecision coefficient, response/covariance disclosure, and immutable
  bundle identity.
- JINC owns its distinct signed estimator and its own limited product state.
  No ordinary MAP rule applies to JINC by analogy.

NOI preserves all parent meanings and causes. It may bind a fixed parent state
or explicitly rerun named upstream operations as a separately identified
relearned method, but it cannot silently redefine those operations.

## Stable Conventions

- Array identity is `a1100`, `a1400`, or `a2000`, never a container position.
- Observation, detector/channel occurrence, scan/subscan/segment, sample,
  realization, map/group/Stokes, WCS/frame, support, and product generation are
  separate identities.
- In-memory indices are zero-based; persisted FITS/WCS pixel coordinates are
  one-based. WCS, not memory layout, defines persisted spatial meaning.
- Requested, effective, observation-resolved, and realized states are distinct
  one-way lifecycle stages.
- Missing, unavailable, unsupported, invalid, and failed are explicit states,
  not undocumented zero, NaN, infinity, or empty-container sentinels.
- A required publication or identity-join failure prevents realized success.
- Assignment signs and correlation objects are dimensionless. A realization
  has its parent signal unit. Variance/covariance has squared signal unit.
  Inverse-variance/precision has inverse-squared signal unit. Standardized
  signal is dimensionless.
- Equal units do not establish equal scientific role.
- Enabled polarimetry and measured-R execution remain outside active authority.

## Fixed Versus Relearned State

Every generation method enumerates RTC, CAL, PTC, AST, MAP/JINC, filtering,
source-model, coadd, mask/support, and consumer-selection state as one of:

- immutable/fixed from an exact parent;
- rerun/relearned by an exact named procedure;
- not applicable; or
- unavailable/unknown with consequence.

“Relearned” is not a generic label. Two methods that rerun different stages,
training subsets, masks, weights, convergence rules, or random perturbations
have different identities. Their realization members cannot be pooled without
a separately derived mixture estimand.

## Same-Operator Boundary

For a fixed-state realization to use the same MAP/coadd/filter operator as its
signal parent, the exact membership, projection, grouping, coefficients,
normalization, support, boundary/edge treatment, response handling, and
publication domain must be fixed or identically prescribed. Re-estimation or
selection from a realization defines another method and changes the inference
problem.

## Validity And Support

Sample validity, realization-member availability/QC, empirical-estimator
validity, covariance-domain support, and standardized-product eligibility are
separate. Sign multiplication cannot clear a cause, make a missing occurrence
valid, or create support. A downstream finite value cannot promote an invalid
parent. A later named use may accept a limitation without rewriting the parent
claim.

## Uncertainty And Weight Meanings

The following roles are not interchangeable:

- second moment;
- conditional randomization variance/covariance;
- repeated physical-noise variance/covariance;
- calibrated empirical variance/covariance;
- marginal inverse variance;
- full or regularized precision;
- consumer-effective weight;
- PTC/MAP analysis or gridding coefficient; and
- standardized signal.

An empirical NOI weight remains a companion result. Any future mapmaking,
coadd, filter-fitting, or adaptive use of it requires an explicit consuming
method and cannot retroactively modify an immutable parent.

## Downstream Ownership

- FLT owns deterministic and inference-bearing filter transfer, response,
  support, edges, covariance transformation, and local validity.
- SRC/MODE/BEAM own source, Pointing, OOF, and Beammap estimators and their
  consumer-specific qualification.
- FRUIT owns source subtraction/add-back, feedback, learning, recurrence,
  convergence, restart, and adaptive selection.

NOI may provide declared ensembles or uncertainty companions to these
packages. It does not validate a consumer from the fact that the consumer used
them.

## Claim Separation

Algebraic contract correctness, implementation conformity, representation
fidelity, numerical validation, observational/statistical calibration,
achieved performance, readiness, and production authorization are separate
claims. No Stage B scientific derivation may infer the latter claims from
historical execution evidence.
