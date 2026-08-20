# SCI-PTC v0.1 — Binding Author Supersession Cover

Status: scientific-owner approved; effective for content-bound Stage B
authorship

This cover is inseparable from the frozen
`SCI-PTC-001_INDEPENDENT_CORE.tex` at
`01ee247461d6c19bc4db81ccac4fec21af162c88`, SHA-256
`82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2`.
The frozen core remains immutable. Where it conflicts with this cover, this
cover governs SCI-PTC v0.1 authorship.

## 1. Package And Status Translation

- Historical `SCI-PTC-001` is predecessor material; the durable library family
  is `SCI-PTC`, proposed contract version `v0.1`.
- Audit-launch identities, application SHAs, finding hypotheses, source-trace
  plans, audit phases, implementation status, validation status, production
  status, and reopen instructions are historical context and must not enter
  the scientist-facing or engineering-facing contract.
- The reusable mathematical content is authority for derivation, not evidence
  that any implementation conforms or any observational claim is achieved.

## 2. Ordered Input Boundary

For SCI-PTC v0.1, replace the generic RTC-to-PTC input with:

`admitted SCI-RTC conditioned x -> admitted SCI-CAL calibrated detector signal
-> SCI-PTC`.

The admitted PTC signal is the SCI-CAL primary detector stream in the
top-of-atmosphere, point-source-equivalent mJy-per-fixed-nominal-beam
convention, with exact RTC parent, calibration factor and target-atmosphere
lineage, validity/quality, conditional and nuisance-uncertainty state,
detector/sample identity, and upstream response status. PTC emits transformed
`x` in the same declared unit and convention but does not automatically
preserve point-source peak, absolute level, extended-source response, or beam
shape.

An unavailable or invalid RTC/CAL state makes the affected calibrated PTC
science product unavailable. PTC must not reconstruct, repair, approximate,
or relabel missing upstream authority.

Raw-`Delta f/f` Beammap PTC, polarimetry, and other signal/unit roles are
outside v0.1. PTC retains the complete RTC atomic parent, including raw-`r`
identity and causal `r` lineage, but has no numerical raw-`r` science branch.
A separately authorized, PTC-compatible conditioned-`r` product may support
`r`-only diagnostic PCA. Under resolved `PTC-OWNER-Q001`, that analysis is
inert or advisory in base v0.1: it may not alter calibrated-`x` membership,
subtraction, output, or coefficients, and it may not supply an `x` subtraction
basis. Unconstrained joint `x/r` PCA is deferred.

## 3. Approved D001--D006 Authority

The author must apply the following successor decisions:

1. **Disabled semantics.** If PTC is disabled, no PTC centering, cleaning,
   coefficients, PTC product, or MAP product on the PTC-dependent v0.1 route
   is realized. A requested upstream terminal product may still be emitted.
   PTC may run without MAP for a requested transformed TOD. A direct CAL-to-
   MAP route is separate authority.
2. **Decision-stage validity.** Distinguish `fit_invalid`,
   `postfit_output_reject`, and `weight_only`. Only fit-invalid causes require
   refit or fitted-state invalidation; later output rejection or coefficient-
   only noncontribution does not automatically invalidate the fitted state or
   other detectors.
   Direct ALIGN-synthesized or RTC-replaced occurrences are excluded. The
   blanket core rule that all noncenter descendants with transitive influence
   are universally ineligible is superseded: preserve the cause and response
   effect, then apply PTC's explicit use-specific fit/output policy.
3. **Response identity.** PTC owns its sample-domain response. The optional
   `estimated_map_center_point_source_response` is a declared functional of an
   exact source template, propagated response, and exact named reference map
   operator, bound to band, mode, configuration, detector/mask/selection,
   realization, RTC/CAL parent, and calibration/validation/uncertainty status.
   It is not ordinary mapmaking or the general PTC response. Status distinguishes
   `computed_published`,
   `not_computed_or_not_requested_for_this_product`, `invalid`, and
   `unavailable`. The estimate is not universal off-center, extended-source,
   cross-band, or cross-mode authority.
4. **Coefficient and covariance meaning.** Fitted mode loadings,
   centering/scaling parameters, diagnostic coefficients, and downstream
   analysis/gridding coefficients are distinct families. Each has exact type,
   role, unit, gauge or normalization, support/group, lifecycle, numerical use,
   permitted consumers, and prohibited interpretations. Only an explicitly
   named analysis/gridding family may be MAP-facing. No family is formal
   precision, inverse variance, significance, sensitivity, or independent-
   noise authority absent complete proof. Full covariance is not mandatory;
   unsupported covariance is unavailable and dependent claims fail closed.
5. **Product and provenance burden.** The in-memory transformed TOD is the
   authoritative PTC-to-MAP intermediate, not an independent sky estimator.
   Final consumers record material PTC state. Persisted PTC TOD declares
   `diagnostic_artifact` or `requested_derived_analysis_product`; provenance
   and replay burden follow declared consumption, and partial output is never
   complete.
6. **Eligibility, fallback, and reproducibility.** Fitted-state arithmetic
   uses eligible finite samples. Surrogates shift signal and associated
   validity/eligibility together. Insufficient support is unavailable or
   rejected, never a valid zero fallback. Fallbacks have cause and stage.
   Material randomness preserves generator/version, seed, and input identity;
   every realized shift and selection uncertainty need not be stored unless a
   declared consumer depends on them.

## 4. Operator Ownership Corrections

- Preserve the core's abstract data-dependent cleaner, conditioned linear
  special case, common-mode/PCA identities, state lifecycle, response,
  covariance, coefficient, provenance, and falsification reasoning.
- Do not select current method names, defaults, thresholds, component counts,
  convergence values, or fallback behavior by inference.
- PTC owns its correlated-mode fit and application masks. A temporal notch or
  line filter remains RTC or separately named temporal-conditioning science,
  even if applied immediately before PTC.
- Model subtraction, add-back, map feedback, multi-pass recurrence, and
  restart are FRUIT or another explicitly declared recurrence owner's science.
  PTC may consume an exact model/residual/mask parent and realize one fit
  inside that recurrence without absorbing the recurrence.
- VAL, MAP, NOI, BEAM, FLT, and SRC/MODE retain the downstream responsibilities
  in the sanitized ownership record.

## 5. Scope-Review Additions D007--D017

The author must also apply these binding scope rules:

- **Scientific estimand and null space.** Organize the rationale around
  calibrated astronomical signal plus a fitted shared/template component and
  remaining noise. Correlation does not identify the component's physical
  origin. Publish the fitted correlated signal, removed subspace,
  additive-reference state, null space, and permitted sky attenuation.
- **Centering/scaling.** Declare axis, population, support, weights,
  location/scale estimator, masks, boundary, unit, reversibility, gauge, and
  null space. Invert internal standardization before ordinary output.
- **Flag/support semantics.** Map every cause independently into basis fit,
  loading fit, application, output, coefficient/QC, response companion,
  empirical, simulation, and downstream supports. Zero-filling is not missing-
  data exclusion. Express fit-excluded/apply-allowed behavior explicitly.
- **Estimator families.** Base families are robust group common modes, fixed-
  template regression, masked/weighted PCA/SVD, and conditioned-`r`-only
  diagnostic PCA. Joint sky/noise recurrence and correlated-noise maximum-
  likelihood mapmaking remain adjacent/successor authority.
- **Grouping.** Base fits are hierarchical within one array, with explicit
  array-wide, network/electronics, and optional local/focal-plane components.
  Joint versus sequential order is state. Data-derived groups are learned.
  Cross-array fitting needs separate spectral/calibration/beam/response
  authority.
- **Mode selection.** Choose the least aggressive member of a finite candidate
  set for which every required residual-contamination, astronomical-transfer,
  conditioning, support, stability, and QC predicate passes. A failed
  predicate is not compensated by a scalar score. Declare candidate ordering
  and deterministic ties; compare nonnested candidates through the complete
  removed subspace and response. No universal rank, variance fraction,
  eigengap, or singular-value threshold is authority.
- **Detector refinement.** A finite fit-diagnose-classify-refit process records
  stable support/subspace, new-threshold events, maximum refinements,
  oscillation, insufficient support, and nonconvergence. Fit-support changes
  require a new fit; output-only and coefficient-only decisions do not.
  Residual, loading, influence, stability, source-response, and `x/r`
  diagnostics declare their population or approved noise/signal reference,
  normalization, support, uncertainty, and policy role; distinguish detector
  pathology from sky/model/mask/calibration/position/sensitivity effects; and
  use only owner-controlled numerical thresholds. Every support-changing
  refinement refits one complete selected model from the same immutable
  admitted CAL parent and applies the final model once. A cleaned output is
  not the numerical parent. Sequential residual fitting is permitted only as
  an explicit ordered stage of one complete hierarchical estimator with
  cumulative subspace, response, covariance, and parentage.
- **Iteration identity.** Internal estimator iteration, a new immutable-parent
  PTC pass, and external FRUIT recurrence are different states and terms.
- **Response companions.** A fixed-state companion uses the exact frozen fit,
  supports, grouping, centering/scaling, rank, masks, detector classes, and
  fallback without influencing the fit. A full-procedure injection reruns all
  selected learning/fitting/classification/application from the immutable
  admitted CAL parent and may yield a response family. A whole-chain RTC-to-
  CAL-to-PTC injection is a separate cross-package study requiring exact
  upstream owners and companions.
- **Source protection.** A mask protects only its declared source model and
  support. It is not proof that unmasked extended emission or other sky modes
  survive.

`PTC-OWNER-Q001` is resolved diagnostic-only for the first implementation/base
v0.1. An `r`-derived temporal basis may not be fitted to or subtracted from
calibrated `x`, and `r` diagnostics may not control calibrated-`x` membership,
subtraction, output, or coefficients. Stronger use requires a successor owner
decision.

## 6. Core Statements Narrowed Or Removed

The author must not carry forward:

- the core's historical audit-status statements, source-inspection plan,
  finding/repair/reopen routing, or production restrictions as PTC science;
- a universal exhaustive serialization/replay requirement superseded by D005;
- any implication that a generic input unit authorizes raw Beammap or other
  non-CAL roles;
- any wording that calls the admitted or transformed calibrated detector
  signal intrinsically point-source peak rather than point-source-equivalent
  mJy per fixed nominal beam with explicit response;
- any assumption that the retained raw-`r` parent shares the conditioned-`x`
  grid/operator or may be numerically analyzed without separate authority;
- any blanket late-exclusion rule superseded by the D002 three-way
  decision-stage distinction;
- the blanket upstream replacement/synthesis descendant-ineligibility rule.
  Direct occurrences remain excluded; noncenter influence remains traceable
  and receives the declared PTC use-specific decision;
- any response vocabulary superseded by D003; or
- any claim that a scalar coefficient or map denominator is precision,
  significance, independent noise, or complete covariance.

## 7. Unresolved Authority

The core and this cover do not select a mandatory estimator family, numeric
threshold, rank/component policy, source-mask geometry, convergence tolerance,
coefficient family, response construction, covariance approximation, or
validation threshold. Cross-channel `x <- r` subtraction and `r`-controlled
`x` decisions are excluded from base v0.1 by resolved `PTC-OWNER-Q001`. The
Stage B author must derive only what follows from
the packet, state honest unavailable conditions, and place every remaining
owner choice in `AUTHOR_DRAFT_DECISIONS.md`.

The author may not open other files to resolve an ambiguity. It must return one
precise question to the manager when the exact packet is insufficient.

## 8. Frozen-Core Conflict Preflight

This cover explicitly supersedes or narrows every older core clause that could
conflict with the approved packet on: point-source-equivalent signal wording;
removed-subspace, additive-reference, gauge, and null-space state; per-cause
stage-specific flag/support semantics; coefficient-family taxonomy;
diagnostic-only conditioned `r`; immutable-parent post-fit refinement;
within-array hierarchical grouping; conjunctive least-aggressive mode
selection; and fixed-state versus PTC-full-procedure response companions.

No older core statement may be used to reintroduce point-source-peak meaning,
zero-filled missing data, universal flag actions, cross-array or cross-channel
authority, cleaned-output refit parentage, compensating scalar rank scores, or
whole-chain response ownership.
