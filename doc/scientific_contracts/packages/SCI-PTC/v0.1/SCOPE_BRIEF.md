# SCI-PTC — Correlated-Mode Cleaning And Detector Coefficients Scope Brief

Status: scientific-owner approved Stage A scope; frozen for Stage B authorship

Scientific owner: Grant Wilson

Proposed version/date: `v0.1`, `2026-08-19`

Approved source identifier: exact bytes bound by `AUTHOR_PACKET_MANIFEST.md`

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: Codex manager, `2026-08-18`; revised after two
  scientific scope reviews supplied by Grant on `2026-08-19`
- Existing material adopted: the frozen implementation-independent
  `SCI-PTC-001` core and approved PTC decisions D001--D006, with a binding
  supersession cover
- Existing material abstracted or excluded: current implementation and
  interfaces, audit findings, repairs, handoffs, tests, validation, Unity,
  production status, the optional transfer-study execution plan, and the raw
  model-protected-notch note
- Genuinely new scientific work: reconcile the reusable core with the
  `SCI-RTC -> SCI-CAL -> SCI-PTC` boundary; define the estimand, removed
  subspace, centering/scaling, coefficient taxonomy, support-aware fitting,
  conditioned-`r` diagnostic branch, post-fit detector refinement, grouping,
  rank objective, response companion, and map-center diagnostic ownership
- Proposed author references: this Scope Brief; the binding supersession cover
  paired with the exact frozen independent core; the sanitized conventions and
  ownership record; and the bounded method-reference summary
- Author-packet exclusions: all unlisted files and all implementation, audit,
  repair, test, reduction, validation, Unity, and production material

This opening is approved for Stage B authorship: `yes — Grant Wilson,
2026-08-19`.

## 1. Package Name And Scientific Purpose

**SCI-PTC — Correlated-Mode Cleaning and Detector Coefficients** defines a
support-aware operation that estimates and subtracts declared shared detector
modes while preserving exact knowledge of the astronomical modes that may also
be removed.

The organizing scientific model is

\[
y^{\rm CAL}_{td}=s_{td}+\sum_{k=1}^{K}m_{tk}a_{dk}+n_{td},
\qquad
y^{\rm PTC}_{td}=y^{\rm CAL}_{td}
-\sum_{k=1}^{K}\widehat m_{tk}\widehat a_{dk},
\]

or an exact family-specific equivalent. Here `s` is astronomical signal, `m`
is a shared temporal mode or supplied template, `a` is its detector loading,
and `n` is remaining detector noise. Correlation does not establish physical
origin: a fitted shared subspace may contain atmosphere, electronics,
calibration residuals, detector noise, astronomical signal, or mixtures.

PTC therefore owns not only the cleaned samples but the removed subspace,
additive-reference and null-space state, fit/application supports, fitted
gauge, and response consequences. Retaining the declared signal unit does not
prove preservation of absolute level, extended emission, compact-source
amplitude, detector combinations, or beam shape.

The package must distinguish the full data-dependent procedure from a
conditioned fixed-state operator and keep fitted mode loadings,
centering/scaling parameters, diagnostics, and downstream analysis/gridding
coefficients scientifically distinct.

The Stage B science rationale must lead with: the physical model and its
non-identification of origin; the removed subspace and astronomical loss;
centering/scaling/grouping/gauge; source protection; estimator families;
coefficient taxonomy; fixed-state versus full-procedure response; covariance,
validity, and distinct supports; and finally iteration, recurrence, products,
and open decisions. It must not read like a methods catalogue or engineering
state inventory.

## 2. Scientific Boundary

The primary v0.1 science branch begins only after:

1. SCI-RTC has produced its admitted conditioned-`x` output and complete
   response/support/influence state; and
2. SCI-CAL has applied the selected absolute detector calibration and target
   atmosphere operator, yielding an admitted top-of-atmosphere,
   point-source-equivalent detector quantity in mJy per fixed nominal beam,
   with explicit response and uncertainty state.

The primary branch ends with a PTC-transformed detector timestream in the same
declared unit and calibration convention, plus the removed-subspace,
additive-reference, validity/influence, realized-fit, coefficient, response,
covariance, and provenance state required by declared consumers. PTC preserves
the unit and calibration convention; it does not automatically preserve
point-source peak, extended-source response, absolute level, or beam shape.

V0.1 may also consume a separately authorized, PTC-compatible conditioned-`r`
diagnostic parent on an exact declared relation to the `x` grid. The base
authority admits `r`-only PCA for electronics/readout diagnostics. The `r`
coordinate is electronics-enriched, not electronics-only, and its optical-
leakage state remains explicit. PTC does not apply the `x` RTC/CAL operator to
raw `r` by assumption.

`PTC-OWNER-Q001` is resolved for v0.1: `r` analysis is diagnostic-only. An
`r`-derived temporal basis may not be fitted to or subtracted from calibrated
`x`, and an `r` diagnostic may not alter `x` fit membership, subtraction,
output retention, or coefficients. Any such control or cross-channel
subtraction is a successor capability requiring new owner authority.
Unconstrained joint `x/r` PCA is also deferred from v0.1.

The transformed `x` timestream is an authoritative intermediate state, not an
independent sky estimator. Optional SCI-MAP may consume it. When PTC is
disabled, no PTC product and no SCI-MAP product on the PTC-dependent v0.1
route are realized. A direct SCI-CAL-to-SCI-MAP route is outside this scope
and requires separate authority.

V0.1 does not authorize a raw-`Delta f/f` Beammap PTC role. Any current or
future Beammap common-mode cleaning in raw units remains deliberately unowned
by this package and is routed to cross-package follow-up.

## 3. Legitimate Inputs

The contract may admit only explicitly identified inputs appropriate to the
selected operation:

1. **Calibrated `x` signal:** a sample-by-detector SCI-CAL signal in
   top-of-atmosphere, point-source-equivalent mJy per fixed nominal beam,
   including exact once-only factor and target-atmosphere lineage,
   calibration validity/quality, conditional uncertainty, nuisance-correlation
   scope, and response status.
2. **Complete RTC parent:** conditioned `x`, exact sample grid/rate/time,
   temporal response, support/influence, operation state, immutable paired raw
   `r` parent identity, and every `r`-derived selector, segmentation,
   validity, response, or uncertainty lineage that causally affected `x`.
   PTC has no numerical raw-`r` science branch.
3. **Optional conditioned-`r` diagnostic:** a separately authorized signal
   with its own conditioning operator, unit, response, validity, detector/time
   identity, optical-leakage state, and exact relation to the calibrated-`x`
   grid. Missing compatibility makes the diagnostic branch unavailable.
4. **Response companion:** a named detector-time source template or kernel on
   the exact signal grid, with source model, amplitude convention, unit,
   detector identity, RTC/CAL response lineage, support, and declared role as
   a fixed-state companion or end-to-end injection parent.
5. **Identity:** observation, scan, coherent segment/chunk, sample time,
   detector occurrence and stable UID, array/network/group, stage, product
   role, internal estimator iteration, PTC pass, FRUIT recurrence, and parent
   identities. Dense positions are not external identities.
6. **Cause and support policy:** direct validity, finite/missing state, typed
   causes, and upstream influence accumulate without erasure. For every
   PTC-owned use, an immutable versioned policy combines a base predicate and
   every applicable fact/cause predicate conjunctively; any exclusion controls,
   while no exclusion plus an unknown required predicate yields
   `decision_unavailable`. The policy declares exact inputs, Boolean rule,
   missing behavior, scope, use, owner, and version. PTC preserves facts for
   downstream-science uses rather than deciding their owners' admission rules.
7. **Decision-stage disposition:** enough state to determine whether an
   occurrence participated in the fit, may receive the fitted subtraction,
   may be retained, may contribute a coefficient, and whether a classification
   change requires refit. At minimum, `fit_invalid`,
   `postfit_output_reject`, `weight_only`, advisory-only, and
   `fit_excluded_apply_allowed` behavior must be expressible without treating
   one token as a universal action.
8. **Masks and supports:** separately typed basis-fit, loading-fit,
   subtraction/application, output, coefficient/QC, kernel, empirical,
   simulation, and downstream supports, with polarity, frame, unit, parent,
   registration, boundary convention, and derivation status.
9. **APT and grouping state:** immutable detector binding and a declared
   within-array hierarchy. Base groups may include an array-wide common
   component, network/electronics components, and optional local or
   focal-plane coordinate-basis components. Fit order or joint fitting is
   explicit. Data-derived groups are learned state. Cross-array modes require
   separate spectral/beam/calibration/response authority.
10. **Selected PTC request:** estimator family, conjunctive scientific
    admission predicates, candidate ordering and tie rule, centering, scaling,
    gauge, candidate ranks/components, robust loss/noise metric,
    cause-to-support policy, bounded fit-refinement policy, source protection,
    coefficient families, response/kernel request, covariance request,
    simulation request, and output roles. Absence differs from an explicit
    value.
11. **Externally supplied source/residual state:** an optional source model,
    mask, residual, or prior-pass parent only when its owner, unit, coordinate
    frame, registration, validity, response, support, and recurrence identity
    are explicit. A mask protects only its declared model and support; it does
    not prove preservation of unmasked extended emission or other sky modes in
    the fitted subspace.
12. **Random state:** generator family/version, seed or stream identity, and
    input identity whenever randomness can change a scientific output.

Unknown unit, identity, order, grouping, calibration, fit admission,
conditioning, response, or required support fails closed for the affected
claim. Replacing a flagged value by zero is never equivalent to excluding it
from a fit unless the selected estimator proves that identity.

## 4. Required Outputs

For every realized PTC product role, the contract must provide or explicitly
mark unavailable:

1. transformed calibrated `x` on the declared output support and in the same
   unit/calibration convention as its admitted CAL parent, with no stronger
   response claim inferred from the retained unit;
2. the removed subspace or template family, modeled correlated signal,
   additive-reference state, centering/scaling state, null-space status, and
   the astronomical modes known or permitted to be attenuated;
3. direct validity, typed causes, causal influence/support, complete
   cause-to-stage support decisions, and distinct downstream-consumer inputs;
4. requested, effective, observation-resolved, and material realized PTC
   state, including selected samples/detectors/groups, centering/scaling,
   mode/subspace, gauge, rank, coefficients, internal iteration, PTC pass,
   convergence/exhaustion, source protection, and fallback state;
5. distinct coefficient products for fitted mode loadings,
   centering/scaling parameters, diagnostic coefficients, and downstream
   analysis/gridding coefficients, each with type, role, unit, gauge or
   normalization, support/group, lifecycle, numerical use, permitted
   consumers, and prohibited interpretations;
6. post-fit detector diagnostics and resolved classes, including residual PSD
   or variance, residual coherence with retained or removed modes, removed-
   variance fraction, loading magnitude/stability, leverage, segment/scan
   stability, source-template response, `x/r` diagnostic consistency, and
   residual non-Gaussianity or transient contamination where scientifically
   applicable. Every diagnostic declares its population or approved noise/
   signal reference, normalization, support, uncertainty, and policy role,
   and distinguishes detector pathology from astronomical signal, source-mask
   or model failure, calibration state, focal-plane position, and expected
   detector-sensitivity variation. Numerical thresholds remain owner-
   controlled policy inputs;
7. covariance status and the exact conditioning/approximation domain for any
   formal covariance, marginal variance, empirical scatter, response
   uncertainty, calibration/systematic term, selection uncertainty, or
   cross-coordinate covariance;
8. sample-domain PTC response and every requested response companion. A
   fixed-state conditional companion uses the frozen realized operator without
   changing the fit; a full-procedure response reruns the complete selected
   estimator from the immutable admitted CAL parent under a declared injection
   and may be a response family;
9. when requested and supported, the optional
   `estimated_map_center_point_source_response`, computed only as a declared
   functional of an exact source template, propagated sample-domain response,
   and exact named reference map operator; it is not the general PTC response
   or SCI-MAP authority;
10. for the optional conditioned-`r` branch, diagnostic modes/loadings,
    optical-leakage state, grouping, support, stability, and cross-coordinate
    comparison status without calibrated-`x` or sky-signal meaning;
11. immutable PTC-to-MAP parentage and material realized state affecting the
    declared consumer;
12. for persisted PTC TOD, an explicit role of `diagnostic_artifact` or
    `requested_derived_analysis_product`, with honest completion and output-
    failure policy; and
13. scientifically named diagnostics with estimator, unit, support, validity,
    parentage, and whether they are inert, advisory, policy-selecting, or
    response-changing.

Internal invertible scaling is inverted before publication. The additive
detector location removed by centering is not restored: detector `x` has no
scientifically meaningful optical DC response, so the conditioned output is
`P(x - lambda)`, not `lambda + P(x - lambda)`. Frozen-state response holds the
learned `lambda` fixed; full-procedure response re-estimates and again discards
it. No partial
artifact may be labeled complete. Required-output failure propagates. A
best-effort diagnostic failure does not invalidate an otherwise valid product
unless that diagnostic was declared required or entered selected policy.

Every fit-support-changing refinement refits one complete selected model from
the same immutable admitted SCI-CAL parent. A previously cleaned output is not
the numerical parent of the refit, and the final complete model is applied
once to that parent. Sequential residual fitting is admitted only as an
explicit stage of one complete hierarchical estimator with declared order,
cumulative removed subspace, response, covariance, and parentage.

## 5. Upstream And Downstream Responsibilities

- **SCI-RTC** owns raw paired `x/r` identity, raw-domain conditioning, donor
  replacement, temporal filters/notches, phase-zero sampling, conditioned-`x`
  response, causal support/influence, and the atomic bundle. RTC or another
  separately authorized conditioner must produce any PTC-compatible
  conditioned-`r` diagnostic and its relation to the `x` grid.
- **SCI-CAL** owns absolute `flxscale`, target atmosphere, once-only
  composition, the point-source-equivalent fixed-nominal-beam convention,
  calibration validity/quality, conditional uncertainty, nuisance
  correlations, and lineage. PTC consumes and transforms that quantity but
  never repairs or strengthens it.
- **AST/ALIGN** own time/coordinate mapping, frame, registration, detector
  binding, coordinate validity, and their uncertainty. PTC consumes only
  admitted abstractions needed by masks or response.
- **PTC** owns the selected correlated-mode estimand; basis and loading fits;
  centering/scaling/gauge; fit, application, and coefficient supports;
  bounded fit-diagnose-classify-refit state; transformed TOD; typed
  coefficients; sample-domain response; and covariance availability.
- **RTC or a separate temporal-conditioner** owns any temporal notch or line
  filter and its response. Hook placement before PTC does not transfer that
  science to PTC.
- **FRUIT or another recurrence owner** owns source-model subtraction/add-back,
  map feedback, external recurrence, pass parentage, and restart. PTC internal
  numerical iteration, a new immutable-parent PTC pass, and FRUIT recurrence
  are distinct.
- **SCI-VAL** may own common types, knowledge-state semantics, cause
  preservation, reusable policy-evaluation machinery, provenance
  requirements, and shared profile vocabulary. It does not own or invent PTC,
  MAP, RTC, or other package-specific scientific admission policies. PTC owns
  the composite predicates for PTC-local uses; every downstream named-use
  owner owns its own rule and consumes PTC's preserved facts.
- **SCI-MAP** owns sample-to-map estimation, gridding normalization, map
  support/response/covariance, coaddition, and any direct CAL-to-MAP route. An
  exact named reference map functional may be imported solely for an optional
  PTC diagnostic without transferring MAP authority.
- **NOI** owns empirical noise realizations, scatter/covariance calibration,
  and significance authority. It must bind exact PTC selection and response.
- **BEAM** owns beam/source inference and broader response interpretation. A
  PTC map-center diagnostic does not establish off-center, extended-source,
  arbitrary-morphology, universal-band, or universal-mode response.
- **FLT and SRC/MODE** own map filtering and source/pointing/OOF inference.
  PTC does not absorb those estimators.

## 6. Externally Imposed Conventions

1. The primary v0.1 `x` branch admits and emits only the SCI-CAL
   top-of-atmosphere, point-source-equivalent quantity in mJy per fixed nominal
   beam with explicit response. PTC preserves its unit and convention, not
   point-source peak, absolute level, extended-source response, or beam shape.
2. PTC retains the full RTC atomic parent, including raw-`r` identity and all
   causal `r` lineage. It has no numerical raw-`r` science branch. An optional
   `r` diagnostic must be a separately conditioned, PTC-compatible product.
3. Primary timestreams are samples by detectors. Stable UID/occurrence identity
   governs cross-product joins; dense positions are local only.
4. Observation, scan, segment/chunk, sample, detector, array, group, stage,
   product role, internal iteration, PTC pass, FRUIT recurrence, band, mode,
   and parent are distinct identities.
5. Missing, non-finite, invalid, rejected, disabled, automatic, unavailable,
   and zero are distinct. Finite does not imply eligible.
6. A flag is a cause. Causes accumulate without erasure. For every PTC-owned
   use, the selected policy combines its base predicate and all applicable
   fact/cause predicates conjunctively; one exclusion controls, while an
   unknown required predicate with no known exclusion yields
   `decision_unavailable`. Every composite declares inputs, rule, missing
   behavior, scope, use, owner, and policy/version. Downstream use owners apply
   their own policies to preserved PTC facts. No flag implies zero-fill,
   universal rejection, zero coefficient, or refit by name alone.
7. Only eligible finite samples enter fitted arithmetic. `fit_invalid`
   requires refit or fitted-state invalidation; `postfit_output_reject` and
   `weight_only` do not retroactively change the fit. A fit-excluded occurrence
   may remain application/output eligible when policy explicitly says so.
8. Direct ALIGN-synthesized or RTC-replaced occurrences are excluded.
   Noncenter transitive influence is preserved and handled by PTC's declared
   use-specific fit/output/response policy.
9. Every centering/scaling operation declares axis, population, support,
   weights, location/scale estimator, masks, boundary, unit, gauge, failure
   behavior, and resulting null space. Internal scaling is inverted before
   ordinary publication. The learned additive detector location is discarded,
   not restored; frozen-state response holds it fixed and full-procedure
   response re-estimates and again discards it.
10. Fitted loadings, centering/scaling parameters, diagnostic coefficients,
    and downstream analysis/gridding coefficients are distinct. Only an
    explicitly named analysis/gridding family may be consumed by SCI-MAP. A
    loading never inherits weight, precision, significance, sensitivity, or
    independent-noise meaning.
11. Base fitted modes remain within one array. Hierarchical group identity,
    order/jointness, and coordinate basis are scientific state. Data-derived
    groups are learned state. Cross-array fitting is unavailable without
    separately authorized spectral, calibration, beam, and response models.
12. Rank/component selection follows a finite declared candidate set and
    chooses the least aggressive candidate for which every required residual-
    contamination, astronomical-transfer, conditioning, support, stability,
    and QC predicate passes. A failed predicate is not compensated by a
    scalar score. Candidate ordering and deterministic ties are declared; for
    nonnested candidates, comparison uses the complete removed subspace and
    response rather than mode count alone. No universal mode count, variance
    fraction, eigengap, or singular-value threshold is authority. A later
    role-specific scalar utility requires separately declared authority.
13. Basis sign, scale, and rotations within a degenerate subspace are gauge.
    Scientific claims attach to the removed subspace and modeled subtraction,
    not an unpinned component vector.
14. Requested, effective, observation-resolved, learned/resolved, and realized
    state flow one way. Learned masks, groups, ranks, modes, thresholds,
    coefficients, and convergence branches are realized random quantities.
15. Response status distinguishes at least `computed_published`,
    `not_computed_or_not_requested_for_this_product`, `invalid`, and
    `unavailable`. Fixed-state conditional and full data-dependent response are
    distinct. Within SCI-PTC, full-procedure response reruns the complete
    selected PTC fitting, selection, rank, classification, and application
    procedure from the immutable admitted SCI-CAL parent. A whole-chain
    RTC-to-CAL-to-PTC injection is a separately named cross-package study
    requiring exact upstream owners and companions.
16. Full covariance may be unavailable. Unavailable uncertainty is never zero,
    and scalar coefficients are nonprecision unless stronger conditions are
    proved.
17. Shifted/null surrogates shift signal and associated producer facts,
    causes, support decisions, boundaries, and identity together. Insufficient
    support is unavailable or rejected, never a valid zero fallback.
18. Required outputs fail closed and cannot be represented as complete after
    partial publication.

## 7. Questions The Contract Must Answer

1. For each admitted estimator family, what exact objective/loss, noise metric,
   basis/loading parameterization, support, grouping, gauge, and null space
   define the fitted correlated signal?
2. What physical or statistical assumptions distinguish robust group common
   modes, fixed-template regression, masked/weighted PCA/SVD, and `r`-only
   diagnostic PCA?
3. Which causes map into basis fit, loading fit, application, output,
   coefficient/QC, kernel, empirical, simulation, and downstream supports, and
   what changes require refit?
4. What centering/scaling coordinate is used, which transformations are
   inverted, and which additive or multiplicative modes are irretrievably
   removed?
5. What within-array hierarchy is fitted jointly or sequentially, and how are
   overlapping array/network/local subspaces prevented from becoming order-
   dependent undocumented behavior?
6. What finite candidate ordering, conjunctive scientific predicates, and
   deterministic tie rule select the least aggressive admitted subspace; how
   are nonnested and degenerate/near-degenerate subspaces handled?
7. How does a source mask or residual protect only its declared astronomical
   model/support, and how are unmasked compact or extended sky modes treated?
8. Which fitted mode loadings, centering/scaling parameters, diagnostic
   coefficients, and MAP-facing coefficients are produced, and what are their
   gauges, units, uses, consumers, and prohibited interpretations?
9. Which residual, loading, influence, stability, source-response, and `x/r`
   diagnostics are evaluated against which approved noise/signal model or
   population reference, and which can change detector fit support,
   application, output, coefficients, or only advisory health state? What
   finite stop, oscillation, insufficient-support, and nonconvergence rules
   govern refit from the immutable admitted CAL parent?
10. How are internal estimator iteration, a new PTC pass with an immutable
    parent, and external FRUIT recurrence represented without conflation?
11. When is a conditioned fixed-state operator valid, and when do rank, masks,
    grouping, classification, clipping, or convergence require end-to-end
    perturbation response or explicit unavailability?
12. How is a response companion propagated through exactly the signal's
    realized fit/application/output support without entering or changing the
    fit, and when must the full learn/fit/select procedure be rerun?
13. How is `estimated_map_center_point_source_response` derived from an exact
    source template, propagated sample-domain response, and named reference
    map functional without becoming ordinary mapmaking authority?
14. Which formal covariance, empirical scatter, response uncertainty,
    calibration/systematic term, selection uncertainty, cross-coordinate
    covariance, and omitted correlation are supplied or unavailable?
15. What `r` optical-leakage, grouping, support, conditioning, and stability
    state makes an `r`-only diagnostic interpretable?
16. How is the resolved diagnostic-only `r` branch kept inert or advisory so
    that it cannot alter `x` fit membership, subtraction, output, or
    coefficients, and how is any requested successor capability rejected as
    unavailable?
17. What output role and material state are required for the in-memory
    PTC-to-MAP interface, diagnostic artifacts, and requested derived products?
18. Which analytic limits and perturbation tests falsify identity, null space,
    gauge, scale/unit behavior, missing-data support, group hierarchy, detector
    refinement, response companions, covariance, and one/two-pass parentage?

## 8. Non-Goals

SCI-PTC v0.1 does not:

- define RTC temporal filtering/replacement or silently condition raw `r` with
  the `x` operator;
- derive or alter SCI-CAL calibration factors, atmosphere physics, units,
  quality, or unresolved numerical authority;
- authorize raw-`Delta f/f` Beammap PTC, polarimetry, cross-array fitted modes,
  unconstrained joint `x/r` PCA, or other signal-unit roles;
- authorize `r`-derived subtraction from calibrated `x` or permit `r`
  diagnostics to control the `x` branch in base v0.1;
- infer physical origin from correlation or call a source mask proof of all-sky
  preservation;
- select AST coordinates or ALIGN timing;
- own temporal notch filtering merely because it executes near PTC;
- own iterative joint sky/common-mode estimation, correlated-noise maximum-
  likelihood mapmaking, FRUIT recurrence, model add-back, restart, MAP, NOI,
  BEAM, FLT, or source-fitting science;
- redesign mature PTC numerical algorithms, choose production thresholds,
  optimize hot paths, or infer correct science from current code;
- inspect or repair implementation, run tests/reductions/Unity, perform a
  validation campaign, or issue conformity/production-readiness claims; or
- require dense full covariance or exhaustive archival replay when no declared
  consumer or reproducibility claim needs it.

## 9. Allowed References

The proposed implementation-blind author packet contains only:

1. this owner-approved Scope Brief;
2. the exact frozen `SCI-PTC-001_INDEPENDENT_CORE.tex` at
   `01ee247461d6c19bc4db81ccac4fec21af162c88`, paired inseparably with
   [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md);
3. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md);
   and
4. [`AUTHOR_METHOD_REFERENCE_BOUNDARY.md`](AUTHOR_METHOD_REFERENCE_BOUNDARY.md),
   which supplies bounded paraphrases of six primary method references without
   authorizing their assumptions or asking the author to open the papers.

The exact content hashes and firewall are in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

## 10. Owner Decisions And Remaining Ambiguities

The following package-scope decisions are proposed for owner approval:

1. **PTC-SCOPE-D001 — ordered boundary.** V0.1 uses
   `SCI-RTC -> SCI-CAL -> SCI-PTC -> optional SCI-MAP`, including complete
   calibration uncertainty and response lineage.
2. **PTC-SCOPE-D002 — signal role.** The primary branch accepts and emits only
   the admitted SCI-CAL top-of-atmosphere, point-source-equivalent quantity in
   mJy per fixed nominal beam. Raw Beammap PTC is deferred explicitly.
3. **PTC-SCOPE-D003 — upstream availability.** Invalid or unavailable RTC/CAL
   authority makes the affected calibrated PTC product unavailable.
4. **PTC-SCOPE-D004 — operator ownership.** PTC owns correlated-mode fitting,
   subtraction, source-protection supports, additive/null-space state, and the
   rule that correlation does not identify physical origin. Temporal notches
   remain RTC/separate-operator science; recurrence/add-back remains FRUIT.
5. **PTC-SCOPE-D005 — route-specific downstream behavior.** PTC may run without
   MAP for requested transformed TOD. Disabled PTC produces no PTC or MAP
   product on this route; direct CAL-to-MAP is separate authority.
6. **PTC-SCOPE-D006 — author packet.** Stage B reuses the frozen core through
   the exact cover, sanitized conventions, and bounded method-reference
   summary; implementation/audit material remains excluded.
7. **PTC-SCOPE-D007 — flagged-sample semantics.** Every cause maps independently
   into basis-fit, loading-fit, application, output, coefficient/QC, kernel,
   empirical, simulation, and downstream supports. Zero-fill PCA and universal
   flag actions are not authorized.
8. **PTC-SCOPE-D008 — paired `r` diagnostic branch.** PTC may consume a
   separately conditioned, PTC-compatible `r` parent for `r`-only diagnostic
   PCA. Raw-`r` processing and unconstrained joint `x/r` PCA are excluded.
   Under resolved `PTC-OWNER-Q001`, `r` diagnostics are inert or advisory and
   may not alter `x` membership, subtraction, output, or coefficients;
   applying `r`-derived modes to `x` is outside base v0.1.
9. **PTC-SCOPE-D009 — post-fit detector assessment.** PTC supports a finite
   fit-diagnose-classify-refit process; fit-support changes require refit,
   while output-only and coefficient-only changes do not retroactively alter
   the fit. Every support-changing refit fits one complete selected model from
   the same immutable admitted CAL parent and applies the final model once.
   Sequential residual fitting is allowed only as an explicit ordered stage of
   one complete hierarchical estimator with cumulative subspace, response,
   covariance, and parentage.
10. **PTC-SCOPE-D010 — estimator families.** Base v0.1 admits robust group
    common modes, explicit fixed-template regression, masked/weighted PCA/SVD,
    and `r`-only diagnostic PCA. Cross-channel `r` templates are gated;
    iterative joint sky/noise and correlated-noise ML mapmaking are adjacent or
    successor authorities.
11. **PTC-SCOPE-D011 — grouping.** Base fitting is hierarchical within one
    array, with explicit array, network/electronics, and optional local or
    focal-plane components. Data-derived groups are learned state; cross-array
    fitting requires separate authority.
12. **PTC-SCOPE-D012 — mode selection.** A finite candidate set and named
    conjunctive science policy select the least aggressive candidate for which
    every required residual-contamination, astronomical-transfer,
    conditioning, support, stability, and QC predicate passes. Failed
    predicates cannot be offset by a scalar score. Ordering and deterministic
    ties are declared, and nonnested candidates are compared through their
    complete removed subspace and response. No universal rank or singular-
    value threshold is authority.
13. **PTC-SCOPE-D013 — kernel and response propagation.** Every requested
    companion follows the exact fixed realized operator and support. Fixed-
    state conditional response and full data-dependent injection response are
    distinct products. PTC full-procedure response begins from the immutable
    admitted CAL parent; whole-chain injection is separate cross-package work.
14. **PTC-SCOPE-D014 — estimand and null space.** The contract publishes the
    fitted correlated model, removed subspace, additive-reference state, null
    space, and permitted astronomical attenuation; unit retention is not
    response preservation.
15. **PTC-SCOPE-D015 — coefficient taxonomy.** Loadings,
    centering/scaling parameters, diagnostics, and downstream analysis/gridding
    coefficients are separate families; only the last may be MAP-facing.
16. **PTC-SCOPE-D016 — map-center diagnostic ownership.** PTC owns its
    sample-domain response. `estimated_map_center_point_source_response` is an
    optional functional of exact source and reference-map inputs, not general
    response or MAP authority.
17. **PTC-SCOPE-D017 — centering and scaling.** Every transform declares its
    axis, population, support, estimator, masks, units, reversibility, gauge,
    and null space; internal standardization is inverted before ordinary
    output.

One owner decision is now resolved:

- **PTC-OWNER-Q001 — diagnostic-only `r` (`resolved 2026-08-19`).** In the
  first implementation/base v0.1, `r` analysis is diagnostic-only and inert or
  advisory with respect to calibrated `x`. It may not supply subtraction
  modes or alter `x` membership, output, or coefficients. A stronger branch
  requires a successor owner decision.

Approved historical PTC D001--D006 remain binding to the extent preserved or
explicitly superseded. Stage B must place any further scientific choices in an
owner ledger rather than inventing defaults.

## 11. Independence Statement

This brief defines the scientific objects, boundaries, adjacent ownership, and
open question without prescribing current Citlali behavior as the answer. The
proposed packet contains only this brief, the reusable independent core with
its binding cover, sanitized conventions/ownership, and bounded primary-method
paraphrases. If those inputs are insufficient, the author must return a
precise owner question and stop rather than inspect implementation, audit
history, or unlisted literature.
