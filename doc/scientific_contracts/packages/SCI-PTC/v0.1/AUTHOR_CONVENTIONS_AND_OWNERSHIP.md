# SCI-PTC v0.1 — Sanitized Conventions And Ownership

Status: scientific-owner approved Stage B author reference

This record contains only stable conventions and conditional adjacent-package
boundaries needed for implementation-blind authorship. It is not evidence of
implementation, conformity, validation, or production readiness.

## Signal And Unit Boundary

- The v0.1 primary signal is the admitted SCI-CAL ordinary `x`/`xs` detector
  quantity in top-of-atmosphere, point-source-equivalent mJy per fixed nominal
  beam, with explicit response state.
- PTC preserves that unit and calibration convention through transformed `x`.
  It does not automatically preserve point-source peak, absolute level,
  extended-source response, or beam shape.
- Raw `Delta f/f`, `MJy/sr`, `Jy/pixel`, temperature, integrated photometry,
  extended-source calibration, cross-array fitting, and polarimetry are outside
  the primary v0.1 branch.
- The complete RTC bundle retains raw-`r` parent identity and causal `r`
  lineage. PTC has no numerical raw-`r` science branch. A separately
  conditioned compatible `r` product may enter `r`-only diagnostic PCA. In
  base v0.1 that analysis is inert or advisory and may not alter calibrated-
  `x` membership, subtraction, output retention, or coefficients.
- Fitted loadings, centering/scaling parameters, diagnostics, and downstream
  analysis/gridding coefficients are different families. A coefficient
  proportional to inverse signal squared is not thereby formal precision,
  total inverse variance, significance, or independent-noise authority.
- Unknown or unavailable uncertainty is never zero.

## Identity, Shape, And Ordering

- Primary detector timestreams are matrices with samples on rows and detectors
  on columns.
- Observation, scan, coherent segment/chunk, sample time, detector occurrence,
  stable detector UID, array, network/group, stage, product role, band, mode,
  internal estimator iteration, PTC pass, FRUIT recurrence, and parent are
  distinct identities.
- A detector index, row, column, map index, file order, or local integer is not
  a stable external identity. Cross-product detector joins require the admitted
  occurrence/UID relation.
- Sample time carries declared scale, epoch, and unit. Actual timestamps and
  gaps remain authoritative over a nominal rate.
- Flattening and storage order must be declared and must not change the
  mathematical identity of an operand.

## Validity, Support, And State

- Missing, non-finite, invalid, rejected, disabled, automatic, unavailable,
  and numeric zero are distinct states. Finite does not imply eligible.
- Direct validity and transitive causal influence are distinct. Compact
  over-approximation of influence is permitted; under-approximation is not.
- Only eligible finite samples enter fitted-state arithmetic.
- A flag is a cause, not an action. Policy maps each cause independently into
  basis-fit, loading-fit, application, output, coefficient/QC, kernel,
  empirical, simulation, and downstream supports. Zero-fill is not exclusion.
- `fit_invalid`, `postfit_output_reject`, `weight_only`, advisory-only, and
  fit-excluded/apply-allowed behavior are distinct. Only a fit-support change
  invokes refit or fitted-state invalidation.
- Direct ALIGN-synthesized or RTC-replaced occurrences are excluded. Noncenter
  transitive influence remains traceable and participates in PTC's declared
  use-specific admission, output, and response policy; it is not universally
  collapsed to automatic ineligibility.
- Basis fit, detector-loading fit, subtraction/application, output retention,
  coefficient/QC, kernel, empirical estimation, simulation, and downstream-
  science supports are distinct.
- A finite post-fit detector refinement records fit, diagnostics, resolved
  class, refit decision, and stable/exhausted/oscillating/insufficient-support
  stop state. Output-only and coefficient-only dispositions do not rewrite the
  fitted state. Residual, loading, influence, stability, source-response, and
  `x/r` diagnostics declare population or approved noise/signal reference,
  normalization, support, uncertainty, and policy role, and distinguish
  detector pathology from astronomical signal, source-model/mask failure,
  calibration state, focal-plane position, and expected sensitivity
  variation. Numerical thresholds are owner-controlled policy inputs.
- Every fit-support-changing refinement refits one complete selected model
  from the same immutable admitted CAL parent; a cleaned output is not its
  numerical parent, and the final model is applied once. Sequential residual
  fitting is permitted only as an explicit ordered stage of one complete
  hierarchical estimator with cumulative subspace, response, covariance, and
  parentage.
- Requested, effective, observation-resolved, learned/resolved, and realized
  state flow one way. Data-derived masks, groups, selections, modes, ranks,
  thresholds, coefficients, and convergence branches are realized random
  quantities even when computation is deterministic.
- Shifted/null surrogates shift signal and associated validity/eligibility
  together. Insufficient support is unavailable or rejected, not a valid zero.

## Response, Covariance, And Coefficients

- The organizing model separates astronomical signal, a declared shared or
  template component, and remaining detector noise. Correlation alone does not
  identify atmosphere, electronics, calibration residual, noise, or sky.
- The removed subspace, additive-reference state, centering/scaling, gauge,
  null space, and permitted astronomical attenuation are required state.
- Every centering/scaling transform declares axis, population, support,
  weights, estimator, masks, boundary, unit, reversibility, gauge, and null
  space. Internal standardization is inverted before ordinary output.
- Base fitting is hierarchical within one array. Array-wide,
  network/electronics, and optional local/focal-plane components declare joint
  versus sequential order. Data-derived groups are learned state. Cross-array
  modes require separate authority.
- A source mask protects only the declared source model and the exact support
  it covers. It does not establish preservation of unmasked extended emission,
  arbitrary morphology, or structure outside that support.
- Rank selection chooses the least aggressive member of a finite candidate set
  for which every required residual-contamination, astronomical-transfer,
  conditioning, support, stability, and QC predicate passes. Failure of one
  predicate cannot be compensated by a scalar score. Candidate ordering and
  deterministic ties are declared; nonnested candidates are compared through
  complete removed subspace and response. No universal mode threshold is
  authority.

- A fixed realized fit may define a conditioned operator. Data-derived
  selection, rank, mask, clipping, coefficient feedback, and convergence make
  the full procedure generally nonlinear and state-dependent.
- A complete response composes the admitted RTC and CAL response with every
  response-changing PTC stage. A partial kernel cannot be relabeled complete.
- PTC owns its sample-domain response. A fixed-state response companion uses
  the exact frozen realized operator and does not alter the fit. A full-
  procedure injection reruns all selected learning/fitting/classification and
  application from the immutable admitted CAL parent and may yield a response
  family. A whole-chain RTC-to-CAL-to-PTC injection is separately named cross-
  package work requiring the exact upstream owners and companions.
- The approved stored identity
  `estimated_map_center_point_source_response` is an optional functional of an
  exact source template, propagated sample response, and exact named reference
  map operator. It is not the general PTC response or MAP authority.
- Response status distinguishes `computed_published`,
  `not_computed_or_not_requested_for_this_product`, `invalid`, and
  `unavailable`.
- A map-center point-source response does not establish off-center,
  spatially varying, extended-source, arbitrary-morphology, cross-band, or
  cross-mode response.
- Formal covariance, empirical scatter, calibration/systematic uncertainty,
  response uncertainty, and selection uncertainty are distinct.
- Full covariance may be unavailable. No consumer may make a stronger claim
  that requires unavailable covariance.
- Every coefficient family has exact type, role, unit, gauge or normalization,
  support/group, lifecycle, numerical use, permitted consumers, and prohibited
  interpretations. Only an explicitly named analysis/gridding family may be
  MAP-facing; re-estimation creates a new realized state.

## Scientific Responsibility Boundaries

- **SCI-RTC** owns paired raw `x/r` identity, conditioned-`x`, raw-`r` parent
  lineage, donor replacement, temporal filters/notches, phase-zero sampling,
  local response, and causal support/influence. RTC or another separately
  authorized conditioner owns any PTC-compatible conditioned-`r` product.
- **SCI-CAL** owns `flxscale`, target atmosphere, factor composition,
  point-source-equivalent fixed-nominal-beam meaning, calibration
  validity/quality, conditional and nuisance uncertainty, response state, and
  lineage. Its unresolved authority may make a numerical parent unavailable.
- **AST/ALIGN** own coordinate/time mapping, frames, registration, detector
  binding, validity, and uncertainty.
- **SCI-PTC** owns the correlated-mode estimand, basis/loading fits,
  centering/scaling/gauge, support mapping, detector refinement, transformed
  TOD, typed coefficients, sample response, and covariance availability.
- **A temporal-conditioner** owns temporal line/notch filtering even when
  invoked next to PTC.
- **FRUIT or a declared recurrence owner** owns model subtraction/add-back,
  map feedback, external recurrence, pass parentage, and restart. PTC internal
  iteration and a new PTC pass remain distinct from that recurrence.
- **VAL** owns reusable eligibility and cause precedence.
- **SCI-MAP** owns sample-to-map estimation, gridding normalization, map
  response/support/covariance, coaddition, and any direct CAL-to-MAP route. A
  named reference functional may enter an optional PTC diagnostic only.
- **NOI** owns empirical noise realizations, calibration of scatter/covariance,
  and significance authority.
- **BEAM** owns beam/source inference and broader response interpretation.
- **FLT** owns map-domain filtering. **SRC/MODE** owns pointing, OOF, and
  source inference.

Applying or consuming another package's product does not transfer ownership of
its estimator or authorize PTC to strengthen an unavailable state.

## Output Roles And Failure

- The in-memory transformed TOD is the authoritative PTC-to-MAP intermediate,
  not an independent sky estimator.
- PTC may run without MAP for an explicitly requested transformed TOD. On the
  PTC-dependent v0.1 route, MAP does not run when PTC is disabled; no claim is
  made about a separately authorized direct CAL-to-MAP route.
- A persisted PTC TOD declares `diagnostic_artifact` or
  `requested_derived_analysis_product`. Persistence alone grants no stronger
  scientific role.
- Provenance and replay burden follow material scientific state and declared
  consumption. Exhaustive internal-state serialization is not universal.
- Required output failure propagates. A partial artifact is not complete. A
  best-effort diagnostic failure affects science only when that diagnostic was
  declared required.
- `PTC-OWNER-Q001` is resolved diagnostic-only for the first implementation/
  base v0.1. An `r`-derived temporal basis may not be fitted to or subtracted
  from calibrated `x`, and `r` diagnostics may not control calibrated-`x`
  membership, subtraction, output, or coefficients. Stronger use requires a
  successor owner decision.

## Sanitization Boundary

This record abstracts owner-approved scientific content and adjacent package
boundaries. It intentionally supplies no current implementation detail,
default, finding, repair, test, validation result, Unity evidence, production
status, or numerical threshold.
