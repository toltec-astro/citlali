# SCI-RTC v0.1 r0.6 change log

Status: implementation-blind bounded-correction record

## Preserved

R0.6 preserves every r0.5 identifier and the paired-input, leakage-diagnostic,
level-shift, learn--resolve--apply, complete-plan, replay, response, support,
covariance, provenance, phase-zero, role-plan, and claim-layer architecture.

## Corrected

- Withdrew the false r0.5 owner approvals for atmospheric-template subtraction
  and donor replacement before level-shift learning.
- Restricted atmospheric templates to diagnostic evidence and routed numerical
  common-mode removal to separate PTC or successor authority.
- Required original-pair spike masking/downweighting for shift learning, with
  donor replacement only after segmentation and within stable segments.
- Made conditioned $x$ the required numerical output and raw $r$ the immutable
  paired parent; a conditioned $r$ product now requires separate authority.
- Replaced the universal affine IQ mapping with a general nonlinear mapping
  plus explicitly local affine/Jacobian representations.
- Removed the $\zeta$ symbol collision by using $\epsilon$ for leakage and
  $\eta$ for residuals.
- Made donor and phase-zero equations explicitly $x$-domain and standardized
  “raw aligned paired parent” terminology.
- Strengthened role-specific plateau support and pre/post-shift optical-response
  comparison, including explicit unavailable and additive-only-block states.
- Restored the atmospheric-estimator shared-data/noisy-coordinate bias warning.
- Added the missing scientific falsifiers without claiming validation results.

## Inventory delta

- Definitions: unchanged at 38, with amended output and estimator boundaries.
- Displayed equation tags: unchanged at 37, with corrected semantics.
- Assumptions: unchanged at 12, with amended scope.
- Requirements: 105 to 108.
- Predictions: 63 to 71.
- Author decisions: 23 to 24.
- Owner ledger: 71 to 74 entries; 64 open, 5 resolved, 5 deferred.

No implementation, conformity, validation, science-qualification, or readiness
result is created by this revision.
