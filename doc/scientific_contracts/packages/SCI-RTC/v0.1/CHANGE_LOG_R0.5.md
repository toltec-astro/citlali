# SCI-RTC v0.1 r0.5 change log

Status: implementation-blind revision record

## Preserved

R0.5 preserves every r0.4 identifier and the learn--resolve--apply,
complete-plan, replay, response, support, covariance, provenance, phase-zero,
role-plan, and claim-layer architecture.

## Changed

- Replaced the single-channel input premise with mandatory exact paired raw
  $x/r$ coordinates and independent member validity.
- Renamed the native-row symbol to $m_{\rm native}$.
- Added upstream IQ-to-$x/r$ mapping identity and covariance requirements.
- Added optical-response/leakage models, separate atmosphere and bright-source
  diagnostics, scalar/frequency status, and pre/post response identity.
- Added persistent paired level-shift events, transition masks/guards,
  segmentation, reset/carry, plateau diagnostics, multiple-event handling, and
  an explicit no-stitch boundary.
- Recorded the owner-selected despike/replacement-before-shift suborder.
- Recorded raw atmospheric-template removal before later filters and mandatory
  noncommutation accounting, distinct from downstream SCI-CAL atmosphere.
- Clarified $x$-only SCI-CAL handoff, diagnostic $r$, and exclusion of
  polarimetry, automatic correction, $r$ calibration, and $r$ donor use.
- Replaced ambiguous direct-cell language with exact representative occurrence
  $(d,Mn)$ in the acquired pre-replacement stream.
- Corrected bounded refinement to distinguish actual attempts $A$, configured
  maximum $A_{\max}$, and accepted plans $K$ without artificial no-ops.
- Reorganized the twelve-section rationale around paired coordinates, shifts,
  leakage, filters, segmentation, downstream interpretation, and validation.

## Inventory delta

- Definitions: 29 to 38.
- Displayed equation tags: 31 to 37.
- Assumptions: unchanged at 12, with amended scope.
- Requirements: 82 to 105.
- Predictions: 46 to 63.
- Author decisions: 18 to 23.
- Owner ledger: 50 to 71 entries; 65 open, 2 resolved, 4 deferred.

No implementation, validation, science-qualification, or readiness result is
created by this revision.
