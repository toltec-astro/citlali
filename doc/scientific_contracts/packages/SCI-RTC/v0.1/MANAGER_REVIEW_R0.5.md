# SCI-RTC v0.1/r0.5 Manager Review

Date: `2026-08-18`

Disposition: the bounded r0.5 paired-coordinate revision is complete and
accepted for scientific-owner review and freeze disposition. This is not
scientific approval, implementation conformity, representation fidelity,
observational validation, science-impact qualification, or production
readiness.

## Independence And Authority

The revision remains implementation-blind. It applies the supplied r0.5
scientific-owner directive and the owner's two explicit ordering decisions
without inspecting source behavior, tests, audits, reductions, or production
evidence. The six shared files under `src/common/` remain the sole normative
authority and are imported exactly once by each audience view. The engineering
wrapper contains no independent displayed mathematics.

## Directive And Scientific-Consistency Review

- Every role begins with an exact paired raw $x/r$ occurrence, independent
  member validity, and retained upstream IQ-to-$x/r$ mapping authority.
- Raw RTC precedes SCI-CAL. SCI-CAL consumes conditioned $x$ only; $r$ is not
  calibrated, substituted for $x$, used as an $x$ donor, or silently mixed
  through an $x-\gamma r$ operation.
- Atmospheric and bright-source optical-leakage estimators retain distinct
  parents, response, support, uncertainty, and validity. Scalar versus
  frequency-resolved status is explicit.
- The owner-selected ordering is recorded: admitted despike/replacement
  precedes level-shift estimation, while the selected raw RTC atmospheric
  template precedes later temporal filters with an exact composed response or
  an owner-approved noncommutation bound.
- Accepted level shifts create paired detector/channel amplitudes, transition
  masks and guards, coherent plateau segments, and explicit reset/carry state.
  V0.1 does not silently stitch plateaus.
- Joint covariance, selector dependence, noncenter replacement influence,
  original-input replay, and actual-versus-maximum refinement-attempt counts
  remain explicit and mutually consistent.
- The twelve-section rationale explains the science, while exact state
  machinery, requirements, and predictions remain in the shared formal core.

## Mechanical, Build, And Visual Review

- Shared inventory: 38 definitions, 37 displayed equation tags, 12
  assumptions, 105 sequential requirements, and 63 sequential predictions.
- Records: 23 author decisions and 71 owner-ledger entries: 65 open, two
  resolved, and four deferred.
- Exact crosswalk coverage, stable identifiers, directive source hash, and
  once-per-view shared-core imports pass the package verifier.
- Both PDFs compile without warnings or errors. Each has a one-page contents
  section. The rationale is 47 pages with exactly 12 substantive pre-appendix
  narrative pages; the engineering view is 37 pages.
- All 84 final Poppler-rendered pages were inspected. No clipping, overlap,
  broken table, bad glyph, header/footer defect, orphaned requirement row, or
  unreadable page remains.
- Both canonical PDFs are US Letter, unencrypted, and contain no forms or
  JavaScript.

The remaining 65 open owner decisions retain named unavailable effects. No
numerical default, implementation result, conformity result, validation
result, or readiness claim was inferred during this review.
