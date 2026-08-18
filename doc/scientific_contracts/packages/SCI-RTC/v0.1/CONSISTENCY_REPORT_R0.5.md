# SCI-RTC v0.1/r0.5 consistency report

Status: author-side semantic and structural review; not conformity evidence

## Boundary checks

- Paired $x/r$ exists in every role and a missing partner is malformed input.
- Independent member validity is not collapsed into pair validity.
- Upstream mapping remains upstream authority; RTC retains but does not infer it.
- Raw $x$ is not called Stokes I; $r$ remains diagnostic; polarimetry is excluded.
- SCI-CAL consumes only $x$ after RTC; no calibrated RTC branch is introduced.
- RTC atmospheric-template removal is distinct from target SCI-CAL atmosphere.
- No automatic $r$ correction, $r$ calibration, $r$-as-$x$ donor, or plateau
  stitching appears.

## Operator and state checks

- Despike/replacement precedes shift estimation; shift events create explicit
  masks, guards, segments, and reset/carry state.
- Atmospheric-template removal precedes later filters and carries response or
  bound authority for noncommutation.
- Representative occurrence remains exactly acquired $(d,Mn)$, distinct from
  support centroid or largest response coefficient.
- Actual attempts, maximum attempts, and accepted plans are distinct and
  satisfy $K\le A\le A_{\max}$.
- Pair identity, cross-coordinate covariance, support, causes, response,
  uncertainty, and provenance remain in the atomic bundle.

## Claim checks

No statement asserts scientific approval, implementation conformity,
representation fidelity, observational performance, science qualification,
validation completion, or production readiness. Open numerical and
methodological choices have typed unavailable consequences; closed r0.5
architecture is not reopened as an owner question.
