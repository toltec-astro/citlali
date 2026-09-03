# SCI-RTC v0.1 change log: r0.1 → r0.2

Date: `2026-08-18`

## Scientific genre and organization

- Retitled the rationale **TolTEC Raw-Timestream Conditioning: Learn–Apply
  Filtering, Response, and Scientific Consequences**.
- Replaced the audit-first narrative with 12 science-team sections beginning
  with present status and learn → resolve → immutable apply.
- Added five explanatory diagrams: lifecycle, source spectrum/filter response,
  donor information flow, group-delay displacement, and prefilter/decimation
  aliasing.

## Formal authority

- Expanded definitions from 20 to 26 with learning population, learned
  evidence, resolved plan, apply state, online-adaptive estimator, and
  scientific filter design.
- Expanded equation tags from 24 to 30 with projected Gaussian crossing time,
  generic notch response, constrained FIR-order selection, FIR group delay,
  scan-direction displacement, and the lifecycle identity.
- Retained 12 bounded assumptions while broadening learned-mode and validation
  language.
- Expanded requirements from 54 to 70 and predictions from 26 to 38.
- Preserved all r0.1 identifiers and meanings unless explicitly strengthened;
  new authority is appended sequentially.

## Scientific clarifications

- Defined every filter by purpose, signal model, learned/selected quantities,
  resolved parameters, astronomical and calibration effects, and failure.
- Defined notch width/depth/Q, drift, state, overlap, and adaptation limits.
- Made FIR tap count an owner-constrained design result.
- Distinguished continuity surrogate, raw-domain scaling, donor admissibility,
  and scientific eligibility.
- Added the prior-authority/circularity rule for Beammap donor `flxscale`.
- Related decimation and filter choices to projected beam, scan speed and
  direction, aliasing, phase, and coordinate registration.
- Required complete RTC-plan compatibility for calibration transfer.
- Added design-variation studies and kept science qualification separate from
  algebraic correctness.

## Decision and traceability records

- Added eight r0.2 owner decisions (`OWNER-029`--`036`).
- Added an exact r0.2 rationale-to-contract crosswalk, consistency report, and
  cross-package follow-up list.
- No implementation, test, audit, reduction, or production evidence was used.
