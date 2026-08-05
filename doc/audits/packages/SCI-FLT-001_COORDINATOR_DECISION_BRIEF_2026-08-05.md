# SCI-FLT-001 remaining owner decision brief — 2026-08-05

Record ID: `SCI-FLT-001-OWNER-BRIEF-2026-08-05`
Status: FLT-D001 and FLT-D002 owner decided; no implementation, evidence
campaign, repair, re-audit, Unity action, application integration, or
production change is authorized by this brief.

## Decided boundary

`SCI-FLT-001-D001` is approved: same-map median fill is only a numerical
boundary device, scientific admission must be eroded so no admitted output
stencil reaches fill, and fill-influenced pixels are invalid for scientific
use. No fill-covariance calculation is authorized for invalid pixels.

MAP supplies an accepted bounded signal, nonprecision gridding/normalization
coefficient, kernel, facts/support/validity, and centered coefficient-weighted
coaddition contract. It does **not** supply precision. PTC/VAL, SCI-NOI-001,
SCI-NOI-002, and CAL remain limiting authorities; CAL does not yet authorize
absolute photometry.

## Owner-approved SCI-FLT-001-D002 — aperture-photometry contract

Primary scientific use is aperture photometry, especially cross-band dust
spectral-index beta analysis. `signal_I` remains a fixed convolved map
amplitude; no automatic point-source correction is required or authorized.

`kernel_I` must be the corresponding mapmaking kernel convolved with the
identical realized convolve operator, centering, and valid-region policy as
`signal_I`. Its unit-source-kernel convention must be explicit and verified.
With that convention, point-source peak recovery may use the signal peak
divided by the convolved-kernel peak. Arbitrary aperture/background weighting
may use the ratio of identically weighted signal and convolved-kernel sums.
These are user-applied response corrections, not automatic photometry.

Persist the convolved kernel plus truthful normalization, filter, and
input-kernel identity, with inexpensive convenience metadata sufficient for
its peak, signed integral, pixel solid angle, and effective beam solid angle.
Exact keyword and schema names remain implementation details. Compatibility
point-source aliases remain explicitly labeled convolved amplitudes, not
automatic photometry.

CAL owns absolute calibration, passband/color correction, and cross-band
calibration covariance. SCI-NOI-002 owns aperture uncertainty and covariance.
This decision authorizes no direct flux product, aperture catalog, response
plane, engineering expansion, implementation, evidence run, or re-audit.

Read-only source inspection at pushed `origin/codex/refactor-mainline` commit
`d5015fe716971bf8ea617e8a187311bf5af05185` characterizes the current code:
the sequential and OpenMP paths each assign `mb.kernel[map_index]` to the
convolve input, call `run_convolve()`, then perform the analogous signal
sequence; their convolve denominator is one. This is implementation evidence
requiring focused signal/kernel equality, centering/valid-region, and
identity/normalization metadata tests at a later exact successor and fresh
re-audit. It is not closure of F005 or F006.

## Still-held scientific-policy choices

### SCI-FLT-001-D004 — empirical calibration policy

Choose the intended scientific status of any empirical filtered product:

1. Keep empirical products diagnostic and not scientifically interpretable
   until SCI-NOI-001/002 are approved.
2. Permit a later named global empirical calibration policy after the NOI
   contracts define its realization, calibration-region, and covariance limits.
3. Reserve a spatial empirical model for a separately scoped successor rather
   than folding it into this amendment.

No choice converts MAP `weight_I` into precision or permits significance before
the required NOI and validity evidence.

### SCI-FLT-001-D005 — filtered covariance and downstream use

Choose the downstream contract after D001 admission is enforced:

1. Keep every multi-pixel, morphology, confidence, and feedback consumer fail
   closed until a covariance representation is approved and validated.
2. Name the smallest covariance representation or calibration that a specific
   downstream consumer needs, together with its consumer allowlist and
   validation gate.
3. Defer all filtered covariance consumers to a separately scoped successor.

This choice must not repurpose a diagonal variance, `coverage_bool_I`, support,
or response as covariance, confidence, or feedback gain.

## Bounded return template

For D004--D005, return the selected option (or an explicit deferral), the
intended product/consumer boundary, and any required acceptance evidence. Do
not add source files, output formats, simulations, campaigns, or new estimator
modes unless a later authorization names them explicitly.

## Proposed next owner question

Should empirical filtered products remain diagnostic until SCI-NOI-001/002
are approved, or should the owner reserve one named global empirical
calibration policy for a later bounded successor?
