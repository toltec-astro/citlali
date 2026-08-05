# SCI-FLT-001 remaining owner decision brief — 2026-08-05

Record ID: `SCI-FLT-001-OWNER-BRIEF-2026-08-05`
Status: FLT-D001 through FLT-D003 owner decided; no implementation, evidence
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

## Owner-approved SCI-FLT-001-D003 — empirical calibration and covariance boundary

FLT retains one robust global empirical calibration of the formal spatial
pattern. Direct per-pixel jackknife variance and S/N remain diagnostics pending
SCI-NOI-002; they are not scientifically admitted uncertainty, significance,
or confidence products. Identical realized operator identity and complete
realization provenance are required.

No spatial empirical model, full covariance matrix, or long-term
realization-stack requirement is authorized in FLT. Aperture uncertainty uses
blank apertures or a future compact SCI-NOI-002 product, never an
independent-pixel summation. Until the applicable NOI and CAL contracts are
complete, covariance-dependent multi-pixel significance, morphology,
confidence, and feedback consumers remain fail closed. This resolves the
previous empirical-calibration and filtered-covariance policy choices; it does
not close implementation, evidence, dependency, or re-audit gates.

## Operational realization-count note

This is not a FLT scientific finding, implementation requirement, or new
default. At pushed `origin/codex/refactor-mainline`
`d5015fe716971bf8ea617e8a187311bf5af05185`, `NoiseConfig` validation has no
hard maximum and requires only `n_noise_maps >= 0` when enabled. Current mode
configurations request 10 for science and beammap, 5 for pointing, and 1 for
OOF. `write_realizations` is false unless explicitly requested, while
in-memory realization storage scales linearly with the requested count.

The available resource-admitted high-count validation tier of 64 is not a
universal requirement, routine default, or beammap expectation. Beammap may
remain at 10; any higher beammap count awaits a later memory/resource study.
Routine configurations remain unchanged. This amendment adds no artificial
hard ceiling and authorizes no FLT streaming or memory work.

## Resulting coordination boundary

All substantive FLT scientific-policy choices are now owner-resolved by
D001--D003. The next coordinator/owner action is to authorize, if desired, an
exact-successor repair/evidence and fresh re-audit package against the stated
gates. It must not infer a new realization-count cap/default or authorize
application integration, production expansion, source changes, or a new
campaign from these decisions alone.
