# SCI-FLT-001 coordinator amendment — 2026-08-05

Status: current coordination amendment; no implementation, re-audit, Unity
activity, application integration, production expansion, or new evidence
campaign is authorized.

## Authority and preserved history

This amendment updates only the live SCI-FLT-001 coordination record. It does
not alter the immutable original audit at
`800e8ae433f87d3fb7521fcb1a7fdf1d32532949`, its artifact
`doc/CONVOLVE_SIGNAL_UNCERTAINTY_AND_RESPONSE_CONTRACT.tex` (SHA-256
`8d336242bbc260273c4f8f2819a13f3150091a44cd6bb6692fca92abd2f6ae60`), the
2026-08-01 returned-evidence review (SHA-256
`2331c9f4c4809a97d7b3dcc1c187db92470bb14fb96bedc7cd5ab417df4b1357`), or
the immutable CAL-to-FLT handoff `SCI-FLT-001-XAUD-001`.

The MAP authority reconciled here is the accepted bounded contract at exact
application candidate `af0c849ce59a5f80e5efc8db435bb6662863052f`, final
independent re-audit `8fc716557ca78b0d220200a92be46fa3545797e9`, and canonical
coordination record `c7bb0214edfd57fddf31165923f08784dfd1b8c9`. Its separately
authorized application-mainline documentation child is
`d5015fe716971bf8ea617e8a187311bf5af05185`; its decision artifact has SHA-256
`3420e63145cdd776f0bc1dc3210515d6fdb787b598a1b8a103eccd14afec07e9`.

## MAP reconciliation

MAP's bounded estimator, normalization, product identity, distinction between
raw scientific validity and numerical computability, and local evidence are
accepted within SCI-MAP-001 scope. The exact accepted application candidate is
`af0c849ce59a5f80e5efc8db435bb6662863052f`; the separately authorized
application-mainline documentation child is
`d5015fe716971bf8ea617e8a187311bf5af05185`.

Accordingly, SCI-FLT-001's MAP dependency moves from `open` to `conditioned`,
not `satisfied`. The admitted MAP `weight_I` identity is a **nonprecision
gridding/normalization coefficient**. It is not formal inverse variance, and
FLT may not derive a variance, precision, covariance, uncertainty, S/N, or
significance claim from it. A filtered precision or variance identity requires
separate authorization from the applicable PTC/VAL/NOI contracts.

MAP validity remains conditioned on SCI-PTC-001 and SCI-VAL-001, and MAP
production remains `existing_use_only`. SCI-NOI-001 and SCI-NOI-002 remain
not started and block empirical significance and covariance claims. SCI-CAL-001
remains nonconformant and in progress; the completed bounded D007 atmosphere
evidence does not authorize absolute photometry. The CAL-to-FLT handoff remains
held for re-audit and continues to require CAL unit, response, covariance, and
validity authority before filtered absolute or significance claims.

## Owner direction SCI-FLT-001-D002 — aperture-photometry boundary

Authority: project owner. Status: owner_decided/approved 2026-08-05.

The primary expected scientific use of convolved maps is aperture photometry,
especially cross-band dust spectral-index beta work. No turnkey
point-source-amplitude estimator or automatic response correction is required.
Users performing point-source photometry from convolved amplitudes remain
responsible for applying the named kernel response.

`signal_I` remains a fixed convolved map amplitude. `kernel_I` must be the
corresponding mapmaking kernel convolved with the identical realized operator,
centering, and valid-region policy. Its unit-source-kernel convention must be
explicit and verified. An interior point-source peak may be recovered by the
signal peak divided by the corresponding convolved-kernel peak, subject to that
normalization and CAL authority. Arbitrary aperture/background weighting may
use the ratio of identically weighted signal and convolved-kernel sums when the
unit-source response convention applies. These are user-applied response
corrections, not new direct flux products or automatic pipeline corrections.

The smallest useful aperture-photometry contract comprises the convolved
kernel map, valid-region mask, pixel solid angle, and truthful normalization,
filter, input-kernel identity, peak, signed integral, and effective-beam solid
angle metadata under the unchanged map-unit convention. Exact keyword and
schema names are implementation details. Compatibility point-source aliases
remain clearly labeled convolved amplitudes, not automatic photometry. This
does not require a continuous edge-response plane. Aperture uncertainty/
covariance remains conditioned on SCI-NOI-002; absolute cross-band calibration,
passband/color correction, and cross-band calibration covariance remain
conditioned on SCI-CAL-001.

Read-only inspection of pushed `origin/codex/refactor-mainline` at
`d5015fe716971bf8ea617e8a187311bf5af05185` found both the sequential and
OpenMP paths set `filtered_map` from `mb.kernel[map_index]`, call
`run_convolve()`, then perform the analogous signal sequence. In convolve mode,
`run_convolve()` sets `denom` to one. This is implementation evidence only.
A later bounded successor must demonstrate exact sequential/OpenMP
signal/kernel equality and truthful identity/normalization metadata before
re-audit; it is not a current closure.

## Owner decision SCI-FLT-001-D003 — empirical calibration and covariance boundary

Authority: project owner. Status: owner_decided/approved 2026-08-05.

FLT retains one robust global empirical calibration of the formal spatial
pattern. Direct per-pixel jackknife variance and S/N remain diagnostics pending
SCI-NOI-002; neither is a scientifically admitted uncertainty, significance,
or confidence product. FLT is not authorized to add a spatial empirical model,
full covariance matrix, or long-term realization-stack requirement. The
identical realized operator and full realization provenance remain required.

Aperture uncertainty must come from blank apertures or a future compact
SCI-NOI-002 product, not an independent-pixel summation. Until the applicable
NOI and CAL authorities are complete, this preserves the fail-closed boundary
for covariance-dependent multi-pixel significance, morphology, confidence, and
feedback consumers. It resolves the remaining empirical-calibration and
filtered-covariance owner-policy choices without closing their implementation,
evidence, dependency, or re-audit gates.

### Operational realization-count note

This is an operational note, not a new FLT finding or implementation
requirement. Read-only inspection of pushed
`origin/codex/refactor-mainline` at `d5015fe716971bf8ea617e8a187311bf5af05185`
shows `NoiseConfig` accepts `n_noise_maps >= 0` when noise is enabled and has
no hard maximum. Current application mode configurations request 10 for
science and beammap, 5 for pointing, and 1 for OOF; `write_realizations` is
false unless explicitly requested. In-memory realization storage scales
linearly with the requested count.

The available resource-admitted high-count validation tier of 64 is neither a
universal requirement nor a default or beammap expectation. Beammap may remain
at 10; any higher beammap count is deferred to a later memory/resource study.
Routine configurations remain unchanged, no artificial hard ceiling is
introduced, and no FLT streaming or memory work is authorized.

## Owner decision SCI-FLT-001-D001 — fill-boundary policy

Authority: project owner. Status: approved 2026-08-05.

1. The same-map median fill is retained only as a numerical boundary device.
2. Edge- or fill-influenced pixels are never scientifically valid and receive
   no scientific weighting, significance, photometry, confidence, or feedback
   interpretation.
3. No computational effort is authorized to propagate, model, or estimate
   median-fill covariance for those invalid pixels.
4. The scientifically admitted region must be conservatively inset or eroded
   by the declared effective filter footprint so that no admitted output stencil
   reaches the fill. Numerical computability and scientific validity remain
   distinct states.
5. A later repair and re-audit must verify that fill cannot contaminate admitted
   signal regions, including sequential/OpenMP equality and explicit edge and
   guard fixtures.

SCI-FLT-001-F004 remains `open`. Its owner policy is resolved and bounded
remediation is approved, but implementation, exact-successor evidence, and a
fresh re-audit are still required. This decision does not retrospectively make
the reviewed partial Unity bundle a scientific-validity or covariance pass.

## Current restrictions and next gate

The package remains contract `proposed`, implementation
`conditionally_conformant` only under its recorded numerical assumptions,
validation `in_progress`, production `fail_closed`, verdict `amend`, and
re-audit `required`. F005--F009 remain open. In particular, filtered products
remain unavailable for scientific weighting, significance, photometry,
confidence, feedback, or a multi-pixel covariance interpretation.

All substantive FLT scientific-policy choices are owner-resolved by D001--D003.
The next gate is a separately authorized exact-successor repair/evidence and
fresh re-audit package that implements and verifies those bounded policies;
this amendment authorizes none of that work.
