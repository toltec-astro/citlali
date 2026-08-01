# SCI-CAL-001 owner decision brief

## Decision requested

**Do not select or implement a successor operator yet.** The evidence now supports a bounded primary candidate, but not a final versioned operator or operational domain.

When the missing raw evidence is supplied, evaluate `piecewise_linear_los_tau_v0` as the primary candidate and `pchip_los_tau_v0` as the challenger. Select piecewise-linear line-of-sight optical depth if it meets the preregistered one-percent raw-grid correction-error gate over the approved domain; it preserves exact anchors and introduces no unsupported curvature. Keep the cubic only as a stress test.

For lineage closure, preserve the recovered monochromatic convention at 272.73, 214.29, and 150.00 GHz relative to 225.00 GHz. Treat any passband-integrated replacement as a separately named scientific model change requiring an explicit spectral convention.

## Evidence now established

- Exact q25/q50/q75 raw NPZ grids are recovered with SHA-256 and matching TolTECA registry MD5 identities. q95 is still missing.
- The legacy coefficients are exactly reproduced after eight-decimal rounding by a degree-six `numpy.polyfit` in elevation radians over 31 raw nodes from 20 through 80 degrees in two-degree steps.
- The recovered fit uses `T(nu_band)/T(225 GHz)` at the four monochromatic frequencies; it does not integrate a TolTEC passband.
- The isolated degree-six ratio fit has worst recovered raw-node correction error `0.024111%` (q75/a2000). The owner-required full-sample-airmass anchor reconstruction using repair-base coefficients has worst raw-anchor error `0.427091%` (q75/a1100). This is not a claim about the current application correction; its missing sample-airmass factor remains separate mandatory repair scope.
- A post-hoc raw q50 leave-one-model-out check, predicted linearly in line-of-sight optical depth from raw q25 and q75, has worst correction error `0.012264%` (a1400). Using the full-airmass q25/q75 anchor reconstructions gives `0.243563%` (a1400). q50 was already inspected during provenance recovery, so this is not a preregistered or blinded holdout and not a full-domain result.
- The recovered nominal-frequency q25/q50/q75 raw surfaces have zero increasing-opacity or increasing-elevation monotonicity violations at line-of-sight-tau tolerance `1e-12`.
- Frozen phase-0 evidence finds material hard-selector discontinuities in all 36 tested above-q25 boundary/band/elevation rows; a continuous operator remains scientifically motivated.
- All three diagnostic candidates preserve the exact source anchors and approved zero-to-q25 identity, remain finite and positive, and are opacity-monotone on the 30--80 degree diagnostic grid.
- Candidate disagreement at above-q25 interval midpoints is at most `0.259200%`; legacy-fit leave-one-anchor-out error is at most `0.313498%`. Neither statistic is independent raw truth.
- Every exact-anchor candidate inherits the q95/a2000 elevation feature: `0.839827%` maximum wrong-way correction excursion on the 0.1-degree grid. Its disposition should follow inspection of the missing q95 raw calculation.

The partial raw errors are comfortably below one percent, but they do not cover q75--q95, independent intermediate profiles in all intervals, or an approved operational domain. They are provisional numerical representation evidence only. They do not establish 5--10% absolute flux accuracy or approximately 5% repeatability.

## Evidence and choices still required

The exact request is machine-readable in `owner_input_request.json`. The blocking decisions are:

1. Supply the q95 artifact registered as TolTECA datafile ID 461 with expected MD5 `0ca7b331823237767d26016d19bffb3d`, plus SHA-256 custody provenance.
2. Supply the historical `am` payload, argv/configuration, q profiles, MERRA percentile construction, site/geometry, and output directives if full atmosphere regeneration is required.
3. Approve physical intermediate-profile runs in each q interval and an operational zenith-opacity/aligned-elevation domain. No extrapolation is authorized. The present 30--80 degree range is diagnostic; recovered raw support is 20--80 degrees.
4. Choose the spectral lineage: recovered monochromatic convention (recommended here) or a newly versioned band-integrated convention. The latter also requires immutable passbands, detector/array aggregation, normalization, source spectrum, and quadrature rules.

The proposed eventual domain should be no broader than zenith `tau225` from zero through the source-derived q95 coordinate and aligned elevation from 20 through 80 degrees, and only after q95 verification, raw support, and held-out fidelity exist throughout it. Until then, no exact endpoints are approved and requests outside the owner-declared domain must fail closed.

## Separate gates and open dependency

Software correctness, atmosphere-representation fidelity, and observational performance remain separate gates. Many samples reduce stochastic error but do not reduce shared calibrator, Beammap-extinction, selector, or airmass systematics.

The late SCI-ALIGN-001 handoff `SCI-CAL-001-XAUD-001` remains open and held for CAL re-audit. Any eventual atmosphere operator may use aligned elevation only with explicit ordered sample identity, timing-gap/interpolation origin, duration, and original-versus-synthesized eligibility. It does not alter the atmosphere equation or justify broader work here.

This package stops before Citlali application changes, repair implementation, re-audit, Unity access, or coordination-registry edits.
