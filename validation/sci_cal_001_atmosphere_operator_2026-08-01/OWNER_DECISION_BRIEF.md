# SCI-CAL-001 owner decision brief

## Direction recorded and decision boundary

**Evaluate a separately versioned AM 12.2 successor to determine whether it warrants adoption. This is study authorization only, not operator adoption or an operational-domain declaration.**

Machine status is `evaluation_only_not_adopted`. Study results are pending and remain unbound from the living package.

1. **Selected evaluation path:** define and test a versioned AM 12.2 successor with an explicit copied-profile rule, H2O-scale construction, grid, spectral convention, unresolved-line warning policy, independent validation design, and bounded q95-excluding study domain.
2. **Historical provenance disposition:** retain the generic-q custody gap, including generic q95, as unresolved historical/diagnostic evidence. The copied products are distinct from the generic products, while their historical generic-generator association is not established. Do not describe the successor as reproducing or replacing the generic products.

Retain piecewise-linear line-of-sight optical depth as the baseline and PCHIP as the challenger only. Neither is authorized. A future adoption decision must pass exact-anchor, finite-positive-transmission, continuity, opacity-monotonicity, and fail-closed-support gates, plus the provisional one-percent independent model-representation gate throughout the declared successor study domain. Elevation monotonicity must pass within that domain or receive an explicit owner scientific disposition supported by independent model evidence. The sub-percent legacy q95/a2000 excursion remains diagnostic-only and is not a successor release gate.

## Evidence now established

- Exact generic q25/q50/q75 raw NPZ grids are recovered with SHA-256 and matching TolTECA registry MD5 identities. Generic q95 remains missing at expected datafile ID 461/MD5 `0ca7b331823237767d26016d19bffb3d`.
- The legacy coefficients are exactly reproduced after eight-decimal rounding by a degree-six `numpy.polyfit` in elevation radians over 31 raw nodes from 20 through 80 degrees in two-degree steps.
- The recovered fit uses `T(nu_band)/T(225 GHz)` at the four monochromatic frequencies; it does not integrate a TolTEC passband.
- The isolated degree-six ratio fit has worst recovered raw-node correction error `0.024111%` (q75/a2000). The owner-required full-sample-airmass anchor reconstruction using repair-base coefficients has worst raw-anchor error `0.427091%` (q75/a1100). This is not a claim about the current application correction; its missing sample-airmass factor remains separate mandatory repair scope.
- A post-hoc raw q50 leave-one-model-out check, predicted linearly in line-of-sight optical depth from raw q25 and q75, has worst correction error `0.012264%` (a1400). Using the full-airmass q25/q75 anchor reconstructions gives `0.243563%` (a1400). q50 was already inspected during provenance recovery, so this is not a preregistered or blinded holdout and not a full-domain result.
- The recovered nominal-frequency q25/q50/q75 raw surfaces have zero increasing-opacity or increasing-elevation monotonicity violations at line-of-sight-tau tolerance `1e-12`.
- Frozen phase-0 evidence finds material hard-selector discontinuities in all 36 tested above-q25 boundary/band/elevation rows; a continuous operator remains scientifically motivated.
- The diagnostic exact-anchor candidates preserve source anchors and the approved zero-to-q25 identity, remain finite and positive, and are opacity-monotone. Their `0.839827%` q95/a2000 wrong-way feature belongs to the legacy q0--q95 diagnostic and remains useful characterization evidence, but q95 is outside the selected successor study.
- C1 tests the legacy-anchor candidates against 16 copied AM 12.2 profiles over the broader legacy q0--q95 diagnostic range. Its quoted a1100 maxima, `1.738766%` and `1.738068%`, occur for `LMT_MAM_75` at modified-secant tau225 `0.294218`, in the upper q75--q95 interval. Whether that point is outside the eventual study domain depends on the still-unapproved upper endpoint. C1 remains post-discovery stress evidence, not a blinded successor holdout or an authorization result.
- A distinct native GCC-15 AM 12.2 build exactly reproduces all five parsed fields and numeric lines for all `180/180` annual cases. All cases retain the accepted unresolved-line warning/status 1. Co-staging and matching output-build identity do not prove that the exact copied Linux ELF generated the historical files.
- R1 passes its preregistered 0.1% frequency-resolution diagnostic: the 10-MHz maximum correction difference from 1 MHz is `0.000340%`. Status-1 unresolved-line warnings persist, so this is not a clean software-success or warning-policy decision.

Canonical P1 directly ran all 100 copied-profile/H2O-scale hypotheses over 50,001 frequencies and 31 elevations. It preserves 100/100 exact parsed T225 anchors and produces 100 scale, 1,200 metric, and 1,050 coefficient rows. Its 13,667 referenced AM runs comprise 9,792 status-0 anchors and 3,875 complete status-1 full grids; other warnings, errors, and failed canonical attempts are zero.

| Generic target | Separate rank | Rank-one profile | H2O scale | RMS residual | Maximum residual | Maximum correction error (fraction) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| q25 | transmission | MAM5 | `1.81225445269332575` | `0.00511939194` | `0.0226840` | `7.79414741` |
| q25 | Rayleigh-Jeans | DJF5 | `3.01439309124786581` | `0.777548 K` | `3.5928 K` | `834.500816` |
| q50 | transmission | MAM25 | `0.915696647246186712` | `0.00323305754` | `0.0195999` | `0.998458439` |
| q50 | Rayleigh-Jeans | DJF25 | `2.02963214820032256` | `0.604530 K` | `2.2234 K` | `736.370583` |
| q75 | transmission | DJF50 | `1.88602893644962655` | `0.00156476455` | `0.0103363` | `4.83405417` |
| q75 | Rayleigh-Jeans | DJF75 | `1.01048455031671569` | `0.492757 K` | `1.8464 K` | `14.1911868` |
| q95 | nominal ratio surface | DJF25 | `6.88363302058917359` | `0.00541090730` | `0.0188189832` | `0.0119094929` |

Transmission and Rayleigh-Jeans ranks are separate; the correction column for a Rayleigh-Jeans winner is the transmission-derived correction error for that same profile and is not part of its Rayleigh-Jeans rank. No composite score or near-exact cutoff is used. The frozen addendum's descriptive statement that the closest same-percentile family is DJF is not a registered ranking.

P1's complete-grid result is a provenance mismatch: no q25/q50/q75 profile passes one percent over all 0--500 GHz samples. Even the smallest maximum correction errors are `97.968871%` (q25/MAM25), `99.845844%` (q50/MAM25), and `98.987223%` (q75/annual50). This must be distinguished from the legacy nominal frequencies: all 25 hypotheses pass one percent at 272.73, 214.29, and 150.00 GHz for q25/q50/q75, with worst error `0.665829%` at q75/a1100/JJA5. For q95, only the weaker 93-point ratio surface is available; none passes one percent, the best maximum is `1.117452%` (annual25), and the RMS winner's maximum is `1.190949%` (DJF25). For q25/q50/q75, direct copied-AM `atmTaun` is authoritative on the candidate side, while the generic truth NPZs contain only `atmTtx`, so P1 reconstructs truth-side line-of-sight tau as `-log(atmTtx)`. q95 is ratio-only and reconstructs both sides as `-log(T_band/T_225)`. The frozen P1 report and manifest state this authority too broadly; that wording is interpreted as candidate-side-only and is superseded by this package-level clarification.

P1 is post hoc and cannot establish generic custody or serve as an independent operator holdout. The nominal-frequency results are provisional numerical diagnostics only. They do not establish 5--10% absolute flux accuracy or approximately 5% repeatability.

## Successor study inputs and later choices still required

The exact request is machine-readable in `owner_input_request.json`. The selected evaluation path still requires:

1. Select and version the AM 12.2 profile family and H2O-scale construction. P1 fitted scales narrow hypotheses but do not make this scientific choice.
2. Supply or approve genuinely independent intermediate-opacity model runs across the q95-excluding study range. P1 fitted scales and C1 copied profiles were inspected post hoc and are not substitutes.
3. Resolve structural gates within the successor study domain and later approve exact zenith-opacity/aligned-elevation endpoints only where raw and independent model support exists. No extrapolation is authorized; out-of-domain requests must fail closed.
4. Choose and version the spectral convention. A monochromatic choice fixes 272.73/214.29/150.00 GHz relative to 225.00 GHz. Band integration requires immutable passbands, aggregation, normalization, source spectrum, and quadrature rules.
5. Approve the treatment of AM's unresolved-line status-1 output; R1 does not make that policy choice.
6. Retain generic q95 and exact generic-generator custody as nonblocking historical provenance. Their absence does not prevent the successor evaluation.

No exact opacity or elevation endpoints are approved here.

## Separate gates and open dependency

Software correctness, atmosphere-representation fidelity, and observational performance remain separate gates. Many samples reduce stochastic error but do not reduce shared calibrator, Beammap-extinction, selector, or airmass systematics.

Zenith `tau225` must be applied with each eligible sample's full modified-secant airmass and top-of-atmosphere pivot `X_ref=0`. The late SCI-ALIGN-001 handoff `SCI-CAL-001-XAUD-001` remains open and held for CAL re-audit. Any eventual atmosphere operator may use aligned elevation only with explicit ordered sample identity, timing-gap/interpolation origin, duration, and original-versus-synthesized eligibility. It does not alter the atmosphere equation or justify broader work here.

This package stops before Citlali application changes, repair implementation, re-audit, Unity access, or coordination-registry edits.
