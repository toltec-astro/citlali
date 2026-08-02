# SCI-CAL-001 owner decision brief

## Direction recorded and decision boundary

**The separately versioned AM 12.2 successor study is complete.  It is much
better behaved than the hard legacy selector, but no frozen candidate passes
the preregistered one-percent representation gate over the complete
q0--q75/20--80 degree study domain.  Machine status is
`numerical_adoption_evidence_fail`; adoption remains
`evaluation_only_not_adopted`.**

1. **Completed evaluation path:** AM 12.2, two frozen profile/H2O-scale lanes,
   two continuous line-of-sight-optical-depth operators, TolTECA v1 ECSV
   primary passbands, representative FTS challengers, source indices
   `alpha=-1,0,2,4`, and direct held-out midpoint calculations over q0--q75
   and 20--80 degrees.
2. **Historical provenance disposition:** retain the generic-q custody gap, including generic q95, as unresolved historical/diagnostic evidence. The copied products are distinct from the generic products, while their historical generic-generator association is not established. Do not describe the successor as reproducing or replacing the generic products.

No operator or operational domain is authorized.  The sub-percent legacy
q95/a2000 excursion remains diagnostic-only and is not a successor release
gate.

## Brass-tacks calibration result

All four lane/operator candidates pass provenance, exact-anchor, positivity,
continuity, opacity/elevation monotonicity, fail-closed support, and complete
evidence-coverage gates.  They all fail the full-domain numerical decision at
the same clear-to-q25 point:

| Quantity | Result |
| --- | ---: |
| Worst primary-ECSV correction error | `1.159949%` |
| Location | a2000, `alpha=-1`, DJF5, q0--q25 midpoint, EL21 |
| Primary a2000 `alpha=0` maximum | `1.069577%` |
| Worst FTS-challenger representation error | `1.101467%` |
| Best above-q25 maximum (`conditioned_djf_v1` + PCHIP) | `0.163540%` |
| Simplest above-q25 maximum (`fixed_djf25_v1` + piecewise linear) | `0.288111%` |
| Maximum fixed-versus-conditioned lane difference | `0.049709%` |
| Maximum piecewise-linear-versus-PCHIP difference on holdouts | `0.123225%` |

The failure is therefore narrow, not a broad model breakdown: every primary
band/alpha combination other than a2000 `alpha=-1` and `alpha=0` passes, and
the q25--q50 and q50--q75 intervals are comfortably sub-percent.  The common
failure comes from the frozen analytic q0-to-q25 segment, so changing the
above-q25 lane or interpolant cannot cure it.

At the worst row the direct AM correction is `1.057175916` and the operator
returns `1.044913213`: a `1.159949%` flux under-correction.  The sign therefore
does not hide a cancellation or merely symmetric scatter.

For comparison, the legacy hard selector has instantaneous extinction-
correction jumps at q50/q75 of `0.267763%` through `7.735359%` across the
tested a1100/a1400/a2000 rows at representative EL30--80.  A discontinuity is
not itself a truth-error measurement, but it is a real calibration step that
the AM12 continuous candidates remove.  Against direct AM12 truth, the new
continuous representation is no worse than `1.159949%` over the complete
study and no worse than `0.288111%` above q25 for even the simplest candidate.

Passband definition is now the larger calibration ambiguity.  Direct truth
integrated through the representative FTS spectra versus the TolTECA v1 ECSV
passbands differs by as much as `3.336470%` in a1100, `0.408717%` in a1400,
and `3.474613%` in a2000.  Those differences are not interpolation error and
are large enough to matter within a 5--10% absolute-flux objective.

The post-result diagnostic is also informative but is not a new passing
study: restricting the already inspected rows to EL25--80 gives maximum
representation errors of `0.989845%` for the primary passbands and
`0.949738%` for the FTS challengers.  The corresponding maximum FTS-versus-
ECSV truth difference remains `3.047492%`.  Because EL25 was identified after
the EL20--80 result was seen, it cannot be used to relabel the frozen study as
a pass.

## Evidence now established

- Exact generic q25/q50/q75 raw NPZ grids are recovered with SHA-256 and matching TolTECA registry MD5 identities. Generic q95 remains missing at expected datafile ID 461/MD5 `0ca7b331823237767d26016d19bffb3d`.
- The legacy coefficients are exactly reproduced after eight-decimal rounding by a degree-six `numpy.polyfit` in elevation radians over 31 raw nodes from 20 through 80 degrees in two-degree steps.
- The recovered fit uses `T(nu_band)/T(225 GHz)` at the four monochromatic frequencies; it does not integrate a TolTEC passband.
- The isolated degree-six ratio fit has worst recovered raw-node correction error `0.024111%` (q75/a2000). The owner-required full-sample-airmass anchor reconstruction using repair-base coefficients has worst raw-anchor error `0.427091%` (q75/a1100). This is not a claim about the current application correction; its missing sample-airmass factor remains separate mandatory repair scope.
- A post-hoc raw q50 leave-one-model-out check, predicted linearly in line-of-sight optical depth from raw q25 and q75, has worst correction error `0.012264%` (a1400). Using the full-airmass q25/q75 anchor reconstructions gives `0.243563%` (a1400). q50 was already inspected during provenance recovery, so this is not a preregistered or blinded holdout and not a full-domain result.
- The recovered nominal-frequency q25/q50/q75 raw surfaces have zero increasing-opacity or increasing-elevation monotonicity violations at line-of-sight-tau tolerance `1e-12`.
- Frozen phase-0 evidence finds material hard-selector discontinuities in all 36 tested above-q25 boundary/band/elevation rows; a continuous operator remains scientifically motivated.
- The diagnostic exact-anchor candidates preserve source anchors and the
  protocol-frozen zero-to-q25 identity, remain finite and positive, and are
  opacity-monotone. Their `0.839827%` q95/a2000 wrong-way feature belongs to
  the legacy q0--q95 diagnostic and remains useful characterization evidence,
  but q95 is outside the selected successor study.
- C1 tests the legacy-anchor candidates against 16 copied AM 12.2 profiles over the broader legacy q0--q95 diagnostic range. Its quoted a1100 maxima, `1.738766%` and `1.738068%`, occur for `LMT_MAM_75` at modified-secant tau225 `0.294218`, in the upper q75--q95 interval. That point is outside the proposed q0--q75 follow-up support; the proposed support is not an authorized operational domain. C1 remains post-discovery stress evidence, not a blinded successor holdout or an authorization result.
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

## Owner decision now required

The frozen 20--80 degree study cannot recommend adoption by its own rule.
Choose one `LOW-001` scientific path before another selection study is
registered:

1. **Keep EL20 support.** Replace or augment the analytic q0 anchor with an
   explicitly physical dry-atmosphere/low-opacity construction, then freeze
   and test new q0--q25 opacity holdouts.  This addresses the actual failure
   mechanism without weakening the one-percent gate.
2. **Set a higher minimum elevation.** Declare whether EL25 is an acceptable
   proposed follow-up floor, retain q0 through the exact q75 selector anchor
   (`tau225=0.158313198574890929`), and run a new preregistered confirmation at
   opacity/elevation nodes not used to discover that floor.  Existing rows
   support this path numerically but do not constitute a blinded confirmation.
3. **Relax the representation gate.** Explicitly accept at least `1.159949%`
   (a practical rounded contract would need to be chosen) over EL20--80.  This
   is scientifically defensible only as an owner change to a numerical
   representation criterion; it must not be described as one-percent
   photometry.

Independently choose the `BAND-001` production passband convention.  The TolTECA v1 ECSV
files are immutable and reproducible, but the single-network FTS challengers
show up to `3.474613%` calibration sensitivity and are not an array-average
operational passband.  Either explicitly select the TolTECA ECSV convention
for a named successor version or supply detector/array-weighted operational
passbands with immutable provenance and repeat the band integrations.

If the owner chooses the EL25 path and the TolTECA v1 ECSV convention, the
protocol's simplicity tie-break conditionally points to
`fixed_djf25_v1` + `am12_piecewise_linear_los_tau_eval_v0` on closed support
`0 <= tau225 <= 0.158313198574890929`, `25 <= EL <= 80`, with fail-closed
behavior.  This is a precise follow-up candidate, not an adoption or
operational authorization.  The observed conditioned/PCHIP improvement above
q25 is only `0.124571` percentage point in worst correction error and remains
small compared with the unresolved passband sensitivity.

The exact `DOMAIN-001` support, `WARN-001` warning-status-1 policy, and later
`OBS-001` observational campaign also remain required.  Generic q95 and exact generic-generator custody stay nonblocking
historical provenance and do not enter any of these choices.

## Separate gates and open dependency

Software correctness, atmosphere-representation fidelity, and observational performance remain separate gates. Many samples reduce stochastic error but do not reduce shared calibrator, Beammap-extinction, selector, or airmass systematics.

Zenith `tau225` must be applied with each eligible sample's full modified-secant airmass and top-of-atmosphere pivot `X_ref=0`. The late SCI-ALIGN-001 handoff `SCI-CAL-001-XAUD-001` remains open and held for CAL re-audit. Any eventual atmosphere operator may use aligned elevation only with explicit ordered sample identity, timing-gap/interpolation origin, duration, and original-versus-synthesized eligibility. It does not alter the atmosphere equation or justify broader work here.

This package stops before Citlali application changes, repair implementation, re-audit, Unity access, or coordination-registry edits.
