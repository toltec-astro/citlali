# Fruit-Loop Convergence Criteria Discussion

Date: 2026-07-27

Status: complete 108-observation evidence analyzed; no production stopping
policy adopted

## Immediate Decisions

The historical pointing-table `sig2noise` value is not a convergence or
scientific S/N metric. It is fitted amplitude divided by the standard
deviation of the complete map, so recovered source structure contributes to
its denominator. Population analysis retains it as
`legacy_peak_over_full_map_rms` for reproducibility and removes it from the
combined convergence candidate.

Convergence metrics must not share one numerical tolerance merely because they
are dimensionless. Amplitude, PSF shape, centroid, map change, valid support,
learning state, and noise health answer different scientific questions and
need separately justified limits.

## Extinction And The Amplitude Tolerance

Absolute extinction uncertainty is relevant to the final flux error budget,
but a fixed extinction correction is multiplicative and common to every saved
iteration of one observation. It therefore cancels from the fractional
iteration-to-iteration amplitude change. Extinction uncertainty is a reason
not to spend substantial runtime chasing scientifically irrelevant precision
far below the absolute calibration floor; it is not, by itself, the
statistical uncertainty of the convergence measurement.

The amplitude tolerance should instead be chosen from:

1. measured late-iteration repeatability;
2. the residual difference between a simulated early stop and the retained
   full sequence;
3. empirical point-source amplitude uncertainty in correlated map noise; and
4. the fraction of the final photometric error budget allocated to incomplete
   fruit-loop recovery.

## Preliminary Population Evidence

This evidence uses the first 50 complete Stage B observations downloaded on
2026-07-27. It comprises 150 source-associated array trajectories through
iteration 9 and is order-biased; it is not the final 108-observation result.

The endpoint test is the maximum absolute kernel-normalized amplitude change
over transitions 7 to 8 and 8 to 9:

| Candidate amplitude tolerance | a1100 | a1400 | a2000 | All arrays |
|---:|---:|---:|---:|---:|
| 1% | 43/50 | 28/50 | 36/50 | 107/150 |
| 2% | 50/50 | 39/50 | 44/50 | 133/150 |
| 2.5% | 50/50 | 45/50 | 49/50 | 144/150 |
| 3% | 50/50 | 47/50 | 49/50 | 146/150 |
| 4% | 50/50 | 48/50 | 50/50 | 148/150 |
| 5% | 50/50 | 50/50 | 50/50 | 150/150 |

A simulated amplitude-only rule requiring two consecutive passes no earlier
than iteration 6 gives:

| Tolerance | Resolved by iteration 9 | Median stop | Median residual to iteration 9 | 90th percentile residual | Maximum residual |
|---:|---:|---:|---:|---:|---:|
| 2% | 134/150 | 6 | 0.52% | 1.50% | 3.70% |
| 2.5% | 144/150 | 6 | 0.67% | 1.84% | 3.77% |
| 3% | 146/150 | 6 | 0.82% | 2.20% | 3.77% |
| 5% | 150/150 | 6 | 1.12% | 4.02% | 7.28% |

Five percent obtains complete endpoint yield by allowing scientifically
material remaining motion. One percent rejects many trajectories after their
remaining changes are already small compared with realistic absolute
calibration uncertainties. Three percent is a reasonable discussion starting
point: it gives 97.3% endpoint yield in this partial sample while keeping the
90th-percentile simulated residual near 2.2%. It is not yet an adopted value.

The nonzero maximum residual after a two-pass rule demonstrates that amplitude
alone is insufficient. A temporary small increment can be followed by renewed
motion, especially before learning state is demonstrably stable.

## Complete 108-Observation Evidence

The final analysis includes all 108 observations, 324 array trajectories, and
3,240 saved array maps through iteration 9. Stage A passes 16/16 job audits and
Stage B passes 92/92. The analysis includes empirical blank-sky point-source
S/N and one convergence plot per observation. No production policy changed.

Radio sources and planets are evaluated separately. The 75 unresolved radio
observations use each iteration's realized point-source kernel. The 33 Uranus
and Neptune observations use that kernel convolved with an observation-epoch
JPL Horizons uniform disk. This changes the expected planet FWHM materially:

| Array | Median planet-template major-axis broadening | Maximum |
|---|---:|---:|
| a1100 | 5.20% | 8.38% |
| a1400 | 3.07% | 5.72% |
| a2000 | 1.88% | 2.58% |

The morphology-aware amplitude-only two-transition simulation gives:

| Morphology | Tolerance | Resolved trajectories | P90 residual to iteration 9 | Maximum residual |
|---|---:|---:|---:|---:|
| Unresolved | 2% | 211/225 | 1.24% | 3.57% |
| Unresolved | 2.5% | 221/225 | 1.64% | 3.57% |
| Unresolved | 3% | 225/225 | 1.87% | 3.57% |
| Unresolved | 5% | 225/225 | 3.39% | 7.12% |
| Planet disk | 2% | 54/99 | 1.45% | 3.08% |
| Planet disk | 2.5% | 66/99 | 2.24% | 3.13% |
| Planet disk | 3% | 77/99 | 2.47% | 4.38% |
| Planet disk | 5% | 95/99 | 5.74% | 7.94% |

Three percent remains the best current amplitude candidate for unresolved
sources: it resolves every radio-source array trajectory while keeping the
P90 residual below 2% and the worst residual below 4%. Five percent is too
loose. The planet result does not justify loosening the global tolerance:
planet amplitudes are still moving at iteration 9 and require their own
continuation assessment.

The complete candidate V0 rule resolves 57/108 observations by iteration 9:
51/75 unresolved sources and 6/33 planetary disks. For these simulated stops,
the stopped-to-iteration-9 source-aperture residual is 1.35% at the median,
2.84% at P90, and 4.28% at maximum. Its median and P90 values in units of the
final source-free background sigma are 0.77 and 2.67. No stopped observation
exceeds the provisional 5% aperture-residual guard.

A strict-state variant that also requires exactly unchanged effective
sample-mask and detector-penalty counts resolves only 39/108, with median stop
iteration 8. Exact count equality is retained as a sensitivity analysis, not
silently promoted to the core rule: the scientific guard is stable realized
map support and weights while learning is in the `apply` phase. The
continuation blocks should determine whether changing event counts can remain
scientifically immaterial under those realized-state guards.

The unresolved population is not one undifferentiated continuation set:

- 57 observations satisfy V0;
- 23 are measurement-limited, primarily because the pointing Gaussian width
  is censored at its upper bound; and
- 28 have individually measurable but unresolved trajectories: 15 unresolved
  radio sources and 13 planetary disks.

More fruit-loop iterations are appropriate only for the 28 trajectory cases.
The 23 measurement-limited cases first need estimator review; repeating the
same censored fit does not establish a PSF.

The historical apparent S/N loss is not present in the scientific diagnostics.
At iteration 9 the median empirical point-source S/N ratios to the seed are
1.96, 2.88, and 3.07 for unresolved-source a1100, a1400, and a2000 maps. The
corresponding planet values are 2.78, 3.83, and 3.67. Background sigma is not
monotonically increasing and its median endpoint ratios are at or below 0.99
in every array/morphology group. The legacy peak/full-map-RMS ratio may still
fall because the growing source changes the denominator.

The calibration-reference conclusions remain separate:

1. **Astrometry:** qualified after the centroid gate. All 75 unresolved-source
   observations and 28/33 planets obtain two all-array centroid steps below
   0.1 arcsec; 107/108 obtain stable source association. The remaining five
   planets require trajectory or fit review.
2. **Effective PSF:** qualified only where the fit is interpretable. FWHM
   change alone becomes small in 108/108, but only 85/108 obtain two
   all-array, uncensored PSF assessments. A stable bound-hit is not a measured
   beam.
3. **Photometric amplitude:** real-source iteration convergence is established
   for the unresolved sample at 3%, but absolute photometric transfer is not.
   The controlled injected-source attenuation and external flux uncertainty
   remain separate error-budget terms.
4. **Associated science response:** still unmeasured. No pointing-derived
   correction is approved until an exact pointing/science injection shows
   predictive transfer under the science configuration.

## Candidate V0 Rule For Full-Population Simulation

Evaluate this rule offline against every retained iteration; do not enable
runtime early stopping yet.

1. Do not evaluate before absolute iteration 6.
2. Require two consecutive passing transitions.
3. Require every active array to pass independently.
4. Require valid, source-associated fits; a censored or unavailable PSF fit is
   a failed shape assessment, not convergence.
5. Require kernel-normalized amplitude change below 3%.
6. Test FWHM relative to the realized kernel at separate 3% and 5% candidate
   limits; do not silently inherit the amplitude tolerance.
7. Retain the current 0.1 arcsec centroid-step candidate until the scientific
   pointing requirement supplies a stronger authority.
8. Evaluate successive source-aperture and whole-map change separately. Test a
   5% whole-map candidate while developing a source-aperture residual
   normalized by empirical background noise.
9. Require quantitatively stable valid support and weights while learning is
   in the `apply` phase. Report a strict-state sensitivity variant that also
   requires unchanged effective mask/penalty counts.
10. Treat robust source-free background sigma and pixel roughness as health
    diagnostics. A provisional guard is no more than a 10% increase relative
    to the seed and no unexplained late rising trend; their S/N ratio is not a
    convergence metric.
11. Report formal `amp / amp_err` and empirical blank-sky PSF S/N, but do
    not require either ratio to become constant.
12. Retain `max_iters` as a hard bound and record a terminal reason of
    `converged`, `max_iters`, or `metric_unavailable`.

## Empirical Point-Source S/N

Population analyzer schema v2 applies a fixed circular Gaussian, using the
geometric-mean realized-kernel FWHM, to both the source and deterministic
blank-sky positions in the 40 to 120 arcsec source-free annulus. Each fit
solves amplitude plus a constant background with formal map weights. The
normal-scaled MAD of blank amplitudes divided by their formal uncertainties
calibrates the source amplitude uncertainty. At least 12 valid blank fits are
required.

This first empirical estimator accounts for correlated-noise scale errors and
coverage-dependent formal uncertainty while using the same amplitude
estimator on source and blank sky. Its estimator name, geometry, minimum count,
and scale definition are persisted in the analysis manifest. Before it becomes
a product-level scientific S/N, validate its bias and coverage with synthetic
sources, the exact injected-source pairs, and the full quality-stratified
population.

## Evidence Needed Before Production

1. Continue the 28 trajectory-unresolved observations in three-iteration
   checkpoint-v2 blocks; do not spend continuation runtime on the 23
   measurement-limited cases without changing the measurement.
2. Review the PSF estimator and planetary fit treatment for the 23
   measurement-limited observations.
3. Run the exact control/injected transfer subset for normal, marginal, and
   stress observations.
4. Decide the acceptable fraction of the photometric and pointing error
   budgets attributable to incomplete iteration.
5. Select separate amplitude, PSF, centroid, map, support, learning, and noise
   guards.
6. Only then implement versioned runtime convergence state and provenance
   under retained-debt item D15.
