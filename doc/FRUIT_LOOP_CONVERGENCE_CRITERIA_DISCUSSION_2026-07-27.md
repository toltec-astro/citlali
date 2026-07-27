# Fruit-Loop Convergence Criteria Discussion

Date: 2026-07-27

Status: discussion draft; no production stopping policy adopted

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
9. Require stable valid support, weights, and learning state in the `apply`
   phase.
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

1. Complete all 108 real-source trajectories and repeat the tables by array,
   source, and frozen quality stratum.
2. Simulate each candidate stop and compare the stopped product with iteration
   9, including amplitude, centroid, PSF, source-aperture residual, and
   empirical noise.
3. Run the exact control/injected transfer subset for normal, marginal, and
   stress observations.
4. Decide the acceptable fraction of the photometric and pointing error
   budgets attributable to incomplete iteration.
5. Select separate amplitude, PSF, centroid, map, support, learning, and noise
   guards.
6. Only then implement versioned runtime convergence state and provenance
   under retained-debt item D15.
