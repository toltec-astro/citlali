# Temporal-change study and map tradeoffs

**Disposition: useful direct-change diagnostic; not ready for automatic admission
or map-safety bounds. No harm threshold selected. Optical cut remains0.10.**

The new fixed-reference test avoids the two earlier stationary false-veto examples
(p=0.172 and0.115). But its estimated-dependence intervals fail their nominal99%
coverage goal. Strong steps are detectable; small changes, drifts and temporary
excursions can be missed. A non-detection must not be called stability.

[Method and assumptions](METHOD.md), [work order/prior recovery](WORK_ORDER.md),
[temporal results](RESULTS.json), [map sweep](MAP_RESULTS.json) and
[known-noise control](ORACLE_RESULTS.json) preserve the exact study.

## What we learned from the temporal test

12,000 synthetic realizations,999 resamples each; another3,000 realizations in an
oracle diagnostic, plus160 deterministic map cases and256 map-noise realizations.
These are generated data only; no observed data, Citlali runtime, reduction or
frozen trial was read or executed by the study scripts. Main Monte Carlo runtime
was36.35s on the local Python3.13.2/NumPy2.2.3 Mac environment.

| Constant-response case | Flags at nominal1% | Actual coverage of nominal99% simultaneous intervals |
| --- | ---: | ---: |
| Independent Gaussian noise | 8/1000 (0.8%) | 97.5% |
| Correlated noise, true AR coefficient0.6 | 15/1000 (1.5%) | 90.9% |
| Correlated noise with changing amplitude | 13/1000 (1.3%) | 91.8% |
| Block offsets | 15/1000 (1.5%) | 90.2% |
| Weak excitation in last block | 13/1000 (1.3%) | 90.1% |
| Gaps | 17/1000 (1.7%) | 91.2% |
| Heavy-tailed innovations | 20/1000 (2.0%) | 91.1% |
| Quiet healthy detector | 10/1000 (1.0%) | 97.9% |
| Strongly correlated noise, true coefficient0.95 | 8/250 (3.2%) | 62.0% |

Monte Carlo uncertainty is in RESULTS.json. For example, the95% interval for the
heavy-tail flag rate is1.30–3.07%; the gaps rate is1.06–2.71%. Those intervals
exclude the nominal1% target. This is a measured limitation, not a new tolerated
false-rejection rate. These flags do not implement rejection, and the maximum
search accounts for shapes/panels within one detector, not all detectors in an
observation.

| Injected temporal behavior | Detected /250 | Missed /250 |
| --- | ---: | ---: |
| 5% midpoint step | 2 (0.8%) | 248 |
| 10% midpoint step | 29 (11.6%) | 221 |
| 30% midpoint step | 230 (92.0%) | 20 |
| 80% midpoint step | 250 (100%) | 0 |
| 10% full-observation drift | 8 (3.2%) | 242 |
| 30% full-observation drift | 46 (18.4%) | 204 |
| 30% one-block excursion | 13 (5.2%) | 237 |
| Late response becomes0.03 or−1 | 250 each (100%) | 0 |

These powers apply only to the declared reference, noise and sample geometry.
Full-block offsets prevent additive offsets from being called gain changes but
discard between-block response information. Adding a complete search over short
excursions is not a free fix: it changes the search and its uncertainty calibration.

Attribution also matters: drifting reference flags12/250, noisy reference3/250,
stationary nonlinear coupling248/250, isolated impulse56/250. A temporal flag
does not prove a detector's optical gain changed. Outliers require the existing
disturbance policy or a separately tested robust estimator; OLS here is not native
Huber fitting. No-excitation and constant-last-block cases are unavailable250/250
each, not confidently weak or stable. The complete-search unavailable policy
retains the conservative limitation described in METHOD.md.

## Why the uncertainty fails

The encompassing fit estimates AR coefficient about0.49–0.50 when the true value
is0.6. Its many fitted response/offset terms absorb low-frequency noise; resampling
then conditions on a noise model that is too optimistic. Even iid noise has a
negative fitted coefficient near−0.06 and undercoverage. This diagnoses an
important mechanism, not the exclusive explanation of every failure.

In a separately declared oracle control, the same contrast geometry with the
**known true** Gaussian noise law covers98.8%,99.0%,99.3% of1000 realizations for
true coefficients0,0.6,0.95. All three Monte Carlo intervals include99%. This
supports the direct-contrast construction while isolating noise-law learning as
a major remaining problem. The oracle is not deployable, does not repair the
learned test, and does not validate uncertain references or non-Gaussian cases.

Do not widen intervals by an empirically chosen multiplier or tune the confidence
level to make these cases pass. The next statistical design should account for
estimating the noise law jointly with response (e.g. a carefully tested restricted
likelihood/dependence-uncertainty treatment), include independent held-out seeds,
and measure coverage before making map-safety claims. This report selects no new
operational covariance model.

## Map-impact tradeoffs before choosing harm budgets

The toy is a four-pass cross-linked raster with one unit-FWHM beam, source peak20,
atmospheric RMS30 and measured map-noise benchmark0.9989 brightness units. Its
256 noise-only realizations give amplitude scatter1.839%, centroid scatter
0.00781/0.00816FWHM in x/y, and width scatter1.643%. These are toy scales, not
TolTEC uncertainties or array-specific forecasts. All160 source fits completed
inside their bounds.

For a30% midpoint response step:

| Affected coherent map-weight share | Atmospheric-only structured RMS / map noise | Amplitude bias, atmospheric only | Amplitude bias, all-signal gain | Independent-noise cost if dropped |
| --- | ---: | ---: | ---: | ---: |
| 0.1% | 0.00184 | 0.0084% | 0.0234% | +0.050% |
| 1% | 0.0184 | 0.0839% | 0.2339% | +0.504% |
| 5% | 0.0918 | 0.419% | 1.169% | +2.598% |
| 20% | 0.3673 | 1.675% | 4.675% | +11.803% |

At20% weight, the atmospheric-only case shifts the centroid0.01746FWHM and
width1.284%. These are fitted biases in the particular scan. If exactly that
artifact repeats in16 independent-noise observations, its background ratio grows
from0.367 to1.469. This is a conditional coherent-coadd scaling, not a simulated
real multi-observation coadd. Randomly signed, differently projected contamination
would behave differently.

The dominant lessons are:

1. The response-change percentage alone cannot define harm. A single tiny-weight
   contribution and a coherent high-weight cohort have very different consequences.
2. Map noise alone cannot protect calibration: all-signal gain adds source bias
   even when atmospheric-background contamination is nearly unchanged.
3. Deeper coadds can expose systematics hidden by single-observation noise.
4. A fixed constant-atmosphere cleaner is not learned ALS/FRUIT. This sweep neither
   estimates basis contamination nor predicts source-feedback amplification.

The sweep provides effect sizes, not safe/unsafe labels. We still need owner
budgets for pointing error (arcseconds), calibration bias (fraction), and false
structure at science/coadd angular scales. Those budgets should be translated
using real beam/scan/noise context later, not chosen to admit a particular saved
detector cohort. Production sensitivity cost also requires actual weight/coverage
and noise correlations; the independent-noise formula is illustrative only.

## Local checks and limitations

20 focused tests verify exact nested-regression equivalence, null/offset removal,
shared-panel dependence, deterministic output, missing excitation, invalid input,
gap handling, fixed-cleaner and map associations, source-gain distinction and
oracle covariance recursion. Tests are implementation checks; the statistical
coverage goal **fails**, regardless of passing unit checks. Full CTest, config,
build and baseline gates are not triggered because no application/config/build
code changed. Existing optical policy/tests remain unchanged.

Initial map checks exposed the same local NumPy matrix-multiply warning as the
earlier study, inside `pinv`. SVD with explicit bounded summation replaced that
multiply. No warnings are suppressed; final tests and studies run with warnings
as errors. No new thresholds or parameter retuning followed the observed failures.

Compact quiet-mode study outputs retain case summaries, counts, uncertainty,
parameters and deterministic seeds. Per-draw arrays and per-realization logs are
not persisted. The main result includes elapsed wall time, which naturally differs
on reproduction; scientific fields must reproduce exactly on this environment.

Reproduce from this checkout with `/Users/gwilson/tolteca/bin/python -B -W error`:

```bash
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /Users/gwilson/tolteca/bin/python -B -W error -m pytest -q -p no:cacheprovider validation/temporal_change_2026-10-29/test_temporal.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/gwilson/tolteca/bin/python -B -W error validation/temporal_change_2026-10-29/experiment.py --output /tmp/temporal-results.json
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/gwilson/tolteca/bin/python -B -W error validation/temporal_change_2026-10-29/oracle_control.py --output /tmp/temporal-oracle-results.json
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/temporal-study-mpl XDG_CACHE_HOME=/tmp/temporal-study-cache OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/gwilson/tolteca/bin/python -B -W error validation/temporal_change_2026-10-29/map_tradeoffs.py --output /tmp/temporal-map-results.json --figure /tmp/temporal-map-tradeoffs.png
```

Scientific: evidence after failed uncertainty-calibration gate; no admission
approval. Architecture: standalone study, no pipeline changes. Repository:
bounded study/status additions; frozen reductions, APTs and runtime untouched.
Independent exact-SHA review is recorded after candidate commit. Unity/Spack,
integration, push, activation and production validation were not performed.
