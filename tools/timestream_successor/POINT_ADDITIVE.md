# Full-observation conditional additive POINT companion

Run `run_point_additive.py` with the same immutable MAP/source description and
verified empirical/pairing companion directories. Required profile is
`saved-descriptive-v1`. Example (from this repository):

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-additive-mpl \
XDG_CACHE_HOME=/private/tmp/point-additive-cache \
/Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_additive.py \
  --maps /absolute/path/to/map-pass/products \
  --measurement /absolute/path/to/source-measurements.json \
  --empirical /absolute/path/to/companion-r2 \
  --pairings /absolute/path/to/comparisons-r2 \
  --profile saved-descriptive-v1 --workers 4 \
  --trials 256 --sensitivity-trials 64 --output /absolute/path/to/new-directory
```

This separate diagnostic estimates candidate additive-noise scatter conditional on
the processed map/source model. It is not SCI-NOI detector-sign randomization,
a pending formal POINT error method, absolute calibration/attenuation uncertainty,
or a FRUIT decision. Existing upstream state and estimator weights are immutable.
See POINT_EMPIRICAL.md for the exact recovered saved fit and fixed template.
The Gaussian-plus-constant fit uses a **40 arcsec radius** about AST center.
Every free parameter and original peak search, initialization, bounds and explicit
solver setting are reused; covariance is never used as new fit weights. Both the
full-map fit and fixed-template measurement must numerically reproduce their
verified parent. All seven source parameters also match the existing companion.

The learned approximation is C(p,q)=s(p)s(q)rho(p−q). One ordinary source-masked
background mean is removed; broad/linear modes are not detrended. Five equal-count
coverage bins use second moments about that common mean, with n−1 normalization.
This is an empirical piecewise-constant amplitude versus coverage approximation,
not an inverse-square-root-hit law. Robust scale/tails are diagnostics only.
No residual is clipped. At least100 pixels/bin are required. Values outside the
observed background coverage range receive an explicitly invalid endpoint
continuation for assumption-dependent calculations, not empirical support.

Observed pair-product sums are divided by mask-pair counts (at least200), retaining
both spatial axes and negative correlations. Single-mean finite-window effects
remain; mask-pair division does not undo them. The base32-pixel lag square receives
a cosine-product taper, is embedded on a grid strictly larger than twice both
retained lag and measurement-stamp size, Fourier transformed, projected to
nonnegative spectral power, and normalized to mean1. Spectrum repair magnitude
and pair-model residuals are published. Unsupported/beyond-domain covariance is
unknown physically, even though the computational model supplies a value.
Fourier draws use real white noise filtered by sqrt(spectrum), a central crop,
and then spatial amplitude modulation. qTCq uses exactly that discrete spectrum.
A doubled Fourier domain tests numerical-domain sensitivity.

The clean baseline is the saved fitted source plus fitted constant; A0 is a fresh
application of the unchanged estimator to this noiseless baseline. Trials add
new zero-mean Gaussian correlated fields at the actual source position/depth.
The centered sample SD uses B−1, without division by sqrt(B). Mean shift is
reported separately and never applied to the original amplitude. Failed trials
withhold the primary aggregate; finite constrained outcomes remain included and
flagged. MC precision is conditional on the inferred model and does not quantify
model instability or authorize Gaussian observational tail probabilities.

Prespecified sensitivity: source masks3/4/5 measured major FWHM; lag16/32/48 pixels;
cosine-taper powers0.7/1/1.4; north/south training. Base256 trials, others64 using
common array/trial random streams. Region/coverage diagnostics compare unaveraged
2D covariance and directional slices; source/background network/chunk populations
come from original admitted MAP operator positions. Hit matching alone does not
establish exchangeability. Signed cumulative lag contributions are diagnostic
projections, not separate positive variance models. Optional local-linear fit
weights assess covariance range relevance but never replace the full refits.
Constant/gradient/80–320 arcsec sinusoidal perturbations expose responses to
unknown broad modes; their physical amplitudes are not inferred.

Earlier blank-template predictions use each placement's own amplitude field and
exact weights; overlap and variable means prevent interpreting their count as
independent. At most four deterministic existing source-plus-background refit
locations are simulated with32 trials each. Pure-noise unconstrained Gaussian
fits are never compared to source-centered uncertainty. Missing placements and
source-depth extrapolation remain unavailable rather than changing footprints.

Products: per-array summaries, every trial and constraint/failure, PSD/amplitude/
mask arrays, empirical pair counts/covariance, source-depth and regional/network
comparisons, all sensitivity models/trials, existing-blank comparisons, three
representative source/noise FITS pairs, diagnostic figures and exact input/source/
configuration/seed/runtime/checksum provenance. Pixel-noise amplitude FITS is
**not** an image of fitted-amplitude uncertainty. `regeneration.json` and
`model.npz` suffice to regenerate base trials with `draw_noise` and `measure`.

Independent inference validation extends `validate_point_empirical.fixture` and
its unchanged fitter, using independent moving-average kernel generators.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-additive-mpl \
XDG_CACHE_HOME=/private/tmp/point-additive-cache \
/Users/gwilson/tolteca/bin/python tools/timestream_successor/validate_point_additive.py \
  --outer 192 --inner 16 --workers 4 --output /absolute/path/to/new-validation
```

Each outer observation learns its own masked-background covariance; mean estimated
variance is compared with actual across-observation variance. Every noiseless
reference, model-mismatch shift, trial and bound/failure remains recorded. Cases
include white, anisotropic signed correlations, varying amplitude and incomplete
coverage, long modes, spatially changing correlation and non-Gaussian wings.
Gaussian fixed-template internal propagation uses a99% chi-square comparison;
outer calibration uses the predeclared sampling-aware tolerance in the work order.
Challenge cases remain scientific limitations even if scalar checks pass. No
synthetic source recovery through cleaning is validated. No optional detector-time
to-frozen-MAP simulation is performed; the available verifier requires complete
real parent artifacts and is not a standalone synthetic boundary.

Implementation review and model applicability are separate dispositions. A
numerically valid covariance plus passing tests is insufficient to establish that
background statistics represent the source. Read the per-array assessment before
using a candidate value as an observational error bar.
