# Provisional bright-POINT connected-support candidate

This companion implements only the owner's bounded POINT152389 experiment. It
reads immutable MAP/source/additive products, writes candidate images for review,
and performs no FRUIT iteration. It does not establish a production threshold,
blind detection significance, faint-source completeness or extended-source rule.

Run from the successor checkout with a new output directory:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-admission-mpl \
XDG_CACHE_HOME=/private/tmp/point-admission-cache \
/Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_admission.py \
  --maps /absolute/path/to/map-pass/products \
  --measurement /absolute/path/to/source-measurements.json \
  --additive /absolute/path/to/reviewed-additive-companion-r2 \
  --profile saved-descriptive-v1 --output /absolute/path/to/new-directory
```

The command requires the reviewed Sept23 companion manifest; it is deliberately
bound to this observational experiment. Existing MAP/PTC verification, exact
saved source measurement, noise-model provenance and generator/fitter code
hashes are checked. No covariance model is learned or uncertainty campaign rerun.

For each array, keep the original AST-centered40″ radius domain. The measured
centroid anchors a nearest pixel (zero-based floor(pixel+0.5), exact half ties
toward increasing index). D=M-b0 uses the saved fit's constant; trial centroids
and constants are remeasured per realization. Joint MAP support, finite map,
noise validity and positive finite pixel scale are mandatory. Z=D/(scale×factor).
Only the eight-connected threshold component containing the anchor is admitted.
Absent/invalid/below-threshold anchor outcomes are recorded, with no alternate
source or maximum fallback. Values on the component are exactly D, valid excluded
pixels are zero, and invalid/outside-domain pixels are NaN. The fitted background
and negative bowls are excluded. No smoothing, dilation, hole filling, taper,
rescaling or fitted-Gaussian replacement is used. Dilation/erosion are used only
to diagnose adjacency/boundary contact, never to alter the candidate support.

Exactly four controls: (threshold,scale multiplier)=(3,1),(2,1),(5,1),(3,2).
The last equals threshold6 under original scale; it is not an independent test
of the spatial noise model, or an upper bound on the true noise amplitude.

The unchanged seven-parameter fitter (POINT_EMPIRICAL.md) measures both unmasked
D and candidates over the same40″ radius disk. Missing support withholds that
full-footprint diagnostic; no smaller intersection is substituted. Candidate
fits are model diagnostics, not new observed flux measurements. Frozen original
fixed-template weights/footprint project D, candidate and signed excluded D−F,
with the same background convention on each; their projections add when supported.
Raw sums are not called integrated flux. Full parameters/constraints/failures,
area/extent, anchor ratios, disconnected islands and threshold-change masks remain
available; unchanged maximum pixels are not treated as robustness evidence.

Use precisely32 saved trial identities per array: seed23152389,array ID,trial0..31.
Generate baseline fitted Gaussian+constant plus fresh correlated noise using saved
PSD/source scale/support; raw refits reproduce existing trial parameters. Each
raw refit supplies its own centroid/background before applying all four controls.
Every case receives the same realization. Retain constrained/failed outcomes and
label partial distributions. This tests conditional Gaussian behavior only, not
actual source morphology, PTC recovery or a2000 spatial representativeness.

Per-array candidate FITS planes: CANDIDATE (mJy/nominal beam), ADMITTED (binary),
VALID (binary joint support), EXCLUDED (signed D−F), RATIO (dimensionless Z).
The common source-relative FITS supplies D, pixel scale, analysis-domain mask,
MAP support, noise validity and joint support. Pixel scale may contain endpoint
continuations where NOISE_VALID is false; these are never admitted. Trial NPZ
arrays have axes[trial,row,column], candidate cubes and admitted masks per case,
and separate static valid/domain masks. Failed raw measurements have NaN trial
candidates and empty masks, with cause/state in trials.json; they are not valid
zero astronomical models. JSON tables, source/code hashes, controls, seeds,
RUN.json profiles and FILES_SHA256SUMS accompany the figures and report.

One owner-facing ASSESSMENT.md records the recommendation after inspection.
The command does not automatically promote a candidate or choose an admission
threshold. Any subsequent FRUIT experiment requires separate authorization and
must use immutable parents, relearn input-dependent PTC state after subtraction,
reconstruct the total, and infer a revisable/removable replacement model.
