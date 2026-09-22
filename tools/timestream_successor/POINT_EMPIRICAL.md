# Saved POINT empirical companion

`run_point_empirical.py` reads immutable successor MAP bundles and their exact
PTC parents. It writes a new directory and never replaces the six MAP planes.
It reuses the explicitly selected saved descriptive measurement, with exact
parent hashes and numerical reproduction checked before companion publication.
The generic SCI-POINT zero-background compatibility method remains unavailable
in the recovered authority; this command does not instantiate or rename it.

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-empirical-mpl \
XDG_CACHE_HOME=/private/tmp/point-empirical-cache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_empirical.py \
  --maps /absolute/path/to/map-pass/products \
  --measurement /absolute/path/to/diagnostics/source/source-measurements.json \
  --profile saved-descriptive-v1 \
  --output /absolute/path/to/new-companion-directory
```

Run from the checked-out successor repository. `--request request.json` can
supply the explicit `Request` fields in `point_empirical.py`. The effective
request is always serialized. The default mask is three **measured major-axis
FWHM**, with two/three/four-FWHM background sensitivity. A fixed-template disk
uses radius three measured FWHM; apertures use one, 1.5 and two measured FWHM.
Neither changes the nominal calibration beam or the actual source-fit domain.

The fixed-template measurement is `sum(t*M)/sum(t*t)` on its complete disk,
with unit-peak shape/centroid frozen to the saved source fit. Signed blank
measurements are retained. No offset or annulus is subtracted. Aperture
measurements are raw sums multiplied by pixel angular area, in
`mJy / nominal beam * arcsec^2`. The recovered nominal FWHM and peak-calibration
unit do not authorize a new integrated-flux conversion; no Gaussian beam-area
conversion to mJy is made here.

The nonlinear refit inserts the fitted Gaussian source (without its additive
fit constant) into complete real blank 40-arcsec stamps and reruns the exact
saved seven-parameter Gaussian-plus-constant procedure. Source and blank
selection are distinct. Failed refits are recorded and withhold the aggregate
scatter; finite bound-limited results are included in the scatter and counted,
never silently discarded. Too few complete matched stamps makes that particular
diagnostic unavailable, even when compact-template/aperture diagnostics exist.
The empirical scatter is never divided by sqrt(number of placements).

Blank selection uses support, source exclusion and template-squared-weighted
coverage within +/-30% of source coverage, never signal values or quietness.
Complete footprints are mandatory. A greedy disjoint subset is selected first,
then additional placements with two measured FWHM center spacing. All positions,
counts and overlap limitations are retained. Coverage-only spatial
exchangeability is an assumption exposed by the quadrant and mask diagnostics,
not a statistical theorem. No inverse-square-root hit-count law is imposed.

Outputs include REPORT.md, source-measurements.ecsv, exact request/provenance,
per-array background.json, measurements.json with every blank/refit outcome,
blank/refit location tables, background-companion.fits, split-maps.fits,
split-report.json and three diagnostic figures. Source refit and aperture
measurements are not interchangeable. Raw A-B split maps compare interleaved
whole scans from already-cleaned PTC publications; they are not an ensemble of
independently relearned reductions or automatically full-map noise realizations.
All source and PTC inputs pass the existing exact MAP bundle/replay boundary.

RUN.json records companion-only wall/CPU/peak-RSS, code identities, dependencies,
and original MAP preservation. FILES_SHA256SUMS binds the new products. This
stage makes no full-chain response, calibration-uncertainty, physical total-flux,
significance, FRUIT-admission or inverse-variance-weight claim.

Independent-realization validation:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-empirical-mpl \
XDG_CACHE_HOME=/private/tmp/point-empirical-cache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/Users/gwilson/tolteca/bin/python tools/timestream_successor/validate_point_empirical.py \
  --output /absolute/path/to/new-validation-directory --realizations 48
```

Five cases cover white noise, spatial correlation, additive offset, incomplete
coverage and non-Gaussian wings. Every independent realization and empirical
scatter is retained. Exact known-kernel linear variance provides a second check.
The comparison uses RMS within-map scatter versus between-realization scatter;
its finite Monte Carlo precision is reported. Boundary-limited optimizer
outcomes remain part of the preserved procedure; tests verify their accounting
and nonconvergence handling. Model mismatch is measured separately using the
noiseless fitted amplitude versus the true source peak. None of these map tests
validates recovery through PTC.
