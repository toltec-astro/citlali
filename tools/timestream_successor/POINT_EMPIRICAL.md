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

## Balanced reconstruction and local coverage extension

`run_point_pairings.py` is the bounded follow-up. It preserves the initial
companion outputs, but supersedes their partial-footprint split fits for
comparisons requiring identical domains. It does not reinterpret the earlier
split difference as a noise-only null or final flux uncertainty.

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-pairings-mpl \
XDG_CACHE_HOME=/private/tmp/point-pairings-cache \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
/Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_pairings.py \
  --maps /absolute/path/to/map-pass/products \
  --measurement /absolute/path/to/diagnostics/source/source-measurements.json \
  --processing-request /absolute/path/to/inputs/processing-request.json \
  --profile saved-descriptive-v1 \
  --output /absolute/path/to/new-pairing-directory
```

The request is tied to the exact realized processing generation. Its adjacent
`effective-point.yaml`, telescope file, native time hashes, per-network science
intervals, PTC slots and original MAP operator are checked. Native sample ranges
are zero-based and half-open; reported first/last UTC values are sample centers,
not exposure boundaries. Absolute epoch uncertainty stays unquantified as in
the parent. Raw telescope ScanNum and processing chunk IDs are distinct.

For 152389 the bound configuration forces six approximately 10-second processing
chunks (0..5) within raw observation 152389_0_2. Every complementary balanced
partition is enumerated once (ten three-versus-three pairings), along with all
six individual chunks. Nothing selects four of six chunks. No four-scan identity
is found in the bound telescope/processing records; a separate owner definition
would require a separately recorded grouping rather than silently relabeling
these chunks. The generic partition function also tests all three balanced
partitions of a four-element domain.

Each subset is accumulated as sum(N)/sum(Q), preserving original admitted
samples and unit coefficients. Its 0.5 coverage cut is evaluated from its own
normalization. Exact original counts and roundoff-bounded numerator recovery
must pass. No upstream operation is rerun. Complete 40-arcsec disk support is
required before calling the unchanged saved seven-parameter fitter. Missing
pixels make that fit unavailable. The fixed template and each aperture retain
full-map centroid, shape and radii, each with their own complete-support gate.
No fit on a shrunken common intersection is substituted.

Per-region coverage includes core (one measured major FWHM), full template,
40-arcsec fit, fit periphery outside the template, and each aperture. Quantiles
are min/10th/median/90th/max and include zero coverage for each subset. Pair
normalization ratios additionally state how many pixels have positive coverage
on both sides; support completeness is a separate fact. The background constant
is fitted jointly on the whole fit disk. Periphery statistics describe that
local context; they are not a new annulus estimator. No arbitrary extra
coverage-ratio acceptance threshold is added. Complete support alone does not
establish equal precision, effective response or broader scientific adequacy.

`groups.json` retains every source measurement, aperture, local support and
normalization distribution, per-network and per-detector accepted contribution
count, and missing parent segment. `replay-audit.json` binds detector occurrence
identities once per network, all native time ranges, whole-segment detector
cause-code histograms and MAP exclusions. Cause codes are preserved as codes;
no new upstream reason is guessed. Excluded PTC samples have no serialized MAP
position, so their whole-segment exclusion counts are not assigned fabricated
source-region positions. Local populations are exact admitted contributions.

Each subset FITS stores SIGNAL/NUMERATOR/NORMALIZATION/SCI_SUPPORT; with the
verified uniform coefficient family, NORMALIZATION is also the hit count.
Difference FITS store A_MINUS_B/COMMON_SUPPORT/Q_A/Q_B/Q_A_OVER_Q_B. Measurement
masks, source table, timing, comparisons, overview figures, runtime and complete
hash manifest are included. Reusing multiple balanced partitions does not turn
them into additional independent observations. Inherited cleaning attenuation
is unresolved; the command neither estimates it nor forces source agreement.
