# Contrast with owner-selected redu09

The supplied redu09/a1100 saved map is much brighter than both no-FRUIT maps
under the same unweighted, freely rotated Gaussian-plus-constant40arcsec fit.
It is also broader in both axes than the current successor M0.

| Saved map / check | Peak above constant (mJy/beam) | FWHM major×minor (arcsec) | Radial offset (arcsec) | Offset +3major FWHM (arcsec) |
| --- | ---: | ---: | ---: | ---: |
| Earlier V2 native1arcsec | 251.70 | 8.16×5.94 | 8.91 | 33.38 |
| Earlier V2 averaged to2arcsec | 242.78 | 8.30×6.09 | 8.89 | 33.78 |
| Current successor M0,2arcsec | 223.28 | 10.81×4.84 | 8.83 | 41.27 |
| **redu09 native1arcsec** | **574.59** | **12.57×7.31** | **8.86** | **46.56** |
| redu09 averaged to2arcsec | 559.89 | 12.66×7.52 | 8.86 | 46.84 |
| redu0915arcsec core-only fit | 586.21 | 11.37×6.41 | 8.87 | 42.99 |

The redu09 native peak is2.28times the earlier V2 peak and2.57times current M0.
Its fitted constant is+32.59mJy/beam, compared with−1.61 and−2.04 respectively;
the quoted amplitudes exclude those constants. The source radial offset remains
approximately8.9arcsec. The historical redu09 ECSV gives an axis-aligned amplitude
580.58mJy/beam and10×10arcsec widths; those are a different fit and are not
substituted into the common-fit comparison.

The2arcsec sampling check retains559.89mJy/beam, so pixel averaging does not
account for the large peak difference. The core-only check changes the fitted
constant to60.48mJy/beam and narrows the widths; thus redu09 has more sensitivity
to the fitting region than the previous maps. Even that fixed check leaves a
footprint extending42.99arcsec. Neither alternate fit changes campaign admission.
The normalized residual RMS is8.77% of the fitted peak for native redu09, versus
8.21% for prior V2 and10.29% for current M0; these are descriptive residuals, not
comparable noise estimates or evidence of which map is scientifically best.

## Processing-stage distinction

The supplied FITS records FRUITLOOPS_ITER=4 and CONFIG.FRUITLOOPS=true,
standard_pca with5modes, validated weights, downsampling, array grouping and
JINC parameters R1.5/A1.1/B1.67/C2. The directory index identifies
v4.0.0-423-gffc6b907 and2026-07-19.13:32:39. The earlier V2 map records no FRUIT;
current successor M0 likewise precedes any FRUIT applications. This is not a
comparison of equivalent feedback stages or a controlled single-variable test.
The increased peak is consistent with the motivation for iterative source
preservation, but these saved products cannot isolate FRUIT's contribution from
other reduction, calibration, weighting, detector or mapmaking differences.
Peak amplitude is not integrated flux or proof of recovered astronomical truth.

## Consequence for the failed footprint gate

This addition qualifies the earlier recommendation. The owner-selected historical
redu09 map has a larger three-major-FWHM footprint than the failed current M0.
The unchanged actual linear_weights helper rejects its fitted geometry with
“fixed-template footprint exceeds source stamp” on the81×81 one-arcsecond source
stamp. Thus failure of that auxiliary containment rule alone is not sufficient
evidence of a bad reduction or a newly introduced physical broadening defect.
The earlier V2 comparison still establishes a morphology difference, but it is
not a universal successful-reduction width requirement. We should distinguish
source-shape diagnosis from applicability of the fixed-template estimator before
choosing a repair. No40arcsec feedback/fit rule is relaxed by this report.

## Evidence and limits

Only three new fixed diagnostic fits ran: native redu09,2x2 mean, and15arcsec
core. The previous three comparison values are read from immutable bound metrics;
no earlier fit, reduction or feedback was rerun. The common helper, solver and
all inputs are SHA256-bound in results/metrics.json; original inputs are unchanged.
All fits converged with no active bounds. results/VERIFICATION.json records hash
checks and the actual helper rejection. The plot was inspected headlessly.

Unlike the prior V2 map, redu09's index contains no historical SHA256 manifest
binding and its FITS has no internal CHECKSUM/DATASUM. The present-day SHA256
is captured before analysis and verified afterward; historical byte authenticity
is not independently established. Native old/redu09 Alt-Az and successor FK5
frames are not registered; widths and radial offsets are rotation invariant,
while orientation and pixelwise differences are not compared. No covariance-
calibrated uncertainties, significance, calibrated PSF or upstream-cause claims.

Scope remains the parent Tier2 evidence comparison on the same worktree, base
0132dc8ca2c72e0b7fd35439196f29365646b6ec. Application code, shared helper and
pipeline configuration are unchanged. No build/config/CTest gate is newly
triggered by this isolated evidence addition. A fresh-context independent review
binds the exact final commit in an external receipt. No push, Unity, cleanup,
activation, automatic launch or policy change is performed.
