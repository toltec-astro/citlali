# POINT152420 saved 1.1 mm map comparison

The current saved M0 map is more elongated under a common descriptive fit. The
old-versus-new difference survives matching the Gaussian model, averaging the old
map to2arcsec pixels, and a fixed core-only fitting check. This is evidence of a
morphology difference between the saved reductions, not identification of a
particular upstream defect or a calibrated beam/PSF measurement.

| Map / fixed diagnostic | Major FWHM | Minor FWHM | Radial centroid offset | Offset +3 major FWHM |
| --- | ---: | ---: | ---: | ---: |
| Earlier V2, native1arcsec | 8.156″ | 5.936″ | 8.908″ | 33.375″ |
| Earlier V2,2x2 mean to2arcsec | 8.296″ | 6.085″ | 8.894″ | 33.783″ |
| Current M0, native2arcsec | 10.814″ | 4.842″ | 8.832″ | 41.274″ |
| Earlier V2,15arcsec core fit | 8.173″ | 5.948″ | 8.907″ | 33.427″ |
| Current M0,15arcsec core fit | 10.630″ | 4.754″ | 8.834″ | 40.724″ |

The current major axis is32.6% larger than the old native-map fit, or30.3% larger
than the old map averaged to2arcsec. Its minor axis is18.4% smaller than the old
native fit. The geometric-mean FWHM rises only4.0%: the dominant descriptive
change is elongation. Axis ratios are approximately1.37 before and2.23 now.
These are point estimates without a covariance-derived uncertainty; they are not
statistical significance, astronomical truth, or a transfer-function calibration.

## What the comparison resolves

The earlier reported7.35×6.13arcsec widths came from the historical axis-aligned
fit. Letting orientation vary under the current shared solver changes that old
estimate to8.16×5.94arcsec. Thus the earlier comparison overstated the major-width
difference, but the matching fit leaves a substantial difference in the saved
maps. The current fit reproduces all seven saved guard parameters exactly.

The source offset is stable at about8.9arcsec. The existing analysis helper uses
a circular radius of3times the **major** FWHM around that displaced center;
its farthest continuous extent is41.27arcsec now versus33.38arcsec before.
That is an analysis aperture, not a41arcsec source width. The core-only check
still produces an extent40.72arcsec, so distant background structure is not
sufficient to explain away the current boundary failure. The actual campaign
check remains the original pixelwise guard; these continuous extents explain it
and do not redefine its numerical acceptance.

The plot's half-maximum contours and principal-axis cuts show the current narrow
minor direction and extended major direction in the central source. Residuals
also differ, but normalized residual RMS is not a like-for-like noise estimate.
Plot interpolation is used only for display cuts, never for the native fits.

## What remains unresolved

The maps were not made by an otherwise identical pipeline. The older header
records standard PCA with5modes, validated weighting,1arcsec Alt-Az pixels and
no FRUIT. Current M0 uses rank10 observed-entry ALS, uniform MAP weighting,
2arcsec FK5 tangent pixels and networks1–4, with the successor RTC/CAL/support
contracts. Detector contributions, projection and filtering can differ. This
comparison cannot attribute the morphology to one of those differences, nor
establish that a pointing error rather than weighting/filtering/support produced
it. No reduction, new injection, parameter sweep or feedback cycle was run.

Native frames differ. We fit in each map's own physical arcsecond coordinates,
using a freely rotated ellipse; major/minor lengths and radial offsets are
invariant under rigid changes of axes. No single inferred parallactic rotation
is imposed on a time-integrated Alt-Az map, and no registered sky-plane difference
image or claim of equal orientations is made. Projection during mapmaking remains
a possible source of differences; coordinate invariance of the fitted lengths
does not eliminate that possibility. The2x2 mean is a sampling control with its
correct half-pixel-shifted block centers, not a reproduction of the new mapmaker.

Recommendation: preserve the40arcsec gate for now and investigate source shape
through the saved per-network contributions, detector offsets and mapping support
before choosing whether to change the analysis footprint. This recommendation
is not authorization to execute a new reduction or change the gate.

## Validation and provenance

`results/metrics.json` binds both maps, saved guard, analysis script and shared
solver by SHA256 and records all fit parameters, masks, convergence and environment.
`results/VERIFICATION.json` verifies the old map against its retained product index
and the new map against its observation/array-bound producer bundle. Old FITS
lacks internal checksums; the external SHA256 matches. New FITS internal checksums
pass. All input digests remain unchanged. All fitted parameters are interior to
the shared bounds. Synthetic rotated-Gaussian recovery, orthogonal-coordinate
invariance, invalid-support rejection and block mean/center checks pass.
`results/comparison.png` was rendered headlessly and visually inspected.

This is a bounded Tier2 evidence follow-up from basea84eb2456fef4a8c2e768144dd75a47f21a83654;
application code/configuration is unchanged. CTest/build/config gates are not
rerun because no covered implementation changed; the prior successful runtime
preflight remains a historical fact, not new observational success. Local tolteca
Python/NumPy/SciPy/Astropy versions are in metrics; no Unity work or Spack
reproduction is claimed. Independent fresh-context review binds the final exact
commit in an external review receipt. No integration, push, activation, cleanup,
rerun or launch command follows from this comparison.
