# FRUIT compact-source development check: pointing 152389

Status: **exploratory development evidence only; not a qualification result or
stopping-rule decision**

## The simple question

Does established Citlali FRUIT move this point source away from the
artificially narrow PCA result and toward the reduction's expected point-source
response, without moving its centroid or obviously making the surrounding map
worse?

This is one real pointing observation of source `1146+399`. It is a useful
first check, but it has no independently known injected flux or PSF. It cannot
by itself establish absolute flux recovery, the true diffraction-limited beam,
or performance on extended low-signal emission.

## What is used as the beam reference

Two references are shown, and neither is mislabeled as independent truth:

1. `kernel_I` is the primary same-run comparison. It represents the compact
   response carried through this reduction's APT assumptions, processing, and
   JINC mapmaking. Comparing the measured source width with this kernel tests
   whether the PCA-processed source is narrower than the response the reduction
   itself predicts.
2. FITS `BMAJ` and `BMIN` provide the secondary observation-specific beam
   geometry carried into the reduction from its APT/calibration state. Their
   geometric means are `5.546`, `6.782`, and `9.067` arcsec for `a1100`,
   `a1400`, and `a2000`.

The generic `5.0`, `6.3`, and `9.5` arcsec values found elsewhere in Citlali
configuration and historical analysis are not used as the target here. In the
active pointing configuration, similarly named values are Wiener-template
settings, and repository evidence does not establish them as the independent
physical diffraction truth for this observation. A formal compact-source
profile still needs an owner-approved numerical diffraction/beam reference or
an injected source with known truth.

## What happened

The width entries below are the geometric mean of the two fitted Gaussian
axes. A ratio of one means that the fitted source width matches the named
reference.

| Array | Source/kernel width, iter 0 | iter 1 | iter 2 | Source/APT width, iter 2 | Peak amplitude, iter 2 / iter 0 | Centroid shift, iter 2 | Background sigma, iter 2 / iter 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `a1100` | 0.932 | 1.003 | 1.043 | 1.099 | 1.090 | 0.068 arcsec | 0.880 |
| `a1400` | 0.949 | 1.011 | 1.051 | 1.092 | 1.197 | 0.025 arcsec | 0.784 |
| `a2000` | 0.937 | 0.994 | 1.041 | 1.090 | 1.299 | 0.040 arcsec | 0.801 |

The initial PCA result is 5--7 percent narrower than the processed kernel in
all three arrays. After one feedback pass, every array is within about one
percent of that kernel. That is the behavior expected if FRUIT restores
point-source response suppressed by PCA. After the second feedback pass, the
source is 4--5 percent broader than the processed kernel and about 9--10
percent broader than the APT-header reference. This is a measured trajectory,
not a declaration that the extra broadening is harmful.

The centroid remains stable to less than `0.07` arcsec. The robust background
sigma in the 40--120 arcsec annulus is lower at iteration 2 by approximately
12, 22, and 20 percent for the three arrays. However, the background pattern is
still changing: the iteration-1-to-2 background-map change is 48, 57, and 61
percent of the preceding background RMS. The central 40-arcsec map also changes
by about 10--11 percent in that transition. Therefore these three saved
iterations do not establish a stable terminal point.

Peak amplitude is not monotonic: `a1100` and `a1400` peak at iteration 1 and
then decrease, while `a2000` continues to increase. Without injected or
external flux truth, none of those points can be selected as the scientifically
best iteration.

## Measurement definitions

- Peak amplitude, Gaussian widths, centroid, and formal fit S/N come from the
  persisted pointing ECSV tables.
- The source and processed-kernel width comparison uses geometric-mean FWHM,
  preserving the source and kernel axis ratios separately in the CSV.
- Background level is the median and scaled median absolute deviation in the
  coverage-valid 40--120 arcsec annulus around the fitted source.
- Pixel roughness is the scaled median absolute deviation of adjacent-pixel
  differences in that annulus, divided by `sqrt(2)`.
- Successive-map change is RMS of the pixel difference divided by RMS of the
  preceding map, evaluated separately inside 40 arcsec and in the 40--120
  arcsec annulus on common valid support.
- Centroid motion is measured both step-to-step and from iteration 0.

The exact numerical definitions and all nine iteration/array rows are in
[`iteration_metrics.csv`](iteration_metrics.csv). The plotted summary is
[`point_152389_iteration_summary.png`](point_152389_iteration_summary.png).
[`manifest.json`](manifest.json) records the operational-control identity,
configuration, executable hash, and hashes of every product read by the
analysis. The source reduction and its canonical APT remain unmodified.

## What this establishes, and what comes next

This observation supports the qualitative statement that FRUIT reverses the
PCA narrowing of a compact source: one feedback pass brings the fitted width
into close agreement with the processed response kernel. It also demonstrates
stable astrometry and improving annular robust noise by iteration 2.

It does **not** establish true recovered flux, an absolute diffraction-limited
PSF, a stopping iteration, extended-mode recovery, or superiority of a new
recurrence. The next scientifically decisive compact-source test is a
controlled timestream injection into this same development-only observation.
That test can provide known input flux and shape while retaining the real
atmosphere and detector behavior. Its metrics and any acceptance bands must be
declared before using it to tune or rank candidate recurrences.

## Reproduction

From the Citlali repository root, with the downloaded reduction and preserved
executable snapshot in their recorded locations:

```bash
MPLBACKEND=Agg \
MPLCONFIGDIR=/tmp/fruit-point-dev-mpl \
XDG_CACHE_HOME=/tmp/fruit-point-dev-xdg \
$HOME/tolteca/bin/python \
  tools/fruit_loops/analyze_compact_pointing_development.py \
  --reduced /Users/gwilson/work_toltec/local_data/citlali-validation/v2/point/refactor/reduced \
  --obsnum 152389 \
  --output validation/fruit_loop_point_152389_development_2026-09-02 \
  --control-id SCI-FRUIT-C31-OPERATIONAL-CONTROL-POINT-152389-R0.1 \
  --software-id v4.0.0-3753-gc31a60a0 \
  --executable-snapshot /Users/gwilson/work_toltec/local_data/citlali-validation/v2/point/refactor/.tolproj/citlali-snapshots/9d3ce4f31260bbdf3f630dbaa09e0c18eb92b3ebcc8be15eb146024acd6f7f65/citlali
```
