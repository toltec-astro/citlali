# Saved a1100/a1400 FRUIT feedback inspection

2026-10-30. Noise weighting remains fixed. No reduction, new source fit, detector
selection change, calibration, Unity operation or runtime/configuration change.
This inspection reads the completed paired 152418 noise arm and evaluates existing
fit parameters mathematically; the 1″ example is not a new map or trial.

**The feedback correctly carries the selected central source pixels, but excludes
positive shoulder emission. The case for omitted source wings is stronger in
a1400; a1100's excluded structure is more asymmetric. The 2″ representation is
coarse enough to leave local shape residuals, but neither finding establishes the
cause or size of the previously measured amplitude difference.**

## What the saved feedback actually contains

The active selector subtracts the saved fitted constant, takes pixels above three
times the frozen initial-map BASE scale, then retains the connected component
containing the fitted source center. Other valid pixels are exactly zero. It
feeds back measured pixel brightness, not the fitted Gaussian, with gain one and
replacement each pass. The C0/BASE source-selection models stay frozen from M0;
native detector-noise weights remain separately owned and relearned. Threshold 3,
gain 1 and the 40″ domain are currently fixed in this source-selection code.

The last source model is constructed from M9 and consumed by M10; no M10 model
is used in this run. Both arrays' selection stops changing after M2:

| Array | M0 → M1 → M2 → M9 selected pixels | Final selected area | Existing fitted Gaussian brightness outside mask |
|---|---:|---:|---:|
| a1100 | 32 → 34 → 36 → 36 | 144 arcsec² | 8.47% |
| a1400 | 44 → 49 → 50 → 50 | 200 arcsec² | 5.57% |

For every one of the 20 saved models, the inspection checks its hash against
RUN, feedback record and next request, verifies the next native cycle/plan and
its request/model identity, and verifies FITS checksums. Model/map coordinate
associations are exact. The full 40″ domain is valid, neither selected component
touches the boundary or unavailable support, and every admitted value equals
its source-map pixel minus the saved fit constant exactly. Every valid unselected
value is zero. No missing-file, stale-model, coordinate-offset or extra amplitude
normalization discrepancy was found in these checked associations.

The Gaussian fractions above evaluate the already-saved shape at map pixel
centers, without refitting. They are not fractions of known true source flux;
the Gaussian itself may miss a broad optical component. They also are not large
enough to equate directly to the earlier 33.2%/16.6% peak differences.

## What is left out

Circular means below are centered on each M9 fitted centroid, using native pixels
and that fit's constant subtraction. Units are mJy per nominal beam. These are
spatial descriptive means, not independent noise samples or confidence intervals.

| Array | Radius interval | Map mean | Feedback mean | Mean omitted from feedback |
|---|---|---:|---:|---:|
| a1100 | 6–8″ | 20.60 | 15.34 | 5.26 |
| a1100 | 8–10″ | 10.08 | 4.01 | 6.08 |
| a1100 | 10–12″ | 2.39 | 0 | 2.39 |
| a1100 | 12–14″ | −2.70 | 0 | −2.70 |
| a1400 | 6–8″ | 42.20 | 38.75 | 3.45 |
| a1400 | 8–10″ | 14.08 | 6.97 | 7.11 |
| a1400 | 10–12″ | 5.16 | 0 | 5.16 |
| a1400 | 12–14″ | 2.28 | 0 | 2.28 |

In a1400, unselected 8–16″ pixels average positive brightness in **all four
quadrants**: 4.34, 3.32, 2.96 and 2.17. The circular profile exceeds the narrow
Gaussian in the outer shoulder. This is consistent with omitted source wings,
including a non-Gaussian optical component, but correlated background can also
produce positive means. It is not a detection significance or an optical diagnosis.

In a1100, the corresponding quadrant means are +0.07, −2.19, +5.84 and −0.39.
Its positive shoulder is much less symmetric, and the radial mean becomes
negative at 12–16″. Enlarging the mask indiscriminately risks feeding back
background structure. Asymmetric optical structure remains possible; these
saved maps cannot distinguish it from residual atmosphere/cleaning structure.
Quadrants are azimuth/elevation offset angles in increasing atan2 order from
−180° to +180°, not celestial north/east labels.

The displayed mask contours join pixel centers and illustrate the selection;
the actual mask and feedback are pixel-valued. The exact native masks, not the
contour interpolation, are used for every numerical measurement. No annular
smoothing, source refit or uncertainty calibration is performed.

## What 2″ sampling does and does not demonstrate

The weighted route deposits a sample into its containing map pixel. Native FRUIT
then reads the value of the containing source-model pixel at the detector position.
It does not interpolate a continuous beam within that cell. These are consistent
choices for the current discrete map representation, not evidence of a forward/
backward coordinate bug. A real narrow source can nevertheless vary substantially
inside a cell; that unresolved variation can remain in the timestream after source
subtraction and interact with cleaning. The saved quiet-mode products do not
measure that detector-time interaction.

To isolate representation geometry, the script evaluates the saved M9 Gaussian
at finely sampled positions, then compares it with the value of that same Gaussian
held constant throughout each containing pixel. The fit center, widths and angle
are unchanged. Uniform area quadrature covers an ample grid centered on the
existing offset origin. Full Gaussian support is used: this calculation deliberately
separates pixel sampling from threshold-mask truncation. It does not regenerate a
source product or pass any synthetic data through PTC/MAP.

Define relative shape error as sqrt(integral[(held − continuous)²] /
integral[continuous²]). It is an L2 shape mismatch, **not fractional flux loss**.

| Saved shape | 2″ containing-pixel shape error | 1″ illustrative shape error |
|---|---:|---:|
| a1100 | 20.84% | 10.48% |
| a1400 | 18.24% | 9.16% |

The signed integrated brightness difference is numerically negligible (less than
1e−12 relative) for these smooth Gaussians on the ample grid: positive and negative
local errors cancel. Thus it would be wrong to say that 2″ sampling loses 18–21%
of source flux. The actual binned map also contains noise, non-Gaussian structure,
weighted within-pixel averages and a hard selection edge; it is not identical to
this ideal center-sampled Gaussian. Detector sampling is not uniformly distributed
in area, and no saved detector-coordinate distribution or cleaner response is
propagated here. This is a representation warning and a motivation for a controlled
test, not a numerical explanation of the observed amplitude deficit.

The quadrature was repeated with 16×16 and 32×32 subpositions per pixel; relative
L2 changes are below 0.00030 (0.030 percentage points). The script enforces a 0.001
absolute quadrature agreement gate, solely as numerical verification—not a
scientific harm threshold. No production pixel size or interpolation method is
selected by this arithmetic.

## Recommendation

Retain noise weighting. Focus the next bounded investigation on feedback rather
than adding identical FRUIT iterations or changing detector admission. The current
positive-shoulder evidence particularly motivates a1400; a1100 needs a way to
separate asymmetric source structure from background.

If Grant authorizes a new controlled trial, isolate **one change at a time**:
first a defined source-support/wing treatment at the same 2″ map grid, keeping
noise estimation, ranks, health and all other settings fixed; separately assess
source representation/sampling. Define the candidate support rule before execution
and inspect admitted background as well as peak/aperture recovery. A broad raw
mask or lower global threshold is not automatically a better source model, and a
single narrow Gaussian would not represent every optical wing. A 1″ full reduction
also changes map sampling and noise selection, so it is not by itself a pure test
of the feedback lookup. Native projection consistency must be retained if a finer
or continuous feedback representation is proposed.

No specific new threshold, wing model, pixel size or trial is selected/prepared
here. The existing noise-weighted result remains calibration-ineligible. Absolute
source recovery and the influence of optical structure remain unresolved.

## Reproduction and evidence

Base: `ea61862a5f4bdc9c3dad0443a7ca295be1ff7e93`. Native paired run source remains
`d042083be71f76a406fb698707f7bde94c928d62`; this audit introduces no executable.
Data root: `/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-weight-control-152418-20261030/reduced/redu00`.
Script and [metrics](metrics.json) retain exact RUN/model/map/feedback and figure
bindings. Existing paired audit supplies full detector/sample support verification.

[Mask overlays and profiles](/Users/gwilson/.codex/visualizations/2026/10/23/01a1688f-ccdb-73b0-99e9-0ff65be99341/feedback-inspection-20261030/masks-profiles.png)
show initial/last source selections and the M9-to-M10 model.
[Sampling illustration and mask growth](/Users/gwilson/.codex/visualizations/2026/10/23/01a1688f-ccdb-73b0-99e9-0ff65be99341/feedback-inspection-20261030/sampling.png)
show existing fitted shapes evaluated mathematically.

From this checkout, saved-only reproduction:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/feedback-inspect-mpl XDG_CACHE_HOME=/tmp/feedback-inspect-cache \
 /Users/gwilson/tolteca/bin/python -B -W error \
 validation/point_feedback_inspection_2026-10-30/inspect_saved.py \
 --output /tmp/feedback-inspection.json --figures /tmp/feedback-inspection-figures
```

Focused saved-evidence assertions, Ruff check/format, quadrature refinement and
visual inspection pass. Broader build/CTest/config gates and Unity/Spack validation
are not triggered by this evidence-only work. Exact-SHA independent review and
closure are recorded separately. Runtime, detector policy and all saved reductions
remain unchanged.
