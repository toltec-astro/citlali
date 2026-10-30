# 152418: completed equal/noise-weight comparison

2026-10-30. Saved-product assessment only. Grant ran the reduction; this audit
read existing products and generated descriptive arithmetic and figures. No new
reduction, real-source refit, calibration, runtime change or Unity access occurred.

**Detector and sample support matched in every array and every cycle. Noise
weighting produces much quieter backgrounds, but also lowers the final fitted
peak by 33.2% in a1100, 16.6% in a1400 and 2.1% in a2000.** Most of the a1100
difference exists before FRUIT; further iterations are not the indicated remedy.
The experiment establishes an effect of the combined cleaning/mapmaking weighting
choice. It does not establish which arm recovers the true source brightness.

## What was held fixed and checked

Both arms completed M0 plus ten FRUIT applications under the same executable,
received configuration, original CAL inputs and frozen optical-health publication.
The paired requests have identical network/input associations, effective-config
reference and health reference. Ranks remain 3/4/6, pixels 2 arcsec, coverage cut
0.15 and optical-response cut 0.10. Independent source-model trajectories are
consequences of weighting under the same FRUIT recipe; they are not frozen equal
models after M0. The prepared work order and handoff define the experiment.

The audit reauthenticated saved cycle publications and map plans, then repeated
all 33 array-cycle support comparisons: 726 segment pairs, each with six matching
hashes for ordered detector/slot domain, input eligibility, source-free selection,
residual support, cleaned support and actual mapped support (4,356 hash pairs).
Observation/grid identities, ranks and cleaner convergence also match. Each of
the 66 native maps passes FITS checksum/data checks and its bundle hash; final
published maps match native M10 and the saved fit's map hash. HITS planes are
identical between arms in all 33 comparisons.

| Array | Contributing detectors in each arm | Contributing samples per map, each arm | Weak / reversed / unresolved health decisions |
|---|---:|---:|---:|
| a1100 | 2,185 | 15,667,500 | 6 / 2 / 358 |
| a1400 | 782 | 5,642,569 | 4 / 4 / 167 |
| a2000 | 667 | 2,307,810 | 0 / 0 / 119 |

These contributor counts hold for M0 through M10. The disturbance rule newly
excludes the same one detector in shared upstream processing: a2000 network 11,
channel 92. There are no new disturbance drops in a1100 or a1400. Prior-unavailable
counts in the health population are 622/411/191; these are a later-stage population
than RTC's drop diagnostic and should not be added to RTC counts as disjoint cuts.

There are 10/0/4 health-retained detector-segment occurrences without a map weight
per cycle in a1100/a1400/a2000, respectively. All recorded cases have
`no-eligible-input` reference noise and `reference-noise-unavailable` remaining
noise. These are segment occurrences, not additional whole detectors dropped.
Both arms omit the same samples; equal weighting does not rescue unavailable noise.

**Equal sample support does not mean equal output coverage masks.** Normalization
and science masks use thresholds on accumulated coefficients, so they change with
weighting even at the fixed coverage cut. M10 normalization-mask differences are
838/679/1,187 pixels and science-mask differences are 2,088/2,498/2,746. The nearest
science-mask differences over all cycles lie 104.74/71.84/78.46 arcsec from the
respective fixed analysis centers. All apertures through 45 arcsec are fully in the
common domain. Background arithmetic uses the same supported pixels across both
arms and every cycle; this includes using a common 45–75 arcsec sky selection for
the aperture comparison (one a1400 outer-annulus pixel is omitted from both).

## Final maps and pointing fits

Peaks and background scatter below are in mJy per nominal beam. Background STD
is measured on common supported pixels 30–75 arcsec from the noise-weighted final
centroid. It includes residual sky structure and noise; it is not a calibrated
noise estimate, independent-sample error bar or sensitivity certificate.

| Array | Equal peak | Noise peak | Noise/equal peak change | Background STD: equal → noise | Centroid separation |
|---|---:|---:|---:|---:|---:|
| a1100 | 581.565 | 388.735 | −33.16% | 29.371 → 9.805 | 0.347″ |
| a1400 | 599.071 | 499.795 | −16.57% | 20.490 → 7.756 | 0.206″ |
| a2000 | 694.620 | 680.054 | −2.10% | 15.472 → 7.230 | 0.113″ |

Background scatter falls by factors 3.00, 2.64 and 2.14. Robust MAD-based scales
agree qualitatively: 26.24→9.50, 17.97→6.68 and 14.03→7.00. Visual inspection shows
less striping and broad positive/negative background structure with noise weights,
especially in a1100, although structured residuals remain.

| Array | Equal minor × major FWHM | Noise minor × major FWHM | Equal centroid (Az, El) | Noise centroid (Az, El) |
|---|---:|---:|---:|---:|
| a1100 | 5.966 × 7.603″ | 5.895 × 7.255″ | (8.911, 0.363)″ | (8.798, 0.690)″ |
| a1400 | 7.105 × 8.837″ | 6.876 × 8.080″ | (8.848, 0.180)″ | (8.847, 0.386)″ |
| a2000 | 9.435 × 10.030″ | 9.405 × 10.048″ | (8.981, 0.073)″ | (8.871, 0.094)″ |

All six final saved fits were accepted by the existing guard, finite, converged
and free of active parameter bounds. The centroid shifts are small relative to
the fitted beams, but no new pointing tolerance or acceptance decision is implied.
Saved `sigma_A` is a conditional frozen-C0 reference scale, not a qualified final
empirical flux uncertainty; it is not used to declare the differences significant.

The a1100 peak loss is not explained by the beam becoming broader: the weighted
fit is slightly narrower. Its fitted Gaussian brightness-area is 37.0% lower.
The 12-arcsec aperture brightness-area is also 37.0% lower with common annulus
subtraction, or 37.3% lower subtracting each saved fit's constant. Thus the compact
brightness change survives an alternative measurement to the Gaussian peak.

| Array | Gaussian brightness-area change | 12″ aperture change, common sky | 12″ aperture change, fit constant |
|---|---:|---:|---:|
| a1100 | −36.97% | −37.00% | −37.34% |
| a1400 | −26.18% | −25.36% | −27.45% |
| a2000 | −2.23% | −1.77% | −2.06% |

These are brightness-area diagnostics in (mJy/nominal beam) × arcsec², not
independently calibrated total fluxes. Apertures use a shared fixed center and
8×8 pixel-footprint quadrature; reported fractional areas can differ from unity
by about 0.1% from quadrature. The wider aperture curves remain background
sensitive: a1100 declines outside 12 arcsec in the noise arm, while a1400's 45″
ratio changes markedly with background convention. Wider sums cannot resolve
absolute source recovery here. a2000's compact peak, shape and apertures are much
closer between arms, with a 20″ common-sky aperture difference of only −0.55%.

## What FRUIT did

| Array | Equal M0 → M10 peak | Noise M0 → M10 peak | FRUIT gain: equal / noise | Noise/equal deficit: M0 → M10 |
|---|---:|---:|---:|---:|
| a1100 | 533.579 → 581.565 | 360.872 → 388.735 | +8.99% / +7.72% | −32.37% → −33.16% |
| a1400 | 516.353 → 599.071 | 442.113 → 499.795 | +16.02% / +13.05% | −14.38% → −16.57% |
| a2000 | 601.294 → 694.620 | 376.246 → 680.054 | +15.52% / +80.75% | −37.43% → −2.10% |

a1100 already has nearly the full relative deficit at M0; both arms settle
quickly at different levels. a1400 behaves similarly with a smaller persistent
difference. a2000 differs substantially: the weighted initial map is much more
attenuated, and its independent FRUIT sequence recovers most of that deficit.

The noise-weighted M9→M10 peak changes are approximately 4.2e−8, 6.4e−6 and
0.0195 mJy/nominal beam; the a2000 last increment is 0.0029% of its final peak.
Corresponding map-change RMS inside 20″ is 5.0e−7, 6.9e−6 and 0.0167. This is
numerical settling in the saved sequence, not a new convergence or scientific
acceptance threshold. Both runs stopped because the fixed ten-pass budget ended.

The equal arm's available coefficients are exactly one. Noise-arm map coefficients
span 2.88e−11–9.24e−4, 1.17e−8–3.11e−3 and 9.16e−7–1.68e−3 beam²/mJy² in
a1100/a1400/a2000 at both endpoints. The noise-weighted rule relearns residual noise
through FRUIT and uses the reference-variance floor. Equal extrema do not establish
unchanged interior weights. With `verbose=false`, full valid-detector weight
distributions were not saved; these compact records cannot support claims about
individual high-weight detectors, effective detector count or weight/source-response
correlations. The existing 10% optical cut was shared and unchanged.

## Interpretation and next decision

We can now attribute the paired differences to switching the coefficients in
cleaning and mapmaking, plus the resulting FRUIT evolution, rather than a changed
detector/sample selection. This is a larger a1100 contrast than the earlier
uncontrolled comparison. Both arms use the same acquisition and optical state:
time-varying telescope M1 conditions between separate observations cannot explain
this between-arm difference. That does not turn the result into evidence of faulty
detectors or rule out optical-response differences among detectors within the data.

Cleaner backgrounds are a useful result, but neither a higher equal-weight peak
nor a lower noise-weighted peak establishes the correct source brightness. Possible
contributions include different source removal by the cleaner, different residual
contamination and different emphasis on detectors with different responses. This
pair changes both cleaning and map weights, so it cannot separate those mechanisms.

Recommended next bounded experiment, only after Grant authorizes preparation:
separate the two weighting choices with hybrid cleaning/mapmaking arms, retaining
this frozen membership and identical settings. Start the comparison at M0 to locate
the direct effect before interpreting feedback. If transfer bias remains ambiguous,
define a known-source response test before calibration qualification. Do not simply
add FRUIT iterations, retune the 10% cut, restore unresolved detectors, match the
old peak by construction or treat this as a reason to return globally to equal
weights. Both arms remain `calibration_eligible=false`; no calibration is issued.

## Evidence, reproducibility and limitations

Root: `/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-weight-control-152418-20261030/reduced/redu00`.
Noise final products: `152418/raw`; equal final products: `comparison/equal/152418/raw`.
All original products and reference reductions are preserved. Reported total paired
run time is 1,130.62 seconds (18m51s), including shared upstream and both arms;
this is not a per-arm speed comparison.

Source snapshot: `d042083be71f76a406fb698707f7bde94c928d62`.
Native executable SHA256: `436c8ff369be4d3ea0a190567d8b0890e6680852e15b21f1b4d24ed659ceadb6`.
Received YAML SHA256: `f2250d7bebc8efbc21f0cb7e15075b485fbe05afd0c53d10ee44cd5a4254ae1c`.
Frozen health SHA256: `55d81af8bded42a88d2ad818bcdbd1a64de1f755974c75c496a1c6372d58b1a9`.
Exact RUN, diagnostic, final-fit and all-cycle map bindings and numerical results
are in [assessment-metrics.json](assessment-metrics.json).

[All-array map comparison](/Users/gwilson/.codex/visualizations/2026/10/23/01a1688f-ccdb-73b0-99e9-0ff65be99341/point-weight-assessment-20261030/maps.png)
uses an identical nonlinear color scale throughout. SHA256:
`bd6ed1590d8f83c27566592f46eb6b32de269667584c3ca3870f05e2764143fe`.
[FRUIT and aperture curves](/Users/gwilson/.codex/visualizations/2026/10/23/01a1688f-ccdb-73b0-99e9-0ff65be99341/point-weight-assessment-20261030/fruit-apertures.png)
SHA256: `091e68256df1e6cec0d276656ec746286d31bb82938ec6838318dbeb7cbe07e4`.

Reproduce only the saved-data audit, from this checkout:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/point-weight-assess-mpl XDG_CACHE_HOME=/tmp/point-weight-assess-cache \
  /Users/gwilson/tolteca/bin/python -B -W error \
  validation/point_weight_control_2026-10-30/assess_saved.py \
  --output /tmp/152418-weight-assessment.json --figures /tmp/152418-weight-figures
```

The script and Ruff checks pass. An initial extra assertion incorrectly expected
normalization masks to match; inspection of `ForwardMapOperator` established their
coefficient-dependent threshold semantics. The audit now records those differences
and requires exact HITS and sample-support agreement. This was an audit assumption
repair, not a reduction change or waived input-support failure. No unexpected
error-level lines were recorded by either saved run. Broad application/build/config
gates are not retriggered for this evidence-only assessment; local observational
evidence is not Unity/Spack qualification. See the assessment closure record for
exact-SHA independent review and final repository state.
