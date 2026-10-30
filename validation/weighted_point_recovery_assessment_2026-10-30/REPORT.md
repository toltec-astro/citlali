# Five-case 152418 source recovery and seed memory

**Compact-source recovery is close to unity on all three arrays at M10. Broader
source recovery is incomplete, most clearly on a2000. The unsupported feedback
seed decays strongly rather than remaining imprinted. M5 is sufficient to describe
the settled a1100/a1400 results here, but it misses a measurable part of a2000's
remaining broad-source recovery.** These are descriptive findings, not new release
acceptance thresholds.

The owner-run campaign completed all five cases and54 map-cycle calls in2448.59s
(40.81 minutes), including its one prerequisite CAL restoration. Assessment used
saved FITS, receipts and fit records only. No reductions, restoration, new source
fits, configuration/runtime changes, Unity operations or pushes were performed.

## Validity of the comparison

The frozen trial manifest is
`229ec59da325db0a7949e08a4fec75d150a3f834ddd2f30756f40fbc403abead`, with reviewed
implementation `d0d3eeb7981975a2f43f1a55878668410aa6c20b`. We authenticated18,187
unique files, including all761 prelaunch bindings and16,565 restored native-time/CAL
payloads against their original hashes. Original metadata and archives match.
The compact input audit records an aggregate identity rather than duplicating the
large payload inventory already frozen in the trial manifest.

REF exactly reproduces all198 saved scientific FITS planes:11 cycles ×3 arrays
×6 planes. All54 actual native cycle records report success; no unexpected error
lines or failed cleaner segments were recorded. Original detector/sample eligibility,
source-free masks and ordered segment domains match. Cleaned and mapped sample
support also match across cases. Injected-parent hashes stay identical through each
trajectory; all726 SEED segment comparisons equal the matched REF parent.
The seed addition is recorded only at M1. Noise-weighting coefficients are allowed
to adapt; there is no equal-weight treatment in this campaign.

There are nine array/cycle comparisons with small full-map coverage-mask changes
(1–4 changed pixels, in M0/M1). **None loses a pixel in the fixed40″ benchmark**.
The66 recorded native weighted-reference checks have maximum absolute error
3.77e-13 mJy/nominal beam. Their admitted weights/normalizations were used by the
native operator; the old unit-weight assumption was not imported. Our independent
saved-map calculation reproduces every recorded gain/distortion to1e-12 tolerance.
Source geometry,2″ sampling, health decisions and all calibration inputs remain fixed.

## What “recovery” measures

Here gain is the projection of the case-minus-REF map onto the known injected
raster on all1257 pixels of the fixed40″ disk, divided by the raster's squared norm.
Unity is ideal. CORE− uses its signed negative truth. Gain is **not an integrated
flux fraction**, a fitted-peak ratio, or an absolute calibration measurement.
Distortion is the norm of recovered-minus-known signal divided by known-signal
norm; it includes shape errors and signal at pixels where the injection is zero.

| Array | CORE+ gain M10 | CORE− gain M10 | Shoulder gain M5 | Shoulder gain M10 | Shoulder distortion M10 |
|---|---:|---:|---:|---:|---:|
| a1100 |101.41%|99.71%|93.01%|93.02%|12.42%|
| a1400 |100.77%|99.82%|95.14%|95.19%|9.85%|
| a2000 |99.86%|99.46%|79.96%|84.37%|21.72%|

Compact shape distortion at M10 is3.62%/2.11% for positive/negative a1100,
2.82%/1.71% for a1400 and1.63%/1.47% for a2000. The compact fitted-peak increments
agree with their ideal map-addition controls to within0.21% of the substantial
ideal peak increment on every array/sign. This supports the current compact POINT
response, while the nonzero distortion prevents claiming exact linear recovery.
The even-response norms are2.46%,1.79%,0.93% of the compact injection norm;
positive-minus-negative gains differ by1.70,0.96,0.40 percentage points respectively.

The broad signal is less well recovered than the compact signal. Its fitted peak
can look satisfactory while its wings remain deficient. At M10 the SHOULDER
peak increments are6.857/11.290/17.884 mJy per nominal beam, versus ideal increments
6.538/11.178/17.713. Those small positive peak differences do not negate the broad
negative recovery error visible around the source in the error maps.

The following uses the frozen aperture of radius **2×the REF M0 major FWHM**, with
pixel-center inclusion and4 arcsec² per pixel. Entries are increments relative to
REF, in mJy/nominal beam × arcsec²; no aperture or beam-area correction is applied.
“Constant-subtracted” removes each total map's own saved fitted constant. Both
forms are retained because changing that constant can partly mask missing brightness.

| Array | Actual raw area increment | Ideal raw increment | Actual constant-subtracted increment | Ideal constant-subtracted increment | Constant-subtracted deficit |
|---|---:|---:|---:|---:|---:|
| a1100 |3106.57|3485.33|3061.96|3308.05|246.09 (7.44%)|
| a1400 |5306.73|5797.00|5081.83|5374.20|292.37 (5.44%)|
| a2000 |10267.92|12940.60|9897.01|11600.47|1703.46 (14.68%)|

The raw deficits are10.87%,8.46%,20.65%, respectively. These well-defined,
non-negligible denominators make the descriptive percentages useful; they are not
pass thresholds. All1,1.5,2-FWHM aperture terms and absolute ideal-control errors
remain in `metrics.json` and the original per-case `IDEAL_COMPARISON.json` files.

## Seed memory

SEED modifies only the first feedback operand, never the original data. The table
reports the **norm of SEED−REF divided by the known shoulder norm**, so it measures
all remaining structure, not only its projection onto the original seed shape.

| Array | Remaining seed norm M1 | M5 | M10 | Largest seed residual pixel M5 → M10 (mJy/nominal beam) |
|---|---:|---:|---:|---:|
| a1100 |32.35%|0.0152%|0.00000145%|0.00478 → 0.000000607|
| a1400 |35.27%|0.0409%|0.0000186%|0.02789 → 0.00001544|
| a2000 |67.35%|1.4985%|0.01552%|1.06921 → 0.01014|

Seed-shaped projections at M10 are0.00000101%,0.0000107%,0.01320%, respectively.
The replacement-candidate differences also decay, and their final m9 admitted
masks match REF exactly on every array. This is strong evidence against persistent
memory of this particular one-time seed under the ordinary replacement policy.
It is not a proof about infinite iteration, other seeds or a Gaussian wing that
would be imposed again every cycle. A permanently forced extension is a different
intervention and is not cleared by this test.

## M5 versus M10

| Array | REF peak change | REF centroid shift | Shoulder response gain change | Largest peak change among five cases |
|---|---:|---:|---:|---:|
| a1100 |+0.000089%|0.0000167″|+0.0091 percentage points|0.000187%|
| a1400 |+0.001406%|0.0000260″|+0.0573 percentage points|0.002258%|
| a2000 |+0.201880%|0.0021921″|+4.4125 percentage points|0.225528%|

For a1100/a1400 there is no appreciable late change in the reported source metrics
at these scales. Their applied feedback masks at M5 versus M10 are unchanged for
all five cases. The M5 cap is well supported for this observation's compact POINT
use and saves work without changing this interpretation.

For a2000 the distinction between **pointing/peak stability** and **broader response**
matters. The REF widths still grow by0.0272″/0.0354″ (major/minor), and its constant
changes by0.1453 mJy/nominal beam, despite the small peak and centroid changes.
CORE+ gain moves98.24%→99.86%; CORE−98.95%→99.46%. The shoulder gains4.41 percentage
points, while distortion drops25.52%→21.72%. Its applied feedback mask changes at
five pixels between the candidates that generate M5 and M10. Recovery improves
through M6/M7 (82.35%/83.60%) and more slowly thereafter (M9 84.29%,M10 84.37%).
The remaining M10 shoulder deficit is therefore not explained simply by stopping
at M5, and more iterations are not justified as an automatic remedy.

The shoulder-case total-map change at a2000 has0.887 mJy/nominal-beam RMS across
the fixed disk, versus0.000827/0.00522 on a1100/a1400. Its largest pixel change is
5.93 mJy/nominal beam. These are descriptive changes, not calibrated significances.
The background-ring STD stays nearly fixed: REF M5→M10 is9.92838→9.92837,
7.68796→7.68797,7.05981→7.05996 mJy/nominal beam. The late differences principally
concern source response, rather than a large change in that descriptive noise scale.

## Interpretation and recommendation

Retain noise weighting and the2″ grid. This experiment supports compact-source
recovery and ordinary feedback's ability to shed an unsupported initial feature.
It does not support forcing a new Gaussian wing merely to make maps brighter.
The remaining broad-response issue is strongest in a2000; a1100/a1400 also retain
measurable shape/area deficits even after their trajectories settle.

Keep the owner's M5 budget as a **prospective economy for POINT trials**, but label
a2000 broad-response conclusions at M5 as less settled. M5's small pointing/peak
drift may be adequate for the immediate pointing objective; this assessment does
not set the release tolerance. Do not state that all three arrays' full source
recovery is finished by M5, and do not silently change the recorded budget.

The bounded next diagnostic question is whether the broad negative recovery error
coincides with source signal omitted by the ordinary feedback selection, especially
on a2000. Saved candidates already show only75.46% shoulder projection in m9 versus
84.37% in the total map, with19 added pixels relative to REF. a1100/a1400 candidate
projections are79.60%/93.00%. These observations motivate inspecting the existing
selection boundary and residuals; they do **not** isolate selection causality from
freshly learned cleaning and weights. Use saved evidence first. Any remedy should
be one explicit bounded policy comparison retaining data and seed controls, not
an automatic longer run or open-ended tuning exercise.

The earlier a1100 equal-versus-noise-weighted amplitude discrepancy cannot be
identified with a20–30% loss of a compact added signal: this controlled experiment
finds almost full compact incremental recovery. It does not recover the unknown
true flux/profile of the original source, validate calibration, or measure telescope
M1 deformation. The perturbations enter after CAL and therefore bypass the actual
optics. Source-free overlap is retained as planned:71,399 a1100 eligible samples
see nonzero tapered input, but their squared-signal sum is1.22e-29 for CORE+ and
0.002892 for SHOULDER; a1400/a2000 have zero overlap. No mask was changed to obtain
these findings, and no isolated fixed-weight/feedback-only attenuation is claimed
for the data-injected cases.

[Trajectory and seed-decay figure](trajectories.png) ·
[Known shoulder, recovery error and late changes](shoulder_maps.png) ·
[Reproducible metrics](metrics.json) · [Assessment source](assess.py)

Assessment verification: all saved-map metric comparisons and identity assertions
passed. The first arithmetic implementation emitted NumPy matmul runtime warnings;
it was replaced with explicit finite-array scalar sums and rerun with runtime
warnings treated as errors, successfully and with unchanged conclusions. Figures
were inspected headlessly. This record requires independent exact-source review
before its final closure. No scientific release qualification is claimed.
