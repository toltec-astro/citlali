# Where the broad-source response is lost

The saved 152418 experiment supports incomplete wing protection as a leading
explanation to test. The feedback ends while measurable injected-source response
continues beyond it, and most final recovery error lies outside that feedback.
However, a2000 also loses response inside the mask. The saved evidence does not
establish that enlarging the mask alone will restore the source.

This is a read-only audit of completed products. Noise weighting and the 2″ grid
were verified and left unchanged. No reductions, source fits, selector relearning,
restoration, Unity operation, calibration issuance or runtime/config edits occurred.

## What was compared

Use the known tapered SHOULDER raster I and matched maps D = SHOULDER − REF.
Recovery error is E = D − I; it is not a Gaussian-fit residual or the residual
timestream presented to the cleaner. Gain is sum(D I)/sum(I²), not integrated flux.
All numerical sums use the fixed 40″ disk about nominal origin (1,257 pixels).
Brightness-area sums use equal 4 arcsec² pixels and retain nominal-beam units;
no measured beam-area or aperture correction is inferred.

Compare the applied feedback difference F = m_SHOULDER − m_REF. M5 consumes m4;
M10 consumes m9. There is no m10 feedback candidate. Profiles use 2″ annuli about
the frozen REF M0 fitted center; they average the offset shoulder, not an assumed
circular source. Figures are diagnostic views, not new fitting or pixel resampling.

Frozen trial SHA256:229ec59da325db0a7949e08a4fec75d150a3f834ddd2f30756f40fbc403abead.
RUN SHA256:c143873de7181443fc96181d5449df55307100cde1a8f9c821f1da03bd38b4ca.
Runtime source:d0d3eeb7981975a2f43f1a55878668410aa6c20b.
The [previous assessment](../weighted_point_recovery_assessment_2026-10-30/REPORT.md)
and its independent review establish CAL restoration, sample/detector support,
all five cases and seed behavior. This narrower audit authenticates 348 consumed
files, including 66 maps, 60 candidates, their bindings and next-cycle plans.
It does not rehash the original CAL payload archive. See INPUTS.json and assess.py.
Current admission/publishing source matches the runtime commit for the files read.

## The masks omit substantial wings

At m9, all candidates are retained. All 1,257 domain pixels are valid; no mask
contacts the domain boundary or unsupported data. Selected candidate values equal
the preceding saved total map minus its saved fitted constant exactly. Excluded
valid pixels are zero; invalid pixels are NaN. All 60 next-cycle source-model
bindings match, with inverse-noise weighting and ±2″ grid steps in every plan.
There is no evidence here of a failed fit guard, wrong feedback file, missing
coverage, or clipping at the 40″ domain boundary.

| Array | REF → SHOULDER mask pixels | Known shoulder brightness outside SHOULDER mask | Shoulder power outside | M10 squared recovery error outside |
|---|---:|---:|---:|---:|
| a1100 | 36 → 41 | 50.05% | 30.24% | 91.58% |
| a1400 | 50 → 67 | 39.71% | 19.20% | 91.40% |
| a2000 | 85 → 104 | 42.16% | 22.48% | 83.27% |

“Brightness outside” is the fraction of the **known added raster** falling outside
the mask, not the fraction lost from the final map. The cleaner/map path recovers
some source outside feedback. “Power” and error fractions are sums of squares.
These fractions should not be interchanged with flux or gain.

The quiet products contain CANDIDATE, ADMITTED and VALID but no per-pixel selection
ratio or frozen selection-scale raster. Thus we cannot locate each threshold
failure or quantify its distance below threshold. Nevertheless, the saved summaries
count only 1, 7 and 5 above-threshold pixels disconnected from the anchor. Even
assigning those pixels the largest omitted truth values leaves at least 48.48%,
33.80% and 39.05% of the entire shoulder brightness on below-threshold pixels,
respectively. This is a conservative positional bound, not a threshold replay.
Most omitted shoulder brightness therefore occupies pixels below the current
selection threshold; disconnected islands cannot account for most of the omission.
The threshold is applied to the total source-relative map using each trajectory's
frozen M0 scale, not to the known injection alone or to current detector weights.

## The loss is predominantly in the wings, but a2000 is not just an edge problem

| Array | M10 gain over whole disk | Local gain inside m9 SHOULDER mask | Local gain outside mask |
|---|---:|---:|---:|
| a1100 | 93.02% | 96.67% | 84.59% |
| a1400 | 95.19% | 97.82% | 84.16% |
| a2000 | 84.37% | 90.57% | 63.01% |

Local gains use the same projection definition restricted to each stated region;
they combine to the whole-disk gain weighted by truth power. a2000 retains a
9.43% projected deficit even inside the selected region and a 36.99% deficit
outside. This rules out describing the observed error as simply missing output
pixels: there are no missing benchmark pixels, and retained regions are affected.

The [maps](feedback_residuals.png) show a broad negative error, strongest toward
the displaced shoulder side, with small positive features at some mask edges.
The [profiles](profiles.png) show feedback terminating while the recovered signal
continues, with a shallow negative recovered-difference bowl farther out. a1100
and a1400 show the same qualitative wing deficit at smaller magnitude.
This is consistent with source remaining in the residual stream and being partly
removed by cleaning, but the actual detector-time residual arrays and cleaner-mode
projections are not available in these quiet products. The measured response also
includes adaptive cleaner/weight changes and independent M0 selector initialization
for each data case. The audit cannot uniquely apportion the loss between them.

At m9 the saved fitted-constant differences are 0.07435, 0.29438 and 0.34394
mJy/nominal beam. They are part of candidate construction. The spatial error maps
and local gains above are computed from raw map differences, without subtracting
these constants; the loss is not an artifact of subtracting a new fitted background
in this audit. The previous assessment separately reports aperture results both
with and without the saved fitted-constant subtraction.

## Mask expansion makes feedback differences nontrivial

On common pixels F is the change in candidate brightness. On newly admitted pixels
F is the entire SHOULDER candidate brightness, including the original source and
background realization at that location; it is not just the injected shoulder.
No REF pixels are removed at m9 in these pairs.

| Array | Feedback projection F onto I | Common-pixel contribution | Newly admitted contribution |
|---|---:|---:|---:|
| a1100 | 79.60% | 60.30% | 19.30% |
| a1400 | 93.00% | 60.14% | 32.86% |
| a2000 | 75.46% | 60.74% | 14.71% |

These are exact geometric accounting terms, not isolated causal contributions to
final recovery. They explain why the feedback-difference image can have bright
edge pixels and why its projection can exceed the fraction of truth power inside
the mask. Simply equating feedback gain with source flux protected would be wrong.

## What M5–M10 adds

SHOULDER mask sizes m0…m9:
- a1100:35,39,41,41,41,41,41,41,41,41.
- a1400:51,60,66,67,67,67,67,67,67,67.
- a2000:51,69,86,93,99,102,104,104,104,104.

For a2000, m4→m9 adds five pixels and feedback projection rises 67.25%→75.46%.
Of that 8.21 percentage-point change, 6.15 points are changes on common mask pixels
and 2.07 points are newly admitted pixels. Thus the later improvement is mainly
continued brightness evolution within existing feedback, not just expanding its edge.
The mask is unchanged from m6 onward, yet the maps continue evolving.
The matched output gain rises 79.96%→84.37%; a substantial deficit remains at M10.
For a1100/a1400 masks and profiles are essentially settled by M5.

The already reviewed one-time seed difference decays strongly; at M10 its full norm
is 0.01552% of the shoulder norm for a2000 (smaller for the other arrays). That
supports forgetting a one-time seed. It does not validate continually supplying
invented wings. Preserve the owner's prospective M5 cap; these results do not
authorize automatic extra iterations or a change in general SCIENCE defaults.

## Recommendation and limits

Keep noise weighting and the 2″ grid. Treat source-feedback selection as the next
bounded question, with a2000 the most informative case. The evidence supports a
controlled, spatially limited wing-protection comparison, not a global lowering of
the threshold or an assumption that more iterations will solve the problem.
Define the candidate policy and matching feedback-only control before implementing
or running it. Judge both inside/outside recovery, background structure and seed
memory; do not select success by peak alone. Retain M5 as the future run budget and
report any unresolved endpoint drift. No harm or release threshold is chosen here.

For any approved future test, saving the small 41×41 selection-ratio raster and
frozen scale once per array would resolve pixel-level threshold questions at modest
cost, without broad verbose tracking. This is a recommendation, not an implemented
change or a request to repeat the completed run merely for diagnostics.

This observation and post-CAL injected shoulder do not diagnose real telescope M1
loss, absolute source calibration, detector failures or faint-SCIENCE performance.
The known injection bypasses telescope optics. a1100's small source-free overlap,
already quantified in the previous assessment, remains part of the measured
procedure. No new threshold, calibration correction or production claim follows.

## Verification

Warning-as-error saved-data analysis passed, with exact FITS checksums, hash chains,
candidate-value semantics, mask validity, source-model associations and geometric
sum identities. Ruff and whitespace checks passed; both figures rendered headlessly
and inspected. Supplemental local Python evidence; build/config/CTest and new
Unity/Spack validation are not applicable. Independent exact-SHA review is recorded
in REVIEW.md upon closure.
