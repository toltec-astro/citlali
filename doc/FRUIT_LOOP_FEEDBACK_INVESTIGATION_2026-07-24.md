# Pointing Fruit-Loop Feedback Investigation

Date: 2026-07-24

Status: active; the first five controlled Unity ablations are complete. A
high-S/N-only model-support ablation and full-PTC injected-source test remain
required before a production default change.

## Question

Five pointing observations show monotonic fitted amplitude and FWHM growth over
four fruit-loop feedback passes. The purpose of this investigation is to
distinguish valid recovery of signal suppressed by timestream cleaning from an
incorrect feedback recurrence or an interaction with learning, map-template
tapering, detector weights, kernel handling, or model support.

The controlling evidence is the TolAPT report at commit `fd514b5`:

`outputs/hero-pointing-comparison/fruitloops5-rc1-convergence/analysis/report.md`

The seed iteration is an exact control: `redu00` is identical to the existing
no-fruit-loop RC1 products.

## Production Recurrence

For a map projection operator \(P\), timestream cleaner \(C\), input data
\(d\), and previous map \(m_n\), Citlali implements:

1. subtract the projected map: \(d - Pm_n\);
2. clean the residual: \(C(d - Pm_n)\);
3. add the identical projected map back:
   \(C(d - Pm_n) + Pm_n\);
4. map the restored timestream to form \(m_{n+1}\).

For a fixed linear cleaner and exact map/project operators, this is the
standard transfer-function recurrence. It can produce monotonic growth while
converging to the true source. Monotonicity alone therefore does not establish
a fault.

The implementation uses the same loaded map buffer, interpolation policy,
pixel-selection gate, and calibration geometry for subtraction and add-back.
Residual-weight resetting can add detector-sample flags before add-back; the
new diagnostics measure that difference explicitly. A regression test with
unchanged flags confirms that an immediate subtract/add round trip restores
both signal and kernel TOD to numerical precision. A second controlled test
injects a Gaussian of known amplitude and width, applies a known linear
attenuation, and exercises the production map-to-TOD and naive-mapmaking
recurrence. Its amplitude approaches the injected truth, iteration changes
shrink, and its PSF width remains stable.

These tests reject a basic sign error or unconditional double-add. They do not
yet validate the nonlinear production PTC cleaner or iteration-dependent
learning and weight policies.

## Existing-Run Evidence

For observation 133410, successive whole-map changes become smaller after the
large first pass. The final relative whole-map RMS changes are approximately
6.2%, 5.6%, and 4.7% for a1100, a1400, and a2000. This is compatible with a
stable recurrence, but the changing detector policy after iteration two means
it is not a fixed-transfer-function proof.

The configured `beammap_source.fluxes` values do not provide a known pointing
source truth. Citlali uses that photometry contract for Beammap flux
calibration; pointing map calibration remains APT based. The real-data
trajectory therefore cannot be accepted or rejected by comparing its fitted
amplitude with those config values.

### Feedback support is not source-only

The frozen pointing policy combines its selection gates with logical OR:

- `sig2noise_limit: 100`;
- `array_flux_limit: [12, 18, 10] mJy`; and
- inactive peak-fraction and local-S/N gates.

The low absolute flux cut consequently selects many positive background
fluctuations even when they fail the S/N cut. After applying the configured
map-weight taper, the seed maps have:

| Array | Active flux-selected pixels | Fraction beyond 40 arcsec | Fraction of tapered positive model sum beyond 40 arcsec |
|---|---:|---:|---:|
| a1100 | 21,879 | 96.8% | 79.1% |
| a1400 | 23,036 | 95.2% | 77.0% |
| a2000 | 27,054 | 97.6% | 57.4% |

The central positive source region has full template-weight gain, but the
taper removes about 31--32% of the full flux-selected positive model sum at
the seed. The projected model is therefore a bright compact source plus a
large, one-sided positive background/noise field. Repeatedly protecting that
field from cleaning can interact with the PTC cleaner and map normalization.
Detector-weight recomputation has now been ruled out for this observation.
Broad model support remains an untested candidate, not a demonstrated root
cause.

## Provisional Cause Classification

| Candidate | Current disposition |
|---|---|
| Fundamental subtract/clean/add-back recurrence | The low-level recurrence and immediate round trip are mathematically correct. The production PTC cleaner/map-normalization interaction is now the dominant remaining cause class. |
| Learned masks or detector exclusions | Ruled out as the origin. Disabling learning leaves iterations zero and one exact and changes fitted amplitude by at most 0.32% later. |
| Template weight taper | Ruled out as a material cause for this observation. Disabling it changes final fitted amplitudes by less than 0.001%. |
| Source-subtracted detector weights | Ruled out for this observation. The recompute policy executed, but all signal, kernel, and weight image arrays remained exactly identical to the control. |
| Kernel normalization | Captures the recovered PSF-width evolution but does not correct the amplitude overshoot. It is not the sole cause. |
| Interaction | The remaining interaction is the production recurrence with PTC cleaning/map normalization. Broad one-sided model support remains the next isolated candidate within that interaction. |

No production default has changed.

## Controlled Ablations

`tools/fruit_loops/prepare_feedback_ablation.py` verifies the frozen obsnum
133410 policy and generates:

1. unchanged full-policy diagnostic control;
2. learning disabled;
3. map-template weight feedback disabled;
4. detector weights recomputed after add-back; and
5. all three changes together.

All configs retain five iterations and every iteration product. Every config
has an independent output root. The unchanged-policy rerun is an
instrumentation control and provides stage diagnostics unavailable in the
existing run.

After those five variants completed, the generator gained a sixth one-change
test:

6. low absolute-flux selection disabled, retaining only the high-S/N model
   support.

The opt-in `timestream.fruit_loops.diagnostics_enabled` path records, for each
scan and array:

- loaded signal/kernel map count, sum, RMS, extrema, and absolute peak;
- loaded map-weight count, sum, RMS, and extrema;
- projected signal/kernel sample count, sum, and RMS for subtraction and
  add-back;
- signal and kernel TOD summaries before subtraction, after subtraction, after
  cleaning, and after add-back;
- source-subtracted and final detector-weight summaries; and
- final fitted map amplitude, widths, centroid, and amplitude/whole-map-RMS
  diagnostic. The established pointing-table S/N remains the comparison
  authority.

`tools/fruit_loops/compare_feedback_ablation.py` compares all saved iterations
and records fitted source, kernel, map-weight, successive-map, and feedback
support metrics.

## Completed Unity Ablations

All five initial variants completed `redu00` through `redu04` without an
error-level message. Each emitted the complete expected set of 1,455
diagnostic records: 15 final-map fits, 576 TOD-stage summaries, 288 loaded-map
summaries, 288 projection summaries, and 288 detector-weight summaries.

The unchanged diagnostic control is array-for-array identical to the original
five-iteration run for every signal, kernel, and weight image. The
instrumentation therefore does not perturb the result.

| Variant | Maximum fitted-amplitude difference from control | Interpretation |
|---|---:|---|
| Learning disabled | 0.32% | Small later-pass effect; not the origin |
| Template weight taper disabled | 0.00093% | Materially irrelevant to fitted-source growth |
| Weights recomputed after add-back | 0% | Exact image-array identity; not causal here |
| All three changes | 0.31% | Numerically equivalent to learning disabled |

The subtraction and add-back calls load numerically identical signal and
kernel maps. Add-back projects 0.0015--0.0127% fewer selected samples,
depending on iteration and array, because residual-weight resetting flags
those samples between the two calls. The retained projected signal sum is at
least 99.993% of the subtracted sum. Those newly flagged samples are also
excluded from mapmaking, and the difference is far too small to explain the
30--48% first-pass source growth.

The TOD sum identities close to \(1.1\times10^{-10}\) relative for signal and
\(3.6\times10^{-13}\) for kernel. This verifies that the recorded subtraction,
cleaning, and add-back stages are internally arithmetically consistent.

### Shape recovery versus amplitude truth

Observation 133410 is 3C273. The matched APT records the contemporaneous
calibrator-flux reference used by the reduction as 3981.3, 4799.7, and
6331.6 mJy for a1100, a1400, and a2000. This is a calibrated real-source
reference rather than an injected absolute truth.

The fitted source width divided by the propagated kernel width moves from
0.847, 0.912, and 0.935 at the seed to 1.001, 1.029, and 1.011 at iteration
four. Fruit loops are therefore recovering the PSF shape represented by the
kernel.

Amplitude does not show the same truth convergence. After division by kernel
peak, the calibrated-flux errors evolve as follows:

| Array | Seed | Iteration 1 | Iteration 4 |
|---|---:|---:|---:|
| a1100 | -14.6% | +8.5% | +7.0% |
| a1400 | +4.3% | +47.1% | +37.5% |
| a2000 | -24.4% | +2.2% | +16.5% |

This is mixed behavior: the PSF transfer function is recovered, but the
amplitude recurrence overshoots the calibrated source truth, especially for
a1400. Extending the iteration count cannot resolve that normalization
problem.

## Decision Rules

- If subtraction and add-back differ for samples retained by mapmaking, or by
  enough to explain the source change, the recurrence has mutated material
  state. The measured difference is currently confined to a tiny sample set
  newly flagged during residual-weight resetting.
- If only learning-disabled runs change after iteration two, learning modifies
  convergence but does not explain its origin.
- If disabling template taper changes only broad support while central growth
  remains, taper is not the root cause.
- If recomputing weights after add-back removes growth while preserving
  shrinking changes and source shape, source-subtracted detector weights are
  the dominant interaction.
- If all three ablations retain comparable growth, the next one-change test
  must restrict the low absolute-flux model support before altering any
  production policy. This condition is now met; the generated
  `snr_only_model` variant sets only `array_flux_limit: [0, 0, 0]`, retaining
  the existing S/N threshold of 100.
- A production policy is acceptable only after a realistic injected-source
  test with the full PTC cleaner converges to known amplitude and PSF with
  shrinking changes.

The current synthetic injection establishes the recurrence test seam; the
next implementation increment should replace its controlled scalar cleaner
with the configured production PTC path or a fixture that exercises that path
without external Unity data.

## Local Verification

The investigation code passes the local `citlali_cli` and `citlali_test`
builds. CTest passes all 505 enabled tests; one unrelated lifecycle test
remains disabled. The complete config preflight passes 123 Python unit tests,
all four mode kits, compact compatibility, schema generation, and every typed
authority audit.

The ablation generator accepts the downloaded frozen obsnum 133410 low-level
config and produces the initial five independent configs plus the next
high-S/N-only model-support config. The comparison tool reproduces all
array/iteration records and feedback-support measurements from the existing
and completed ablation runs.

The high-S/N-only Unity product and full-PTC injected-source fixture remain
open. Existing evidence establishes correct PSF-shape recovery alongside an
amplitude trajectory that does not converge to the calibrated source
reference. It narrows the fault to the production
recurrence/cleaner/map-normalization interaction without changing defaults.
