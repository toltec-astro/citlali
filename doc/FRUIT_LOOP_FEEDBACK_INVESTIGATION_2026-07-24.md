# Pointing Fruit-Loop Feedback Investigation

Date: 2026-07-24

Status: active; both controlled Unity matrices are complete. The first
full-PTC injected-source pair completed, but its exact-restart control gate
failed because checkpoint schema v1 omitted retained processed-weight
validation state. That pair is quarantined. Checkpoint schema v2 and a
mandatory uninterrupted-versus-restarted control gate are implemented
locally; a corrected Unity pair remains required before accepting or changing
a production policy.

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
| Fundamental subtract/clean/add-back recurrence | The low-level recurrence and immediate round trip are mathematically correct. The ten-iteration production run reaches a stable fitted-source endpoint. |
| Learned masks or detector exclusions | Ruled out as the origin. Disabling learning leaves iterations zero and one exact and changes fitted amplitude by at most 0.32% later. |
| Template weight taper | Ruled out as a material cause for this observation. Disabling it changes final fitted amplitudes by less than 0.001%. |
| Source-subtracted detector weights | Ruled out for this observation. The recompute policy executed, but all signal, kernel, and weight image arrays remained exactly identical to the control. |
| Broad model support | Ruled out as a material cause. Compact adaptive models use only 3--8% as many projected detector-samples on the first pass yet reproduce final fitted amplitudes within 1%. |
| PTC cleaning | Dominant driver. Seed attenuation and feedback correction increase strongly with PCA depth. This is expected transfer-function behavior, not by itself a fault. |
| Projection/mapmaking | Secondary. Bilinear and historical projection variants change final amplitudes by at most 2.7%; naive mapmaking retains a similar growth pattern. |
| Kernel normalization | Tracks the recovered PSF and much of the amplitude transfer. Real-source endpoints do not equal the matched-APT flux, but cleaner-free fits are farther from that reference, so this does not isolate a kernel fault. |
| Interaction | No production fault is demonstrated by the real-source matrices. The full-PTC injected-source test is the remaining authority for absolute amplitude and PSF recovery. |

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

After those five variants completed, the generator gained a follow-up matrix.
It includes:

- high-S/N-only support at thresholds 50, 100, and 200;
- compact source-centered support using either 5% of local peak or local
  S/N 5;
- PTC cleaning disabled, one PCA mode, ten PCA modes, and a 30-arcsec PTC
  source mask;
- bilinear projection, the historical truncating projection/center
  convention, and naive mapmaking; and
- an unchanged ten-iteration trajectory.

These are subsystem-isolation and dose-response tests. No member is a proposed
production policy.

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
a1400. The follow-up cleaner matrix shows that this real-source comparison
cannot by itself establish a normalization fault: the cleaner-free
kernel-corrected a1400 fit is about twice the same reference, consistent with
strong residual-background contamination. Injected truth remains necessary.

## Completed Follow-up Matrix

Twelve variants completed every requested iteration with executable
`v4.0.0-3592-gdba226d0`. The `snr_only_s200` run stopped cleanly after
iteration one because all three arrays selected zero detector-samples. Its
no-op guard behaved as designed. Empty stderr files and no unexpected
error-level messages were recorded for the twelve completed runs.

The five-iteration control and the first five iterations of the independent
ten-iteration control are exact for all 45 signal, kernel, and weight image
arrays. Every support- or projection-only seed is likewise exactly identical
to the control seed.

### Cleaner-strength response

The fitted-amplitude ratio between the final and seed maps is:

| PTC PCA modes | a1100 | a1400 | a2000 |
|---:|---:|---:|---:|
| 0 (cleaning disabled) | 0.878 | 0.773 | 1.096 |
| 1 | 1.002 | 0.734 | 1.172 |
| 5 (control) | 1.382 | 1.555 | 1.805 |
| 10 | 2.023 | 2.646 | 3.065 |

Stronger PCA cleaning progressively attenuates the seed and produces a larger
feedback correction. At ten modes, the seed kernel-corrected amplitudes are
about 50% below the matched-APT reference in all arrays; fruit loops recover
them to -9.8%, +21.0%, and +6.5%. The zero- and one-mode a1400 maps remain
strongly contaminated and exceed the same reference even before feedback.

A 30-arcsec PTC source mask nearly eliminates growth for a2000 and reverses it
for a1100 and a1400, while materially changing the seed map. This confirms
that the cleaner acts on source-bearing samples, but the masked result is a
mechanistic test rather than a candidate production policy.

### Model-support response

The control projects 1.39 million detector-samples on its first feedback pass.
The compact 5%-of-peak and local-S/N-5 models project only 52,164 and 48,567
samples, respectively, yet their final amplitudes differ from the control by
at most 0.97% and 0.41%. Broad positive off-source support is therefore not
driving the fitted-source growth.

Global S/N-only selection is too brittle for this dataset:

- S/N 50 selects no a1400 samples after the first pass;
- S/N 100 selects no a1400 samples at all and no a1100 samples until the third
  pass; and
- S/N 200 selects no samples in any array.

The compact adaptive gates maintain support in all arrays and are the more
useful source-only diagnostic.

### Projection and convergence response

Bilinear projection changes final amplitudes by at most 2.14% relative to the
Jinc control. The historical truncating projection and center convention
change them by at most 2.70%. Naive mapmaking changes the seed as expected but
retains similar final/seed growth. Projection choice is not the origin of the
observed correction.

The unchanged ten-iteration run converges. Between iterations eight and nine,
the fitted amplitude changes by -0.043%, -0.008%, and +0.007% for a1100,
a1400, and a2000. Relative whole-map RMS changes continue shrinking and reach
1.69%, 1.04%, and 1.16%. Iterations zero through four remain exactly
reproducible.

## Decision Outcome and Remaining Gate

- Subtraction/add-back differences are confined to a tiny sample set newly
  flagged during residual-weight resetting and are too small to explain the
  source change.
- Learning modifies later convergence by less than 0.32% and does not explain
  its origin.
- Template taper and source-subtracted detector weights are not material for
  this observation.
- Compact-support trajectories reproduce the broad-support result; broad
  positive model support is not causal.
- Growth scales strongly with PCA depth, and source masking materially changes
  it; PTC source transfer is the dominant mechanism.
- Projection and mapmaking variants retain the behavior; map/project operator
  consistency is not the dominant mechanism.
- The unchanged ten-iteration trajectory reaches an asymptote; the current
  recurrence is stable on this observation.
- A production policy is acceptable only after a realistic injected-source
  test with the full PTC cleaner converges to known amplitude and PSF with
  shrinking changes.

The current synthetic injection establishes the recurrence test seam. The
production-PTC injected-source pair described below replaces its controlled
scalar cleaner with the configured production path while retaining the real
observation background.

## Production-PTC Injected-source Pair

Current development after the follow-up matrix adds an opt-in
`timestream.fruit_loops.injected_source_test` diagnostic. It does not export
and reload a multi-gigabyte timestream. Instead, on and after a requested
iteration it scales the pristine unit-kernel TOD by a declared per-array
amplitude and adds that source immediately before the previous signal and
kernel maps are subtracted.

The operation is:

\[
  d_\mathrm{PTC,in}
    = d_\mathrm{RTC} + A K_\mathrm{RTC} - M_{n-1},
\]

where \(d_\mathrm{RTC}-M_{n-1}\) is the ordinary converged residual,
\(K_\mathrm{RTC}\) is the production unit-source kernel after the same RTC
operations, and \(A\) is the known injected amplitude. The existing kernel
TOD is not modified by the injection. No additional full-size matrix is
allocated.

The authoritative experiment is paired:

1. restart `control` and `injected` from the same completed checkpoint;
2. retain identical learned masks, exclusions, cleaner policy, weights,
   mapmaking, and iteration count;
3. change only `injected_source_test.enabled`;
4. save every iteration; and
5. fit the `injected - control` signal map relative to the propagated kernel.

This construction cancels the real residual background to first order while
preserving any nonlinear interaction between the injected source and the
production cleaner. It measures:

- recovered amplitude divided by injected amplitude;
- recovered major/minor FWHM divided by kernel FWHM;
- recovered-source/kernel centroid separation;
- shrinking successive transfer-map differences;
- control/injected kernel and map-weight differences; and
- ordinary control/injected pointing fits and S/N.

The diagnostic is deliberately fail-closed. It is accepted only for
pointing/OOF reductions with fruit loops, kernel generation, diagnostics, and
saved iterations enabled. The start iteration must be at least one and below
`max_iters`; amplitudes must contain three finite nonnegative values in
`[a1100, a1400, a2000]` order, with at least one positive value. Runtime also
requires `mJy/beam`, matching signal/kernel shapes, finite kernel samples, and
at least one realized nonzero projected sample.

Production defaults remain disabled. The setup and comparison commands are
documented in `tools/fruit_loops/README.md`. The comparator derives absolute
iteration identity from the map FITS header, since a restarted run in a fresh
output root writes absolute iteration 9 to `redu00`, and rejects paired config
drift beyond the output root and injection enable switch.

## First Injected-source Pair: Quarantined

The first control/injected pair completed absolute iterations 9--13 with
executable `v4.0.0-3594-ga5fcad29`. Both branches loaded the same iteration-8
checkpoint, completed without error-level messages, and the injected branch
projected 465,874 nonzero model samples in every pass.

The required continuation control nevertheless failed. The restarted control
at absolute iteration 9 differs materially from the uninterrupted iteration-9
product generated immediately after the checkpoint source:

| Array | Signal relative RMS | Kernel relative RMS | Weight relative RMS |
|---|---:|---:|---:|
| a1100 | 16.5% | 0.62% | 21.4% |
| a1400 | 26.6% | 2.43% | 49.7% |
| a2000 | 4.61% | 0.87% | 8.35% |

The configs differ only by the expected new output root, absolute stop
iteration, restart path, and disabled injected-source diagnostic. Threads,
inputs, and all scientific settings are otherwise unchanged. The executable
between the uninterrupted and restarted products differs only by
documentation/tooling and the inactive injection path.

The root cause is missing operational checkpoint state. Validated PTC
weighting learns detector factors once and retains these fields in `PTCProc`
across fruit-loop iterations:

- accumulated/finalized identity;
- ratio and atmospheric sums and counts; and
- final detector penalty and validation vectors.

Schema v1 stored reduction-learning masks and detector exclusions, but none of
these processed-weight validation fields. A fresh process resumed at iteration
9 with empty state, omitted the established validation factors for that pass,
and then relearned different factors from iteration 9. The observed detector
weight and map differences are the direct result. This disproves the v1
checkpoint's claim of exact continuation for validated weighting.

The paired subtraction still produced a stable-looking trajectory: recovered
PSF widths approached the propagated kernel, centroids remained within 0.05
arcsec, successive changes shrank, and iteration-13 amplitudes reached about
85.0%, 83.5%, and 89.7% of the injected a1100, a1400, and a2000 truth.
Those numbers are not scientifically authoritative because both branches
started from the wrong realized weighting state and then learned
branch-dependent replacements.

Checkpoint schema v2 now stores and validates the complete retained
weight-validation state plus a canonical processed-timestream policy
snapshot. Version-1 checkpoints fail closed. The comparator now requires an
uninterrupted continuation reference and demands exact signal, kernel, and
weight image identity before measuring injected-source transfer.

The corrected Unity sequence is:

1. run one uninterrupted ten-iteration control with the v2 executable;
2. restart a control and injected branch from its `redu08` checkpoint;
3. require restarted control iteration 9 to equal uninterrupted `redu09`
   exactly; and
4. only then interpret the injected-minus-control transfer trajectory.

## Local Verification

The investigation and schema-v2 repair pass the local `citlali_cli` and
`citlali_test` builds. All 514 enabled CTests pass; one unrelated lifecycle
test remains disabled. Six focused restart tests cover finalized and
partially accumulated weight state, malformed state, policy mismatch, split
learning state, and lifecycle restoration. The complete config preflight
passes 123 Python unit tests, all four mode kits, compact compatibility, schema
generation, and every typed authority audit.

The ablation generator accepts the downloaded frozen obsnum 133410 low-level
config and can produce either the completed initial matrix or the independent
follow-up matrix. The comparison tool reproduces all array/iteration records
and feedback-support measurements and now discovers runs longer than five
iterations.

The full-PTC injected-source implementation and local test seam are complete,
but the first Unity pair is invalidated by the v1 restart defect. Existing
ablation evidence establishes stable PSF-shape and amplitude transfer recovery
whose size scales with cleaner strength. It does not demonstrate a production
fruit-loop fault, nor does it establish absolute amplitude correctness.
Production defaults remain unchanged.
