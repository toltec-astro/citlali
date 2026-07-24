# Pointing Fruit-Loop Feedback Investigation

Date: 2026-07-24

Status: active; controlled Unity ablations required before a production
conclusion or default change.

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
pixel-selection gate, detector flags, and calibration geometry for subtraction
and add-back. A new regression test confirms that an immediate subtract/add
round trip restores both signal and kernel TOD to numerical precision. A
second controlled test injects a Gaussian of known amplitude and width,
applies a known linear attenuation, and exercises the production map-to-TOD
and naive-mapmaking recurrence. Its amplitude approaches the injected truth,
iteration changes shrink, and its PSF width remains stable.

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
field from cleaning can interact with detector weights and map normalization.
This is currently the strongest additional mechanism exposed by the code and
product audit, but it is not yet a demonstrated root cause.

## Provisional Cause Classification

| Candidate | Current disposition |
|---|---|
| Fundamental subtract/clean/add-back recurrence | Basic recurrence and round trip are mathematically correct. An interaction with the production cleaner remains open. |
| Learned masks or detector exclusions | Cannot cause the first large jump because application starts at iteration two. It may alter later convergence. |
| Template weight taper | Full gain applies across the central positive source region. It materially changes broad low-weight positive background support and may interact with cleaning, but is unlikely to be the sole central-amplitude cause. |
| Source-subtracted detector weights | Active on the first feedback pass and remains a plausible source of iteration-dependent transfer. The recompute-after-add-back ablation directly tests it. |
| Kernel normalization | Not the sole cause: source/kernel-corrected amplitude and width still grow. It may track part of the transfer change. |
| Interaction | Most plausible current class. The leading interaction is broad one-sided model support with cleaning and source-subtracted detector weights; learning may modify later passes. |

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

## Decision Rules

- If subtraction and add-back projection summaries differ within a scan, the
  recurrence has mutated state and the native path is faulty.
- If only learning-disabled runs change after iteration two, learning modifies
  convergence but does not explain its origin.
- If disabling template taper changes only broad support while central growth
  remains, taper is not the root cause.
- If recomputing weights after add-back removes growth while preserving
  shrinking changes and source shape, source-subtracted detector weights are
  the dominant interaction.
- If all three ablations retain comparable growth, the next one-change test
  must restrict the low absolute-flux model support before altering any
  production policy.
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
config and produces the five independent configs described above. The
comparison tool reproduces all 15 array/iteration records and the feedback
support measurements from the existing run.

Unity ablation products and the full-PTC injected-source fixture remain open.
Consequently, this record does not claim that the observed real-source growth
is correct recovery or a production feedback fault. It narrows the leading
hypothesis to an interaction and supplies the measurements needed to
distinguish the candidates without changing defaults.
