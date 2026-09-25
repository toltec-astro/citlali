# POINT 152389 guarded feedback comparison — 2026-09-25

The supplements have an array-dependent benefit, with a material tradeoff in a2000. a1100 is a null comparison: all three policies produce identical new trajectories. In a1400, CORR improves the injected shoulder response beyond both BASE-POINT and DIAG without appreciable seed persistence. In a2000, both supplements improve shoulder recovery, but CORR leaves a seed-induced difference and worsens compact positive-response distortion. These results do not support blanket adoption of CORR or DIAG.

All 45 trajectories reached T10: exactly 450 feedback applications, no blocked/stopped cell, no new processing failure or source loss. Twelve authenticated physical M0 products were reused; there were zero initial PTC/MAP calculations, RTC/CAL reruns, or new calibration/validation fields. The 27 ideal-addition fits and nine endpoint covariance fits all completed. Every one of the 495 recorded stage guards passed; minimum frozen-reference amplitude S/N was 79.97 including M0, and 98.13 across the 450 feedback outputs. T10 consumed m9; m10 and all same-map cross-candidates remain unapplied.

The results concern finite full-procedure response at the post-RTC/CAL boundary. They neither isolate the true sky component nor establish absolute flux calibration. No new pass criterion, combined score, endpoint source error, or ten-cycle false-feedback guarantee is inferred.

## Definitions and fixed measurement contract

POINT-EXPECTED-v1 is unchanged: one unweighted seven-parameter elliptical Gaussian plus constant, deterministic initialization and 3,000-evaluation limit, 40 arcsec fit disk; positive peak above the constant, centroid radius ≤20 arcsec, both sigmas strictly >1.5 arcsec with lower-width-bound rejection, and A/sigma_A >10. Full frozen C0 propagates through all seven nuisance columns for every policy. sigma_A remains a conditional reference-scale guard uncertainty, not a new endpoint uncertainty. Fixed templates, apertures, original centroid/radial bins and original benchmark masks retain their archived definitions. The fit's major/minor widths and Cartesian ellipse tensor avoid angle-gauge artifacts. Apertures report background-subtracted brightness area, with raw brightness area and fitted-constant contributions separately retained.

Brightness and peak are mJy/nominal beam; offsets/FWHM are arcsec; aperture quantities are (mJy/nominal beam) arcsec². Nominal beams remain 5, 7.5 and 10 arcsec and WCS remains FK5/J2000. g is unweighted projection onto the declared injection-only MAP reference, eta=||Delta−I||/||I|| and rho=||Delta−gI||/||I||; eta²=(g−1)²+rho². All response references use each policy's own REF forward support, bypassing PTC. Zero-injection pixels remain in the benchmark. All 297 comparisons retain the entire original physical benchmark, with no paired native support loss/gain or normalization change. Both original and native-domain results are saved.

## Historical BASE versus BASE-POINT

The new guarded fit is not substituted into old records. New BASE-POINT and historical BASE have identical candidate membership at all stored stages across all five branches and arrays, but their fitted constants and retained brightness values are not bitwise identical. Largest all-stage source-stamp map changes are 2.48e−7, 0.00180 and 4.40e−7 mJy/beam for a1100/a1400/a2000. The a1400 exception comes from C-core-minus T6: its angle contacts the upper bound, producing a −0.16840 peak and +0.005278 constant change relative to the historical fit. This is accepted by the unchanged guard, which rejects lower-width bounds, not every active constraint. The corresponding DIAG T6 fit has the same flag. These are the only two active-bound fits among the 495 new stages; none of the 27 ideal fits is constrained.

At T10, the largest absolute BASE-POINT-minus-historical peak changes across the five branches are only 1.77e−5, 1.60e−5 and 4.02e−6 mJy/beam. The a1400 compact-minus retained brightness still differs by up to 2.78e−5 mJy/beam at T10. Its transient map-response gain difference is visible in the historical figure. Thus the fit change is measurable and fully retained, but does not explain the substantial shoulder response changes from supplemental admission.

## T10 response on the original physical benchmark

Each cell below is g / eta / rho. Core-minus g uses the signed negative truth. Values near unity in g alone do not establish low distortion.

| Array | Policy | Compact + | Compact − | Shoulder |
|---|---|---|---|---|
| a1100 | BASE-POINT | 1.00074 / 0.01392 / 0.01390 | 1.00023 / 0.01187 / 0.01187 | 0.97437 / 0.07571 / 0.07124 |
| a1100 | CORR | 1.00074 / 0.01392 / 0.01390 | 1.00023 / 0.01187 / 0.01187 | 0.97437 / 0.07571 / 0.07124 |
| a1100 | DIAG | 1.00074 / 0.01392 / 0.01390 | 1.00023 / 0.01187 / 0.01187 | 0.97437 / 0.07571 / 0.07124 |
| a1400 | BASE-POINT | 0.99744 / 0.00894 / 0.00857 | 1.00524 / 0.03317 / 0.03276 | 0.91107 / 0.14559 / 0.11527 |
| a1400 | CORR | 0.99760 / 0.00855 / 0.00821 | 1.00541 / 0.03327 / 0.03283 | 0.96444 / 0.09561 / 0.08875 |
| a1400 | DIAG | 0.99744 / 0.00894 / 0.00857 | 1.00524 / 0.03317 / 0.03276 | 0.92774 / 0.12821 / 0.10591 |
| a2000 | BASE-POINT | 1.00831 / 0.04679 / 0.04605 | 1.01537 / 0.04688 / 0.04428 | 0.84988 / 0.22117 / 0.16242 |
| a2000 | CORR | 1.02343 / 0.09119 / 0.08813 | 0.99643 / 0.06040 / 0.06029 | 0.99210 / 0.11327 / 0.11299 |
| a2000 | DIAG | 1.01300 / 0.05475 / 0.05318 | 0.98831 / 0.08650 / 0.08571 | 1.00450 / 0.13126 / 0.13118 |

a1100 adds no supplemental pixels in any branch: the map responses, candidates and finite sign asymmetry coincide. The shoulder gain is 0.97437 with eta 0.07571; a residual bowl remains visible in the signed profiles. The compact gain is close to unity but the positive and negative residuals are not perfectly antisymmetric.

a1400 CORR raises shoulder g from 0.91107 to 0.96444 and lowers eta from 0.14559 to 0.09561; DIAG gives g 0.92774 and eta 0.12821. The CORR benefit therefore survives this matched DIAG comparison for this array and injection. Compact response changes are small and mixed: positive eta improves slightly, negative eta rises slightly. The shoulder deficit in the central/wings profile is reduced, not removed.

a2000 CORR and DIAG raise shoulder g from 0.84988 to 0.99210 and 1.00450, reducing eta from 0.22117 to 0.11327 and 0.13126. CORR has lower shoulder distortion than DIAG, but positive compact eta rises from 0.04679 to 0.09119 (DIAG 0.05475), and negative compact eta rises from 0.04688 to 0.06040 (DIAG 0.08650). Compact even-map RMS is 0.42930 / 0.93656 / 0.82313 mJy/beam for BASE-POINT/CORR/DIAG. Sign asymmetry and measurement dependence remain substantial; the gain improvement is not a uniform recovery improvement.

## Ideal map-addition controls and model mismatch

The table shows the actual shoulder increment and its signed discrepancy from REF plus the injection-only reference, using the same full-map fit. Absolute discrepancies precede any ratios in the full measurement tables. Undefined/negligible ideal increments have unavailable ratios. No positive-source fit was applied to a signed difference map.

| Array | Policy | Actual peak increment | Peak discrepancy | Actual template increment | Template discrepancy | Actual aperture-2 increment | Aperture-2 discrepancy | Fitted constant: actual / ideal increment |
|---|---|---:|---:|---:|---:|---:|---:|---|
| a1100 | BASE-POINT | 20.3270 | 0.0647 | 64.3021 | -0.8303 | 6293.1717 | -263.8524 | 0.38689 / 0.60105 |
| a1100 | CORR | 20.3270 | 0.0647 | 64.3021 | -0.8303 | 6293.1717 | -263.8524 | 0.38689 / 0.60105 |
| a1100 | DIAG | 20.3270 | 0.0647 | 64.3021 | -0.8303 | 6293.1717 | -263.8524 | 0.38689 / 0.60105 |
| a1400 | BASE-POINT | 15.5171 | 0.6328 | 44.8334 | -1.1793 | 5941.9193 | -634.2297 | 0.22240 / 0.62801 |
| a1400 | CORR | 14.8536 | -0.0258 | 45.4994 | -0.5137 | 6309.1946 | -267.1068 | 0.40004 / 0.62777 |
| a1400 | DIAG | 15.2241 | 0.3398 | 45.0042 | -1.0085 | 6047.2840 | -528.8650 | 0.26756 / 0.62801 |
| a2000 | BASE-POINT | 20.6278 | 0.3811 | 52.7623 | -2.4847 | 11549.1184 | -1951.0998 | 0.41245 / 1.44820 |
| a2000 | CORR | 18.6693 | -1.4739 | 55.3081 | 0.0387 | 13613.5180 | 100.8899 | 1.16106 / 1.43704 |
| a2000 | DIAG | 18.6334 | -1.5958 | 55.4292 | 0.1754 | 13604.9137 | 100.9711 | 1.14033 / 1.44485 |

The broad shoulder shifts the fitted constant even in the ideal control: it is not represented exactly by one Gaussian. For a1400, CORR reduces both peak and aperture discrepancies. In a2000 the shoulder aperture discrepancy improves from −1951.10 to +100.89 (CORR) or +100.97 (DIAG), while the peak discrepancy worsens from +0.38108 to −1.47392 or −1.59576 mJy/beam. The constant's shift toward its ideal value accompanies better recovery; it is not independently a noise or bias estimate. Peak, fixed-template, aperture and full-map response answer different questions. The shoulder signal is not to be replaced by a Gaussian feedback model.

## Data-supported shoulder versus first-candidate-only seed

Projection is the dimensionless projection onto the archived shoulder raster, including its deliberate zero pixels. A value of 1 corresponds to the supplied shoulder amplitude, not a fraction of the entire source. The seed-only parent data are unchanged.

| Array | Policy | Data shoulder projection | Seed total projection T10 | Seed candidate projection T10 | Seed full-map RMS | Seed constant shift | Seed aperture-2 shift |
|---|---|---:|---:|---:|---:|---:|---:|
| a1100 | BASE-POINT | 0.9743651 | 2.420785e-07 | 1.791397e-07 | 6.254806e-07 | 4.149783e-07 | 0.001802328 |
| a1100 | CORR | 0.9743651 | 2.420785e-07 | 1.791397e-07 | 6.254806e-07 | 4.149783e-07 | 0.001802328 |
| a1100 | DIAG | 0.9743651 | 2.420785e-07 | 1.791397e-07 | 6.254806e-07 | 4.149783e-07 | 0.001802328 |
| a1400 | BASE-POINT | 0.9110713 | 3.821998e-07 | 2.876977e-07 | 9.181673e-07 | 8.646021e-07 | 0.002318116 |
| a1400 | CORR | 0.9644394 | 3.907437e-07 | 2.98385e-07 | 9.336274e-07 | 7.819564e-07 | 0.002435549 |
| a1400 | DIAG | 0.9277432 | 3.821998e-07 | 2.876977e-07 | 9.181673e-07 | 8.646021e-07 | 0.002318116 |
| a2000 | BASE-POINT | 0.849881 | 6.374848e-06 | 4.56095e-06 | 2.173798e-05 | 1.631587e-05 | 0.08306319 |
| a2000 | CORR | 0.992104 | 0.01023426 | 0.02030964 | 0.09629517 | 0.07139818 | 173.5552 |
| a2000 | DIAG | 1.004496 | 6.176615e-06 | 4.437593e-06 | 2.129721e-05 | 1.668057e-05 | 0.080749 |

a1100 and a1400 show vanishing seed differences by T10, and a2000 BASE-POINT/DIAG are also still decaying toward their respective REF trajectories. a2000 CORR instead retains a total shoulder projection of 0.010234 (about 1.02% of the artificial shoulder), with candidate projection 0.020310. It is much smaller than the data-supported shoulder but far above the other policies' seed residuals. CORR's seed endpoint has 92 admitted pixels/22 extras versus its REF's 83/13. This demonstrates initialization sensitivity over ten cycles; it does not establish distinct asymptotic fixed points or prove that the observation lacks physical shoulder structure.

The a2000 CORR seed projection changes only from 0.010265 at T9 to 0.010234 at T10. The corresponding absolute seed-map aperture-2 drift is +0.41762, while REF drifts +0.79183; the seed-minus-REF aperture residual therefore changes by −0.37421. CORR's data-shoulder aperture still changes by +19.1695 from T9 to T10, compared with +2.23859 BASE-POINT and +1.35516 DIAG. No cycle 11 was run.

## Admission, source shape and same-map checks

| Array | Policy | REF peak | East / north | FWHM major / minor | Constant | REF pixels / extras | Shoulder pixels / extras | REF peak T10−T9 |
|---|---|---:|---|---|---:|---|---|---:|
| a1100 | BASE-POINT | 937.42168 | 4.62774 / -5.14008 | 6.35309 / 5.51348 | -1.58352 | 35 / 0 | 46 / 0 | 0.0000406 |
| a1100 | CORR | 937.42168 | 4.62774 / -5.14008 | 6.35309 / 5.51348 | -1.58352 | 35 / 0 | 46 / 0 | 0.0000406 |
| a1100 | DIAG | 937.42168 | 4.62774 / -5.14008 | 6.35309 / 5.51348 | -1.58352 | 35 / 0 | 46 / 0 | 0.0000406 |
| a1400 | BASE-POINT | 664.73408 | 4.39358 / -5.29799 | 7.38825 / 6.91647 | 1.71226 | 49 / 0 | 54 / 0 | 0.0000858 |
| a1400 | CORR | 664.70103 | 4.39372 / -5.29828 | 7.38847 / 6.91775 | 1.71621 | 50 / 1 | 66 / 6 | 0.0000805 |
| a1400 | DIAG | 664.73408 | 4.39358 / -5.29799 | 7.38825 / 6.91647 | 1.71226 | 49 / 0 | 58 / 4 | 0.0000858 |
| a2000 | BASE-POINT | 813.61770 | 4.60435 / -5.20012 | 9.67932 / 9.37208 | 1.43260 | 68 / 0 | 76 / 0 | 0.0017673 |
| a2000 | CORR | 812.88474 | 4.61026 / -5.18929 | 9.71074 / 9.38810 | 1.58931 | 83 / 13 | 142 / 52 | 0.0019675 |
| a2000 | DIAG | 813.36386 | 4.60503 / -5.19488 | 9.68849 / 9.38094 | 1.48350 | 71 / 1 | 129 / 39 | 0.0015857 |

[Same-map T10 cross-check: all 27 unapplied candidates](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/CROSS_CANDIDATES.md). These share the producing map's fit, separating selection changes from subsequent cleaning changes. All-stage candidate masks, turnover, brightness area, unavailable tests and PTC/MAP accounting are retained. No source is lost; no candidate reaches the inherited forbidden boundary.

## Endpoint background and covariance applicability

Only the nine REF T10 endpoints receive the existing covariance-family analysis, with original masks and request. No covariance was fed back. Background scatter describes the frozen background region, not the source's uncertainty. Median scale ratios compare the same supported background pixels; matched neighborhood variance uses the same weights and exclusions in endpoint and C0 calculations.

| Array | Policy | Background std / robust scale | Median scale / C0 | CORR variance median / C0 | DIAG variance median / C0 | Matched neighborhoods |
|---|---|---|---:|---:|---:|---:|
| a1100 | BASE-POINT | 15.57133 / 14.27279 | 0.997529 | 0.947151 | 0.967163 | 962 |
| a1100 | CORR | 15.57133 / 14.27279 | 0.997529 | 0.947151 | 0.967163 | 962 |
| a1100 | DIAG | 15.57133 / 14.27279 | 0.997529 | 0.947151 | 0.967163 | 962 |
| a1400 | BASE-POINT | 12.50709 / 11.76534 | 0.995064 | 0.980604 | 0.999862 | 812 |
| a1400 | CORR | 12.50706 / 11.76461 | 0.995060 | 0.980550 | 0.999858 | 812 |
| a1400 | DIAG | 12.50709 / 11.76534 | 0.995064 | 0.980604 | 0.999862 | 812 |
| a2000 | BASE-POINT | 20.66608 / 16.50328 | 0.992466 | 1.014578 | 0.982918 | 641 |
| a2000 | CORR | 20.66638 / 16.51586 | 0.992464 | 1.016582 | 0.983064 | 639 |
| a2000 | DIAG | 20.66576 / 16.49393 | 0.992456 | 1.015142 | 0.982873 | 639 |

Endpoint background scales remain close to C0: the median ratios are about 0.998/0.995/0.992, and the directional correlations remain similar. There is no observed 25% background amplitude increase comparable to the earlier stress case. This does not erase that stress evidence: at 1.25 amplitude the prior joint exceedance counts were 33/23/17 per256 and final CORR supplemental events 0/1/2, versus zero DIAG final events. Those reused tests are not ten-cycle false-feedback probabilities.

All 1,257 source-disk pixels and the full/nonzero-weight footprints of the matched endpoint neighborhoods lie in the endpoint amplitude-interpolation domain. This coverage statement does not validate transferring background covariance to the source. Regional covariance, source/background depth and network mixtures remain distinct and can be confounded. The original a2000 source-transfer/model qualification remains; frozen guard sigma_A is not recalibrated. The source-location covariance is not observed by these masked-background comparisons, nor can separate variances be added to infer uncertainty of paired map differences.

## Numerical, resource and evidence checks

Independent trajectory inspection confirms 9,900 processing segments received fresh residual-specific rank-10 PTC, with unchanged eligibility, causes and application-failure counts. All 150 appearances of the inherited a1100 network-3/scan-5 nonconvergence retain its 1,155 failed application times; no new segment failure is introduced. Maximum subtract/rejoin error is 4.09e−11 and maximum applied-operand remap error 4.30e−11, below the frozen 1e−8 tolerance. All 198 preformed injected parents and nine initial seed candidates satisfy their exact controls. No r-channel injection or learned-state reuse on changed residuals occurred.

The four sequential feedback phases took 262.54, 272.18, 502.90 and 400.01 seconds (23.96 minutes summed phase wall time, including final phase authentication). They used 3/6/12/12 independent workers. Native CPU summed across processes is 7,201.39 seconds, not elapsed time. Largest sampled aggregate resident memory was 16.64 GB; native products total 123.40 GB by recorded byte counts. Native workers and requested numerical threads were each one; actual Accelerate library thread count was not observable. The separate report process took 290.93 seconds wall, with 276.99 seconds recorded before final authentication/manifest. Figure-only publication and full seal timings are separately preserved.

Two report-publication attempts failed (FITS header object and existing non-overwriting FITS output). Both are preserved. The first had already completed one endpoint fit and one ideal fit; these were carried byte-for-byte into the final publication, so the final census remains nine and27 scientific fit attempts, with no repeated fits. First-attempt duration ~508.89 seconds is a filesystem-lifetime proxy, not a dedicated stopwatch. The second report attempt took 97.61 seconds wall; the subsequent figure-only publication took 17.55 seconds, with zero fits. These are reporting failures, not suppressed science failures.

Validation includes native seven controls,20 CLI tests, nine focused CTest checks,207 baseline tests plus137 subtests, required configuration preflight, and all305 successor tests covered by the corrected304-pass/one-skip run plus the separately corrected frozen fixture (111 subtests also passed). Focused reporting/closure tests pass. Initial setup failures and unbuilt unrelated CTest targets are retained; no full CTest claim is made. This is local cached-dependency evidence, with the recorded dirty dependency revision, not a Unity/Spack replication. Original consumed inputs are authenticated before/after phases and again at closure. Every final payload is freshly hashed and every nested ledger reconciled during sealing.

## Figures and detailed products

All27 scientific figures were inspected. Historical labels were shortened to avoid clipping; each residual row shares a scale across the three policies, with the bright known injection on its own labeled scale. Original figure publications are preserved. This display-only repair performs no fit or reduction. Source-region map figures cover ±41 arcsec; full response FITS preserve the full benchmark. Seed map rows have their own visibly labeled scale: tiny a1100/a1400 residuals must not be compared by apparent color saturation with a2000 CORR.

- a1100: [response](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1100/response.png), [seed and data](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1100/seed.png), [residuals](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1100/residuals.png), [profiles](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1100/profiles.png), [endpoint](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1100/endpoint.png), [ideal measurements](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1100/MEASUREMENTS.md), [guards](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1100/GUARDS.md), [drift and raw apertures](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1100/DRIFT.md).
- a1400: [response](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1400/response.png), [seed and data](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1400/seed.png), [residuals](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1400/residuals.png), [profiles](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1400/profiles.png), [endpoint](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1400/endpoint.png), [ideal measurements](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1400/MEASUREMENTS.md), [guards](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1400/GUARDS.md), [drift and raw apertures](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a1400/DRIFT.md).
- a2000: [response](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a2000/response.png), [seed and data](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a2000/seed.png), [residuals](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a2000/residuals.png), [profiles](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a2000/profiles.png), [endpoint](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a2000/endpoint.png), [ideal measurements](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a2000/MEASUREMENTS.md), [guards](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a2000/GUARDS.md), [drift and raw apertures](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/a2000/DRIFT.md).

[All histories, all nine figure types per array and snapshot tables](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/REPORT.md); [machine-readable full results](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925/report/SUMMARY.json).

## Complete execution matrix

| Array | Policy | Branch | Result |
|---|---|---|---|
| a1100 | BASE-POINT | REF | T10 complete |
| a1100 | BASE-POINT | C-core-plus | T10 complete |
| a1100 | BASE-POINT | C-core-minus | T10 complete |
| a1100 | BASE-POINT | C-shoulder | T10 complete |
| a1100 | BASE-POINT | D-seed-shoulder | T10 complete |
| a1100 | CORR | REF | T10 complete |
| a1100 | CORR | C-core-plus | T10 complete |
| a1100 | CORR | C-core-minus | T10 complete |
| a1100 | CORR | C-shoulder | T10 complete |
| a1100 | CORR | D-seed-shoulder | T10 complete |
| a1100 | DIAG | REF | T10 complete |
| a1100 | DIAG | C-core-plus | T10 complete |
| a1100 | DIAG | C-core-minus | T10 complete |
| a1100 | DIAG | C-shoulder | T10 complete |
| a1100 | DIAG | D-seed-shoulder | T10 complete |
| a1400 | BASE-POINT | REF | T10 complete |
| a1400 | BASE-POINT | C-core-plus | T10 complete |
| a1400 | BASE-POINT | C-core-minus | T10 complete |
| a1400 | BASE-POINT | C-shoulder | T10 complete |
| a1400 | BASE-POINT | D-seed-shoulder | T10 complete |
| a1400 | CORR | REF | T10 complete |
| a1400 | CORR | C-core-plus | T10 complete |
| a1400 | CORR | C-core-minus | T10 complete |
| a1400 | CORR | C-shoulder | T10 complete |
| a1400 | CORR | D-seed-shoulder | T10 complete |
| a1400 | DIAG | REF | T10 complete |
| a1400 | DIAG | C-core-plus | T10 complete |
| a1400 | DIAG | C-core-minus | T10 complete |
| a1400 | DIAG | C-shoulder | T10 complete |
| a1400 | DIAG | D-seed-shoulder | T10 complete |
| a2000 | BASE-POINT | REF | T10 complete |
| a2000 | BASE-POINT | C-core-plus | T10 complete |
| a2000 | BASE-POINT | C-core-minus | T10 complete |
| a2000 | BASE-POINT | C-shoulder | T10 complete |
| a2000 | BASE-POINT | D-seed-shoulder | T10 complete |
| a2000 | CORR | REF | T10 complete |
| a2000 | CORR | C-core-plus | T10 complete |
| a2000 | CORR | C-core-minus | T10 complete |
| a2000 | CORR | C-shoulder | T10 complete |
| a2000 | CORR | D-seed-shoulder | T10 complete |
| a2000 | DIAG | REF | T10 complete |
| a2000 | DIAG | C-core-plus | T10 complete |
| a2000 | DIAG | C-core-minus | T10 complete |
| a2000 | DIAG | C-shoulder | T10 complete |
| a2000 | DIAG | D-seed-shoulder | T10 complete |

## One next bounded action — recommendation only

Diagnose the existing a2000 CORR seed-persistence path using saved T1–T10 maps, candidate brightness/membership, neighborhood evidence and PTC receipts, matched to its own REF and DIAG. Locate when the lasting membership difference appears and how it relates to the compact sign-asymmetry and shoulder residuals. Use only saved products: no new fitting, trajectories, seeds, threshold/noise changes or production adoption. This recommendation has not been executed.

Stop here. No policy/default change, new uncertainty campaign, cycle11, m10 application, JINC, Unity, merge or push. Exact source, binary, review and final manifest identities are in COMPLETION.json and provenance/SOURCE.json; the repository closeout records the final seal.
