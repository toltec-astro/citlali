# POINT step assessment — 2026-09-27

The unchanged B method admits ten times as many detector-event groups in selected POINT152420 as in POINT152430. Apply follows its declared exclusions and complete filter footprints exactly. The strongest next issue is **Learn coverage for finite, modest-slope transitions that never produce a candidate**: two discovery windows show persistent x changes passing into CAL and largely being removed only by PTC. Their final astronomical impact is not established by this bounded sample. No treatment revision is made or automatically authorized by this finding.

| Finding | 152430 B | 152420 B |
|---|---|---|
| Full population | nw11 486 D; nw12 491 D | Same detector counts, own original/tune/APT inputs |
| Native candidate instances → event groups → admitted detector-event groups | 754 → 109 → 9 | 333 → 248 → 89 |
| Admitted affected D | 7/486 nw11 (1.44%); 0/491 nw12 | 26/486 nw11 (5.35%); 25/491 nw12 (5.09%) |
| Input-valid x detector-time | 61,374.783 s | 60,354.552 s |
| Admitted groups per 1,000 valid detector-s | 0.147 | 1.475 |
| Original event-window review | 2 plausible steps, 8 plausible non-steps, 2 unresolved / 12 | 9 plausible steps, 1 plausible non-step, 2 unresolved / 12 |
| Original unflagged-window review | 0 plausible steps, 11 plausible non-steps, 1 unresolved / 12 | 2 plausible steps at one shared r-dominated epoch, 9 plausible non-steps, 1 unresolved / 12 |
| Gross admitted physical-transition / broader jump exclusions | 4.178 / 124.035 detector-s | 4.768 / 1,162.691 detector-s |
| Actual CAL/MAP contributions, M0 and T5 | 2,870,933 | 2,814,220 |
| Plan versus realized filter support | Exact; zero crossing or replacement influence | Exact; zero crossing or replacement influence |
| Raw-to-result command exit (one diagnostic-heavy run) | 75.186 s, CPU 74.101 s | 75.516 s, CPU 72.709 s |

The two observations are development challenge cases, not alternative treatments of the same sky/data. Absolute flux, noise, and runtime differences do not measure treatment improvement. The pre-replay inventory distinguishes retained facts from missing diagnostics. `compact/report/B-REF-rep1/BASELINE_ASSESSMENT.md` was established before the selected trajectory.

## What was exercised

152420/0/2 was frozen from three candidates (152420, 152432, 152434) before any new final map. Original I/Q excerpts supported repeated finite transitions, including three separated epochs with multiple detector responses. The raw discovery denominator used APT-usable receiver time; the table above instead uses realized native valid x integration time and all detector identities. Do not mix those denominators.

The new production result confirms a richer challenge. Its admitted groups form seven descriptive 50-ms clusters in nw11 and one 25-detector cluster in nw12, versus five small clusters in baseline nw11. These are time groupings, not eight independently proven physical events. Median admitted count per detector is zero in both observations; selected nw11 p95 is 1 and maximum 6, compared with baseline p95 0 and maximum 3. Repeated steps, finite transitions, overlapping support, short admitted runs, x/r-specific triggers, source-adjacent events and unresolved extents are present. Each network has one physical native run: real acquisition gaps and natural donor replacements are absent here and remain fixture-only evidence.

Selection uses no new map quality or source recovery. Four retained discovery time windows (their other tones are peer context) plus eight production-evidence windows comprise the selected 12-event review, not 16 windows. Each observation has 12 additional time/coverage-selected candidate-free windows. Selection files precede cleaned extraction. Adjudications are separately stored, do not change production masks, and are not a truth catalogue or precision/recall estimate. Unreviewed detector-time remains unassessed for completeness.

## Learn → Consider → Apply

Learning uses original aligned paired raw Δf/f, independent coordinate validity and spike evidence. Candidate evidence is tied to coordinate-local 10-s noise blocks; those software blocks are not physical gaps. Current plateau assessment uses primary cubic-plus-offset fits with 2-s flanks and at least 64 usable samples, with candidate/neighbor guards. A 1-s short fit must corroborate amplitude (both ≥5 frozen difference-noise scales), sign, and consistency within two scales. Transition/recovery evidence uses ±4 frozen residual scale and 50-ms sustained support, within the existing ±2-s search. The September 12 plateau rule and September 13 admitted-jump amendment are retained. One existing support-overlap reassessment uses frozen scales and explicitly records request, refit support/availability, and final reasons; unavailable isolated replacement authority is not silently granted.

Consider admits only supported persistent changes with complete recovery-search evidence and the existing source-authority conditions. Either coordinate can authorize paired consequences. An empty production protection mask is authority state, not a claim that no astronomical source was crossed. Geometric source overlays are separate diagnostics, using the unchanged own-source pointing definition.

Accepted jumps exclude affected detectors across intersected existing processing scans; three admitted original groups for a detector invoke the existing whole-observation exclusion. Unresolved extent can withhold a physical run. No offset subtraction, plateau stitching, shortened PCA segments, rank change, recalibration of r, or atmospheric subtraction was introduced. Physical/reset and downstream segmentation ownership remain native.

Apply uses B's 307-tap finite low-pass filter (153 native samples each side), factor-two phase-zero cadence, and complete admissible support. It does not pad or renormalize edges or bridge exclusions. The extractor independently eroded every declared admitted run and obtained the exact realized x support for both observations. Excluded seeds and broader jump regions are unavailable downstream; this is not a zero-residual measurement. Speed exclusions govern science centers separately from filter support. No replacement or donor influence occurred. Per-event native identities, original evidence, immutable decisions, and realized seed state are linked in `event-chain.yaml`; complete dispositions and full-population plots are in the saved report.

Native fixtures cover persistent versus spike behavior, finite transitions, overlap/short support, coordinate-specific paired decisions, pulse/source context, real gaps versus numerical boundaries, forbidden donors, plan realization, and active x-only perturbation. The new fixture changes learned candidates while preserving r/validity; the real source perturbations happen not to change discrete decisions.

## Exposure and residual diagnosis

The ledger first removes already-invalid input, then separately attributes outside-processing-science support, accepted transitions, jump scan exclusions, transient decisions, unresolved extent, other operation-unavailable support, remaining resolved/donor domain, and complete-filter expansion. Pairwise overlaps and the exact union are retained. Order-dependent incremental attribution is explicit; gross terms must not be added. Fitting guards protect evidence fits and are **not** automatically science-output exclusions.

For 152430, incremental physical transition + scan exclusion costs 119.407 detector-s after outside-science removal; unresolved extent adds 653.328 s and other unavailable support 7.430 s. For 152420 those corresponding costs are 1,123.582 s, 861.635 s, and 15.016 s. Complete-filter support expansion costs 2,476.212 s and 2,616.558 s respectively; this includes ordinary boundaries and all exclusions, so it is not all step-caused. Short admitted runs unable to host a complete footprint number 6 versus 20. Gross broader jump loss is much larger than the tiny admitted transition cells. Stable-looking stretches are consequently discarded, but their full recoverability and the benefit of a different policy are not established without the forbidden step-off counterfactual. This is a measured cost requiring care, not proof that all loss is excessive.

Native scheduled geometric source cells fall from 375,418 valid inputs to 326,669 RTC/CAL centers in baseline, and 410,563 to 360,522 in selected data. Masks preserve source/background overlap attribution. MAP normalization/hit/support statistics are retained. The source overlay itself never becomes a science mask.

Selected `n11_w4` (D462) and `n12_w5` (D206), chosen from original discovery evidence, have no nearby learned event and retain all native RTC samples. Their original x post-minus-pre contrasts are +4.748e-6 and −9.043e-6, versus adjacent-difference MAD 0.541e-6 and 1.614e-6. Corresponding RTC contrasts remain +4.848e-6 and −8.979e-6; CAL contrasts are +1,965.7 and −1,875.8 mJy/beam. At M0 they are −33.2 and −98.2; at T5 residual/sky-restored total −35.1 and −86.8 mJy/beam. These are descriptive window medians, not new plateau fits or independent significance. PTC strongly suppresses the common structure; good final maps therefore do not demonstrate complete RTC detection. Exact identity, original peers, masks and raw values are retained.

Two candidate-free nw12 control windows also show a smaller r-dominated plateau and return at the same epoch. This is a possible missed finite excursion; r is not an optically blank truth channel. Its x effect is less distinct from common variation. No universal missed-event rate or final-map harm follows from these examples. Accepted examples instead lose entire reviewed intervals; containment is demonstrated by absent admissible footprints, while residual leakage there is unmeasurable. Partially retained rejected/source-pulse windows and quiet controls show no identified crossing of a resolved exclusion. Remaining smooth offsets or oscillations outside masks are not by themselves forbidden leakage.

## Astronomical response

Own 152420 M0 instantiates the fixed raster recipe and admission-noise package; own T5 supplies the conditional uncertainty scale. Compact ±10% and displaced broad +5% x-only inputs run active Learn/Consider/Apply independently before CAL, with original r and native validity unchanged. All unmodified reference MAP contributions have valid mapping. The full unmodified reference ideal was retained: **zero** unavailable RTC comparison contributions, lost/gained science pixels, or changed HITS/normalization at M0/T5. All 248 events match by detector/native physical seed in every branch, with zero appeared/disappeared/shifted groups or changed decisions/support. Thus no decision sensitivity can be blamed for the remaining response discrepancy in this bounded injection test.

| Response | RTC projected gain | T5 peak ratio | T5 template ratio | T5 aperture 1 / 1.5 / 2 FWHM | T5 spatial gain / distortion |
|---|---:|---:|---:|---|---|
| compact +10% | 0.99620 | 0.99006 | 0.99349 | 0.99860 / 0.99251 / 0.99292 | 0.99293 / 0.03156 |
| compact −10% | 0.99620 | 0.98943 | 0.99384 | 0.99567 / 0.99643 / 1.00233 | 0.99037 / 0.02617 |
| broad +5% | 0.99897 | 1.00951 | 0.96242 | 0.97805 / 0.90429 / 0.86241 | 0.87338 / 0.20486 |

Ratios preserve signed increments. M0 compact spatial gain is about 0.496–0.499; feedback recovers most compact response by T5. Broad outer response remains consequential relative to its injected morphology (13.8% outer-aperture deficit), despite nearly unit RTC projected gain. This is downstream response limitation, not evidence that step decisions suppress the source. RTC-only fits are projected differences added to fixed M0, not independent RTC maps and receive no T5 uncertainty interpretation.

T5 compact peak discrepancies are −0.442/+0.470 mJy/beam against own conditional peak sigma 5.307; centroid discrepancies are ≤0.0021 arcsec against 0.0395/0.0337 arcsec scales. Broad template discrepancy −1.104 is 0.23 of the 4.797 conditional scale; outer-aperture discrepancy −996.2 is 0.73 of 1,367.6. These are scale comparisons, not paired significance. Raw-aperture and fitted-background terms remain separate in `effects.json` and full response records.

Do not overlook a fit-shape qualification: compact-minus reaches the existing angle bound near −π while still passing the normal guard. Its sorted major/minor widths differ from ideal by −0.447/+0.360 arcsec, unlike plus (0.0012/0.0039 arcsec) and broad (−0.0264/−0.0124 arcsec). No rescue fit or new trajectory was run. Ellipse components and exact parameters are retained; this limits beam-shape interpretation and is not a measured step-selection effect. Reference 152420 T5 peak is 641.863 mJy/beam, widths 10.767/9.362 arcsec, background robust scale 11.268 mJy/beam. Baseline background scale is 15.473; this is not a same-data treatment comparison.

## Decision and completion discipline

1. **Richer steps: observed.** More admitted groups, affected detectors, repeated temporal structure, and original persistent examples substantiate the selection; real gaps and replacement remain unexercised observationally.
2. **Misses/confusion: bounded evidence.** Two persistent x examples pass RTC; two smaller r windows share one possible missed epoch. Source-like/recovery cases remain distinct from plausible steps. Completeness is not measured.
3. **Rejection cost: measured, excess not established.** Whole-scan/run treatment dominates physical transition loss. No counterfactual supports reclaiming all associated exposure.
4. **Apply: effective at the measured support scale.** Exact planned/realized footprints, no donor crossing; no independent zero-leakage claim in fully excluded windows.
5. **Response: compact flux effective at this scale.** Active decisions/support are stable; broad downstream morphology and the minus-fit orientation boundary remain explicit limitations.
6. **Single next issue: Learn.** A subsequent owner-authorized effort should explain coverage of the retained finite-transition examples before proposing a detector/threshold change. Neither this finding nor final-map quality authorizes changing Consider or Apply now.

Science source `43b510b3d01323c2de9cc41b2dfdb8d316b9ca4d`, tree `7394c516a91913cbfc9b028a55aa2b0a246f074b`, executable SHA256 `e8ecd34c74253fd7cbfd28c5c0bd77a27bd81fab744d16d13d0d586bd6ed56c8`. Diagnostic-only B replay is exact against the sealed prior numerical/discrete baseline. Five trajectories, 25 feedback applications, five parents/endpoints, no extra RTC-only capture, no replacement. Ten actual network Apply starts/completions; plan identity 1001 is not an attempt count. M0 is cold; rank-10 warm ALS fully re-solves each residual; T5 consumes m4, m5 is unapplied.

Two network workers, one numerical thread, one independent workload. Command exits span 73.801–75.899 s; RTC wall occupancy bounds are 10.906–10.986 s baseline and 9.993–12.923 s selected REF, with process-CPU overlap bounds and detailed preparation/Learn/Consider/Apply/serialization intervals saved. Diagnostic capture costs are included. No median, speedup or Unity claim. Inherited launch prose incorrectly says POINT152430 for selected runs; authoritative prospective/input/native identities bind 152420. The original launch record is preserved and this stale descriptive label is not used for routing.

Local gates: 21 native jump fixtures, 207 baseline-tool tests, 36 Python tests, full config preflight, and 152/152 CTest. The first CTest attempt had five unbuilt targets; the separate `CTEST_COMPLETE.txt` closes that gap. Every scientific log has zero unexpected native error-level messages. Each run emitted three known NumPy contraction warnings; independent non-BLAS contraction reproduces finite saved operators and all endpoint sigmas exactly (zero difference). Local toolchain evidence remains supplemental.

Compact preservation, isolated saved-only regeneration, independent final review and guarded enumerated cleanup are recorded separately. B remains the development reference; fixed11 and the original cold-rank10/T10 profile remain parked/preserved. No push, merge, production-default change, tuning or automatic next campaign.
