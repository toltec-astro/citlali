# POINT 152430 a2000 — completed replacement comparison

**The shoulder suppression recurs, but the substantial supplement benefit seen on 152389 does not.** CORR added no pixels and is exactly identical to BASE-POINT in every map plane and candidate at all 55 paired branch/stage comparisons. DIAG provides a small shoulder improvement, about 0.22% of the whole-source aperture and 0.064 conditional aperture sigmas. This acquisition does not demonstrate a scientifically useful advantage from the fixed supplements, or added value from full covariance over DIAG. This is a result for the frozen methods and specified test signals, not a general proof about covariance-aware admission.

All 15 trajectories completed M0–T10; no source rejection, failed native call, processing error, boundary stop or retry occurred. High coverage is now informational while low-coverage, numerical and genuine support/boundary checks remain. The freshly learned M0 training range was 114–456, with three otherwise valid pixels reaching 493. Their unchanged high-tail scales were 15.3470 (C0) and 15.4390 (BASE) mJy/nominal beam: finite, positive and continuous at the training maximum. No threshold or numerical noise prediction was retuned.

| Policy | REF | CORE-PLUS | CORE-MINUS | SHOULDER | SEED-SHOULDER |
|---|---|---|---|---|---|
| BASE-POINT | T10 complete | T10 complete | T10 complete | T10 complete | T10 complete |
| CORR | T10 complete | T10 complete | T10 complete | T10 complete | T10 complete |
| DIAG | T10 complete | T10 complete | T10 complete | T10 complete | T10 complete |

Each cell includes M0 and ten feedback applications; m10 is saved and unapplied. Detailed M0/T1/T3/T5/T10 rows and all seven fit parameters, gauge-safe shapes, templates, apertures and admission changes are in [MEASUREMENTS.md](compact/report/MEASUREMENTS.md) and [histories.json](compact/report/histories.json).

## Leading measurement effects

These are signed actual-minus-ideal T10 effects using **each policy's own REF and ideal-addition control**. Peak/template units are mJy/nominal beam; aperture units are mJy/nominal beam × arcsec², not integrated flux. Percentages use the corresponding full REF source measurement. Conditional sigmas come from the matching policy's single REF T10 covariance companion, including refitted constant nuisance. Effect/sigma is a practical scale comparison, never calibrated significance of paired differences.

| Policy / shoulder measurement | Signed mismatch | % whole source | Conditional sigma | Effect / sigma |
|---|---:|---:|---:|---:|
| BASE-POINT and CORR / peak | +0.53231 | +0.07092% | 6.17089 | +0.08626 |
| BASE-POINT and CORR / fixed template | −2.73503 | −0.34076% | 7.88152 | −0.34702 |
| BASE-POINT and CORR / aperture 2 | −2017.792 | −2.54980% | 2731.237 | −0.73878 |
| DIAG / peak | +0.51177 | +0.06820% | 6.16669 | +0.08299 |
| DIAG / fixed template | −2.39113 | −0.29778% | 7.88023 | −0.30343 |
| DIAG / aperture 2 | −1842.174 | −2.32193% | 2730.665 | −0.67462 |

The DIAG shoulder aperture improvement is +175.618 brightness-area units relative to BASE's mismatch. The broad negative residual remains coherent in all policies. In the 40-arcsec source disk, shoulder gain/distortion (g, eta) is (0.84456, 0.22107) for BASE/CORR and (0.85443, 0.21182) for DIAG. These describe response relative to the artificial increment; 15.5% gain loss is **not** 15.5% loss of the whole source.

Fitted-background absorption matters. For BASE/CORR the shoulder raw aperture increment is 11287.282 versus ideal 14462.106, a −3174.824 difference. Its fitted-background contribution differs by −1157.032, leaving the reported background-subtracted mismatch −2017.792. DIAG's corresponding raw and background differences are −2931.713 and −1089.539, leaving −1842.174. The aperture discrepancy therefore cannot be read solely from a pixel maximum or fitted peak.

Compact response remains close to ideal. BASE/CORR ± core aperture mismatches are −263.762/+67.094 (−0.3333%/+0.0848% of source; −0.0966/+0.0246 conditional sigmas). DIAG gives −138.851/−87.946 (−0.1750%/−0.1108%; −0.0508/−0.0322 sigmas). Source-disk g for plus/minus is 0.99075/0.99910 for BASE/CORR and 0.99503/1.00482 for DIAG; eta is 0.02292/0.02439 versus 0.02490/0.03854. DIAG slightly changes the compact distortion, but these measurement effects do not reproduce a consequential compact-response tradeoff. Signed plus/minus increment sums are retained separately; aperture sums −196.669 (BASE/CORR) and −226.811 (DIAG) expose modest nonlinearity without treating it as a new experiment trigger.

Seed sensitivity is small beside data-injected shoulder loss. BASE/CORR seed-minus-REF aperture is +0.0615; DIAG is +36.1009, only 0.04550% of source and 0.01322 conditional sigmas. DIAG seed peak changes −0.07535 mJy/nominal beam (−0.01222 sigmas). Across all branches the largest absolute T9→T10 peak change is 0.08474 mJy/nominal beam and largest aperture drift 68.189, about 0.014 and 0.025 matching conditional scales. Park these isolated seed/drift effects; this is not a claim of asymptotic convergence.

REF peaks at T10 are 750.60360 for BASE/CORR and 750.39240 for DIAG. The endpoint pixel-noise scale over the source patch is roughly 14.98–20.08 mJy/nominal beam; it is distinct from the fitted-peak sigma ≈6.17, fixed-template sigma ≈7.88 and aperture-2 sigma ≈2731. High-coverage transfer remains model-based. Fixed-threshold portability has not calibrated a false-admission rate on this acquisition, and this post-RTC/CAL experiment does not validate optical/absolute flux response.

The retained 152389 assessment reported a2000 shoulder aperture mismatch changing from −2.3021% of source for BASE to +0.1182%/+0.1189% for CORR/DIAG. Here it remains −2.5498%/−2.5498%/−2.3219%. Its strong shoulder benefit therefore does not carry over unchanged. No old uncertainty field or geometry was imported; the older aperture noise scale was historical context and is not used as this run's error bar.

## Figures, checks and resources

- [Matched response residuals](compact/report/response-residuals.png): common color scale for all policies/signals.
- [Matched endpoints and seed differences](compact/report/matched-endpoints.png).
- [Admission masks](compact/report/admission-masks.png): high-coverage flags remain separate from eligibility.
- [REF histories](compact/report/reference-histories.png).

Exactly **150 feedback attempts**, four physical inputs, nine ideal T10 controls and three endpoint covariance analyses; 155 native invocations including upstream, ordinary M0 and three injected M0 calls. **Zero input transfers.** Maximum subtract/restore error 6.37e−12, remapped operand error 6.60e−12 mJy/nominal beam; native injection/control discrepancy zero. All native return codes are zero and error-level logs are absent. BASE endpoint arithmetic warnings were checked against saved operators: independent elementwise pseudoinverses match exactly for all policies, Jacobian identity residual ≤6.74e−14, and recomputed saved sigmas match exactly. No extra fit or covariance learning was used for that check.

Science stage wall times total **391.081 s (6.52 min)**; summed native invocation wall time is 1499.351 s because branches run concurrently. BASE measured peak process-family RSS 0.889 GB; the twelve-worker response phase peaked at 7.816 GB sampled RSS. Initial native processing used two network workers; numerical libraries used one thread each. Largest observed disk stage snapshot was **2.901 GB**, below the enforced **16.159 GB** allocation ceiling; this is a sampled stage size, not a continuous filesystem high-water mark. The first saved-product report took 2.769 s excluding module/font-cache initialization. Cleanup removed 1,202,019,404 native numeric bytes (16,026 files). Lossless compression reduced 806,924,160 FITS bytes to 165,762,798, and 250,316,063 large metadata bytes to 9,790,600. The tracked compact product copy contains 234,820,215 bytes; both explicit product copies total 469,640,430 bytes. Inputs are unchanged. CLOSEOUT.json records exact scope.

The high-coverage/M0-control regression suite, independent 52-test review, native 31-test gate and full configuration preflight passed. This existing local build is supplemental evidence, not Spack-backed V2 validation. Exact source/binary/input/configuration and reproduction commands are retained. Superseded stopped-attempt products stay deleted; Git history was not rewritten. No push, merge, production/default promotion, threshold change, extra cycle or follow-on experiment.

**Next action: retain this as a negative portability result and leave method/default decisions parked pending owner review.** No further numerical work is started.
