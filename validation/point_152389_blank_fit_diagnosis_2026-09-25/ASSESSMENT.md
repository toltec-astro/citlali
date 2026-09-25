# POINT 152389: bounded blank-fit diagnosis

**The sampled failures exhaust the function-evaluation budget while fitting narrow noise peaks.** All nine exact-setting failed replays return optimizer status0, “The maximum number of function evaluations is exceeded,” at nfev=3,000. All 1,257 fit pixels, terminal parameters and residual evaluations are finite. There is no sampled NaN, missing-coverage or solver-exception failure.

All 18 archived outcomes reproduce exactly. The three isolated 6,000-evaluation probes still fail. They use the original start, data, objective, bounds and tolerances; only the budget differs. These are diagnostic fits, not replacement calibration outcomes.

| Array | Probe trial | Amplitude at 3000 | Amplitude at 6000 | Relative cost decrease | Minor FWHM at 3000 /6000 (arcsec) |
|---|---:|---:|---:|---:|---:|
| a1100 | 8 | 77.80856 | 103.33954 | 1.3245e-06 | 1.3297 / 1.2693 |
| a1400 | 11 | 48.69658 | 48.95675 | 1.5182e-08 | 1.4640 / 1.4603 |
| a2000 | 3 | 104.35332 | 131.92623 | 0.00011351 | 1.3790 / 1.2794 |

Amplitudes are mJy/nominal beam. Relative cost decrease is `(cost3000-cost6000)/cost3000`, not a fractional amplitude error or detection statistic.

The nine failed fits have minor FWHM below the 2-arcsecond pixel spacing. Several approach the inherited sigma lower bound of 0.5 arcsec (FWHM 1.1774 arcsec). The two larger amplitude drifts accompany narrowing widths and little objective improvement. This is consistent with weakly determined subpixel noise peaks and slow progress along coupled amplitude/shape directions. It is a diagnosis of the selected cases, not proof of a unique cause for every archived failure.

The paired core controls converge in 15/17/15 evaluations for a1100/a1400/a2000. The successful blank controls converge in 30–562 evaluations but can also fit narrow or bound-limited noise structures. In particular, one successful a1100 control sits extremely near the width bound without an active-bound flag. Neither numerical success nor that flag is a scientific source-presence test.

## Why blank scenes enter the positive-source fitter

The inherited initializer searches for the largest pixel within 20 arcsec, subtracts the median within 40 arcsec, then fits a positive elliptical Gaussian plus constant. Its pre-fit absence check is only whether that maximum exceeds the median. This is not a noise-aware association or significance test. No archived blank outcome satisfies that absence check; noise peaks enter the seven-parameter optimizer. The strict 1e-12 tolerances and broad shape freedom are inherited descriptive-fit choices. Their individual causal contributions have not been isolated, and this task does not select replacements.

## Scope and preserved failures

| Array | Calibration blank failures /1024 | Validation blank failures /1024 | Stress blank failures /256 |
|---|---:|---:|---:|
| a1100 | 125 | 126 | 38 |
| a1400 | 165 | 173 | 50 |
| a2000 | 233 | 226 | 53 |

The full archived census reports only the generic nonconvergence cause for failures; optimizer termination details were not originally saved. Evaluation-limit exhaustion is established for these nine replayed failures, not retrospectively for all failures. The sample is deterministic (first three failed calibration IDs per array), small and not randomly representative.

No NaN/input repair, automatic source-absence relabeling, larger-budget adoption, threshold from surviving draws, source-model change or feedback rerun is justified by this diagnosis. All six launch gates remain unavailable. No source fitter, admission rule, map, original trial record or sealed parent was changed.

**Recommended next action:** separately define and review the source-association/no-candidate behavior for these known-bright-source null controls, keeping optimizer failure distinct from absence. Preserve detailed optimizer diagnostics in any later implementation. A blanket budget increase is not supported by the three probes. No rule or repair is selected here.

Runtime: 14.547s for 18 replays and 3 budget probes; zero new independent noise fields, zero upstream reductions. Ten focused correlation-admission tests pass in the existing local Python environment. This does not claim Unity/Spack validation.

![Fixed diagnostic traces](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-blank-fit-diagnosis-20260925/report/blank-fit-traces.png)

[Every recorded case and optimizer result](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-blank-fit-diagnosis-20260925/CASES.json)
[Full archived outcome census](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-blank-fit-diagnosis-20260925/CENSUS.json)
[Fixed selection, input hashes and environment](/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-blank-fit-diagnosis-20260925/PLAN.json)
