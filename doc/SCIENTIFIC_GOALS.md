# Scientific goals

Source: Grant Wilson's owner statements of **September 17, 2026**,
“Establish scientific goals and use them to decide what is good enough.”
These goals guide development judgment; they are not new requirements,
automatic thresholds, or replacements for the scientific contracts.
**Timely delivery of useful products to project PIs is part of scientific
utility.** Performance worse than an aspiration may be worthwhile when it
materially advances that delivery; no numerical delivery deadline is set.

| Purpose | Scientific objective | Working goal or reference | Interpretation |
|---|---|---|---|
| SCIENCE — calibration | Deliver usefully calibrated astronomical measurements. | **Owner aspiration:** approximately 5% calibration error would be excellent. | End-to-end goal, not a universal rejection limit or an RTC allowance. |
| SCIENCE — PSF fidelity | Preserve the astronomical response for useful interpretation. | **Owner aspiration:** approximately 5% final PSF broadening or distortion would be excellent. | Name the measured quantity: width, shape residual, peak loss and timestream waveform error are different. Keep compact and extended-source evidence distinct. |
| POINT | Recover displacement of a bright, approximately central source; retain useful telescope/observing-condition checks. | **Recovered owner decisions:** preserve the working per-array fit; amplitude and shape support quality checks. **Numerical accuracy target and adopted stopping conditions: not recovered.** | Preserve POINT's own scope and priorities; do not replace them with SCIENCE or BEAM amplitude goals. See recovery notes below. |
| OOF | Measure actual PSFs at each defocus, preserving the structure used to infer dish deformations. | **Owner-reported empirical reference:** current shape modeling fits with approximately 2–3% residual error. | Not a processing-distortion allowance or dish-deformation accuracy. Preserve real asymmetry and aberration; a more ideal-looking image or smaller fit residual need not be more faithful. |
| BEAM — amplitude | Measure beam amplitudes with useful precision. | **Tentative owner suggestion:** approximately 5% amplitude precision (“5%?”). | Precision, bias and absolute calibration remain separate; this is not a mandatory threshold. |
| BEAM — PSF | Measure the actual beam response with useful shape fidelity. | **Owner aspiration:** approximately 5% PSF measurement accuracy would be excellent. | Use existing shape measures and state what each measures; one successful width or peak does not establish the whole PSF. |

The POINT [owner scope](scientific_contracts/packages/SCI-POINT/v0.1/SCIENTIFIC_OWNER_STAGE_A_DIRECTION_2026-09-02.md)
and [decisions 001, 004, 007–009](scientific_contracts/packages/SCI-POINT/v0.1/SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md)
preserve per-array displacement, the established six-parameter fit, honest
formal-error limitations, and amplitude/shape as supporting quality checks.
The [historical stopping study](FRUIT_LOOP_CALIBRATION_REFERENCE_INVESTIGATION_2026-07-26.md#candidate-stopping-policy)
explicitly calls 0.1″ a test value, not an adopted requirement. Searches of
owner decisions, POINT development/handoff records and related operational
records did not recover an adopted accuracy number or practical stopping rule.
Those specific gaps do not block compact-source RTC development.

Use end-to-end goals as context without inventing per-stage budgets. Report
each percentage's quantity, reference and normalization; prefer existing
metrics and distinguish direct measurements from proxies. Compare OOF
processing and modeling errors only with compatible measures. A value below
5% is not evidence of adequacy when it describes another quantity. Conversely,
an incomplete end-to-end budget need not prevent a proportionate development
decision. Retained sample counts are not measured sensitivity or statistical
information. Processing runtime and development still needed for PI delivery
are separate costs.

For each substantive RTC increment, name the relevant goal and the useful
capability or important risk it addresses. Before another experiment ask:
**What goal is threatened or capability blocked? Could the result change our
choice or readiness to move on? What is the smallest work that resolves it?**
Without concrete answers, stop that line. Conclude plainly: good enough for
the stated use; one specific additional check is justified; or defer because
the likely benefit does not justify the cost. These are development judgments,
not runtime states. Prefer representative evidence; reopen optimization for a
goal-relevant problem, changed use or credible meaningful gain, not uncertainty
alone.

For the [reviewed compact-source RTC case](../handoff/TIMESTREAM_SUCCESSOR_RTC_TREATMENT_OUTCOME_001_2026-09-17.md#purpose-consequence-completion),
**retain the selected notch and stop optimizing it for this tested case**.
Worst controlled additional template-amplitude error fell from 2.304% to
0.052%; relative timestream waveform RMS error rose from 0.059% to 0.095%,
normalized by the injected waveform's RMS on the common crossing support.
Eligible paired-output retention fell by 1.155 percentage points. The
source-error measurements have one supported detector crossing; the support
count covers the twelve-detector cohort. These quantities are not total
calibration accuracy, final PSF accuracy or sensitivity. The measured trade supports proceeding in this
bounded compact-source domain; it identifies no goal-relevant reason for
another notch sweep. Missing fast/extended and POINT/OOF/BEAM evidence remains
a limitation, not a prerequisite for this next development step.

The next practical priority is a complete RTC-only observation result using
this existing treatment. The [development status](REFACTOR_STATUS.md#scientific-goals-and-rtc-stopping-judgment--2026-09-17)
records the single proposed connection. Further numerical limits or broader
scientific qualification are not prerequisites for that bounded development
step; production authorization and existing data-integrity obligations remain
unchanged.
