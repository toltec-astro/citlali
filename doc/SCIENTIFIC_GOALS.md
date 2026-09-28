# Scientific goals

## 2026-09-27 — bounded POINT step assessment completed

The [assessment](../validation/rtc_step_point_2026-09-27/ASSESSMENT.md) at science
source `43b510b3d` completes five trajectories / 25 feedback applications, no
extra RTC-only captures or replacement. Frozen original-domain selection152420
exercises 89 admitted detector-event groups versus152430's9. B replay is exact;
planned/realized filter support agrees, with no donor crossing. Broader scan/run
exclusions carry measured cost; their avoidable fraction is not established.
Two retained unflagged x-transition examples survive RTC/CAL and are largely
removed by PTC. Best-supported next issue is Learn coverage, pending a separate
owner decision; no tuning or follow-on is authorized here. Active injections
change no event decisions or support; compact T5 spatial gain0.990–0.993, broad
0.873. Compact-minus fit-angle-bound width discrepancy remains explicit.
B stays development reference, fixed11 stays parked. Compact preservation,
saved-only isolation, independent final review and guarded cleanup govern the
local closeout. See [handoff](../handoff/RTC_STEP_POINT_2026-09-27.md).


## Fixed-notch final-map tradeoff measured — 2026-09-27

The [fixed11 assessment](../validation/rtc_fixed11_2026-09-27/ASSESSMENT.md)
completes the explicitly authorized applicability and fixed-design comparison.
Treating 331/486 and 151/491 detectors suppresses the target line, but final
conditional peak uncertainty rises 3.32%, MAP contributions fall 3.1884%, and
median exit time rises 12.39%. Compact response stays near 0.992; broad aperture
response slightly worsens. Retain B development reference; do not adopt C-fixed11
for this endpoint. This resolves the prior admission decision and closes the test.
No new threshold/design search, PTC/FRUIT optimization, or automatic follow-on.

## RTC cost/source response measured — 2026-09-27

The [RTC assessment](../validation/rtc_cost_response_2026-09-27/ASSESSMENT.md)
supersedes the earlier RTC deferral for this bounded POINT observation. Keep the
owner-selected warm rank-10/five-application endpoint for development. RTC earns
its measured antialias work; the 11 Hz residual remains consequential. Equivalent
serialization reduces CPU work, with no demonstrated whole-reduction speedup.
Compact transfer loss is small; broader final aperture response remains about
84%, with morphology and fitted-background terms retained. Conditional scales
are not paired significance, and the approximately 5% final science aspiration
is not a per-RTC threshold. Transient completeness and donor efficacy remain
unestablished. The subsequent fixed11 work above resolved that historical next
action; no further PTC/FRUIT optimization or science run starts.

## POINT latency/PCA comparison measured — 2026-09-27

The [bounded comparison](../validation/point_speed_pca_2026-09-27/ASSESSMENT.md) keeps
current rank10 as reference. Warm ALS strongly reduces native fit work, but the
unchanged campaign-wide storage scans confound later raw timing. Covariance has
mixed compact scales and changed extended measurements. No rank15 method extension
was selected. N5 is a useful observed checkpoint, not an automatic stopping rule.
Resolve accounting/provenance overhead before a fresh timing comparison; RTC and
independent-observation science remain unexecuted.

## Bounded PTC rank comparison completed — 2026-09-26

The [rank-response/noise assessment](../validation/ptc_rank_response_2026-09-26/ASSESSMENT.md)
uses matched compact response and conditional measurement uncertainty to expose
the tradeoff: rank15 improves compact scales but loses more broad shoulder response.
Keep rank10 as reference pending one independent-observation confirmation. This
completes the previously deferred 5/10/15 study only; no RTC or method expansion.


## Bright-POINT downstream reference frozen — 2026-09-26

FRUIT method testing is parked. The next lane is RTC/PTC evaluation through the
ordinary MAP + BASE-POINT endpoint, using the single [pipeline quickstart](BRIGHT_POINT_PIPELINE.md)
and [handoff](../handoff/RTC_PTC_MAP_FRUIT_BASELINE_2026-09-26.md). The isolated
original-input POINT152430/a2000 regression completed M0 and ten applications on
its first attempt at b47c13e1f5b745b94880218e347f77b4142686ff. Every compared MAP and
candidate plane, measurement, initial noise field and conditional endpoint record
is exactly equal to the retained reference; no tolerance relaxation. Temporary
native arrays were removed after compact preservation. This is a bright-POINT
development reference, not production/all-mode qualification.

Fresh observations/upstream settings initialize their own M0/C0/BASE quantities;
PTC relearns per network/processing segment on every residual. Ten cycles, measured
brightness, unity gain, full replacement and unapplied m10 remain fixed. The
existing high-coverage continuation is informational, with genuine guards intact.
CORR/DIAG/JINC/CUT3/seed/extra-cycle studies, the proposed mode-count comparison,
and a pre-RTC injection boundary are deferred. No upstream experiment starts in
this closeout. Older next-step directions below are historical where superseded.


## Measurement importance and retention — owner direction 2026-09-26

Future experiment proposals must lead with the affected measurement and units,
the best supported matching noise/uncertainty scale, the expected practical
consequence, and the decision that the experiment could change. State source
versus injected-increment denominators. Label endpoint, conditional-model,
historical-context and unavailable scales separately. Ratios describe practical
size, not significance of paired reductions that share observed noise.

Park effects comfortably below the relevant scale and scientific goals when no
consequential instability or correctness issue is demonstrated. Retain material
coherent response questions; report unmatched scales as unresolved without
automatically starting another experiment. There is no universal sub-mJy/sigma
cutoff, combined score, or new numerical replay tolerance. The approximately 5%
calibration/PSF aspirations remain context, not partial-boundary acceptance tests.

Compact scientific evidence and exact reproduction records are durable. Large
intermediate arrays are temporary unless tied to a named near-term use or a
specific recovery limitation. Regeneration/restoration is an explicit operation,
never a side effect of reading a report. Byte-verified retained counterparts may
replace duplicate payloads through a reviewed manifest; original seals remain
historical and compact representations receive their own index. Protected
observations, indispensable inputs and inaccessible/unique evidence are retained.
Do not keep a large array indefinitely solely because it might be useful later.

The September 26 POINT importance/compaction task parks the unexecuted three-pixel
counterfactual and proposed second-observation comparison. Its closeout records
the actual retained footprint and concrete producer-specific recovery exceptions.

The later September 26 whole-local-data purge directive supersedes the preceding
retention exceptions specifically for identified inactive Citlali RTC/PTC streams:
those outputs and solely associated axes/masks have been deliberately removed,
including unique processing parents. Compact evidence and upstream inputs remain;
reprocessing is explicit, and instant restart is no longer promised. This bounded
execution does not establish an automatic future deletion policy or disk quota.
See `validation/local_data_rtc_ptc_purge_2026-09-26/REPORT.md`.

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
