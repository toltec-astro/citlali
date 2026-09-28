# Codex directive: scientific consequences of RTC step decisions
## POINT 152430 — readout networks 1–4

**Owner:** Grant Wilson  
**Date:** September 28, 2026  
**Status:** Execute the bounded baseline assessment below, independently review it, preserve compact evidence, commit locally, and stop. Do not return only a proposal.  
**Scientific target:** POINT **152430**, specifically **networks 1, 2, 3, and 4**. Recover the exact acquisition identity; the earlier work used **152430/0/2**. The owner identifies these networks as especially step-rich. Verify and describe their behavior; do not substitute another observation or select only the easiest network.

> **What evidence is present, why does the pipeline act—or not act—on it, and what are the consequences for astronomical response, uncertainty, usable exposure, and processing cost?**
>
> The objective is scientifically useful data, not a more complete event catalogue or a cosmetically smooth timestream. Apply this standard to both treatment and non-treatment. Correct execution is necessary, but does not prove a scientifically appropriate decision. A good final map does not certify every upstream operation; a visible residual does not automatically justify another operation.

This principle applies across the pipeline. **This execution milestone remains a step-consequence assessment**, not a simultaneous reopening of spikes, spectral lines, atmospheric removal, PCA, mapmaking, or FRUIT. Learn coverage is one possible issue, not the predetermined answer. No new scientific treatment is authorized here. Conclude with at most one evidence-supported comparison for an owner-approved next milestone, or explain why no change is justified.

## 1. Recover the completed baseline, not an obsolete campaign state

Start from the completed POINT step closeout, reported local commit **`b33787d1c`**, after owner-approved scratch cleanup. Verify actual checkout, ancestry, build/source identity, branch and unrelated changes. Discovery anchors, not reset commands:

- `validation/rtc_step_point_2026-09-27/ASSESSMENT.md`, retained evidence, review and closeout; `handoff/RTC_STEP_POINT_2026-09-27.md`.
- `validation/rtc_fixed11_2026-09-27/` and `validation/rtc_cost_response_2026-09-27/`, only for inherited B definitions and relevant saved evidence.
- `doc/REFACTOR_STATUS.md`, `doc/SCIENTIFIC_GOALS.md`, `doc/BRIGHT_POINT_PIPELINE.md`, repository instructions and accepted RTC/ALIGN/AST/CAL/PTC/MAP/VAL contracts and amendments.
- The previous `CODEX_RTC_STEP_LEARN_CONSIDER_APPLY_POINT_VALIDATION.md` for retained definitions, not renewed execution allowances.

The reported worktree is `~/.codex/worktrees/ptc-rank-response/citlali-refactor`. Preserve the original user checkout, unrelated work, prior compact evidence and original cold-rank10/T10 tag/profile. Use existing `tools/timestream_successor/point_pipeline.py` and native scientific owners; extend existing diagnostic/reporting code rather than creating another pipeline or generic framework.

Earlier findings on networks 11/12 and POINT 152420 are context, not the treatment control for networks 1–4. Read them from compact evidence. Do not rerun those campaigns or treat their absolute fluxes, noise or event counts as directly comparable treatment outcomes.

## 2. Bind networks, array identity and original dependencies

Inspect all four networks before drawing a population conclusion. Establish their actual array membership from the acquisition/APT metadata. **Do not inherit the old `a2000` label or its numerical beam, calibration, source raster or noise field merely because the obsnum is unchanged.** Verify that the existing route supports the selected network subset and truthful array-specific outputs; a partial-array map must be labeled as such.

Inventory complete original readout parts, telescope/auxiliary files, paired x/r availability, native timing/validity, exact tune associations, APT and calibration identities. Record paths, sizes and readiness. Use original-derived inputs, not saved RTC/PTC streams, as scientific parents. Keep every acquisition label, launch scope, receipt and report consistent with the actual obsnum and networks.

Local execution is the default. Targeted, read-only recovery from Unity of genuinely missing original dependencies for **152430/networks 1–4 only** is permitted through existing configured access. Establish exact paths, sizes and bindings before transfer. No bulk survey, other observation, remote computation, archive modification, credential changes or retrieval of old processed streams. A missing dependency blocks only the dependent work; do not silently omit a network and describe the remainder as a complete four-network assessment.

Use the appropriate existing array profile under the inherited B method. Freeze and report numerical configuration before scientific execution. Do not tune it to this dataset. If the four networks require multiple array chains, account for each explicitly; the execution cap in Section 8 is not permission to multiply the campaign. Report an unsupported configuration rather than silently broadening scope or inventing calibration.

## 3. Preserve the scientific method and active decision-making

Keep **B** as the development reference; **C-fixed11 remains parked for this POINT round**. Preserve raw Δf/f RTC units, CAL-after-RTC ordering, independent coordinate validity, paired x/r policy, antialiasing, cadence, physical gaps, speed exclusions, segmentation, guards and donor rules. Recover the current implemented/approved treatment; do not reconcile historical descriptions by silently changing it.

Keep the downstream endpoint: **warm-initialized, fully re-solved observed-entry ALS, rank 10; M0 plus exactly five outer FRUIT feedback applications**. M0 starts cold; later PTC fits recompute centering and fully solve their current residual. Warm state cannot cross branches, networks or physical segments. **T5 consumes m4; m5 remains unapplied.** RTC runs once per trajectory; feedback uses that trajectory's immutable post-RTC/CAL parent. Record internal reassessment/application attempts separately.

Preserve ordinary MAP, the established grid/projection, admission/feedback and source-fit/guard policies, weights, numerical tolerances and uncertainty procedure. Bind their actual array-appropriate settings. Retain the inherited 2-arcsec grid, 40-arcsec feedback radius and threshold-3 source-connected BASE policy unless the existing approved array profile explicitly supplies a different setting; report such a profile difference, not a new optimization. Preserve measured-map feedback brightness, unity gain, complete model replacement and high-coverage continuation.

Allow the unchanged method to learn and decide normally on each input. Do not transplant an old event list, PCA basis, segmentation, map or noise realization. No step-off observational arm, hand-edited event catalogue, threshold/guard sweep, plateau stitching, coordinate rotation, new donor method, rank change or extra feedback iterations. Instrumentation defects may be repaired and tested; a scientific-behavior defect must be identified separately, not hidden in a diagnostic patch.

## 4. Lead with the population, then explain representative decisions

Reuse the existing event-chain, population-mask and window machinery. Produce per-network detector-versus-time views and counts/rates with explicit eligible detector-time denominators. Separate shared physical disturbances, detector–event instances, affected detectors and repeated events. Preserve x-only, r-only, joint, rejected, unresolved and unassessed cases. An r-only admission is evidence in r, not proof that x is unaffected or that every paired exclusion is necessary.

Establish original-domain behavior independently of production admission counts. Finite transitions require supported pre/post behavior; an isolated spike, source crossing, smooth atmospheric variation or changed common-mode amplitude is not automatically a step. Peer context is supporting evidence, not a requirement that every event be network-wide. Stable stretches need not be flat.

Freeze a bounded review selection from original/evidence-domain properties before inspecting cleaned outcomes: **up to four decision/event examples and four unflagged/control windows per network**, at most 32 windows total. Distribute across physical time and, where present, ordinary/large events, repeated transitions, short plateaus, source-adjacent cases, r-only decisions, broad exclusions and quiet controls. Do not fabricate absent categories. Label unavailable review coverage. Targeted examples do not establish population precision/recall or an independently complete truth catalogue.

For every reviewed decision or plausible omission, trace:

1. **Evidence:** original x/r, local variability, transition duration, pre/post support, source and peer context, and interaction with spike exclusions.
2. **Learn and Consider:** whether a candidate existed, each actual admission/rejection/unresolved reason, selected boundary/guard, segment and paired-coordinate consequences. A missing final event must not automatically be attributed to candidate generation; distinguish never proposed from proposed but rejected.
3. **Apply and downstream:** planned versus realized support, actual retained/excluded samples, donor/filter influence and residual behavior through CAL and PTC.

Preserve physical-time alignment, units, actual cadence and independent validity. Diagnostic centering, filtering or decimation must be explicit and must not hide offsets, transitions or guard leakage. Do not use donor-created values as original measured evidence. Distinguish physical gaps from numerical block boundaries. Correct plan realization and no forbidden donor crossing remain independent correctness checks even when the particular map effect is small.

## 5. Assess consequences in both directions

Organize findings around **what the decision costs or accomplishes**, not which stage produced the largest catalogue. Populate one compact consequence table with: network/event class, evidence, action or omission and reason, affected detector-time/source crossings, RTC residual, downstream residual, response/uncertainty/exposure/cost evidence, attribution limits, and disposition. Allow unavailable entries rather than invented zeros or unsupported causal explanations.

### Untreated or unresolved transitions

At matched physical intervals inspect RTC input/output, CAL/pre-PTC, and PTC residuals at M0 and T5 where supported. Distinguish the residual from sky-restored total. Compare persistent structure and ringing outside exclusions with comparable local noise and quiet controls using existing estimators.

If PTC largely removes a transition, report that attribution. Do not conclude either that RTC was effective or that PTC paid a consequential penalty without evidence. Inspect available residual/noise, source-crossing and map-context diagnostics; a visually apparent upstream transition alone does not prove final harm. Likewise, no obvious map feature is not a demonstrated upper bound on all harm. Classify insufficient attribution as not established.

### Accepted events and exclusions

Separate physical transition loss, additional guards/filter influence, short/unstable segment rejection, scan/run exclusions and donor-supported filtering. Show overlaps and their **union**, plus incremental loss among samples eligible immediately before each rule. Declare any sequential accounting order; incremental allocation is bookkeeping, not a unique causal decomposition.

Track eligible detector-time, RTC-retained science centers, CAL-admitted contributions, actual MAP contributions and weighted coverage/normalization. Separate losses near source crossings from background losses where supported by the unchanged pointing/source definition. Do not add already-invalid samples to the cost of step treatment or count synthesized support as acquired exposure.

Large exclusions deserve explanation, not automatic condemnation. Identify whether retained original data support potentially usable stable stretches, but distinguish **gross excluded exposure**, **plausibly recoverable exposure**, and **demonstrated recoverable scientific information**. This unchanged baseline does not authorize unmasking real observations and cannot establish that all associated loss is avoidable.

### Scientific importance and correctness

Interpret effects against matching estimator uncertainty/local noise and existing final-product goals. Report physical magnitudes and uncertainty-relative context, not only percentages. Do not chase inconsequential sub-noise differences, impose the final approximately 5% goal as a universal per-event threshold, or infer sensitivity from sample counts alone. A baseline assessment without an intervention cannot generally establish the causal benefit of adding/removing treatment; say so precisely.

Contract violations remain defects independently of one small map effect. Valid policy choices are judged by scientific benefit. Classify each finding as: **adequate at the measured scale**, **evidence of a consequential residual**, **potentially costly treatment requiring a comparison**, **correctness defect**, or **not established**. Do not call a large exclusion proven over-rejection without adequate evidence.

## 6. Measure active astronomical response on this same four-network subset

Run an unmodified B reference to T5 and capture all needed stage/population diagnostics during that trajectory. Establish its own source measurements, admission-noise package, final conditional uncertainty and support. A failed ordinary source guard retains diagnostics and established failed-fit/empty accepted-output behavior; do not weaken the guard or switch observations/networks to obtain a favorable response result.

Where source admission and mapping permit, execute the established **compact +10%, compact −10%, and displaced broad +5%-amplitude** branches once each. Construct the deterministic rasters from this selected subset's own unmodified reference using the established geometry/taper recipe and correct array calibration. Bind rasters, normalization, coordinates and hashes before injected executions. Do not transplant the networks 11/12 raster or move an injection to avoid steps.

Inject once into original-derived aligned x, before RTC learning, using validated raw-unit calibration and pointing mapping. Leave input r and original validity unchanged. Quantify mapping-unavailable cells; verify complete mapping of the reference MAP-contribution domain. An unavailable injection is not zero response. This remains an x-only optical perturbation, not validation of real optical leakage into r.

**Keep Learn–Consider–Apply active independently in every branch.** Cross-match by detector identity and physical intervals, retaining ambiguity, new/missing/shifted events and changed support. Do not freeze the unmodified event decisions. A changed decision requires explanation against the original event and injected signal; it is not automatically a false detection.

Compare injected-minus-own-REF at RTC, M0 and T5 with ideal addition on **unmodified reference support**, retaining lost/gained support and estimator unavailability. Never hide losses with an intersection-only domain. Use this subset's own reference M0 admission-noise/guard package and T5 conditional-noise endpoint under the inherited recipe; do not import noise operators from the older subset.

Report spatial gain separately from peak/template/aperture response, centroid, widths and distortion. Retain aperture/background terms and fitted-parameter boundary/degeneracy limitations. Projected RTC differences added to a fixed map are conditional transfer diagnostics, not independently reduced RTC sky maps. Do not apply T5 uncertainty to RTC-only fits or claim paired significance using an ordinary single-map noise scale.

Broad morphology is a regression probe, not permission to restart shoulder reconstruction or FRUIT optimization. Attribute attenuation only as far as the stage evidence supports. Favorable compact response does not validate every step decision; unchanged event/support decisions under active injections are useful evidence of local stability, not universal immunity to sky signals.

## 7. Select the next question only after the evidence

Answer whether the strongest issue is missed evidence, an admission/segmentation choice, overly broad treatment, inadequate containment, downstream handling, or no actionable deficiency at the measured scale. Do not select Learn simply because the earlier campaign found two unflagged transitions.

Where one issue warrants testing, deliver **one proposed matched comparison**, specifying the affected population and physical mechanism; exact single change and unchanged controls; expected benefit; source/exposure/noise/correctness risks; matching uncertainty interpretation; and a small prospective execution budget. Prefer a class of consequential behavior over two hand-selected examples. A candidate must have a plausible path to scientific benefit, not merely more flags or cleaner-looking traces.

**Do not implement or run that treatment comparison in this milestone.** Baseline associations and source-injection results may motivate it but cannot substitute for its counterfactual. “Existing handling appears adequate” and “consequence remains unresolved; this is the minimum discriminating test” are legitimate conclusions. No automatic follow-on or candidate promotion.

## 8. Hard execution and resource bounds

A trajectory means one original-input branch on the selected four networks, followed by its M0–T5 endpoint. Count array-specific chains separately if metadata requires them; the caps apply to the total, not per network or array.

| Work | Maximum M0–T5 trajectories | Outer feedback applications |
|---|---:|---:|
| Unmodified B reference, networks 1–4 | 1 | 5 |
| Compact plus, compact minus, broad response | 3 | 15 |
| **Planned maximum** | **4** | **20** |

No new 152420 or networks 11/12 trajectories. Permit **one additional RTC-only four-network capture total**, only if required diagnostics cannot be obtained from the planned runs and retained evidence. It is not a policy comparison. Permit **one replacement trajectory, at most T5**, only after a documented build/routing/instrumentation defect is repaired, not for disappointing science or failed source admission. Absolute cap: **five full trajectories/25 feedback applications plus one RTC-only capture**. Do not stack budgets from prior directives.

Record actual launches, original-input parents, network RTC Apply starts/completions, existing internal reassessments, array endpoints and feedback applications. Four planned four-network trajectories imply 16 initial network RTC Apply executions; internal attempts must still be counted from actual execution, not plan IDs. Failed launches that enter science consume budget.

Choose a fixed, documented, resource-safe local network-worker/numerical-thread policy before launch. Record actual concurrency, Learn/Consider/Apply/filtering/serialization costs, total RTC/raw-to-result wall time, CPU and peak memory. Include diagnostic costs honestly. No timing-repeat or scaling campaign; a single run is not a speedup measurement. Avoid campaign-growing storage scans during timed execution; report unavoidable overhead without starting a performance redesign.

## 9. Evidence, review and disposable scratch

Use one new campaign directory under `~/work_toltec/local_data/citlali-validation/`, with corresponding compact records/code under `validation/rtc_step_consequences_2026-09-28/`. Establish a numerical peak-disk budget from actual dependencies and allocation before launching. Reuse packed diagnostics and bounded excerpts; avoid per-sample/per-cycle tiny-file proliferation and duplicating prior campaigns. Aim for a few hundred MB of durable evidence; justify any excess.

Run focused existing native/reporting tests and add only missing small fixtures for neutrality, paired evidence, finite transitions/guards, union accounting, physical gaps versus block boundaries, forbidden donor crossing, event matching and active-selection response. Verify instrumentation neutrality with the established numerical comparator and exact discrete identities/support using existing fixtures or a small bounded replay. Any new scientific replay of real observational inputs consumes Section 8 budget; no extra old-observation campaign. A behavior-changing defect is a separate finding, not an excuse for hidden tuning.

Retain enough event/decision records, masks/ranges, stage excerpts, maps, response/noise summaries, actual timings and input/config/source bindings to regenerate the assessment and figures without original-data access, deleted full streams, scientific fitting or a new uncertainty ensemble. Independently review scientific interpretation, execution counts, support/response accounting and compact reproducibility. The review uses saved evidence and small fixtures, not another observational reduction.

Prepare a precise, campaign-only plan for removing inactive native scratch after committed compact preservation, saved-only regeneration and independent review. Use the existing guarded cleanup and required approval mechanism. Protect originals, tune/APT/calibration/team data and prior evidence; **do not reuse the previous campaign's deletion approval as authorization for new paths**. Never bypass an approval block or fabricate review authorization. If direct owner approval is required, request it once with exact scope and preserve the completed report while pending.

Perform exhaustive scratch verification once in the guarded procedure, not repeatedly in independent review. Record actual deletion, protected-input checks, remaining metadata/FITS/logs and post-cleanup regeneration; do not claim cleanup or zero disk use without the receipts. No indispensable code or reproduction record may exist only in disposable `local_data`.

## 10. Final report and stop condition

Lead with the **scientific consequence assessment**, followed by the four-network population view—not commit IDs, successful-run counts or the largest step count. Include the consequence table, representative evidence, source response and uncertainty, exposure accounting, timing and limitations.

Answer directly:

- What did the original data present, and why did each consequential class receive its action or non-action?
- What did RTC contain, what remained for PTC, and what downstream consequence is actually supported?
- Which exclusions are explained, which are potentially costly, and what avoidable loss remains unmeasured?
- Are compact response, source decisions and support stable at the measured scale? What remains limited for broad morphology or individual estimators?
- Is any change justified now? If so, what single comparison would discriminate benefit from cosmetic improvement?

Mark **observed**, **inferred** and **untested** conclusions explicitly. Maintain truthful failures and unavailable quantities. Update status, goals, pipeline documentation and a concise handoff to preserve the consequence-first principle without changing numerical contracts or defaults. Commit locally. **B retained; fixed11 parked; no push, merge, production promotion, treatment revision or automatic next campaign.**
