# Codex directive: fixed 11 Hz notch — population applicability and final-map benefit

**Owner:** Grant Wilson  
**Date:** September 27, 2026  
**Status:** Execution order. Resolve the bounded experimental applicability below, implement and run one fixed-design comparison, assess, independently review, commit locally, and stop. Do not return only another proposal.  
**Scope:** POINT **152430/0/2**, **a2000**, networks **11/12**, with the original inputs and exact dependencies used by the completed RTC audit.

> Determine whether the already selected 1 Hz / 3-second notch removes the measured near-11-Hz contamination at a worthwhile cost in astronomical response, usable data, and runtime. Choose its affected detector/segment population from the unmodified observation before running the candidate. Judge the result through the fixed five-feedback map endpoint, not merely by a cleaner spectrum. This is one experimental comparison, not a new notch-design, detector-classification, or PCA project.

## 1. Recover the completed work and preserve the control

Start from the completed RTC cost/source-response effort, reported local closeout **`ab43474a7`**, not the earlier PCA comparison or original cold-start freeze alone. Recover:

- `validation/rtc_cost_response_2026-09-27/ASSESSMENT.md`, its compact population/spectral/source/timing records, B equivalence evidence, reproduction instructions, final review, and closeout.
- `handoff/RTC_COST_RESPONSE_2026-09-27.md`, current status/goals, `doc/BRIGHT_POINT_PIPELINE.md`, and applicable RTC/ALIGN/AST/CAL/PTC/MAP/VAL contracts and accepted amendments.
- `CODEX_RTC_COST_CONTAMINATION_AND_SOURCE_RESPONSE.md` and the historical characterization of the accepted **1 Hz / 3-second** notch, including its exact numerical design and coefficient-generation recipe.

The reported checkout is `~/.codex/worktrees/ptc-rank-response/citlali-refactor`; verify actual checkout, ancestry, common Git directory, build identity, and unrelated changes. Quoted identities are discovery anchors, not reset instructions. Preserve the user checkout, prior compact evidence, and original cold-rank10/T10 tag/profile. Keep scientific execution identities distinct from later report/closeout commits.

The completed audit reports effective antialiasing, residual near-11-Hz contamination, compact RTC spatial gain near 0.9963, and exact scientific equivalence for implementation candidate B. These are checks against the saved evidence, not substitutes for its full-precision measurements or acceptance tolerances. B's serialization/CPU savings did not establish end-to-end speedup.

Use the established `tools/timestream_successor/point_pipeline.py` route and native scientific owners. Carry forward B's verified implementation and corrected run-local accounting in **both** new arms. No further implementation-speed candidate is authorized.

## 2. Explicit applicability authorization: one fixed experimental population

**This order resolves the earlier permission to leave C unselected solely because the historical notch had not been authorized for this observation's detector population.** Applying that existing design to one evidence-supported population in this observation is now authorized. It is not permission to notch every detector by default or to adopt a production admission law.

The owner delegates the **one-time experimental selection and enumeration** of that population to Codex under the following rules. Do not return for approval of each detector. A new universal significance threshold, residual-line allowance, or automated classifier is not a prerequisite to this bounded test.

### Establish and freeze the set before candidate execution

Use the completed audit's source-excluded, unmodified **x-channel** population diagnostics, network by network, with temporal/physical-segment information intact. Read the full population, not only historically illustrated pathological detectors. Use the existing spectral estimator, continuum definition, uncertainty/quality information, and line/persistence evidence. Do not refine these methods in this effort.

Include every supported detector/physical-segment pair with a clearly identified target-band feature in those diagnostics: a resolved feature near the fixed center with excess over its local continuum, supported by the available time-resolved evidence rather than solely a positive-excess statistic, one isolated noisy bin, or a source crossing. Record the numerical evidence and the reason for each inclusion. Existing audit criteria or classifications may supply this evidence even where they previously did not confer treatment authority.

Pairs with no identified feature stay on the control treatment. Equivocal or diagnostically unsupported pairs also stay on the control treatment and are labeled **unassessed/uncertain**, not line-free. Do not expand a network-wide average or another detector's coherence into automatic treatment of an unassessed detector. r and coherence can corroborate the x evidence; an r line is not required, and r/coherence alone is not sufficient.

This is a documented, observation-specific experimental selection, not a claimed calibrated detection rule. Do not invent a numerical threshold merely to disguise the selection as an already validated policy. Do not choose or revise membership using candidate spectra, injection recovery, map appearance, or uncertainty improvement. If the saved diagnostics genuinely cannot distinguish any affected population, report that exact evidentiary limitation rather than forcing a classifier or treating every detector.

Prefer the existing detector-by-physical-segment plan granularity. Do not introduce new fine temporal subdivisions, operator switching around source crossings, or new segmentation to improve the result. Evidence masks exclude source crossings from **learning**, not from the measured convolution support of otherwise valid samples.

Before the first C run, save and bind a compact **applicability manifest** containing the original observation/tune/APT/detector identities, exact network and physical-domain identifiers, selected and unselected dispositions, evidence locations/values, and the fixed design identity. Report selected, unselected, and uncertain counts with full denominators for each network. An independent preflight checks this set against the saved evidence. Freeze its hash before candidate execution.

**The same notch applicability and design are used in all C repeats and all C injection branches.** This is an explicitly fixed-applicability transfer experiment. It does not measure how a future adaptive notch selector would react to injected sources. All other existing RTC learning, masks, validity decisions, and downstream processing remain active; do not freeze the whole baseline RTC plan or its output support to manufacture agreement.

## 3. Freeze the filter design and retain current scientific semantics

Use the historically selected candidate with numerical center **11.0062558446 Hz**, nominal **1 Hz width**, and **3-second design duration**, as defined by the actual accepted characterization. The long numerical center specifies reproducibility, not that the physical line is measured to that precision.

Recover the exact family, window, depth, width convention, duration/tap convention, phase/delay convention, application order, and coefficient-generation recipe. Generate/bind the realized coefficients at the actual input cadence as required by that recipe. Record the coefficients or hashes and realized combined response with the existing antialias filter. Do not reinterpret “1 Hz” as a different full/half-power width, reinterpret “3 seconds” as padding or a new guard length, copy coefficients across incompatible rates, or substitute an IIR/FIR design with the same label.

No center/width/duration/depth sweep, per-detector recentering, line tracking, coherent subtraction, or second notch is authorized. A mismatched/drifting feature remains a reported limitation, not permission to broaden the design. Do not use the earlier approximately 2.2% worst-tested fast-source peak attenuation as a universal tolerance or as a measurement on this observation.

Reuse the **mathematical filter design**, not obsolete RTC orchestration. Preserve current raw-unit RTC, CAL-after-RTC ordering, x/r policies, transient/shift handling, donor rules, speed exclusions, antialiasing, cadence, physical gaps, reset rules, and support semantics. The notch's required support and edge effects must be represented honestly; do not shorten guards, bridge real gaps, or invent measured exposure to retain coverage.

Apply the shared notch consistently to x and the requested conditioned-r companion under the current paired-coordinate contract, without cross-coordinate mixing. Preserve immutable raw r; do not calibrate r, use r as substitute sky data, or manufacture missing support. The source injections change x only. Source leakage into r and optical/IQ response remain outside this test.

The only new scientific treatment is **the fixed notch on the frozen set**. Its inherent response/support consequences are allowed and must be measured. Necessary native configuration/plan plumbing is authorized; implementing the filter numerics in the campaign driver is not. Missing exact design evidence or an actual numerical/support incompatibility is a truthful blocker for dependent work, not permission to substitute a different filter.

## 4. Controlled arms and fixed downstream endpoint

Name the two arms clearly:

| Arm | RTC treatment | Downstream configuration |
|---|---|---|
| **B-control** | Completed, scientifically equivalent B implementation; existing RTC policy, without the newly added fixed-notch intervention | Warm-initialized, fully re-solved observed-entry ALS; rank 10; M0 plus exactly five FRUIT feedback applications |
| **C-fixed11** | Same implementation and policy, plus the fixed notch on the frozen affected set | Identical configured downstream method |

Keep acquisition **152430/0/2**, its exact **152430/0/1** tune associations, networks 11/12, and the bound APT/calibration/telescope inputs. Verify the existing local manifest; no download or Unity execution is expected or authorized. Do not substitute archived processed streams or a nearby tune.

Keep ordinary MAP, the 2-arcsec grid and containing-pixel projection, 40-arcsec feedback radius, existing seven-parameter Gaussian-plus-constant fit/guard, source-connected threshold-3 BASE admission, measured-map feedback brightness, unity gain, and complete model replacement. Preserve high-coverage continuation. No new source model, weights, calibration, noise estimator, admission law, JINC, rank, solver, or stopping rule.

M0 starts cold. Each subsequent PTC solve recomputes centering and fully solves the current residual with unchanged weights, validity rules, tolerance, and iteration limit. Warm state is confined to the same arm, trajectory, branch, rank, network, and physical segment. **T5 consumes m4; m5 is not applied.** Five means outer feedback applications, not five ALS iterations.

Run RTC once per trajectory. Feedback uses that trajectory's immutable post-RTC/CAL parent, subtracts the current sky model, freshly solves/applies PTC, restores the exact subtracted operand, and maps the total. Different C maps may produce different downstream candidates under the unchanged method; do not transplant B's feedback history.

## 5. Execute only the bounded comparison

Use fresh B controls for contemporaneous timing and regression, while reusing the completed A/B compact injection evidence for the unchanged no-notch science. Do not rerun the entire preceding audit or its no-notch injection trajectories.

| New work | Trajectories | Outer feedback applications |
|---|---:|---:|
| B-control: independent unmodified raw-to-T5 repeats | 3 | 15 |
| C-fixed11: independent unmodified raw-to-T5 repeats | 3 | 15 |
| C-fixed11: compact-plus, compact-minus, and shoulder pre-RTC injections, once each | 3 | 15 |
| **Planned total** | **9** | **45** |

Run unmodified timing trajectories serially in the predeclared order **B1, C1, B2, C2, B3, C3**, with a fixed CPU/thread budget, fresh processes and scientific state, identical output/instrumentation policy, and no concurrent science, build, or report workload. Retain first-use/cache labels; do not discard an inconvenient first run or claim controlled cold caches without evidence. Use B1 and C1 as the predetermined scientific representatives, not the most favorable repeats.

B1 must reproduce the completed B reference through the existing comparator and exact discrete plan/mask/identity checks. Verify the inherited no-notch response bindings through the completed A↔B equivalence evidence. A failed control regression must be explained/corrected; it cannot be absorbed into the notch's scientific difference.

Allow **at most two extra RTC-only diagnostic executions**, only to recover missing stage/applicability or response-accounting evidence from these same inputs when saved records and planned runs cannot supply it. They do not add PTC/FRUIT trajectories or parameter trials. Reuse original-derived parents only for explicitly untimed diagnostics/injection work. Count all parent generation and internal RTC attempts.

Allow **one replacement trajectory, at most T5**, only after a documented implementation/build/routing defect is repaired. Absolute new-work limits are **10 trajectories, 50 feedback applications, and two additional RTC-only executions**. Scientific rejection, disappointing suppression, excessive loss, or slow performance do not earn a replacement. New limits do not reopen unused budgets from earlier campaigns.

## 6. Measure what is removed and what reaches the map

Lead with unmodified data and full populations. Reuse the audit's fixed spectral/continuum recipe and physical frequency bands. Measure, separately by network and relevant segment, input → RTC output → CAL/pre-PTC → PTC residual at M0 and the last feedback pass. Distinguish cleaned residuals from restored totals; restoration of sky is not automatically a return of interference.

For both the treated subset and the entire network, report absolute target-band power, excess over local continuum, residual line-to-continuum measures where supported, prevalence, persistence, and existing coherence summaries. Show typical and upper-tail behavior with denominators, including unselected/uncertain detectors. Report what PTC removes after each RTC treatment and what remains. Do not claim a final-map gain just because RTC removes a line or PTC has less variance to remove.

Match physical intervals, source exclusion, units, cadence, resolution, and valid support. Keep raw and calibrated units explicit. Do not zero-fill invalid samples for naive PSDs, hide loss through an intersection-only comparison, or call a frequency removed merely because it is unavailable on an output grid. A notch-shaped spectral hole alone is not an independently validated residual-contamination allowance.

Use compact support/event diagnostics to report the notch's incremental detector-time loss, guard/boundary influence, donor influence, retained MAP contributions, normalization, and coverage. Keep full B support visible alongside C support before any matched-domain plot. A quieter result obtained primarily by losing useful data is not automatically an improvement. Inspect ringing and structure near existing events/gaps without changing their treatment.

At T5 compare the unmodified source measurements, conditional uncertainties, background residual/correlation summaries, and common-scale map/difference figures already supported by the route. Show physical differences and matching uncertainty scales. These are shared-data method sensitivities, not independent noise realizations or a new absolute calibration truth. Do not start a new map-domain uncertainty or artifact-estimator project.

## 7. Repeat the pre-RTC astronomical-response controls for C

Reuse the exact signed **±10% compact** and **+5%-amplitude displaced broad/shoulder** rasters, including geometry, offset, taper, and full-precision identities. Add each once to original-derived aligned x using the validated raw Δf/f calibration/pointing mapping. Do not replace this with post-RTC injection, invert a filter, refit the source definition, or modify input r/validity.

Retain and quantify the known mapping-unavailable native cells. Verify the mapping of reference MAP contributions for the current arm rather than assuming the previous counts still apply. Unsupported injection mapping is not zero source response and must not quietly remove contributions from an ideal control.

Use C's **own unmodified M0** admission-noise/guard package across C response branches and its own T5 conditional-noise endpoint. Keep measurement geometry common to the comparison. Do not transplant B's numerical noise fields or learn admission noise from injected brightness.

Compare injected-minus-own-REF at RTC output, M0, and T5 against the established ideal-addition construction. Keep each arm's own-reference support convention explicit, and separately expose B→C reference-support losses so that redefining C's ideal cannot conceal notch-caused loss. Never redefine an ideal using newly lost injection-branch support.

Report compact spatial gain/distortion, signed peak/template/aperture response, centroid, major/minor widths, and broader residual morphology. Keep compact signs separate. Retain raw-aperture and fitted-background terms for the shoulder. Compare C to the saved no-notch response with exact control bindings; inherited B response is not newly remeasured B response.

Label projected-difference-plus-fixed-map RTC-only fits as **conditional RTC transfer diagnostics**, not independent RTC maps or T5 uncertainty endpoints. The new notch set is frozen, while all other active processing follows its normal branch-specific path; do not claim an adaptive-selector robustness test.

Assess final response and noise together. Use response-normalized uncertainty only where a scalar compact gain is supported; do not apply a scalar shoulder correction or rescale science maps. Do not demand exact scientific identity from C, but do not excuse a software regression as sub-noise. Earlier final-product goals guide interpretation, not a universal 5% per-RTC pass threshold. Sub-noise differences without consequential bias, instability, or correctness implications are not a reason for another campaign.

## 8. Cost, evidence, tests, and closeout

Report B/C repeat values, ranges, and medians for **raw-to-usable POINT**, **raw-to-completed analysis**, and **command exit**, together with RTC critical-path time, actual CPU work, memory, and any changed support. Keep synchronous output/required noise work included. Separate diagnostic overhead and parallel-worker sums from elapsed latency. Do not extrapolate this local two-network test to Unity/all arrays or convert serialization savings into an unmeasured end-to-end speed claim.

Use one campaign-owned directory under `~/work_toltec/local_data/citlali-validation/`, with a numerical peak-disk bound based on actual allocation. Retain compact population tables, applicability/design bindings, necessary maps, source/noise measurements, timings, configs, and reproduction code. Aim for a few hundred MB of durable evidence. Avoid redundant deterministic payload copies and new per-sample/per-cycle tiny-file proliferation; do not launch a storage redesign.

After committed compact preservation and saved-only report verification, remove only this campaign's enumerated inactive native scratch. Protect original/tune/APT/calibration/team inputs, prior compact evidence, and unrelated work. Perform exhaustive scratch verification once in the shared guarded cleanup path, not again through a reviewer. Report remaining files honestly rather than claiming the entire campaign directory is gone.

Use focused native/Python fixtures for fixed-set membership, unselected-detector identity, state isolation, paired x/r application, coefficient/phase/cadence correctness, physical gaps and guards, injection mapping, and failure propagation. No broad synthetic parameter sweep. Require a bounded preflight and final independent review of the frozen set, scientific comparison, control regression, and evidence/cleanup procedure. Review uses saved evidence and small fixtures, not another observation campaign.

Executed failures or missing required outputs return nonzero with truthful failed/blocked states. Preserve diagnostic products and the established failed-source-fit/empty accepted-output behavior. Do not drop a network, change membership, relax a guard, or silently revert the candidate to rescue its result.

## 9. Deliver the decision, not just completed-run counts

Lead the owner report with one B-versus-C cost–benefit table: selected population, residual near-11-Hz contamination before/after RTC and after PTC, final compact and broad response, matching uncertainty, usable-data loss, and elapsed seconds.

Answer directly: **Does this fixed notch improve the final five-feedback POINT result enough to justify its response/support/runtime cost on the measured affected population?** Distinguish a useful current-observation candidate, suppression without demonstrated final-map benefit, an unfavorable tradeoff, and insufficient evidence. Do not force a favorable verdict. If PTC already makes the final-map difference practically negligible, say so; if the notch improves useful sensitivity while preserving acceptable measured response, quantify it.

Update status, goals, reproduction instructions, and the handoff. Preserve B as the current development reference and C as a named experimental candidate; recommend adoption or rejection with the stated scope, but do not silently change defaults. Commit source, compact evidence, and review locally. **No push, merge, production activation, new observation/array, new threshold/design search, or automatic follow-on.**

**Stop after this one fixed-design comparison. Do not reopen PTC/FRUIT optimization.**
