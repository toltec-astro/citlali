# Codex directive: establish POINT map quality, then hand off to OOF

**Owner:** Grant Wilson  
**Date:** October 1, 2026  
**Immediate scope:** A bounded POINT map-quality assessment on the two existing a1100 anchor observations, followed by concrete preparation for the first OOF experiment.  
**Status:** Execute the POINT assessment and prepare the OOF handoff. Do not return only another testing proposal. New OOF observational reductions are the next milestone, not part of this numerical allowance.  
**Execution mode:** Prepare and preflight; launch a bounded batch; end the active agent turn; resume on the owner's message.

> Determine whether the current successor produces better astronomical measurements and more trustworthy maps—not merely cleaner-looking timestreams. Finish this POINT assessment without another optimization campaign. OOF is the next development target, ahead of a broader POINT survey or SCIENCE campaign. Carry forward validated infrastructure and evaluation principles, not untested POINT-specific assumptions.

## 1. Recover current state and use the calibration investigation

Recover the actual checkout, HEAD, worktree changes, applicable instructions, active processes, completed products, investigation ledger, and latest calibration/amplitude assessments. Discovery anchors—not reset instructions—include:

```text
~/.codex/worktrees/ptc-rank-response/citlali-refactor
validation/point152420_amplitude_investigation_2026-09-30/
validation/point152420_map_comparison_2026-09-29/
validation/rtc_point_transfer_2026-09-28/
validation/rtc_two_stage_2026-09-28/
```

Read the actual closeout of the APT-v2 investigation and any subsequent repairs. Do not assume the September 30 directive is a completion report, assume the issue is still open, or restart completed lineage work. Summarize the demonstrated origin of the factor difference, applicable workflow legitimacy, repairs, remaining discrepancy, and exactly which previous products remain usable. Continue a genuinely unresolved link only to the extent needed for an interpretable comparison; preserve other findings explicitly rather than declaring them resolved.

Establish calibration equivalence at the appropriate detector/sample boundary when holding calibration fixed. Do not align finished maps with a fitted scalar or treat a median factor ratio as a complete transformation. Do not use a correction derived from the target POINT as independent validation of that same target's photometry. Follow `redu09` provenance through its parent directories when a dependency remains unresolved.

Scope uncertainty to affected claims. An unresolved absolute scale does not necessarily block every morphology or stability diagnostic, but detector-dependent calibration differences can also change weighting, cleaning, and shape. Demonstrate which comparisons remain interpretable; do not simply label all non-flux metrics calibration-independent.

Preserve active batches and their bound source/inputs. Account for completed work and retire superseded unstarted controls with a short explanation; do not run them merely because they were prepared.

## 2. Freeze the POINT experiment

Use **POINT152420/a1100 and POINT152430/a1100, networks 1–4**, verifying exact acquisitions, tunes, detector identities, and available dependencies. These are development/regression anchors, not previously untouched validation data. Do not add a third POINT, another array campaign, or a fresh B/Q treatment search before OOF.

The successor working reference remains **B**, with any demonstrated correctness repair explicitly versioned: current RTC/CAL and paired x/r validity/support semantics; cold M0; warm-initialized, fully re-solved rank-10 observed-entry ALS; ordinary MAP on the existing 2-arcsecond grid; existing threshold-3, measured-brightness, 40-arcsecond POINT feedback policy; exactly five feedback applications. **T5 consumes m4; m5 is not applied.** Verify later accepted changes rather than blindly restoring this historical description. Keep Q and fixed11 parked.

No PCA-rank, FRUIT-iteration, admission-radius, threshold, notch, JINC, weighting, or source-model optimization is authorized. Necessary correctness repairs and measurement-tool repairs are authorized, with tests and explicit identification of invalidated results.

Define the comparison arms before numerical work:

| Role | Meaning |
|---|---|
| **R: earlier processing reference** | A recoverable, explicitly identified earlier treatment of the same acquisition, with understood calibration, population, units, geometry, and endpoint. Use existing implementations; do not build a parallel legacy pipeline. |
| **S: current successor** | The frozen current B route, including any verified repair required for valid execution. |
| **Historical context** | Original saved products, including `redu09`, retained unchanged. They are evidence, not ground truth. |

If R needs a documented calibration-aligned rerun, distinguish it from the original historical product. If other inputs or populations differ, disclose them: an operational comparison can remain informative without identifying a unique algorithmic cause. Do not make causal attribution stronger than the controls support.

M0-to-T5 measures feedback's effect; B-to-B measures reproducibility. Neither establishes improvement over earlier processing. If a defensible R cannot be recovered for one anchor, report that limitation and complete the valid successor diagnostics; do not substitute Q or manufacture a baseline.

## 3. Measure astronomical utility and residual contamination

Implement one compact scorecard, reusing current diagnostics and uncertainty machinery:

| Dimension | Required evidence |
|---|---|
| **Pointing** | Centroid in a valid stated frame, applicable uncertainty, subset stability, and failed/ambiguous fits. Real-data agreement is not known-truth accuracy; use known injected positions where applicable. |
| **Source response** | Peak above background, major/minor widths, asymmetry, wings/bowls, and suitable aperture/template or descriptive integral measures. State beam/unit conventions; a Gaussian angular integral is not automatically flux density. |
| **Background** | Source-excluded noise on relevant spatial scales, anisotropic correlation/striping, positive and negative tails, and source-like artifacts. Distinguish valid empirical uncertainty from formal weights or descriptive scatter. |
| **Support** | Accepted exposure and source crossings, common-supported performance, and separately reported lost/degraded support. Missing samples or pixels are unavailable, not zero sky. |
| **Practical cost** | Actual raw-to-final and major-stage wall time, memory where available, and evaluation overhead reported separately. No new repeated timing campaign. |

Preserve free centroid/shape measurement. Do not recenter maps independently on their fitted sources and then claim pointing agreement. Use difference images only with valid frame registration and compatible units/grid interpretation. Do not subtract native Alt-Az and sky maps without that work, or smooth away a shape difference merely to improve agreement.

Predeclare source/background regions and spatial-scale summaries from existing justified geometry, not the most favorable final map. Compare common support while also accounting for the original reference domain and everything that disappeared. Report inadequate background support rather than training a noise model on source emission.

Judge noise together with measured response. Where applicable, report local input-equivalent amplitude uncertainty, `sigma_A / abs(g_A)`, using a compatible estimator and measured injection response. Do not claim a sensitivity gain from reducing source and noise together. This quantity does not validate absolute calibration or remove additive bias.

Inspect **uninjected maps as well as injected-minus-own-reference responses**. The latter can cancel persistent contamination. Preserve compact/shoulder evidence already available, but do not reopen shoulder reconstruction. Use existing applicable uncertainty methods, recomputing bindings when required; do not borrow incompatible uncertainties or launch a new ensemble at every feedback cycle. Shared data/calibration and learned-model dependence must remain explicit; do not add paired error bars in quadrature under an unsupported independence assumption.

Use documented scientific goals and relevant measurement uncertainty. Do not invent a POINT latency deadline, centroid tolerance, or single-pass attenuation gate. A brighter, narrower, smoother, or more heavily rejected map is not automatically better.

## 4. Fill only the consequential evidence gaps

**Saved-product assessment first.** Inventory usable maps, M0/T5 measurements, intermediate feedback summaries, calibration bindings, source-response controls, noise products, and runtime records. Rerun only when a necessary product is missing or invalidated. A saved map is not a resumable timestream checkpoint.

**One subset-stability test.** Before new outcomes, choose one anchor and one geometry-supported split into two non-overlapping, reasonably coverage-balanced scan groups. Prefer POINT152420 where a matched R/S split is feasible; otherwise justify choosing 152430 from acquisition/support information, not favorable reduction results. Compare both methods on the same split.

For an independent-reduction claim, refit the applicable adaptive stages within each subset, preserve physical gaps/filter support, and disclose shared calibration or other dependencies. Submaps from one full-observation solution remain conditional diagnostics, not independent reductions. Interpret time-dependent pointing/atmosphere and changed coverage explicitly. If no defensible split exists, report why rather than trying many partitions.

**Matched compact response only where missing.** Reuse valid full-learning evidence first. If it cannot support the R/S comparison, run the existing positive/negative compact perturbation recipe on that same anchor, with common physical injection definitions and independently checked normalization. Inject at the original-derived x boundary before RTC; retain measured r and native acquisition validity. Let each arm relearn its normal RTC/PTC/feedback decisions. Label frozen-state results separately. Do not inject into final maps and describe the result as an end-to-end response test.

## 5. Numerical allowance and launch discipline

This is a new, bounded map-quality milestone. Its allowance is a **ceiling for missing evidence, not a target**:

| New numerical roles | Maximum starts |
|---|---:|
| Uninjected R/S on the two anchors | 4 |
| Two subsets, R/S, on one anchor | 4 |
| Compact positive/negative perturbations, R/S, on that anchor | 4 |
| Replacements after diagnosed and tested implementation defects | 2 |
| **Maximum new starts** | **14** |

Reuse compatible completed roles. Do not repurpose unused slots for new treatments, partitions, observations, repeated timing, or optimization. Fourteen is not fourteen additional successful experiments: at most twelve planned roles plus two replacements. Record all failed starts, upstream parent generation, subset reductions, feedback applications, and resource costs. No extra observation-wide RTC passes are free because they are called diagnostics. Small fixtures and saved-map arithmetic are ordinary development.

Freeze the minimal runnable matrix and exact scientific identities after the inventory. Do not stack unused allowances from older campaigns onto this one. Keep prior and already-active work in its original accounting. A necessary change beyond this matrix is a substantive budget decision, not an automatic retry.

Use the established launch-and-manual-resume workflow. Preflight the actual native route; freeze code, inputs, and configuration; launch using an already verified persistent mechanism; make at most one bounded startup check; report status/resume paths; then end the active turn. No polling or monitoring subagents. If persistence is unverified, supply one exact owner-terminal command and report readiness, not a fictitious launch.

On the owner's resume message, verify completion, inspect compact results first, and perform authorized diagnosis/repair within the remaining allowance. If still running, report that once and stop monitoring. Never modify source bound to active numerical work.

## 6. Close POINT on evidence—not perfection

For each anchor, conclude **improved, comparable, degraded, or not established**, separately for the important measurements. Support each conclusion with effect size, applicable uncertainty, contamination/support cost, and runtime. An inconclusive metric needs a specific reason, not a generic demand for more validation.

Separate demonstrated correctness improvements, measured scientific improvements, diagnostic capabilities, and untested claims. Identify the largest remaining scientifically consequential limitation, but **do not automatically begin its repair or another POINT tuning cycle**. Carry a nonblocking limitation into the OOF handoff. A shared correctness defect affecting OOF must be fixed or explicitly block dependent OOF execution; do not present unsupported output as valid.

A mixed or negative POINT result can close this milestone. Production acceptance and universal improvement are not prerequisites for moving development attention to OOF.

## 7. Prepare the first OOF experiment as the next development step

After the POINT assessment, complete this preparation without another permission request. It is part of this assignment; **new OOF observational reductions remain outside this batch**.

Recover current OOF contracts, code paths, archived controls, and the mode-aware owner decisions, including `SCI-FRUIT-owner-directive-2026-09-09.md` and any later accepted amendments. Use the following existing direction rather than importing the POINT source prior:

**Do not require a bright core, high-S/N seed, ring, contiguous footprint, Gaussian shape, or narrow predetermined morphology family.** Preserve the known-source observing context without assuming the distorted structure that OOF is intended to measure. Cross-focus/coherence-based inference remains a candidate, not an already adopted algorithm.

Inventory local data and available archive metadata. Nominate **one complete acquisition-defined OOF focus sequence**, not one favorable map, for the first experiment. Identify every member, actual focus setting/sign, array/network population, source, timestamps, elevation/scan context, telescope streams, exact tunes/APTs/calibration dependencies, and historical reduction provenance. Prefer a sequence offering useful variation in defocus while remaining feasible. Do not assume a particular number of focus states or invent observation IDs. State local completeness and exact missing paths/sizes. Do not launch a broad archive survey or bulk download.

Prepare a concrete first-comparison plan covering:

- **Reference and endpoint:** recover the actual historical OOF/S/N-based treatment and a clearly specified successor route. Keep focus states distinct; do not coadd them as one unchanged sky image. Carry over shared processing components only where their contracts apply. POINT's 40-arcsecond support, Gaussian guard, threshold-3 policy, rank choice, and five-feedback endpoint do not become validated OOF defaults.
- **Scientific scorecard:** measure distributed structural recovery, scale-dependent attenuation, low-surface-brightness structure, false promotion, correlated residuals, support, and computational cost. Separate amplitude/normalization from shape. Where an established OOF forward model or solver exists, specify how map changes affect its residuals and recovered wavefront/surface parameters without treating agreement with a fitted model as independent truth. Do not build a new inversion solver here.
- **Readiness and first discriminator:** identify the earliest unsupported boundary—mode routing, frame/projection, footprint/background support, uncertainty, PTC behavior, or feedback admission. Specify the smallest first batch, required inputs, exact settings, and work/storage budget. Separate historical reproduction from deliberately changed controls. Name a real scientific decision if one is missing; do not hide it behind a POINT fallback or an undocumented morphology prior.

Use existing mode-aware interfaces and native scientific owners. Necessary shared interface repairs or fixtures implementing established contracts are authorized; a second pipeline, new OOF inference policy, cross-focus pooling method, or general framework is not. Historical and visually persuasive OOF maps are comparison evidence, not quantitative truth. OOF is the next test of generality, not already validated SCIENCE performance.

## 8. Autonomy, resources, and deliverables

Carry forward `CODEX_POINT152420_COMPLETION_AND_BOUNDED_AUTONOMY.md`: own relevant implementation, maintainable local refactoring, tests, diagnostic repairs, documentation, and local commits. Do not ask permission for each helper edit. Ask only at a genuine unresolved scientific, scope, resource, or external-action boundary; continue independent authorized work.

Respect current CPU/thread, memory, and shared-storage limits, including the last recorded shared 30 GB ceiling; verify its accounting scope and any later accepted amendment. Use local evidence first. Existing configured Unity access may be used for targeted read-only recovery of identified missing POINT dependencies and bounded OOF metadata/inventory. List OOF raw-data retrieval needs for the next batch rather than downloading the archive now. No remote computation, archive writes, credential changes, pushes, merges, production activation, or telescope-control changes.

Keep compact maps, metrics, manifests, logs, and reproduction inputs. Delete owned reproducible scratch when no active job or necessary continuation needs it, preserving compact failure evidence. Protect raw/tune/APT/calibration inputs, original historical maps, unrelated work, and authoritative archives. Do not retain bulky failed intermediates merely for posterity.

Deliver one compact POINT assessment with machine-readable metrics and common-scale comparison figures, plus one OOF handoff. Reuse existing reporting structures; suitable new locations are `validation/point_quality_2026-10-01/` and `handoff/POINT_TO_OOF_2026-10-01.md`. Include exact executed identities, actual work/retry accounting, reproduction/launch commands, and one focused independent review of the substantive conclusions—not a review for every edit.

Lead the owner report with: **Did POINT improve, in which measurements, and at what cost? What did calibration resolution change? What remains unproven? Which OOF sequence comes next, and what exact comparison is ready or specifically blocked?**

**Finish with an honestly assessed POINT checkpoint and an actionable OOF handoff—not another open-ended POINT campaign.**
