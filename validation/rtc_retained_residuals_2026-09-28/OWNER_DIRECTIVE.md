# Codex directive: contamination in accepted POINT outputs
## POINT 152430/0/2 — a1100, networks 1–4

**Owner:** Grant Wilson  
**Date:** September 28, 2026  
**Status:** Execution order. Recover the completed baseline, assess retained contamination, and execute at most one evidence-supported additional-rejection comparison under the bounds below. Independently review, retain compact evidence, commit locally, and stop. Do not return only another testing proposal.

> The input is badly contaminated. Large rejection fractions are acceptable. Our priority is the trustworthiness of the accepted timestreams, final maps, and POINT measurements—not recovering exposure or reducing the number of flags. Determine what contamination survives, why it survives, and whether additional rejection produces a scientifically better output.

## 1. Owner decision and scope

Retain **B** as the control and keep **fixed11 parked**. Defer the previous proposal to replace whole-physical-run unresolved exclusions with processing-scan containment. **No exposure recovery, unmasking, releveling, or plateau stitching is authorized.** Network 3's reported 50.27% RTC loss is not, by itself, a reason to change treatment.

This milestone concerns accepted residual contamination on this same four-network a1100 subset. Do not expand to another observation or array, redesign FRUIT, optimize broad-source reconstruction, increase PCA rank, reopen notch design, or begin a performance campaign.

The completed assessment is encouraging but bounded: inspected RTC-missed transitions were largely suppressed by PTC; a local outlier remained in n1_w5; compact-source recovery was good; positive injections changed some support decisions. None of those observations alone proves that more flags help or that the reference output is clean. Source-response differences can cancel additive contamination shared by the reference and injection branches.

## 2. Recover the completed state and preserve the method

Read the completed assessment, closeout, reproduction instructions, consequence table, window-selection erratum, response records, maps, and compact evidence under:

- `validation/rtc_step_consequences_2026-09-28/`
- Current `doc/REFACTOR_STATUS.md`, `doc/SCIENTIFIC_GOALS.md`, `doc/BRIGHT_POINT_PIPELINE.md`, applicable repository instructions, and accepted RTC/ALIGN/AST/CAL/PTC/MAP/VAL contracts.

Use reported science source `839f31b28b52de6a46bc9c2dc623673378839cfa` and tree `591de3682d5626fe494a61388c66800b5c95f99d` as discovery anchors, not reset instructions. Recover the actual later closeout and current worktree. Preserve unrelated work, the original checkout, prior compact evidence, and the older frozen references. Distinguish execution source from reporting-only commits. Do not merge, push, rewrite history, or promote a production default.

Use the existing `tools/timestream_successor/point_pipeline.py` route and native scientific owners. Preserve the accepted CLI scope repair. No parallel pipeline or generic diagnostics framework.

Bind the same complete original 152430/0/2 inputs, exact tune associations, telescope streams, APT, and calibration. These are **a1100 partial-array** outputs. The assessment reports that all dependencies were already local and that previous native scratch was deleted. Verify readiness; do not assume deleted RTC/PTC arrays are still available. No download is expected. Use configured, read-only Unity recovery only for an exactly identified missing original dependency; no remote computation or archive modification.

Keep the established numerical method:

- The accepted a1100 RTC profile: 67-tap lowpass, factor 1, native integration 0.008192 s; existing event logic, filter support, paired x/r semantics, independent validity, and donor rules.
- Rank-10, observed-entry, flag-aware ALS; cold M0; warm initialization followed by a **full new solve** of each current residual; branch/network/physical-segment-local state.
- Exactly five FRUIT applications; ordinary MAP; 2-arcsec pixels; 40-arcsec feedback; threshold-3 source-connected BASE; unity gain and full replacement. **T5 consumes m4; m5 remains unapplied.**

Numerical processing scans are not physical discontinuities. Do not turn their boundaries into new physical containment assumptions.

## 3. Start with accepted residuals, not the event catalogue

Use saved evidence first. Establish what is present, regenerable, or not measured. The previous 32 windows are targeted examples, not a complete contamination survey. Missing support is unavailable, never a measured zero. Distinguish raw Learn candidates, event-group seeds, rejected proposals, unresolved extent, and actual Apply support; carry forward the selection-label correction.

Produce compact detector/physical-run summaries across **all four networks' retained population**, using the existing stage products or the single permitted reference replay. Assess CAL/pre-PTC and the actual M0 and T5 PTC residuals—not sky-restored totals. Keep original-domain evidence and physical units traceable. Capture summaries and bounded excerpts during that replay instead of repeatedly regenerating full streams.

Prioritize residual transitions, finite excursions, isolated surviving outliers, unstable stretches, and ringing or leakage outside existing exclusions. Start with n1_w5 and the existing retained-transition examples, but apply the same diagnostic rule across the population. Do not select only the most dramatic two detectors. Retain at most 16 additional targeted/control windows beyond the existing record; compact population statistics need not be limited to those windows.

For each consequential residual class, answer:

1. What supports a non-astronomical origin in original x/r, temporal structure, peer context, or the event chain?
2. Why did existing Learn–Consider–Apply retain it, and how much survives PTC?
3. Does it enter actual final MAP contributions, where does it project, and what output consequence is observed or still unresolved?

A large CAL step removed by PTC does not qualify automatically. Neither does an isolated large noise realization, a source crossing, smooth atmospheric variation, or a change in detector-space common mode. Use existing source geometry and quiet controls for diagnostics without altering production optical-protection authority or scientific masks.

Local adjacent-difference MAD is a descriptive scale, **not** uncertainty on a step contrast. Inspect off-center features and tails rather than relying only on a window-center statistic. Do not claim catalogue precision/recall or formal significance from the targeted examples.

## 4. Judge the actual output, independently of injection recovery

Assess the unmodified reference's final map and accepted POINT measurement directly. Use existing estimators, background/coverage definitions, and noise machinery. Examine coherent residual structure, source-adjacent contamination, residual tails, and the fitted source's amplitude, centroid, and shape. Report physical magnitudes and the relevant local noise or estimator uncertainty.

Trace suspicious samples to their real AST/MAP positions and contributions. Any event-associated diagnostic projection must use the native coefficients and be labeled as a contribution diagnostic, not an independent astronomical map. Do not interpret a sky-restored signal as residual contamination or an absent map feature as a universal upper bound.

Keep three conclusions separate:

- Contaminant suppression in accepted timestreams and maps.
- Preservation of astronomical response and the POINT measurement.
- Changes in coverage, normalization, noise, and usable source crossings.

Cleaner appearance or lower residual variance alone is insufficient. Large additional loss is acceptable, but removing nearly everything, losing source admission, or substantially degrading the measurement is not a successful cleaning result. Do not demand publication-grade uncertainty before a bounded test, and do not pursue sub-noise differences without a consequential stability or correctness concern.

Use the known positive-injection support changes as a robustness diagnostic. Their proposed block-noise-normalization mechanism remains an inference. Inspect existing records where useful; do not silently add a normalization or threshold redesign to this experiment. Broad-response limitations remain reported, not a new reconstruction project.

## 5. One authorized candidate: conservative additional quarantine

If the audit identifies a defensible class of **retained post-PTC contamination**, run one additional-rejection comparison. Demonstrated map bias is not a prerequisite—that is part of the test—but require reproducible contaminant evidence and a traceable path into accepted output. High pre-PTC amplitude or a high rejection fraction is not sufficient.

The permitted candidate is a **validation-only detector/physical-run quarantine**: additionally withhold complete physical runs of detectors belonging to that residual class. This deliberately conservative comparator tests whether keeping those suspect contributions harms the result. It is not a proposed universal detector-health rule or an optimized minimum-loss mask.

Before candidate execution, freeze a compact selection record containing the class definition, reproducible evidence rule, complete selected detector/run list, supporting original and residual evidence, projected source/background overlap, and expected mechanism of improvement. Apply the rule consistently across all four networks. The owner delegates this bounded experimental selection; individual detector approvals and a new general classifier are not prerequisites.

Use the existing native support/admission interface. Selected contributions must be unavailable to downstream scientific fitting and MAP, not merely blanked after PTC or removed from the displayed map. Ensure excluded data cannot supply donor support or leak through permitted filter influence. Preserve measured originals and independent x/r validity; record the experimental veto separately from acquisition invalidity. Paired handling must remain contract-consistent.

The **only deliberate scientific change** is this additional veto. Do not narrow old exclusions, change event thresholds, widen selected guards at the same time, retune filters, change rank, or repair values. Natural recomputation of nuisance fits, support-dependent quantities, PTC, and FRUIT is required. If the native interface cannot express the veto without other scientific policy changes, report the specific blocker rather than invent another processing route.

Freeze the same quarantine population for the candidate reference and its injection branches. Leave the existing RTC Learn–Consider–Apply decisions active independently in each branch; do not transplant B's learned events, PCA basis, or feedback maps. Injection results therefore test the existing active pipeline **conditional on this fixed experimental quarantine**, not a newly validated automatic quarantine selector.

If no defensible residual class emerges, stop after the assessment with the measured limits and retain B. Do not manufacture a candidate to spend the execution budget.

## 6. Matched source, residual, and noise comparison

Use the saved B reference and all three completed B injection branches as controls after verifying compatibility. Permit one unmodified B M0–T5 replay for missing population diagnostics and instrumentation-neutrality checking. Verify established numerical comparators and exact discrete support/event identities; a failed baseline equivalence check blocks the candidate comparison rather than licensing hidden tuning or extra runs.

The candidate has four original-input M0–T5 trajectories: its own unmodified reference, compact +10%, compact −10%, and displaced broad +5%-amplitude. Reuse the exact frozen a1100 rasters, coordinates, normalization, taper, and hashes from the completed assessment. Do not redraw them from the candidate map or move a source to avoid contamination. Inject once into original-derived x before RTC; preserve original r and acquisition validity.

Measure each policy's injected-minus-**its-own-reference** response. Do not subtract B's reference from a candidate injected map and call that a source transfer. Keep matched ideal-addition definitions and report reference-relative coverage/support losses explicitly. Conditional common-pixel gains are useful diagnostics, not full-domain transfer claims; unavailable contributions must not become zeros or disappear from the accounting.

Report final peak, template, aperture, centroid, and shape increments with the established absolute-error conventions. Also compare the two uninjected reference maps and POINT measurements directly: this is where persistent additive contamination matters. Give both global output changes and class-associated residual changes, including quiet regions/windows not used to select the quarantine. Do not attribute every map difference to a single event when PTC and feedback have recomputed.

Use saved B conditional uncertainties and the unchanged uncertainty machinery for the candidate's own coverage/noise. Report matched physical units and uncertainty-relative context. Single-map sigmas are not paired-difference significance; a same-data, outcome-selected quarantine is an in-sample development test, not independent validation. Do not rebuild the noise estimator or launch a new uncertainty-method campaign.

**Success means evidence of reduced consequential contamination while preserving a useful, stable POINT result.** Recovery of samples is not a success criterion. If stricter rejection merely changes coverage or removes features already adequately handled by PTC, retain B.

## 7. Execution and resource bounds

This is a new, finite budget; do not reuse allowances from previous directives.

| New work | Maximum M0–T5 trajectories | Feedback applications |
|---|---:|---:|
| Unmodified B diagnostic/equivalence replay, only as needed | 1 | 5 |
| Candidate reference plus the three frozen injections | 4 | 20 |
| **Total** | **5** | **25** |

No separate RTC-only captures, second quarantine population, mask-strength sweep, extra observation, timing repeats, or automatic replacement allowance. A failed launch that enters science consumes its trajectory slot. Perform routing/configuration/fixture checks before expensive launches. If comparison validity cannot be established within budget, preserve completed work and report the limitation.

Count actual original-input parents, initial network RTC Apply starts/completions, internal reassessments, endpoints, and feedback applications separately. Five complete four-network trajectories imply at most 20 initial network Apply executions; internal attempts are additional accounting, not fictitious new trajectories.

Reuse the established local resource policy unless required for safety, and record any necessary difference. Record wall time, CPU, sampled peak memory, and diagnostic overhead without claiming a speedup. No concurrency-tuning campaign.

Use one new campaign root under `~/work_toltec/local_data/citlali-validation/`, with durable records under `validation/rtc_retained_residuals_2026-09-28/`. Preflight actual required disk against a documented ceiling; start from the previous 30 GB ceiling rather than silently expanding it. Process sequentially where useful and retain packed summaries/excerpts rather than full duplicate timestream campaigns. Stop before a resource-bound violation.

## 8. Evidence, review, cleanup, and handoff

Add only focused tests needed for diagnostic neutrality and the experimental veto: B unchanged when disabled; exact selected run identity; paired support and independent validity; exclusion from fitting, mapping and donor supply; filter influence; physical gaps versus numerical scan boundaries; and truthful support/normalization accounting. Reuse existing tests and reporting code.

Retain compact population summaries, selected/control excerpts, the selection manifest, native support receipts, final maps, response/noise records, timings, source/config/input bindings, and reproduction instructions. Saved-only report regeneration must not require deleted full streams, fresh scientific fitting, or another uncertainty ensemble. Do not preserve failed bulky products merely for historical completeness.

Independently review the scientific claims, selection bias, comparison validity, source/support accounting, execution counts, and reproducibility. Follow the existing guarded cleanup and approval procedure for this campaign's own newly generated scratch; prior deletion approval does not authorize new paths. Protect originals, tunes/APT/calibration, team products, previous compact evidence, and unrelated work. Keep the scientific report available even if cleanup awaits approval.

Lead the final report with answers to these questions:

1. What contamination survives into accepted RTC, PTC, and final MAP output, and at what measured scale?
2. Which suspected features are adequately handled by PTC, which remain consequential, and which remain unestablished?
3. Did conservative additional rejection improve the final residuals and POINT result, rather than merely reduce samples or variance?
4. What happened to compact response, source admission, centroid/shape, noise, support, and active decisions?
5. Should B remain the working baseline, or does this bounded test justify a separately considered treatment change?

Distinguish **observed**, **inferred**, and **untested** conclusions. Report the additional loss, but do not frame it as a defect in itself. Update status/goals and a concise handoff to preserve the owner's contamination-first priority. Commit locally and stop. **No automatic promotion, exposure-recovery experiment, FRUIT redesign, fixed11 activation, push, or next campaign.**
