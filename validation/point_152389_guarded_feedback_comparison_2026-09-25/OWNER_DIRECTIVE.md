# Codex directive: POINT 152389 — guarded BASE-POINT / CORR / DIAG feedback comparison

**Owner:** Grant Wilson  
**Date:** September 25, 2026  
**Status:** Implement the bounded continuation, validate, execute, assess, independently review, and stop.  
**Scope:** POINT 152389; a1100, a1400, a2000; ordinary MAP; three matched admission policies; the existing compact/shoulder response and shoulder-seed tests.

> We have a working expected-source guard and a completed admission screen. Now test whether admitting additional measured brightness improves source recovery without increasing dependence on an artificial starting model. Keep the method fixed. More admitted pixels or a brighter map are not, by themselves, improvements.

## 1. Scientific question and new authorization

This order explicitly authorizes the feedback trajectories withheld by the expected-source-guard work order. It does **not** reopen source recognition, threshold calibration, neighborhood design, covariance learning inside feedback, or JINC.

Compare three policies that share the **same implemented expected-source guard, fitted-background convention, and measured-brightness operand**:

| Policy | Admission after the shared source guard passes |
|---|---|
| BASE-POINT | The guarded source-connected threshold-3 baseline, with no supplemental admission. |
| CORR | BASE-POINT plus the existing full-covariance neighborhood supplement at its frozen calibrated threshold. |
| DIAG | BASE-POINT plus the existing diagonal-covariance neighborhood supplement at its separately frozen calibrated threshold. |

There are three scientific contrasts: BASE-POINT against historical BASE exposes the guard/fit change; CORR against BASE-POINT measures the effect of the supplement; CORR against DIAG tests whether accounting for correlations adds value beyond neighborhood aggregation and separate threshold calibration. Neither new policy is presumed preferable.

Fresh residual-specific PTC learning/application, ordinary MAP accumulation, the exact archived pre-PTC signal injections, and the first-candidate-only shoulder-seed intervention are authorized **only within the matrix below**. Existing science contracts and component ownership remain in force. This is a diagnostic campaign, not a production/default change.

## 2. Recover the accepted implementation and freeze the experiment

Read applicable repository instructions and recover the current reviewed implementation. Primary authorities are:

- `CODEX_POINT_152389_EXPECTED_SOURCE_GUARD.md` and its completed assessment, frozen thresholds, checks, reviews, and sealed products.
- `CODEX_POINT_152389_CORRELATION_AWARE_ADMISSION.md` and its completed preparation, original BASE replay, and screening evidence.
- `CODEX_POINT_152389_RESPONSE_AND_ROBUSTNESS_CD.md` and the authenticated original response/seed operands, measurements, domains, and histories.
- The existing POINT/MAP+/PTC/AST/NOI/FRUIT contracts, original ten-cycle and A+B records, and applicable governance.

The owner-provided guard closeout reports all six admission screens passed, all tested cores accepted, all blanks rejected, and no new feedback trajectories. Discovery anchors are local commit `01be8c312` and:

```text
/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/
  successor-point-152389-expected-source-guard-20260925-r2/
  successor-point-152389-correlation-admission-20260924/
  successor-point-152389-response-robustness-cd-20260923/
```

These are discovery hints, not substitutes for checking the worktree, source, and authenticated inputs. Recover exact stored values and hashes. Do not infer an implementation from this document's shorthand or silently substitute a nearby run.

For identification only, the reported thresholds are:

| Array | CORR, rounded display | DIAG, rounded display |
|---|---:|---:|
| a1100 | 3.83549 | 6.91257 |
| a1400 | 3.70923 | 7.16588 |
| a2000 | 3.68998 | 4.68608 |

**Use the full saved thresholds, not these rounded values.** Missing, mismatched, or unusable threshold/covariance evidence blocks the dependent policy/array; it does not authorize threshold 3 as a fallback. Reuse the completed calibration, replayed validation, and 1.25-noise stress evidence. Do not generate new fields, repeat that campaign, or recalibrate.

Before execution, freeze a compact manifest binding this order, exact code/binaries/environment, input identities, numerical tolerances, policy definitions, thresholds, C0/K0, original admission-noise scale, geometry, response/seed operands, measurement rules, matrix, and stop rules. Use existing provenance tools. Preflight review is a bounded implementation check, not another owner-parameter-selection loop.

## 3. Hold these choices fixed

### 3.1 Source guard and measurement

Use the reviewed `POINT-EXPECTED` implementation unchanged: one finite, converged elliptical-Gaussian-plus-constant fit; positive peak above the fitted constant; radial centroid displacement at most 20 arcsec from the recorded expected origin; both principal sigmas strictly above 1.5 arcsec, including lower-bound-state rejection; and fitted-amplitude S/N strictly above 10. Retain the single deterministic attempt and inherited 3,000-function-evaluation ceiling.

The amplitude error remains the local-linear sensitivity of the free seven-parameter fit propagated through the frozen **full C0 on the fit support**, including background and shape/position nuisance parameters. DIAG uses the same full-covariance source guard; its diagonal substitution applies only to the later supplemental-neighborhood variance. Do not change fit weights, bounds, initialization, stopping tolerances, rank handling, or uncertainty normalization.

The S/N > 10 condition recognizes the bright POINT source; it is not a per-feedback-pixel threshold. Use the new shared fit for the new trajectories and their full-map measurement comparisons. Preserve historical fits separately rather than copying old backgrounds or parameters into new results. Signed difference maps and injection-only reference maps do not undergo positive-source recognition.

### 3.2 Admission and feedback brightness

Keep the original AST-centered 40-arcsec-radius feedback disk, 2-arcsec grid, eligibility, source anchors, connectivity, original frozen threshold-3 admission-noise scale, unity gain, and complete replacement semantics.

Keep full C0 and K0 frozen from the authenticated original REF M0. Do not redefine K0 from BASE-POINT, a later map, a perturbed source, or a newly expanded candidate. Preserve all implemented supplemental exclusions: original core, current baseline mask, and the anchor's own pixel. Preserve signed neighbor brightness, full-footprint support, weight-concentration checks, fixed beam-scale weights, strict score comparisons, positive anchor brightness, and final source connectivity. No sequential within-map growth or borrowing evidence from newly added pixels.

The candidate publishes **exactly the original measured brightness operand on selected pixels**. It never publishes a Gaussian fit, neighborhood average, predicted noise, deconvolution, or injected truth, except for the explicitly authorized initial seed intervention. Valid excluded pixels remain intentional model zeros; unavailable pixels remain unavailable; outside-domain feedback remains absent without removing observed samples or asserting zero sky.

Do not use C0 as MAP weights or new PTC coefficients. Frozen C0 is a campaign evidence scale, not a claim that the actual noise is constant through feedback. Carry forward the a1100 network-3/scan-5 condition and the a2000 background-model/source-transfer qualifications without repairing them here.

### 3.3 Upstream processing

Reuse immutable post-RTC/CAL, pre-PTC parents and native AST coordinates. Retain the accepted per-network/processing-segment observed-entry rank-10 PTC recipe, fresh centering/learning rules, solver/stopping conventions, native eligibility/normalization, ordinary MAP, and direct containing-pixel model projection. Paired-r behavior remains inert as in the parent; no r-channel signal injection or new treatment. No new RTC/CAL reduction or dependency migration.

## 4. Exact trajectory matrix and execution bound

Run the following five branches for **each of BASE-POINT, CORR, and DIAG in each array**:

| Branch | Immutable pre-PTC input | First candidate | Later candidates |
|---|---|---|---|
| REF | Original y | Its own policy candidate from unperturbed M0 | Same policy |
| CORE-PLUS | y plus the exact archived +10% compact operand after native projection | Its own policy candidate from its perturbed M0 | Same policy |
| CORE-MINUS | y plus the exact archived −10% compact operand after native projection | Its own policy candidate from its perturbed M0 | Same policy |
| SHOULDER | y plus the exact archived +5% broader/offset operand after native projection | Its own policy candidate from its perturbed M0 | Same policy |
| SEED-SHOULDER | Original y; no data injection | Its own REF m0 plus the exact archived shoulder raster | Same policy from m1 onward |

**Ceiling: 45 logical trajectories, ten feedback applications each, at most 450 feedback applications.** M0 is not a feedback cycle. Preserve M0 and T1–T10; T10 uses m9; save m10 but never apply it. Complete ten cycles unless a stop condition in section 8 occurs. Small drift does not authorize early stopping; persistent drift does not authorize cycle 11.

BASE-POINT gets the **complete five-branch matrix**, not just a REF trajectory. Historical BASE cannot replace this matched control merely because the guard retained historical membership on 15 saved maps: changed fit backgrounds or brightness operands can change subsequent cleaning and admission. Reuse authenticated historical BASE results for context; do not rerun the old low/high-seed, threshold-5, or entire C+D campaign.

Execute REF before its dependent comparisons for each policy/array, then the compact response pair and the paired SHOULDER/SEED-SHOULDER tests. Run independent work concurrently where practical. Report every matrix cell, including blocked or stopped cells; do not silently select only arrays or policies that show extra pixels.

M0 processing precedes policy selection. The four distinct physical inputs per array can therefore share authenticated, mathematically identical M0 processing across policies; the seed branch shares REF M0. Reuse existing verified M0 products when fully applicable. Account for any necessary initial PTC/MAP calculations separately from the feedback ceiling.

Reuse existing step results only when immutable inputs, complete applied brightness operands, geometry/support, processing configuration, and required numerical identity establish exact equivalence. Identical masks, equal pixel counts, approximate map agreement, or a shared historical endpoint are insufficient. Count logical trajectories separately from actual calculations. Do not build a general caching/scheduling framework to obtain this reuse.

## 5. Signal intervention and native recurrence

Recover the exact archived compact plus/minus and shoulder rasters, taper, physical axes, support, normalization, and projection. Do not regenerate them from rounded fits or redefine them using a new policy's endpoint. The compact decrement is applied to an already bright source; it is not a negative-source detection experiment.

For a data-injection branch, form its immutable parent once:

```text
y_I = y + P(I)
```

Apply the signal coherently to the relevant valid pre-PTC entries, including source-bearing entries used by PTC but not final MAP contributors. Preserve invalid entries and all native support. No extra optical beam convolution, calibration factor, filter, independent network signal, or repeated injection.

Starting from each branch's own M0, cycle k uses only its latest candidate m_(k−1):

1. Begin again with that branch's immutable y or y_I.
2. Project and subtract m_(k−1), then freshly learn/apply the unchanged PTC recipe to that residual.
3. Restore exactly the operand subtracted, preserving the native centering and rejoin conventions, and accumulate ordinary MAP T_k.
4. Measure/guard T_k and generate m_k from that map using the branch's policy and complete replacement.

Do not subtract from a previously cleaned residual, accumulate old models, reuse historical learned bases on changed residuals, or add the known injection as a separate restoration term. Changed admission is expected to change later residuals and learned PTC state; this is the experiment, not a reason to freeze old state.

For SEED-SHOULDER, first require the normal REF source guard to pass, then add the archived shoulder to **that policy's own REF m0** on the original supported domain. This is the only first-candidate-only admission exception. Do not threshold the seed away before its first application. The data remain original; all subsequent candidates receive no seed exemption, permanent support, forced connection, or truth information.

## 6. Required measurements and scientific comparisons

### 6.1 Admission, source retention, and trajectory evolution

At every stage retain the guard status/reason, amplitude/error/S/N, centroid radius, widths/bound flags, fitted constant, baseline mask, proposed and final extra support, unavailable tests, exact candidate brightness, added area/brightness, candidate turnover, and PTC/MAP support/state summaries. Preserve complete empty outputs on source rejection rather than stale products.

Show histories of peak, physical centroid, gauge-safe shape/widths, constant, original frozen-template and fixed-aperture measurements, candidate size/brightness, and terminal drift. Use all stages for histories and summarize M0, T1, T3, T5, and T10. Keep raw brightness-area quantities and fitted-background contributions separate.

Reuse the sealed same-map saved-candidate screen. At each new REF T10, additionally evaluate all three policies on that **same map** as unapplied candidates, sharing its fit. This small cross-check separates selection differences on identical brightness from differences between trajectories whose cleaning has already diverged. No additional feedback follows from these cross-candidates.

For historical BASE versus BASE-POINT, explicitly report guard/fit/background/candidate changes and resulting response/seed differences. For CORR/DIAG versus BASE-POINT, use the matched new trajectories, not historical BASE as the sole comparator.

### 6.2 Compact and shoulder recovery

For policy h, signal I, and stage k, compare each injected branch with its **own policy REF**:

```text
Delta_h,k(I) = T_h,k(I) - T_h,k(REF)
```

Recover or construct the injection-only ordinary-MAP reference using that policy's REF forward support, coefficients, and normalization at the corresponding stage, bypassing PTC cleaning/centering. These controls are diagnostic projection/regridding, not new upstream science reductions. Reuse exact identical references when established; expose policy-dependent support changes.

Use the existing C+D unweighted projection gain g, total distortion eta, and orthogonal distortion rho, with their established definitions and domains. Preserve the original physical benchmark domain wherever supported; report common-support and native-domain comparisons separately when necessary. Do not hide unfavorable residuals with a new cut or redefine the benchmark around admitted pixels.

Report signed response and residual maps/profiles, compact positive/negative differences and the even finite-difference map, shoulder/wings and bowl/pedestal behavior, and fixed-template/aperture responses. These are finite full-procedure responses at the post-RTC/CAL boundary, not an isolated recovered sky component or an absolute flux calibration.

At T10 only, compute the ideal map-addition measurement controls for all three signal perturbations per policy/array: compare the measurement change in REF + injection-only reference with that in the actual injected branch. This is at most 27 logical ideal-addition controls, with exact reuse allowed. Use the same new full-map measurement/guard convention; preserve failures and active constraints. Do not fit signed difference maps with a positive-source Gaussian. Undefined or numerically negligible ideal increments yield unavailable ratios, not invented gains. Report absolute differences first.

### 6.3 Initial-model dependence

Compare SEED-SHOULDER against its **own policy REF** at matching stages in both total maps and replacement candidates. Track the shoulder-shaped projection, residual norm, aperture/background shifts, retained extra support, and remaining T9-to-T10 drift.

Place the data-injected SHOULDER response next to the seed-only response. The key question is whether the method recovers a shoulder present in the data without showing comparable persistence of a shoulder supplied only to the starting model. A persistent seed-induced difference demonstrates initialization sensitivity; it does not prove the unchanged observation physically lacks similar structure. No claim of distinct asymptotic fixed points from ten cycles.

### 6.4 Endpoint background and covariance applicability

On each completed policy/array REF T10 only, use the existing background companion and original background/source masks to characterize the background and, when supported, fit the existing compact covariance family once. There are at most nine logical endpoint analyses, with exact reuse allowed.

Report background scatter and directional/beam-scale correlations, model support/transfer qualifications, and how endpoint neighborhood-variance estimates compare with frozen C0 for matched usable weights. These are diagnostic model comparisons, not newly established source-location truth. Relate material noise-amplitude increases to the already completed 1.25 stress result without rerunning that stress campaign.

Never feed endpoint covariance back into scores, thresholds, source acceptance, or completed trajectories. No new Monte Carlo/refit ensemble, adaptive-noise simulation, or source-error recalibration. Label sigma_A from frozen C0 as the guard's conditional reference-scale uncertainty, not a complete uncertainty on the new endpoint. Do not attach old BASE/JINC scatters to new maps, add separate variances to infer uncertainty of paired differences, or claim a per-map null screen guarantees a ten-cycle false-feedback rate.

## 7. Bounded preflight and numerical controls

Reuse and extend existing tools/tests rather than constructing a parallel pipeline. Before accepting trajectory results, establish:

- **Authority and replay:** consumed parent hashes and exact frozen policies agree; existing original BASE replay evidence remains applicable. Verify the integrated guard/candidate path against the saved 15-map controls and recorded policy candidates under their established exact/tolerance rules. Do not run a redundant historical ten-cycle replay solely to restate existing evidence. An intervening change that invalidates that evidence blocks dependent observation execution until the demonstrated inconsistency is resolved within the existing bounded tests; do not silently add historical trajectories or relax tolerances.
- **Intervention isolation:** y_I − y equals the declared native projection on valid support; seed-only changes touch m0, not y; upstream invalidity/channel semantics are unchanged; each cycle restarts from the immutable parent.
- **Accounting and learning:** native subtract/restore, centering, normalization, and rejoin checks pass every cycle; each nonidentical residual receives fresh native PTC learning/application. Preserve basis/subspace/state diagnostics and actual support changes.
- **Selection and stopping:** published admitted brightness equals its measured operand; source rejection clears accepted state; supplemental evidence excludes K0/current baseline/anchor; seed exemption expires after m0; complete replacement can remove old support; unavailable inputs remain distinguishable from ordinary rejection; m10 is never applied.
- **Response controls:** existing no-op/known-response small fixtures and zero/signed-input accounting cover the integrated route, including a seed removed by later replacement. Verify policy-matched injection references and ideal-addition measurements without rerunning large simulation campaigns.

Run affected focused, successor/baseline, native/CLI, and required configuration checks as applicable. Record what was actually exercised. Independent preflight review must cover the realized method/interventions; final review must cover numerical accounting, scientific interpretation, and rendered figures. An independent code review is not an external observational replication.

## 8. Failure handling and branch stops

**Expected-source rejection is a defined scientific outcome, not a computational exception.** If a required full branch map fails the shared source rule, preserve that map and diagnostics, return the specified failed-fit message and empty accepted POINT/feedback products, record a source-retention miss, and stop that branch **before another feedback application**. This includes finite evaluation-budget exhaustion and a nonidentifiable amplitude. Do not continue with a stale candidate, use the seed to bypass the guard, retry with relaxed bounds, or reinterpret a fit failure as proof of physical absence.

Missing/corrupted inputs, unusable required covariance, non-finite required calculations, unsupported native projection, failed accounting, a new failed processing segment, or inherited feedback-boundary contact remain separately identified blockers. Preserve the inherited boundary-contact predicate and original handling of the known problematic segment; do not invent a stricter boundary test from harmless tails or conceal a changed processing state.

Freeze the failing state and retain completed diagnostics. Comparisons requiring an uncompleted REF or paired branch are unavailable beyond their common completed stages. Do not substitute a different policy's REF. A failed endpoint diagnostic or ideal-control measurement blocks that diagnostic, not an otherwise valid completed trajectory.

Low recovery, larger residuals, extra noise, seed dependence, no supplemental admissions, nonzero terminal drift, and a qualified but mathematically usable a2000 model are **results**, not invitations to tune or restart. Finish unaffected work. No changes to rank, threshold, gain, masks, covariance, fit bounds, signal shapes, or cycle count to improve an outcome.

## 9. Resources, evidence, and report

Use the existing validated environment. Inspect the available machine/allocation and use useful parallelism across independent arrays/policies/branches; do not inherit an arbitrary four-worker cap. Share immutable inputs and compact arrays, avoid nested numerical-library oversubscription, and record worker/thread settings and actual allocation. Existing Unity execution is permitted when needed, without environment migration or a broad optimization project.

Use initial REF work to check memory and execution costs before scheduling the remaining matrix; this is not permission to drop matrix cells silently. Record logical versus actual calculations, reused products, wall/CPU time, PTC learning/application, MAP/admission, endpoint analysis, serialization/verification, and measured memory scope. Do not add overlapping process durations into an elapsed-time claim or label local checks as Unity/Spack evidence.

Create a fresh bundle, for example:

```text
successor-point-152389-guarded-feedback-comparison-20260925/
```

Save the prospective bindings; exact parent, candidate and injection identities; commands; every stage's maps/measurements/candidates; support/state/accounting summaries; signed response controls; endpoint diagnostics; tests, reviews and reproduction instructions. Preserve failed attempts and verify consumed originals before and after execution. Reference existing immutable archives rather than copying raw data per branch.

Provide one owner-facing assessment with a complete 45-cell status matrix, compact numerical tables, and inspected matched-scale figures. State separately:

1. What changed from historical BASE to BASE-POINT, including non-membership changes that affect brightness or later processing.
2. Whether CORR or DIAG improves compact and shoulder response relative to BASE-POINT, and whether any apparent CORR benefit survives the DIAG comparison.
3. Whether data-supported shoulder recovery improves without greater seed-only persistence, background absorption, instability, or source loss.
4. Whether the frozen noise model remains a useful conditional scale at the endpoints, with observational qualifications explicit.

Show injection gains/distortions, sign asymmetry, shape/background/aperture behavior, seed persistence, endpoint drift, and admission/support changes separately for each array. Greater brightness or support alone is not success; smaller pixel RMS is not automatically better measurement precision. Similar CORR and DIAG results can mean that aggregation and empirical threshold calibration account for the effect rather than correlation handling specifically.

Do not invent a new scalar score or post-hoc pass threshold. The approximately 5% final calibration/PSF goals provide context, not automatic acceptance criteria for this partial-boundary experiment. An unfavorable, null, or qualified result is a complete outcome. Recommend one next bounded action only; do not execute it.

## 10. Stop boundary

Stop after this fixed comparison, assessment, required independent review, and evidence seal. Commit reviewed changes locally using established worktree practice; report final commit, worktree status, and bundle. No merge, push, destructive cleanup, default change, or production activation.

No fresh admission calibration or validation draws, JINC work, new neighborhood scales/thresholds, Gaussian/RBF/Wiener feedback reconstruction, changed sky interpolation, covariance-weighted source fitting, changed RTC/CAL/PTC recipe, damping, extra signal/seed branches, new observation, full adaptive uncertainty campaign, applied m10, or cycle 11 is authorized.

**Completion means a controlled answer about the benefit and risks of these fixed admission supplements in the bright POINT case—not proof of absolute flux recovery or permission to adopt a new method.**
