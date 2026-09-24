# Codex directive: POINT 152389 — bounded correlation-aware admission on ordinary MAP

**Owner:** Grant Wilson  
**Date:** September 24, 2026  
**Status:** NEW EXPERIMENTAL WORK ORDER — recover, implement, validate, execute the gated comparison, assess, independently review, and stop.  
**Scope:** POINT 152389; a1100, a1400, and a2000; ordinary MAP; one explicitly defined neighborhood-admission method and its diagonal-covariance control. No JINC feedback or production change.

> Change which measured pixels are admitted to the feedback model. Do not replace their brightness with a smoothed map, fitted source, noise prediction, or injected truth. Test additional evidence outside the bright core rather than letting the core supply that evidence.

## 1. Scientific question and new authority

The completed map-only JINC comparison did not demonstrate a consistent measurement-precision advantage for its selected gridding configuration. Retain ordinary MAP as the reference representation and leave the sealed JINC experiment unchanged.

The next question is whether spatially coherent, individually subthreshold brightness can be admitted more usefully when its neighborhood evidence is evaluated with the measured pixel correlations. The target remains this known-bright POINT source, not blind source detection or a general extended-emission reconstruction.

This order selects a deliberately simple first method: **positive, beam-scale neighborhood evidence, normalized using the full covariance, as a supplement to the existing threshold-3 source-connected admission**. A matched diagonal-covariance version separates correlation handling from the effects of neighborhood aggregation and the newly introduced noise-amplitude field. This is not a claim that the method is optimal, a covariance-weighted source fit, or an RBF reconstruction.

This order explicitly authorizes:

- A diagnostic covariance-to-admission route, the score and candidate rule below, null calibration/validation, saved-map screening, and the bounded new feedback trajectories.
- Fresh residual-specific PTC learning/application within those trajectories using the unchanged accepted PTC recipe. Changing admission changes later residuals; freezing the old bases would answer a different question.
- Reuse of the exact compact and shoulder signal operands and the first-candidate-only shoulder-seed robustness intervention specified below.

It supersedes earlier prohibitions on correlation-aware admission, new candidates, and new feedback **only for this campaign**. It does not modify frozen MAP/NOI/FLT/FRUIT contracts, activate an unapproved normative product, or authorize a new default. The new route is an explicitly named, diagnostic FRUIT policy. Existing components retain numerical and scientific ownership.

The choices in this order are the experimental selections. They are not inherited defaults or claims of previous authorization. Do not return for another parameter-selection loop when a choice is supplied here. A failed gate is a completed negative screening result, not permission to tune the method until it passes.

## 2. Recover and bind the established experiment

Read applicable AGENTS.md, owner decisions, current worktree status, MAP+/PTC/AST/NOI/FLT/FRUIT boundaries, and the accepted experimental measurement/admission implementation. Recover these work orders and their completed handoffs, assessments, reviews, and manifests:

1. `CODEX_POINT_152389_RESPONSE_AND_ROBUSTNESS_CD.md`.
2. `CODEX_POINT_152389_FULL_OBSERVATION_ADDITIVE_UNCERTAINTY.md` and `CODEX_POINT_152389_POST_FRUIT_NOISE_AND_UNCERTAINTY_AB.md`.
3. The original one-cycle and ten-cycle POINT authorizations and results.
4. `CODEX_POINT_152389_MAP_ONLY_JINC_FULL_RUN.md` and its completed, sealed comparison. Use this as evidence and context, not as a new execution order.

Discovery roots under `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/` include:

```text
successor-point-152389-fruit-one-cycle-20260923/
successor-point-152389-fruit-ten-cycles-20260923/
successor-point-152389-additive-uncertainty-20260923/
successor-point-152389-post-fruit-ab-20260923/
successor-point-152389-response-robustness-cd-20260923/
successor-point-152389-map-only-jinc-full-20260924-r4/
```

Paths and the reported JINC closeout commit `07fff9594` are discovery hints, not substitutes for authenticated identities. Verify consumed originals against their archived hashes, not hashes newly declared authoritative from whatever occupies an old path. Preserve all older bundles and unrelated worktree changes.

Recover the complete original threshold-3 candidate definition: brightness/background convention, eligibility, source anchors, connectivity, handling of failed fits, validity, noise scale, and projection. Do not approximate it from this document's shorthand. Save a compact method binding that distinguishes inherited choices from this order's new choices.

The reference remains ordinary-MAP **threshold 3, unity gain, complete replacement**, with direct containing-pixel model projection, the original coefficients and eligibility, and the original frozen admission-noise scale. Retain the original per-network/processing-segment observed-entry rank-10 PTC recipe, centering, learning, stopping rules, and application semantics. Retain the inert paired-r behavior; no r-channel injection or new treatment.

The feedback domain remains the AST-centered **40-arcsecond-radius disk** on the original **2-arcsecond grid**. Outside-domain feedback remains absent; this does not remove observed samples or assert zero sky. Valid excluded pixels inside the disk remain intentional model zeros. Unavailable pixels remain unavailable.

Keep the original seven-parameter descriptive Gaussian-plus-constant measurement, its footprint/search/initialization/weights/bounds/failure rules, frozen-template projection, fixed apertures, background conventions, and mJy/nominal-beam units. Covariance does not become new POINT fitting weights or MAP coefficients.

Carry forward the a1100 network-3/scan-5 condition and the a2000 background-model/source-transfer qualifications. Do not repair them here. Qualified, mathematically usable covariance supports a qualified diagnostic experiment; perfection is not a prerequisite.

## 3. Freeze a first-pass covariance and the core exclusion

### 3.1 Covariance source and scope

Use the existing compact map-domain family on the **unperturbed ordinary REF M0**:

\[
C_0(p,q)=s_0(p)s_0(q)\,\xi_0(p-q).
\]

Reuse an authenticated original M0 companion realization with the original source/bowl exclusion and its recorded model settings. If the complete realization was not retained, this order authorizes reconstructing it or fitting it once per array using the existing companion algorithm and those original masks/settings. Record which occurred. Do not substitute a T10 covariance, a JINC covariance, a JINC-enlarged exclusion, an arbitrary diagonal model, or a source-fit peak error merely because it is easier to access.

Bind the actual amplitude field, directional correlations, supported lags, pair counts, source-depth transfer, PSD treatment, and generation method. Missing covariance is unknown, not zero correlation. Preserve negative correlations. Establish a coherent joint field over all source-fit, candidate, and neighborhood pixels used together.

Freeze this M0 covariance **for all policies, signal branches, and feedback stages in this campaign**. There is no new covariance learning inside the feedback loop. This isolates admission policy and prevents the new trajectories from changing their own evidence scale. It is a conditional prototype; it does not assume that the true noise remains identical through T10.

The original threshold-3 baseline continues to use its original frozen admission-noise scale. Only the supplemental neighborhood calculation uses C0. The diagonal control uses exactly `diag(C0)` in its score, not a separately fitted amplitude field.

### 3.2 Fixed core exclusion

For each array, define `K0` as the authenticated original unperturbed REF M0 threshold-3 **source-connected admitted mask**. This is a fixed exclusion from supplemental evidence, not a mandatory feedback mask and not an assumed boundary of the true source.

Freeze K0 before inspecting new policy scores. Use it unchanged for both new policies and every branch. Do not derive it from the shoulder, a perturbed map, T10, a favorable new fit, or a newly expanded candidate.

At every stage also exclude that stage's current ordinary threshold-3 admitted mask from supplemental evidence. Thus neither the original protected core nor currently strongly admitted pixels can directly supply the neighborhood brightness that admits additional pixels.

## 4. Exact supplemental policy: CORR-NEIGHBOR-v1

### 4.1 Unchanged baseline calculation

On each input map, run the existing measurement and threshold-3 candidate procedure once. Let:

- `D` be its valid feedback domain;
- `B` be its source-connected threshold-3 admitted mask on this map;
- `b(p)` be the exact signed brightness operand that the existing builder would publish at pixel p before selection, including its existing background convention.

Preserve the original BASE candidate exactly. In particular, do not redefine b from a new Gaussian, introduce another background subtraction, or use the fit amplitude in place of pixel brightness. If the parent procedure cannot establish a valid source association/candidate, do not fabricate one from a neighborhood peak. Preserve the parent failure/absence outcome and apply no supplemental feedback.

### 4.2 One fixed angular scale

Use the inherited nominal beam conventions solely as selected neighborhood scales:

| Array | Neighborhood Gaussian FWHM B_a | Full circular neighborhood radius |
|---|---:|---:|
| a1100 | 5 arcsec | 5 arcsec |
| a1400 | 7.5 arcsec | 7.5 arcsec |
| a2000 | 10 arcsec | 10 arcsec |

These are not new PSF measurements or calibrations. No scale sweep, orientation fit, multiscale maximum, or tuning to the shoulder is authorized.

For every original-grid anchor p in `D \ (K0 union B)`, define the full circular footprint `F_p = {q : distance(p,q) <= B_a}` in the physical tangent coordinates. Its complete original-grid footprint must lie in the valid feedback domain and have the required covariance. Do not truncate it at the domain edge, fill missing values, or shrink it around unavailable pixels to make a test possible. A locally unavailable supplemental test does not invalidate the unchanged BASE candidate.

Within a supported full footprint, define deliberately masked weights:

\[
w_{pq}=\exp[-4\ln(2)\,d(p,q)^2/B_a^2]
\;1_{q\ne p}\;1_{q\notin K0\cup B},\qquad
q_p=w_p/\sum_q w_{pq}.
\]

The center pixel is excluded: the proposed addition must be supported by **neighboring** brightness, not by counting its own fluctuation again. K0/B exclusions are the specified policy, not missing-data filling. Retain negative neighbor brightness in the sum; do not preselect positive neighbors.

Require positive finite normalization and

\[
n_{\rm eff,weights}=1/\sum_q q_{pq}^2\ge3.
\]

This is only a weight-concentration check, **not** a number of statistically independent pixels. Record its value. No other adaptive footprint/weight rule is allowed.

### 4.3 Score and diagonal control

For the same map-dependent weights, compute:

\[
e_p=q_p^T b,\qquad
v_{p,\rm CORR}=q_p^T C_0 q_p,\qquad
v_{p,\rm DIAG}=q_p^T\operatorname{diag}(C_0)q_p,
\]

\[
Z_{p,h}=e_p/\sqrt{v_{p,h}},\qquad h\in\{\rm CORR,DIAG\}.
\]

Both variances must be numerically supported and positive for the corresponding test. Use the existing covariance regularization only; do not repair an invalid variance with a hidden floor. No covariance inversion or inverse-covariance weighting is selected here.

These are **nominal neighborhood evidence scores**, not automatic Gaussian significances. B is data-dependent, and b may contain an estimated background. Consequently qCq alone does not describe the entire distribution after fitting and selection. The complete-procedure simulations below must repeat those operations; do not freeze the trial's fitted background at truth or treat its error as independent.

The diagonal policy differs only in this variance calculation and its separately calibrated threshold. In all simulations the true generating covariance remains full C0 for **both** policies. Simulating independent pixels for the diagonal control would change the experiment.

### 4.4 Admission and brightness publication

For policy h with its frozen threshold tau_h, proposed extra pixels satisfy all of:

- p is in `D \ (K0 union B)` and its supplemental test is available;
- the original brightness operand `b(p) > 0`;
- `Z_p,h > tau_h`, with a strict comparison.

Form `B union E_h` and retain only its components connected to the original source anchors using the inherited connectivity rule. Do not automatically include separate pointwise-threshold components that were absent from B. If B is empty, add nothing. Compute all scores from the same unmodified input map and original B **before** any new pixel is added. There is no within-map iterative flood fill, dilation, borrowing from newly admitted pixels, or forced bridge across a gap.

Publish exactly the original b(p) on the resulting mask, with unity gain and the original zero/unavailable semantics elsewhere. The neighborhood mean, fitted Gaussian, score, covariance, and synthetic truth are never feedback brightness.

At a given map, the new mask is a superset of B. Across feedback stages, regenerate the whole candidate from the new map and replace the previous candidate completely. Never union supports across stages. This is deliberately a **supplemental** test; it is not a replacement/rejection policy for pixels already admitted by BASE.

## 5. Calibrate and validate before any new feedback trajectory

### 5.1 Two declared null scenes

For each array use two frozen map-domain means on the joint source-fit/evidence domain:

1. **Blank:** the original REF M0 fitted constant, with zero astronomical brightness.
2. **Core only:** that constant plus the original REF M0 fitted noiseless Gaussian evaluated only on K0, exactly zero astronomical brightness outside K0.

The sharply truncated core is an intentional contamination-control scene, not a physical optical PSF. It tests whether the bright admitted core, fitting, or selection can manufacture supplemental support where this synthetic truth has no source. It is not the real source model used for feedback.

Generate new correlated Gaussian fields from C0 and add them to these noiseless means, not to an already noisy observed map. The same field is deliberately reused across the two scenes and the three policy evaluations for a paired **simulation** comparison. This supplies known simulation pairing, not the physical cross-covariance of observed adaptive maps.

Re-run the unchanged source measurement, background handling, BASE selection, weight construction, supplemental score, and connectivity as applicable. Share a required fit between policy evaluations on the same input map; do not perform redundant copies. Finite constrained fits are retained and flagged. A legitimate no-source/no-candidate outcome on the blank scene is not a fabricated detection; distinguish it from a solver or numerical failure. Do not replace failed draws.

### 5.2 Selected counts, seeds, and threshold

Use exactly **1,024 calibration fields and 1,024 independent validation fields per array**. Both null scenes reuse each field. The maximum is 6,144 independent joint noise fields across three arrays, not 12,288 independent fields. No trial runs RTC, CAL, PTC, MAP accumulation, or a feedback trajectory.

Use the existing verified generator, freeze its algorithm and version, and derive each seed from the first 128 bits, interpreted big-endian, of SHA-256 of the UTF-8 string:

```text
POINT152389-CORR-NEIGHBOR-v1|<array>|<calibration-or-validation>|<zero-based-trial-as-4-digits>
```

Record actual seeds and generator state conventions. Worker order and Python's randomized hash must not affect draws. Regeneration of a draw is not an additional independent trial.

For each calibration field and each policy, record S as the maximum score over eligible supplemental anchors with positive b(p) in either null scene, before the connectivity restriction. For the core scene consider only anchors outside K0; for the blank scene all potentially supplemental anchors are null. Empty eligible sets contribute minus infinity. Apply the same source-absence/failure rules in calibration, validation, and real selection.

Select the per-array/per-policy threshold:

\[
\tau_h=\max(3,S_{(1015)}),
\]

where `S_(1015)` is the 1,015th smallest of the 1,024 calibration maxima. This is the order-statistic choice `ceil((1024+1)*0.99)`, with no interpolation or fitted tail. It targets a **1% model-based per-map familywise exceedance rate for the supplemental search**, jointly over the two declared scenes. It does not target the false-admission rate of the inherited BASE rule.

Freeze thresholds before opening validation results or inspecting real CORR/DIAG admission maps. The floor of 3 is a selected score floor, not a claim that this is a three-sigma test. Do not lower the threshold because the shoulder is not admitted.

### 5.3 Independent validation and launch gate

On the 1,024 validation fields, record the same combined pre-connectivity exceedance event `S > tau_h` and the actual final supplemental false-admission event. The former conservatively screens the latter under these declared scenes. Also record BASE false support, total candidate support, additional pixel count/area, extra brightness-area sum, and the positive selection bias introduced by retaining measured brightness.

For k pre-connectivity exceedances, calculate the exact one-sided 95% binomial upper bound U. For k < 1024 this is the solution to

\[
\Pr_{\mathrm{Binomial}(1024,U)}(X\le k)=0.05;
\]

for k = 1024 set U = 1. Independently check the zero-event case `U = 1 - 0.05^(1/1024)`.

**A policy/array may enter new feedback only when:**

- Input/candidate replay and numerical fixtures pass;
- its covariance/evidence calculations are mathematically usable;
- all required core-scene fits and simulation calculations have complete, valid outcomes, retaining finite constraint flags;
- `U <= 0.02`; and
- the noiseless core-only and blank controls add no supplemental false pixels.

The 2% upper-bound screen is a selected finite-sample launch tolerance around the 1% calibration target, not proof that the true rate is at most 1%. Genuine source absence on a blank trial is a recorded no-candidate outcome; numerical failures are not silently recoded as absence. Unexplained failures make the affected gate unavailable.

Do not retune/recalibrate after a failed validation gate, draw until success, replace failures, or relax thresholds. Finish other unblocked cells and report the failure. A mathematically valid a2000 model may pass **with its observational qualification**; passing synthetic controls does not validate its source-depth extrapolation.

These controls are conditional on C0 and the fixed scenes. They do not establish a posterior probability for individual real pixels, a false-discovery guarantee on arbitrary skies, or a 1% false-feedback rate over ten adaptive cycles. Repeated use of the same observation is not independent evidence.

### 5.4 One bounded noise-amplitude stress check

On exactly the first 256 validation fields, also multiply the noise field by **1.25** and repeat the two scenes and policies using the frozen C0 scores and thresholds. Refit/reselect normally. This is one preselected covariance-amplitude misspecification check, not 256 new independent draws and not a new threshold calibration.

Report its false-admission/support/bias changes. It is a sensitivity result, not an additional launch gate or a reason to adapt the noise scale. Do not add correlation-length, non-Gaussian, mixture, or amplitude sweeps.

## 6. Saved-map screen and exact baseline replay

Before new policy trajectories, apply BASE, CORR, and DIAG as **unapplied diagnostics** to the authenticated original REF maps M0, T1, T3, T5, and T10 for all three arrays. Use the same frozen thresholds/C0/K0 throughout. Show B, extra proposed pixels, final connected mask, score/variance ratio, unavailable tests, and brightness assigned to extra pixels. Compare on the exact same map: this isolates selection from later changed cleaning.

This screen must not alter thresholds, scales, masks, or run selection. No supplemental pixels is an informative result, not a reason to enlarge neighborhoods.

Reproduce one complete original BASE REF M0–T10 trajectory per array using the current matched implementation. Require agreement with the archived maps, measurements, admission membership/brightness, applied operands, and upstream state under the original exact/tolerance rules. Do not weaken these rules. Replay original saved candidate calculations for the archived injection/seed trajectories used in comparisons; authenticate their full histories. Do not rerun the entire old C+D campaign.

A failed BASE replay blocks dependent new trajectories for that array. Correct demonstrated implementation defects, preserve failed evidence, and revalidate affected work. Do not fix replay by changing science settings or original products.

## 7. Bounded closed-loop comparison

### 7.1 Fixed trajectory matrix

For every policy/array passing its launch gate, execute these five trajectories with **ten feedback cycles**:

| Branch | Immutable pre-PTC data | Initial candidate | Subsequent candidates |
|---|---|---|---|
| REF | Original y | Its own policy candidate from M0 | Same policy |
| CORE-PLUS | y plus the exact archived positive compact operand | Its own policy candidate from its M0 | Same policy |
| CORE-MINUS | y plus the exact archived negative compact operand | Its own policy candidate from its M0 | Same policy |
| SHOULDER | y plus the exact archived shoulder operand | Its own policy candidate from its M0 | Same policy |
| SEED-SHOULDER | Original y; no data injection | Its own REF m0 plus the exact archived shoulder raster | Same policy starting with m1 |

Run this matrix for **CORR and DIAG**, with their separately frozen thresholds. Historical BASE counterparts remain the reference; only the three BASE REF trajectories are newly replayed in full.

The ceiling is therefore **30 new-policy trajectories plus 3 BASE REF replay trajectories**, at most **330 feedback applications**. M0 is not a feedback cycle. Reuse authenticated, mathematically identical M0 products and shared required fits rather than recomputing them redundantly. Count logical trajectories separately from actual native calculations and document exact reuse.

Execute REF before its dependent signal/seed comparisons. Independent arrays/policies/branches may run concurrently. A failed policy gate does not block a separately valid policy/array; mark the resulting comparison limitation explicitly. Do not add a substitute policy, source, scale, or trajectory.

### 7.2 Signal and seed boundaries

Reuse the exact C+D raster operands, normalization/taper, physical axes, and native containing-pixel projection. The compact tests are the saved +/-10% increments; the shoulder is the saved +5% broader/offset increment. These remain regression tests, not policy training targets. Do not redraw them from rounded fit values.

For data-injection branches, apply the archived operand coherently to the relevant valid pre-PTC entries, including source-bearing entries used by PTC but not final MAP contributors. Preserve invalid entries and native support. Do not introduce an optical response, extra beam convolution, calibration factor, or r-channel injection.

For SEED-SHOULDER, the data remain original. Add the archived shoulder raster to **that policy's own REF m0**, not automatically to the old BASE m0. This explicitly permits the same first-candidate-only bypass of admission used in the earlier robustness experiment. Subsequent candidates have no exemption or forced shoulder support. Compare each seed trajectory with its own policy REF to isolate initialization dependence. This control tests whether feedback can preserve an unsupported seed; it is not a recovered real signal.

### 7.3 Native recurrence

At each cycle begin again with the branch's immutable y or y+PI. Subtract only its current candidate, freshly learn and apply the unchanged PTC recipe to that residual, restore exactly the subtracted operand, and construct the ordinary MAP using native mean/mask/support rules.

Do not reuse the old REF basis sequence, freeze old means, subtract from a prior cleaned residual, accumulate candidates, add an injection repeatedly, or hand the known injected shape to the source fitter. PTC values may legitimately differ because the new candidate changes the residual; that is a downstream consequence, not authorization to change the PTC method.

Save every map, fit, candidate, score/support summary, and applied-operand identity. T10 uses m9. Save each m10 **unapplied**. No cycle 11. Run all ten cycles unless the branch becomes numerically/scientifically unavailable under the inherited failure rules; small drift is not permission to stop early and persistent drift is not permission to continue.

## 8. Measurements and inference boundaries

### 8.1 Admission and feedback behavior

At every stage report the ordinary B support, added support, unavailable neighborhood tests, connected components, retained/additional brightness, candidate changes relative to the prior stage, and any changing upstream support/learning outcomes. Track positive-only selection bias and how new support survives or disappears in later complete replacements.

Distinguish same-map policy differences from differences between already diverged trajectories. A larger selected region or a brighter final map is not itself a benefit. A stable mask with evolving brightness is not proven convergence.

### 8.2 Paired finite response

For every policy h, injection I, and stage k, calculate

\[
\Delta_{h,k}^{I}=T_{h,k}^{I}-T_{h,k}^{\mathrm{REF}}.
\]

Never subtract the old BASE REF from a new-policy injected map and call that the new policy's injection response. Construct/reuse the injection-only ordinary-MAP reference with that policy's REF operator/support at the corresponding stage, bypassing cleaning, following the established C+D rules.

Retain the original physical benchmark domain wherever supported and separately label common/native-domain results. Compute the existing projection g, total distortion eta, and orthogonal distortion rho; preserve signed core/wing/bowl residuals and positive/negative nonlinearity. Do not fit signed difference maps with the positive-source Gaussian fitter.

Use the unchanged full-map peak, centroid, shape tensor, frozen-template and fixed-aperture measurements. Preserve fitted-constant and raw brightness-area effects separately. At T10, compute only the required ideal map-addition controls for the three perturbations per new policy/array; reuse exact authenticated controls when truly identical. No unrelated historical or redundant control fits.

The comparison between CORR and DIAG is essential. Similar results can mean that aggregation, rather than spatially varying correlation handling, accounts for the effect. Under spatially uniform correlation inflation, independent null calibration can substantially cancel a global score rescaling. That is an informative limitation of this selected method, not a defect to tune away.

### 8.3 Seed robustness

For each policy compare SEED-SHOULDER against its own REF through T10. Report how much seed-induced structure and measurement shift remain, including their sign, shape, background contribution, and terminal drift. A method that improves the data-injected shoulder while retaining an equally unsupported initial seed has not established reliable recovery.

### 8.4 Background and uncertainty limits

On each completed new-policy REF T10 only, use the existing companion and frozen original background masks to characterize the endpoint background and, where supported, fit its existing compact covariance family once. Compare it with C0 and report neighborhood-scale variance changes/source-transfer qualifications. These are endpoint diagnostics; do not feed them back or revise completed trajectories.

Do not run a new source-flux Monte Carlo ensemble. Do not attach old BASE/JINC peak scatters to the new endpoints. Do not infer uncertainty on paired real-map differences by adding separate variances, or describe covariance-qualified neighborhood scores as calibrated astronomical significances.

This work gives finite response, method sensitivity, conditional null behavior, and feedback-stability evidence. It does not establish a full adaptive uncertainty law, calibrated flux correction, absolute photometry, or validation of the final approximately 5% calibration/PSF goals. Those goals provide context, not automatic pass/fail thresholds for this partial-boundary screen.

## 9. Validation, ownership, and resources

Use the existing candidate, companion, native PTC/MAP, trajectory, measurement, and reporting code. Put the diagnostic policy at the FRUIT selection boundary. Do not turn MAP into a detector, the descriptive POINT fit into reconstruction, FLT into feedback ownership, or NOI covariance into sample weights. An internal neighborhood statistic is not permission to publish a normative filtered-map or FRUIT parent product.

Required focused tests include:

- Exact BASE candidate replay; same-map supplemental masks cannot remove BASE pixels; published admitted brightness equals original b exactly.
- No contribution from K0, current B, or the anchor pixel to neighborhood evidence; negative neighbor brightness is retained; no sequential growth within a candidate.
- Physical scale/axis/edge rules, missing support/covariance, weight concentration, empty B, source-association failure, and disconnected proposed islands.
- qCq against a direct manageable pixel-pair calculation and generated-field variance for fixed test rows. In a genuinely diagonal covariance fixture, CORR and DIAG scores/candidates must agree under identical calibration.
- Full-correlated generating noise for both policies; separate calibration/validation streams; exact order-statistic/tie handling; binomial upper-bound boundary cases; deterministic regeneration.
- Estimated-background and constraint behavior across complete null trials; no replacing failed draws or confusing no-source outcomes with numerical errors.
- Complete replacement, exact restore/subtract accounting, fresh branch-specific learning, first-candidate-only seed exception, and preservation of unapplied m10.

Run affected successor, baseline, native/CLI and required configuration checks, plus independent scientific/numerical preflight and realized-result/figure review. A synthetic gate is not independent observational validation; a fresh-context code review is not an external replication. State what was actually checked.

Use useful parallelism across independent arrays, branches, and small simulations with bounded memory and deterministic publication. Avoid nested BLAS/OpenMP oversubscription and do not inherit a four-worker cap without inspecting the current machine. Prefer compact local covariance calculations and reuse joint noise generation/required fits. No observation-wide dense sample covariance or per-sample provenance expansion.

Use the existing validated local environment when adequate. If resources require Unity, this order permits a bounded execution using the established environment, with requested cores/memory and actual allocation recorded; it does not authorize dependency migration or a general optimization campaign. Distinguish local checks from actual Unity/Spack evidence. Report wall/CPU time, phase costs, worker/thread counts, and measured memory scope without adding overlapping durations into an elapsed-time claim.

## 10. Outputs, review, and explicit stop

Write a fresh bundle, for example:

```text
successor-point-152389-correlation-admission-20260924/
```

Before calibration, freeze this directive, exact code/binary/environment and parent bindings, original method, C0/K0/masks, scale and neighborhood rules, seeds, counts, threshold/gate rules, trajectory matrix, injection/seed operands, measurement rules, and original numerical tolerances. Freeze actual calibrated thresholds before validation and real policy screens.

Deliver one compact owner-facing assessment with an explicit matrix of completed, qualified, unavailable, gate-blocked, and unrun cells. Include:

1. What the new rule actually admitted and whether the bright core was excluded from its evidence.
2. Conditional null and amplitude-stress results, including false support, extra brightness bias, constraints/failures, and covariance applicability.
3. Same-map BASE/CORR/DIAG selection comparisons and full-trajectory differences.
4. Compact +/- response, shoulder response, seed persistence, background changes, and terminal drift, separately for every array.
5. Whether any benefit survives the diagonal control, and what can and cannot be attributed specifically to correlations.
6. One justified next action, recommended only. An unfavorable or null result is a complete scientific outcome; no production winner is required.

Provide legible matched-scale figures for source/admission masks, neighborhood evidence and support, null-score/false-admission results, response/residuals, seed persistence, trajectory drift, and endpoint background. Use explicit signed scales and disclose rescaling. Inspect rendered figures. Preserve trial-level outcomes, fit flags, exact candidates/parents, provenance, timing, tests, reviews, and regeneration commands. Report consumed-original verification before and after execution and retain failed attempts without modifying historical evidence.

Commit reviewed changes locally under the established worktree practice. Do not merge, push, reset, discard unrelated changes, or activate production. Report final commit/worktree status and the sealed bundle.

**Stop after this fixed experiment or its explicitly failed gates.** No JINC work, RBF/Wiener reconstruction, covariance-weighted source fitting, new sky-brightness interpolation, automatic threshold/scale optimization, revised PTC rank/segmentation/learning method, new observation, additional signal shapes, extra independent noise draws, full adaptive-noise ensemble, m10 application, cycle 11, or production/default change is authorized.

Completion means a controlled answer about this one conservative neighborhood-admission supplement on the known-bright POINT case. It does not mean that more admitted pixels are better, the shoulder is solved, synthetic false-alarm calibration is observational truth, or all correlation-aware methods have been tested.

---

### Method references and authority note

The inherited scientific authority is the authenticated local work orders and contracts named above; this document supplies the new bounded experimental choices. Relevant statistical context includes Vio & Andreani (2016), *On the Correct Estimate of the Probability of False Detection of the Matched Filter in Weak-Signal Detection Problems*, arXiv:1602.02392, for the distinction between a fixed-location score and a search false-alarm rate; and the NIST/SEMATECH e-Handbook, section 7.2.4.1, for exact binomial confidence bounds. The present method is not that paper's matched filter or peak-distribution implementation. Its searched, data-dependent decision is calibrated by the complete declared simulations instead of assigning a Gaussian tail to Z.
