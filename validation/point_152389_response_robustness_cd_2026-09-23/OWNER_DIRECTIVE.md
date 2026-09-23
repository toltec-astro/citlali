# Codex directive: POINT 152389 source-response and feedback-robustness screen

**Date:** September 23, 2026  
**Scope:** Steps C and D, bounded to POINT 152389 and arrays a1100, a1400, and a2000.  
**Execution:** Implement, validate, run the prescribed experiments, assess, and stop. This is not a request for another proposal.

## 1. Objective and new authorization

The completed ten-cycle FRUIT trajectory and post-FRUIT A+B analysis are the reference. We now want to distinguish **recovery of a known added signal**, **dependence on the initial feedback model**, and **dependence on an admission-method choice**.

**C — Source response:** Perturb the immutable saved post-RTC/CAL, pre-PTC data by a known compact or weak broader/asymmetric signal. Recompute the initial map and complete ten-cycle trajectory. Measure how the reconstructed result changes relative to a matched unperturbed trajectory.

**D — Feedback robustness:** Keep the observed data unchanged. Separately perturb the first feedback model, and separately compare one more restrictive admission threshold. Determine whether the ten-cycle endpoints and intermediate trajectories depend materially on those choices.

This directive explicitly authorizes the pre-PTC injections, PTC/FRUIT replays, seed interventions, and single threshold comparison specified below. It supersedes the previous A+B prohibition on those activities **only within this new diagnostic campaign**. It does not authorize a new reference result, default admission policy, flux correction, or production activation.

These are finite perturbation and robustness experiments on one observed dataset. They are not repeated-observation noise ensembles, a full adaptive-FRUIT uncertainty calibration, or an absolute astronomical flux-recovery test. The injection boundary excludes unknown upstream RTC/CAL and optical response. Useful qualified results are sufficient; resolving the a2000 background-model problem is not a prerequisite.

## 2. Recover the exact reference; do not reinvent the method

Read applicable AGENTS.md, current owner decisions, and relevant MAP+/PTC/POINT/FRUIT contracts and method bindings. Recover the first-cycle authorization, ten-cycle handoff, and post-FRUIT A+B directive, assessment, and independent review. Preserve the accepted experimental seven-parameter Gaussian-plus-constant measurement; do not replace it with a different draft POINT method.

Discovery hints under `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/` are:

- `successor-point-152389-fruit-one-cycle-20260923/`
- `successor-point-152389-fruit-ten-cycles-20260923/`
- `successor-point-152389-post-fruit-ab-20260923/`
- The immutable MAP/PTC parents and admission products bound by those bundles.

Recover `handoff/POINT_152389_FRUIT_TEN_CYCLES_2026-09-23.md`, `handoff/POINT_152389_POST_FRUIT_AB_2026-09-23.md`, and `CODEX_POINT_152389_POST_FRUIT_NOISE_AND_UNCERTAINTY_AB.md` where available. Paths are discovery hints, not authority to select inputs. Bind actual source revisions, input identities, and hashes.

Reference sanity checks are T10 peaks **937.421686 / 664.734078 / 813.617700 mJy/nominal beam** for a1100/a1400/a2000. Recover full saved measurements rather than treating these rounded values as replacement references.

Hold the existing recipe fixed except for the explicitly identified branch intervention: post-RTC/CAL parents; AST coordinates; direct containing-pixel model projection; per-network/processing-segment observed-entry rank-10 PTC, including its fresh centering and learning rules; ordinary forward MAP; uniform-coefficient and eligibility-resolution rules; unity feedback gain; complete replacement rather than accumulated candidate support; existing fitting procedure; and original frozen admission-noise scale. Paired-r behavior remains as in the parent, including its inert status in this experiment. Do not invent r-channel source injection or new r treatment.

Reference admission remains threshold **3**. Feedback is absent outside the AST-centered **40-arcsecond-radius disk**, without dropping observed samples or asserting zero sky there. Valid excluded pixels inside the disk remain intentional model zeros; unavailable pixels remain unavailable.

Preserve the original M0–T10 products and original unapplied m10. Track the inherited a1100 network-3/scan-5 condition without repairing it. The A+B peak scatters **7.91 / 5.68 / 6.63 mJy/nominal beam** remain contextual conditional calculations with their original qualifications; they are not new admission scales or uncertainties of paired differences.

## 3. Fixed run matrix and campaign bound

Before numerical response runs, write a manifest containing the exact shapes, interventions, seeds, tolerances, run order, and input hashes. Do not tune the matrix after seeing results.

Execute the following eight logical trajectories per array, with at most ten feedback cycles in each trajectory:

| ID | Change to pre-PTC data | First feedback candidate | Subsequent admission |
|---|---|---|---|
| REF | None | Ordinary candidate from ordinary M0 | Threshold 3 |
| C-core-plus | Add the +10% compact increment | Regenerate normally from perturbed M0 | Threshold 3 |
| C-core-minus | Add the −10% compact increment | Regenerate normally from perturbed M0 | Threshold 3 |
| C-shoulder | Add the prescribed weak broad/offset increment | Regenerate normally from perturbed M0 | Threshold 3 |
| D-seed-low | None | 0.8 × original m0 on its original support | Threshold 3 |
| D-seed-high | None | 1.2 × original m0 on its original support | Threshold 3 |
| D-seed-shoulder | None | Original m0 plus the prescribed shoulder | Threshold 3 |
| D-threshold-5 | None | Regenerate from original M0 at threshold 5 | Threshold 5 |

There is **no C×D factorial matrix**, no further amplitude/threshold sweep, and no new observation or blank-field campaign. Execute REF first, then C, then D. Unexpected scientific sensitivity is a result to report, not a reason to add runs or tune the method.

The cap is eight logical trajectories per array: one reference, three data perturbations, three seed perturbations, and one method comparison. Each branch includes its initial map M0 and cycles T1–T10; M0 is not a feedback cycle. Reuse the unchanged M0 for D branches. A newly reproduced REF serves all paired comparisons and the zero-perturbation control. No separate redundant full-length zero run is required.

Run all ten cycles unless a branch becomes invalid under the stop rules below. A small endpoint change is not permission to stop early; lack of convergence is not permission for cycle 11. Use branch-specific names so none of these experimental T10 products can be mistaken for the original reference.

## 4. Define the injected signals and their boundary explicitly

### 4.1 Freeze two test shapes

For each array, recover the original T10 fitted peak A, centroid, two principal widths, and orientation, and the original M0 major FWHM W0. Construct test shapes on the exact reference candidate grid in its physical tangent coordinates. Exclude the fitted constant.

**Compact shape G:** The unit-peak analytic Gaussian with the saved T10 centroid, widths, and orientation. It is a convenient test signal, not a replacement for the empirical feedback model and not a claim about the true PSF.

**Shoulder shape H:** A unit-peak Gaussian with twice both T10 principal widths, the same orientation, and its centroid shifted by **0.5 W0 along the positive first physical tangent-coordinate axis**. Bind that physical axis and its handedness from AST/WCS, not array-index or plotting conventions.

Apply the same explicit compact-support taper to both shapes. Let r be distance from the AST analysis center, in arcseconds:

- w(r) = 1 for r ≤ 30;
- w(r) = [1 + cos(π(r−30)/6)] / 2 for 30 < r < 36;
- w(r) = 0 for r ≥ 36.

Define the actual raster operands as:

\[
I_+ = +0.10 A\,G w,\qquad
I_- = -0.10 A\,G w,\qquad
I_h = +0.05 A\,H w.
\]

The amplitudes refer to the analytic pre-taper Gaussian peaks. Save the actual discrete peak, area-weighted sum, geometry, and any taper loss; do not silently renormalize after rasterization/tapering. The shoulder has low peak brightness but need not have a negligible area integral. This is one deliberately broader, asymmetric increment, not an extended-emission validation campaign.

The taper gives these **test operands** deliberate zero support outside 36 arcseconds; it does not alter the 40-arcsecond feedback domain or any observed-data eligibility. Inspect input geometry and support before execution. If the prescribed shape overlaps unavailable needed pixels, mark the affected test unavailable rather than silently filling, clipping, moving, or rescaling it. Do not enlarge the analysis domain.

### 4.2 Enter before PTC learning, not at the final map

Let y be the immutable saved calibrated pre-PTC parent, and P the exact previously authorized direct containing-pixel projection. For an injection branch form:

\[
y^{I}=y+P I.
\]

This directive authorizes treating I as a known additive brightness operand **at that saved post-RTC/CAL boundary**, using the existing mJy/nominal-beam convention. Do not add another beam convolution, filter, calibration factor, response correction, or physical detector model.

Audit the projection domain. Apply the operand consistently to valid parent entries participating in PTC learning/application, including relevant entries that are not final MAP contributors. Do not inject only map-admitted centers while silently leaving other valid source-bearing PTC entries unchanged. Preserve all invalid-entry and saved-support semantics; an injection must not make unavailable data valid. Missing necessary AST/projection information is a specific unavailable test, not permission to invent positions or call an incomplete injection a full pre-PTC response.

The same operand is projected coherently into the contributing networks using their actual coordinates; no independent random signal is assigned to networks. Verify that unaffected arrays/channels and all upstream parents remain unchanged.

The negative compact branch is a signed decrement to an already bright source, not a separate negative-source detection experiment.

## 5. Replay semantics: do not hand the answer to the feedback loop

For every C branch, recompute its own first PTC-cleaned M0 from y^I and derive its own initial feedback candidate using the ordinary measurement/admission procedure. Keep the original admission-noise scale and expected-source association/search procedure fixed. Do not seed the source fitter at the injected truth, train a new noise scale, or insert the known injection directly into the initial feedback candidate.

At every subsequent cycle, start from that branch's immutable y^I, subtract only its current candidate operand, freshly learn/apply the existing PTC recipe, and restore exactly the operand subtracted. Do not repeatedly add the injection, subtract from the previous residual, reuse the reference learned basis for the perturbation branch, or accumulate previous candidates.

For D branches, y never changes. The low/high variants change only m0 amplitude, not its support or background. The shoulder-seed branch deliberately adds I_h to the first operand within supported pixels, including valid pixels previously excluded by admission. This is an explicitly authorized **first-candidate-only admission exception** for a diagnostic intervention. Do not threshold the added shoulder away before its first application. All later candidates must be regenerated normally at threshold 3 using complete replacement semantics; the artificial shoulder must receive no persistent exemption or forced support.

For D-threshold-5, use threshold 5 both to form its initial candidate and in every later admission step. Change nothing else. This is a method comparison, not an initialization comparison and not adoption of a new preferred threshold.

Preserve all cycle maps, applied/replacement candidate identities, source measurements, and accounting summaries. Save each branch's m10 but do not apply it. Recompute normal data-dependent PTC state; do not hard-code the reference convergence flags, coefficients, or downstream state into perturbed branches.

## 6. C diagnostics: measure response rather than merely a brighter output

### 6.1 Paired map response and known injection reference

For each C branch and cycle k, including M0, compute:

\[
\Delta T_k^I = T_k^I-T_k^{\rm REF}.
\]

Construct the known injection-only map reference using P and the **fixed REF forward-MAP support, coefficients, and normalization**, without PTC cleaning or centering:

\[
I_{\rm map}=\mathcal M_{\rm REF}(P I).
\]

This diagnostic projection is not another scientific reduction. Validate it independently, including containing-pixel behavior, missing support, and normalization. It prevents pixelization/coverage effects from being mistaken for cleaning losses.

Compare ΔT with I_map using signed difference maps, profiles, existing fixed-template/aperture measurements, and a clearly labeled unweighted projection onto the known injection shape. Report a response projection and residual norm on declared common supported pixels:

\[
g_k=\frac{\langle I_{\rm map},\Delta T_k^I\rangle}
{\langle I_{\rm map},I_{\rm map}\rangle},\qquad
\eta_k=\frac{\|\Delta T_k^I-I_{\rm map}\|_2}{\|I_{\rm map}\|_2}.
\]

These are deterministic shape-response diagnostics, not optimal photometry, significance, or uncertainty estimates. Show signed residual structure rather than compressing every result into one gain. Inspect source core, shoulder/wings, fitted constant, and surrounding bowl/pedestal behavior.

Compare compact positive/negative gains and the even finite-difference map T_plus + T_minus − 2 T_REF. Nonzero even response can reveal finite-perturbation nonlinearity or state changes; do not assume a single amplitude-independent transfer function from ±10% tests.

All branches share the same observed noise realization. Their paired differences can include changes to the cleaning of the original source and background, not just an isolated image of the added component. Label the result **finite full-procedure response at the declared boundary**, not an independently observed recovered source.

### 6.2 Keep estimator response separate from map response

Run the unchanged seven-parameter fitter separately on each full branch map. Do not fit the signed difference map with the ordinary positive-source constraints and interpret that as the recovered increment.

For a nonlinear measurement f, also compute the ideal map-addition comparator:

\[
\delta f_{\rm ideal,k}=f(T_k^{\rm REF}+I_{\rm map})-f(T_k^{\rm REF}),
\]

and compare it with:

\[
\delta f_{\rm actual,k}=f(T_k^I)-f(T_k^{\rm REF}).
\]

This final-map addition is authorized **only as a measurement-response control**, not as the source-response experiment. It distinguishes reduction-induced effects from how a Gaussian-plus-constant fit responds to a non-Gaussian increment. Retain its failures and active constraints as for any other fit.

Report absolute discrepancies first; ratios are unavailable when the ideal increment is zero, too small for the numerical comparison, or undefined. Centroid and shape diagnostics should use physical offsets and a gauge-safe ellipse/shape representation; do not interpret an unstable position angle of a nearly circular fit as meaningful distortion.

Preserve original template, aperture centers/radii, background refitting, support, and units. Do not convert aperture integrals to corrected astronomical flux. Report raw aperture increments and fitted-background contributions separately where that distinguishes lost wings from background absorption.

## 7. D diagnostics: distinguish initialization memory from method sensitivity

Compare each D trajectory against REF at matching cycle numbers. Report peak, centroid, widths/shape, fitted constant, fixed-template/aperture measurements, candidate area and turnover, and full unthresholded source-region map differences. Summarize cycles 1, 3, 5, and 10 while retaining all intermediate data.

For low/high seeds, determine whether the differences diminish, persist, change sign, or grow. Report both endpoint spread and remaining cycle-to-cycle drift. Do not label different T10 endpoints distinct asymptotic fixed points without longer-run evidence; this campaign does not authorize obtaining that evidence by extending the runs.

For the shoulder-seed branch, follow the excess relative to REF in both the total maps and replacement models, including its projection onto I_h. Compare with C-shoulder: in C the shoulder is a real known increment to the data; in D it was added only to the initial feedback model. This is the main recovery-versus-initialization-memory contrast.

Do not assert that a persistent D feature is necessarily false astronomical emission: the unchanged data may contain related structure. What the experiment can demonstrate is dependence on the supplied initialization, and whether later replacement admission removes that dependence.

Report threshold-5 outcomes separately from seed outcomes. A threshold-dependent endpoint is method sensitivity, not random noise, not an error bar, and not automatic evidence that either threshold is scientifically preferable.

Do not add D spreads in quadrature with the A+B scatters or divide paired changes by sqrt(2) times those scatters. There is no independently calibrated stochastic law for these paired experiments. No threshold is selected by minimizing apparent scatter or maximizing recovered brightness.

## 8. Numerical controls and failure handling

Before accepting scientific response results, require the following bounded controls:

1. **Reference replay:** REF must reproduce original M0–T10, support/normalization, measurements, and candidates to existing archived tolerances, with exact equality where the deterministic route promises it. Reuse the original expected values and do not relax tolerances after a mismatch. Stop affected dependent comparisons if replay fails.
2. **Injection identity and non-accumulation:** Verify branch y^I−y = P I on its declared valid domain; verify upstream input hashes, channel isolation, candidate-only seed edits, and use of the same immutable parent at each cycle.
3. **Remove/rejoin accounting:** Reuse the existing residual + restored-operand = total checks at every cycle. The known injection must not appear as an additional rejoin term. Retain mean-removal and normalization conventions.
4. **Independent small fixture:** With a no-op cleaning fixture, recover the known projected injection. With a deliberately attenuating simple cleaner, reproduce its analytically specified first-pass and finite-cycle response rather than assuming attenuation must persist after feedback. Include zero injection, signed increments, taper/support, and a case where later replacement removes a seed-only feature. These are numerical fixtures, not an alternative production cleaner.
5. **Fresh-learning and state audit:** Verify perturbed branches actually learn from their own inputs/residuals. Reuse the corrected basis-layout/subspace diagnostic. Track PTC convergence, eligibility, coefficients, and MAP support rather than forcing them to match REF.

Unchanged upstream validity is mandatory. Legitimate downstream state/support changes under perturbation must remain visible and must not be repaired by zero filling, weight freezing, or masked-out failures. Show the common-support comparison and the lost/gained support separately; if the same measurement becomes unavailable or meaningfully different, label a state transition rather than reporting a simple response ratio as though only brightness changed.

Stop an affected branch before the next feedback application for a new failed processing segment, invalid fit/candidate needed for continuation, unsupported projection, feedback boundary contact, non-finite required operand, or failed accounting. Retain the failing state and completed diagnostics. A changed status of the inherited problematic segment is recorded as perturbation sensitivity; it is not silently repaired or forced to its old status. Follow the inherited accepted handling where that original condition persists.

Low recovery, a broad residual, significant endpoint sensitivity, or a2000's qualified noise model are **not software failures**. Complete unaffected branches/arrays, and do not tune those outcomes away. No automatic retries with changed rank, threshold, damping, masks, or fit bounds.

## 9. Execution, evidence, and owner-facing report

Extend existing FRUIT trajectory, projection, measurement, and reporting utilities. Keep implementation compact and reusable; do not create a parallel reduction framework or generalized simulation service. Store this campaign in a new, unambiguous bundle, for example `successor-point-152389-response-robustness-cd-20260923/`. Never overwrite original products or A+B evidence.

Use the existing validated execution environment and useful available parallelism across independent arrays/branches, without nested numerical-library oversubscription. Profile REF before scheduling the remaining matrix. Report actual wall/CPU time, worker/thread counts, memory, learning/application/MAP costs, serialization, and verification separately. No new dependency/build-policy migration or broad optimization project is authorized. C has execution priority over D if resources prevent finishing the complete bounded matrix; explicitly report every not-run case rather than silently reducing coverage.

Save the prospective manifest, source revision, exact input references, shape rasters and projection checks, branch commands, every cycle's total and candidate products, PTC state summaries, measurements, response comparators, failures, support changes, and regeneration information. Reuse immutable input references and compact arrays; do not duplicate the entire raw upstream archive for every branch.

Run affected existing tests and applicable repository/configuration gates. Distinguish local supplemental evidence from Spack-backed validation. Obtain the existing required independent review of intervention isolation, replay/accounting, shape/reference calculations, and interpretation. Keep review finite and relevant; do not initiate an unrelated contract rewrite or provenance framework.

Provide one owner-facing assessment with a compact run-status matrix, numerical tables, and linked figures. Answer:

- Do the ±10% compact increments have approximately unit response after ten cycles, and do the two signs agree? Show the actual discrepancies rather than only a verdict.
- Is the known weak broader/asymmetric increment retained, attenuated, distorted, or absorbed into the fitted background? How does that evolve from M0 through T10?
- How much dependence on the initial model remains at T10, especially for the shoulder-only seed? Is that dependence still drifting?
- What changes under threshold 5, separately from initialization sensitivity? Did support or processing state change?
- Which conclusions are secure for this finite experiment, and what single next bounded action is justified by the findings?

Present map response, measurement response, endpoint robustness, and numerical validity separately. The approximately 5% final calibration/PSF goals may provide context, but they are not automatic pass criteria for this partial-boundary screen. A good local response does not establish absolute flux recovery; a stable endpoint does not establish correctness; a model-based additive scatter does not establish full adaptive uncertainty.

## 10. Stop boundary

Stop after this prescribed C+D matrix, assessment, and review. Preserve all original and experimental m10 candidates as unapplied. Do not run any cycle 11, promote a new preferred map, correct the published source values, adopt threshold 5, or change the frozen admission scale.

Do not build JINC maps, introduce covariance-weighted fitting or correlation-aware/RBF admission, alter RTC/CAL/PTC rank or segmentation, implement damping, repair detector/network health, run new noise/refit ensembles, or launch a broader observation/injection grid. No full upstream optical/RTC/CAL response claim, confidence interval, detection significance, or new total error budget follows from this screen. Do not automatically merge, push, or activate production changes.

Routine implementation choices should be recorded and executed without another approval loop. For missing inputs or an irrecoverable ambiguity that changes scientific meaning, finish unaffected work and identify the precise unavailable component; do not invent a substitute or silently broaden scope.

**Completion means a controlled answer about finite source response and feedback sensitivity for this bright POINT case—not proof of absolute recovery, a new uncertainty calibration, or authorization to modify the science method.**
