# Codex directive: POINT 152389 — frozen-PTC shoulder-response decomposition

**Date:** September 24, 2026  
**Status:** NEW FOLLOW-ON WORK ORDER. The C+D response/robustness campaign is complete; do not execute its directive again.  
**Scope:** Saved-state diagnostic analysis of REF and C-shoulder for a1100, a1400, and a2000.  
**Execution:** Recover the exact saved inputs, implement the bounded diagnostic, validate, execute, assess, and stop. This is not a request for another proposal or approval loop.

## 1. Objective and authorization

The completed C+D campaign found nearly unit compact response, negligible tested initialization dependence, and incomplete/distorted response to the broader shoulder. Determine how the **already measured shoulder response** is assembled from fixed cleaning, the change in the protected feedback candidate, and changes in PTC's fitted means and learned subspace.

Use the saved reference and injected trajectories. Authorize read-only application of saved PTC states to the specific vectors and residuals below, plus diagnostic forward-map accumulation through the saved operator. Do not learn a new PCA/ALS basis, generate a new feedback candidate, or run a new adaptive trajectory. Reusing a saved basis does require solving its application coefficients on a new input when that is what the existing Apply method does; that is not authorization to relearn the basis.

The central deliverable is a **signed map-level accounting that closes against the saved adaptive response**, followed by a mechanistic assessment. Do not infer the cause from a single frozen-cleaner gain. In particular, equal gains from a one-pass frozen calculation and a ten-cycle adaptive result do not establish that admission is irrelevant: protection and state-change effects may cancel.

This is a reference-anchored, conditional decomposition of completed histories, not a unique causal attribution, a new transfer correction, or the result of a hypothetical fully frozen ten-cycle method. The candidate differences already contain the effects of the preceding adaptive histories.

## 2. Recover and bind the exact completed experiment

Read applicable AGENTS.md, current owner decisions, the implemented experimental PTC/FRUIT/MAP bindings, and the C+D directive, handoff, assessment, completion record, and independent review. Preserve the accepted method rather than substituting a different draft POINT or PTC method.

The completed bundle is discoverable under:

`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-response-robustness-cd-20260923/`

Recover its `freeze/`, REF and C-shoulder products, `report/`, `COMPLETION.json`, `REVIEW.md`, `FINAL_REVIEW.json`, `REGENERATION.md`, and `handoff/POINT_152389_RESPONSE_ROBUSTNESS_CD_2026-09-23.md` in the repository. Follow archived bindings to the original post-RTC/CAL parents, AST coordinates, masks, injection, applied candidates, means, bases, application rules, MAP coefficients, and normalization. Paths are discovery hints, not scientific identity.

The supplied completion record identifies native execution source `26af74ac5e59d88a99da1ed4047a7a71a327fca0`; later sources include report-only corrections. Verify actual bindings and establish equivalence for any current diagnostic code. Do not run a stale binary or silently reinterpret saved arrays with a different layout or solver.

Reported T10 shoulder projection checks are **0.97437 / 0.91107 / 0.84988** for a1100/a1400/a2000. Recover full precision, domains, and the corresponding distortion measures from the saved report; do not replace them with these rounded numbers. The original a1100 network-3/scan-5 condition and a2000 background-model limitation remain explicit and unrepaired.

Recover the actual tapered shoulder raster and its projected timestream operand. Do not redraw, move, normalize, or truncate it. It was a 5%-peak, twice-width, offset test shape, not a new source model. Confirm `y_H - y_R = P h` on the original injection domain, including valid PTC entries that are not final MAP contributors. Reconstructing an already bound operand from its exact saved representation is allowed; changing the perturbation is not.

Verify consumed files against their archived hashes, including referenced external inputs, before execution and at completion. Do not simply hash the files currently found at old paths and declare them authenticated. Preserve all original bundles and the existing C+D directive.

## 3. Fixed analysis matrix and indexing

Analyze these five paired snapshots in each array, **15 REF/C-shoulder pairs total**:

| Saved output being explained | Applied reference operand | Applied shoulder-branch operand |
|---|---|---|
| M0 | No feedback operand | No feedback operand |
| T1 | REF m0 | C-shoulder m0 |
| T3 | REF m2 | C-shoulder m2 |
| T5 | REF m4 | C-shoulder m4 |
| T10 | REF m9 | C-shoulder m9 |

Use the state that actually produced each output. Never use a replacement candidate from the same output as though it had already been applied. In particular, **T10 uses m9, not m10**.

The five snapshots are independent saved-state evaluations, not a new sequence: no diagnostic output becomes another snapshot's input. Read other existing maps/receipts only as needed to verify the ancestry and admission of these applied candidates. A reverse-anchor sensitivity check is authorized at **T10 only**, as defined below. Existing compact and D results may be quoted as context; do not execute new diagnostics on those branches.

Freeze the snapshot identities, comparison domains, numerical tolerances, component definitions, and requested figures in one compact prospective manifest before inspecting the new decomposition results. If resources limit completion, prioritize T10 across all arrays, then the other prescribed snapshots; report every unavailable/not-run pair explicitly. Do not substitute extra snapshots or new trials.

## 4. Define precisely what is frozen

For each snapshot, distinguish the original immutable parent, applied feedback operand, residual entering PTC, learned PTC state, cleaned residual, restored operand, and total map.

Freeze the exact per-network/processing-segment grouping, detector/sample identity, fit and application masks, support, weights, rank, saved basis, scaling/transforms, regularization, rank-resolution/tolerance decisions, and MAP operator. The inherited experiment uses observed-entry rank-10 PTC and ordinary forward MAP; do not replace it with a global array basis or dense idealized projector.

**Primary convention: all learned PTC state is frozen, including the numerical fitted means.** A frozen affine Apply may still solve linear coefficients for the new input on the saved design and masks. Do not instead freeze the previously reconstructed correlated signal and subtract that fixed array from every input: that is not the source response of a frozen-subspace projection.

Recover the exact Apply equations and operator order from the execution implementation. On an appropriate common valid domain, denote a frozen cleaned-output map by

\[
\mathcal A_b(z)=Q_bz+a_b,\qquad b\in\{R,H\},
\]

where `R` is REF and `H` is C-shoulder. **Q is the retained-signal operator, not the removed-mode projector.** Affine offsets cancel in differences evaluated under the same saved state. Establish the form with source inspection and numerical checks rather than assuming linearity from the name PCA. Masks and missing-entry solves must remain exact; a bare `Id - B B^T` is not generally a substitute for masked Apply.

Mean recomputation and basis relearning are different operations. Section 7 explicitly permits an additional **mean-recomputed, basis-fixed diagnostic** where the original mean rule is recoverable. Label it separately from all-state-frozen response. Do not silently substitute that diagnostic for the primary convention.

The original observation is not cropped to the source disk for PTC applications. Use the full required segment support, even for vectors nonzero only near the source: centering and subspace projection can create responses elsewhere. Invalid data remain invalid. Physical gaps, omitted samples, and map-only versus PTC-valid support remain distinct. Paired-r behavior stays unchanged and inert as in the parent.

## 5. Primary matched decomposition

Suppress snapshot/array indices in the equations below. Let:

- `y` be the immutable REF pre-PTC data and `u = P h` the known shoulder increment;
- `v_R = P m_R` and `v_H = P m_H` be the actually applied feedback operands;
- `d = v_H - v_R` be their signed difference;
- `x_R = y - v_R` and `x_H = y + u - v_H` be the actual PTC residual inputs;
- `M` be the common saved linear forward-map operator, including the realized support, coefficients, and normalization.

Thus `x_H - x_R = u - d`, and the saved totals satisfy

\[
T_R=\mathcal M[\mathcal A_R(x_R)+v_R],\qquad
T_H=\mathcal M[\mathcal A_H(x_H)+v_H].
\]

First verify that the two branches really share the needed masks/support and MAP operator at each selected snapshot. The completion record reports no downstream transitions; recheck consumed state rather than imposing equality. A genuine incompatibility makes the simple matched decomposition unavailable for that pair; preserve the existing response and explain the gap. Do not hide it by zero-filling or forcing common weights.

### 5.1 Exact ordered accounting

Construct the following maps using saved Apply states only:

\[
D=\mathcal M[\mathcal A_R(x_R+u)-\mathcal A_R(x_R)],
\]

\[
F=\mathcal M[\mathcal A_R(x_H)-\mathcal A_R(x_R+u)+d],
\]

\[
S=\mathcal M[\mathcal A_H(x_H)-\mathcal A_R(x_H)].
\]

Their meanings are:

**D — direct increment under the all-state-frozen reference cleaner.** Change the data while retaining the reference candidate and PTC state.

**F — effect of changing to the saved shoulder-branch feedback operand under that same reference PTC state.** This includes both altered subtraction before cleaning and identical-operand restoration afterward; the remapped operand difference alone is not the feedback-protection contribution.

**S — remaining effect of changing from reference PTC state to shoulder PTC state on the same shoulder residual.** This includes fitted-mean and learned-subspace changes until explicitly separated below.

These telescope exactly:

\[
\boxed{T_H-T_R=D+F+S.}
\]

For a validated affine Apply, compute compact difference actions rather than subtracting large nearly equal arrays when appropriate:

\[
D=\mathcal M Q_Ru,\qquad
F=\mathcal M[(\mathrm{Id}-Q_R)d].
\]

Cross-check the efficient route against the direct telescoping evaluation. Preserve double precision and account for cancellation against the large original background. Do not absorb a closure discrepancy into S: calculate S independently from the two saved states and then test closure.

At M0, both operands are absent and F must be zero. At later snapshots d is the actual difference of the two historical candidates, **not** a mask applied to the known shoulder and not a truth-derived candidate.

### 5.2 Separate fitted means from the learned projection

Where the recovered implementation has the form

\[
\mathcal A_b(z)=Q_b(z-\mu_b),
\]

with the saved mean vector properly broadcast over the original support, create the cross-state application

\[
\mathcal A_{R,\mu_H}(z)=Q_R(z-\mu_H).
\]

Keep the reference projection fixed while substituting the **saved** shoulder mean. Then split S into

\[
C_\mu=\mathcal M[\mathcal A_{R,\mu_H}(x_H)-\mathcal A_R(x_H)]
=-\mathcal M Q_R(\mu_H-\mu_R),
\]

\[
C_Q=\mathcal M[\mathcal A_H(x_H)-\mathcal A_{R,\mu_H}(x_H)]
=\mathcal M[(Q_H-Q_R)(x_H-\mu_H)].
\]

Require

\[
\boxed{\Delta T=D+F+C_\mu+C_Q.}
\]

This gives four signed contributions: direct response, feedback-operand effect, mean change, and remaining projection-state change. C_Q may be called a **learned-subspace change term** only after verifying that the other Q-defining state is unchanged; otherwise name all included changes. It acts on the full residual, including the original atmosphere/source/noise, not just the added shoulder.

Do not invent this factorization if the implemented centering/scaling order differs. Use the exact recovered operations to create the compatible mean-only cross-state calculation, or retain the independently computed unsplit S and report which finer term is unavailable. An unavailable finer split does not erase valid three-term accounting. No native Learn/ALS call is permitted to manufacture a missing state.

A mean-change diagnostic does **not** restore discarded detector means or optical DC to a science map. It measures the difference between two already realized nonrestoring operations.

## 6. T10 reverse-anchor check: expose attribution dependence

At T10 in each array, repeat the algebra using the shoulder projection as the fixed anchor. For the factorization above:

\[
D'=\mathcal M Q_Hu,\qquad F'=\mathcal M[(\mathrm{Id}-Q_H)d],
\]

\[
C'_\mu=-\mathcal M Q_H(\mu_H-\mu_R),\qquad
C'_Q=\mathcal M[(Q_H-Q_R)(x_R-\mu_R)].
\]

These must also sum to the **same saved** Delta T. An equivalent exact telescoping calculation is acceptable for a more general Apply form.

Report how attribution changes under the two anchors. The total is invariant; the allocation of interacting effects need not be. Do not average the decompositions into a supposed unique physical budget or describe their difference as stochastic uncertainty. A material anchor dependence is part of the result, not a reason to add a full factorial experiment.

## 7. Mean-aware fixed-subspace view

The primary all-state-frozen direct term deliberately holds the numerical mean fixed. It therefore does not by itself include the effect of estimating and discarding the perturbation's mean.

If the actual centering rule is the same fixed-support linear functional in the two branches, write

\[
\mu(z)=Jz,\qquad C=\mathrm{Id}-J.
\]

Verify `mu_R = J x_R`, `mu_H = J x_H`, and `mu_H - mu_R = J(u-d)` against the saved means at every analyzed pair. Re-evaluating this original mean functional is authorized as an application diagnostic; learning a new subspace is not.

Then regroup the same response as

\[
D_{\rm centered}=\mathcal M Q_R C u,
\]

\[
F_{\rm centered}=\mathcal M[(\mathrm{Id}-Q_R C)d],
\]

\[
\Delta T=D_{\rm centered}+F_{\rm centered}+C_Q.
\]

This is **basis-fixed with the original mean rule recomputed**, not all-state-frozen PTC and not a newly iterated FRUIT result. Require closure against the four-term calculation. Show the centering-only contribution `M(Cu-u)` and the subsequent projection contribution `M(Q_R C u-Cu)` as a compact diagnostic of direct attenuation, retaining their order.

If the original mean rule is nonlinear but exactly recoverable, its mean-recomputed/basis-fixed diagnostic may be evaluated with finite differences on `x_R`, `x_R+u`, and `x_H`, with distinct labels and numerical closure; do not write a linear J identity for it. If the rule cannot be recovered compatibly, report this view unavailable without changing the valid primary result.

Present this view alongside the primary result so that loss from centering is not mislabeled as PCA-basis relearning. Do not force mean variation into either a favorable or an unfavorable interpretation.

## 8. Explain admission without claiming that geometry equals recovered signal

At each nonzero snapshot, use the exact **applied** candidate selection masks to partition their signed map difference into:

1. changed brightness on support retained in both candidates;
2. newly selected support, carrying the shoulder candidate's brightness;
3. removed support, carrying minus the reference candidate's brightness.

The three pieces must sum to `m_H - m_R` on the permitted model domain. Project each and apply the same linear feedback-effect operator to partition F. Where the mean-aware linear view is valid, partition F_centered too. Keep unavailable pixels unavailable; valid excluded pixels and the authorized outside-disk no-feedback rule retain their original meanings.

This is an arithmetic partition of the **candidate effect**. It does not uniquely isolate the causal effect of thresholding from all earlier learning, background fitting, or source-shape changes. The candidates were obtained from the total branch maps, not by applying a threshold to the known injection alone. Do not call d a measured recovered shoulder.

Show the known shoulder footprint over the two applied masks. Report shoulder brightness-area and squared-brightness fractions inside/outside each mask on the declared supported grid, labeled strictly as geometric overlap. For relevant added/removed/boundary pixels, reuse saved preceding maps, fitted constants, and the frozen original admission scale to show the actual source-relative threshold ratios and connected-component admission. Do not recompute candidates, tune the noise scale, or substitute the A+B fitted-peak scatter for a pixel admission scale.

Spatial PTC mixing can change pixels outside the selected footprint. Do not infer independent inside/outside response from mask overlap alone. The signed propagated component maps, not just the mask fractions, must support any conclusion about insufficient protection.

## 9. Required measurements and figures

Reuse the C+D injection-only map `h_map = M u`, fixed comparison domains, physical coordinates/handedness, and normalization. Recover the exact existing unweighted projection and residual-norm definitions before reporting results.

For each map contribution X report

\[
g_X=\frac{\langle h_{\rm map},X\rangle}{\langle h_{\rm map},h_{\rm map}\rangle}.
\]

The signed component projections must sum to the reproduced adaptive g. Report absolute closure errors and signed contributions, not only percentages of the deficit. Contributions can be negative or greater than unity and can cancel. Do not clip or force them into positive fractions summing to 100%.

Recover total distortion `eta = ||Delta T-h_map||/||h_map||`. Also report `||Delta T-g h_map||/||h_map||` to distinguish a scalar projection deficit from distortion orthogonal to the injected shape. Check the corresponding Pythagorean identity on the same domain. **Unit g does not establish an undistorted map.** Norms are not additive; retain compact component cross-inner-products if needed to explain cancellation.

Apply existing fixed linear template weights and raw fixed-aperture sums to each component. Preserve centers, radii, pixel area, support, and original units: peaks/template-like quantities in mJy/nominal beam, aperture brightness-area sums in mJy/nominal-beam arcsec². No conversion to corrected astronomical flux.

Reuse saved fitted constants to account separately for background subtraction in the original paired aperture/template measurements. For an aperture of valid area Omega, the paired background term is `-Omega (c_H-c_R)`; for a template it is `-(sum q)(c_H-c_R)`. Verify against the actual saved estimator. Do not attribute this nonlinear fitted-background difference arbitrarily among D/F/C_mu/C_Q. Do not rerun source fits or apply the positive-source fitter to signed component maps. Carry the five constrained a1100 shoulder ideal-comparator flags forward whenever those old peak comparisons are discussed; they do not invalidate map accounting.

Produce a compact set of figures per array: (a) signed component projections versus the five snapshots, (b) T10 known injection, measured response, primary components, residual distortion, and closure residual, and (c) applied masks with shoulder contours plus signed profiles/linear aperture accounting. Give cancellation-prone terms and closure residuals explicit scales. Keep a reference/shoulder axis convention consistent throughout. Do not infer scan-oriented striping from plotting orientation alone.

## 10. Numerical acceptance and failure handling

Before interpreting a pair, pass these controls:

- **Saved-state self-replay:** Apply each branch's saved PTC state to its own actual residual and reproduce its cleaned output and total map, including original causes/support and remove/rejoin accounting. No PTC learning is allowed. Reuse established tolerances, exact equality where promised, and report maximum absolute/relative discrepancies. A mismatch blocks that pair's operator-based attribution, not sibling pairs.
- **Affine/operator checks:** On a small independent masked fixture, verify zero-offset handling, difference linearity when claimed, scaling, and correct saved-basis layout. Include nonzero means, incomplete observed entries, a rank-deficient application case handled by the original rule, and distinct fit/application/MAP masks. Do not silently change a solver or regularization to make replay work.
- **Known-identity fixtures:** Verify zero injection/zero operand change gives zero; identical states give zero state-change terms; identical candidates give zero feedback term; M0 gives zero feedback; a no-op cleaner gives direct `M u` and no protection term. For a fixed linear cleaner with candidate difference exactly equal to the injected signal, check cancellation to the ideal response. That last case is a numerical fixture only, not an oracle feedback model inserted into the observation.
- **Independent closure:** Directly evaluate each term, then compare their sum with the independently loaded saved `T_H-T_R`. Check both anchors at T10, the mean-only split where supported, mean-aware regrouping, candidate partitions, and exact linear measurement sums. Never define the final component as whatever makes the sum close.
- **Input/state isolation:** Instrument or otherwise prove zero new native Learn/ALS calls and zero admission updates. No new m_k, T11, altered original file, new row validity, or replacement state is allowed. Tests that deliberately exercise learning on small synthetic fixtures must be separately identified, never observation runs.

Set scale-aware numerical tolerances prospectively, using existing replay controls and independent fixtures; do not use the approximately 5% final science goal as an accounting tolerance. Preserve failure records. Unexpected attenuation, cancellation, or anchor dependence is a scientific finding, not a failed test to tune away.

If a required state or binding is unavailable, finish unaffected components/pairs and state the precise missing artifact or unsupported operation. Do not rerun the original trajectory to recreate missing means/bases under this authority. No hidden zero filling, rescue, candidate alteration, or broad search for a more favorable comparator.

## 11. Implementation, runtime, and evidence

Extend the existing response/reporting and PTC-Apply utilities. A narrowly scoped saved-state native Apply entry point or matching-source diagnostic harness is authorized if needed; it must not change ordinary learning, reduction, or production defaults. A separate implementation must be verified against native self-replay before use. Do not create a generic simulation framework, dense observation Jacobian, full detector covariance, or new per-sample object hierarchy.

Use compact arrays, matrix-free basis applications, bounded segment processing, and reuse of loaded parents/states. Parallelize independent arrays/snapshots where useful without nested numerical-library oversubscription. Do not use a one-worker or low-core setting by habit; choose and report actual available resources. Preserve useful reuse without duplicating the entire raw parent archive. No new dependency installation, build-policy migration, Unity campaign, or unrelated optimization work is authorized.

Write a fresh bundle, for example:

`successor-point-152389-frozen-ptc-shoulder-20260924/`

Use a new handoff such as `POINT_152389_FROZEN_PTC_SHOULDER_2026-09-24.md`. Save this exact directive, prospective manifest, verified parent/state identities, source/binary identities, Apply/replay checks, component FITS/maps with explicit support, numerical tables, figures, failure states, and regeneration commands. Reference large immutable ancestors instead of copying every earlier experiment.

Record loading/verification, Apply, mean evaluation, MAP accumulation, diagnostics, and serialization timings; wall/CPU time; actual worker/library-thread counts; and measured memory with per-process versus aggregate distinctions. No claim of aggregate memory from adding unrelated process high-water marks. Report the number of processed pairs and each not-run/unavailable result.

Run affected tests and required repository/configuration gates in the available environment, distinguishing local supplemental checks from Spack-backed validation. Obtain the existing independent review focused on exact replay, frozen-state semantics, centering, decomposition closure, candidate indexing, and the limits of causal interpretation. Keep review bounded; no unrelated contract rewrite or new governance framework. Commit locally under existing repository practice; do not merge, push, or activate production automatically.

## 12. Owner-facing assessment and stop boundary

End with a concise assessment that answers:

1. Can the saved shoulder response be reproduced from saved Apply states, and do the independent component maps close at all supported snapshots?
2. At T10 in each array, what are the signed direct, feedback-operand, fitted-mean, and learned-projection contributions? What changes when mean recomputation is grouped with fixed-basis cleaning?
3. Which terms produce the projection deficit and which produce spatial distortion? Are there large cancellations hidden by the scalar gain?
4. Is the feedback-operand effect mostly associated with changed retained brightness or support turnover? What is actually supported about incomplete protection, and what remains entangled with earlier history?
5. How much does interpretation depend on the reference versus shoulder anchor? What can and cannot be concluded about relearning from this saved-state accounting?
6. What **single next bounded development or validation action** is justified? Recommend it, but do not execute it here.

Do not infer a cause from comparing a lone frozen one-pass gain with the adaptive ten-cycle gain. Do not label fixed-state loss irrecoverable, blame thresholding solely from geometric overlap, or label the state-change contribution an independent noise realization. These components are signed conditional diagnostics of the same data, not additive uncertainty-budget entries, confidence intervals, or universal transfer functions.

**Stop after the specified saved-state analysis, validation, assessment, and review.** No repetition of the completed 24-trajectory C+D campaign, no new adaptive cycles, no application of any m10, no cycle 11, no threshold/initialization sweep, no new source/noise ensemble, no new observation, and no new PCA/ALS learning on the observation. No changes to RTC/CAL, PTC rank/grouping, MAP weights, source fitting, the reference threshold, the frozen admission scale, source model, beam convention, or published source values. No JINC, correlation-aware/RBF admission, damping, detector-health repair, total error budget, or absolute astronomical flux correction.

**Completion means an authenticated, numerically closed explanation of the measured shoulder response at the saved pre-PTC boundary—not a new reduction or a claim of uniquely separated physical causes.**
