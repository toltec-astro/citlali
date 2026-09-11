:chatgpt-content-reference{index="0"}

# Directive to Codex: bounded RBF-feedback experiment

**Proceed with one bounded experiment using regularized radial basis function fitting as the map-to-feedback estimator.** Compare it with the existing matched pixelwise and single-Gaussian feedback controls. Also evaluate PSF-concentration diagnostics derived from the RBF representation, but keep those diagnostics separate from reconstruction and admission decisions.

This authorizes implementing and running the isolated experimental harness—not merely preparing another discussion packet. Recover the latest experimental state, preserve previous results, and reuse existing input bindings and evaluation machinery. Do not change production contracts, profiles, or defaults.

## 1. Scientific objective

Determine whether RBF feedback can retain the benefits of spatially coherent inference **without forcing the astronomical signal into a single Gaussian morphology**.

The motivating case is a telescope that is out of focus and comatic. Its source image may contain broad shoulders, asymmetric tails, or multiple lobes. Those features are astronomical signal to preserve, not defects for the reduction to remove.

Test two separate hypotheses:

- **Reconstruction:** A controlled RBF representation can recover faint, spatially organized structure more faithfully than independent-pixel selection while accommodating shapes that a single Gaussian cannot represent.
- **Diagnosis:** Concentration measured from the recovered brightness distribution can describe PSF quality without becoming an objective that biases reconstruction toward a compact image.

Neither hypothesis is presumed true. A compact or brighter result is not automatically a better reconstruction. This experiment does not authorize a universal OOF or SCIENCE policy.

## 2. Preserve the established FRUIT recurrence

Change the **map-to-feedback inference step**, not the surrounding processing chain.

Every pass must return to the same immutable pre-PTC parent, subtract the currently applied astronomical model, relearn centering and the correlated subspace from that residual, apply the resolved PTC state, restore the astronomical model, and make the reconstructed total map:

\[
r_k=d-P S_k,
\qquad
\Theta_k=\operatorname{LearnPTC}(r_k),
\]

\[
M_k=\operatorname{MAP}
\left[
\operatorname{ApplyPTC}(r_k;\Theta_k)+P S_k
\right].
\]

Infer the **complete replacement model** \(S_{k+1}\) from \(M_k\). Do not accumulate models, infer only from the residual map, repeatedly clean a previous descendant, or hold the PTC subspace fixed across passes. Previously admitted structure must be able to weaken or disappear.

Keep the existing rank-5 network grouping, ordinary mapmaking, uniform map coefficients, flags, geometry, support conventions, and pass budget fixed. Preserve array-specific results. Do not introduce new detector weighting, JINC changes, cross-array coupling, or a new projection route.

Reconstruct the RBF model on the existing feedback grid and use the existing map-to-timestream projection. **The ordinary total map and inferred feedback model remain distinct products.** RBF regularization belongs to the experimental FRUIT inference step; it does not redefine SCI-MAP output.

## 3. Implement one controlled RBF estimator

Use

\[
S(\mathbf{x})=\sum_j a_j\phi_j(\mathbf{x}).
\]

Start with overlapping, localized, unit-integral Gaussian RBFs on a fixed lattice, with **one characteristic basis width per array**. Centers and widths remain fixed throughout the trajectories. Their sum may be asymmetric, extended, or multilobed even though each building block is radial.

**Use regularized fitting, not exact interpolation.**

For this positive-calibrator screen, use nonnegative coefficients. Treat that as an experimental POINT prior, not a restriction on generic astronomical signals. Fit explicit nuisance-background terms, but exclude them from feedback.

A suitable estimator is

\[
(\widehat a,\widehat\beta)
=
\arg\min_{a\geq0,\beta}
\left[
\|M_k-\Phi a-B\beta\|_W^2
+
\lambda\|La\|^2
\right],
\]

where \(B\beta\) describes the nuisance background and \(L\) is one declared quadratic smoothness operator.

Use uniform valid-pixel fit weights for the initial screen rather than opening another weighting investigation. Before execution, specify the penalty, normalization, units, strength, fitting domain, and background treatment. Do not impose coefficient sparsity or maximize concentration as a reconstruction objective.

The basis must cover the declared, permissive source domain. Do not recenter or shrink that domain around an encouraging peak. Choose basis resolution to represent the narrowest scientifically relevant structure—not the overall diameter of a defocused image. A nominal beam may inform a resolution check, but must not become an enforced source shape or width.

Use locality, reusable numerical work, and bounded computation where practical. Do not default to a dense global interpolator with one center per pixel. Record basis size, conditioning and solver diagnostics, convergence, setup cost, and per-pass inference cost. Fail explicitly on an unusable fit.

**Admission remains a separate decision.** Define and freeze one spatially coherent admission rule using the existing nuisance/noise information. It must permit non-Gaussian structure: do not retain a successful Gaussian fit or bright central seed as a prerequisite.

A fitted positive model, a large peak, or persistence through FRUIT is not sufficient evidence of astronomical origin. Treat an empirical scatter threshold as an empirical rule, not an automatically calibrated false-alarm probability.

## 4. Check representation before testing recovery

Perform a small noiseless representation check before freezing the candidate.

Verify that the chosen basis can represent both an in-focus compact PSF and a defocused/comatic PSF, including lower-surface-brightness structure. **Generate the test truth independently of the fitted RBF basis.**

Include a sub-lattice translation check: the same PSF should not acquire a substantially different shape or concentration merely because it moves relative to the RBF centers. Keep pixelization and integration conventions consistent with forward projection.

Use these checks to distinguish basis inadequacy from FRUIT failure and to make a bounded geometry/conditioning choice. Do not launch a kernel, lattice-spacing, width, or regularization sweep.

Freeze the estimator before examining its empirical trajectories. Report representation error separately from regularization and noise-induced error.

## 5. Add concentration diagnostics—but do not feed them back

For the background-free fitted source over a fixed declared aperture \(\mathcal A\), define

\[
F_{\mathcal A}=\int_{\mathcal A}S(\mathbf{x})\,d\Omega,
\]

\[
\boxed{
A_{\rm eff}
=
\frac{F_{\mathcal A}^{\,2}}
{\int_{\mathcal A}S(\mathbf{x})^2\,d\Omega}.
}
\]

Smaller effective area means a more concentrated model. This is **not** a uniquely calibrated focus measurement and does not establish that the fitted source is real or correctly located.

Compute it directly from the RBF coefficients. Precompute, over the same aperture and discretization,

\[
b_j=\int_{\mathcal A}\phi_j\,d\Omega,
\qquad
H_{ij}=\int_{\mathcal A}\phi_i\phi_j\,d\Omega,
\]

giving

\[
\boxed{
A_{\rm eff}=\frac{(b^Ta)^2}{a^THa}.
}
\]

Include the off-diagonal overlaps. A coefficient-only sum of squares is not a substitute. Verify agreement with direct integration of the reconstructed model.

Also retain the explicitly basis-dependent coefficient-concentration diagnostic

\[
p_j=\frac{a_jb_j}{b^Ta},
\qquad
N_{\rm eff}=\frac{1}{\sum_jp_j^2}.
\]

Using \(a_jb_j\) accounts for clipped basis support. Label \(N_{\rm eff}\) as an internal representation diagnostic, not a universal optical-quality measure. Do not use maximum/minimum coefficient ratios as the principal metric.

Report aperture, basis identity, units, background convention, admission state, and truncation or coverage limitations. Integrated map brightness is not automatically physical flux density without the existing beam convention. Mark absent or insufficiently supported measurements unavailable rather than simply labeling them “poor focus.”

**Do not use concentration to select regularization, source support, admission, stopping iteration, or the winning reconstruction.** Removing real wings could otherwise improve the score while damaging the science.

Quantify concentration bias against known truth, including brightness/noise and grid-phase sensitivity. Compare arrays separately.

## 6. Run a small matched empirical screen

Use **123424** as discovery data. Check and preserve the actual exposure status of **129081**; keep it reserved until the candidate and decision rules are frozen. Never describe previously examined data as untouched holdout data.

Use a small paired test set, not a full Cartesian campaign:

- Existing compact-source injections and null realizations, preserving continuity with the previous experiment.
- An independently generated defocused-and-comatic source, with a fair brightness normalization and a regime where the source is detectable but important structure is faint. Include a limited second brightness level to expose reconstruction and concentration bias.
- The real discovery pointing, treating existing operational pointing outputs as a control rather than truth.

A compact aberrated-PSF simulation or separately qualified template is sufficient; do not turn this into a telescope-optics modeling project. Retain the existing mismatch case where it adds information within the budget.

Include a lightweight background-only check: flexible RBFs can absorb background even when the explicit fitted plane is excluded from feedback.

Use identical geometry, masks, truths, and paired nuisance realizations across arms. Rerun the relevant learning independently for every source/null trajectory. Paired source-minus-null differences are useful response diagnostics, but do not imply exact noise cancellation in nonlinear feedback. Report null trajectories and their admitted models separately.

For asymmetric PSFs, **predeclare the pointing observable and its noiseless reference**. Do not interchange fitted Gaussian center, brightness centroid, peak location, and injected translation. Preserve the established POINT measurement as a common downstream evaluation without letting its Gaussian shape dictate feedback.

Reuse control products only when their bindings genuinely match; otherwise rerun the necessary controls. Historical OG operational results remain a reference, not ground truth or an automatically matched speed benchmark.

## 7. Evaluate fidelity, failure behavior, and cost

Retain existing acceptance tolerances wherever applicable. Before running the new morphology cases, define additional source-shape and integrated-brightness criteria.

Evaluate structure over fixed truth-defined regions, including the coma tail or defocused shoulder. Do not select evaluation regions after seeing which features the candidate recovers.

The primary question is whether RBF feedback improves non-Gaussian structural and brightness recovery while preserving acceptable in-focus recovery, pointing performance, null behavior, and runtime. Report outcomes per array rather than hiding failures in an average.

Inspect reconstructed total maps, applied and next feedback models, residual structure, source measurements, background leakage, false admitted structure, concentration bias, and pass-to-pass behavior.

**Separate what the feedback model estimates from what the output map actually recovers.** Stability, a larger peak, agreement with OG, or an apparently sharper PSF is not proof of correctness.

Charge RBF inference and required admission work to the candidate. Separate reusable setup, PTC, mapmaking, inference, operational diagnostics, evaluation-only work, and output overhead. Preserve the existing map-ready timing convention: unused next-model inference must not inflate time to the current map.

Do not compare isolated harness timings with unmatched full OG runs and call the ratio an end-to-end speedup. Retain established pass and resource budgets. Report a scientifically interesting improvement that misses the POINT runtime requirement as such.

## 8. Close with an explicit decision

Allow **one initial RBF candidate and at most one targeted revision**, justified by a specific observed failure. Preserve the initial result and freeze a revision before rerunning. An inconclusive result must not trigger an open-ended redesign.

Close with a **keep, revise, or reject** assessment that separates the feedback estimator from the concentration diagnostic. Either may be useful even if the other fails.

A “keep” means the bounded evidence justifies the next validation step—not production qualification or demonstrated generality across OOF/SCIENCE. A few null realizations constitute a failure screen, not a measured rare-failure rate.

Deliver one compact experiment report with frozen configuration/input/code identities, comparison figures, machine-readable metrics and timings, failure records, and remaining limitations. Preserve prior work and avoid another large documentation-only gate.

**Recover the PSF the telescope produced, then describe its concentration. Do not make the reduction produce a PSF that scores well.**