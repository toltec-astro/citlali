# Codex directive: POINT 152389 post-FRUIT map noise and conditional source uncertainty

**Date:** September 23, 2026  
**Authorized scope:** Steps A and B only, for a1100, a1400, and a2000.  
**Execution:** Implement, validate, run against the saved products, assess, and stop. This is not a request for another proposal.

## 1. Objective and scientific boundary

The completed ten-cycle POINT experiment is the reference. Characterize its saved maps and estimate the conditional additive-noise scatter of the existing T10 source measurements without changing the science reduction.

**A — Saved-map noise characterization:** Compare M0, T1, T3, T5, and T10 using fixed spatial regions and matched measurement definitions. Determine how background amplitude, directional correlations, and measurement-scale noise evolve. Assess whether the source-masked background can represent the source location, with particular attention to a2000.

**B — T10 conditional source uncertainty:** Reuse the existing full-observation correlated-noise companion to propagate a T10 background-derived noise model through the unchanged descriptive Gaussian peak fit, frozen-template projection, and existing fixed-aperture measurements.

The target is:

> Additive-noise scatter of the existing T10 processed-map measurements under a declared map-domain noise model, conditional on the chosen processed-source baseline, fixed measurement procedure, coverage/support, and the assumption that the learned background noise model applies at the source.

This is useful uncertainty characterization, not full adaptive-FRUIT repeatability. Learning covariance from outside the source does not demonstrate that it captures source-region noise altered by feedback selection. Describe this transfer assumption explicitly; do not equate the companion with an exact conditional law obtained by propagating detector noise through a frozen cleaner.

Deliver practical estimates where supported and assumption-dependent calculations where informative. Do not make perfect error-bar calibration, a universal covariance model, or complete resolution of a2000 prerequisites for completing useful work on the other arrays. Do not interpret this authorization as permission to use the resulting scales for admission or feedback.

## 2. Recover and bind the exact reference

Read applicable AGENTS.md instructions, current owner decisions, relevant MAP+/POINT/NOI/FRUIT boundaries, the completed `handoff/POINT_152389_FRUIT_TEN_CYCLES_2026-09-23.md`, and `CODEX_POINT_152389_FULL_OBSERVATION_ADDITIVE_UNCERTAINTY.md` (also archived with the earlier companion). Extend the current companion and reporting utilities rather than creating a parallel framework. The accepted experimental measurement remains the target; do not replace it with another draft POINT method.

Discovery hints under `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/`:

- `successor-point-152389-fruit-ten-cycles-20260923/`: `ASSESSMENT.md`, `REVIEW.md`, `run/provenance.json`, `run/RUN.json`, `report/trajectory-table.json`, and bound cycle products.
- `successor-point-152389-additive-uncertainty-20260923/`: `RESULTS.md`, final `companion-r2`, `validation`, and the archived owner work order.
- The immutable M0 and T1 bundles referenced by the ten-cycle provenance, including the original source measurements and empirical companions.

Relevant implementation hints are `tools/timestream_successor/point_additive.py`, `run_point_additive.py`, `validate_point_additive.py`, `POINT_ADDITIVE.md`, and the ten-cycle measurement/report utilities. Resolve actual locations and current identities; paths and filenames alone are not authority to select a parent.

Reported T10 peak amplitudes are 937.421686, 664.734078, and 813.617700 mJy/nominal beam for a1100/a1400/a2000. Reproduce the bound measurements to the archived numerical precision/tolerance before attaching uncertainties; these rounded values are sanity checks, not replacements for full saved references.

Verify that T10 is the total reconstructed map made using m9. Preserve saved m10 as unapplied. Carry forward the inherited a1100 network 3 / scan 5 condition and the a2000 noise-model qualification; neither is repaired by this task.

Bind hashes of consumed maps, measurements, model/selection records, and relevant external inputs. Verify them before execution and again at completion. Write new diagnostics into a separate companion bundle. Never overwrite prior maps, measurements, companions, or ten-cycle manifests.

## 3. Freeze the measurements and separate their spatial domains

Recover the exact seven-parameter descriptive Gaussian-plus-constant fit, including parameter conventions, model equation, weights, initialization/search, bounds, solver, nuisance parameters, and failure rules. Retain the original 40-arcsecond-radius fit footprint and its actual coordinate definition. Do not change weights to inverse covariance or initialize simulations at their known generating parameters unless that is already the actual measurement rule.

Recover the original frozen-template projection and all saved fixed apertures. Preserve their centers, radii, pixel-area factors, support, and background conventions. In particular, do not redefine apertures using T10 widths or move them to a newly fitted centroid. Reproduce their saved values as well as the fitted peak.

Distinguish three domains:

1. **Measurement domain:** The exact existing fit, template, apertures, and any pixels used by their background estimator.
2. **Noise-training domain:** Source-masked background on the available saved full map; it need not be restricted to the fit disk.
3. **Cross-cycle comparison domain:** A fixed common-support background region used consistently within each array.

A source-exclusion mask for covariance learning must not become an exclusion mask for the source measurement. Simulation must cover the union of the measurement and background-estimation domains while preserving original missing-pixel semantics.

Keep fitted peaks in mJy/nominal beam. Preserve the saved aperture units, including mJy/nominal-beam arcsec² where applicable. Do not convert aperture sums into integrated flux density, redefine beam area, or introduce aperture/response corrections. Recover nominal beam conventions from the products rather than assuming they equal the fitted T10 widths.

## 4. Step A: comparable saved-map diagnostics

### 4.1 Fixed masks and honest background treatment

Within each array, construct and freeze one common analysis-support mask and one conservative source exclusion for M0/T1/T3/T5/T10. Start from the existing mask prescription and the union/envelope of relevant saved source geometries. Cover plausible wings and source-associated bowls, not merely the compact admitted feedback footprint. Freeze the rule before comparing inferred noise amplitudes and publish its exact geometry.

Use one larger exclusion as a sensitivity case, also fixed across cycles. If this leaves inadequate background or source-depth coverage, report that limitation; do not shrink the mask to obtain a favorable estimate. Keep the original trajectory-report background region as a separate diagnostic if useful, with its different mask clearly identified.

Publish background means/medians, centered dispersion, and second moments relative to the declared reference separately. Regional centering can remove broad structure: record which modes are removed for covariance learning. Do not silently subtract gradients, high-pass the maps, Gaussian-subtract the source and call the remainder noise, or clip tails to make the background quieter.

Describe measured off-source structure as background residual structure until its noise interpretation is supported. These maps alone cannot uniquely separate stochastic background from every deterministic processing residual or unmasked sky feature.

### 4.2 Spatial amplitude and directional correlations

Reuse the existing compact model, with separately learned parameters at each selected cycle:

\[
C_k(p,q)=s_k(p)s_k(q)\rho_k(\Delta x,\Delta y),\qquad \rho_k(0,0)=1.
\]

Keep the model family and main numerical conventions matched across cycles. Assess, rather than assume, whether a shared normalized correlation pattern is adequate within each map.

For each array/cycle report local second-moment scales versus coverage, disjoint region, and available network contribution; robust scale and tail diagnostics; and signed directional covariance over relevant angular separations. Preserve negative correlations and anisotropy. Do not assume inverse-square-root HITS scaling or treat map normalization/weight planes as validated precision.

Use mask-aware pair counts and explicit finite-domain treatment. A fitted two-dimensional spectrum is useful; a raw masked periodogram must be labeled as windowed rather than interpreted as an unbiased sky-noise spectrum. No full spectral-deconvolution project is required.

Record supported lags, pair counts, model smoothing/regularization, and any positive-semidefinite repair. Unsupported long-range correlations remain unknown, not zero. Computational truncation is not evidence of their physical absence.

### 4.3 Noise on the scales of the existing measurements

For the original frozen template and each genuinely fixed linear aperture/background operator, recover exact weights q and compute

\[
V_{k,q}=q^{\mathsf T}C_kq.
\]

Use these matched projections to compare measurement-scale noise across cycles. Include a diagonal-only covariance comparison as a diagnostic, not a replacement; correlations need not always increase the variance of a signed/background-subtracted estimator.

If an aperture uses a nonlinear or refitted background, label any fixed-background analytic projection as narrower and use Step B for its actual T10 uncertainty. Do not claim qTCq describes a nonlinear estimator without establishing that equivalence.

Do not launch full source-refit ensembles at every cycle. Do not describe lower pixel RMS as improved sensitivity without considering measurement-scale covariance and the still-unmeasured source response. Cycle differences describe algorithmic changes in the same data, not independent noise draws. Neither the cleaned residual nor the restored model is an independent noise/source verification product.

### 4.4 Focused a2000 representativeness assessment

Reassess a2000 using the saved coverage and network-contribution information. The earlier north/south discrepancy and changing network 11/12 mixture are hypotheses to investigate, not established causal attribution.

Compare regional noise amplitude and normalized covariance at overlapping coverage conditions. Show where source coverage and contributing mixture fall within, or outside, the background training distribution. Spatial pixels, lag pairs, and overlapping patches are not independent repeated observations.

A single compact coverage-plus-mixture amplitude alternative is allowed if existing products support it. Evaluate it on held-out spatial regions where feasible; retain the base model and report both. Do not select it merely because it fits the training data or gives a smaller source uncertainty. If coverage, position, and mixture cannot be separated with this field, report that confounding rather than force a decomposition.

Do not sum per-network variances assuming independence. Any decomposition must retain cross-network terms or state that they are unavailable. Use archived contribution products or compact summaries derivable without rerunning the science reduction. Missing products are not authorization for new RTC/PTC/MAP runs.

Do not expand this into detector-health analysis, an unrestricted spatially varying covariance model, or a multi-observation campaign. A qualified a2000 result is an acceptable completion outcome.

## 5. Step B: T10 source-plus-noise ensembles

### 5.1 Base experiment

Learn the T10 noise model using the declared source-masked background. Construct a noise-free processed-source baseline from the saved T10 descriptive fit and its constant. Apply each existing estimator to that baseline to establish its own noiseless reference. Keep the actual measured T10 value separate from this reference.

Generate

\[
M^{(b)}=\widehat\mu_{10}+n^{(b)},\qquad n^{(b)}\sim\mathcal N(0,\widehat C_{10})
\]

as the initial model-based experiment. Use **256 deterministic realizations per array with a usable base model**, at the actual source position, depth, grid, and support. An assumption-dependent model can be run and labeled as such; invalid covariance or unavailable measurement support must not be hidden by fabricated data.

Use a clean baseline plus fresh noise, not the observed noisy T10 map plus more noise. Generate a joint field over all pixels needed by the fit, template, apertures, and their background measurements. Preserve spatial correlations between these domains. Use correct padding/cropping to avoid wrap-around, and verify scaling after amplitude modulation.

Refit every parameter that is free in the actual source measurement, including centroid, widths, orientation, and constant. Preserve the real initialization/search and constraints. Do not introduce more optimistic starts or freeze nuisance parameters to reduce scatter.

### 5.2 Aperture and background propagation

Apply the exact saved aperture procedure to each realization. If it subtracts a constant from the same trial's source fit, use that trial's fitted constant. If it uses a separate background statistic, recompute that statistic by the original procedure on the same realization. If it is a raw sum, keep it a raw sum.

Do not treat an estimated background as exact or add source and background errors in quadrature as though they were independent. Their dependence must be present in the same trial or in the exact linear operator where applicable.

Measure the frozen-template projection and all fixed apertures on those same draws. Save their joint covariance/correlation with the peak as a compact diagnostic; these are not independent confirmations of the source.

### 5.3 Statistics, failures, and interpretation

For each measurement report its centered sample standard deviation with B−1 normalization, its mean shift from the corresponding noiseless reference, and conditional-model distribution summaries. **Do not divide the measurement scatter by sqrt(B).** Monte Carlo precision describes the estimated scatter, not the measurement's error bar.

Retain finite bound-limited fits and flag them. Preserve all failure/constraint records. Unless an already accepted rule governs the case, a failed trial makes the primary complete-ensemble refit aggregate unavailable; survivor statistics may be shown only as partial diagnostics. Independently successful linear measurements need not be erased by a separate nonlinear-fit failure.

Do not apply Monte Carlo mean shifts as bias corrections. Do not reinterpret non-Gaussian source-model mismatch, broad source-response features, or a changing fitted background as independently measured random noise. The Gaussian baseline is a conditional reference, not a claim that the true PSF is Gaussian.

Separate three qualifications: finite Monte Carlo precision; uncertainty/representativeness of the learned noise model; and omitted source-response, calibration, upstream, and adaptive-feedback effects. No Gaussian detection probabilities, full-FRUIT error bars, lower-bound claims, or calibrated confidence intervals follow automatically.

### 5.4 Analytic checks and bounded sensitivities

Verify qTCq against simulations for the frozen template and each genuinely linear aperture/background estimator. Include direct dense pixel-pair evaluation on a manageable footprint as an independent numerical check. Check a larger generation domain without changing the scientific covariance. Agreement validates propagation/generation, not observational representativeness.

Before the final ensembles, record a small sensitivity set, with **at most five variants per array**: the larger source mask, north-only and south-only training where usable, one correlation-range or smoothing alternative, and optionally the bounded mixture-aware model for a2000. Do not create a parameter grid or tune until the uncertainties agree.

Use **64 matched trial identities** for base-versus-variant refit comparisons. Do not compare a 64-trial variant directly with the separate 256-trial base as though all variation were a model effect. Reuse common random inputs where compatible and report weakened coupling when domains or representations change. Treat bootstrap/comparison intervals as Monte Carlo precision only.

Flag every variant's source-depth, regional, mixture, and measurement-domain support. An extrapolated calculation is not an empirical upper/lower bound; even an interpolated depth does not prove spatial representativeness. Sensitivity ranges are not confidence intervals. More draws cannot cure lack of representative background.

## 6. Reuse validation without expanding into response work

Run the existing independent map-domain validation appropriate to the reused/changed inference code, retaining its distinctions between internal covariance-generation consistency and recovery across independently generated observations. Reuse established cases and tolerances where applicable; add focused coverage for the new cycle-mask and aperture/background handling.

Required checks include exact source/template/aperture reproduction; fixed cross-cycle masks; signed anisotropic covariance and normalization; missing support; empirical-versus-model second moments; aperture/background cross-covariance; deterministic regeneration under different worker scheduling; failure reporting; and detection/reporting of deliberately nonrepresentative spatial backgrounds.

Reuse existing blank placements where they remain fully supported under the new masks. Evaluate them on the appropriate saved cycle, using their own coverage and the matching estimator. Never carry an M0 blank value into a T10 comparison. Show placement overlap and available support; do not present many correlated placements as independent calibration evidence. Pure-noise nonlinear fits are not substitutes for bright-source measurement uncertainty.

Map-domain synthetic baselines/noise for this companion and its validation are authorized. **No detector/timestream source injection, frozen-MAP injection experiment, PTC/FRUIT replay, or source-transfer-function experiment is authorized.** The optional detector-to-MAP validation mentioned in the earlier directive is outside this A+B work order.

Use prospective, sampling-aware numerical checks. The final SCIENCE calibration/PSF goals are not pass tolerances for this covariance companion. Failed representativeness is a scientific result to report, not a software test to tune away.

## 7. Implementation and evidence

Keep the work in reusable companion utilities using compact arrays, local covariance representations, existing source measurements, and existing provenance. Do not grow Engine/core abstractions, per-detector-sample records, or a generic uncertainty platform.

Use useful local parallelism without nested oversubscription. Bind seeds to array, cycle/model identity, and trial—not worker order. Profile loading/verification, diagnostics, inference, generation, fitting, and plotting separately; record worker/library-thread counts, wall/CPU time, and measured memory with parent/child versus aggregate distinctions. No Unity campaign, dependency installation, or build-policy migration is part of this task.

Run affected tests and applicable repository gates; distinguish newly executed evidence from inherited results. Report an environment-blocked gate honestly rather than calling it passed. Follow existing independent-review/worktree requirements and obtain a review of numerical propagation and claim boundaries where available. Preserve unrelated work; no automatic integration, push, or production activation.

## 8. Required owner-facing results

Produce one concise assessment with linked figures and machine-readable tables, backed by a reproducible bundle.

**A — Map comparison:** Per array and selected cycle, report support/masks, background reference and dispersion, regional/coverage dependence, directional correlation, fixed-template and aperture-scale noise predictions, and diagnostic diagonal-only comparisons. Show source-mask sensitivity and a2000 coverage/mixture applicability. Plot fixed, comparable regions and angular scales.

**B — T10 measurement uncertainty:** For the fitted peak, frozen template, and each fixed aperture, report the unchanged observed value, noiseless reference, conditional additive scatter, relative scatter where meaningful, Monte Carlo mean shift/precision, failures/bounds, and model sensitivity/applicability. Include the joint measurement covariance and analytic/simulated comparisons. Keep units explicit.

Give each requested estimate a clear disposition: provisional model-based conditional estimate, assumption-dependent calculation, or unavailable with a concrete reason. Do not automatically carry the M0 disposition to T10; reassess every array.

Historical M0 additive estimates can appear for context, but clearly identify differences in masks/models. Use matched fixed-operator comparisons for cycle evolution. Do not claim a change in the primary fitted-peak uncertainty from unmatched legacy and new calculations, and do not silently run extra all-cycle refit ensembles to fill a table.

Retain exact inputs/method identities, masks, covariance parameters and validity, pair counts, seeds, trial-level measurements, failure states, representative synthetic FITS patches, validation results, runtime, and checksums sufficient for regeneration. A pixel-noise scale image is not a fitted-source uncertainty image. Reuse existing artifact conventions; do not duplicate the entire upstream evidence bundle.

End the assessment by answering:

1. What changed in the observed background and its measurement-scale noise from M0 to T10?
2. What can we quote for the T10 peak and fixed-aperture additive scatter, under what assumptions?
3. How much do the supported modeling alternatives change those conclusions?
4. Does a2000 remain limited, and what specific missing evidence dominates?
5. What remains unknown about source response and adaptive-FRUIT uncertainty?

## 9. Stop boundary

Stop after A+B implementation, bounded validation/execution, and assessment. Do not select a scientifically best map, alter published amplitudes, update the frozen admission scale, apply m10, run cycle 11, or change any saved feedback state.

Do not run steps C or D: no response injections and no initialization/threshold robustness experiments. Do not build JINC maps, implement correlation-aware admission/RBF feedback, change RTC/CAL/PTC/MAP or fit weights, introduce full relearned noise ensembles, or redefine SCI-NOI/formal POINT products.

Routine implementation choices should be recorded and executed without another approval loop. For an irrecoverable scientific ambiguity or missing input, finish unaffected work and identify the precise unavailable component; do not invent an estimator or broaden scope to bypass it.

**Completion means useful, qualified map-noise and T10 additive-uncertainty results for the current bright-POINT method—not proof of absolute flux recovery and not authorization to change that method.**
