# Codex directive: POINT 152389 — simple expected-source acceptance

**Owner:** Grant Wilson  
**Date:** September 25, 2026  
**Status:** Implement, validate, rerun the existing admission calibration, assess, and stop.  
**Scope:** POINT 152389; a1100, a1400, a2000; ordinary MAP; a bounded expected-source check. No new feedback trajectories in this work order.

> A POINT observation is expected to contain a bright, approximately Gaussian source near the expected position. Look for that source with one bounded fit. If the result does not qualify, return empty accepted POINT/feedback products with a failed-fit message. Do not turn this into a general source-detection project.

## 1. Decisions and authority

Implement the following common acceptance rule for all three arrays:

| Quantity | Required value |
|---|---|
| Source model | Existing seven-parameter elliptical Gaussian plus constant background |
| Numerical fit | Finite, converged result from one bounded optimizer attempt |
| Peak amplitude | Positive amplitude above the fitted constant |
| Amplitude S/N | `A / sigma_A > 10`, using the fitted-amplitude uncertainty defined in section 3 |
| Principal-axis widths | Both Gaussian standard deviations strictly greater than **1.5 arcsec** |
| Centroid | Radial displacement **at most 20 arcsec** from the expected source position |

The 20-arcsec radius is the selected implementation of “near the center.” Use the existing AST/tangent-plane expected source origin, not the center of an available-pixel bounding box or a peak found in the current map. Record that origin. The width limit is on **sigma**, not FWHM or pixels; its FWHM equivalent is approximately 3.532 arcsec.

“Gaussian-ish” does not authorize a new goodness-of-fit cutoff, exact beam-width requirement, circularity requirement, shoulder fit, or source-model family. Keep existing upper bounds and other established fit conventions unless explicitly changed here.

This order supersedes the inherited lower-width bound, the central-maximum/median test as evidence of source presence, and the treatment of ordinary fit rejection as an undefined selection outcome, **for this named diagnostic route**. It also authorizes the lightweight amplitude-uncertainty calculation below. It does not change a frozen normative SCI-POINT contract or a production default.

Read applicable repository instructions and recover the actual current implementation. Historical discovery anchors include:

- `CODEX_POINT_152389_CORRELATION_AWARE_ADMISSION.md` and its failed-calibration assessment.
- `CODEX_POINT_152389_FULL_OBSERVATION_ADDITIVE_UNCERTAINTY.md` and the completed uncertainty companions.
- `validation/point_152389_blank_fit_diagnosis_2026-09-25/` and its assessment, cases, reproduction instructions, and review.
- The sealed correlation-admission and blank-fit-diagnosis bundles under `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/`.

The reported diagnosis closeout commit `cfcde23f4` is a discovery hint, not a substitute for checking the current worktree and bound inputs. Preserve all previous results. Reuse existing fitter, covariance, candidate, simulation, and reporting code; no parallel framework.

## 2. One bounded fit, followed by a simple decision

Retain the existing Gaussian-plus-constant objective, fit footprint, pixel weights, unit conventions, source-independent initialization recipe, and numerical stopping tolerances, with these changes:

**Constrain both sigmas to be at least 1.5 arcsec during optimization.** Do not spend the budget exploring narrower solutions and reject them only afterward. Acceptance requires both to exceed the floor; a lower-bound-active solution is rejected even if numerical interior handling represents it slightly above 1.5. Use the existing solver's bound-state convention rather than inventing an astronomical tolerance.

Restrict peak search/initialization to the 20-arcsec disk. Use the simplest supported central parameter bounds. A box-constrained solver may use `|dx| <= 20` and `|dy| <= 20`, followed by the exact radial acceptance test `hypot(dx,dy) <= 20`; do not claim that the box itself is a disk, and do not build a new constrained optimizer just for this boundary. Do not fall back to a global-map peak.

Use one deterministic initialization and **one optimizer call, with at most the inherited 3,000 function evaluations**. No doubled-budget attempts, restart ladder, randomized initialization, or relaxed tolerances after failure. The old maximum/median calculation may remain solely as initialization bookkeeping, not as a source-presence test. Handle a valid constant map as an ordinary rejected POINT observation, not an initialization exception.

After convergence, apply the position and width checks, then the amplitude S/N check. No separate multiscale detector or pre-fit significance threshold is requested. Do not introduce an amplitude lower bound that forces a fit to satisfy S/N > 10.

## 3. Define amplitude S/N without nested simulations

The numerator is the fitted Gaussian peak **above its fitted constant**, in the existing mJy/nominal-beam units. It is not amplitude plus background, integrated flux, or the largest map pixel.

For this diagnostic acceptance rule, select a **local-linear, covariance-propagated uncertainty of the same free Gaussian-plus-constant amplitude estimator**. Reuse an equivalent existing calculation where available. Otherwise add this small calculation beside the current fitter; do not run a refit ensemble inside every calibration trial.

Let `J` be the model Jacobian on the actual fit pixels, including amplitude, both centroid coordinates, both shape widths, orientation, and the fitted constant. Let `W` be the quadratic weight matrix of the existing fit, unchanged. If the implementation supplies residual multipliers, use their squares in `W`. Define the amplitude's local map sensitivity by

```text
L = pseudoinverse(J.T @ W @ J) @ J.T @ W
q_A = amplitude row of L, represented as a pixel column vector
sigma_A_squared = q_A.T @ C0_fit @ q_A
SNR_A = fitted_amplitude / sqrt(sigma_A_squared)
```

Use a stable QR/SVD or equivalent solve; the displayed formula does not require explicitly forming an inverse. Propagate **all fitted nuisance parameters**, including the background. Do not substitute the fixed-template amplitude variance or freeze nuisance parameters to obtain a smaller denominator. A rank-deficient shape description may be handled only when the amplitude sensitivity is still identifiable; an undefined orientation for an exactly circular Gaussian must not by itself reject an otherwise identifiable amplitude. Reuse existing rank handling where suitable. Do not silently treat an active bound as an exactly known nuisance parameter.

Use the **same full, frozen ordinary-MAP M0 covariance `C0`** already bound for the correlation-admission campaign, restricted to the fit support. Reuse the existing covariance contraction machinery; no covariance inverse, new noise model, new fit weights, or observation-wide dense covariance is needed. BASE-POINT, CORR, and DIAG all use this same source-acceptance calculation. DIAG differs only in the later supplemental-neighborhood variance.

This is an explicitly selected, conditional approximation for a practical bright-source quality cut. It is **not** a replacement for the existing Monte Carlo uncertainty reports, a publication-grade error bar, or an exact ten-sigma false-alarm statement. Frozen `C0` remains a campaign reference scale, not a claim that actual noise is unchanged through FRUIT. Carry its source-depth and regional-model qualifications forward.

Do not use pixel RMS, the optimizer's independent-pixel formal error, a division by the square root of the trial count, an old T10 peak scatter applied indiscriminately to other maps, or a diagonal substitute for `C0`. Do not cap a large uncertainty or floor an invalid variance to force acceptance.

If finite fit geometry leaves amplitude nonidentifiable, return a failed fit with an explicit reason. If required covariance/input data are missing or invalid, retain a computational/unavailable outcome; do not silently convert it into a scientifically valid null trial.

## 4. Empty outputs are a defined selection outcome

For a valid input map, an ordinary rejected result—insufficient amplitude S/N, disallowed width or position, finite nonconvergence/evaluation-budget exhaustion, or nonidentifiable amplitude—has the following behavior:

```text
POINT fit failed: no source satisfying the expected POINT criteria was found.
```

Append a concise reason. Return no accepted source parameters or pointing correction, an empty accepted-source mask, and an empty feedback candidate. On valid in-domain pixels the candidate contains zero admitted brightness; unavailable pixels and outside-domain semantics remain unchanged. Clear prior accepted output state; never return a stale fit or candidate from another map or stage.

Preserve the underlying measured MAP, coverage, and diagnostic fit trace/parameters when available. “Empty” describes accepted POINT/feedback products, not an observational assertion of zero sky. Do not overwrite measured signal, noise, or coverage with zeros, and do not fabricate zero uncertainties.

Keep the implementation small: an explicit accepted/failed status and cause in existing result records are sufficient. Retain the optimizer's original convergence/status information independently. A failed fit remains a failed fit; the **selection result** is now completely defined as empty. Do not claim that failure proves the source is physically absent.

Unexpected exceptions, corrupted input, unsupported covariance, and non-finite numerical calculations remain errors. They must not disappear into a blanket exception handler that reports a successful non-detection. Outward fail-safe empty products do not turn such trials into completed calibration outcomes.

Apply the identical procedure to real maps and both synthetic scenes. Simulation truth labels must never choose the output. For a legitimately empty candidate, that scene has no eligible supplemental anchors and contributes `-infinity` to the maximum-score calculation, using the existing serialization convention. Keep every draw and every failure/rejection reason.

The >10 requirement is **source recognition only**. It does not replace the threshold-3 pixel rule and must not be imposed separately on every feedback pixel. A qualifying source still uses the existing measured-brightness candidate builder. Do not apply this positive-source guard to signed response-difference maps or injection-only reference products.

## 5. Preserve the historical comparison and expose the fit change

Leave the historical fitter and sealed BASE products reproducible. Name the new diagnostic route, for example `POINT-EXPECTED-v1`; call the common source-guarded threshold-3 control `BASE-POINT` in reports so it cannot be confused with the historical BASE.

On a given input map, run the new measurement/guard once and share its result between BASE-POINT, CORR, and DIAG. All three use the same fitted background convention and measured brightness operand. CORR and DIAG only supplement the new baseline mask; neither can invent a source when the guard rejects it.

Keep original `C0`, original `K0`, neighborhood scales, anchor/core exclusions, signed neighbor brightness, connectivity, uniform gain, and zero/unavailable semantics unchanged. Do not recompute `K0` or synthetic scene means from the revised fits. Feedback brightness remains the measured pixel operand, never the Gaussian or the neighborhood mean.

Changing bounds can change a successful optimizer path. Therefore **do not require new bounded fits to be bitwise identical to historical unconstrained fits**, and do not hide differences by copying old backgrounds or parameters into the new result. Require exact replay on the unchanged historical route; separately measure new-minus-historical peak, centroid, widths, constant, brightness operand, and candidate membership.

The completed three historical ten-cycle BASE replays need not be rerun. Reuse their authenticated evidence. This work launches no new PTC/FRUIT trajectories; a later trajectory comparison must account for any material BASE-POINT-versus-historical-BASE change rather than attribute it to correlations.

## 6. Bounded checks before recalibration

Use focused tests and the existing saved cases, not a new survey of algorithms.

**Numerical and decision tests.** Cover strict S/N and width boundaries, inclusive 20-arcsec radial boundary, sigma/FWHM/pixel conversion, positive amplitude above background, constant/no-source maps, finite nonconvergence, stale-output clearing, missing inputs, and ordinary circular Gaussian fits. Check the amplitude sensitivity with local map perturbations and the covariance contraction against a direct small calculation. Reuse existing correlated-field tests; no new Monte Carlo campaign is required.

**Recorded diagnosis cases.** Replay all 18 originally diagnosed outcomes with the new route, including the nine failed blank fits and successful controls. Record old/new status, evaluations, widths, amplitude, uncertainty, S/N, centroid radius, and candidate support. Do not perform another doubled-budget probe. A blank fit may converge and then fail the source criteria; that is normal.

**Bright-source regression.** On the authenticated REF M0, T1, T3, T5, and T10 maps for each array, compare the new result against the saved historical measurement and candidate. These are the expected bright-source controls, not threshold-tuning targets. Also test the original noiseless blank and core-only scene means. The noiseless blank must yield an empty result, and the noiseless core must qualify for that array's source-recognition check to pass.

If an expected bright control fails, report the exact cause without weakening the rules. Continue independent diagnostics for other arrays, but mark that array's repair check as failed and do not present a successful source-retaining screen for it. Do not compensate by lowering S/N, changing covariance, moving the center, or altering the synthetic core.

Review the small implementation and these realized checks before the full replay below. Routine implementation details are not new owner-selection questions; document them and proceed within this order.

## 7. Repeat calibration and saved-map screening, then stop

For each array whose input/covariance and implementation checks are usable, rerun the existing **1,024 calibration and 1,024 validation noise fields**, including both original blank and original core-only scenes. Reuse the exact original seeds, generator, `C0`, grid, masks, and scene means. Reuse a fit across policies on the same scene. Recalculate outcomes under the new rule; never relabel old failures without executing it.

Freeze the new method before the replay. The validation fields remain separate from the calibration fields, but they have already been examined during diagnosis: describe this as **replayed validation, not a new untouched holdout**. Do not add fresh independent fields or tune the rule after inspecting the replay.

For every calibration field/policy, combine the two scene maxima exactly as before. A rejected scene contributes `-infinity`; if the other scene has valid anchors, its maximum still contributes. Computationally unresolved draws are not dropped or replaced and make dependent calibration unavailable.

Retain the original per-array/per-policy threshold rule:

```text
tau = max(3, the 1,015th smallest of the 1,024 combined calibration maxima)
```

Freeze thresholds before evaluating the replayed validation outcomes. Retain strict score comparisons, the original one-sided 95% binomial upper-bound calculation, the `U <= 0.02` diagnostic screen, and the original no-supplemental-pixels checks on noiseless scenes. Repeat the existing 1.25-noise-amplitude stress check on the first 256 validation fields without recalibrating or adding draws.

Report **core-source acceptance alongside false supplemental admission**. Use the same completed core-scene replays to compare predicted amplitude uncertainties with the empirical refitted-amplitude scatter, before the S/N acceptance cut and with all fit failures disclosed. This is an approximation check, not permission to rescale uncertainties until a desired acceptance rate is reached. Ordinary core fit rejection is a recorded miss with empty output, not an unknown score; it must not be hidden. A small false-admission count produced by rejecting the bright cores is not evidence of useful admission. Preserve counts of all acceptance, quality-rejection, finite-nonconvergence, and computational-error outcomes. No new completeness threshold or automatic feedback launch is selected here.

Use the newly frozen thresholds to screen the saved REF M0/T1/T3/T5/T10 maps with BASE-POINT, CORR, and DIAG as **unapplied** candidates. Show source acceptance, baseline mask, extra proposed/final connected pixels, and their measured brightness. Preserve unavailable tests and differences from the historical BASE. An unavailable threshold means an unavailable new-policy screen, not permission to use 3 as an uncalibrated fallback.

This order explicitly permits recalibration after the declared source-policy change; it does not permit repeated tuning until calibration passes. Stop on failed statistical screens for dependent policy/array comparisons and finish independent unblocked results.

**Stop after these results and their review.** The prior work order's full feedback-trajectory permission is not carried forward automatically into this amended selection method. No new RTC/CAL/PTC learning/application, map accumulation, FRUIT trajectory, m10 application, JINC work, or production/default change is authorized here.

## 8. Deliverables and completion

Write a fresh bundle, for example:

```text
successor-point-152389-expected-source-guard-20260925/
```

Use existing provenance and review conventions. Preserve the original bundles, verify consumed inputs, and commit reviewed changes locally without merge/push or unrelated cleanup. Run affected focused, successor/baseline, native/CLI, and required configuration checks as applicable; identify what was actually exercised.

Return a short owner-facing assessment containing:

1. The exact implemented acceptance and amplitude-uncertainty definitions.
2. Per-array old/new blank outcomes, bright-control acceptance, evaluation costs, and fit/candidate differences.
3. Complete calibration, replayed-validation, and stress counts, thresholds, statistical-screen status, and source-miss rates.
4. Matched saved-map candidate figures for BASE-POINT/CORR/DIAG, plus one justified next action recommended only.

Use useful parallelism across independent arrays/trials, share fits and covariance work, and avoid nested library threads. Record measured wall time, evaluation counts, and memory without launching a general optimization project.

**Completion means a working, explicitly bounded POINT acceptance rule and an honest recalibrated admission screen. It does not mean every blank Gaussian fit converges, every bright synthetic realization passes, or correlation-aware feedback has been shown to improve the map.**
