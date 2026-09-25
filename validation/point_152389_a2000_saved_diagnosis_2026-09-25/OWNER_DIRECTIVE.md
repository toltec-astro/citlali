# Codex directive: POINT 152389 — a2000 saved-product diagnosis

**Owner:** Grant Wilson  
**Date:** September 25, 2026  
**Status:** Implement the bounded analysis, execute on saved products, assess, independently review, and stop.  
**Scope:** The completed guarded BASE-POINT / CORR / DIAG comparison; a2000 seed persistence, compact signed response, and their practical measurement consequences. No new scientific reduction or feedback application.

> We have demonstrated better injected-shoulder recovery, but a2000 CORR also retains more dependence on the starting model and changes the compact response. Determine where those differences enter, how they evolve, and how large they are in measurements we care about. Do not treat 1.02% of an artificial shoulder as a 1.02% error in the whole source. Diagnose before changing the method.

## 1. Authority and exact parent

Read applicable repository instructions and recover the sealed completion, assessment, numerical tables, fit caches, saved candidates/evidence, reviews, and manifest for:

```text
/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/
  successor-point-152389-guarded-feedback-comparison-20260925/
```

Discovery anchors are local completion commit `26a3047c6`, branch `codex/map-successor-first-forward-001`, and owned worktree `/private/tmp/citlali-point-additive-20260923`. These identify the reported work; verify the actual checkout, source identities, and parent bindings rather than assuming current HEAD is that commit. Do not touch unrelated work.

Recover these authorities and the implementations/products they bind:

- `CODEX_POINT_152389_GUARDED_FEEDBACK_COMPARISON.md` and its completed handoff/assessment.
- `CODEX_POINT_152389_CORRELATION_AWARE_ADMISSION.md` and `CODEX_POINT_152389_EXPECTED_SOURCE_GUARD.md`, for exact evidence, selection, brightness, and guard semantics.
- `CODEX_POINT_152389_RESPONSE_AND_ROBUSTNESS_CD.md`, for the archived interventions and response/seed measurements.

The supplied closeout reports a2000 shoulder gain/distortion of approximately 0.84988/0.22117 for BASE-POINT, 0.99210/0.11327 for CORR, and 1.00450/0.13126 for DIAG; CORR seed projection 0.010234 versus approximately 6e-6 for the others; increased compact sign asymmetry under both supplements; and worsened positive-compact distortion under CORR. These are discovery checks, not new targets or substitutes for the archived full-precision results.

This order authorizes **analysis of the completed experiment**, not its re-execution or modification. The previously recommended follow-up is now authorized only within the bounds below.

## 2. Allowed calculations and hard exclusions

Permitted work is reading/authenticating saved products; extracting cached measurements and selection records; deterministic differences, sums, dot products, norms, fixed-aperture calculations, and mask bookkeeping; and generating tables and figures. Use the existing diagnostic definitions, physical coordinates, units, masks, and domains. Add only the small analysis/reporting code needed.

**No new source fits, covariance fits, candidate-builder runs, policy evaluations on additional maps, PTC learning/application, AST projection, MAP accumulation, RTC/CAL processing, simulations, signal injections, counterfactual reductions, or feedback applications.** Do not invoke a report path that silently refits on a cache miss. No cycle 11 and no application of m10.

An algebraic audit of an already recorded score is allowed when its exact saved operands are complete. It may verify the recorded numerator, variance, and threshold comparison; it must not rebuild missing selection/weight state, invent scores for unavailable tests, or publish a new candidate. Missing cached inference products make the affected diagnostic unavailable; they do not authorize regeneration.

Keep thresholds, full C0, K0, neighborhood weights/rules, source guard, background convention, PTC recipe, calibration, map representation, and feedback semantics unchanged. No parameter sweep, tuning, damping, reconstruction, new rejection rule, or production change.

## 3. Analysis matrix and comparison discipline

Use the **15 existing a2000 histories**: three policies times REF, CORE-PLUS, CORE-MINUS, SHOULDER, and SEED-SHOULDER. Inspect M0 and all saved T1–T10 maps, and the relevant m0–m10 candidate records. Reuse existing a1100/a1400 summary results only as brief context; do not open another all-array campaign.

Every branch difference is against its **own policy REF at the same stage**. Compare these within-policy differences across policies; never use another policy's REF to define seed memory or injection response. Retain the parent's exact comparison vector, normalization, and supported physical domain separately for maps and candidates. Do not assume a seed raster, its projected map reference, and a fitted source template are interchangeable.

Make the temporal relationship explicit throughout:

```text
m_(k-1) was applied to produce T_k.
m_k was selected from T_k and could affect only T_(k+1).
m10 is a saved, unapplied candidate; it cannot explain T10 as an applied input.
```

m0 differs intentionally in the seed intervention. Distinguish that imposed difference from subsequent unforced selection/brightness differences. Equal masks or equal pixel counts do not imply equal candidates.

Before detailed analysis, reproduce the relevant archived aggregate numbers by extraction or allowed arithmetic, using existing numerical tolerances. Preserve original-domain results. Show any support mismatch explicitly; do not zero-fill unavailable pixels or silently replace the benchmark with a favorable subset.

## 4. Seed history: where does dependence persist?

For each policy h and stage k, use or form from saved products:

```text
S_h,k = T_h,k(SEED-SHOULDER) - T_h,k(REF)
dm_h,k = m_h,k(SEED-SHOULDER) - m_h,k(REF)
```

Track the parent's seed-shaped projection, total seed-difference norm, and residual not parallel to the seed, in both maps and candidates. Reuse the parent's exact normalization. For seed-only differences the desired initialization-independent result is **zero**, unlike the data-injected shoulder, whose desired response is its injection reference. Do not use the injected-signal distortion-to-unit-response metric as a seed-memory error metric.

Plot all saved stages and provide compact checkpoints at M0, T1, T3, T5, T9, and T10. Place the data-injected SHOULDER recovery alongside the seed-only history without conflating their normalizations.

Report whether the difference decreases, grows, changes sign, or switches over the recorded interval. Show absolute T9-to-T10 changes in the seed difference, candidate difference, and cached source measurements; also show each branch's own terminal change. A small change in their difference can coexist with drift shared by both branches.

Use ratios only with a usable denominator. Do not fit an exponential, infer a contraction rate from tiny ratios, extrapolate to later cycles, or declare distinct asymptotic fixed points. Describe a plateau or continued decay only over the observed stages.

## 5. Admission and brightness: locate the first relevant divergence

### 5.1 Stage-by-stage selection record

For SEED-SHOULDER versus its own REF, identify separately the first post-intervention difference in baseline membership, proposed supplemental membership, final source-connected membership, and selected brightness on common membership. Record the corresponding applied candidate and next map. Then compare that sequence across BASE-POINT, CORR, and DIAG.

For each stage retain pixel identities and counts for baseline-only, supplemental-only, rejected-after-connectivity, newly selected, dropped, and shared selected pixels. Distinguish final supplemental membership from proposals that never reached the feedback model. Track baseline/supplemental transitions: a pixel can be admitted by the supplement early and later pass the ordinary baseline rule.

For the implicated pixels, collect available saved brightness, fitted constant, pointwise threshold margin, neighborhood numerator, variance, score, threshold margin, full-footprint validity, weight concentration, exclusion state, and connectivity outcome. Preserve negative neighbor contributions. Show actual angular positions relative to the source and seed.

Detailed figures should cover T1, T10, and the first relevant unforced divergence when distinct. Retain complete tables rather than selecting only pixels that support a preferred explanation. With the archived weights/evidence, identify whether a crossing accompanies a brightness/background change, changed baseline exclusions/normalization, a variance change, or a connectivity change. If the required records are absent, state exactly which explanation remains unresolved.

### 5.2 Exact separation of mask and brightness differences

On common valid support, use the exact saved operand b and final selected masks M. For k >= 1, partition the candidate difference into disjoint terms:

```text
shared membership:  1_(M_seed intersect M_ref) * (b_seed - b_ref)
seed-only members:  1_(M_seed minus M_ref) * b_seed
REF-only members: -1_(M_ref minus M_seed) * b_ref
```

Verify that their sum reproduces the saved candidate difference. Treat the imposed m0 shoulder separately; it is the deliberate exception to ordinary measured-brightness publication.

Report the signed contribution of these terms to the existing seed projection and fixed-aperture differences, together with their norms. Norms need not add, and signed terms can cancel. Where the saved brightness convention permits it, separately expose the contribution from the **already fitted** constant by exact arithmetic; do not refit or relevel the maps. This is bookkeeping of the realized candidate, not a causal decomposition into independent physical sources.

### 5.3 Correlation evidence versus trajectory history

Use the existing **nine a2000 REF T10 same-map cross-candidates** where authenticated: each policy evaluated on each policy's REF endpoint. They distinguish selection differences on identical brightness from differences between already diverged feedback histories. They were never applied, and remain unapplied. Do not generate equivalent cross-candidates on seed or compact maps.

CORR and DIAG use the same neighborhood numerator at the same fixed state, but different variance expressions and separately calibrated thresholds. Compare the saved quantities and margins, not the raw threshold numbers alone. Across different states, the weights can also change because the baseline exclusions change. Do not attribute every score difference to covariance.

Inspect already saved PTC centering, state/support, cleaned-versus-restored contributions, and accounting summaries for the implicated transitions. Explain the recorded chain from applied model to residual cleaning to next selection only as far as those records support it. No fresh basis application, SVD, or hybrid/recombined trajectory. Distinguish direct evidence, an explanation consistent with it, and a causal claim that would require a new intervention.

## 6. Compact response: is the same behavior involved?

Use the saved CORE-PLUS/MINUS differences from each own-policy REF, corresponding injection-only references, response metrics g/eta/rho, and even-response maps:

```text
D_plus  = T(CORE-PLUS)  - T(REF)
D_minus = T(CORE-MINUS) - T(REF)
E       = T(CORE-PLUS) + T(CORE-MINUS) - 2*T(REF)
```

The minus case is a decrement to an already bright source, not negative-source detection. Check the archived signed-reference relation and support before interpreting E; expose any nonzero ideal even contribution instead of assuming it away.

Show where CORR's positive-compact residual and the supplements' sign asymmetry appear and how they evolve. Compare their stage timing, pixel locations, mask transitions, background changes, and available cleaning-state records with the seed diagnosis. Establish whether the same pixels/transitions are implicated, different mechanisms are indicated, or the saved evidence cannot distinguish them. Spatial coincidence alone does not prove a shared cause.

Use cached T10 ideal-addition fits to separate changes in the actual reduction response from the descriptive fitter's response to an ideal added component. Reuse saved fitted peaks, physical centroids, gauge-safe widths/shapes, constants, template measurements, and fixed apertures. Retain constrained/failed measurement flags. Do not refit full maps, signed differences, or ideal controls, and do not create earlier-stage ideal-fit controls.

Keep the data-injected shoulder benefit visible alongside these costs. Neither a gain near one nor a larger admitted mask establishes uniformly better source recovery.

## 7. Put the differences on practical measurement scales

For seed dependence, compact-response changes, and the shoulder benefit, report absolute effects first: brightness in the inherited mJy/nominal-beam units, centroid shifts in arcseconds, saved width/shape changes, fitted constant, and raw/fitted-background-separated fixed-aperture quantities. Use archived aperture centers, radii, support, and pixel-area factors.

Give percentages only with explicit denominators: original seed normalization, compact injection, each policy's REF fitted peak/width, or its matching aperture measurement. Do not divide by a negligible quantity or describe brightness-area integrals as calibrated total source flux without the established conversion.

The reported seed coefficient 0.010234 is a fraction of the **artificial shoulder**, not the whole source. Its nominal peak normalization was 0.05 times the original reference peak. Their product, about 0.000512 or 0.0512%, is only an illustrative seed-aligned peak scale before accounting for the actual archived template/projection and other residual structure. Recover the actual template and separately report:

- the seed-parallel component and the remaining spatial residual;
- the maximum absolute full-map difference on the declared domain;
- the cached fitted-peak difference and fixed-aperture/background effects.

These quantities are not interchangeable. In particular, the seed projection is not an upper bound on total error. Also do not dismiss a small seed-aligned component while ignoring a larger compact or aperture effect.

The approximately 5% final calibration/PSF goals are context, not automatic pass criteria for these post-RTC/CAL perturbation tests. Give the owner the actual scales; do not invent a combined score or new acceptance threshold. Keep the a2000 noise-model/source-transfer qualification explicit. Frozen-C0 guard errors are conditional reference-scale quantities, not newly validated endpoint errors. No significance claim for paired adaptive differences from adding independent variances, and no conversion of initialization sensitivity into a random error bar.

## 8. Implementation, checks, and bounded evidence handling

Before analysis, bind the exact consumed files, their parent manifest entries, cached measurement definitions, code revision, and commands in a small prospective record. Verify bytes for the files actually consumed and confirm they remain unchanged afterward. Reference the existing sealed parent; do not duplicate its roughly 125 GB payload or repeat an unrelated full-parent review merely to close this derivative diagnosis.

Keep the executable route read-only with respect to the parent. Check that cached-only reporting cannot fall through to fitting, selection, or processing. Add focused tests for arithmetic identities, policy/stage matching, mask partition closure, signed controls, zero-versus-unavailable semantics, and m9/T10/m10 labeling. Reuse existing upstream validation evidence; do not trigger a broad native build or rerun the science campaign for reporting-only changes.

A cache miss or unsupported diagnostic remains an explicit limitation. A hash mismatch or failed arithmetic identity blocks affected conclusions; preserve it and finish independent valid work. Do not repair the sealed parent or relax tolerances.

Use compact arrays and useful I/O/analysis parallelism without nested oversubscription. Avoid repeatedly loading large products or building new workflow infrastructure. Record actual analysis/rendering/verification runtime and measured memory scope without presenting overlapping work as elapsed time.

## 9. Deliverables and completion

Create a fresh derivative bundle, for example:

```text
successor-point-152389-a2000-saved-product-diagnosis-20260925/
```

Deliver one owner-facing assessment with concise answers to:

1. Where does a2000 CORR's seed dependence first differ from its matched controls, and which saved admission/brightness/cleaning records explain it?
2. Is the difference still decreasing, approximately stationary, or switching within the recorded late stages?
3. Does the same behavior explain the compact-positive distortion and sign asymmetry?
4. How large are all these effects in source measurements, distinct from their injection/seed-normalized coefficients?

Include machine-readable all-stage tables, a short implicated-pixel ledger, and inspected figures showing seed histories, matched signed seed/candidate differences with membership overlays, compact signed/even residuals, and practical measurement effects. Use consistent angular scales and comparable signed color ranges; label any magnification or constant subtraction. Reuse existing plots when adequate. Do not turn this into another exhaustive gallery or repeat the parent report.

Provide a completion/availability table for the requested analyses, reproduction instructions, input references, focused checks, and an independent review of numerical bookkeeping, interpretation, and figures. Seal only the new derivative products. The review may verify saved-product arithmetic, not rerun forbidden inference or processing. Report zero new scientific fits/processing/candidate evaluations from the actual execution path, not a fabricated counter.

Conclude with the supported explanation and remaining uncertainty. It is acceptable to find that the effects are small on practical scales, consequential, or unresolved by the retained evidence. Recommend **one** next bounded action based on the findings, but do not execute it.

## 10. Stop

Stop after the saved-product diagnosis, assessment, review, and derivative evidence seal. Commit the reviewed diagnostic/reporting changes locally using the existing worktree practice and report the final commit, worktree status, and bundle.

No method adoption, default change, merge, push, destructive cleanup, new source/seed, new observation, JINC work, retuning, new fits, new candidate evaluations, new projection/mapmaking/cleaning, new feedback trajectory, applied m10, or cycle 11.

**Completion means an evidence-based explanation of the existing a2000 tradeoff and its practical scale—not another attempt to improve the result during diagnosis.**
