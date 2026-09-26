# Codex directive: POINT 152389 — a2000 three-pixel m1 counterfactual

**Owner:** Grant Wilson  
**Date:** September 25, 2026  
**Status:** Implement, validate, execute the bounded pair, assess, independently review, and stop.  
**Scope:** a2000 only; ordinary MAP; one targeted CORR seed-branch intervention and one unchanged replay control, continuing from saved m1 through T10.

> Test whether the earliest three CORR-only admitted brightness contributions materially influence the later seed-dependent result. Remove them once from the saved seeded m1, leave every other value unchanged, and let the normal method evolve. Measure both the causal change and the remaining difference from the unseeded reference. Do not optimize the outcome or mistake this seed-path experiment for a compact-photometry validation.

## 1. Scientific question and new authorization

The completed saved-product diagnosis identifies an early policy-specific seed-selection difference in m1, followed by map differences at T2. Its recommendation is to remove the three CORR-only m1 pixels **790, 882, and 1003** from the CORR SEED-SHOULDER branch and inspect the subsequent history.

This order authorizes that new intervention, with a matched **unchanged-m1 replay control** to validate the continuation path. It supersedes the previous saved-product-only prohibitions solely for the operations and pair specified here. This is an execution order, not a request for another proposal.

The primary questions are:

1. Does removing these three contributions once reduce, increase, or redirect the late initialization-dependent map and candidate differences?
2. Are the three pixels admitted again by ordinary CORR selection, when does that happen, and how does the later nine-pixel difference change?
3. Are the effects material in the fitted peak, position, widths, template amplitude, aperture measurements, and local spatial residuals?

The intervention was selected from an exploratory diagnosis and is now fixed before execution. It is not a statistically independent validation sample. It tests the **joint removal of three measured-brightness contributions**; it cannot separate each pixel's influence or a binary membership effect from the brightness removed with that membership.

The existing compact-response and genuine shoulder-recovery results remain context. This test cannot establish that their tradeoff is repaired, even if seed dependence becomes smaller.

## 2. Recover and bind the accepted parents

Read applicable repository/governance instructions and recover the exact completed products, source identities, and method definitions. Primary authorities are:

- `CODEX_POINT_152389_A2000_SAVED_PRODUCT_DIAGNOSIS.md` and its completed assessment, pixel ledger, results, reproduction record, reviews, and seal.
- `CODEX_POINT_152389_GUARDED_FEEDBACK_COMPARISON.md` and its completed trajectories, candidate/fit caches, native receipts, prospective method bindings, reviews, and seal.
- The expected-source-guard, correlation-aware-admission, and original C+D directives and bound implementations, only as needed to recover unchanged semantics and original operands.

Discovery anchors supplied by the completed records:

```text
Saved-product diagnosis completion:
11b4d2c07a81cc5876bd3da833c4168731e34aea

Guarded-comparison completion:
26a3047c654386e38adac860b91491136888bc64

Bundles under:
/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/
  successor-point-152389-a2000-saved-product-diagnosis-20260925/
  successor-point-152389-guarded-feedback-comparison-20260925/

Repository records:
validation/point_152389_a2000_saved_diagnosis_2026-09-25/
validation/point_152389_guarded_feedback_comparison_2026-09-25/

Previously reported owned worktree / branch:
/private/tmp/citlali-point-additive-20260923
codex/map-successor-first-forward-001
```

Confirm the actual worktree and identities rather than assuming current HEAD or a filename supplies the correct parent. Preserve unrelated work. Use the exact accepted native implementation and numerical environment; extend only the narrow experiment runner, intervention record, necessary diagnostic logging, and report.

Authenticate the consumed parent bytes against their sealed records, including the immutable post-RTC/CAL pre-PTC data, native AST coordinates, CORR seeded T1/m1, original CORR REF history, full-precision threshold, C0/K0, measurement definitions, and relevant processing state. Preserve the existing cached source-guard acceptance of T1; do not refit T1 or rebuild its ordinary m1.

Before observation execution, freeze a compact prospective record of these inputs, the three pixel identities, branch definitions, numerical comparison rules, permitted calculations, and stop conditions. Recover the diagnosis's exact definition of “CORR-only” and name its comparator; do not silently reinterpret that term as a different seed-versus-REF or policy comparison.

Reported discovery checks include CORR's T10 seed projection 0.010234, peak difference −0.103601 mJy/nominal beam, maximum full-map difference 4.706088 mJy/nominal beam, and large-aperture difference +173.555237 in the inherited brightness-area units. Use full saved values and definitions, not these rounded figures as numerical targets.

## 3. Exact test matrix and execution ceiling

Both branches inherit the archived CORR SEED-SHOULDER history through T1. Both use the **same original observed pre-PTC data y**, with no new data injection. The original artificial shoulder entered only the historical m0; its first-cycle effect is already present in the shared T1 history.

| Branch | Applied model for T2 | Models for subsequent applications | New feedback applications |
|---|---|---|---:|
| `SEED-M1-REPLAY` | Exact archived CORR SEED-SHOULDER m1 | Newly generated normal CORR replacement models | T2–T10: 9 |
| `SEED-M1-CUT3` | Copy of that m1 with exactly the three specified contributions removed | Newly generated normal CORR replacement models; no persistent exclusion | T2–T10: 9 |

**Ceiling: two continuation branches and 18 new observation-feedback applications.** M0 and T1 are reused, not rerun. There are no new REF, DIAG, BASE-POINT, compact-injection, or data-shoulder trajectories. Failed or aborted observation applications must be counted; the ceiling is not permission for automatic retries. Stop and report if a scientific replay failure or exhausted allowance prevents completion.

Run the unchanged replay first and validate it through T10 before launching CUT3. Reuse its validated new products as the immediate matched control and the archive as its replay oracle. Do not replace the replay with copied historical maps or feed archived later candidates into it: it must regenerate m2–m10 using the current, unchanged continuation path.

Each new full map receives the existing single source fit/guard and normal CORR candidate calculation once, shared by all its downstream diagnostics. At most 18 new full-map source fits are needed. Cache them for reporting; do not repeat inference to change labels or render figures. No new ideal-addition fits, covariance fits, or simulation ensembles are authorized.

Chronology is absolute, not renumbered from the restart:

```text
archived T1 -> archived ordinary m1
                      |
             exact copy / one-time CUT3 override
                      |
                     T2 -> normal m2 -> ... -> normal m9 -> T10 -> m10 [unapplied]
```

No m10 application, T11, longer convergence run, additional pixel subset, or amplitude sweep is authorized.

## 4. Exact one-time intervention

Resolve pixel IDs `{790, 882, 1003}` using the archived ledger's indexing and geometry. Do not assume they are raw flattened FITS indices or transplant them to another raster. Before launch record each ID, its exact array coordinates, physical east/north position, ordinary m1 brightness, validity, baseline/supplemental membership, and available saved score/threshold margin. Verify they are the three final admitted CORR-only members identified by the diagnosis, not rejected proposals or nearby substitutes.

For valid candidate pixels p, define:

```text
J = the three authenticated pixels
m1_REPLAY(p) = archived_m1_SEED(p)
m1_CUT3(p)   = 0                       for p in J
              archived_m1_SEED(p)     otherwise
```

Remove those three members from the **applied feedback model**, setting their model contribution to intentional zero while leaving their observational/sample validity unchanged. Preserve all retained brightness values exactly, including background convention, gain, support outside J, and unavailable values. Do not delete samples, change hits/weights, refit a background, renormalize remaining brightness, subtract an analytic shoulder, or copy a DIAG/REF model into this branch.

This is an explicit **one-application override after the ordinary m1 was constructed**. Retain the ordinary candidate and its provenance separately from the edited applied candidate. Record exactly what changed; do not represent CUT3 as an untouched output of normal CORR selection or reuse the original model hash.

Do not run another connectivity filter or otherwise prune additional members after deletion. If deleting J disconnects other retained members, record that fact and keep them unchanged under this narrow diagnostic override. No general connectivity/source-validity rule is weakened for ordinary candidates. All native numerical, projection, and observed-data validity rules still apply.

The edit expires immediately after its application to produce T2. From m2 onward, run normal CORR selection with no special knowledge of J. The pixels may return as baseline or supplemental members. Never force them back in, blacklist them, or accumulate prior masks. A return in m2 can affect T3, not T2; a return only in m10 has no applied consequence within this experiment.

## 5. Unchanged scientific method

Retain the accepted guarded-comparison recipe exactly:

- Ordinary MAP, original 2-arcsec candidate grid and AST-centered 40-arcsec feedback disk, direct containing-pixel model projection, unity gain, and complete candidate replacement.
- Original frozen full C0/K0 and threshold-3 baseline noise scale; exact archived CORR threshold, neighborhood weights/scale, exclusions, signed-neighbor treatment, availability predicates, strict comparisons, positive anchor brightness, and final ordinary source connectivity.
- The same POINT-EXPECTED Gaussian-plus-constant fit and guard: finite/converged fit, positive amplitude, radial centroid displacement at most 20 arcsec, both principal sigmas strictly above 1.5 arcsec, and amplitude S/N strictly above 10. Preserve all seven free parameters, weights, initialization, bounds, rank handling, and the inherited 3,000-evaluation ceiling. Propagate the same frozen full-C0 reference-scale error.
- The accepted per-network/processing-segment observed-entry rank-10 PTC, centering, learning/application, solver/stopping, coefficients, eligibility, and normalization conventions. Paired-r behavior remains inert as in the parent.

For each k = 2,...,10, start again from immutable y; project and subtract only that branch's current m_(k−1); freshly learn/apply PTC on that residual; restore exactly the subtracted operand; make T_k; then fit/guard and build the normal replacement m_k.

Fresh residual-specific PTC learning/application, native model projection, ordinary mapmaking, and unchanged full-map fitting/admission are expressly authorized **within this pair**. Do not reuse old learned bases on changed residuals, freeze the later selection history, subtract from a previously cleaned residual, or introduce a separate seed restoration term. Any inherited state needed to reproduce the numerical route must be bound and reproduced, not silently reset.

There is no new RTC/CAL reduction, covariance adaptation, noise-threshold calibration, reconstruction, damping, gain change, filter change, JINC work, or source/seed-shape change.

## 6. Preflight, replay, and numerical controls

Before observation execution, independently review the exact intervention and restart route. Use existing small fixtures plus focused tests for an empty edit, exact three-value removal, preservation of all other values/validity, intervention expiry, possible ordinary re-admission, source-failure clearing, and stage ancestry. These are small implementation tests, not new observing-field simulations.

**Replay gate.** SEED-M1-REPLAY must reproduce archived CORR SEED-SHOULDER T2–T10 and m2–m10 under the archived field-specific rules: maps, measured brightness operands, masks, measurements/flags, support/normalization, and native centering/learning/application/accounting. Check exact identities wherever the accepted deterministic route promises them; use only inherited numerical tolerances elsewhere. Raw basis-vector differences must respect the parent's established sign/gauge/subspace comparison rules.

The inherited 1e−8 brightness accounting/remap tolerance is not a universal tolerance for every fit parameter or product. Bind appropriate existing rules before examining the result. Do not relax them after a mismatch or numerically subtract replay drift to manufacture an intervention result. Failure of the replay gate blocks CUT3 and any causal conclusion.

**Intervention gate.** Demonstrate that m1_CUT3 minus m1_REPLAY is exactly the negative of the three saved contributions and zero elsewhere on valid support; input data and all other scientific settings are unchanged. Preserve separate original, selected, overridden, and applied identities.

**Each new stage.** Retain the normal subtract/restore/remap checks, guard outcome, processing statuses, and model/observational support. Legitimate supported downstream changes are recorded, not forced to match the control. Missing data, invalid projection, failed accounting, a new failed processing segment, or inherited feedback-boundary contact block continuation under the established rules.

A required source-guard failure preserves the map and diagnostics, returns explicit failed-fit/empty accepted products, and stops that branch before another application. No stale candidate, relaxed bound, increased fit budget, or alternative initialization is allowed.

A scientifically unfavorable or negligible effect is a result, not a retry trigger. Reporting failures should reuse saved completed science; they do not authorize repeating applications.

## 7. Measurements and diagnostic evidence

Let R_k be the archived **unseeded CORR REF**, S_k the validated SEED-M1-REPLAY, and C_k the CUT3 map at the same stage. Report three distinct quantities:

```text
Original seed dependence:    S_k - R_k
Remaining seed dependence:  C_k - R_k
Effect of the intervention: C_k - S_k
```

Verify `(C_k - R_k) = (C_k - S_k) + (S_k - R_k)` on declared common valid support. Apply the analogous accounting to candidates, treating the imposed m1 edit separately from normal m2 onward. Use the original archived seeded trajectory as an additional replay overlay, not a different policy's REF.

### 7.1 Spatial residuals and finite-cycle behavior

For all stages, report the existing signed seed-shape projection, seed-parallel and orthogonal components, full residual norm, maximum absolute difference and its physical location, and candidate membership/brightness differences. Keep the archived seed raster and any projected response reference distinct; do not renormalize them to the new endpoint.

Preserve the original full-map benchmark and seed-projection domains separately. The diagnosis reports 10,601 full-map pixels and 1,257 finite seed pixels; authenticate these rather than assuming all masks match. Any support change requires explicit lost/gained support and a separately labeled common-domain comparison. Never fill missing data with zero or trim a residual to improve a metric.

Show T2, T3, T5, T9, and T10 checkpoints while retaining every stage. Inspect each branch's own late map/candidate drift as well as the drift of their difference. Do not extrapolate a fixed point, fit a decay law, choose an earlier stopping stage retrospectively, or run extra iterations.

### 7.2 Pixel re-entry and admission/brightness accounting

Track J and the nine archived late seed-only members individually. Record first re-entry, baseline versus supplemental status, proposed versus connected admission, brightness, and applicable score margin. Include all other newly differing members in a complete machine-readable ledger; do not restrict the result to the initially suspected pixels.

For m2 onward, separate changed brightness on shared accepted support from contributions of members unique to each branch. Use the existing exact mask partition and stored-background bookkeeping. Equal counts or equal masks do not imply equal models. Signed projection/aperture contributions can cancel; component norms must not be added as if they were signed contributions.

The previous diagnosis lacked individual neighbor-weight records. During **already-required new CORR evaluations only**, add narrow logging for the fixed union of J and the nine archived late members: actual neighbor IDs/normalized weights, signed brightness contributions, numerator, variance, score and full-precision margin, and existing availability/exclusion/connectivity outcomes where computed. This is observation of the unchanged calculation, not an additional policy evaluation or adaptive diagnostic selection. Do not compute unavailable scores solely to populate the log. Demonstrate that instrumentation leaves the replay unchanged.

Do not rebuild missing T1/m1 evidence or evaluate alternative policies on the new maps. Distinguish newly logged evidence from unavailable historical evidence, and recorded processing changes from an unperformed counterfactual basis analysis.

### 7.3 Practical source measurements

Reuse each new stage's one cached full-map fit and the inherited fixed-template/aperture estimators. Report absolute and explicitly normalized changes for fitted peak, east/north centroid, principal widths/gauge-safe shape, constant, template amplitude, and all original fixed apertures. Separate raw brightness-area and fitted-background contributions.

For T10, put original seed dependence, remaining seed dependence, and intervention effect beside one another. Do not report only the shoulder coefficient: it is neither a whole-source flux error nor an upper bound on local spatial differences. Aperture units remain the inherited mJy/nominal beam arcsec², not automatically calibrated total flux.

Retain the a2000 source-location covariance-transfer qualification. Frozen-C0 guard errors are conditional reference scales, not new endpoint uncertainties or a probability law for these paired deterministic differences. No independent-variance significance, quadrature addition to old errors, or adaptive false-feedback rate is inferred.

## 8. Interpretation and the unresolved compact-response tradeoff

With exact intervention isolation and a passing replay, C−S measures the finite-run downstream effect of **this joint one-time edit** in the configured procedure. It does not isolate whether later cleaning, brightness changes, or selection mediates that effect, or which individual pixel is responsible.

Interpret without a post-hoc pass threshold:

- Reduced shoulder projection is not sufficient if orthogonal/local residuals or aperture effects grow. Assess all retained measurements.
- Smaller C−R establishes less initialization dependence in this tested continuation, not proof of astronomical truth or permission to suppress these pixels in production.
- Similar late endpoints do not establish that the edit had no transient effect, nor that all early selection choices are irrelevant.
- Pixel re-entry does not by itself certify genuine sky or false feedback. Examine its measured consequences and chronology.

Keep a compact context table from the **unchanged archive**: the improved genuine-shoulder response, the compact-positive peak/aperture discrepancies, and DIAG's compact asymmetry despite little late seed dependence. Do not regenerate those branches or ideal fits. Do not claim this experiment resolves the reported 9.28%-of-increment compact aperture discrepancy. The approximately 5% final calibration/PSF goals provide context, not an automatic acceptance gate at this partial boundary.

Recommend one next bounded action based on the findings, without executing it. A small or inconclusive seed effect is a valid conclusion; do not create an endless pixel-removal search before returning to the broader source-response question.

## 9. Resources, reporting, validation, and stop

Use the existing validated environment and useful existing parallelism within independent native segment work. Avoid nested numerical-library oversubscription, unnecessary data copies, and a new optimization project. Record actual allocation, workers/threads, wall/CPU time and measured memory scope. The pair is sequential at the replay gate; do not launch CUT3 speculatively. No environment migration or separate Unity campaign is authorized by this order.

Create a fresh bundle, for example:

```text
successor-point-152389-a2000-m1-cut3-counterfactual-20260925/
```

Keep both parent bundles immutable. Verify consumed bytes before and after execution; bind parent seals and existing reviews instead of copying or rehashing the unrelated full campaign. Seal the new outputs, exact intervention record, source/environment bindings, commands, receipts, cached fits/candidates, diagnostic tables, and reviews. Record actual application/fit counts and all reused or unavailable work, including failed attempts.

Deliver one owner-facing assessment, a complete two-branch status table, all-stage numerical tables, the pixel ledger, and a small set of inspected matched-scale figures. Figures should show the common fork/intervention, spatial and measurement histories, and pixel re-entry; distinguish amplified residual displays and all normalization/domain choices. Preserve signed values and finite constrained/failed measurement flags.

Run focused continuation/intervention/report tests and relevant integration/configuration checks for the touched route. Reuse unaffected parent broad/native evidence; expand checks only where the code change affects it. Independent final review must verify the realized edit, replay, accounting, arithmetic, interpretation, and figures from the new saved products. It must not silently run another observing trajectory outside the ceiling. Preserve a reproducible source/executable binding and verify the derivative seal.

Update the local status/handoff and commit reviewed changes using the established owned-worktree practice. No merge, push, production/default change, or destructive cleanup.

**Stop after this pair, assessment, review, and evidence seal.** No additional pixel subsets, persistent veto, new compact/shoulder injection, new noise realization, new array/observation, covariance adaptation, threshold/rank/gain change, JINC work, applied m10, or cycle 11 is authorized. Missing required evidence or failed replay is a precisely reported bounded outcome, not permission to choose substitute parents or widen the experiment.
