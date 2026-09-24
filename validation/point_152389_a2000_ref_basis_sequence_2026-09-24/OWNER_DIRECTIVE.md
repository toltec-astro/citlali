# Codex directive: POINT 152389 — paired a2000 reference-basis-sequence counterfactual

**Date:** September 24, 2026  
**Status:** NEW FOLLOW-ON WORK ORDER. The C+D campaign and frozen-PTC shoulder decomposition are complete; do not execute either old directive again.  
**Scope:** One paired experiment on a2000 only, using the existing REF data and exact C-shoulder injection.  
**Execution:** Recover and authenticate the inputs, implement the bounded intervention, validate, run the prescribed pair, assess, obtain the required review, and stop. Routine implementation decisions do not require another proposal or approval loop.

> Both branches use the saved original REF basis at each corresponding map-producing stage. Each branch recomputes its own means, application coefficients, maps, source measurements, and ordinary threshold-3 replacement candidates. This is a cycle-varying reference basis sequence—not one basis frozen for all cycles, not frozen numerical means, and not replay of the old shoulder candidates.

## 1. Scientific question and interpretation

Determine how the completed a2000 shoulder response changes when the shoulder is prevented from changing the learned PTC subspace throughout the feedback history, while mean estimation and ordinary feedback selection remain active.

The completed saved-state analysis reports the following a2000 T10 anchors:

| Quantity | Reported value, for orientation only |
|---|---:|
| Adaptive shoulder projection, g | 0.849880964 |
| Total map distortion, eta | 0.221173 |
| Orthogonal distortion | 0.162424 |
| Mean-aware fixed-reference direct projection | 0.819027 |
| Mean-aware historical feedback projection | +0.112053 |
| Historical subspace-change projection | −0.081199 |

Recover full-precision values, maps, domains, and qualifications from the authenticated completed bundle. These numbers are not execution targets or acceptance thresholds.

Removing the old subspace-change contribution at the endpoint would give approximately 0.93108 **while retaining the old feedback history**. That is not a prediction for this experiment: changing earlier cleaning changes subsequent maps and candidates. The purpose is to measure that closed-loop difference rather than infer it from endpoint subtraction.

This is a finite intervention on one observed dataset at the saved post-RTC/CAL, pre-PTC boundary. The original source, atmosphere, and noise remain present. Their treatment can change along with the shoulder response. The saved REF sequence is an offline, dataset-specific control; it is not a deployable basis-selection rule or evidence that ordinary residual-specific relearning should be abandoned.

## 2. New authorization and unchanged boundaries

This work order supersedes the previous saved-state-only prohibitions **only to authorize these two new constrained trajectories**, their ordinary branch-map fits and candidate updates, and the bounded diagnostics below.

No new observation PCA/ALS basis learning is authorized. Import the saved original REF bases. Applying those bases includes recomputing the original mean rule and solving the original application coefficients on each branch's current residual; those operations are required and are not basis learning.

Hold the accepted experiment fixed: original parents and injection; per-network/processing-segment observed-entry rank-10 PTC; original centering and application rules; AST/projection geometry; ordinary forward MAP; coefficients and eligibility rules; seven-parameter Gaussian-plus-constant measurement; fixed template/aperture definitions; nominal-beam convention; unity feedback gain; threshold 3; original frozen admission-noise scale; and complete candidate replacement. Preserve the original source association and connected-admission procedure. The candidate remains the accepted measured-map brightness model, not a fitted Gaussian, injected-truth model, S/N map, or new interpolation model.

The feedback domain remains the AST-centered 40-arcsecond-radius disk. Outside-domain feedback is absent without deleting observed samples or asserting zero sky. Valid excluded pixels inside the domain are intentional model zeros; unavailable pixels remain unavailable. Paired-r behavior stays unchanged and inert as in the parent; there is no r-channel injection or new treatment.

This is a diagnostic exception to the usual residual-specific basis-learning procedure, not an amendment to the production method. No threshold, rank, damping, source model, weighting, admission scale, or default is selected by this run.

## 3. Recover the completed evidence and the entire basis sequence

Read applicable AGENTS.md, owner decisions, and current PTC/FRUIT/MAP bindings. Recover the following original directives and their completed assessments, handoffs, reviews, and manifests:

- `CODEX_POINT_152389_RESPONSE_AND_ROBUSTNESS_CD.md`
- `CODEX_POINT_152389_FROZEN_PTC_SHOULDER_DECOMPOSITION.md`
- `handoff/POINT_152389_RESPONSE_ROBUSTNESS_CD_2026-09-23.md`
- `handoff/POINT_152389_FROZEN_PTC_SHOULDER_2026-09-24.md`

Discovery roots under `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/` are:

- `successor-point-152389-response-robustness-cd-20260923/`
- `successor-point-152389-frozen-ptc-shoulder-20260924/`

Follow their archived references to immutable pre-PTC parents, the exact shoulder raster and projected operand, original REF and C-shoulder trajectories, saved means/bases, admission state, map operators, measurements, and comparison domains. Paths and the reported local completion commit `4120308df` are discovery hints, not substitutes for authenticated identities or numerical-source compatibility.

**Recover REF bases for every map-producing stage: M0 and T1 through T10, for every required a2000 network/processing segment.** The five snapshots used in the decomposition are not a complete execution sequence. Never interpolate between saved bases, carry the previous basis forward across a missing stage, substitute a shoulder basis, or relearn a missing basis.

Before observation execution, authenticate the full required sequence and the replay expectations. If a necessary stage or binding is missing or scientifically incompatible, finish available implementation/fixture validation and report the precise blocker; do not start a known-incomplete substitute trajectory.

Authenticate consumed originals against archived expectations before execution and again at completion. Rehashing whatever is currently at an old path is not sufficient. Verify source/binary compatibility, basis serialization, and numerical equivalence to the native Apply implementation used by the completed work.

Save one prospective manifest binding the two branches, all stage-to-basis mappings, parents, injection, inherited state, candidate/fitting rules, original comparison domain, tolerances, run order, resource policy, and requested diagnostics before inspecting new observation results.

## 4. Fixed experiment matrix and indexing

Run exactly these two logical trajectories:

| Branch | Immutable pre-PTC data | Basis schedule | Initial candidate | Later candidates |
|---|---|---|---|---|
| `REFSEQ-REF` | Original REF parent y | Original REF basis at each corresponding stage | Regenerated normally from this branch's M0 | Regenerated normally at threshold 3 |
| `REFSEQ-SHOULDER` | Original y plus the exact saved shoulder operand u | The same original REF stage-matched sequence | Regenerated normally from this branch's own M0 | Regenerated normally at threshold 3 |

Run `REFSEQ-REF` first as the full replay control. Only after its required replay checks pass may `REFSEQ-SHOULDER` begin. The original adaptive REF and C-shoulder trajectories are read-only comparators; do not rerun them with learning.

Each new branch contains M0 plus T1–T10: at most **22 map-producing stages and 20 feedback applications in total**. M0 is not a feedback cycle. Complete all ten cycles unless an inherited validity stop or a control failure intervenes. Apparent convergence is not an early stopping rule; lack of convergence does not authorize cycle 11.

| New output | Basis imported from original REF | Operand applied from the new branch | Candidate produced afterward |
|---|---|---|---|
| M0 | REF M0 basis | None | m0 |
| T1 | REF T1 basis | Own m0 | m1 |
| Tk | REF Tk basis | Own m(k−1) | mk |
| T10 | REF T10 basis | Own m9 | m10, saved and unapplied |

The basis is indexed by the stage that produced the map, not by the preceding candidate index. Preserve original and new m10 products as unapplied. Use branch-qualified names everywhere.

## 5. Exact intervention and trajectory lifecycle

### 5.1 What is imported, recomputed, and checked

Import the original REF stage-specific learned basis with its exact detector/sample ordering and compatible application design. Preserve the inherited segmentation, upstream validity, masks, scaling/transforms, regularization, rank-resolution rules, and solver tolerances. Recover and declare all state that defines the retained-signal operator; changing only a basis filename is not proof that the intended intervention was isolated.

Recompute from the current branch residual:

- The original PTC numerical means on the original eligible support.
- Application coefficients and the resulting cleaned residual using the imported basis and native application rule.
- All ordinary validity checks, map construction, source fitting, and candidate admission required by the existing lifecycle.

Do not import the saved REF numerical means, fitted coefficients, reconstructed correlated signal, cleaned residual, branch fits, or feedback candidates as substitutes for those computations. Do not invoke Learn merely to obtain means and then discard its new basis. Do not copy a successful learning status and claim that the new residual was learned or converged. Record that the basis was externally supplied from an authenticated REF stage and report current Apply outcomes separately.

A fixed masked Apply is not generally implemented by a bare `Id − B B^T`. Retain the exact observed-entry solves and mask distinctions. Keep full segment support; do not crop PTC to the source disk.

Preserve rules rather than forcing outcomes. If a new residual produces a genuine support, eligibility, rank-resolution, or other operator-defining transition, record it; do not overwrite flags or hard-code old output support to obtain equality. Follow the validity/isolation handling below.

### 5.2 Construct the immutable shoulder parent once

Let u be the exact previously saved projected shoulder. Verify `y_shoulder − y = u` on the full original injection domain, including valid PTC entries that are not final MAP contributors. Preserve invalid entries and physical gaps. Do not redraw, shift, rescale, renormalize, convolve, or further taper the injection. Do not add it again in later cycles.

The known shape may be used to construct/authenticate the already specified parent and to calculate diagnostics. It must not enter source-fitter initialization, source selection, feedback truth masks, or candidate values.

### 5.3 Produce every stage from its branch parent

For branch b, use its immutable parent y_b. At M0 the feedback operand is absent. At stage k ≥ 1, project only that branch's own previously generated candidate:

\[
v_{b,k}=P m_{b,k-1},\qquad x_{b,k}=y_b-v_{b,k}.
\]

Recompute the original mean functional on x, apply the saved corresponding REF basis with freshly solved application coefficients, restore **exactly the same v that was subtracted**, and run ordinary MAP. Schematically, where the verified factorization applies:

\[
\mu_{b,k}=J_k x_{b,k},\qquad
T_{b,k}=\mathcal M_k\left[Q^R_k(x_{b,k}-\mu_{b,k})+v_{b,k}\right].
\]

Use the actual native operation order if implementation details require a more general representation; do not change the numerical method to fit the notation. Q denotes the retained-signal operator.

Fit the resulting full branch map with the unchanged procedure and generate its next complete replacement candidate at threshold 3 using the original frozen pixel-admission scale. The fitted map background is handled exactly as in the accepted candidate procedure; it is not fed back as astronomical brightness.

Never subtract from the previous cleaned residual, accumulate candidates, force support to grow, reuse the old adaptive shoulder candidate, seed with the known injection, or add the shoulder as a second restoration term. The candidate derived from T10 may be saved but must not be applied.

## 6. Reference replay is the gate, not a cosmetic comparison

The original REF basis sequence should reproduce the original REF trajectory when means, Apply coefficients, fits, and candidates are regenerated from their own outputs.

At M0 and every T1–T10, compare `REFSEQ-REF` with the authenticated original REF:

- Numerical means, cleaned outputs, remove/rejoin accounting, total maps, and original map measurements.
- Candidate brightness values, selected membership, connected-component decisions, validity, and the identity of the candidate actually applied at the following stage.
- Upstream eligibility, PTC/application/MAP masks and causes, per-pixel network contributions, coefficients, support, and normalization.
- Correct stage-specific basis identity and zero observation Learn/ALS calls.

Use existing archived numerical tolerances, exact equality where the deterministic route promises it, and exact discrete decisions where required. Bind them before execution. Do not use a science-performance percentage as a replay tolerance or relax tolerances after seeing discrepancies. A numerical fit perturbation that changes membership is not an acceptable replay merely because the peak barely changed.

Regenerate candidates; comparing copied candidates to themselves is not a control. Do not “repair” the control by injecting saved means or candidates into the trajectory. If replay fails, retain the evidence, stop the dependent shoulder run, and identify the implementation or input incompatibility. Do not expand the observation matrix to search for a passing configuration.

## 7. Primary scientific comparisons

Use the exact original injection-only map h_map, original coordinate convention, and original C+D/frozen-decomposition comparison domain. For each stage, compute:

\[
\Delta^{\rm seq}_k=T^{\rm REFSEQ\text{-}SHOULDER}_k-T^{\rm REFSEQ\text{-}REF}_k,
\]

\[
\Delta^{\rm adaptive}_k=T^{\rm old\ shoulder}_k-T^{\rm old\ REF}_k,
\qquad E_k=\Delta^{\rm seq}_k-\Delta^{\rm adaptive}_k.
\]

Using the same domain and unweighted inner product as the original work, report for both responses:

\[
g_k=\frac{\langle h_{\rm map},\Delta_k\rangle}{\langle h_{\rm map},h_{\rm map}\rangle},\quad
\eta_k=\frac{\|\Delta_k-h_{\rm map}\|}{\|h_{\rm map}\|},\quad
\rho_k=\frac{\|\Delta_k-g_kh_{\rm map}\|}{\|h_{\rm map}\|}.
\]

Check `eta² = (g−1)² + rho²`. Report signed `g_seq − g_adaptive`, changes in eta and rho, and the normalized norm and signed structure of E. Retain all stages and summarize M0/T1/T3/T5/T10, with T10 the prespecified endpoint. Include T9→T10 drift without claiming asymptotic convergence.

Verify that the original full comparison domain and the needed operator/support identities remain valid. Do not silently shrink the headline domain to hide new missing pixels or recompute the ideal map with a changed operator. If the primary matched comparison becomes unavailable, preserve the transition and any clearly labeled secondary common-support diagnostics, with their changed domain; they are not substitutes for the original metric.

Inspect both shoulder-like gain and non-shoulder structure, including the original core, wings, surrounding negative or positive residuals, and map-background behavior. A larger g is not automatically a better result; improvement toward unity with reduced spatial error is the relevant comparison. Overshoot and gain–distortion tradeoffs must remain visible.

## 8. Bounded endpoint accounting: distinguish avoided subspace effect from changed history

Add a compact **M0 and T10** accounting check using the existing native saved-Apply diagnostic. Do not repeat the previous full multi-array decomposition campaign or run a third trajectory.

Under the verified common mean rule, masks, and operator, define the mean-aware fixed-reference action `L_k = Q^R_k (Id − J_k)`. Let d_seq be the difference between the actual projected operands applied in the two new branches; let d_old be the corresponding difference from the two original adaptive branches. These are applied m9 operands at T10, not m10.

Independently calculate the new centered feedback term:

\[
F^{\rm seq}_{C,k}=\mathcal M_k[(\mathrm{Id}-L_k)d^{\rm seq}_k].
\]

Reuse the authenticated old centered-direct map `D_C = M L u` where its identical operator and domain have been verified. Check:

\[
\Delta^{\rm seq}_k=D_{C,k}+F^{\rm seq}_{C,k}.
\]

Then calculate, directly from the operand change rather than as a closure remainder,

\[
H_{{\rm history},k}=\mathcal M_k[(\mathrm{Id}-L_k)(d^{\rm seq}_k-d^{\rm old}_k)].
\]

With the saved old subspace-change map `C_Q,old`, check:

\[
\boxed{E_k=-C_{Q,{\rm old},k}+H_{{\rm history},k}.}
\]

This accounts for both branches' operand differences and therefore does not silently discard a small reference-replay residual. At M0 all feedback operands are absent, so both feedback/history terms must be zero. At T10, the first term removes the old endpoint subspace contribution at the old residual, while the second records the propagated change in the realized feedback operands. Their signed maps can cancel; report projections and spatial structure, not only positive percentages.

In projection terms, the question is whether the new a2000 endpoint equals, exceeds, or falls below the approximately 0.93108 old-history fixed-reference value, and what history term accounts for that difference. No result is presumed.

This is conditional algebra for the verified matched operators, not a unique physical allocation or a second causal experiment. Do not call the history term recovered shoulder flux or infer the effect of a different threshold from it. If the required operator/mean/domain identity fails, report this secondary check unavailable rather than inventing an identity or masking a transition. Preserve otherwise valid primary comparisons.

## 9. Feedback history and estimator response

For every newly formed candidate m0–m10, archive its map, admission ratios, source association, selected membership, fitted constant, and provenance. Distinguish formation at Tk from application at T(k+1).

Compare new shoulder candidates with the original adaptive shoulder candidates at matching indices, and retain the paired new/reference differences. Report selected area; retained/added/removed pixels; changed brightness on retained support; original threshold margins and connectivity for pixels whose membership changes; and first departures in the map and candidate histories. Candidate changes are permitted in either direction: no monotonic-growth assumption.

At T10, use **applied m9** for protection accounting. Where the verified linear action above applies, partition new shoulder minus old shoulder into retained-brightness, added-support, and removed-support pieces; independently do the same for new reference minus old reference. Subtract the reference-branch pieces when assembling the paired history difference, and propagate through that same action. The reference pieces should vanish with exact replay; do not discard a nonzero replay residual. Check that the resulting maps sum to the history term. Keep this bounded to T10 and reuse existing partition utilities. Geometric overlap with the known shoulder may be shown for orientation, but it is not an independently recovered/lost-signal fraction.

Run only the ordinary fits required on the full new branch maps. Preserve the original seven-parameter fitter, initial/search rules, constraints, fitting support, and source association. Do not fit signed difference/component maps with positive-source constraints, change fit bounds to improve an outcome, or give the fitter the injection truth.

Report peak, centroid/shape, fitted constant, the original fixed-template measurement, and the three fixed-aperture measurements. Use physical centroid offsets and gauge-safe ellipse diagnostics. Retain original centers, radii, pixel area, weights, and units: mJy/nominal beam for peak/template-like quantities and mJy/nominal-beam arcsec² for brightness-area aperture sums. No conversion to corrected astronomical flux.

Keep paired raw aperture/template increments separate from the fitted-background subtraction contribution. Reuse the authenticated original C+D ideal map-addition measurement comparators after reference replay establishes their applicability. Do not silently create a new comparator when a saved one is unavailable. Report absolute measurement differences; ratios are unavailable for invalid or numerically negligible denominators. Failed or constrained fits stay visible, and a failed optional measurement does not erase usable map response.

## 10. Numerical validation and validity stops

Reuse the established native Apply, FRUIT, MAP, basis-layout, mean, and accounting tests. Add only focused fixtures needed for this intervention:

1. A deterministic short multi-cycle fixture with deliberately different saved bases at adjacent stages, checking the full M0/m0/T1 ordering, current-branch candidate replacement, and non-accumulation. A zero-injection pair must agree without copying outputs.
2. A masked native-Apply fixture with nonzero means, incomplete entries, distinct fit/application/MAP masks, and existing rank-deficient handling. Check fresh means and coefficients with fixed basis; retain native numerical ownership.
3. A no-op and an analytically specified simple-cleaner check of remove/rejoin and the centered/history identities, including exact-protection and changed-operand cases. Truth-based operands are allowed in these synthetic identities only, never in observation admission.

Authenticate basis identity at every observation stage and establish zero new observation Learn/ALS calls using the existing structural/instrumented safeguards. Do not label hard-coded zero counters as interception proof. Independent review must check that no convenience preparation path implicitly relearns a basis.

Stop before the next feedback application for failed replay or accounting, new failed required segment/application, unsupported projection, invalid fit/candidate needed to continue, non-finite required operand, or inherited feedback-boundary contact. A new support or other operator-defining transition that breaks the intended basis-only matched experiment must be retained and reported, not forced away; stop that affected constrained trajectory rather than changing the intervention.

A failure confined to a terminal optional diagnostic does not erase valid completed maps. Attempt the ordinary terminal m10 construction; preserve a terminal failure explicitly without applying anything or changing the fit to manufacture success. Distinguish completed map stages, candidate availability, numerical gate failures, and scientific findings in the status record.

Low recovery, overshoot, non-monotonic evolution, larger spatial error, or remaining T10 drift are scientific outcomes, not reasons to change the settings or add cycles. The known a2000 background-model limitation is not itself a numerical execution failure and must not block this bounded response experiment. No additional noise ensemble or independent-network variance assumption is authorized to resolve it.

## 11. Implementation, resources, and evidence

Extend the existing saved-basis native Apply entry point and experimental FRUIT lifecycle rather than building a parallel reduction framework. Native PTC retains mean/application ownership; existing FRUIT utilities retain source-measurement, candidate, and remove/rejoin behavior; Python may authenticate, orchestrate, accumulate, and report as already established. Reuse numerical code verified by the completed decomposition.

A narrowly scoped diagnostic basis-provider override is authorized, with explicit stage/segment bindings and no production-default effect. Imported bases must retain their original provenance; create honest new diagnostic execution receipts rather than forging learning lineage or bypassing normal production provenance checks globally. Do not install a generic cross-generation state-reuse policy.

Continue in the verified authorized development worktree under existing repository practice. Inspect actual branch/source/status first and preserve unrelated or pre-existing changes. No reset, discard, main-checkout repair, integration, merge, push, or production activation is authorized.

Use compact arrays, reusable immutable parents, cached authenticated bases, and bounded per-segment processing. Cycle dependencies are sequential, and REF must gate the shoulder branch; do not parallelize through those dependencies. Use useful deterministic parallelism within independent networks/segments or existing native numerical work, avoid nested thread oversubscription, and preserve accumulation ordering needed for replay. Select resources from the actual host rather than inheriting a habitual low-core setting. Do not migrate environments or launch Unity/Spack work for this directive.

Record actual workers/threads, wall and CPU times, loading/hash, mean, Apply, MAP, fit/admission, serialization, and diagnostic costs. Distinguish individual-process high-water memory from aggregate memory; do not claim an aggregate from unrelated peaks. Keep performance work subordinate to the scientific control and bounded to reuse of existing paths.

Write a fresh output bundle, for example:

`successor-point-152389-a2000-ref-basis-sequence-20260924/`

Use a distinct handoff, for example:

`handoff/POINT_152389_A2000_REF_BASIS_SEQUENCE_2026-09-24.md`

Preserve this directive, prospective manifest, authenticated input/basis sequence, source and binary identities, per-stage commands/receipts, new means/application outcomes, total maps and candidates, measurements, replay deltas, support transitions, diagnostic maps/tables, failed/unavailable results, and regeneration instructions. Reference immutable ancestors instead of duplicating the complete original archive.

Provide a compact figure set: response trajectories; matched-scale T10 ideal/adaptive/constrained/difference and distortion maps; candidate-history/mask evidence; and the T10 avoided-subspace/history accounting with signed profiles or aperture diagnostics. Keep orientation, domains, scales, and units explicit. Preserve full stage tables even if an optional figure is unavailable.

Run affected tests and required repository/configuration gates in the available environment. Obtain the existing independent review focused on intervention isolation, complete reference replay, indexing, candidate autonomy, endpoint/history identities, and interpretation. Verify original inputs are unchanged. Record actual local validation separately from unperformed Spack V2/Unity validation. Commit locally under existing practice; do not create unrelated governance work.

## 12. Owner-facing assessment and final stop

The assessment must answer:

1. Did `REFSEQ-REF` reproduce the original reference through T10 while regenerating its own means, coefficients, fits, and candidates? Give the numerical maxima and exact membership/support verdicts.
2. For the shoulder branch, how did g, eta, orthogonal distortion, and original estimator response change relative to the saved adaptive history? Separate scalar gain from spatial fidelity.
3. When did the histories first differ, and how did candidate brightness and support changes evolve? What was still drifting from T9 to T10?
4. At T10, how much of the change is accounted for by the avoided old subspace contribution versus the propagated candidate-history term? Did they reinforce or cancel, and do the signed maps close?
5. Did the constrained intervention improve this shoulder response, worsen it, produce a gain–shape tradeoff, or become unavailable because of a state transition? State the actual evidence without declaring a new default or universal winner.
6. What single next bounded action is justified? Recommend it only; do not execute it here.

Retain the original a2000 background qualification. The paired differences are not independent noise realizations; do not attach A+B errors to them, add deterministic changes in quadrature, quote significance, or create a total uncertainty budget. A g near one is not proof of undistorted recovery, and eta/rho are not percent PSF broadening or direct measures of the final 5% photometric/PSF goals. No absolute astronomical flux correction follows from this pre-PTC-boundary intervention.

**Stop after the two prescribed trajectories, bounded diagnostics, assessment, validation, and review.** No new basis learning; no additional array or observation; no second injection, reference-basis alternative, single-basis-for-all-cycles variant, threshold/seed sweep, or repeated adaptive C+D campaign; no old or new m10 application or cycle 11; no JINC, covariance-weighted fitting, RBF/correlation-aware admission, damping, health repair, new noise/refit ensemble, changed published source, or production/default change.

**Completion means a controlled measurement of how sharing the original cycle-varying REF subspaces changes the a2000 closed-loop shoulder response—not merely another endpoint subtraction, not proof that frozen-basis reduction is preferable, and not authorization to adopt it.**
