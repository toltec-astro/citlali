# Codex directive: explain the POINT152420 / redu09 amplitude discrepancy

**Owner:** Grant Wilson  
**Date:** September 30, 2026  
**Scope:** POINT152420, a1100; completed successor B reduction and the actual historical `redu09` lineage. Q remains parked.  
**Status:** Diagnose, implement necessary correctness repairs, execute a bounded causal investigation, independently review the explanation, and report. Do not return only a proposal.  
**Execution mode:** Prepare and preflight; launch a bounded numerical batch; end the agent turn; resume on the owner's message.

> Explain why the completed successor has a fitted peak of approximately 291.81 mJy/nominal beam while historical `redu09` has approximately 574.59, and why the historical profile is also broader. “Different processing histories” identifies work to do; it is not a causal explanation. Do not assume either reduction is correct, and do not tune the successor merely to reproduce the historical brightness.

## 1. Objective, evidence, and disposition

The completed B/Q comparison may remain closed with Q parked. **Successor photometric/source-profile performance remains unresolved.** Retaining B as a comparator is not scientific acceptance of its amplitude.

Recover and verify these reported measurements from their bound products:

| Product | Peak above fitted constant, mJy/nominal beam | Major × minor FWHM | Fitted constant, mJy/nominal beam |
|---|---:|---:|---:|
| Successor B M0 | 223.28 | 10.81″ × 4.84″ | Recover from saved fit |
| Successor B T5 | 291.81 | 10.98″ × 5.19″ | −0.87 |
| Historical redu09 | 574.59 | 12.57″ × 7.31″ | +32.59 |

The reported T5-to-redu09 contrast is 282.78 mJy/nominal beam, a peak ratio of approximately 1.969. From the rounded widths, the Gaussian area ratio is approximately 1.61 and the Gaussian angular-integral ratio approximately 3.17. For a fitted Gaussian, excluding its constant,

`I_G = [pi / (4 ln 2)] × A × FWHM_major × FWHM_minor`.

This has brightness-times-angular-area units. It is **not absolute flux** until beam/calibration conventions are reconciled. Recompute these quantities after validated remeasurement; do not preserve faulty numbers for continuity.

The reported 2″ averaging check leaves redu09 at 559.89 mJy/beam. Verify and reuse that evidence rather than repeating it unnecessarily. The amplitudes already exclude their fitted constants: a constant offset is not, by itself, an explanation. Background/model coupling remains testable. Existing compact incremental-response tests do not establish absolute recovery of the real source or independently validate calibration.

## 2. Authority and operating boundaries

Carry forward the September 29 bounded-development authority. Own the implementation within this objective: inspect source and provenance, repair project-local tooling, repair correctness defects, extend existing diagnostics, run focused tests, regenerate dependent products, and commit locally without asking permission for each step.

**This directive explicitly authorizes diagnostic variations needed to isolate the discrepancy**, including comparisons of documented historical and successor detector populations, calibration application, cleaning methods/ranks, weighting, map normalization, and feedback behavior. Earlier prohibitions on reopening those settings do not block these explanatory controls. This is not a parameter optimization campaign or permission to change production defaults.

Keep the original completed products and their method identities intact. The successor baseline remains the accepted RTC/CAL and paired x/r validity/support semantics, rank-10 observed-entry ALS with warm initialization and full re-solving, ordinary MAP on a 2″ grid, threshold-3 feedback, the 40″ fit/feedback policy, and exactly five feedback applications. T5 consumes m4; m5 is measured but not applied. Distinguish any repaired baseline from the originally executed baseline. Do not relabel old results as repaired results.

A diagnostic spatial-policy variation is allowed only when recovered historical settings, measured edge structure, or actual feedback truncation make it a concrete hypothesis. Keep fitting/background-region changes separate from feedback-support changes. The auxiliary footprint reaching about 41.3″ is not itself evidence that significant emission was cut at 40″. Do not begin with a radius sweep or more iterations.

Ask only at a substantive boundary: another observation/array, unrelated redesign, production adoption, authoritative-input modification, resource-budget expansion, remote writes/pushes, or an unresolved scientific choice outside these diagnostic controls. Continue independent in-scope work while reporting a specific blocker. Do not use uncertainty about one dependency to halt all diagnosis.

## 3. Stage A — recover provenance and locate the first divergence

### A1. Follow redu09's parent directories to its actual lineage

**Owner clarification: redu09's provenance is contained at upstream levels of its directory tree.** Start with the saved comparison's source-map path, resolve the actual historical map location rather than stopping at a copied comparison product, and inspect its parent directories through the relevant observation/reduction roots. Follow recorded references to configurations, scripts, logs, manifests, calibration inputs, and earlier products. Do not declare provenance missing because it is absent from the redu09 leaf directory. Do not ask the owner to reconstruct information already present in the hierarchy.

Discovery anchors, not instructions to reset the checkout:

- `validation/point152420_map_comparison_2026-09-29/`, including `redu09/` and its assessment/source bindings.
- `validation/rtc_point_transfer_2026-09-28/ASSESSMENT.md`, `RESUME.md`, and `completion-results/`.
- `~/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930/`.
- Reported worktree: `~/.codex/worktrees/ptc-rank-response/citlali-refactor`.

The completed campaign reports numerical source `518e05891`; `c87c20d9a` is the later reporting commit, not the source that executed. Recover current HEAD and intervening work without resetting or overwriting anything. Read applicable repository instructions and current scientific contracts.

Create one compact side-by-side provenance table, with exact evidence paths, covering acquisition/subobservation, timestamps/scans, array/network/detector membership, accepted source crossings, raw/tune/APT dependencies, detector-to-calibration mapping, calibration/correction factors, units/nominal beam, pointing/coordinate realization, filtering/decimation, cleaning, weighting/map kernel/normalization, and feedback operations. Distinguish requested configuration from realized execution and inherited parent configuration from current defaults.

Verify the historical network population rather than assuming it is networks 1–4. Trace the exact historical reduction sequence. `redu09` is a directory label until its meaning is established; `FRUITLOOPS_ITER=4` is not proof of five successor-equivalent applications. Earlier V2 is a contextual benchmark, not redu09's no-feedback parent unless provenance establishes that relationship.

Use local hierarchy evidence first. Targeted read-only recovery of genuinely missing dependencies for this same observation from existing configured Unity access is permitted after identifying exact paths and sizes. No bulk archive retrieval, remote computation, credential changes, or archive modification. A referenced but unavailable item is a specific dependency gap, not permission to substitute a different observation or silently omit a network.

### A2. Audit calibration and map meaning numerically

Trace the same representative native detector values through both conversion chains to final brightness units. Include a small deterministic selection of common detectors spanning relevant networks/calibration ranges, plus any population implicated by the evidence. Inspect the full conversion-factor distribution for systematic offsets; do not rely on one favorable detector.

Check delta-f/f or other native signal conventions, tune/APT identity, detector calibration, applied corrections, nominal-beam conventions, pixel/kernel factors, and any duplicated or missing conversion. Calculate representative conversions independently from recorded factors so injection and expected-response code cannot conceal a shared normalization error.

Identify what each FITS product actually contains: residual, total, sky model, weighted numerator, normalized brightness, or another quantity. A common `BUNIT` label is insufficient. Trace map numerator, denominator, and support rules. More detectors alone should not multiply a correctly normalized brightness estimate; population-dependent calibration, beam, pointing, contamination, or source-crossing rejection can change it and must be measured.

### A3. Repair the fitter and remeasure saved maps

Resolve the reported project-local SciPy loading problem before SciPy-dependent validation. Repair or recreate the project-local environment through the established reproducible route as needed; record relevant versions and import/smoke tests. Do not modify machine-wide packages or silently substitute another scientific estimator.

Repair the artificial periodic-angle boundary while preserving the intended Gaussian-plus-constant model, physical parameter constraints, objective, units, and weights. Use a maintainable parameterization, not an observation-specific escape or restart ladder. Verify equivalent angles and axis-swap/orientation representations, a near-circular case where angle is unidentifiable, and the affected saved maps. Compare fitted images and objective values, not merely success flags or absence of active bounds.

Remeasure B M0–T5 and redu09 with the same validated procedure; include available historical intermediate maps and earlier V2 where informative. Inspect affected injection endpoints only as needed to establish the scope of invalidated prior claims. Report amplitude, constant, centroid in its actual frame, widths, Gaussian angular integral, fit state, objective, and residual morphology. Do not claim global optimality solely because the fit is interior.

Separately test whether corrected intermediate fits change accepted feedback, through background subtraction, centroid, admission, thresholding, or other actual consumers. **A corrected final-map fit does not repair a trajectory that consumed faulty earlier measurements.** Rerun a corrected reference when consequential feedback changes are demonstrated; count it in the numerical envelope below. Keep old and repaired histories distinct.

Add a small set of physically matched, fixed-angular-radius profiles/curves of growth and background checks. Use a consistent centering convention, disclose recentering, and retain centroid offsets separately. Inspect core, shoulders/pedestal, and negative bowls. Include at least one background estimate not inherited from the same Gaussian fit. Apply the same diagnostic conventions to both products rather than choosing each map's most favorable aperture or background. Preserve genuine coverage; never zero-fill missing pixels or silently renormalize away unsupported regions.

Native AltAz and sky-coordinate maps are not automatically pixelwise comparable. Use available telescope/AST/WCS provenance for valid transformations where straightforward; do not invent a single rotation for a time-dependent coadd. Otherwise compare physically meaningful native-frame profiles and invariant scalar measurements, with explicit limits, and continue the investigation. Do not build a registration framework just to obtain a difference image.

### A4. Recover trajectories and account for removed/added signal

Recover redu09's own initial map and successive processing/feedback products from its established lineage, using existing saved evidence before rerunning. Compare stage definitions, not iteration labels. Build a compact trajectory showing where peak, background, width, and angular integral diverge.

Trace source-associated signal through calibrated input, RTC, PTC, mapmaking, applied sky model, residual, and published total. Uncleaned data contain atmosphere: their absolute fitted level is not automatically source truth. Prefer matched source-crossing measurements, source-coherent removed components, or controlled perturbations when additive background prevents a meaningful direct comparison. Measure whether differences in accepted source crossings or the removed cleaning model can account for the observed contrast. Rank 10 versus rank 5 is a hypothesis, not an explanation until its realized effect is demonstrated.

For each feedback endpoint, verify which model was projected/subtracted, which cleaned residual was mapped, what was added back, and in which units, pointing, support, and normalization. Test for omitted or duplicate add-back, cumulative-versus-incremental confusion, and mismatched residual/total products. Use a focused synthetic/no-cleaning accounting test where useful. Do not add normalized maps algebraically when their denominators or support differ; perform the check in the compatible contribution domain or reproduce the exact forward operations.

End Stage A with the earliest demonstrable divergence, quantified leading hypotheses, ruled-out explanations, and the smallest causal comparison set needed. **Proceed into authorized Stage B without an owner checkpoint merely to approve routine choices.** If saved-product evidence and local correctness tests fully resolve the discrepancy, do not run an unnecessary observational campaign.

## 4. Stage B — bounded, evidence-led causal comparisons

Use a small number of matched controls, not a grid over every setting. For each proposed role, state the causal question, changed block, fixed inputs/states, distinguishing expected signature, required measurements, and parent/reference identity before launch. Select the first batch from Stage A findings, then use its results to select any follow-on work.

Choose among these mechanisms only when the recovered evidence supports them:

| First divergence / hypothesis | Discriminating comparison |
|---|---|
| Conversion or normalization | Same native samples and independently checked factors; repair/replay the implicated application or mapper normalization. |
| Detector population, weights, or rejection | Native-population results plus a common-population/source-crossing control; attribute differences to specific calibrated contributions rather than detector count alone. |
| RTC or PTC signal removal | Identical compatible upstream samples through the relevant documented methods; measure source-coherent removed signal and downstream amplitude/profile response. |
| Feedback growth or accounting | Matched inputs with actual subtraction/residual/add-back definitions exposed; use historical settings only as identified diagnostic controls. |
| Numerical fit affecting feedback | Original versus correctly remeasured saved products, followed by a corrected trajectory only when feedback changes require it. |
| Historical excess from contamination | Source measurements across informative scan/detector splits and source-versus-background structure, retaining adequate source support. |

Frozen-state/operator controls can isolate a mechanism; fully relearned runs establish the adaptive method's realized consequence. Label them separately. Do not transplant learned states into a run presented as independently relearned, or force identical support and then conceal the effects of actual rejection. Report native-population and common-support results separately.

If incremental injections are necessary, choose the morphology, angular scale, and amplitude needed to distinguish the remaining hypotheses. Use each treatment's own uninjected reference. Do not repeat the earlier small compact test as proof of absolute recovery. Do not use the same unchecked calibration implementation to define both injected physical amplitude and expected output. Count every new reference/injected trajectory in the budget.

Do not insist on arbitrary historical/new stage hybrids when representations are incompatible. Prefer the earliest compatible checkpoint or a complete, explicitly scoped comparison with remaining confounders stated. Reuse existing checkpoints only with complete compatible state and identities; a saved map is not a resumable timestream state.

### Numerical envelope and launch behavior

For this investigation, allow **up to six planned observation-profile trajectory starts, plus at most two diagnosed replacement starts: eight total maximum**. Historical reproductions, corrected references, diagnostic treatments, and injected roles all count. A full observation reduction stopped early also counts; changing runners or bypassing wrapper bookkeeping does not make it free. Saved-map fits, source/configuration audits, and genuinely small localized tests are ordinary development, not full trajectories. Keep their cost proportionate.

This is the new investigation's single envelope. The predecessor's unused replacement allowance is **not additive**. Do not rerun the completed eight-role B/Q campaign. Use the allowance adaptively and stop early when the mechanism is established. Replacement starts require a diagnosed defect, a focused repair/test, and identified invalidated products; no blind retries. If more work is needed, report the demonstrated mechanism or unresolved discriminator and a specific additional request, not a claim of resolution.

Preflight and freeze each batch's code/configuration/inputs. Use the established persistent launch mechanism, make one bounded startup check, provide status/resume paths, and end the agent turn. No model polling, log-watching agents, or automatic model continuation. On the owner's resume, verify completion, assess compact results, perform authorized repairs, and launch evidence-supported follow-on work within this envelope without requesting permission for each helper or control. Do not edit code bound to an active batch.

## 5. Evidence standard, review, and closeout

The requested result is a **quantitative causal account of the peak and profile discrepancy**, not a catalogue of configuration differences or passed execution checks. Prefer a table connecting each demonstrated mechanism to its measured amplitude/profile effect, control, uncertainty/limitation, and remaining discrepancy. Effects of interacting adaptive stages need not add independently: report attribution order and interactions rather than inventing an additive error budget.

Use appropriate existing noise/estimator scales where applicable. Do not manufacture paired significance by adding independent-map errors for reductions of the same data. Conditional fitted-peak errors do not cover calibration, estimator bias, or historical processing uncertainty. Conversely, unavailable perfect uncertainty does not justify postponing investigation of a factor-of-two effect. No default new Monte Carlo ensemble or general uncertainty framework is required.

Assess whether source-consistent signal is suppressed in the successor, inflated/contaminated in redu09, measured inconsistently, or affected by multiple mechanisms. A stronger/brighter or narrower/cleaner-looking map is not automatically more correct. Consistency across independent data subsets is useful but does not itself prove absolute calibration. Use an independent flux/calibration constraint already present in the lineage where applicable; do not invent one.

Prepare one focused independent milestone review of the substantive repairs and causal conclusions through the existing review process. It should challenge the strongest attribution, independently check key numerical comparisons, and distinguish execution correctness, measurement validity, and scientific resolution. Reproducing file hashes or agreeing that Q should stay parked does not resolve this investigation. No review is required for every helper edit or local commit.

Use the existing validation/reporting structure, with a clearly named amplitude-investigation assessment and durable resume record. Retain compact provenance references, configuration diffs, measurements/profiles, accounting checks, causal-comparison results, code/environment identities, actual runtime/resource accounting, and reproduction commands. Lead the owner report with **what explains the discrepancy, how much, and what remains unexplained**. Follow with the status of any invalidated earlier measurement/feedback claims and the next concrete decision, if one remains.

Respect current approved worker/memory/storage limits. The latest completion report cites a shared 30 GB ceiling; verify current governing limits and actual holdings before launch. Reuse prior analysis and compact products. Remove inactive, reproducible scratch owned by this work when regeneration dependencies and compact evidence are secured; do not require an expensive ceremonial replay for every deletion. Protect historical maps, provenance up the directory tree, raw/tune/APT/calibration inputs, unique diagnostic evidence, unrelated work, and authoritative archives. Do not delete original comparison evidence to make room for the investigation.

Commit locally and update current status/handoff without rewriting frozen campaign history. No push, merge, production activation, new target, or ongoing automatic monitoring.

**Completion criterion:** demonstrate where the discrepancy arises and reproduce its consequential size with suitable controls, or explicitly leave the amplitude issue open with a narrowed, evidence-supported unresolved cause. Neither matching 574.59 by tuning nor accumulating successful runs constitutes completion. Do not close photometric validation while an unexplained near-twofold amplitude difference remains.

---

### Source context for this directive

This work order is based on the completed POINT152420 assessment supplied in the Codex conversation, the September 29 `CODEX_POINT152420_COMPLETION_AND_BOUNDED_AUTONOMY.md` authority, and Grant's September 30 instruction to investigate the discrepancy and recover redu09 provenance from its upstream directory hierarchy. Numerical values above are reported starting evidence, not a new execution or independent remeasurement. Resolve and verify their actual source bindings in the repository.
