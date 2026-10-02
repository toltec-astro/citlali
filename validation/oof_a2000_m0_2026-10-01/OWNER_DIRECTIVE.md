# Codex directive: a2000 OOF maps through the operational analysis windows

**Owner:** Grant Wilson  
**Date:** October 1, 2026  
**Status:** Execution work order: implement the necessary bounded preparation, freeze the smallest batch, launch through the established owner-terminal/manual-resume procedure, assess, and stop.  
**Scope:** OOF 152385/0/1, 152386/0/1, and 152387/0/1; a2000 networks 11/12; separate focus states; cold M0 only; no feedback or optical inversion.

> Start with the band used by the operational OOF workflow. Reuse compatible historical a2000 maps, produce the three successor initial maps, and assess the regions the existing analysis actually consumes. Establish whether a reported limitation is encountered in that workflow before turning it into a prerequisite.

## 1. Owner decisions and supersession

The owner states that the current OOF workflow uses **a2000 only**, and that **LMTOOF and a second existing OOF analysis code trim their input maps**. Recover their actual preprocessing rather than inventing a crop or assuming that full-map valid-pixel counts describe the fitted inputs.

This order replaces the proposed a1100/networks 1–4 first experiment with a2000/networks 11/12. It authorizes the bounded OOF reductions below; the earlier preparation-only restriction no longer applies to this batch. It does not reopen POINT, activate production defaults, or authorize an OOF feedback method.

Unequal outer-map coverage is **not an automatic failure or launch blocker**. The earlier fixed-count LMTOOF finding remains recorded, but its relevance must be checked after the operational trimming/masking and at the actual executable boundary. An established analysis window is not an improvised crop to conceal a defect. Conversely, equal cropped dimensions alone do not prove equal valid sample counts.

Keep mapmaking, consumer preprocessing, and optical inversion as separate stages. A defect in an unexecuted optical fitting path does not block otherwise valid M0 maps. Do not make pupil/bandpass optimization, formal solver uncertainty, or a general solver repair prerequisites for this mapmaking milestone.

## 2. Recover and narrow the existing preparation

Read current repository instructions, active worktree changes/processes, accepted OOF preparation, and exact input records. Reuse completed engineering and evidence; do not repeat the entire POINT investigation or an open-ended search for missing historical directives.

Discovery anchors—not reset instructions—are:

```text
~/.codex/worktrees/ptc-rank-response/citlali-refactor
validation/oof_preparation_2026-10-01/READINESS.md
validation/oof_preparation_2026-10-01/EVIDENCE.json
handoff/OOF_PREPARATION_2026-10-01.md
~/work_toltec/local_data/citlali-validation/v2/oof/
~/GitHub/LMTOOF/
```

The supplied preparation closure was `95b2fab06`; verify the current source and later changes. Preserve the closed POINT evidence and retire the unstarted a1100 proposal without deleting unrelated work.

Verify the three acquisitions and networks 11/12 against their actual raw data. The recorded dependencies are Tune 152384/0/1 and canonical matched APT-v2 products with beammap 148670 parent; authenticate rather than substitute these from memory. Preserve requested physical focus settings, all telescope corrections, timestamps, and gaps. The −2/0/+2 mm offsets are relative to the middle acquisition, **not an assertion of best focus**. No cross-focus coaddition or pooled cleaning is authorized.

Recount a2000 detector identities, calibration availability, native cells, and resource needs. Do not reuse the earlier a1100 detector counts or storage reservation as a2000 measurements. Use the existing local sequence; no bulk download or remote computation is authorized.

## 3. Establish the actual consumer windows with saved maps

Trace the operational entry point through preprocessing to the final arrays consumed by each analysis code. Inspect the actual Python wrapper/configuration as well as any C++ reader; do not test only a lower-level reader that bypasses normal trimming.

Locate the second code from local workflow evidence. `OOFAnalysisPipeline` and `MultiBandDataLoader` are older discussion search hints only—not verified current names, locations, or crop settings. If it cannot be located in a bounded search, record that specific gap and continue with the available LMTOOF path; do not invent its behavior or block M0.

Run only saved-map preprocessing/reader diagnostics, with fitting, model inference, telescope writes, and other side effects disabled. For each focus state and consumer, record compactly:

- Input identity, frame, angular pixel scale, units, and original support.
- Actual crop/centering rule, angular boundaries, output dimensions, mask/weight/finite-value rules, valid sample count, and unsupported pixels inside the window.
- Any interpolation, background subtraction, normalization, or coordinate/pointing change performed by the operational preprocessing, including its parameters and operands.

Freeze the established rules before successor outcomes. Apply the same consumer configuration and scientifically equivalent angular selection to reference and successor products; identical pixel indices are not equivalent when pixel scales differ. Preserve any established data-dependent centering rule, but record the resulting shifts/windows and do not optimize them separately to improve agreement. Keep unnormalized, unrecentered parent maps and diagnostics so preprocessing cannot hide amplitude or pointing differences.

**Reassess the LMTOOF warning at the actual post-preprocessing boundary.** Check residual allocation/indexing against the actual valid sample counts and ordering for the selected executable. If preprocessing satisfies that implementation's requirements, classify the unequal-full-map-count warning as not encountered in this invocation. That is not a claim that the general fixed-count assumption has been repaired. If unequal counts still reach an incompatible fitter, retain the evidence and block that fitting path only. Never zero-fill invalid observations or invent a new crop merely to satisfy its allocation.

A narrowly justified correction to Citlali's preparation/adapter guard is authorized when the operational evidence demonstrates the existing guard tests the wrong stage. Preserve rejection of genuinely unsafe inputs. No LMTOOF or other sibling-repository modification is authorized by this order.

## 4. Reuse the historical a2000 reference whenever compatible

Recover the actual saved a2000 products for all three acquisitions, including their upstream reduction provenance. First determine whether networks 11/12 constitute the complete recorded a2000 population and whether those historical maps already use it.

Check acquisition, raw/telescope/tune/APT identities, available detector population, effective calibration/extinction treatment, and the executed reduction configuration. Verify whether processing or normalization couples a2000 to other bands. **The fact that the historical job also processed a1100/a1400 is not, by itself, a reason to reject its a2000 maps.**

Differences in accepted samples caused by the declared algorithms are comparison outcomes to report, not a reason to force identical final masks. Shared APT bytes likewise do not prove identical effective sample calibration.

Reuse compatible maps without rerunning their raw data. Preserve the actual reference recipe and disclose its source protection, cleaning, projection, weighting, and feedback settings. Do not assume all settings from the earlier a1100 summary. The resulting comparison is operational R versus S, not an isolated PCA-versus-ALS experiment or a claim that R is ground truth.

Only reproduce a reference acquisition when a specific missing product or consequential input mismatch prevents reuse, and an existing supported a2000-only reference route can supply it within the allowance below. Document the reason before launch. Do not build a parallel legacy pipeline, bypass canonical APT admission, or process all eleven networks simply to work around subset selection. If matched R remains unavailable, continue independently useful S characterization, label historical context honestly, and identify exactly which comparative claims remain unavailable.

## 5. Produce the three successor M0 maps

Use the shared native OOF route and scientific owners. Implement necessary a2000 routing/configuration fixes and focused fixtures without asking permission for each engineering edit.

The frozen successor recipe is:

- **a2000 networks 11/12**, each acquisition reduced independently from its own original inputs; cold initial map and **zero FRUIT applications**.
- Existing validated **a2000** RTC/CAL/filter/segmentation behavior with paired x/r and native timing/gap semantics. Do not relabel an a1100 filter recipe, tune a new notch, or reopen the parked fixed-11-Hz comparison.
- Observed-entry ALS **rank 3**, cap 500 and tolerance `1e-5`, retaining the proposed OOF rank control rather than POINT rank 10. Report the recovered reference rank and any difference; no rank search or silent substitution.
- Existing ordinary MAP on the **2-arcsecond grid**, current coefficient/support semantics, and supported native coordinate route. No POINT source-admission gate, Gaussian/ring/core requirement, inherited 40-arcsecond source boundary, or new source-protection mask.

Retain full native M0 support in the parent output. Consumer trimming is downstream analysis; it must not silently truncate RTC/filter support or redefine mapmaking validity.

Resolve the selected telescope-coordinate semantics as far as necessary for the actual map route, preserving original and recomputed stream provenance. A demonstrated incompatibility blocks the dependent operation; an unrecovered historical rationale alone is not evidence that the coordinates are wrong or a reason to stop unrelated diagnostics.

Do not relabel FK5 maps as AltAz or treat a single finished-image rotation as exact sample-level reprojection. Reuse an already validated physical-coordinate adapter or native AST/MAP attachment where available. Otherwise retain valid native M0 products, limit consumer/cross-frame comparisons explicitly, and do not redesign the projection system to satisfy an optical solver that is not being run here.

## 6. Minimal execution budget and manual resume

**Expected new work is three S reductions and zero R reductions.** The ceilings are three S starts plus up to three specifically justified missing-reference R starts. Each planned role has one start; unused R slots are not retries, replacements, or optimization allowance. No additional arrays, sequences, injections, noise ensembles, feedback, or optical inversions are included.

Prepare and freeze the actual minimal matrix, exact source/binary/config/input identities, and resource plan. Reuse existing accounting/report infrastructure rather than creating another governance framework. Run relevant changed-interface tests and required regression gates on the final executable candidate; reuse still-applicable evidence. Obtain one focused independent review of the substantive implementation and plan, not one review per helper edit.

Run serial focus states/roles, at most two network workers and one numerical thread per worker, with one active role. Respect the shared **30 GB derived-storage ceiling** and separate **2 GB free-disk reserve**, unless a later explicit owner amendment exists. These are disk constraints, not RAM predictions. Recalculate active and retained needs for a2000 and the actual matrix; do not reserve six runs merely because six are allowed.

Authorized cleanup includes owned, reproducible, inactive intermediate products after verifying that compact evidence and regeneration inputs are sufficient and no remaining assessment needs those intermediates. Retain maps, masks, calibration/support summaries, relevant learned-state diagnostics, configurations, source/input identities, and concise logs. Protect raw/tunes/APTs, original historical maps, authoritative archives, unrelated work, and active jobs. Do not retain multi-GB per-role scratch by default or raise the ceiling automatically.

Use the established **owner-run foreground batch**: provide one exact tested terminal command once ready. Preparation scripts are not launch commands. The owner starts the batch; it stops on the first failed role and never retries automatically. Do not silently switch to a background launcher, poll, or create monitoring subagents. If the owner has already started a bound batch, do not modify its source or inputs.

On the owner's **“done”** message, verify completion from receipts and inspect compact results first. If still running, report that once and stop monitoring. Diagnose failures within this scope, preserving compact evidence; additional numerical attempts require a separately revised allowance. No push, merge, production activation, remote computation, optical fitting/model inference, or telescope-control change is authorized; the preprocessing and reader-only checks above remain authorized.

## 7. Assess the operationally relevant maps and stop

Repeat the consumer preprocessing checks on the new a2000 maps. Judge each focus state primarily over the recovered operational analysis window, with original-domain context and common-support results reported separately.

Report distributed structure and relative brightness across focus, broad/asymmetric wings, positive and negative artifacts, bowls entering the analysis window, internal support holes, accepted/calibrated exposure, and actual runtime/peak memory/retained disk. Preserve amplitude, background, morphology, and any operational normalization as distinct observables. Do not fit a scalar to force R/S agreement or equate a narrower/higher peak with better OOF data.

Use consistent physical axes and display scales. Perform numerical map differences only with justified unit/frame/grid alignment; disclose interpolation and do not convert a qualified visualization into an exact paired measurement. Measure background scatter/correlation where genuinely source-free support exists. If the window is signal-filled, do not train a noise model on the OOF pattern or claim a sensitivity gain. No optical solution, transfer-function validation, or absolute flux-recovery claim follows from this M0-only comparison.

Outer support differences matter when they intrude into the actual analysis, compromise needed background/support, or reveal consequential artifacts—not merely because whole-map pixel counts differ. Preserve full maps, but do not turn unused peripheral coverage into a blanket veto.

Deliver one compact assessment, machine-readable measurements, comparison figures, and reproducible launch/resume information using existing structures; a suitable new location is `validation/oof_a2000_m0_2026-10-01/`. Update readiness/status to distinguish superseded a1100 findings, verified a2000 findings, unexecuted fitting concerns, and genuine remaining limitations.

Lead the closeout with:

> Were compatible historical a2000 maps reused? What does the operational trimming actually pass to each consumer? Do the three successor M0 maps preserve useful OOF structure with acceptable contamination and support, and at what cost?

Conclude **improved, comparable, degraded, or not established** for the supported measurements, with concrete reasons. Finish this bounded map assessment before proposing feedback or an optical-inversion milestone. POINT stays closed.

---

**Basis:** Owner clarification and agreement in this conversation on October 1, 2026; `CODEX_POINT_MAP_QUALITY_AND_OOF_HANDOFF_2026-10-01.md`; and the supplied Codex OOF preparation/readiness report. Local paths, historical identities, and earlier audit findings above are discovery records to verify, not claims of a new filesystem or scientific validation by this directive.
