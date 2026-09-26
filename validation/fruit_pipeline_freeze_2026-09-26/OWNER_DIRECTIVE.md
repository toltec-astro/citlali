# Codex directive: freeze FRUIT, consolidate the end-to-end pipeline, and hand off RTC/PTC testing

**Owner:** Grant Wilson  
**Date:** September 26, 2026  
**Status:** Execute this repository closeout: inspect, consolidate, test, clean up, review, commit, and stop. Do not return only a plan.  
**Scope:** The existing Citlali successor path and its established bright-POINT MAP+/BASE-POINT feedback endpoint. No new scientific comparison or method optimization.

> We built MAP+ so that we could evaluate RTC and PTC through astronomical maps and measurements. We now have enough downstream capability for that purpose. Freeze FRUIT method development here and turn the working pieces into one clear, runnable, maintainable end-to-end development pipeline. Leave a durable checkpoint and a self-contained handoff for a fresh Codex thread. Do not start the RTC/PTC experiments in this closeout.

## 1. Establish the baseline and keep the task bounded

Read applicable repository instructions, the current status, and the accepted RTC/CAL/PTC, ALIGN/AST, MAP, FRUIT, validity, and companion boundaries relevant to this route. Recover existing decisions; do not undertake a package-wide contract rewrite or another architectural survey.

Use these discovery anchors, verifying their actual identities and relationships before changing anything:

- The completed POINT 152430 a2000 replacement comparison, reported final local commit `ac1bf4eec`, with science revision `ac2bee1febaab82b808f7f751374b797de033c6a` and records under `validation/point_152430_no_upper_coverage_veto_2026-09-26/`.
- `doc/REFACTOR_STATUS.md`, `doc/SCIENTIFIC_GOALS.md`, and the latest upstream and MAP+ handoffs.
- `CODEX_INDEPENDENT_POINT_A2000_FROZEN_METHOD_VALIDATION.md`, `CODEX_POINT_152430_REMOVE_HIGH_COVERAGE_VETO_AND_RERUN.md`, the completed local-data inventory/purge, and the September 9 FRUIT owner directive.
- The existing runner/reporting code, including `tools/timestream_successor/run_independent_point.py`, `report_independent_point.py`, and their actual scientific dependencies. Names are discovery leads, not requirements to preserve the present organization.

Distinguish the tested numerical implementation from subsequent reporting-only commits. Inspect the live branch, worktree, permanent Git common directory, uncommitted changes, and active processes. Do not reset to a quoted hash or assume an old temporary checkout is authoritative.

Work locally on an owned branch descended from the verified completed implementation. Preserve unrelated changes. Ensure the result is available in a durable checkout and reachable history in the permanent Citlali repository, not only under `/private/tmp` or `local_data`. Report its relationship to the repository's integrated mainline without silently moving that mainline. No unrelated branch merges, push, remote deletion, history rewrite, or production activation.

This order authorizes necessary behavior-preserving integration and cleanup within the active successor path. It does not authorize new RTC/PTC algorithms, admission methods, or observing-mode policies. Complete independent work when a specific dependency is blocked; report the exact gap without broadening the task.

## 2. Freeze the downstream experimental reference

Establish one clearly named **bright-POINT downstream development reference** using ordinary MAP and BASE-POINT. This is the default documented route for the next upstream-testing thread, not a claim of a universally qualified production default.

Preserve the completed reference's exact settings and semantics:

- Ordinary MAP, its authorized coefficients and support, the 2-arcsec grid, containing-pixel projection, and 40-arcsec feedback radius.
- The existing seven-parameter Gaussian-plus-constant measurement and expected-source guard: finite accepted fit, positive amplitude, fitted-amplitude S/N strictly greater than 10, both principal sigmas strictly greater than 1.5 arcsec, and centroid within 20 arcsec of the expected origin. Preserve the exact uncertainty convention, bounds, and rejection behavior.
- Source-connected threshold-3 BASE admission, accepted measured-map brightness as feedback, unity gain, and complete replacement of the candidate at each stage. No fitted-Gaussian brightness substitution, persistent union of masks, or automatic gain modification.
- M0 followed by ten feedback applications. T10 uses m9; m10 is saved but unapplied. An iteration limit is not a convergence claim.
- The current RTC/CAL treatment and per-network/per-processing-segment iterative observed-entry rank-10 PTC as the unchanged upstream reference for this closeout. Preserve paired-x/r behavior, flags, validity, centering, solver, timing, calibration, and existing observation-specific exclusions.
- Observation-specific initialization of the existing M0 noise/guard quantities, frozen within a trajectory under the current recipe. Preserve the separate endpoint conditional uncertainty calculation and all existing measurement definitions.
- High coverage above the background training maximum remains informational, with the existing numerical continuation. Preserve genuine data, variance, covariance, projection, low-coverage, and geometric-support checks. Do not reinstate the veto indirectly.

Freeze **algorithms and configuration recipes**, not learned numerical answers. Each new input or future upstream configuration must generate its own appropriate initial map, noise quantities, learned PTC states, and feedback candidates under those recipes. PTC must still relearn on every current sky-subtracted residual. Do not freeze an old PTC basis, mask, or numerical noise field to make the upstream testing path appear simpler.

Mark CORR, DIAG, JINC comparisons, CUT3, shoulder-specific refinements, seed investigations, and extra-cycle studies **parked**. Keep their useful compact evidence and necessary regression fixtures; remove them from the active quickstart and default execution path. Do not run them to close out this task.

## 3. Make the actual data flow explicit and executable

The repository must expose the real flow below, with the existing scientific ownership and required companions preserved:

```text
Raw detector acquisition + exact tune/APT/calibration dependencies
Telescope/auxiliary streams + shared identity/timing context
                         |
            Paired ingress and ALIGN
                         |
                        RTC
                         |
                        CAL
                         |
              immutable PTC-input parent y
                         |
                  PTC -> ordinary MAP -> M0
                                         |
                              POINT measurement/admission -> m0
                                         |
                         repeated feedback applications:
                         y - project(m_k)
                                  |
                      fresh PTC Learn/Consider/Apply
                                  |
                     restore exactly the subtracted operand
                                  |
                       ordinary MAP -> total T_(k+1)
                                  |
                        measurement/admission -> m_(k+1)
```

**AST and validity/support are coordinated side paths, not disposable annotations.** Document where AST supplies native pointing, RTC-matched timing/grid information, detector coordinates, and sky projection; show the actual dependencies rather than forcing these into a false one-way sequence. Preserve the current x/r, sample identity, gap, calibration, response, and support relationships through every boundary.

FRUIT starts from the immutable post-RTC/CAL parent. It does not rerun RTC on an already filtered stream, repeatedly clean the previous residual, or accumulate previous feedback models. Restoring a projected model and mapping it must preserve the accepted native subtract/restore accounting; do not substitute direct map-space addition without its required identity.

Identify required NOI/FLT/response companions in their actual roles. Preserve implemented companions and truthful unavailability; do not fabricate replacements or implement unrelated downstream packages to fill out this diagram.

Publish the reconstructed total map separately from the feedback candidate. A failed source check must yield the existing explicit failed-fit/empty accepted-output outcome, clear stale feedback, and stop dependent iterations while retaining the diagnostic map. A scientifically rejected source and a processing failure must remain distinguishable.

## 4. Consolidate the implementation, not just its documentation

Provide **one recommended repository entry point** and a versioned reference configuration for running the supported route from original inputs. Reuse existing CLI/configuration conventions where practical. A thin Python orchestrator calling existing native modules is acceptable; a monolithic rewrite is not required.

The documented interface must support:

1. Preflight that resolves inputs, associations, environment, selected configuration, output location, and resource limits without starting scientific processing.
2. Ordinary first-pass processing with feedback disabled, using the same implementation as M0 in the feedback route.
3. The frozen BASE-POINT feedback route and its compact map/measurement/uncertainty products.
4. Saved-product inspection/reporting that never silently runs scientific processing when a product is missing.

Observation, array/network selection, input roots, output roots, and executable/environment locations must be explicit configuration or command inputs, not campaign constants buried in scientific functions. Keep currently supported arrays usable; clearly label observing modes and capabilities that have not been implemented or qualified. Do not turn this into an all-mode implementation project.

Separate reusable processing and measurement code from observation-specific experiments, plots, and orchestration. Scientific modules own their algorithms; the runner sequences them. Runtime code must not import numerical algorithms from a dated validation/report directory or depend on manual notebook steps, shell history, private scratch scripts, or one developer's temporary working directory.

Move or extract only what is necessary. Preserve tested numerical kernels and accepted operation order. Avoid a new plugin system, universal workflow framework, scheduler, cache platform, generic provenance system, or language/toolchain migration. Remove duplicate live implementations after callers have been redirected. Retain a thin historical wrapper only where it serves an actual retained report or reproduction command.

Ordinary runs must not require another observation's result directory, the old fifteen-trajectory experiment driver, archived RTC/PTC timestreams, or a previous successful reduction. Historical results may be reference data for a comparator, never hidden scientific inputs.

## 5. Leave usable boundaries for the next RTC/PTC thread

Make the existing boundaries callable and identifiable without duplicating their algorithms:

- Original inputs through RTC/CAL to the immutable PTC parent, with its required AST/validity/response context.
- PTC applied to that parent or a current sky-subtracted residual.
- Ordinary MAP, total reconstruction, POINT measurement, BASE admission, and conditional uncertainty/reporting.
- The existing post-RTC/CAL, pre-PTC injection/projection controls and compact/shoulder test definitions.

Expose already implemented RTC/PTC configuration through its proper owner rather than hardcoding it in the downstream runner. Do not add a new scientific option merely to make the interface look general. Do not alter settings or run a mode-count comparison here.

Document which quantities are relearned per observation/upstream configuration, which remain fixed within a trajectory, and which are recomputed each feedback application. Distinguish memory residency, learning scope, and processing-block scope; consolidation must not silently convert the existing per-segment PTC into observation-wide learning or change physical gap handling.

The current pre-PTC injection boundary is available for future PTC tests. An earlier native-domain injection needed to measure RTC transfer is **future work**. State its required units/response/AST considerations, but do not implement or test that new injection route in this task.

## 6. Clean the active repository without destroying useful work

Make a short, actionable inventory of the relevant active code, duplicate helpers, campaign-specific drivers, current documentation, task-owned scratch, and retained scientific records. This is not another whole-disk inventory or mass payload-hashing exercise.

After checking callers and ownership, remove obsolete duplicate code, superseded active instructions, unused task-owned generated files, and inactive temporary numerical products. Update imports, tests, entry points, and affected reproduction references. Use normal tracked deletion; Git history already records prior tracked versions. Do not manufacture an archive for every deleted script or failed scratch attempt.

Preserve governing scientific sources, the useful compact 152389/152430 results and regression operands, raw/tune/APT/calibration inputs, team products, unrelated work, and any source/environment information needed to rebuild. Do not recreate the already erased 152430 stopped attempt. Do not delete evidence merely because a method is parked.

Keep current documentation unambiguous: one quickstart, one downstream reference definition, and one handoff. Link historical records as historical rather than rewriting them as current instructions. Preserve a clearly labeled legacy/comparison lane where it actually exists; it must not silently supply the successor path.

No broad `git clean`, forced worktree removal, branch pruning, symlink traversal, filesystem snapshot cleanup, or Unity modification. Existing unrelated worktrees/branches are not deletion targets. Remove only task-created worktrees when their work is safely committed and they are inactive; otherwise report them.

`~/work_toltec/local_data` remains disposable. Keep required source, configuration, compact scientific evidence, and reproduction/handoff documentation in durable repository history. Use temporary RTC/PTC arrays only as necessary for execution; delete them at closeout. Reference existing compact results instead of copying the 235-MB bundle into every new handoff directory. Record before/after sizes and cleanup scope without claiming tracked deletion removes historical Git objects.

## 7. Prove the consolidated route with a bounded cold-start regression

This is an integration regression, **not a new FRUIT science campaign**.

### Focused checks

Run the affected unit, interface, configuration, native, and regression tests. Cover the shared forward/feedback route, exact identities and units, x/r/validity preservation, projection and subtract/restore accounting, fresh PTC learning, candidate replacement, source-failure behavior, high-coverage semantics, relocatable inputs/outputs, and read-only report behavior. Reuse retained fixtures to exercise the existing injection controls without launching new real-data injection trajectories.

### One required real-data demonstration

Use the completed reference acquisition **POINT 152430/0/2, a2000, associated tune 152430/0/1, networks 11/12**, verified against its actual retained input manifest. Do not infer these associations from proximity. No additional observation or array is needed for this closeout.

From a clean checkout of the candidate in durable storage, using the established build environment and an empty output root, run the consolidated entry point from the original raw/tune/calibration/telescope inputs through M0 and BASE-POINT REF T10. Build the candidate by the documented procedure; record the exact source and executable identities. Do not let an old binary or imports from an earlier worktree bypass changed code. No environment migration is required; distinguish this local evidence from any separate Unity/Spack validation.

The processing process must have no usable access to old derived campaign products. Isolate those paths with the existing practical mechanism and verify the denial, while explicitly allowing the original inputs, declared environment, and candidate code. The comparator may read the retained reference afterward. Do not delete existing products to simulate cold-start conditions or invent a new security framework for this test.

Initialize the run's M0 noise quantities normally and perform its existing REF endpoint uncertainty analysis. The successful run supplies both first-pass and feedback evidence; do not duplicate the raw reduction just to demonstrate two command names.

Compare M0 and every completed T1-T10 stage against the retained BASE-POINT REF: map planes, coverage/support, candidates, source acceptance, fitted parameters, template/aperture measurements, and applicable uncertainty records. Require exact discrete decisions and agreement within the existing justified numerical tolerances for floating results. Establish those tolerances before inspecting differences. Ignore only explicitly irrelevant serialization/path/timestamp differences. A source-relative or sub-noise agreement is not permission to weaken numerical regression checks.

**Execution allowance:** at most two fresh raw-to-M0 builds, two BASE-POINT REF trajectories, twenty feedback application attempts total, and two endpoint uncertainty analyses, including failed attempts. One successful cold-start regression is required. A second attempt is available only after diagnosing and correcting a specific integration/build/path defect; do not use it to tune science or silently replace the reference. No other real-data trajectories, injections, null ensembles, parameter sweeps, or extra cycles. Focused small tests are not charged as real-data trajectories.

No input transfer is expected. Only an exact missing dependency of this recorded acquisition may be recovered through already authorized read-only Unity access. Do not download other arrays, observations, or saved RTC/PTC checkpoints. Unresolved inputs block this demonstration, not all independent cleanup work.

Use bounded scratch and useful parallelism without changing scientific learning scope or nested-thread oversubscription. Reuse the current workspace budget or establish a small run-specific ceiling from actual allocations. Report runtime, memory, disk scope, and all attempts. After saving compact results, remove new inactive numerical intermediates and verify that reporting works from the retained compact products alone.

## 8. Produce a durable freeze and a fresh-thread handoff

Use existing documentation locations where suitable; avoid parallel status frameworks. Deliver:

**A concise pipeline quickstart.** Include the actual data-flow diagram, component ownership, environment/build command, tested preflight/forward/feedback/report commands, input manifest and retrieval instructions, supported scope, output meanings, and temporary-storage policy. Commands must work from the durable checkout with explicit user-configurable roots, not depend on `/private/tmp` paths or undisclosed setup.

**A frozen reference configuration and compact validation record.** Bind the scientific method/settings, tested source and binary, exact input identities, regression outcomes/tolerances, resource/cleanup counts, and known limitations. Reference existing results rather than copying them unnecessarily. Preserve the distinction between the tested numerical commit and a later documentation-only closeout commit.

**One handoff for the new RTC/PTC thread**, for example `handoff/RTC_PTC_MAP_FRUIT_BASELINE_2026-09-26.md`. It must identify the exact starting branch/checkpoint, shortest required reading list, executable commands, module/configuration boundaries, available compact response controls, and real-data readiness. State what upstream work remains and which downstream choices are frozen. Include a short ready-to-paste new-thread kickoff grounded in the finished repository, not provisional commands.

Record limitations explicitly: the present feedback reference is for bright POINT, conditional uncertainties are not a full adaptive error law, the shoulder-response limitation remains, existing injections bypass RTC/CAL, and the local regression does not establish production or all-mode qualification. Preserve any relevant existing upstream exclusions and failures without transplanting them to other observations.

Update the current status and goals records to say that FRUIT method testing is parked and the next development lane is RTC/PTC evaluation through this endpoint. The proposed PTC mode-count comparison and future upstream injection are **deferred work**, not tasks to execute now.

Commit reviewed changes into the permanent repository's reachable history. Create one new, non-overwriting local checkpoint tag at the completed freeze using repository naming conventions. Do not retarget existing tags, advance unrelated branches, push, or merge into another development/production line. The new thread should start from this verified checkpoint, not guess which old worktree contains the latest code.

## 9. Acceptance and stop

Obtain a bounded independent final review of the actual changed scope: scientific behavior preservation, real data flow, input/output dependencies, cold-start evidence, cleanup safety, and handoff usability. Use the repository's existing preflight/review requirements; do not create a chain of ceremonial reviews or claim tests that were not run.

Completion requires a runnable consolidated route—not just a diagram, wrapper around hidden historical state, or a collection of individually tested scripts. The closeout must answer:

- What exact command takes the recorded original inputs to M0 and the frozen feedback endpoint?
- Where can the next thread change RTC or PTC without editing downstream scientific code?
- What is demonstrably identical to the frozen reference, and what is still unavailable or unvalidated?
- What code/products were removed, what remains durable, and what exact checkpoint starts the new thread?

Correct ordinary consolidation defects within the authorized scope. If a scientific discrepancy or missing dependency prevents acceptance, do not relax the method or report completion: deliver the implemented portions and a precise blocker. Do not launch another science campaign to settle it.

**Stop after the reviewed repository closeout and handoff. Do not run the 5/10/15-mode comparison, change RTC filters/donor rules, implement pre-RTC injection, refine FRUIT, add observing modes, or start the fresh thread's experiments.**
