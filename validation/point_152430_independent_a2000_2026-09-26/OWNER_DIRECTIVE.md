# Codex directive: independent bright POINT — a2000 frozen-method validation

**Owner:** Grant Wilson  
**Date:** September 26, 2026  
**Status:** Resolve inputs, implement the narrow extension, validate, execute, assess, review, and stop.  
**Scope:** One additional bright POINT observation, a2000 only; ordinary MAP; BASE-POINT, CORR, and DIAG; five branches per policy, ten feedback applications per branch.

> Determine whether the shoulder-recovery benefit and compact-response tradeoff found in POINT 152389 recur on a different observation, and whether their sizes matter for the measurements. Test the existing methods; do not optimize them on the new observation. Keep raw inputs recoverable from Unity and local processing products disposable.

## 1. Authority and prerequisites

This order releases the previously parked **second-observation comparison only**. CUT3, further 152389 pixel/shoulder diagnosis, JINC, additional arrays, and method tuning remain parked. It authorizes the new observation's ordinary upstream processing and the bounded experiment below, not a production/default change.

Read applicable repository instructions and recover the actual current implementation and completed counterparts of:

- `CODEX_LOCAL_DATA_INVENTORY_AND_RTC_PTC_PURGE.md`
- `CODEX_POINT_152389_SCIENTIFIC_IMPORTANCE_AND_ARCHIVE_COMPACTION.md`
- `CODEX_POINT_152389_GUARDED_FEEDBACK_COMPARISON.md`
- `CODEX_POINT_152389_EXPECTED_SOURCE_GUARD.md`
- `CODEX_POINT_152389_CORRELATION_AWARE_ADMISSION.md`
- `CODEX_POINT_152389_RESPONSE_AND_ROBUSTNESS_CD.md`

Use the completed inventory, purge receipt, importance assessment, and retained method bindings—not directive text as proof of completion. Finish any still-required inventory/purge prerequisite under its existing safety rules before new science; do not repeat completed housekeeping or rebuild historical caches. The reported importance closeout commit `84eb4d919` is a discovery anchor, not an assumed HEAD. Preserve unrelated work.

## 2. Select one observation and resolve Unity inputs

Choose **one distinct POINT obsnum, not 152389 or a split/reprocessing of it**, with an independently established bright-source expectation and usable a2000 acquisition. Prefer a locally complete, reduction-ready package; among comparably ready choices, favor a different observing session or tune realization. Independence of the acquisition does not establish statistical independence of all calibration errors.

Select from inventory, observing metadata, logs, or existing team summaries—not by trying the three policies and choosing a favorable result. Record the choice and rationale before a new reduction. Do not assume that a nearby obsnum has the right mode, source, tune, or calibration. If the selected observation subsequently fails the source check, report that result; do not silently replace it.

Produce a compact dependency/transfer manifest:

| Input | Required action |
|---|---|
| Detector raw acquisition | Verify all acquisition parts for every network required by the a2000 route, including its actual channel/time coverage. Retrieve only missing required files. |
| Telescope and auxiliary streams | Verify the complete pointing/time streams and other metadata consumed by the existing route. Retrieve missing inputs. |
| Associated tune inputs/products | Resolve exact associations from metadata, including network-specific associations. Reuse sufficient existing processed products; obtain raw tune data only when the accepted route actually needs them. |
| APT, detector matching, calibration | Bind the correct identities, validity, and provenance. Retrieve missing accepted products, not unrelated calibration campaigns. |
| Team maps/reduction summaries | Use existing compact products when helpful for source identity or gross sanity checks. They are not mandatory inputs, substitutes for raw processing, or flux ground truth. |

Distinguish **complete raw observation**, **a2000-complete input set**, and **reduction-ready**. Missing files differ from networks that were absent during acquisition. Do not fetch other arrays merely to describe this pass as a complete all-array acquisition.

**Download authorization:** Through already authorized access, inspect Unity read-only and download the missing manifest-listed inputs for this one observation and its genuine dependencies. Record exact remote/local paths, sizes, identity checks, and transfer commands. Download nothing when the local package is already sufficient. Do not mirror a night, use broad recursive transfers, substitute an adjacent tune/APT, or download Citlali RTC/PTC checkpoints. If an accepted APT/tune product suffices, do not also fetch its entire raw beammap/tune history. A need to develop or recalibrate a missing input is a dependency blocker, not a new calibration project.

If access or an indispensable association is unresolved, finish the supported preflight and report the exact gap; do not fabricate a location or proceed with substitute calibration. No Unity modification or deletion is authorized.

## 3. Workspace and execution budget

Use the workspace root and any hard budget already adopted following the inventory. Otherwise place this campaign's new files in one dedicated directory under the existing `~/work_toltec/local_data/citlali-validation/` root; reference existing inputs in place rather than copying them. Do not reorganize the whole historical tree in this task.

Before transfer/execution, record unique missing-input bytes, retained-result allowance, temporary-parent/scratch allowance, expected concurrency, and an explicit peak disk-use ceiling derived from actual file sizes and the runner's allocations. Enforce that ceiling and available free space; stop before exceeding it. The earlier approximately 25 GB derived-evidence target is provisional, not an approved total-workspace quota or permission to delete raw data. Do not invent an approved global cap.

Regenerate the selected observation's RTC/CAL and AST inputs with the accepted scientific recipe. Keep one shared immutable post-RTC/CAL, pre-PTC parent for active use, preferably in memory. Any necessary detector-time spool is temporary, bounded working storage—not a retained checkpoint. Avoid full per-policy/per-cycle copies; remove this run's temporary RTC/PTC products at closeout after checking compact outputs. Do not resurrect pruned 152389 timestreams merely to reuse the old harness.

Use the validated environment and useful available parallelism without an arbitrary four-worker cap or nested thread oversubscription. Measure the first REF's memory and runtime before scheduling the rest. No environment migration, scheduler, or general caching framework.

## 4. Freeze the method; initialize observation-specific quantities once

Preserve the reviewed ordinary-MAP geometry, 2-arcsec grid, AST-centered 40-arcsec feedback radius, containing-pixel projection, uniform-coefficient/eligibility rules, unity feedback gain, and complete candidate replacement. Preserve the accepted per-network/processing-segment observed-entry rank-10 PTC, centering, solver, and fresh residual-specific learning/application. Preserve paired-r behavior; no r injection or new r treatment. Do not carry 152389-specific failed-segment exceptions onto this observation.

All policies use the same existing seven-parameter Gaussian-plus-constant fit and expected-source guard: one bounded attempt; finite convergence; positive amplitude; fitted-amplitude S/N strictly above 10; both principal sigmas strictly above 1.5 arcsec; centroid within 20 arcsec of the recorded expected source origin. Keep existing fit weights, bounds, uncertainty propagation, numerical limits, and rejection semantics.

| Policy | Unchanged selection rule |
|---|---|
| BASE-POINT | Guarded source-connected threshold-3 baseline; no supplement. |
| CORR | BASE-POINT plus the existing full-covariance neighborhood supplement. |
| DIAG | BASE-POINT plus the existing diagonal-covariance neighborhood supplement. |

Reuse **full-precision a2000 supplemental score thresholds** from the completed calibration, with no recalibration or sweep. Missing authoritative values block the dependent policy; rounded values are not substitutes.

**Do not transplant 152389's numerical noise covariance, admission-noise amplitude, source mask, coordinates, or pixel IDs.** Build the new observation's unperturbed M0 using the unchanged initial PTC/MAP recipe. Initialize its noise scale, full covariance C0, excluded core K0, support, and measurement geometry once using the existing source-masked full-observation procedure. This authorizes one initial covariance-learning operation for the new observation, not a new covariance family. Freeze these quantities before comparing policies or injecting signals; share them across all branches.

The full C0 supplies the common source-guard uncertainty for all three policies. DIAG substitutes diagonal covariance only in supplemental neighborhood evidence. Preserve exclusions, signed-neighbor evidence, support checks, strict comparisons, and source connectivity. Candidates contain the accepted **measured map brightness**, not a fitted Gaussian, neighborhood mean, or injected truth.

Applying fixed score thresholds to a newly initialized observation tests portability; it does **not** establish the original calibration's false-admission rate on this observation. Label that limitation. Do not launch a new null/Monte Carlo calibration or update covariance inside feedback.

Before execution, bind code/configuration/environment identities, inputs, full thresholds, construction recipes, comparison tolerances, execution limits, and stop rules in a compact prospective record. Freeze realized M0-dependent operands when available.

## 5. Exact experiment and injection construction

Run BASE-POINT REF first. It is part of the matrix, not an extra trajectory. Construct the new observation's common test shapes from its accepted BASE-POINT REF T10 fit and original M0 major FWHM, following the existing C+D recipe. Freeze the actual rasters before examining CORR/DIAG endpoints or running response/seed branches. Do not use a favored policy's fit to redesign the tests.

Let A, centroid, widths, and orientation come from that BASE-POINT T10 fit, excluding the constant. G is its unit-peak Gaussian. H has twice both widths and is displaced by 0.5 times the M0 major FWHM along the positive first physical tangent axis. Use the inherited radial taper: unity through 30 arcsec, `[1 + cos(pi*(r-30)/6)]/2` from 30 to 36 arcsec, and zero beyond 36 arcsec. The operands are `+0.10*A*G*w`, `-0.10*A*G*w`, and `+0.05*A*H*w`, with no post-taper renormalization. Record physical axes, support, actual peaks/integrals, and taper loss. Unsupported required shape pixels make that test unavailable; do not clip or reshape it to pass.

For **each** of BASE-POINT, CORR, and DIAG:

| Branch | Data | Initialization |
|---|---|---|
| REF | Unmodified parent y | Its own normal policy candidate. |
| CORE-PLUS | y plus the projected +10% compact operand | Its own candidate from its perturbed M0. |
| CORE-MINUS | y plus the projected -10% compact operand | Its own candidate from its perturbed M0. |
| SHOULDER | y plus the projected +5% broader/offset operand | Its own candidate from its perturbed M0. |
| SEED-SHOULDER | Unmodified y | Its own REF m0 plus the same shoulder raster, once only. |

**Ceiling: 15 logical trajectories and 150 feedback applications.** Each trajectory has M0 and T1–T10. There are four distinct initial physical inputs; share mathematically identical M0 calculations across policies, and share REF M0 with SEED-SHOULDER. T10 uses m9; retain m10 as an unapplied compact candidate. No cycle 11, extra seeds, alternative amplitudes, or second new observation.

Inject coherently at the regenerated post-RTC/CAL, pre-PTC boundary using native AST projection, including valid source-bearing entries used by PTC but not final MAP. Preserve invalidity. At every cycle restart from that branch's immutable y or y+P(I), subtract its current candidate, freshly learn/apply PTC, restore exactly the subtracted operand, and accumulate MAP. Never reinject each cycle, accumulate old models, or reuse another branch's learned basis on a changed residual.

The seed branch changes only the first candidate after the normal source guard passes. Later candidates have no seed exemption or forced support. These tests do not measure upstream optical/RTC/CAL response or absolute astronomical calibration.

## 6. Measurements and noise-relative importance

Reuse the existing measurements and freeze their new-observation geometry across policies. Retain compact histories for M0–T10; summarize M0, T1, T3, T5, and T10. Report peak, centroid, gauge-safe widths/shape, constant, fixed-template and aperture measurements, source acceptance, candidate/support changes, and T9-to-T10 drift.

Compare each injected branch with its **own policy REF**. Construct injection-only ordinary-MAP controls with that REF's corresponding support/coefficients, bypassing PTC. Report inherited gain/distortion diagnostics, signed residuals, compact sign asymmetry, and shoulder/background absorption. At T10 perform the existing ideal-map-addition measurement controls: at most nine logical controls, three signals per policy. Do not fit signed difference maps with a positive-source Gaussian.

Compare each seed-only branch with its own REF and place it beside the data-injected SHOULDER result. A larger seed projection is initialization sensitivity, not automatically a whole-source flux error or proof of false emission.

On each completed policy REF T10, run the existing source-masked background/covariance companion once: **at most three endpoint analyses**. Use supported existing sensitivity operators to propagate matching conditional measurement uncertainties, including correlations for templates/apertures. These diagnostic endpoint covariances never enter feedback or retroactively alter source acceptance. No new noise/refit ensemble. When source-location transfer or a measurement uncertainty is unsupported, label the scale unresolved rather than inventing precision.

Lead the assessment with signed effects in physical units, their fraction of the corresponding full-source measurement, and their size relative to a matching uncertainty. Distinguish local map noise, fitted-peak error, template error, and correlated-aperture error. A percentage of an injected increment is not that percentage of the source. Old 152389 uncertainties are context only, not the new observation's error bars.

Paired branches share observed noise: an effect/noise ratio is a practical scale comparison, not a calibrated significance test of their difference. Do not add independent variances or import a universal sub-mJy cutoff. Park inconsequential effects; retain material bias, coherent residuals, source loss, or instability. The approximately 5% final calibration/PSF goals are context, not automatic acceptance criteria for this partial-boundary experiment.

## 7. Validation, failures, and stop rules

Use retained compact fixtures and existing tests for guard/selection semantics, native injection identity, channel/validity preservation, subtract/restore accounting, fresh learning, candidate replacement, noise units, and policy-matched response controls. Do not rerun the historical 152389 campaign merely to restate validation. Keep established numerical tolerances even when differences are below observational noise. Obtain bounded independent preflight and final review.

Expected-source rejection produces the defined failed-fit message and empty accepted POINT/feedback outputs; preserve the diagnostic map, clear stale candidates, and stop that branch before another application. Stop affected work for missing dependencies, invalid required covariance/projection, failed accounting, non-finite operands, new processing failures, or the inherited feedback-boundary predicate. Do not relax fits, change thresholds, or swap observations to obtain success.

A failed BASE-POINT reference blocks shapes/tests that depend on it. Other failures block only genuinely dependent comparisons; report common completed stages and every unavailable matrix cell. Low recovery, no extra admission, seed sensitivity, qualified-but-usable covariance, and terminal drift are reportable outcomes—not authorization to tune or extend the run.

## 8. Compact closeout and decision

Deliver one owner-facing assessment with the selected obsnum, dependency/transfer manifest, full 15-cell status matrix, matched-scale figures, leading noise-relative measurement comparisons, and comparison with retained 152389 evidence. Explain separately whether any benefit comes from supplemental aggregation generally or from full covariance specifically: CORR must be compared with DIAG, not only BASE-POINT.

Retain maps, candidates/masks, small measurement tables, injection rasters, covariance descriptions, configurations, input identities/retrieval commands, reproduction commands, tests, and reviews. Keep indispensable source and compact reproduction/scientific records in the permanent repository outside disposable `local_data`, not solely in a temporary worktree. Do not retain bulk RTC/PTC timestreams or per-cycle restart packages. Delete only this campaign's enumerated inactive temporary products; protect raw/tune/calibration inputs and unrelated work. Report actual download bytes, peak/final storage, runtime, memory, and thread settings.

Conclude whether the fixed supplements show a scientifically useful benefit, an immaterial difference, a consequential tradeoff, or insufficient evidence on this new observation. Recommend at most one next action without executing it. Commit reviewed changes locally using established practice and report the exact record locations and commit. No merge, push, default promotion, further observation, additional array, threshold/rank/covariance tuning, JINC, or extended uncertainty campaign.

**Stop after the bounded comparison, compact report, review, and temporary-product cleanup.**
