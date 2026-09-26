# Codex directive: resume timestream testing with a bounded PTC rank comparison

**Owner:** Grant Wilson  
**Date:** September 26, 2026  
**Status:** New-thread execution order. Recover the frozen checkpoint, implement only the necessary experiment wiring, validate, run the comparison, assess, review, commit locally, and stop. Do not return only a proposal.  
**Scope:** POINT 152430/0/2, a2000, networks 11/12; PTC ranks 5, 10, and 15; the frozen ordinary MAP + BASE-POINT endpoint. RTC/CAL and FRUIT methods remain unchanged.

> MAP+ now provides the astronomical measurements needed to evaluate timestream cleaning. Use it as our testing endpoint, not as the next optimization project. Determine how PTC cleaning strength trades residual contamination against source preservation and conditional measurement uncertainty. The shoulder is one response diagnostic, not the sole objective. This order releases the previously deferred 5/10/15-mode comparison only; it does not release RTC changes or further FRUIT development.

## 1. Start from the completed pipeline, not an old experiment tree

Verify the following local checkpoint and its actual Git relationships before editing:

```text
Tag:       timestream-successor-bright-point-base-v1-20260926
Commit:    5435bdbdb4e6708e884ae4cb2b0fd93e74102ce5
Branch:    codex/fruit-pipeline-freeze-20260926
Checkout:  /Users/gwilson/.codex/worktrees/fruit-pipeline-freeze/citlali-refactor
Git store: /Users/gwilson/GitHub/citlali/.git
```

The completed closeout reports an exact original-input POINT 152430/a2000 M0–T10 regression. Its tested numerical source is `b47c13e1f5b745b94880218e347f77b4142686ff`; later closeout commits contain documentation and evidence. Recover those records rather than repeating the closeout. Verify the checkpoint; do not reset an unrelated working tree or retarget its tag.

Read the applicable repository instructions and the shortest relevant recovery set:

- `handoff/RTC_PTC_MAP_FRUIT_BASELINE_2026-09-26.md`
- `doc/BRIGHT_POINT_PIPELINE.md`, the current status/goals, and the applicable PTC/AST/VAL boundaries
- `data/development/bright_point_base_v1/reference.json`
- `validation/fruit_pipeline_freeze_2026-09-26/REPORT.md` and its comparator, input, and environment bindings
- The compact 152430 response controls under `validation/point_152430_no_upper_coverage_veto_2026-09-26/`, including their exact injection rasters and measurement definitions

Work on a new owned branch descended from the verified tag, using the established worktree practice. Preserve the frozen profile, tag, unrelated changes, and mainline. No merge, push, history rewrite, production promotion, or repository-wide cleanup is authorized. Earlier orders that parked the rank comparison are superseded only for this experiment.

## 2. Scientific question and controlled change

Test **rank 5, rank 10, and rank 15** in the existing iterative, observed-entry, flag-aware PTC method. Rank 10 is the reference. The values 5 and 15 are explicitly authorized experimental settings, not proposed new defaults.

Change rank through the existing native PTC owner/configuration interface. Verify the effective rank in native receipts for M0 and every feedback application. Each rank must learn its own solution; truncating the saved rank-10 basis is not the selected experiment. Preserve the existing initialization recipe, centering, solver tolerances, iteration limits, missing-entry treatment, paired-x/r semantics, and per-network/per-processing-segment learning. Do not introduce cross-rank warm starts or choose favorable solver seeds. An unsupported rank must be reported, not silently clipped or substituted.

The effective configurations must differ scientifically only in `ptc.rank`; profile names, paths, and bookkeeping may differ. Check the resolved configuration diff, not just the requested overrides. Leave RTC filters, notch learning, donor handling, flags, decimation, CAL, detector matching, ALIGN/AST, segment boundaries, and MAP coefficients unchanged. Full-observation data residency does not authorize observation-wide PTC learning. Preserve physical gaps and actual sample eligibility.

Keep the downstream method fixed: ordinary MAP; the reference 2-arcsec grid and containing-pixel projection; 40-arcsec feedback radius; existing seven-parameter Gaussian-plus-constant measurement/guard; source-connected threshold-3 BASE admission; measured-map brightness feedback; unity gain; complete replacement; M0 and ten applications. T10 consumes m9; m10 is retained but not applied. Do not reinstate the high-coverage veto or weaken genuine validity, uncertainty, projection, or support checks.

This tests the complete rank-dependent PTC + fixed-policy feedback response. It is not a fixed-subspace transfer calculation. PTC must relearn on every current sky-subtracted residual, and the unchanged selector is allowed to produce different candidates when rank changes the maps.

## 3. Inputs and reusable experiment wiring

Use the already recorded **152430/0/2** acquisition, its exact **152430/0/1** tune associations, networks **11/12**, and the bound APT/calibration/telescope inputs. Verify against the completed manifest. No new observation, other array, or transfer is expected. Recover only an exact missing dependency through already authorized read-only Unity access; do not substitute an adjacent tune, mirror data, or download old RTC/PTC checkpoints.

Use the consolidated `tools/timestream_successor/point_pipeline.py` path and its reusable processing boundaries. Preserve its preflight/forward/feedback/report separation. A thin experimental configuration/branch driver may sequence these existing functions. Do not copy a dated campaign's implementation of RTC, PTC, MAP, feedback, or source measurement into a second live pipeline.

The allowed code extension is narrow: expose the already implemented rank setting for explicitly named experimental profiles, connect the existing pre-PTC injection controls to the shared route, and add the comparison report/tests. Keep the frozen reference profile unchanged. Do not bypass all configuration guards merely because its default rank is 10. New profile identity must distinguish an upstream experiment from an unchanged reference run.

Regenerate one shared immutable post-RTC/CAL parent `y` from originals, with its required AST/VAL/context. Reuse it across ranks and branches where mathematically valid. When the existing upstream call also produces the rank-10 REF M0, use that as the matrix cell's M0 rather than computing it again. Do not import a saved historical timestream as the parent. Support in-memory reuse or a bounded temporary spool; do not create twelve copies of the same observation.

Runtime processing may read the declared original inputs and explicit experiment operands/configuration. Archived maps and measurements are comparison data, never substitutes for newly produced science. The retained injection rasters are permitted, declared test inputs—not hidden restart state.

## 4. Exact matrix and signal definitions

Run the following four branches for each of the three ranks:

| Branch | Immutable input at the pre-PTC boundary | Purpose |
|---|---|---|
| REF | `y` | Noise, source measurement, coverage, and normal feedback behavior |
| CORE-PLUS | `y + P(I_plus)` | Response to the existing positive compact increment |
| CORE-MINUS | `y + P(I_minus)` | Response to the existing negative compact increment |
| SHOULDER | `y + P(I_shoulder)` | Response to the existing broader, displaced increment |

This is **12 planned trajectories and 120 feedback applications**, not another 15-cell admission-method comparison. No seed-only branch is included.

Reuse the exact three retained 152430 signal rasters and their physical-coordinate/taper conventions. Their defining recipe is +10% and −10% of the reference fitted amplitude for the compact Gaussian, and a +5%-amplitude Gaussian with twice the widths and the established offset for the shoulder, with the existing 30–36 arcsec taper. The stored full-precision operands and metadata, not rounded prose, define the injections.

If a raster needs reconstruction from its retained definition, regenerate it deterministically with the established construction and verify against its retained identity before science. Do not refit or resize the injection using a rank-5 or rank-15 result, move it to more favorable support, or renormalize after tapering. A genuinely unavailable required operand blocks that branch.

Inject with the existing native AST projection at the **post-RTC/CAL, pre-PTC boundary**, including the valid source-bearing entries used by PTC even when they do not contribute directly to MAP. Preserve the accepted x-only injection and unchanged r input; do not invent new r response. These tests do not measure RTC/CAL or optical transfer.

Each branch computes its own initial PTC/M0 and candidate. Every subsequent application begins from that branch's original `y` or `y + P(I)`, subtracts its current model, freshly learns/applies the selected PTC rank, restores exactly the subtracted native operand, and maps the total. Never inject repeatedly, clean a previous residual, reuse another branch's learned basis, accumulate old models, or substitute direct map addition for native restoration.

### Rank-specific noise and common measurement geometry

For each rank, initialize the established M0 admission-noise and source-guard quantities from that rank's **unperturbed REF M0**, once. Share that rank's frozen initial noise/covariance/core-exclusion realization across its four paired branches, following the established injection-experiment convention. Do not use injected brightness to train a branch-specific admission-noise model. Do not transplant rank-10 numerical noise fields onto the other ranks or relearn admission covariance during feedback.

Use the same retained physical measurement aperture/template and response-diagnostic geometry across ranks and signal branches. Preserve the existing fit and background-estimation recipes; fitted parameters remain free to respond. Do not move an aperture or redesign a template to favor a cleaner. Rank-specific noise-model learning follows its unchanged recipe, with any resulting domain differences reported.

On each completed REF T10, run the existing conditional endpoint uncertainty analysis once. That rank's endpoint companion supplies the practical measurement scales for its injected-branch comparisons. It does not enter feedback, calibrate paired-difference significance, or supply uncertainty for another rank. Where useful M0 scales can be obtained from already initialized companions, report them without an additional covariance campaign.

## 5. Execution sequence and bounded validation

Before real-data execution, record the exact source/binary/environment, configuration differences, input and injection identities, comparator tolerances, measurement geometry, resource ceiling, and attempt limits in the existing validation structure. Add focused tests for rank propagation into both M0 and feedback, independent learning, unchanged x/r/validity, injection identity and sign, shared processing, subtract/restore, replacement, and source-failure behavior. Use existing small fixtures; do not create a Monte Carlo campaign. Obtain one bounded independent preflight review of the changed scope.

**First run rank-10 REF through T10.** This is one of the 12 matrix cells, not an additional trial. Compare it against the retained frozen reference at M0 and every cycle: map/candidate planes, discrete support and decisions, measurements, initial noise quantities, and endpoint uncertainties. Use the existing comparator and predeclared tolerances. Do not copy historical values into new output or weaken a numerical regression because its discrepancy is sub-noise.

If rank-10 REF does not reproduce, diagnose the integration/configuration issue before releasing other cells. The mismatch is not evidence about cleaning rank. Do not repeat the whole freeze exercise, historical fifteen-trajectory campaign, or old storage purge.

After this gate passes, execute the remaining 11 matrix cells with useful bounded parallelism. Run rank-5 and rank-15 REF cells before their dependent injected branches. Reuse the rank-10 REF result; do not run it again. Compare the newly generated rank-10 injection outcomes with the existing compact outcomes as an additional regression of the injection wiring, allowing only declared non-scientific path/metadata differences and established tolerances. Establish these injection-wiring comparisons before relying on rank-5/15 injected outcomes; a shared injection defect blocks affected work, not permission to reinterpret it as a rank effect.

| Operation | Planned maximum | Absolute maximum including repair replacements |
|---|---:|---:|
| Raw-to-post-RTC/CAL parent generation | 1 | 2 |
| Real-data trajectory starts / branch M0 computations | 12 | 14 |
| Feedback application attempts | 120 | 140 |
| Unperturbed-M0 noise/guard initialization packages | 3 | 5 |
| REF endpoint uncertainty analyses | 3 | 5 |
| T10 ideal-addition measurement controls | 9 | 11 |

The extra allowance permits **at most two replacement trajectory starts in total**, only after a documented implementation, build, or input-routing defect is corrected. It is not permission to retry scientific rejection, poor recovery, rank-dependent solver failure, or an unfavorable result. Use a fresh branch-output attempt; retain honest counts and invalidate superseded affected results. Failed attempts consume the limits. Parent or control repairs are likewise for diagnosed implementation defects, not tuning. Unit tests and preflight checks that never process real observations do not consume scientific attempts.

A source rejection preserves the diagnostic map, emits the established failed-fit/empty accepted output, and stops that branch. A real processing/solver failure remains a failure. An invalid REF blocks its dependent comparisons, not unrelated valid ranks. Preserve actual exclusions; do not discard problematic networks/segments to make a rank pass. Large attenuation, no noise benefit, or coherent residuals are reportable outcomes, not reasons to tune settings or add cycles.

## 6. Measurements: evaluate useful astronomical information

Lead with the unmodified REF comparisons, then the controlled response tests. Summarize M0, T1, T3, T5, and T10 while retaining small all-cycle records. Do not reduce the assessment to the shoulder or to the lowest pixel RMS.

### A. Reference-map and timestream behavior

For each rank report peak, position, shape, fixed-template and aperture measurements; the corresponding supported conditional uncertainties; spatially structured residuals; source admission; retained detector/sample counts; effective coverage; and T9–T10 changes. REF measurement differences versus rank 10 are method sensitivity, not absolute flux truth.

Include a small **network-by-network PTC diagnostic** for networks 11 and 12 at M0 and the last application: requested/realized rank, contributing detector/sample counts, solver termination and residual metrics, cleaning strength, and existing before/after residual-power or correlation summaries where available. Lead with network/population behavior, not a few selected detectors. Evaluate both the remaining contamination and source transfer; lower timestream variance alone is not success.

Use existing diagnostic implementations and reuse data while resident. Do not add a new spectral-analysis method, SPOD/common-mode project, residual-line investigation, or per-detector gallery. Missing optional diagnostics are a stated limitation, not a reason to postpone the map comparison or retain bulk arrays.

Keep full native-support results visible. Verify equality of coverage/support where expected. When a valid rank-dependent difference occurs, distinguish it from source response; add a common-support comparison only where meaningful and never hide the amount or location of lost coverage.

### B. Signal response with each rank's own reference

For each injected branch, compare with its own rank REF and an injection-only ordinary-MAP control that bypasses PTC using that REF's corresponding mapping support/coefficients. At T10 run the established ideal-map-addition measurement control: three signals per rank, nine total planned controls. Do not fit a signed difference map with the positive-source Gaussian.

Report actual-minus-ideal peak, template, aperture, position/shape where meaningful, and the existing spatial gain/distortion summaries. Show physical units, fraction of that rank's full REF source measurement, and size relative to its matching conditional uncertainty. Preserve the distinction between a percentage of the artificial increment and a percentage of the original source.

Separate raw aperture response from the fitted-background contribution. Retain compact plus/minus asymmetry and broader residual morphology as diagnostics, without opening separate follow-on studies. One source-disk gain cannot correct all spatial distortion.

### C. Noise relative to recovered compact response

For measurement Q and injection j, report the finite-response diagnostic

```text
Delta_Q_actual = Q(injected result) - Q(own REF)
Delta_Q_ideal  = Q(own REF + projected/map-sampled injection) - Q(own REF)
g_Q,j         = Delta_Q_actual / Delta_Q_ideal
sigma_equiv   = sigma_Q(own REF endpoint) / abs(g_Q,j)
```

Use the same estimator and fixed geometry in numerator and denominator. Report plus and minus compact responses separately. This scalar calculation is authorized from the already produced measurements; it requires no new fit or uncertainty ensemble. Report it only for supported, finite, nondegenerate responses with a meaningful scalar interpretation. Sign reversal or severe distortion is a warning, not something to disguise with an absolute value. For a distorted shoulder, retain the spatial response instead of pretending one corrected noise number is sufficient.

`sigma_equiv` is an input-equivalent **conditional measurement-uncertainty diagnostic** near this bright-source test, not a new calibration applied to maps, blank-field sensitivity, mapping speed, or a full statistical risk measure. Its purpose is to prevent a cleaner from appearing better merely because it suppresses signal as well as noise.

Use matched effect/uncertainty ratios as practical scales, not p-values. These branches share observed noise; do not add independent variances or claim a paired significance without the required joint uncertainty. The approximately 5% final SCIENCE goals provide context, not a universal POINT or first-pass pass/fail threshold. Park isolated inconsequential effects, while keeping coherent response bias and data loss visible.

## 7. Resources and compact retention

Use the documented working local build/environment; no migration is required. Verify the new binary against the experiment source rather than using a stale freeze binary. State that local AppleClang/Homebrew results are not Unity/Spack qualification. Preserve unrelated cached dependencies; do not repair the whole toolchain in this task.

Place new working outputs in one dedicated directory under `~/work_toltec/local_data/citlali-validation/`, for example `successor-point-152430-a2000-ptc-rank-20260926/`. Establish and enforce its peak-disk ceiling from actual shared-parent sizes, concurrency, and output allocations before execution. Reuse raw/compact inputs in place. Use useful cores with single-thread numerical libraries where needed to avoid nested oversubscription; report actual concurrency, elapsed time, and the scope of sampled RSS/storage maxima. Do not change learning scope to save memory or claim sampled storage is a continuous high-water measurement.

Retain compact maps/masks, injection definitions, measurement and PTC diagnostic summaries, figures, configurations, exact identities, review, and reproduction commands in durable repository history. Aim for a few hundred MB of new evidence, not another multigigabyte restart archive; quantify any exception. Reference existing 26.2-MB freeze and 235-MB comparison evidence rather than copying them into the new record.

Delete this campaign's enumerated inactive RTC/PTC/native detector-time intermediates after compact preservation. Protect original/tune/APT/calibration/team inputs and unrelated work. Do not manufacture a permanent archive for failed scratch attempts. Retained reporting must work without original run arrays and must never silently rerun science. Keep the established single pipeline quickstart accurate; add only the short experimental invocation and result/handoff links needed here.

## 8. Deliver, decide, and stop

Deliver one owner-facing assessment and a machine-readable 3-rank × 4-branch status table. Explain:

1. How much rank changes residual contamination and the uncertainty of the actual source measurements.
2. How much compact and broader source response is lost or recovered, before and after feedback.
3. Whether an apparent improvement reflects signal suppression, data loss, a meaningful response/noise tradeoff, or negligible differences.
4. Whether both networks support the conclusion, and which limitations remain.

Use matched-scale figures for REF histories, source response residuals, and the compact-response/uncertainty comparison. Include runtime/storage and all attempt counts, source/build/configuration identities, input preservation, cleanup, and independently reviewed outcomes. Review the implemented comparison and scientific interpretation; do not substitute a code review for real-data evidence or rerun the whole campaign during review.

Conclude whether this dataset supports keeping rank 10 as the reference, identifies a promising alternative for independent confirmation, exposes a consequential problem, or shows no practically useful distinction. Do not declare an observationally general optimum or silently promote a rank. Recommend **at most one** next bounded action, without executing it; that action may be independent-observation confirmation, a specific PTC diagnosis, or returning to RTC transfer tests, according to the results.

Commit the reviewed experiment and compact record locally on the new branch; update status and the handoff. Leave the original freeze tag/profile untouched. Do not retest 152389, add other arrays, implement a pre-RTC injection, vary PTC learning scope or method, change RTC, resume CORR/DIAG/JINC/FRUIT work, or start another comparison under this order.

**Stop after this PTC rank-response/noise comparison. The purpose is to use the completed downstream pipeline to make progress on timestream science, not to reopen its construction.**
