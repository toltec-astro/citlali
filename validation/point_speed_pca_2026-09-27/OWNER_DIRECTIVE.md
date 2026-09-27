# Codex directive: POINT latency versus feedback count, then a bounded PCA comparison

**Owner:** Grant Wilson  
**Date:** September 27, 2026  
**Status:** Execution order. Establish the unchanged timing baseline first; then implement and test the two specified experimental PTC approaches. Review, commit locally, and stop. Do not return only a proposal.  
**Initial scope:** POINT 152430/0/2, a2000, networks 11/12, with the existing ordinary-MAP + BASE-POINT endpoint. No additional observation or array processing in this task.

> POINT must be fast. Before expanding to other observations, measure the actual time from original inputs to usable POINT results as a function of feedback count. Then determine whether we can obtain comparable or better astronomical information more quickly using a different PTC/PCA approach. Do not confuse a fast solver, low residual variance, or parallel campaign throughput with a fast, scientifically useful end-to-end POINT reduction.

## 1. Recover the completed implementation and preserve the control

Start from the verified completed PTC rank comparison, not the older freeze alone. Discovery anchors supplied by its closeout are:

- Branch `codex/ptc-rank-response-20260926`, reported final local commit `11b0c313c`.
- Scientific execution source `e3cbf1eeadda256e4d9f400a834f9f7d462c1ae2`; later commits include reporting, failure-status repair, and closeout.
- `validation/ptc_rank_response_2026-09-26/ASSESSMENT.md`, `REPRODUCTION.md`, compact results, and final review.
- `handoff/PTC_RANK_RESPONSE_NOISE_2026-09-26.md`, `doc/BRIGHT_POINT_PIPELINE.md`, current goals/status, and applicable native PTC/AST/VAL contracts.
- The unchanged tag `timestream-successor-bright-point-base-v1-20260926`, commit `5435bdbdb4e6708e884ae4cb2b0fd93e74102ce5`.

Verify actual Git identities, ancestry, permanent common directory, checkout, build, and unrelated work. Use a new owned development branch descended from the completed comparison; do not reset to a quoted hash. Reuse the existing `tools/timestream_successor/point_pipeline.py` route and native scientific owners. Do not create a parallel scientific pipeline or move numerical algorithms into the campaign/reporting driver.

Rank 10 with the existing iterative, observed-entry, flag-aware PTC remains the reference. Rank 15 remains an experimental comparator. This directive supersedes earlier deferrals only for timing instrumentation, explicit shorter feedback budgets, within-trajectory warm initialization, and the covariance-based comparator below. It does not promote a setting or release RTC/FRUIT method development.

## 2. Fixed scientific conditions and inputs

Use the exact recorded 152430 acquisition, its 152430/0/1 tune associations, and bound APT/calibration/telescope dependencies. Verify local holdings against the existing manifest. No download is expected. Recover an exact missing original dependency only through already authorized read-only Unity access; do not substitute a nearby tune or download archived RTC/PTC streams.

Keep RTC/CAL, timing/ALIGN/AST, detector identities, x/r semantics, validity, sample weights, detector normalization, physical gaps, segment boundaries, and MAP coefficients fixed. Preserve the actual per-network/per-processing-segment learning scope. Observation-wide residency is not permission to change learning scope.

Keep ordinary MAP, the 2-arcsec grid, containing-pixel projection, 40-arcsec feedback radius, existing Gaussian-plus-constant source measurement/guard, threshold-3 source-connected BASE admission, measured-map brightness, unity gain, and complete model replacement unchanged. Keep the existing high-coverage continuation; do not reintroduce its veto. Keep genuine support, uncertainty, and numerical checks.

Distinguish two unrelated iteration counts in code, configuration, and reporting:

- **N:** outer FRUIT feedback applications after M0. N=0 means M0 only. T_N consumes m_(N-1); a saved m_N is not applied.
- **J:** internal iterations required by the PTC low-rank solver for a particular network/segment/residual.

Changing N is explicitly authorized for this benchmark. Changing solver stopping tolerances, maximum J, rank automatically, or feedback/admission rules is not. No runs beyond N=10.

This measures all processing stages for the validated a2000 two-network route. Label that scope prominently. Do not call it all-three-array or telescope-host operational latency, multiply it by an array count, or extrapolate the Mac timing to Unity. Record any remaining requirement for a later full-instrument benchmark.

## 3. Instrument the whole critical path before optimizing anything

Instrument the existing implementation with low-overhead monotonic timestamps, including child processes. Measure the actual outer invocation from process launch, before input reading and pipeline initialization. Preserve an unmodified-science control and prove the instrumentation/configuration extension does not change its outputs.

Publish these separate milestones:

1. **Raw-to-usable POINT latency:** time until the map, source acceptance/failure status, and existing peak/centroid/width measurements for M0 or T_N are actually available to the caller. Include every noise/guard calculation required to obtain those results, normal synchronous writes, and any preceding work on the critical path. If results are not available before process exit, say so; an internal timestamp is only a potential future availability milestone.
2. **Raw-to-completed numerical analysis:** the same invocation including its ordinary endpoint conditional-uncertainty calculation and final numerical outputs. Do not omit an expensive default stage and call the remaining number end-to-end.
3. **Command exit and validation/reporting costs:** separately identify finalization, cleanup, plotting, archival, and campaign comparison. Exclude synthetic-control campaigns, builds, and independent review from single-observation latency, while reporting their costs separately when measured.

Measure, rather than assume, the stage breakdown: input/process setup; ingress/ALIGN; RTC; CAL; parent construction; PTC preparation/centering/initialization; PTC solve; PTC application; projection and subtract/restore; MAP; source fitting; initial admission-noise/guard learning; each feedback update; endpoint conditional uncertainty; synchronous I/O/serialization; finalization. Use inclusive and exclusive labels. Parallel child durations cannot be added to obtain wall time; report critical-path wall time, CPU work, and unattributed overhead separately.

Record machine/CPU, available cores, memory, OS, compiler/build type, executable/source identity, numerical library and thread settings, network/segment workers, input sizes and acquisition duration, actual eligible entries, process startup counts, and cache state. Report sampled aggregate process-tree RSS as sampled RSS, not a guaranteed system high-water mark.

All primary latency runs process **one POINT workload at a time**. Fix one documented single-observation resource policy before collecting them; use the existing supported intra-observation parallelism and avoid nested oversubscription. No concurrent independent observations, ranks, injections, reports, or builds. Do not optimize worker layout halfway through the reference series.

Every independent timing repetition begins in a fresh process/output with original inputs and recomputes its RTC/CAL parent and learned numerical states. Reuse that parent's immutable values within the trajectory; do not rerun RTC for every feedback cycle. A warm OS file cache is allowed but must be identified. Fresh scientific state does not prove a cold filesystem cache. Do not flush system-wide caches or discard a slow first run. Reuse of a saved parent is allowed for response controls only and must never be labeled raw-to-result timing.

## 4. Phase A — unchanged end-to-end speed and useful iteration count

Run the current cold-initialized iterative PTC at **ranks 10 and 15**:

| Run family | Runs per rank | Purpose |
|---|---:|---|
| Independent original-input M0 through T10 | 3 | Repeated cumulative latency at every N=0...10; ordinary T10 endpoint and exit timing |
| Independent original-input runs terminating at N=0,1,3,5 | 1 at each N | Verify actual shortened invocation, endpoint/finalization cost, and absence of hidden later work |

Total: **14 original-input trajectories and 78 feedback applications**. The three long runs provide medians and observed ranges at every prefix. The shorter terminating checks are single measurements, not three-repeat statistics. Label measured prefix milestones, actual terminating invocations, and any derived estimates distinctly.

Run rank-10 REF first. Compare M0 and every cycle, discrete decisions, support, measurements, initial noise, and endpoint with the retained reference using existing predeclared numerical tolerances. Also verify rank-15 against its completed record. Treat a regression failure as an implementation/configuration problem, not a PCA result. Do not relax numerical regression requirements because a difference is sub-noise.

For shortened runs require exact agreement with the corresponding valid long-run prefix within those tolerances, and prove no later feedback application is performed. Preserve the normal final output/failure contract. N=0 must not create an unnecessary ten-cycle route behind the scenes.

Report T_ready(N), actual endpoint/exit times where measured, per-cycle marginal cost, and J by network/segment/cycle. A fitted fixed-cost-plus-per-cycle model may summarize the results, but show departures from linear behavior and do not replace measurements with it. Do not divide the previous 596-second parallel campaign by its 120 applications to infer POINT latency.

Alongside runtime, show peak, centroid, widths, fixed-template and aperture measurements, fit status, source-region structure, and changes relative to T10. Use the already retained reference injection trajectories to show compact plus/minus and shoulder response at N=0,1,3,5,10, after the new REF regression establishes their compatibility. They are comparison evidence, never restart state for the timing runs. Reuse retained map-domain controls or evaluate the existing ideal-addition measurement on the corresponding saved maps; do not repeat native reference injections unnecessarily.

Keep the existing conditional uncertainty recipe. The directly terminated REF runs supply endpoint scales at N=0,1,3,5; the long runs supply N=10. These endpoint companions never enter a subsequent feedback cycle. For other prefix changes, T10 uncertainty may be used as an explicitly labeled practical comparison scale, not as that earlier map's inferred uncertainty or a paired significance.

Identify the earliest *observed* useful prefix separately for pointing/centroid, compact brightness, and broader structure. Show when all later recorded measurements remain within 0.25 of the matching supported T10 conditional sigma, as a **diagnostic plateau marker only**, and show physical/percentage changes where that sigma is unavailable. This is not a new stopping rule, accuracy requirement, or proof of truth. Check cumulative difference to T10 and controlled response, not merely the last adjacent step. Do not retrofit the selector or stop production automatically.

Write the Phase A timing table and leading bottleneck findings before implementing the alternative methods. No owner pause is required to continue the explicitly authorized next phase after the reference gate passes.

## 5. Phase B — compare two approaches at the same rank first

Hold **rank 10** fixed initially. Compare the unchanged control with exactly two candidates. Document the actual current solver equations, initialization, and existing fallback before naming anything new; do not relabel an already implemented EM/alternating algorithm as a different method.

### B1. Warm-initialized, fully re-solved observed-entry fit

Use the current objective, centering, weights, validity, rank, stopping tolerance, and maximum internal iterations. M0 starts cold under the reference recipe. At later feedback cycles, initialize from the immediately preceding converged state belonging to the **same observation, method, rank, signal branch, network, and physical segment**. Recompute the current residual and centering, and solve the current problem to its existing convergence criterion.

This is a warm start, not a frozen basis, scheduled basis reuse, reduced-J shortcut, or fixed-subspace transfer experiment. Never seed an injection from REF, one rank from another, one segment/network from another, or a run from a previous timing repetition. Bind detector ordering, weights/support, and other state identities. Reject incompatible cached state before use; any structural cold fallback must be predetermined and logged, never triggered to conceal unfavorable science.

Measure initialization/solve costs and J. Report current observed-entry objective, convergence, and applied operator differences. Weighted low-rank problems need not reach the same solution from different initializations; equal tolerances alone do not establish scientific equivalence. Compare resulting maps/response and subspaces or applied operators rather than signs of individual eigenvectors. A solver failure stays a failure; do not retry cold and report only the favorable answer.

### B2. Direct masked-covariance PCA with validity-aware coefficient fitting

Test a genuinely different basis-estimation approach: estimate detector-pair covariance from jointly eligible current-residual samples, solve its symmetric eigenproblem, then fit/apply the requested basis with the existing validity-aware native coefficient/application machinery.

First inspect the retained non-iterative fallback. Reuse it if it implements the specified comparator; otherwise leave it intact and add only the small explicitly named native experimental method needed here. Do not bring a new scientific Python PCA implementation into the reporting driver.

For the existing binary observed-entry case, define the experimental covariance explicitly. If z_ti is the residual after the unchanged native centering and a_ti is its learning eligibility, use

    C_ij = sum_t(a_ti a_tj z_ti z_tj) / sum_t(a_ti a_tj).

Use the actual recorded learning domain and all eligible pair overlaps; do not restrict training to times when every detector is simultaneously valid. Require at least max(rank+1,2) joint samples for every covariance entry used by the full detector population. Unsupported entries make the candidate unsupported for that segment; do not assign zero covariance or delete detectors to rescue it. If the reference objective uses nonbinary weights, do not invent a weighting conversion: document that mismatch and leave this specific binary comparator blocked.

Use a native deterministic symmetric eigensolver with no whitening, per-detector RMS standardization, new noise weighting, smoothing, robust clipping, or shrinkage. Select the requested number of largest **positive algebraic** eigenvalues; do not sort by absolute eigenvalue. Pair-overlap normalization can produce an indefinite matrix. Retain the signed spectrum, overlap-count distribution, and negative-eigenvalue magnitude/mass relative to positive power as explicit diagnostics. Do not silently clip or repair the matrix to claim a valid conventional covariance. Fewer than rank positive supported eigenvalues, nonfinite entries, or failed eigen-residual checks is an explicit failure. This is a named approximate correlation-based cleaner, not a claim of equivalence to minimizing the observed-entry low-rank objective.

Preserve the native x/r operator semantics and application eligibility. Incomplete time rows use the existing observed-entry coefficient solve and its rank/support guards; no dense projection pretending missing entries are measured zeros. Any masked multiplication must handle invalid/NaN operands safely without making fills scientifically valid. No imputation may become map support. Fully observed small tests must reproduce ordinary centered PCA's subspace/operator to justified numerical tolerance.

Relearn this covariance/basis on **every** current residual. The candidate's missing-data approximation is part of its scientific comparison and must remain visible in the report.

### Not additional live candidates in this task

Record whether decomposition itself is expensive enough to motivate a later truncated/randomized-SVD experiment. Do not implement or benchmark a third solver here. Randomized decomposition is not by itself a missing-data treatment. Do not add robust PCA, independent components, frequency-domain modes, observation-wide learning, new detector weighting, or automatic rank selection.

## 6. Candidate matrix and source-response controls

For each of B1 and B2 at rank 10, plan **three independent raw-to-T10 REF timing runs** under the same machine/resource/output policy as Phase A. Use the first successful, predetermined REF as the science comparison member; the others measure runtime variability, not independent astronomical noise. Do not choose the best-looking repetition.

For that method/rank, run the retained CORE-PLUS, CORE-MINUS, and SHOULDER branches once each through T10. Use the exact existing full-precision rasters, coordinates, offsets, and taper. Inject once at the post-RTC/CAL, pre-PTC x boundary; r stays unchanged. The necessary original-derived parent may be kept as a bounded temporary spool for these controls. Every application starts from its branch's immutable parent, subtracts its current sky model, learns/applies PTC, restores exactly that subtracted operand, and maps the total.

Each method/rank learns its own unperturbed-M0 noise/guard package and freezes it under the established recipe. Its injected branches share that package; they do not train their own from injected brightness. Common physical template/aperture geometry remains fixed. Do not transplant rank-10 control noise, learned bases, or candidates into a new method.

Record all-cycle measurements; emphasize N=0,1,3,5,10. After timing completes, run the unchanged endpoint uncertainty companion on the representative candidate REF maps at N=1,3,5, in addition to T10. Keep this offline checkpoint analysis outside the measured processing history and label its cost; do not pretend it was an actual early-terminating invocation. Evaluate matching ideal-addition controls at the checkpoint maps without further native processing.

Use sigma_equiv = sigma_Q(own REF at N) / abs(g_Q at N) only for supported scalar compact responses with the same measurement geometry. Keep plus/minus separate. Show shoulder spatial distortion and raw-aperture versus fitted-background effects rather than a scalar correction. Show physical effects, fraction of the full source, and matching uncertainty scale; those scales are not paired p-values. Conditional noise remains conditional, not a full adaptive-noise ensemble.

Report full detector/sample eligibility, network populations, realized rank, solver/eigen residuals, application failures, and complete map support before any common-support comparison. A method that is faster because it loses samples or fails source acceptance is not a successful speed improvement. Do not lower the denominator by hiding lost coverage.

Phase B maximum: **12 trajectories, 120 feedback applications** across its two candidates, including their REF timing repeats. An invalid REF blocks its dependent injections; a scientific failure is a result, not a reason to alter masks or retry.

## 7. Phase C — at most one candidate at rank 15

From the completed rank-10 comparison, take at most one candidate forward only when it is numerically valid, preserves the input/application support contract, and shows a useful end-to-end time versus astronomical-quality tradeoff. Identify the choice transparently using repeat ranges, compact/pointing response, uncertainty, and broad structure—not pixel RMS or solver time alone. Do not force a winner when differences are inconsequential or all candidates are worse/unsupported.

For the selected method at rank 15, repeat the same three independent REF timing runs and three signed/broad response branches through T10, with its own noise package and checkpoint analysis. Compare against the measured current-method rank-15 control as well as its own rank-10 result. Do not also run rank 5 or combine multiple new methods.

Phase C maximum: **6 trajectories, 60 feedback applications**. This is experimental confirmation within this observation, not promotion or independent-observation validation.

## 8. Bounds, failures, resources, and preservation

The full planned maximum is **32 trajectories and 258 feedback applications**: A=14/78, B=12/120, C=6/60. Permit at most two additional trajectory attempts, each no longer than T10, only after documenting and correcting a specific implementation/build/routing defect. Thus the absolute ceiling is **34 trajectories and 278 feedback applications**, including failed attempts. No replacements for scientific rejection, disappointing response, unsupported covariance, or unfavorable speed.

The planned independent raw-parent count is 23 (A=14, B=6, C=3). One additional exact original-parent regeneration is allowed solely if a response-control spool must be recreated; the two defect replacements may add two more. Do not retain or copy whole observations for every branch. No separate real-observation profiling campaign or thread-layout sweep is authorized; obtain the stage profile from the instrumented runs.

Use the existing conditional-noise ensemble size and deterministic policy unchanged. Count timing-related repeated endpoints separately from the additional map-only checkpoint companions. At most 32 endpoint companion evaluations are planned, or 34 including defect replacements. No new seeds, injection amplitudes, faint-source ensembles, or statistical estimator development.

Keep truthful processing-failed/unsupported/blocked states. A command returns nonzero when required executed work fails or is incomplete; preserve unrelated completed cells. Deliberately unselected Phase C is distinct from failure. Retain the recently repaired later-cell failure propagation and add tests covering this experiment's dependencies, shortened N, warm-state isolation, and covariance failure paths. Distinguish a missing-data matrix with no valid solution from an implementation crash.

Use one bounded directory under `~/work_toltec/local_data/citlali-validation/`. Establish a numerical peak-disk ceiling before execution from actual allocations, not the old campaign's summed transient writes. Keep baseline timing I/O unchanged; account for cleanup wherever it lies on the measured critical path. Save compact maps at necessary checkpoints, small all-cycle measurements/receipts, timing events, configurations, and exact identities. Save only hashes and timing summaries for redundant deterministic repeats where sufficient. Aim for a few hundred MB of durable evidence, documenting any justified excess.

After committed compact preservation and verification that reporting needs no original run arrays, remove only this campaign's enumerated inactive detector-time scratch. Protect original/tune/APT/calibration/team inputs and unrelated work. Do not repeat a repository-wide purge or preserve failed scratch for historical reasons.

## 9. Deliver, review, and stop

Deliver an owner-facing report that leads with actual seconds, not solver jargon:

- Raw-to-usable and raw-to-completed-analysis runtime versus N for both control ranks, with repeat ranges, first-use/cache labels, actual terminating checks, and acquisition-duration ratios.
- The largest fixed and per-cycle costs, internal solver iterations, actual CPU usage, and the distinction between output availability and final process exit.
- What scientifically changes between N=0,1,3,5,10; when pointing, compact brightness, and broader structure become practically stable; no unsupported automatic stopping claim.
- A small current/warm/covariance table at matched rank, followed by any authorized rank-15 extension: total runtime, compact response-normalized uncertainty, centroid/shape, broad response, support, and failures.
- Conclusions at both equal feedback count and comparable achieved map/measurement quality. A faster per-cycle method that needs more cycles must not be described as an end-to-end improvement without that qualification.
- The fastest supported configuration in this bounded test, or an explicit finding that no alternative is yet preferable. State that no numerical operational POINT deadline was previously supplied and do not invent one as an acceptance test.

Produce machine-readable timing/matrix/measurement records and a small set of runtime-versus-N, measurement-versus-N, and response/uncertainty-versus-time figures. Separate panels/plots and labels must distinguish conditional statistical scales, systematic response deficits, and hardware scope. Final findings should make clear whether PCA is actually the leading optimization opportunity or whether RTC, initialization, I/O, process startup, or uncertainty analysis dominates.

Use the repository's established focused tests and bounded preflight/final independent review; do not rerun the science campaign during review. Commit code, compact evidence, and a concise reproduction/handoff locally. Preserve the old baseline tag/profile. No push, merge, production activation, automatic early stopping, other observation/array processing, RTC change, or further FRUIT method work.

**Stop with the measured speed-quality comparison and one recommendation for the next bounded action. Independent-observation testing remains the following stage, not part of this order.**

### Method references for implementation context

These references motivate the limited shortlist; they do not prescribe replacing native scientific contracts or adopting another package's defaults.

- Stephen Bailey (2012), *Principal Component Analysis with Noisy and/or Missing Data*, PASP; arXiv:1208.4122. Weighted/missing-entry PCA and initialization/convergence caveats.
- Ludovic Delchambre (2014), *Weighted principal component analysis: a weighted covariance eigendecomposition approach*, MNRAS; arXiv:1412.4533. Covariance-based estimation is distinct from the global weighted low-rank fitting objective. The binary covariance comparator above is explicitly defined here; do not import unrelated smoothing or regularization.
- Nathan Halko, Per-Gunnar Martinsson, and Joel A. Tropp (2011), *Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions*, SIAM Review; arXiv:0909.4061. A possible later decomposition acceleration, not an authorized third candidate or a missing-data solution.
