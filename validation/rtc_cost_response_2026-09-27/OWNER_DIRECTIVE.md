# Codex directive: RTC cost, contaminant removal, and source preservation

**Owner:** Grant Wilson  
**Date:** September 27, 2026  
**Status:** Execution order. Establish the fixed development endpoint, instrument and assess RTC, implement bounded evidence-driven improvements, test, review, commit locally, and stop. Do not return only a proposal.  
**Scope:** POINT 152430/0/2, a2000, networks 11/12, using the original acquisition and its exact dependencies. No new observation or array campaign.

> Settle on warm-start rank-10 PTC and five FRUIT feedback applications for this development effort. Return our attention to RTC. Determine where its time goes, whether it removes the intended contaminants, and what it costs in astronomical response and usable data. Optimize useful scientific work—not merely the appearance of the timestream. MAP/FRUIT is the measurement endpoint, not the next development project.

## 1. Decisions and authority

The working downstream configuration is **warm-initialized, fully re-solved observed-entry ALS; rank 10; M0 followed by exactly five FRUIT feedback applications**. This is an explicit development selection, not production promotion, a universal optimum, or an automatic stopping rule. Do not require another PCA timing comparison or independent-observation confirmation before starting RTC work.

This order supersedes earlier RTC-development deferrals only for the audit, pre-RTC injection wiring, implementation optimizations, and bounded cleaning revision described here. Keep the original cold-start rank-10/T10 reference and completed PCA comparisons intact. Do not resume rank selection, covariance PCA, other decompositions, FRUIT/admission development, or JINC comparisons.

The owner's estimate that RTC consumes roughly half the time is a motivation, not a verified stage measurement for the newly selected configuration. Re-establish its actual share. No numerical operational POINT deadline has been supplied; do not invent one.

### Recover the current implementation

Start from the completed POINT speed/PCA work, reported local closeout **`9c7586bb0`**, not the older freeze alone. Discovery anchors are:

- `validation/point_speed_pca_2026-09-27/ASSESSMENT.md`, its compact records, reproduction instructions, review, and closeout.
- `handoff/POINT_SPEED_PCA_2026-09-27.md`, `doc/BRIGHT_POINT_PIPELINE.md`, current status/goals, and applicable RTC/ALIGN/AST/CAL/PTC/VAL contracts and accepted amendments.
- `validation/ptc_rank_response_2026-09-26/` and the original tag `timestream-successor-bright-point-base-v1-20260926` at `5435bdbdb4e6708e884ae4cb2b0fd93e74102ce5`.

Verify actual ancestry, checkout, common Git directory, build, and unrelated changes. Use an owned development branch descended from the completed implementation; quoted identities are discovery anchors, not reset instructions. Preserve unrelated work, the frozen profile/tag, and prior compact evidence. Recover current accepted RTC behavior from implementation and owner decisions; do not silently substitute an older draft's x/r, learning-scope, or donor policy.

Use the existing `tools/timestream_successor/point_pipeline.py` route and native scientific owners. Add only the necessary configuration, instrumentation, diagnostic, and injection interfaces. Do not create another scientific pipeline or move RTC/PTC algorithms into the experiment driver.

## 2. Fixed endpoint and inputs

Use acquisition **152430/0/2**, its exact **152430/0/1** tune associations, networks **11/12**, and bound APT/calibration/telescope dependencies. Verify local holdings against the completed manifest. No download is expected. Unity remains the authoritative archive, but this order does not authorize a remote campaign or a new transfer. A missing indispensable original blocks only dependent work; do not substitute a nearby tune or an archived processed timestream.

Keep the existing ordinary MAP, 2-arcsec grid, containing-pixel projection, 40-arcsec feedback radius, seven-parameter Gaussian-plus-constant measurement/guard, source-connected threshold-3 BASE admission, measured-map feedback brightness, unity gain, and complete model replacement. Preserve high-coverage continuation and genuine numerical/validity checks. Do not add a Gaussian-only sky model, new admission statistic, or stopping rule.

**N=5 means five outer feedback applications after M0, not five internal ALS iterations.** T5 consumes m4; m5 may be recorded but is not applied. M0 starts cold. Each later PTC solve uses only the preceding converged state of the same observation, RTC variant, signal branch, rank, network, and physical segment. Recompute centering and solve the current residual to the existing criterion. Keep the objective, weights, validity treatment, rank, solver tolerance, and maximum internal iterations unchanged. No cross-run warm state, frozen basis, shortened solve, or favorable cold retry.

Every trajectory uses its own immutable post-RTC/CAL parent. Feedback subtracts the current model from that parent, freshly solves/applies PTC, restores the exact subtracted operand, and maps the total. Run the RTC workflow once per trajectory, not once per FRUIT cycle. Record any existing internal RTC plan-review/reapplication attempts separately; do not hide their costs.

Use saved warm-method T5 prefixes and completed response records as regression evidence. The first baseline invocation below must establish the genuinely terminating warm/N5 route, including its endpoint analysis. Do not repeat a T10 campaign to qualify that selection. A regression defect must be corrected, not hidden by switching methods.

## 3. First establish trustworthy RTC timing

### Remove the known accounting distortion

Before new timing, repair campaign-size-dependent storage accounting in the existing shared route. Use run-local accounting, tracked allocations/writes, and bounded reconciliation rather than recursively rescanning accumulated campaign files during each stage. Retain a meaningful disk ceiling and truthful failure handling. Test independence from unrelated historical file counts with a small synthetic filesystem fixture.

Apply the same corrected accounting and output policy to every new configuration. Preserve the old timing record as confounded evidence; do not subtract an estimated overhead and relabel it a measured speed. This repair is part of the RTC task, not another prerequisite campaign.

### Measure actual work

Use lightweight monotonic timing, actual CPU work, worker utilization, and sampled process-tree memory. Separate critical-path wall time from sums of parallel worker durations. Break RTC into its actual implemented operations, including, where present:

- Learning/review, spectral analysis and line decisions, and plan construction.
- Transient/shift detection, masks, support/replacement preparation, and boundary handling.
- Filter preparation/application, antialiasing/decimation, and output construction.
- Allocation/copying, serialization, filesystem work, bookkeeping, and unattributed overhead.

Also retain the surrounding ingress/ALIGN, CAL, PTC, MAP, fitting/admission, endpoint uncertainty, and finalization times. Distinguish native RTC computation from costs merely charged to its wrapper. Identify repeated spectra, filter construction, full-array copies, oversized per-sample records, avoidable serialization, and idle workers as hypotheses to measure—not presumed findings.

Report **raw-to-usable POINT**, **raw-to-completed numerical analysis**, and **command exit** separately. Required source-guard/noise work and synchronous output costs remain included. Label internal availability as internal if the caller cannot actually obtain the result until exit. Do not omit an ordinary default calculation to improve the advertised time.

Run one observation workload at a time with a fixed, documented resource policy and no concurrent science, builds, or reports. Each timing repetition begins in a fresh process with original inputs and fresh scientific state. Label OS-cache state and retain first-use observations. Do not extrapolate local two-network timings to all arrays or Unity. If the implementation candidate changes scheduling, hold its total available CPU/thread budget fixed and report the change explicitly; no thread-layout sweep.

Heavy diagnostics and optional dumps are not normal POINT work. Compute them outside the timed path where possible. Otherwise measure and include their collection cost, with identical instrumentation policies across compared runs. Never present an overhead-subtracted estimate as observed latency.

## 4. Audit actual contaminant removal

Establish the sequence **RTC input → RTC output → CAL/pre-PTC → PTC output**, using M0 and the final current-residual PTC pass where informative. Compare the sky-subtracted residual and restored total explicitly; do not mistake feedback restoration for contaminant return.

Lead with the unmodified observation and **full detector populations, network by network**. Report typical behavior, upper-tail behavior, affected fractions, and denominators. A few detector examples may illustrate findings but cannot supply the population conclusion. Keep temporal/segment variation visible.

Use one documented spectral/contaminant measurement recipe, preferably existing tested code. Match physical intervals, source exclusion, units, spectral resolution, and genuine support. Account explicitly for calibration and cadence changes. Do not zero-fill invalid samples for a naive spectrum, compare unrelated support, or call frequencies absent from the output grid “fully removed.” Show full-domain support losses separately before any matched-support comparison.

| Target | Required evidence |
|---|---|
| Narrow interference lines | Identify lines actually present, including the previously suspected near-11-Hz feature if present. Measure line excess above local continuum, absolute residual power, population prevalence, persistence, and residual coherence. State how much remains for PTC. A suppression ratio alone is insufficient. |
| Spikes and level changes | Event/exclusion counts and duration, residual event-like structure, false rejection around source crossings, and contamination near transitions. Detector flags alone are not independent proof that all events were found. |
| Replacement and boundaries | Where replacement was used, where it influenced retained samples, ringing or residual structure near gaps/events, and loss of usable support. Keep synthesized filter support distinct from measured science samples. |
| Filtering and sampling | Realized passband/source response, unwanted high-frequency residuals, edge behavior, and alias-control evidence. Use small deterministic fixtures where observational separation is unavailable. |
| Correlated residuals | Distinguish plausible instrumental residuals from atmospheric/spatial structure and source crossings. Slow common signal after RTC is not automatically an RTC failure or detector-health change. |

Keep source contamination out of background diagnostics with the existing physical sky/source definition projected through the actual detector pointings. Do not equate every low-frequency signal with atmosphere or every x/r correlation with electronics. Preserve the current raw-r parent and conditioned-r diagnostic roles; r is not assumed to be an optically pure noise witness.

Do not extend RTC into atmospheric common-mode subtraction or new calibration/rejection/weighting rules. Preserve accepted aggressive handling of suspicious data rather than inventing releveling, and preserve genuine speed-excluded samples as fixed-filter support where already authorized while excluding their output centers. Respect physical gaps, existing donor policies, and current Learn/Resolve/Apply scope. Observation-wide residency is not permission to change scientific learning or boundary scope.

For each target, conclude **effective at the measured scale**, **consequential residual**, **excessive loss/cost**, or **not established here**, with the evidence and limitation. A quiet observation does not validate a contaminant treatment it never exercised. Do not download a pathological observation or manufacture a large injection campaign to fill every cell.

## 5. Measure source response through RTC

The prior artificial sources entered after RTC/CAL. They remain useful downstream controls but do not establish RTC response. This order explicitly authorizes the missing **pre-RTC x-channel injection** at the admitted ALIGN→RTC boundary.

### Input and control construction

Reuse the exact retained compact-plus, compact-minus, and displaced broad/shoulder sky rasters: the existing signed ±10% compact increments and +5%-amplitude broader increment, including full-precision geometry, offset, and taper. Do not refit or renormalize them for a favorable RTC result.

Project the same physical sky onto the actual RTC-input detector timestamps/pointings. Use the accepted detector responsivity/calibration mapping to express the increment in RTC's raw Δf/f units. Do not insert calibrated brightness directly into raw x, invert an RTC filter, or interpolate backward through decimation. Verify the unit/cadence/projection mapping with small deterministic and calibration-only controls before real-data injection.

Add each increment once to an immutable original-derived aligned x parent. Leave input r unchanged and preserve immutable input validity. This is an explicitly x-only finite-response test beginning at RTC, not validation of optical response, tuning, raw-IQ conversion, ALIGN, or unknown source leakage into r. If the required raw-unit mapping is genuinely unavailable, report that limitation and complete the independent timing/contamination work; do not silently substitute the old post-RTC injection.

Each branch follows the **actual RTC learning, selection, and application path** and then the fixed CAL/PTC/MAP/T5 path. Preserve branch-specific plans, masks, decisions, and any changes induced by the injection. A frozen-plan linear replay alone cannot establish the adaptive pipeline's finite response. Such a replay may provide an explicitly conditional diagnostic, not replace these tests.

Use each RTC variant's own unperturbed M0 admission-noise/guard package, shared across its response branches under the established convention. Do not learn that package from injected brightness or transfer another RTC variant's numerical noise fields. This convention conditions the downstream test; it is not a full adaptive-noise ensemble.

### Measurements

Compare injected minus own REF at the RTC output and at M0/T5. Construct the existing ideal-addition control with the same physical sky definition, normal output grid/MAP coefficients, and that variant's unperturbed reference support, without RTC/PTC attenuation of the artificial increment. Do not redefine the ideal using newly lost injection support. Keep changes in coverage, normalization, and eligibility visible.

Measure peak, centroid, beam widths/shape, fixed-template and aperture response, broader residual morphology, and relevant conditional uncertainty. Keep compact signs separate. For the shoulder, retain raw-aperture and fitted-background terms rather than reducing all distortion to a scalar gain.

Show physical changes, fractions of the source, and matching uncertainty scales. Use response-normalized uncertainty only where a scalar compact response is supported; do not rescale science maps or claim paired significance from shared-data differences. Keep the existing noise estimator, ensemble size, and deterministic policy. No new uncertainty-calibration or source-model project.

Apply existing scientific goals to the relevant final product. Do not convert an approximately 5% final SCIENCE goal into a universal per-RTC acceptance threshold, or dismiss a numerical regression because it is sub-noise. When a tolerance is not established, show the measured effect and uncertainty rather than inventing a pass.

## 6. Make bounded, attributable improvements

Complete the baseline cost/contamination assessment before selecting changes. Record the selected rationale, exact configuration/source differences, comparators, and expected effect before candidate execution. No additional owner pause is required for work authorized here.

### B: one implementation-speed candidate

Implement **up to two tightly targeted implementation changes**, combined into one candidate, addressing measured leading costs. Examples are removing duplicate work, reusing immutable filter preparations, compact array representation, or supported intra-observation parallelism. These are permissions, not required changes without evidence.

Keep the RTC scientific method, resolved choices, numerical tolerances, x/r semantics, support, and downstream configuration unchanged. Reuse observation-wide information where already permitted; preserve scan/block boundaries and correct filter support. Do not obtain speed by skipping contamination checks, dropping detectors, shortening filters, loosening a solver, or changing cadence.

Require the established numerical regression tolerances plus exact identity for discrete plans/masks/eligibility where applicable. Test changes with small edge/transient/source fixtures. If an optimization changes scientific behavior, reject the equivalence claim; it is not silently absorbed into this candidate.

### C: at most one targeted cleaning revision

Only if the baseline exposes consequential contamination or unnecessary response/support loss, test **one separately named RTC-policy revision**, using an existing approved policy family and already authorized numerical choices. State the targeted contaminant and predicted benefit/cost first. A diagnostic bypass of one suspect optional treatment may instead fill this slot; it is not a production setting.

Keep all other scientific choices fixed and use B's implementation only if its equivalence gate passed. Otherwise use A's implementation. Do not simultaneously change notch, transient, donor, and sampling policies. No threshold/width sweep, new adaptive filter architecture, detector-selection tuning, or source-driven choice of favorable settings. Never bypass input validity, physical gaps, or required antialias protection.

If the evidence calls for a genuinely new scientific/numerical owner decision, identify that exact decision and leave C unselected. This does not block baseline completion or equivalent implementation improvements. No justified revision is an acceptable outcome; do not force a candidate to demonstrate activity.

## 7. Execution matrix and bounds

**A** is the existing scientific RTC with corrected accounting and the selected warm-rank10/N5 endpoint. **B** changes implementation only. **C** is the optional single scientific revision or diagnostic ablation.

| Family | Independent unperturbed raw-to-T5 runs | Pre-RTC response branches | Maximum trajectories |
|---|---:|---:|---:|
| A: required baseline | 3 | Compact +, compact −, shoulder: once each | 6 |
| B: one equivalent implementation candidate, if justified | 3 | None required beyond A's response evidence and focused equivalence tests | 3 |
| C: one targeted RTC-policy candidate, if justified | 3 | Compact +, compact −, shoulder: once each | 6 |

Maximum planned work is **15 trajectories and 75 outer feedback applications**. The first A REF is the endpoint regression and predetermined diagnostic/scientific representative, not an extra run. Use the first successful predetermined REF of each family for scientific comparison; timing repeats are not independent sky/noise realizations or a selection pool.

A and C provide measured pre-RTC response. B inherits the unchanged-method interpretation only after its equivalence checks; label its response evidence as inherited, not independently remeasured. A failed B equivalence gate blocks claims based on inheritance.

Allow **at most four additional RTC-only diagnostic executions** if needed for stage capture, conditional operation attribution, or response accounting that cannot be obtained from existing runs. They use the same observation/populations, do not launch additional PTC/FRUIT trajectories, and are not parameter searches. Reuse captured diagnostics whenever possible.

Allow **at most two replacement trajectory attempts**, each no longer than T5, only after a documented implementation/build/routing defect is corrected. Absolute limits are **17 trajectories, 85 feedback applications, and four extra RTC-only executions**. Failed attempts consume limits. Existing internal RTC plan-review bounds remain unchanged and their actual attempts must be counted. No replacement for scientific rejection, poor response, residual contamination, or disappointing speed.

Run only the families justified by preceding evidence. Shortcuts through saved original-derived parents are allowed for non-timing diagnostics/response branches, with truthful labels; independent REF timing always starts from originals. Do not fill unused budgets, add T10 runs, or start a new observation for completeness.

## 8. Evidence, disk use, and review

Use one campaign-owned directory under `~/work_toltec/local_data/citlali-validation/`. Establish a numerical peak-disk ceiling from actual allocations and available space before execution. Keep at most the necessary temporary stage/branch payloads; avoid per-sample/per-cycle tiny-file proliferation and observation copies for deterministic repeats. Track actual footprint separately from cumulative transient writes.

Preserve compact population diagnostics, spectra/band summaries, events/support counts, necessary map checkpoints, measurements, timing records, configurations, input identities, and reproduction code. Aim for a few hundred MB of durable evidence and explain any justified excess. Repeated deterministic runs generally need hashes/timings, not another full payload.

After durable compact preservation and a saved-only report check, remove only enumerated inactive scratch owned by this campaign. Protect original/tune/APT/calibration/team data, unrelated work, and indispensable evidence. Verify protection before and after cleanup. Perform any required exhaustive verification once in the shared guarded cleanup path; reviewers should verify that procedure and the compact evidence rather than launching redundant full scratch-file rehashes. Do not weaken safety checks or make historical workspace scanning part of observation latency.

Use the established focused native/Python tests and one bounded preflight/final independent review. Cover the warm/N5 route, unchanged RTC output for implementation changes, state isolation, injection units/sign/placement, real support through boundaries, and failure propagation. Reuse small fixtures rather than launching a review science campaign.

Required failed or incomplete executed work returns nonzero and retains truthful failed/blocked states; deliberately unselected B/C is distinct. A failed source guard preserves diagnostic evidence and the established failed-fit/empty accepted-output behavior. Do not switch methods, drop networks, or relax guards to rescue a result.

## 9. Deliver, commit locally, and stop

Lead the assessment with a compact **RTC cost–benefit table**: operation/target, seconds, contamination before and after, what PTC subsequently removes, source/support cost, and disposition.

Answer plainly:

1. What fraction of this actual warm-rank10/N5 reduction is RTC, and which costs are scientific computation versus avoidable overhead?
2. Which contaminants are convincingly reduced, which remain consequential, and which are not established by this observation?
3. What response, pointing, beam, uncertainty, or usable-support cost does RTC introduce before and after the fixed five-cycle endpoint?
4. Which implemented changes deliver a measured benefit at preserved or explicitly quantified scientific performance?

Give per-network population findings and matched raw-to-result repeat ranges. Clearly separate local/two-network scope, conditional uncertainty, source-transfer effects, inherited equivalence evidence, and untested contaminant classes. Do not equate lower RMS with better science or use PCA to conceal an RTC residual.

Update the development profile, scientific goals/status, reproduction instructions, and a concise RTC handoff. Record the selected endpoint and any retained improvement as development choices; keep the frozen reference intact. Commit source, compact evidence, and review locally. No push, merge, production activation, remote run, additional observation/array, or automatic follow-on.

**Stop with a measured answer to whether RTC earns its runtime, the bounded improvements actually demonstrated, and one clearly prioritized next action. Do not reopen PTC/FRUIT optimization.**

---

### Prior records to recover, not execute again

- `CODEX_POINT_SPEED_BASELINE_AND_PCA_APPROACHES.md`: fixed downstream policy, warm-state rules, timing definitions, and completed alternative-method scope.
- `CODEX_PTC_RANK_RESPONSE_NOISE_FIRST_TEST.md`: original inputs, retained signed/broad source definitions, common measurement geometry, and the post-RTC injection boundary.
- Completed September 27 speed/PCA assessment and handoff at the discovery anchors above: achieved results, timing limitations, and current implementation identity.

The decisions in this directive supersede the older deferrals only within its explicit scope. Historical results remain bound to their original configurations and execution identities.
