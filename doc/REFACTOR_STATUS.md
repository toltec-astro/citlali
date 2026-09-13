# Citlali Refactor Status

This is the living roadmap and completion ledger for the Citlali refactor.
Update it when a phase gate, governing decision, or validated snapshot changes.

## Governing Decision

On 2026-07-10 the project formally adopted the five-phase roadmap from the
[independent architecture review](../handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md).
The review verdict, **sound with material reservations**, is accepted.

The original [structural refactor plan](STRUCTURAL_REFACTOR_PLAN_2026-06-29.md)
remains the historical statement of intent. This document governs current
sequencing and exit criteria where the original plan differs.

The project will improve the existing tree incrementally. It will not restart
as a broad rewrite and will not rewrite the granular history of the validated
branch. The exact validated tree will remain available for forensic review.

## Current Integration Model

As of 2026-07-31, `codex/refactor-mainline` is the canonical application
integration branch. Continued scientific, diagnostic, configuration, and
operational development remains active there. The previous
`codex/structural-refactor` and `codex/fruit-loop-calibration-reference`
branches are retained as historical pointers rather than competing
application authorities.

The successor build proceeds independently on `codex/conan2-adaptation` in a
separate worktree created from the application mainline. It incorporates
mainline regularly and may return only after the bounded Adapt gates pass. It
does not wholesale-merge `citlali/v4.x_conan2`, replace the refactored
application, or mix numerical algorithm changes into build integration.

The live branch, upstream revision, gate, and import policy are recorded in
[`INTEGRATION_LEDGER.md`](INTEGRATION_LEDGER.md). The durable rationale is
[ADR 0008](adr/0008-application-mainline-and-build-adaptation-lanes.md).

## Documentation Contract

As of 2026-08-07, the [`doc/` documentation guide](README.md) governs routing
for new user-visible scientific work. Team-facing workflow and product meaning
belong under [`doc/user/`](user/README.md). Non-obvious reusable mathematics
and statistics belong in the registered method library under
[`doc/science/`](science/README.md). A technique is explained once and every
pipeline stage using it links to the same stable method ID; audits and product
guides do not duplicate derivations.

The framework is intentionally demand-driven. A user-visible scientific change
must update the applicable documentation before final acceptance, but no note
or guide is created solely to populate the framework. Existing dated reports
remain historical evidence and are not being reorganized wholesale.

On 2026-08-16 the project added the
[`Scientific Contract Library Program`](scientific_contracts/README.md) as the
governing process for new implementation-independent scientific contracts.
Every package must begin by linking to that charter and completing a prior-work
recovery record before fresh derivation is commissioned. The recovery step
adopts, cites, abstracts, supersedes, defers, or excludes earlier material so
approved scientific reasoning is not needlessly repeated while implementation
behavior, audit findings, repairs, and validation evidence remain outside the
independent author channel. The initial proposed pilot order is CAL, shared
mapmaking/coaddition, then Beammap; Beammap contract work remains separate from
active ALIGN work.

On 2026-08-26 the scientific owner approved and archived the downstream
contract sequence from MAP closure through separate JINC, NOI, filtering,
source/mode, and FRUIT tranches. On `2026-08-31` the owner changed the remaining
order so recovery-first SCI-FRUIT follows the completed single-pass
MAP/JINC/NOI/filtering authority line and precedes source-fitting and
Pointing/OOF. The owner also approved the horizontal
ALIGN-to-MAP decisions governing the mandatory PTC-to-MAP route, MAP-owned
admission policy, honest response and covariance disclosure, versioned later
derivatives, and coordinate/projection ownership. A bounded SCI-MAP Stage A
reopening is now active to reconcile those authorities with the older MAP
packet. The existing formal MAP clauses, 52 requirement IDs, 25 prediction
IDs, and local owner-decision ledger remain unchanged at launch; no Stage B
author, implementation change, conformity finding, validation result, Unity
work, freeze, performance claim, or readiness claim follows from this status.

The SCI-FRUIT v0.1 Stage A branch begins exactly at the conditionally frozen
SCI-FLT-FIXED record commit
`7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5`. Its recovery packet classifies
ordinary MAP, JINC, FLT-FIXED, and the provisional matched-filter snapshot at
`faff97565ee27e375e1337febe5a0a6681507c3b` as four separate candidate routes;
none is numerically admitted. After the owner rejected the original cumulative
immutable-increment framing, Stage A recovered the exact historical reference:
each iteration rereads the original observations, derives and subtracts a model
from one selected complete predecessor route product, runs residual-only PTC/
weight/noise operations, restores the accepted model, and carries a newly
completed raw or filtered observation/coadd product. The revised packet uses
`F_{k+1}=U_k(F_k,R_{k+1})`, does not make ordinary addition normative, and marks
the required projection/remapping, response, support, WCS/grid, weighting,
normalization, mask, filtering, and learning equivalence conditions as
unavailable or unproven. Revised ODQ-001 separates carried-state identity,
transition law, contribution status, and bounded persistence, then offers three
choices: preserve history, prove and validate an equivalent reformulation, or
intentionally adopt a new recurrence. No scientific choice, Stage B author
packet, algorithm change, frozen-package edit, validation, production, or Unity
activity is authorized or claimed.

The owner now provisionally favors the intentional-new-method category while
retaining the recovered historical recurrence as the mandatory compatibility
reference and scientific control. "Better" is to be established by controlled
comparison with an exact, versioned existing-Citlali benchmark. Scientific
quality and computational performance remain separate vectors: required
scientific domains include recoverable angular scale, recovered flux fraction
for declared astronomical modes, atmosphere/other residual leakage, and flux
convergence. The exact benchmark profile, metric definitions, tolerances,
uncertainty, non-inferiority constraints, improvement rule, and tradeoff policy
remain open. This direction selects no recurrence and authorizes no Stage B,
implementation, benchmark execution, validation, production, or Unity work.

The owner subsequently approved the constrained multi-objective comparison
framework as ODQ-001E: protect designated scientific dimensions with
non-inferiority constraints; require validity, response/uncertainty honesty,
exact-restart, and failure-disclosure gates; require material improvement in at
least one owner-prioritized scientific domain to justify an incompatible new
recurrence; and report computational performance separately unless an explicit
trade is later approved.

The owner then identified that an iterative FRUIT method cannot be selected by
an engineering-conformance-first sequence alone: hypothesis testing and tuning
may be required, and bright compact PSF recovery for OOF and faint extended SZE
recovery motivate potentially different policies or methods. Stage A r0.7
therefore reframes ODQ-001F as three owner choices: complete universal
parameterization before development; a staged profile-qualified empirical
method-development/qualification lane; or historical-compatibility v0.1 with
new-method R&D deferred to a successor. The fully bounded staged candidate
separates development, untouched qualification, and challenge populations;
permits only frozen offline tuning, qualified deterministic bounded adaptation,
or explicitly experimental expert override; freezes metrics/split rules before
development and the exact method/threshold/protocol before qualification; and
admits only a sanitized owner-approved qualified-method record to later
method-specific Stage B authorship. FRUIT recovery profiles do not acquire OOF
or SZE inference authority.

On `2026-09-01` the owner approved Disposition B in principle but required
targeted `r0.8` repair before final approval is recorded. The empirical program
does not seek one globally optimal FRUIT algorithm. Exact historical Citlali is
the paired scientific control, not truth; historical compatibility and any
operational fallback are separate roles, and no fallback is authorized.
Absolute truth/null metrics remain required. Qualification requires protected
scientific non-inferiority plus material, statistically credible paired
improvement in an owner-prioritized dimension, with prospective access,
multiplicity, inference, tail, stratum, support, failure, unavailable-state,
and catastrophic-regression safeguards. The `r0.8` candidate also separates
method, claim, evidence, and decision identities; states the complete
qualification-to-production claim layers; requires causal operational
stopping; and uses generic `compact_high_snr_response_recovery` and
`extended_low_snr_mode_recovery` profiles without acquiring OOF/SZE authority.
Valid outcomes remain one broad method, conditional policies, multiple
materially specialized methods, restricted-domain improvement, or no
replacement. The owner accepted the exact content-bound `r0.8` result on
`2026-09-01`, closing ODQ-001F, and requested Stage B at xhigh effort. No task
has been dispatched: the accepted sequence requires a separately authorized
empirical-development lane, frozen held-out qualification, an owner-approved
qualified-method record, and a sealed method-specific author packet, none of
which exists. Dispatch now would silently replace the accepted sequence with a
framework-only Stage B task. The owner has now preserved the accepted sequence
and directed preparation of the empirical-lane authorization packet. The
successor `r0.1` packet presents only a bounded Gate-0 registration-preparation
candidate because the exact historical-control execution profile, population,
metrics, thresholds, candidate family, and execution/resource bindings needed
for development are not yet available. The owner approved exact Gate 0 against
`EMPIRICAL_LANE_BUNDLE_MANIFEST_R0.1.md` on `2026-09-01`; work proceeds on
`codex/sci-fruit-v0.1-empirical-lane` from accepted packet commit
`90e55fa7d04fcab5cc716f7d258032275d6f6d7d`. The repository-only pass recovered
the exact historical recurrence source but found that source commit identity
alone cannot reproduce the executable because historical direct/transitive
dependencies floated. A preserved artifact or an anchored pinned
reconstruction and a human-custodied outcome-free population/control inventory
are required. A broad initial search also exposed historical
`validation/fruit_loop*` outcome evidence; that entire family and its lineage
descendants are quarantined from untouched qualification. No frozen
qualification population existed and no tuning or ranking occurred. Gate-D
launch is not ready. No profile, population, metric, threshold, recurrence,
parent route, numerical implementation access, empirical execution,
qualification, validation, readiness, production, fallback, Stage B, or Unity
activity is authorized.

On `2026-09-02` the owner separately authorized the local copied pointing data
under `fruit-development/point-152389` for development-only tests while
preserving the source data and original reductions, then approved proceeding
with a first controlled compact-source injection. This is not Gate-D launch or
qualification-population access. The frozen local pair injects a centered 100
mJy/beam source after RTC processing, starts from a fresh single-thread
iteration-0 checkpoint, and compares absolute FRUIT iterations 1 and 2 with an
otherwise identical control. The restarted iteration-1 control is bitwise
equal to the uninterrupted reference in signal, kernel, and weight for all
arrays. Kernel-normalized central amplitude recovery rises from 78--87 percent
at iteration 1 to 89--93 percent at iteration 2; by iteration 2 the fitted axes
are within roughly 3--5 percent of the realized processed kernel and centroid
separation is below 0.05 arcsec. The full response still changes by 37--51
percent of the preceding response-map RMS, so the result selects no stopping
point. It also supplies no pre-RTC, off-center, extended-mode, qualification,
historical-superiority, or candidate-ranking claim. The hashed compact record
is in
`validation/fruit_loop_point_152389_injected_development_2026-09-02/`.

The owner then approved extending that same frozen pair continuously through
absolute iteration 6 rather than launching a longer Unity run. The parent
iterations replay bit-for-bit. By iterations 5--6, kernel-normalized central
recovery is approximately 96--98 percent, the last-step central gain is only
0.12--0.35 percentage points, fitted axes are within 1.8 percent of the
same-iteration kernel, and centroid separation is below 0.09 arcsec. This is a
descriptive plateau for one bright centered compact source, not a general
convergence or stopping-rule result. A diagnostic replay from the exact
iteration-4 checkpoint then failed the required complete-trajectory restart
identity: iteration 5 is bitwise exact, but iteration 6 differs in `a1100`,
with signal relative RMS `0.121206`. The uninterrupted process retained prior
map-pixel-outlier records, used them to enable targeted contributor tracing,
and learned a new scan-local detector exclusion for UID 1489 at iteration 5.
The restarted process did not, because the checkpoint persists effective
masks and penalties but not that causally consumed diagnostic history. This
realizes the unsafe case anticipated by `FRUIT-GAP-011`; exact restart is
unavailable for this enabled learning path until repaired and validated. The
same replay completed iterations 5--6 in 80.74 seconds, so the original
1,361/2,650-second timings are a transient unresolved execution anomaly and
not performance evidence. The hashed development record is in
`validation/fruit_loop_point_152389_injected_convergence_development_2026-09-02/`.

A bounded D19 owner-review candidate now distinguishes the complete
iteration-boundary state from the diagnostic archive. It recommends resolving
and checkpointing the small next-iteration map-pixel target set rather than
retaining all prior map-pixel-outlier events. It also records that a candidate
detector ID is not sufficient: the detector is discovered only when the next
iteration traces contributors to those pixels. The proposal keeps
uninterrupted development available, while exact-restart, restart-dependent
qualification, and restart-dependent stopping claims remain unavailable for
the affected feature combination. See
[`EL_D19_ITERATION_BOUNDARY_STATE_CANDIDATE_R0.1.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_D19_ITERATION_BOUNDARY_STATE_CANDIDATE_R0.1.md).

On `2026-09-02` the scientific owner approved Choice A against that exact
candidate after reporting its proposal commit pushed. The bounded local repair
now resolves one policy-capped next-iteration target set per observation at
the completed-iteration boundary and makes uninterrupted and restarted
execution consume the same state. Current-iteration resolution evidence is
separate from the capped diagnostic archive, so record retention is no longer
causal. Checkpoint v3 carries the resolved scope, source/apply iteration, map
shape, and target tuples; explicit empty scopes are valid, while missing,
malformed, policy-incompatible, and grid-incompatible required state fail
closed. It does not retain map-pixel-outlier history or move detector
selection earlier. The CLI builds, 16 focused learning/checkpoint tests and 3
isolated real consumer-path tests pass, all 602 enabled CTests pass with one
pre-existing disabled test, all 147 baseline tests and 17 fruit-loop tool
tests pass, and full config preflight passes after updating its moved pointing
fixture locator. Repair commit `2b59ad642` then passed the real point-152389
split control recorded at `dd342448b`: all 27 signal, kernel, and weight planes
are bitwise equal at absolute iterations 5--7, and every checkpoint variable
is bitwise equal at all three boundaries. The restart restores 12 targets for
iteration 5; both trajectories then resolve the same 13, 3, and 3 targets and
retain the same seven detector penalties. Both runs have no unexpected
error- or critical-level message. D19 is closed for this enabled path; the
result validates exact continuation but does not qualify the FRUIT recurrence,
choose a stopping rule, or authorize restart-dependent scientific claims
beyond the tested state contract. The exact approval is recorded in
[`SCIENTIFIC_OWNER_D19_CHOICE_A_APPROVAL_2026-09-02.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_D19_CHOICE_A_APPROVAL_2026-09-02.md).
The hashed replay is in
[`validation/fruit_loop_point_152389_restart_v3_choice_a_2026-09-02/`](../validation/fruit_loop_point_152389_restart_v3_choice_a_2026-09-02/).

After reporting closure commit `9ba82bdaf` pushed, the owner agreed to resume
the empirical FRUIT lane and prepare the first candidate-recurrence experiment.
The current Gate-D assessment is
[`EL_G0_GATE_D_READINESS_R0.2.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_G0_GATE_D_READINESS_R0.2.md): D19 is no longer the blocker,
but the exact `f70701ad` historical executable and a protected finite
population remain unavailable. The preserved pointing artifact is instead a
refactor-era `c31a60a0` executable. The proposed
[`EL-F1 compact relaxation screen`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F1_COMPACT_RELAXATION_OWNER_REVIEW_R0.1.md)
therefore compares fixed `alpha=1.25` and `1.50` candidates only with an exact
same-build `alpha=1` compatibility control on the already exposed
point-152389 injection. The proposal is bound by
[`EL_F1_BUNDLE_MANIFEST_R0.1.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F1_BUNDLE_MANIFEST_R0.1.md)
and was approved exactly by the owner on `2026-09-02`. The isolated,
default-disabled prototype and focused tests passed before real-data execution.
The alpha-one control/injected trajectories completed iterations 0--6, but the
alpha-1.25 control stopped before iteration-1 scan processing. Two permitted
replacement attempts isolated first a non-spatial WCS representation mismatch
and then a bit-level mismatch between pre-serialization median RMS and the
decimal value reloaded from the FITS header (relative differences
`4.3e-16`--`3.1e-15`). The replacement allowance is exhausted; alpha-1.25
injected and both alpha-1.50 trajectories were not run. Diagnostic rebuilds
also mean the completed alpha-one pair cannot serve as a same-binary control
for a future repaired candidate. The screen is therefore **invalid**, not a
negative scientific result: no candidate map was ranked and no recurrence was
selected. A recommended successor would keep weights/RMS in the
checkpoint-bound ordinary complete product and keep only relaxed signal/kernel
plus exact spatial identity in the separate feedback state, then rerun all six
trajectories under one newly frozen executable if separately authorized. The
execution record is
[`EXECUTION_RESULT_R0.1.md`](../validation/fruit_loop_point_152389_el_f1_compact_relaxation_2026-09-02/EXECUTION_RESULT_R0.1.md).
No Gate D, qualification, Stage B, production, fallback, historical-superiority
claim, or Unity action is authorized or implied.
The narrowly bounded owner-review successor is
[`EL_F1_R1_COMPOSITE_STATE_RETRY_OWNER_REVIEW_R0.1.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F1_R1_COMPOSITE_STATE_RETRY_OWNER_REVIEW_R0.1.md),
content-bound by `EL_F1_R1_BUNDLE_MANIFEST_R0.1.md`. It recommends removing
duplicate weight/RMS authority, retaining the original scientific screen, and
rerunning all six trajectories under one frozen binary. The owner approved it
exactly on `2026-09-02`. The narrow repair makes the checkpoint-bound ordinary
complete map the sole weight/`MEDRMS` authority and limits the separate
relaxed state to signal/kernel plus exact identity. All six fresh trajectories
then completed iterations 0--6 on their first attempt under executable SHA-256
`a49082dde8f71d6f50edd8c378ad94195496b5eb0e0855b746e189f3442acbcc`,
with no error- or critical-level message and practical wall-time parity.
Alpha 1.25 and 1.50 both reached the alpha-1.00 iteration-5 central recovery
by iteration 4 in all arrays and passed every final recovery, width, centroid,
and full-map residual check, but failed the frozen final a1100 annular-residual
limit at 1.249 and 1.542 times the alpha-one residual, respectively, where at
most 1.10 was permitted. The valid prospective classification is therefore
**not promising on this compact case** for both candidates. No exact-restart
follow-up is required and neither candidate is promoted. The complete bounded
result is
[`EXECUTION_RESULT_R0.1.md`](../validation/fruit_loop_point_152389_el_f1_composite_state_retry_2026-09-02/EXECUTION_RESULT_R0.1.md).
This remains development evidence only and does not launch Gate D or Stage B,
qualify a recurrence, change production behavior, establish historical
superiority, or authorize Unity work.

The owner then favored a prospective follow-up that tests whether fixed
`alpha=1.25` stopped at iteration 5 can preserve the compact-source result of
`alpha=1.00` through iteration 6 on an independent pointing observation. The
date-separated observation 123424 was selected before inspecting any of its
FRUIT outcomes, and the owner retrieved its 12 raw detector files from Unity.
Local header checks bind observation 123424/sub-observation 0/scan 2 across
the raw and recomputed telescope files; the matched legacy APT covers the same
12 networks. APT v2 packaging is unavailable, so the exact legacy ECSV is
frozen as a common development input with an explicit non-qualification
limitation. The four-trajectory, unequal-terminal-iteration proposal is
[`EL_F2_INDEPENDENT_POINTING_EARLY_STOP_OWNER_REVIEW_R0.1.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F2_INDEPENDENT_POINTING_EARLY_STOP_OWNER_REVIEW_R0.1.md)
and is content-bound by `EL_F2_BUNDLE_MANIFEST_R0.1.md`. The owner approved
Choice A exactly on `2026-09-02`. The bounded unequal-terminal-iteration
analyzer extension then passed 10 focused tests, and the executable and
analysis were frozen before execution. The first scheduled trajectory stopped
before iteration 0 after 1.22 seconds: the approved local overlay incorrectly
labeled the KIDs fit-report path unused, producing 12 missing-fit-report
critical messages and a fail-closed NaN stop. No map or checkpoint was
created; the complete failed attempt is retained and the first of two allowed
environmental replacements is consumed. The owner then supplied 12 matching
processed tune fit reports covering the same observation and networks. The
bounded r0.2 correction adds their hashes and one path-only overlay while
leaving the scientific question, recurrence, matrix, metrics, thresholds,
order, and limits unchanged. It is
[`EL_F2_INPUT_BINDING_CORRECTION_OWNER_REVIEW_R0.2.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F2_INPUT_BINDING_CORRECTION_OWNER_REVIEW_R0.2.md)
and the owner approved it exactly against `EL_F2_BUNDLE_MANIFEST_R0.2.md` on
`2026-09-02`. All bundle and input identities reverified, but the authorized
replacement exposed a second packet error before iteration 0: the supplied
files are processed tune NetCDFs, while this executable selects ECSV/ASCII
text fit reports with per-network `.txt` regular expressions. It emitted 12
missing-fit-report critical messages, produced no map or checkpoint, and used
the second and last r0.1 environmental replacement. A read-only local search
then found all 12 required text tables. Each uniquely matches the executable's
observed pattern, parses as 14-column ECSV, carries the correct observation/
sub-observation/tune-scan/network metadata, and has the same row count as the
corresponding tune NetCDF's tone dimension. Acquisition provenance beyond the
existing local development path is unavailable. The exact r0.3 owner-review
candidate is
[`EL_F2_TEXT_FITREPORT_CORRECTION_OWNER_REVIEW_R0.3.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F2_TEXT_FITREPORT_CORRECTION_OWNER_REVIEW_R0.3.md),
content-bound by `EL_F2_BUNDLE_MANIFEST_R0.3.md`. The owner approved it exactly
on `2026-09-02`, authorizing one explicitly final environmental replacement.
That replacement and the other three primary trajectories completed in the
frozen BAAB order with the expected iteration sets and no error- or
critical-level messages. Candidate pair-mean wall time was 172.175 seconds
versus 201.610 seconds for the reference, a 14.600-percent reduction that
passes the 10-percent performance target. The scientific protections do not
pass. Candidate iteration 5 fails width and annular-residual limits in all
three arrays; a1400 additionally falls from 0.890451 recovery at iteration 4
to 0.822828 at iteration 5, with 0.299-arcsec centroid error, 7.253 times the
reference annular residual, and 2.399 times its kernel-residual structure. The
valid prospective classification is therefore `does_not_replicate`. The same
candidate is not rerun, and the conditional exact-restart replay is neither
required nor authorized. The complete result is
[`EXECUTION_RESULT_R0.3.md`](../validation/fruit_loop_point_123424_el_f2_early_stop_2026-09-02/EXECUTION_RESULT_R0.3.md).
This independent negative result does not promote a method, stopping rule,
APT, or fallback and does not launch Gate D, Stage B, production, or Unity
work.

The owner then approved a read-only causal-diagnostic comparison of the
completed EL-F1 and EL-F2 trajectories. The 51-row analysis keeps paired
injection truth out of candidate warning signals and finds no tested simple
single-run diagnostic that reliably separates helpful from harmful updates
across the two exposed development cases. Update growth catches the large
EL-F2 a1400 iteration-5 failure but misses both late EL-F1 a1100 failures and
has direct counterexamples on improving or acceptable array states. A new hard
detector penalty likewise is not universal. It does, however, localize a
specific EL-F2 mechanism candidate: the injection changes UID 4460 on scan 5
from three contributing a1400 pixels to the configured threshold of four at
iteration 4, producing a newly carried zero-factor exclusion before the
iteration-5 collapse. The control has no such exclusion. Because no
counterfactual has yet been run, this is an association rather than a causal
claim. Every EL-F2 alpha-1.25 all-array state from iterations 1--5 still fails
at least one inherited science protection, so this evidence does not rescue an
iteration-4 early stop. The complete result and recommended separately
authorized counterfactual are in
[`CAUSAL_DIAGNOSTIC_RESULT_R0.1.md`](../validation/fruit_loop_causal_diagnostic_discovery_2026-09-02/CAUSAL_DIAGNOSTIC_RESULT_R0.1.md).
No stopping rule, candidate recurrence, method promotion, qualification,
production change, Gate D, Stage B, or new reduction is authorized.

The owner then delegated the bounded recommended counterfactual and encouraged
it to proceed. `SCI-FRUIT-EL-F3-LATE-PENALTY-COUNTERFACTUAL-R0.1` copied the
exact EL-F2 alpha-1.25 iteration-4 checkpoints, replayed an untouched control,
and removed only the carried factor-zero UID 4460 a1400 penalty from the
injected copy before advancing each state once. The sham control reproduces
all iteration-5 signal, kernel, and weight images bitwise in all arrays and all
checkpoint variables value-for-value. In the counterfactual, a1400 central
recovery is 0.901542 rather than 0.822828 and annular residual over truth is
0.00288776 rather than 0.0214741. The two prospectively registered reversal
fractions are 1.16401 and 1.17020, so the result is
`substantial_causal_contribution` with full reversal. Every a1100 and a2000
signal, kernel, and weight image remains bitwise equal to the original
injected iteration 5; only a1400 changes. UID 4460 is learned again at the end
of the counterfactual iteration, so the test isolates the consequence of
applying the prior record and makes no later-iteration claim. The inherited
all-array science screen still fails: counterfactual a1400 remains narrowly
outside both width protections, while a1100 and a2000 retain their prior
failures. This establishes a causal penalty-policy problem in one exposed
checkpoint, not a general stop rule, detector judgment, qualified recurrence,
or rescue of EL-F2. The complete record is
[`EXECUTION_RESULT_R0.1.md`](../validation/fruit_loop_point_123424_el_f3_penalty_counterfactual_2026-09-02/EXECUTION_RESULT_R0.1.md).
Implementation ordering shows that the causally consumed detector-dominance
learner currently sees the complete post-add-back raw-observation map rather
than a residual-only map. A later proposal should first test a prospective
model-bypass/source-protection policy against harmful and benign events in
multiple observations; provisional penalties remain a separate fallback
hypothesis. No additional run is authorized by EL-F3.

After reporting the EL-F3 interpretation pushed, the owner authorized the
bounded EL-F4 feedback-model-bypass screen and encouraged it to proceed. The
default-disabled development option changes only hard detector-penalty
evidence from the complete map `Q_k` to the map-domain view
`E_k = Q_k - F_(k-1)`; complete products, mapdiag products and archives, D19
target selection, and every other learned state remain unchanged. This is not
claimed to be a literal sample-domain residual or an equivalent recurrence.
All eight registered fresh trajectories completed on their first attempt under
one frozen executable with no error- or critical-level message. With the
option disabled, 234 signal/kernel/weight planes reproduce the corresponding
EL-F1-R1/EL-F2 products bitwise; the 18 candidate iteration-zero planes are
also bitwise equal before injection.

The primary mechanism gate passes with full reversal. The candidate omits the
injection-specific UID 4460 a1400 penalty, and observation 123424 iteration-5
a1400 recovery is `0.901542068` rather than `0.822828191`, while annular
residual is `0.00288776008` rather than `0.0214741060`. The registered
reversal fractions are `1.16401066` and `1.17020049`. The global bypass is
nevertheless regressive: it removes 15 complete-map penalties, retains only
two, and introduces four protected failures. Observation 123424 a2000
kernel-residual ratios are `1.53114`, `1.23537`, and `1.12811` at iterations
3--5, and observation 152389 a1100 iteration-4 annular residual is `1.20870`
times its same-build complete-map control, all above the registered `1.10`
limit. No new penalty or timing shift occurs. Pair-mean wall time changes are
only `+0.467%` and `-0.733%` for observations 123424 and 152389, respectively.
The prospective disposition is therefore
**`mechanism_helpful_but_regressive`**: the causal concern is confirmed, but
this exact wholesale policy does not advance. The development option remains
default-off evidence instrumentation, not a production behavior or qualified
FRUIT method. The complete result is
[`EXECUTION_RESULT_R0.1.md`](../validation/fruit_loop_feedback_model_bypass_2026-09-02/EXECUTION_RESULT_R0.1.md).
No additional run, candidate, recurrence, stopping rule, Gate D, Stage B,
fallback, production change, or Unity activity follows automatically. A
future proposal must prospectively distinguish the harmful injection-sensitive
penalty from ordinary detector evidence rather than merely retuning this broad
bypass after seeing the failures.

Before EL-F5, the owner requested a read-only interpretation check of the
observation-123424 injection geometry. The check found that the synthetic
source was placed at the nominal map origin while the real Neptune centroid
was offset by about 14 arcsec (`13.608` arcsec in a1400 at the relevant
iteration-4 boundary). The source cores are not coincident, but both occupy
the same central region: the a1400 control map already contains
`476.849 mJy/beam` at the injection origin before the recovered synthetic
increment of about `72.186 mJy/beam` is added. Injected-source recovery is
measured from injected-minus-control maps, so it is the causal response to an
added source in this real Neptune field, not a guaranteed isolated-source
response under nonlinear and stateful processing. The UID 4460 penalty pixels
are instead 107--109 arcsec from the injection and 94--95 arcsec from Neptune;
the injection changes their map values by only 0.50--0.58 mJy/beam but moves
the detector from three qualifying pixels to the hard threshold of four.
EL-F3 therefore retains its exact causal conclusion while losing any generic
isolated-source interpretation, and EL-F4 still rejects the wholesale bypass
without establishing a generic source/penalty mechanism. The read-only record
is
[`INTERPRETATION_CHECK_R0.1.md`](../validation/fruit_loop_point_123424_injection_geometry_interpretation_2026-09-03/INTERPRETATION_CHECK_R0.1.md).
The owner then rejected an automatic centered replication because the existing
centered output already supplies that comparison. The accepted next step is
the smaller EL-F5 location control: prove exact zero-offset compatibility, run
a same-build no-injection control, and place one 100 mJy/beam source at the
prospectively fixed `(AZOFFSET, ELOFFSET) = (0, -60)` arcsec position under the
unchanged complete-map policy. The position was selected from existing control
weights only and has adequate three-array coverage. A fresh centered
trajectory is conditional on a failed compatibility check or a specific
post-test ambiguity; it is not part of the registered matrix. The owner
authorized this bounded registration, diagnostic implementation, local build
and test, freeze, and two-trajectory execution on 2026-09-03. This does not
authorize a recurrence or penalty-policy change, NGC4449 campaign, Neptune
subtraction, qualification, production use, Gate D, Stage B authoring, or
Unity activity. The prospective definition is
[`TEST_DEFINITION.md`](../validation/fruit_loop_point_123424_el_f5_off_source_injection_2026-09-03/TEST_DEFINITION.md).

The EL-F5 diagnostic implementation is now locally verified but has not yet
been executed. It adds only finite map-world AZOFFSET/ELOFFSET controls to the
default-disabled injected-source test, routes the source through the normal
RTC kernel and existing filtering/cleaning path, and records the position in
effective configuration, processed-timestream provenance, NetCDF variables,
FITS headers, logs, and response identity. Missing or explicit zero offsets
retain the literal centered-kernel branch; a sample-level unit test verifies
bitwise equality, and a displaced-kernel test verifies the signed map-world
position. The analysis window, EL-F4-derived annulus, compatibility comparator,
target event, and four-way descriptive classification were fixed in
[`ANALYSIS_MANIFEST_R0.1.yaml`](../validation/fruit_loop_point_123424_el_f5_off_source_injection_2026-09-03/ANALYSIS_MANIFEST_R0.1.yaml)
before output inspection. The production CLI and test binary build, all 618
enabled CTest cases pass (one unrelated test remains disabled), 13 focused
Python tests pass, and the complete required configuration preflight passes
without changing its raw-execution census. No trajectory, compatibility
finding, off-source result, or scientific interpretation is claimed at this
point.

The verified EL-F5 executable and exact eight-file setup were subsequently
copied into the new development output root and frozen before either
trajectory. The executable SHA-256 is
`6431c6653ed46ff6e1dfa5512cd27e8169525f7a110207b0b24505786f39dbbe`
and its clean source identity is
`fd760cdbf59940f803ab38323088b35682f342cd`. The immutable record and fixed
merge order are in
[`FROZEN_INPUTS_R0.1.md`](../validation/fruit_loop_point_123424_el_f5_off_source_injection_2026-09-03/FROZEN_INPUTS_R0.1.md).
The registered control remains the mandatory first trajectory, followed by
its bitwise comparison with EL-F4 before the off-source output may be opened.

Both registered EL-F5 trajectories have now completed on their first attempt.
The new disabled-injection control reproduces all 54 EL-F4 observation-123424
signal/kernel/weight planes bitwise over iterations 0--5. The off-source
kernel and transfer centroids remain within `0.239` arcsec of the declared
`(0, -60)` arcsec position across all arrays and injected iterations. The
complete 18 response maps are retained with deterministic hashes. The same
injection-specific hard record reappears at iteration 4: scan 5, UID 4460,
a1400, score 4, factor zero. The control has three qualifying contributor
pixels for that detector and the injected run has the threshold four.

The prospectively registered disposition is
**`same_event_replicated_off_source`**. In a1400, the annular residual rises
from `0.00327048` to `0.0231344` (`7.074x`) and kernel-residual relative RMS
rises from `0.320673` to `0.727804` (`2.270x`) from iteration 4 to 5. The
Gaussian/kernel-normalized central response decreases only from `1.043975` to
`1.037055`; because both exceed unity, this is slightly closer to correct
amplitude rather than a flux degradation, while whole-kernel recovery is
essentially unchanged (`1.056239` to `1.056704`). Thus central Neptune overlap
is not necessary for the exact penalty event in this observation, and the
scientifically adverse replication is shape/residual leakage, not lost total
flux. The result strengthens a same-observation injection-state/penalty
hypothesis but does not establish generality across observations or causality
for the off-source response change. Aggregate runtime was `344.67` seconds,
maximum resident memory was `0.912` GiB, and no error/critical message was
emitted. The complete bounded result is
[`EXECUTION_RESULT_R0.1.md`](../validation/fruit_loop_point_123424_el_f5_off_source_injection_2026-09-03/EXECUTION_RESULT_R0.1.md).
No blank-field test, recurrence change, penalty-policy change, qualification,
production use, Gate D, Stage B, or Unity activity follows automatically.

After pushing the EL-F5 result, the owner authorized the recommended bounded
EL-F6 causal replay. The registered test copies the EL-F5 off-source injected
iteration-4 checkpoint twice, requires an untouched sham to reproduce the
original iteration 5 exactly, then removes only the carried scan-5 UID 4460
a1400 factor-zero penalty before a second one-iteration replay. Its primary
causal quantities are prospectively fixed reversal fractions for the two
actual EL-F5 degradations: kernel-residual relative RMS and 40--120 arcsec
annular residual. Central amplitude is reported by distance from unity and is
not a causal gate. The registration is
[`TEST_DEFINITION.md`](../validation/fruit_loop_point_123424_el_f6_off_source_penalty_counterfactual_2026-09-03/TEST_DEFINITION.md).
The existing checkpoint editor and the off-source-specific analyzer pass all
222 baseline and FRUIT-loop Python tests. Both source copies were recursively
verified before the registered one-row intervention; its audit confirms that
only the exact UID 4460 record was removed. The executable, seven-file stacks,
source hashes, transformed checkpoint, analysis identities, and fixed run
order are frozen in
[`FROZEN_INPUTS_R0.1.md`](../validation/fruit_loop_point_123424_el_f6_off_source_penalty_counterfactual_2026-09-03/FROZEN_INPUTS_R0.1.md).
Both one-iteration replays then completed normally. The untouched sham passed
its gate: all nine image planes are bitwise equal to the original EL-F5
iteration 5 and every checkpoint variable is value-identical. The
counterfactual also completed, but the first analyzer invocation stopped
before retaining or displaying a result because the timing parser accepted
GNU token order but not the local macOS order. A tested pre-result portability
repair accepts both forms without changing any scientific calculation or
gate; its hashes and scope are recorded in the frozen-input note. Scientific
analysis then completed under the unchanged registered rules. Removing only
the carried UID 4460 penalty reverses `0.957935` of the a1400 kernel-residual
loss and `0.960110` of the annular-residual loss. Both exceed the prospective
0.5 threshold, so the valid classification is
**`substantial_causal_contribution`**; neither reaches the 1.0 full-reversal
threshold. Central and full-kernel recovery remain effectively unchanged and
move slightly closer to unity. All six a1100/a2000 planes remain bitwise equal
to the original injected iteration 5; only a1400 changes. UID 4460 is learned
again at the end of iteration 5, so this result makes no iteration-6 claim.
EL-F6 establishes this off-source causal effect for one observation and
location, not a generic detector or policy rule. The complete result is
[`EXECUTION_RESULT_R0.1.md`](../validation/fruit_loop_point_123424_el_f6_off_source_penalty_counterfactual_2026-09-03/EXECUTION_RESULT_R0.1.md).
No further test, recurrence change, penalty-policy change, qualification,
production use, Gate D, Stage B authoring, or Unity activity follows
automatically.

The scientific owner subsequently identified that the EL-F6 a1400 residual
image appeared to spread the source over several arcs and approved a focused
interpretation correction. The preserved r0.1 numerical result and registered
`substantial_causal_contribution` classification remain unchanged, but their
scientific meaning is narrower. The injected compact component remains at its
declared position with a `0.065`-arcsec response/kernel centroid separation and
only 1.4--1.6 percent width excess over the processed kernel. Applying the
UID 4460 exclusion changes the 20-arcsec injected-source region by only
`0.2581 mJy/beam` RMS, but changes the 20-arcsec fitted-Neptune region by
`2.4370 mJy/beam` RMS; without the penalty the latter injected-minus-control
residual is only `0.1024 mJy/beam` RMS. The trigger pixels also lie on the
observation-wide UID 4460 scan trajectory to about 1.1 arcsec. Because the
injected and control branches have different detector participation before
shared RTC/PTC processing, real-field and scan-synchronous material does not
cancel and appears as several arcs in the total paired response. The active
[`r0.2 result`](../validation/fruit_loop_point_123424_el_f6_off_source_penalty_counterfactual_2026-09-03/EXECUTION_RESULT_R0.2.md)
therefore treats the registered annular and kernel-residual measures as total
paired-response leakage, not an isolated astronomical point-source halo or a
telescope-pointing failure. It also corrects the diagnostic image's horizontal
orientation to follow the frozen FITS `AZOFFSET` WCS; no numerical metric is
affected. Future source-transfer experiments must distinguish total
end-to-end adaptive response from matched-operator source transfer and report
real-field leakage separately. No reduction, algorithm, configuration, or
authorization changed.

After reviewing the study summary, the scientific owner agreed to formalize
those response distinctions and proceed toward the next narrow empirical test.
The resulting owner-review packet proposes
`SCI-FRUIT-EL-F7-SHARED-START-RESPONSE-DECOMPOSITION-R0.1`. A read-only check
found that the EL-F5 injected and control iteration-4 checkpoints differ not
only by the target UID 4460 penalty but also in feedback maps, accumulated
weight evidence, one mask interval, and learned target pixels. EL-F7 therefore
places a single off-source injection into iteration 5 after both branches
start from exact copies of the same EL-F5 control iteration-4 state. One
no-injection sham first has to reproduce the already existing control
iteration 5 exactly. The one new probe map then completes the exact telescoping
decomposition `T5 = S5 + H5 + D4460,5`, separating the total adaptive response
into a shared-incoming-state one-step response, other inherited injection
history, and the already isolated UID 4460 effect. The proposed
[`response-measurement frame`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_RESPONSE_MEASUREMENT_FRAME_R0.1.md)
does not call the shared-start quantity a fully matched-operator transfer;
same-iteration data-dependent processing can still differ. The exact
[`owner-review proposal`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F7_SHARED_START_RESPONSE_OWNER_REVIEW_R0.1.md)
and
[`bundle manifest`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F7_BUNDLE_MANIFEST_R0.1.md)
were approved exactly by the scientific owner on 2026-09-03. The bounded
authorization is recorded in
[`SCIENTIFIC_OWNER_EL_F7_AUTHORIZATION_2026-09-03.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F7_AUTHORIZATION_2026-09-03.md).

Both registered one-iteration replays completed on their first attempt. The
no-injection sham reproduces all nine signal/kernel/weight planes bitwise and
the complete existing control iteration-5 checkpoint value-identically. The
realized paired configurations, units, WCS/grid, normalization, and finite
support match, and the four-component identity closes to
`1.4211e-14 mJy/beam` against predeclared bounds of
`1.1695e-12`--`1.5575e-12 mJy/beam`. The new shared-start response recovers
kernel-normalized central amplitudes `0.858066`, `0.885121`, and `0.739569`
for a1100, a1400, and a2000 after one injected transition. Earlier injected
history adds a positively aligned compact component while strongly cancelling
shared-start annular structure (annular cosines `-0.915`, `-0.557`, and
`-0.511`). In a1400 the separate UID 4460 effect remains nearly equal in RMS
to the total response around Neptune (`2.43696` versus `2.44263 mJy/beam`) and
in the registered annulus (`2.27360` versus `2.29897 mJy/beam`), while changing
the injected-source region by only `0.262866 mJy/beam`. The shared-start probe
does not relearn UID 4460; its completed checkpoint differs from control only
in feedback signal and kernel maps, not persisted penalties, masks, weight
accumulators, or target pixels. The active
[`EL-F7 result`](../validation/fruit_loop_point_123424_el_f7_shared_start_response_2026-09-03/EXECUTION_RESULT_R0.1.md)
therefore supports a future narrow carried-hard-penalty safeguard proposal
that preserves accumulated source history. It does not establish a fully
matched operator, select that safeguard, qualify a recurrence, launch Gate D
or Stage B, change production, or authorize Unity activity.

After EL-F7, the scientific owner asked the manager to lead the next step but
questioned how one detector could make such a large map imprint without being
flagged or deweighted. A read-only implementation/evidence check resolves the
apparent contradiction: UID 4460 was APT-accepted and moderately deweighted
(`0.7587`, near the 36th percentile from the low-weight end among 445
validated unflagged a1400 detectors), then the four-pixel map-dominance trigger
fully flagged all 676 raw samples in scan 6 before RTC during iteration 5. The
four detector-specific leave-one-out significances were only `1.70`--`2.12`;
the current hard rule requires four globally unusual pixels but no separate
minimum leave-one-out significance. Removing the detector before shared
cleaning can change every a1400 detector's processed response, so the arcs are
not a direct image of UID 4460 alone. The exact
[`mechanism note`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F8_UID4460_MECHANISM_NOTE_R0.1.md)
supports a narrow proposed EL-F8 placement decomposition: preserve the
complete FRUIT feedback state and the hard decision, but compare the existing
pre-cleaning exclusion with an opt-in exclusion applied only before final map
accumulation. The
[`owner-review proposal`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F8_PENALTY_PLACEMENT_OWNER_REVIEW_R0.1.md)
and exact
[`bundle manifest`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F8_BUNDLE_MANIFEST_R0.1.md)
were approved exactly by the scientific owner on 2026-09-03. The bounded
authorization is recorded in
[`SCIENTIFIC_OWNER_EL_F8_AUTHORIZATION_2026-09-03.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F8_AUTHORIZATION_2026-09-03.md).
The two existing-placement compatibility replays must pass before either
mapmaking-placement result may be interpreted. After fail-closed repairs to
historical restart-policy compatibility, exact checkpoint intervention,
application accounting, macOS timing parsing, and result serialization, four
fresh R0.4 trajectories completed with zero unexpected error or critical
messages. The two existing-placement replays reproduce all nine EL-F5
signal/kernel/weight planes bitwise and all scientific checkpoint values. In
the map-placement injected replay, all 676 UID 4460 raw samples enter shared
RTC/PTC cleaning; the learned record then excludes all 305 proposed
downsampled samples before map accumulation (271 newly flagged and 34 already
flagged). The registered component identity closes to
`1.4211e-14 mJy/beam` against a `1.4452e-12 mJy/beam` bound in a1400.

The scientific classification is **mixed, with the direct mapped contribution
larger in the registered Neptune and annular regions**. Around Neptune the
direct and early/shared terms contribute 76.86 and 24.51 percent of the
squared-RMS accounting, with a -1.37 percent cross term; in the 40--120 arcsec
annulus they contribute 81.65 and 17.95 percent, with a +0.40 percent cross
term. Across the complete map the split is closer to 58.09/42.29 percent.
Moving the penalty therefore removes a material shared-cleaning interaction
but does not remove the larger direct scan-trajectory imprint. The compact
a1400 injection remains stable: central recovery changes from `1.037055` to
`1.037009`, and the early/shared aperture response is about -0.7 percent. The
active
[`R0.4 scientific interpretation`](../validation/fruit_loop_point_123424_el_f8_penalty_placement_2026-09-03/SCIENTIFIC_INTERPRETATION_R0.4.md)
answers the owner's detector-leverage concern without judging UID 4460 or
selecting a safeguard. The production default, a soft factor, threshold
change, recurrence selection, qualification, Gate D, Stage B, and Unity remain
outside the authorization. The recommended next decision is a separately
authorized read-only map-leverage and flagging audit before any candidate
safeguard comparison.

The scientific owner then authorized that bounded read-only audit. Registered
EL-F9 source inspection corrected an important prospective premise before it
was interpreted: for the fixed JINC products, `weight_formal_I` is the
nonlinear finalized coefficient `G^2/V`, not an additive detector-weight
plane. The pre-normalization grid denominator `G`, variance accumulator `V`,
signal numerator `S`, and detector-resolved components are not published.
The paired formal-coefficient difference is materially negative in 2,019
pixels and positive in 2,448, which proves it cannot be used as UID 4460's
fractional leverage. EL-F9 therefore honored its availability stop: it ran no
new reduction and substituted no proxy leverage.

The remaining flagging trace completed. UID 4460 is APT-accepted; its scan-5
PTC detector weight is at the 79th percentile from the low end and its
flagged fraction sits near the midpoint of a large tie, while its residual
RMS and standard deviation are high at the 91st and 93rd percentiles and its
median is near the 1st percentile. The map rule records four globally extreme
pixels for which the detector-specific leave-one-out significances are only
`1.70`--`2.12`; four is exactly the repeat threshold, and the resulting
factor-zero record withholds all 305 proposed scan-5 mapmaking samples. The
four trigger pixels themselves have exactly zero direct N5/A5-map difference,
so the trigger and response locations are spatially decoupled. The observed
arcs are the response to the whole scan-local action, not removal of just four
pixels. Exact leverage-versus-processed-signal contrast remains unresolved.
The active
[`EL-F9 result`](../validation/fruit_loop_point_123424_el_f9_map_leverage_audit_2026-09-03/EXECUTION_RESULT_R0.1.md)
recommends a separately reviewed diagnostic-only JINC accounting replay that
persists total and target-UID `S`, `G`, and `V` components and exact support,
then reconstructs the already retained map-only result. No instrumentation,
replay, safeguard, policy, recurrence, Gate D, Stage B, or Unity action is yet
authorized.

The scientific owner agreed with that next direction. The resulting proposed
[`SCI-FRUIT-EL-F10-TARGETED-JINC-ACCOUNTING-R0.1` packet](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F10_JINC_ACCOUNTING_OWNER_REVIEW_R0.1.md),
bound by its exact
[`bundle manifest`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F10_BUNDLE_MANIFEST_R0.1.md),
now states the missing accounting in frozen SCI-JINC notation (`N`, signed
`C`, and quadratic `Q`) and keeps signed normalization share distinct from
absolute coefficient mass, quadratic support, hit count, and unique-detector
count. It proposes one diagnostic-on replay from the exact EL-F6 no-record
iteration-4 checkpoint.
Bitwise reproduction of the existing science maps is a prerequisite to any
interpretation; a pre-registered forward-error bound governs reconstruction
of the existing EL-F8 map-only result. The diagnostic sidecars are explicitly
not calibrated sky products, checkpoint state, a JINC-conformity assessment,
or a production interface.

The scientific owner approved that exact packet on `2026-09-04`. EL-F10 then
added the disabled-by-default diagnostic, passed the full local build and test
gates, froze its executable and 19 inputs, and completed the one authorized
local copied-checkpoint replay in 33.29 seconds with no error or critical log
records. All nine ordinary science planes and all three formal-coefficient
planes match EL-F6 N5 bitwise. The registered analysis nevertheless stopped at
checkpoint compatibility before opening the target accounting values: in
addition to the allowed `creator_version` change, the new checkpoint
explicitly serializes
`map_pixel_outlier_detector_exclusion_application: pre_cleaning`, while the
older EL-F6 checkpoint omits that historically implicit default. The
checkpoint structures and all other values match. The active
[`EL-F10 result`](../validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/EXECUTION_RESULT_R0.1.md)
records a compatibility failure, not a scientific accounting result. No
second replay is warranted. The next decision is a separately approved,
no-replay repair that freezes the retained output hashes and permits only this
already established absence-to-`pre_cleaning` normalization; every other
neutrality, closure, bound, scope, and claim gate remains unchanged.
That proposed repair is now presented as
[`SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F10_R1_COMPATIBILITY_NORMALIZATION_OWNER_REVIEW_R0.1.md),
bound by its exact
[`EL_F10_R1_BUNDLE_MANIFEST_R0.1.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F10_R1_BUNDLE_MANIFEST_R0.1.md).
The owner approved R1 on `2026-09-04`. Its 21 exact hashes and repaired
checkpoint gate passed, including the required observed difference set and
one-key normalized-policy equality. The frozen analyzer then failed on the
receipt's `schema_identity` before reading any `N`, `C`, or `Q` plane or
opening the target ledger: NetCDF returned a native Python string and the
helper incorrectly called `.item()` on it. No result product was written. The
active
[`R1 result`](../validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/EXECUTION_RESULT_R0.2.md)
therefore records an analysis-reader stop, not an accounting result. The
proposed
[`SCI-FRUIT-EL-F10-R2-NETCDF-SCALAR-READER-REPAIR-R0.1`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F10_R2_NETCDF_SCALAR_READER_OWNER_REVIEW_R0.1.md),
bound by its exact
[`EL_F10_R2_BUNDLE_MANIFEST_R0.1.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F10_R2_BUNDLE_MANIFEST_R0.1.md),
was approved on `2026-09-04`. The scalar repair and two subsequently exposed
routine diagnostic-path defects were resolved under the owner's standing
narrow-defect direction: FITS receipt comparison now applies the documented
output column reversal, and target-ledger collection now excludes the
noise-only JINC pass.
The latter required one isolated local replacement replay; its accumulator
receipt and checkpoint are byte-for-byte identical to the retained defective-
ledger replay, while the corrected ledger contains exactly 305 unique target
samples (271 admitted and 34 already final-flagged). Full local build, test,
configuration, and formatting gates passed.

The final
[`EL-F10 result`](../validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/EXECUTION_RESULT_R0.6.md)
passes all registered neutrality, compatibility, exact-closure, ledger,
forward-error, and support gates. Removing the target `N_t`, `C_t`, and `Q_t`
reconstructs the retained EL-F8 map-only counterfactual with maximum signal
difference `6.82121e-13 mJy/beam` and no support changes. The exact identity
shows that the localized response is signed local leverage times processed-
signal contrast. At the worst arc pixel, a 4.87% signed share multiplies an
approximately `-2032 mJy/beam` contrast to give the observed
`-98.97 mJy/beam` response; 164 unique detectors contribute at that pixel.
UID 4460's scalar detector weight is elevated but remains well below the
pipeline's logged upper limit, so ordinary weighting does not measure this
interaction. The four original trigger pixels and the off-source injected
aperture have zero direct target contribution.

This is exact but bounded evidence for one observation, UID, scan, and
iteration. It neither judges the detector nor establishes a generic mechanism
or selects a safeguard. The next significant owner decision is whether to
authorize a prospective EL-F11 influence study comparing the existing repeat-
count action with response-aware hard and bounded soft or map-local candidates,
followed by independent-pointing replication before any policy selection.

The scientific owner agreed with that recommendation on `2026-09-04`. The
resulting EL-F11 proposal deliberately resolves one prerequisite before any
intervention comparison: one short replay would measure UID 4460 scan 5's
exact whole-map JINC deletion response in iteration 4, using only state
available before its iteration-5 hard exclusion, and compare it descriptively
with the retained EL-F10 iteration-5 response. The target is retrospectively
known, so this is explicitly an oracle-targeted persistence feasibility test,
not yet a deployable candidate-selection method. The exact
[`owner-review proposal`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F11_PROSPECTIVE_INFLUENCE_OWNER_REVIEW_R0.1.md),
[`design`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F11_PROSPECTIVE_INFLUENCE_PERSISTENCE_DESIGN_R0.1.md),
and
[`bundle manifest`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F11_BUNDLE_MANIFEST_R0.1.md)
are prepared for owner review. No setup, replay, safeguard, threshold,
recurrence, Gate D, Stage B, production, or Unity action is authorized.

The scientific owner approved
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1` exactly against
`EL_F11_BUNDLE_MANIFEST_R0.1.md` on `2026-09-04`. The complete copied
iteration-3 restart directory matches its preserved EL-F5 source, all 23
registered files pass exact identity checks, all 110 fruit-loop Python tests
and the complete configuration preflight pass, and the single local replay
and analysis method are frozen in
[`REGISTRATION_R0.1.yaml`](../validation/fruit_loop_point_123424_el_f11_prospective_influence_2026-09-04/REGISTRATION_R0.1.yaml).
No new accounting values have been opened. The one authorized local replay is
the next action; its outputs must be hash-bound before analysis.
The first launch then stopped in the restart guard before any FRUIT iteration
because the override repeated fresh-run `start_iteration: 1` instead of the
checkpoint's required next iteration `4`. The failed log and empty lock are
preserved. The method-preserving correction is recorded under the owner's
standing routine-defect direction and frozen in `REGISTRATION_R0.2.yaml`; the
single scientific replay remains unused and is still the next action.
The corrected replay then completed absolute iteration 4 in 31.59 seconds at
871,579,648 bytes maximum resident set size with no error or critical log
records. Its exact outputs and the retained EL-F10 comparison inputs are bound
in `REGISTRATION_R0.3.yaml` before any new accounting value was opened. The
single scientific replay is consumed; compatibility-first analysis is the next
action and cannot relax a failed gate or authorize another run.
That analysis reproduced all nine ordinary science planes and all three formal
planes bitwise, then stopped before opening the JINC receipt because the
complete learning CSVs were not byte-identical. Read-only diagnosis established
that the uninterrupted historical file is cumulative across iterations 0--4,
whereas the checkpoint replay file contains only iteration 4. Their headers
and 439 ordered iteration-4 rows match exactly in every raw field. Map-
diagnostic and checkpoint compatibility also pass their registered semantic
comparisons. The active result is therefore a compatibility failure, not a
prospective-influence result, and no replay is warranted. The proposed
[`SCI-FRUIT-EL-F11-R2-LEARNING-LEDGER-SCOPE-NORMALIZATION-R0.1`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F11_R2_LEARNING_LEDGER_SCOPE_NORMALIZATION_OWNER_REVIEW_R0.1.md)
would preserve the failed whole-file check and replace it, prospectively and
without replay, with exact ordered equality of the registered iteration-4
rows. It is bound for review by the exact
[`EL_F11_R2_BUNDLE_MANIFEST_R0.1.md`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F11_R2_BUNDLE_MANIFEST_R0.1.md).
Every other EL-F11 gate and claim limit would remain unchanged.
The owner approved that exact R2 repair on `2026-09-04`. The repaired analyzer
and 39-file no-replay registration passed all 115 fruit-loop tests, Ruff, and
the complete configuration preflight before accounting values were opened.
All compatibility, accumulator, ledger, support, grid, and deletion-identity
gates then passed. Iterations 4 and 5 have the same 4,229 conditioned target
pixels. Their UID 4460 scan-5 deletion responses have normalized inner product
`0.9996376`, fitted scale `1.0040724`, and a `2.69205%` scaled residual; the
top 1% absolute-response pixel sets are identical. The registered Neptune and
annular regions are similarly persistent, while the injected-source aperture
has no direct conditioned target occurrence in either iteration. The complete
[`EL-F11 result`](../validation/fruit_loop_point_123424_el_f11_prospective_influence_2026-09-04/EXECUTION_RESULT_R0.5.md)
therefore establishes strong oracle-targeted temporal persistence in this one
case: the harmful iteration-5 map consequence was already measurable from
iteration-4 JINC state before the carried hard action took effect. It does not
define a causal candidate selector, compare an intervention, establish
generality, or qualify a safeguard. This closes the diagnosis-only UID 4460
branch. The next significant decision is whether to authorize one bounded
response-aware intervention screen; another explanatory study of the same
event is not recommended.

On `2026-09-04`, repository recovery at `d39d4685b` confirmed that final
EL-F11 result and both preserved review-archive identities. The requested
next milestone is now a concrete
[`EL-F12 owner-review proposal`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F12_RESPONSE_AWARE_INTERVENTION_OWNER_REVIEW_R0.1.md),
with a
[`bounded design`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F12_RESPONSE_AWARE_INTERVENTION_DESIGN_R0.1.md)
and exact
[`proposal manifest`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F12_BUNDLE_MANIFEST_R0.1.md).
It proposes a causal census of all new map-dominance hard-exclusion records,
individual and joint JINC deletion-response screening, and two fixed
alternatives: hold the selected exclusion or retain half its map coefficient
through the fixed iteration-6 horizon. All arms start fresh with the historical
`alpha=1` recurrence; the rejected alpha-1.25 diagnosis checkpoints are not
reused as control state. The proposal retains all-array source, morphology,
leakage, useful-exclusion, support, convergence and performance protections.
Eight primary trajectories and at most four conditional restart checks are
proposed; no run has been staged or performed. A favorable result could only
advance to separately authorized independent-pointing replication before any
policy recommendation. The next significant decision is exact owner approval
or revision of this screen. No intervention implementation, replay, new
scientific result, Gate-D launch, qualification, Stage B, production or Unity
activity is authorized by proposal preparation.

The owner approved Choice A against the exact EL-F12 manifest at proposal
commit `d78d4d94a` on `2026-09-04`. The
[`authorization record`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F12_AUTHORIZATION_2026-09-04.md)
permits the bounded prototype, local gates, registration and execution matrix.
All 26 proposal-member identities and 27 external science/configuration/text
fit-report inputs reverify. No EL-F12 trajectory has begun. Replication, policy
recommendation, full Gate D, qualification, Stage B, production and Unity
remain outside this authorization.

Source inspection after approval exposed a missing method rule: the ordinary
hard-exclusion cap is evaluated both before RTC and before PTC, and these
checks can disagree as the detector population changes. Eight focused tests
of the unchanged handler pass, including synthetic rejection-before-RTC and
acceptance-before-PTC evidence. The
[`two-stage cap amendment`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F12_TWO_STAGE_CAP_OWNER_REVIEW_R0.1.md)
proposes that Half requires permission for that key at either applicable
stage in the current iteration. Its
[`exact manifest`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F12_TWO_STAGE_CAP_MANIFEST_R0.1.md)
binds the rule and
[`test/preflight evidence`](../validation/fruit_loop_el_f12_preimplementation_2026-09-04/CAP_BOUNDARY_TEST_RESULT_R0.1.md).
The next significant owner decision is `SCI-FRUIT-EL-F12-CAP-001-R0.1`;
Choice A remains approved, but intervention implementation and replay pause
at this method decision. No intervention or full pre-execution gate has been
implemented or completed. Original proposal bytes, both review archives and
all retained reduction products are preserved.

On `2026-09-05`, the owner approved CAP-001 against the exact manifest at
`7327e642e`; the
[`CAP-001 authorization`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F12_CAP_001_AUTHORIZATION_2026-09-05.md)
resolves the two-stage cap decision. Bounded EL-F12 prototype implementation
and pre-execution gates resume under Choice A and CAP-001. No replay has begun;
all original science and resource limits remain in force.
The default-disabled prototype and local pre-execution gates now pass:
652 enabled CTests, 270 baseline/FRUIT Python tests, CLI build, complete
configuration preflight and focused lint. The
[`pre-execution evidence`](../validation/fruit_loop_el_f12_response_intervention_2026-09-05/PRE_EXECUTION_GATES_R0.1.md)
records routine fixes, exact input recovery and active-assignment restart
coverage. No real-data trajectory has begun. Prospective executable/analyzer
registration and the already approved bounded matrix are the next actions.
The exact
[`prospective registration`](../validation/fruit_loop_el_f12_response_intervention_2026-09-05/REGISTRATION_R0.1.md)
now freezes native source 10478fa0e, controller 80e608eb8, the rebuilt binary,
6,273 artifact identities and all primary/conditional commands before any
observation run. Local execution of H0 uninjected is the next authorized step.
Both H0 trajectories have now passed; uninjected H0 reproduces all retained
EL-F2 alpha-one map planes bitwise through iteration 6, and the pair matches
at iteration zero. H uninjected completed all seven iterations, then exposed
an analyzer omission for the added audit scalar's length-one NetCDF dimension.
The [routine comparison repair](../validation/fruit_loop_el_f12_response_intervention_2026-09-05/SCALAR_DIMENSION_REPAIR_R0.2.md)
retains the original failure and passes all science-plane, checkpoint/D19,
ordered learning-row and map-diagnostic comparisons without replay or binary
change. All 272 Python tests pass. Three primary trajectories/21 passes are
consumed; no alternative or diagnostic replacement has run. H injected is
next after successor analyzer registration.
Under the
[`r0.2 analyzer registration`](../validation/fruit_loop_el_f12_response_intervention_2026-09-05/REGISTRATION_R0.2.md),
H injected also completes all seven iterations and passes every neutrality
check. Both H0/H pairs and both iteration-zero pair checks pass. Four primary
trajectories/28 passes are complete, with no diagnostic replacement. The
[`control-gate progress`](../validation/fruit_loop_el_f12_response_intervention_2026-09-05/EXECUTION_PROGRESS_R0.2.json)
binds those receipts. The approved alternatives may now run in their fixed
order; Half uninjected is active. Neither benefit nor policy is established.

EL-F12 is now complete under the unchanged r0.2 registration. All eight primary
trajectories/56 passes and both paired analyses are retained. The
[`scientific interpretation`](../validation/fruit_loop_el_f12_response_intervention_2026-09-05/SCIENTIFIC_INTERPRETATION_R0.1.md)
and [`result manifest`](../validation/fruit_loop_el_f12_response_intervention_2026-09-05/RESULT_MANIFEST_R0.1.md)
record **Half and Hold as `not_promising`**. Both fail a2000 support and
uninjected-Neptune stability protections, and neither improves the fixed
a1400 leakage endpoints. a1100/a1400 remain bitwise equal to H. H's a1400
Neptune response is already below 0.1 mJy/beam at both endpoints, making the
declared 0.1 mJy/beam improvement unattainable on this pair; this limitation is
disclosed without changing the gate. The independent protection failures
also prevent advancement. Each candidate has all 18 required measurement
rows, with 23 of 315 protection checks failing. No conditional restart or
diagnostic replacement ran, and no independent pointing is authorized.
All 6,274 registered identities reverify; the binary, historical-control
boundaries, existing reductions and both review archives remain preserved.
The next significant owner decision is whether to commission a separate
development generation addressing attainable historical-control endpoints
and beneficial action. Neither this negative screen nor the earlier oracle
diagnoses establish a policy, generic result, qualified method, full Gate D,
Stage B, production change or Unity authorization.

The owner-approved EL-F13 scan-agreement feasibility experiment is complete
on `2026-09-05`. The [authorization](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F13_AUTHORIZATION_2026-09-05.md)
binds proposal commit `9b75104dd6df1708fe28f7de2178fcb2f85cf3c5`; the immutable
proposal and all three implementation registrations remain retained. The
[result](../validation/fruit_loop_el_f13_scan_agreement_2026-09-05/SCIENTIFIC_INTERPRETATION_R0.1.md)
records six H/uninjected boundary evaluations, exact historical reconstruction
for all arrays, and two eligible a2000 keys. Boundary 1's 6,489-pixel fixed
domain loses 7/13 required pixels under half/full retention, so both probes
are unavailable and the primary negative challenge is unassessed. Boundary 2's
5,541-pixel domain supports half retention, which worsens RMS agreement with
both references (88.4629→91.9114 and 97.1460→102.1210 mJy/beam); full retention
loses one required pixel and is unavailable. There is no positive empirical
benefit evidence or policy recommendation.

All six predictions were frozen before the reporter joined the already-exposed
EL-F12 outcomes. Thirty-two focused tests, all 304 offline FRUIT/baseline tests,
and full required configuration preflight passed. Routine macOS startup and
native FITS-orientation repairs preserved failed attempts and changed no method,
gate, input, resource limit or scientific interpretation. Maximum worker RSS
was 1.442 GiB; six worker evaluations took 127.011 seconds in total. The
[manifest](../validation/fruit_loop_el_f13_scan_agreement_2026-09-05/RESULT_MANIFEST_R0.1.md)
retains exact clocks, complete maps, support accounting, intervals and inputs.
All 36 bound input files and both review archives remain unchanged. Zero
Citlali runs, interventions, new observations, Unity work or qualification.
The next significant owner decision is whether to commission a separate
benefit-signal design addressing shared-reference contamination and attainable
historical-control endpoints. Any intervention still requires concrete joint
action/duration/cap rules, all science and resource protections, exact paired
inputs, and independent-pointing replication before policy. EL-F12 stays
negative, the UID 4460 diagnosis remains closed, and Stage B remains separate.

On `2026-09-06`, the requested next benefit-signal design is ready as the
[`EL-F14 owner-review proposal`](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F14_BENEFIT_SIGNAL_OWNER_REVIEW_R0.1.md),
with a fixed [synthetic design](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F14_SKY_TIME_SEPARATION_DESIGN_R0.1.md)
and [exact manifest](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/EL_F14_BUNDLE_MANIFEST_R0.1.md).
It proposes using simultaneous detector differences to constrain sky contrasts
under an explicit shared additive time model, then bounding the benefit of
half/full retention against deletion over every compatible sky. Thirteen fixed
scientific fixtures and four integrity evaluations would test positive,
harmful, ambiguous and inconsistent cases. Identical-data/different-truth twins
must expose the inability to distinguish arbitrary sky-shaped contamination.
The toy contrast margin is not an astronomical endpoint or an EL-F12 gate
change. A pass could establish only conditional synthetic identifiability;
real-data timing, response, error bounds and transfer remain unestablished.
No EL-F14 helper, evaluation, real-data scoring or intervention has been
implemented or run. The next significant owner decision is exact approval or
revision of `SCI-FRUIT-EL-F14-SKY-TIME-FEASIBILITY-R0.1`, limited to this
synthetic method and population. Historical control, all later intervention
protections and independent-pointing replication before policy remain
mandatory. Existing reduction products and both review archives are preserved;
EL-F12/EL-F13 conclusions, the closed UID 4460 diagnosis, full Gate D,
qualification, Stage B, production and Unity boundaries are unchanged.

After the owner questioned EL-F14's overlap with PTC and requested a bounded
return to scientific basics on `2026-09-06`, the
[evidence review](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/BOUNDED_EVIDENCE_REVIEW_R0.1.md)
is complete. Fourteen retained H/uninjected PTC/checkpoint files were inspected
only for schema and identity; they do not establish a replay-ready fixed PTC
operator. Even a conditional source-response measurement would leave the
real-field sky/nuisance benefit target unresolved. The manager therefore
recommends no further retention experiment from this review and withdraws the
recommendation to execute EL-F14 r0.1. Its proposal bytes remain preserved,
unapproved and unrun. This is a planning conclusion, not proof that no better
method exists; no new scientific score or transfer was measured.

The concrete next owner decision is the
[scientific-core scope proposal](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_CORE_CONTRACT_SCOPE_OWNER_REVIEW_R0.1.md),
bound by its [manifest](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/BOUNDED_REVIEW_MANIFEST_R0.1.md).
It proposes a conditional scientific core with no admitted numerical profile,
and an explicit narrow exception to the qualification-before-authorship
sequence. The current accepted sequence remains in force pending that exact
decision. No Stage B is dispatched; author-input sanitization and separate
launch approval remain mandatory. Historical control, empirical outcomes,
independent-pointing replication before policy, full Gate D, qualification,
production and Unity boundaries remain unchanged. All retained products,
EL-F14 bytes and both known review archives are preserved.

On `2026-09-06` Grant Wilson approved
`SCI-FRUIT-SCIENTIFIC-CORE-SCOPE-R0.1` and its narrow sequence exception,
recorded in the separate
[scope approval](scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_SCIENTIFIC_CORE_SCOPE_APPROVAL_2026-09-06.md).
A conditional scientific core may enter separately approved authorship before
a qualified numerical replacement, while admitting no numerical profile.
The numerical development, qualification and independent-pointing evidence
requirements remain intact. The original approved proposal and bounded-review
bytes are preserved; their earlier proposal-status prose is historical.

The [successor Stage A packet](scientific_contracts/packages/SCI-FRUIT/v0.1/scientific_core/README.md)
is prepared for exact owner review, bound by its
[manifest](scientific_contracts/packages/SCI-FRUIT/v0.1/scientific_core/PACKET_MANIFEST_R0.1.md).
It supplies a sanitized Scope Brief, resolved core/deferred numerical decision
map, prior-authority reuse record, restricted scientific boundary extracts
and a bounded author task. The six author inputs and their manifest contain
no empirical results or implementation-informed dossier. The next significant
owner decision is exact packet approval plus an explicit Stage B launch
instruction; this preparation does not dispatch an author.

The current benefit-selector exploration is closed without a qualifying
replacement. EL-F14 remains unapproved and unrun. Historical Citlali remains
the mandatory empirical control and a compatibility candidate, not an
automatically selected scientific method. No new experiment, reduction,
replay, implementation, numerical admission, full Gate D, qualification,
production change or Unity activity occurs in this step. All retained
products, empirical conclusions and both known review archives are preserved.


On `2026-09-06` the owner requested a review of the Stage A output
instructions. The [manager instruction review](scientific_contracts/packages/SCI-FRUIT/v0.1/scientific_core/r0.2/STAGE_A_INSTRUCTION_REVIEW.md)
found the core r0.1 packet insufficient: its scientific input/output boundary,
individual decision dispositions, explicit successor dossier and current
navigation were incomplete. Hash/link checks had not established scientific
readiness. The manager withdraws the r0.1 approval/launch request while
preserving its exact bytes and the already approved core scope.

The [r0.2 successor](scientific_contracts/packages/SCI-FRUIT/v0.1/scientific_core/r0.2/README.md)
repairs the complete eleven-section Scope Brief, maps all eighteen existing
questions separately, designates the dated internal dossier and recovery
chain, and updates current package/index navigation. It is a corrected Stage A
draft for owner scientific review; exact packet approval and explicit launch
remain outstanding. The explicit xhigh author-effort direction is retained.
No scientific derivation, numerical choice, experiment, replay, qualification,
production change or Unity action occurs. All earlier empirical conclusions,
reduction products, r0.1 input bytes and both review archives are preserved.

On `2026-09-08`, Grant Wilson accepted and conditionally froze the exact
`SCI-FRUIT-NORMATIVE-CORE v0.1/r0.4`, recorded in the separate
[conditional freeze](scientific_contracts/packages/SCI-FRUIT/v0.1/scientific_core/SCI_FRUIT_STAGE_B_R0.4_OWNER_CONDITIONAL_FREEZE_2026-09-08.md).
It binds the seven canonical sources through inventory SHA-256
`3dc912d9f6a58d9f92561ad10b9442485c5b47232e5327389d2fb3b89c296e92`.
Stage A and the exact Stage B source/PDF/archive bytes remain unchanged.
The earlier Stage A draft/not-launched statements above are historical.
Program-order authorship authority, conditional composition authority and
numerical-method authority remain separate. The numerical family remains
`unavailable_pending_separate_owner_approval`; the freeze creates no Registry,
activation, implementation conformity or numerical qualification.

The owner then [approved historical-control recovery and method-scope preparation](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/historical_control/r0.1/OWNER_PREPARATION_AUTHORIZATION_2026-09-08.md).
The [r0.1 owner proposal](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/historical_control/r0.1/README.md)
now contains a completed prior-work record, internal historical-source dossier,
separate eleven-section sanitized Scope Brief, all twenty-one required method
classes, exact proposed references and five grouped owner decisions. Its
proposed first use is compact-source pointing with one observation, independent
array maps and raw normalized JINC feedback. That route is not yet selected.
The recovery keeps original-observation rerun and complete-map replacement as
historical facts, and separates source identity from executable reproducibility.
It exposes missing residual-PTC/rejoined-signal/JINC permissions, numerical
coefficient/parameter bindings and any selector-required map companions; frozen
JINC's base bundle does not supply them by implication.

The next significant owner decision is the exact proposed scope and bounded
paper-only closure of its method/upstream questions. This preparation does not
approve numerical operators, a method instance, a population or an experiment.
No new author, implementation assessment, replay, diagnostic, qualification,
production change or Unity operation was launched. Historical control remains
mandatory, the UID 4460 diagnosis stays closed, EL-F14 remains unapproved/unrun,
and independent-pointing replication is still required before policy. All
existing reduction products, frozen deliveries and both known untracked review
archives are preserved. Document verification establishes packet integrity and
scope coverage, not scientific approval or numerical availability.

Later on `2026-09-08`, the owner supplied a sidebar recommending ordinary
(naive) MAP as the first new FRUIT method setting. The manager agrees and
supersedes the unapproved JINC-first recommendation with the
[r0.2 ordinary-MAP-first proposal](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/r0.2/README.md).
Historical-control reconstruction and first-method selection are explicitly
separate: the historical JINC control and exact r0.1 packet remain unchanged.
The proposed successor uses one observation, separate arrays and raw ordinary
MAP, with the feedback projection separately specified. A matched
containing-pixel example illustrates conditional restoration of the applied
model on admitted supported rows; it selects no projection and establishes no
unit sky response, convergence, independent noise or benefit.

The successor retains all twenty-one method classes and the full sanitized
Scope Brief, replaces direct JINC author references with exact frozen MAP
science, and exposes the MAP-facing PTC coefficient, numerical support-policy/
`coverage_cut`, rejoined-signal handoff and companion dependencies. JINC
specialization is deferred. D001–D005 remain pending exact owner disposition;
the sidebar is not recorded as numerical-method or experiment approval.
No frozen core, control, reduction, code/configuration, empirical outcome or
opaque archive changed. No new author, implementation assessment, experiment,
replay, qualification or Unity activity was launched. The next owner decision
is the revised exact scope and paper-only method-definition direction.

Subsequently on `2026-09-08`, Grant Wilson
[approved D001–D005](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.1/OWNER_SCOPE_APPROVAL_2026-09-08.md)
as ordinary-MAP scope and bounded paper-preparation directions, with explicit
two-sided PTC boundary and source-control corrections. Those scope decisions
are closed; the preceding pending-scope statements and exact r0.2 packet are
historical. No further standalone scope review is requested.

The [paper method and boundary proposal r0.1](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.1/README.md)
now supplies concrete rules for all twenty-one method classes, U01–U08
dispositions, a historical comparison and precise method/upstream questions.
The proposed candidate uses an unseeded k=0 bootstrap, previous complete-map
replacement, support-only model selection of either sign, matched containing-
pixel removal/rejoin, PTC state fitted on bootstrap CAL then held fixed, and
a proposed PTC-owned uniform analysis/gridding family retained with current
compatibility/QC. These are new proposals, not scope-approved numerical choices.
L includes bootstrap; successful completion selects final required k=L-1 for
all required arrays. Any required array failure fails the observation request;
earlier immutable products remain in their actual states and do not rescue it.

The two-sided review identifies a specific unavailable numerical route:
`unavailable_under_current_frozen_parent_permissions`. Frozen PTC's admitted
CAL parent does not automatically permit this FRUIT residual; frozen MAP's
PTC-product input does not automatically permit the rejoined FRUIT child.
BC-IN and BC-OUT propose bounded controlled scientific permissions, including
reference/gauge compatibility; no successor is adopted. Coefficient-family/QC,
exact PTC plan and numerical MAP support-policy bindings remain required.
MAP stays exactly v0.1/r0.7.1. The three program-reference flags now agree in
the current source controls with process-only authority. Manager evidence stays
outside any future independently authored packet.

The next substantive owner review is Q01/Q02: the concrete method rules and
named boundary/family actions. Exact numerical-plan/support and later execution
bindings are explicit remaining prerequisites. This delivery includes no
experiment design, numerical method approval, author dispatch, implementation,
replay, qualification or Unity work. Generic-core and upstream frozen bytes,
historical JINC control, existing reduction products and both opaque review
archives remain preserved.

Following review of an owner-supplied PCA discussion on `2026-09-08`, Grant
Wilson accepted the manager's recommendation for a targeted residual-relearning
successor. The [direction record](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.2/RELEARNING_DIRECTION_AND_REVIEW.md)
binds that assent as
`SCI-FRUIT-OD-RESIDUAL-RELEARNING-DIRECTION-2026-09-08`.
The learning schedule is resolved for the intended paper method: learn current
centering and correlated subspace from each newly constructed model-subtracted
residual under the same declared PTC recipe, then hold that resolved state
fixed for its own application. This addresses astronomical influence on fitting;
it does not require eigenmodes to change or establish improved recovery.

The [method-definition r0.2 successor](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.2/README.md)
has proposed identity `FRUIT-FEEDBACK-METHOD/ordinary-map-residual-relearning@r0.2`.
It replaces the fixed-bootstrap recommendation without changing the exact
method-definition r0.1 packet/archive or historical control. BC-IN now specifies
residual learning, resolution and application admission. Per-pass learning
influence, centering/subspace state, response conditioning, uncertainty and
persistence records are updated. Fixed-state response remains conditional;
full-procedure response includes the prescribed relearning and its transitions.
Ordinary configured-rank/centering/scaling and zero-refinement rules are reused;
adaptive rank, changing populations or gridding-weight estimation are not added.

Scope and learning direction are not new open questions. The other candidate
rules remain proposals, especially support-only selection and uniform gridding
coefficients. Exact PTC/MAP amendments remain unapproved and unadopted;
`unavailable_under_current_frozen_parent_permissions` describes those contract
limits, not an inability to calculate PCA or a need to know eigenvectors in
advance. Existing approved recipe decisions are reused; applicable effective
settings and missing request-specific values are bound separately before any
execution request. The 24 proposed-reference entries and corrected process-only
flags are unchanged; the additional discussion stays manager-only.

No generic core, frozen upstream contract, historical product or prior archive
changed. No author dispatch, experiment design/execution, implementation,
qualification, production or Unity action occurred. The next review is the
remaining method combination and exact boundary work, with no repeat scope or
learning-schedule approval requested.

On `2026-09-09`, Grant Wilson supplied an earlier boundary-closure directive
and a later mode-aware inference directive, explicitly giving the later one
precedence. The [r0.3 direction record](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.3/OWNER_DIRECTION_AND_CROSSWALK.md)
binds `SCI-FRUIT-OD-MODE-AWARE-AND-BOUNDARY-2026-09-09`. Ordinary MAP first,
original-parent reconstruction and current-residual PTC relearning remain
resolved directions. The leading feedback proposal now re-infers a replacement
model from the reconstructed total with explicit mode policies. It separates
admission, retained calibrated flux and application; previously admitted
structure can change or be revoked. Fixed MAP science domain/contributor
populations are distinct from variable admitted-model support. The all-support,
both-sign model remains a comparison baseline; historical PCA/S/N and its actual
optional flux cut remain a practical baseline and fallback, with JINC preserved.

The [r0.3 review packet](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.3/README.md)
contains the scientific definition, directive crosswalk, prospective engineering
requirements and minimal experiment proposal. POINT, OOF, BEAM and distinct
SCIENCE inference regimes have explicit required interfaces and limits.
Unity application is a hypothesis; flux floors, non-unity gain, coherence,
detector borrowing and alternative PTC estimators remain experimental.
Convergence, perturbation stability and measured recovery are separate claims.

The actual bounded amendment annex addresses PTC's residual parent and one-fit
rules, current fixed-state application, centering/reference loss, total removed
signal and REQ-099 exclusions. It also proposes a scoped rejoined-descendant
MAP boundary/profile successor and constant gridding/current-QC/retained-use
records. All are proposals, not adopted authority. Exact MAP arithmetic stays
v0.1/r0.7.1. Effective-plan recovery reuses decided PTC and MAP rules and returns
only missing/request-specific bindings and exact mode/input conflicts.

The newer directive authorizes focused experiment proposals, superseding the
earlier no-design boundary only for that paper work. E00 establishes exact
historical and naive references; E01 proposes one POINT selector comparison
with PTC recipe held fixed but relearned separately per arm. Independent-pointing
replication is required before recommendation. A bounded OOF follow-on is
conditional on a specific justified question. No experiment, injection, replay,
implementation, Unity work, qualification, production, author dispatch or
amendment adoption occurred. The next substantive review remains Q01/Q02;
required numerical policy, effective-plan/support, evidence and execution
bindings remain explicit. Frozen core/upstream sources, prior packets/products
and both opaque review archives are preserved. The 43 copied inputs include
five new owner/manager sources; the 24 proposed references remain unchanged.

After reviewing r0.3, the owner supplied targeted feedback and instructed that
it be addressed and the definition locked. The
[r0.4 scientific-definition freeze](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/SCIENTIFIC_OWNER_FREEZE_R0.4.md)
records `SCI-FRUIT-OD-ORDINARY-MAP-DEFINITION-FREEZE-2026-09-09`.
Q01's scientific definition is accepted/frozen after three clarifications:
POINT/E01 is a recommended first bounded screen, subject to separate owner
selection; fixed D_a/contributor/support populations are controlled-reference
restrictions, with general FRUIT support changes requiring explicit attribution
and comparison accounting; and binary admission times reconstructed-total flux
is the leading reference estimator, not a universal inference restriction.
The derived conformance requirements reflect those same limits. No broad
rewrite or reference-algorithm change occurred.

Q02 remains open. The B01–B12 annex, including PTC/MAP permissions, uniform
gridding, QC identities and correction embedding, is byte-identical to r0.3 and
not adopted by the scientific-definition freeze. Numerical policy slots under
Q01, effective-plan/support bindings Q03/Q04, and evidence/execution Q05/Q06
remain unavailable until separately bound and authorized. No experimental
ordering, case, run, author dispatch, implementation or qualification is
approved by this lock. The next substantive review is Q02 separately; the
scientific definition is not reopened merely because its numerical bindings
remain incomplete.

The [r0.4 packet and frozen-definition archive](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/README.md)
preserve the r0.3 packet/archive, historical controls/products and all upstream
and generic-core freezes. All 44 source copies match their exact sources;
the inherited 43 rows and the 24 proposed references are unchanged. Both known
untracked review archives remain present and opaque. No numerical work, Unity
action, upstream amendment adoption or production change occurred.

The owner then authorized Q02 review preparation. The
[three-decision Q02 sheet](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.1/README.md)
recommends A: the scoped residual PTC pass; B: compatible containing-pixel
correction/removal/rejoin; and C: explicit rejoined MAP input with uniform
occurrence gridding for the controlled reference and fresh current QC. All
three decisions remain pending. Uniform coefficients are a deliberate
reference choice, not a consequence of naive mapping or a noise-optimal claim.

The review identifies one necessary proposed clarification S1: frozen VAL
profile supersession rules require new immutable versions when the residual
domain or authoritative source changes. Preserve ordinary-route @1 records;
residual basis/loading/application/retention and requested conditional-response
uses must bind appropriate successor versions, and downstream QC/MAP must
reference the current residual-route decisions. The registered VAL source is
already bound by frozen MAP r0.7.1. S1 clarifies B05 and dependent B09–B11
references without editing the frozen annex. After scientific disposition,
exact controlled successor/source-binding records still require review and
adoption. Q01 remains frozen; mode registration, numerical/evidence/settings
and execution remain at their existing gates. No amendment adoption, author
dispatch, numerical work or implementation occurred.

On 2026-09-09 the owner **approved Q02-A and Q02-B** and requested tests of
uniform weighting because detector quality and noise vary. The
[r0.2 disposition](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.2/README.md)
records A's B01–B07 plus S1 and B's B08/M04 scientific substance as approved for
controlled amendment preparation. Exact successor sources/profile bindings
still require adoption; ordinary-route records and the frozen annex are
unchanged. Q02-C remains open pending weighting evidence, including its bundled
MAP handoff; no separate handoff approval or scientific rejection of uniform
weighting is inferred.

The [bounded weighting test proposal](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.2/WEIGHTING_TEST_PLAN.md)
compares uniform occurrences with one proposed inverse-bootstrap-scatter family
whose values are available before map/model action and fixed through the loop.
It proposes elementary noise/quality cases and paired maps from identical PTC
outputs first, with a separately decided feedback continuation if warranted.
The tests retain detector influence, noise, source recovery, morphology,
leakage, native/common support, stability/convergence and cost measurements.
Acceptability requires predeclared bounded losses and independent-pointing
replication for a pointing recommendation. Neither an estimated scatter nor Q
is precision evidence. Next bind and review the exact first screen's inputs,
experimental permissions, coefficient/QC identities, settings, evidence,
margins and execution scope under existing Q02–Q06. No candidate coefficient
family, mode, case, run, implementation or author dispatch is approved here.
A/B do not need another scientific-substance vote. All numerical routes remain
unavailable; Q01/core/upstream freezes, prior packets/archives and reduction
products are preserved. This revision performs document/source checks only.

The owner then approved the r0.2 weighting design and directed preparation to
proceed. The [r0.3 first-screen record](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.3/README.md)
preserves A/B approval and C's open disposition, and proposes exact T1 cases,
training, 1,024 trials per case, numerical loss limits and local resource bounds.
Those newly supplied numbers still require disposition in the completed
execution record; no comparison, coefficient adoption or FRUIT method was run.

The owner offered to create/download full PTC timestreams from Unity and asked
for observation numbers. The [short collection request](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.3/DATA_COLLECTION_REQUEST.md)
names 123424 (Neptune, 2024-11-27) for discovery alongside existing 152389, and
129081 (3c273, 2025-03-04) as reserved weighting replication. Both requested
integrations are 60 seconds; local telescope time spans are 62.730 and 63.151
seconds. No beammap, RTC dump, intervention, or multi-iteration FRUIT run is
requested. Collection is owner-run; Codex performed no Unity access or transfer.

Metadata-only recovery found full PTC signal/flags for 152389, with all 12
chunks, 3,628 samples and 5,518 detector slots, FRUIT/injection disabled,
network cleaning and rank five. Its exact occurrence-coordinate join and
scientific input admission remain unverified; its `mini` output is not a
contract-conformity result. Within the inspected fruit-development tree,
123424 has diagnostic products but no full PTC signal file. Old normal/stress
strata use map outcomes and are excluded from current detector-noise selection.
The proposed discovery pair's noise contrast remains unmeasured; the training
inventory must establish it before evaluation maps are opened. 129081 is
reserved only for new weighting outcomes, not untouched qualification.
Next receive exact input/provenance bundles and finish T2's input, support,
response/noise and interval bindings before execution review. Frozen authority,
prior packets, archived products and numerical-method gates are unchanged.

The owner subsequently supplied the local directory
`/Users/gwilson/work_toltec/local_data/beammaps/pointings/reduced/redu01`.
The [input intake](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/input_intake/r0.1/README.md)
finds both requested full PTC files, with signal, flags, detector-by-sample
coordinates and 12 declared chunks. Their combined size is 1,470,806,154 bytes.
However, the saved config enables FRUIT with max_iters=10 and both files report
FRUITLOOPS_ITER=9. These final-state candidates are not admitted as the requested
ordinary single-pass inputs. The legacy Boolean header is not relied upon.
Preserve the delivered products and collect fresh ordinary exports of the same
two observations with FRUIT disabled and max_iters=1, retaining config/run log
and identities. No new observation or experiment is selected by this correction.
129081's signal/noise/weighting outcomes remain unexamined and reserved.
The intake is metadata/identity only; no weighting statistic, map comparison,
implementation, numerical experiment or Unity action occurred. A/B approval,
C's open status and all execution/authority gates remain unchanged.

The owner then replaced the contents of the same `redu01` directory. The
[replacement intake](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/input_intake/r0.2/README.md)
clears the previous iteration-state mismatch: the YAML disables FRUIT and both
PTC files report iteration 0 with effective max_iters=1. The inactive YAML
max_iters=10 is consistent with source at the reported e0090e2d revision, which
forces one pass when disabled; no further export is requested for that setting.
Both full signal/flag/coordinate schemas and all 12 declared chunk records remain
present, with new payload hashes and combined size 1,470,806,136 bytes. Prior
intake bytes/hashes remain historical; the owner replaced the external payloads.
The replacement manifest also corrects the prior helper's malformed FRUIT-config
field without changing the old disposition. 129081's scientific values remain
reserved. No run log/provenance sidecar was found; exact scientific input,
support, response/noise and execution bindings remain pending. No numerical
comparison, science-array inspection, upstream adoption or Unity action occurred.

The owner next authorized completing local preflight and the execution proposal.
The [Q02 r0.4 review](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.4/README.md)
recovers a historical chunk-label defect in both discovery exports: the stored
3,628 rows have contiguous lengths 289, ten times 305, then 289, whereas the
reported intervals omit 160 rows. Exact source recurrence plus 152389 chunk
logs and 123424 raw indices support a separately bound recovery; products are
unchanged. The legacy detector-coordinate relation is recoverable from 152389's
embedded inputs and agrees with 123424's full export to 4.34e-19 rad. A local
123424 APT candidate matches all ten inspected identity/geometry/calibration
columns; original executable/raw-input identities and the new run log remain
missing. None of this establishes frozen PTC/AST/MAP conformity.

The unchanged first-half/64-sample training proposal fails input feasibility:
2,875 detector/chunk groups in 152389 and 2,001 in 123424 have evaluation
occurrences but too little source-excluded training at the flag/coordinate
level; respectively 507 and 227 have none. These counts precede final signal/
grid admission and are not noise outcomes. T2 remains stopped; no threshold,
guard, population or coefficient law was relaxed. The concrete next review
proposes standalone T1 execution with the exact r0.3 synthetic cases/margins,
and a separate paper revision of T2 toward observation-pooled detector training.
Both decisions are open. The revised T2 method, uncertainty, input permission
and run still require a completed decision. 129081 stays reserved; no signal
matrix, stored weights, noise estimate, weighted map, injection or replay was
read or computed during this preflight. No upstream adoption, Unity action or
push occurred. Q01, A/B, C's open status and all frozen/archived bytes remain.

The owner subsequently approved an initial goal of about 20% fractional
uncertainty in relative detector weights at one standard deviation. The
[dated Q02/Q05 direction](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/WEIGHT_PRECISION_DIRECTION_2026-09-09.md)
records that provisional design objective, with actual precision unmeasured.
Training windows and any sample-count requirement should be justified against
precision and noise stability; 64 is no longer the leading T2 design criterion.
The next preparation assesses shorter weight-estimation windows on the existing
discovery reductions before proposing observation-wide pooling. PCA chunks,
products and source/evaluation separation stay intact. Joint coefficient
normalization, temporal correlation, changing noise and bias need explicit
handling. The exact uncertainty method and execution scope remain to be bound.
This approval does not approve standalone T1, observation-wide pooling, map-loss
margins, a detector veto or a numerical run. Earlier preflight counts remain
historical evidence, and 129081 remains reserved. No new data analysis occurred.

The owner then directed preparation to proceed. The
[Q02 r0.5 diagnostic proposal](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.5/README.md)
now binds one training-only assessment on the exact 123424 and 152389 discovery
exports. It compares one-, two- and four-chunk training windows, with unchanged
PCA reductions and source/evaluation separation. The proposed first-order
uncertainty calculation propagates centered-scatter correlations and whole-array
normalization jointly over six fixed elapsed-time correlation spans. Temporal
scatter changes, sparse support and sensitivity are reported alongside precision;
finite-data coverage of the error estimate remains explicitly unavailable.
There is no certified 20% pass, detector veto or automatic window selection.
One pending decision covers the precise diagnostic method, limited legacy-input
permission and bounded local implementation/execution together. The proposal
does not authorize T1, T2 maps, 129081 access or feedback. No new signal analysis,
coefficient computation, numerical implementation or run occurred; all earlier
packets, frozen authority and reduction products are preserved.

The owner then approved the exact r0.5 diagnostic and directed execution in the
[2026-09-09 approval record](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/WEIGHT_PRECISION_APPROVAL_2026-09-09.md).
The [completed diagnostic report](../validation/fruit_q02_weight_precision_2026-09-09/SCIENTIFIC_REPORT.md)
was closed out on 2026-09-10. One local attempt completed in 43.676 seconds,
with 0.827 GiB peak memory, eleven deterministic checks and independent saved-
product verification passing. Both exact discovery input hashes are unchanged.
The one/two/four-chunk designs leave respectively 230/4/0 required weights
unavailable in 123424 and 529/12/2 in 152389. Four-chunk normalized precision
is available in all 123424 arrays and 152389 a2000; one zero-training group in
each of 152389 a1100/a1400 keeps those full-array results unavailable.

In the four available cases, occurrence-weighted median fractional uncertainty
is about 8–12% across the prescribed correlation spans. The 123424 tails are
material: groups above 20% at any span supply 23–27% of evaluation occurrences.
Four-chunk child-scatter max/min ratios have medians about 1.5–1.9; the screen
does not separate true noise changes from sampling or source effects. There is
no certified precision, selected window, qualified weighting policy or map-
benefit result. Close this precision diagnostic as evidence. The next owner
decision is a completed U/N mapping-screen design, including explicit handling
of missing training and map-level evidence/loss criteria. Do not choose only
successful arrays, fill unavailable weights or alter PCA chunks by inference.
No evaluation moment, weighted map, injection, reduction, FRUIT execution,
Unity action or push occurred. 129081 remains reserved; historical control,
Q01/A/B, generic/upstream freezes, earlier packets and both opaque archives
are preserved. Numerical FRUIT availability remains unchanged.

The owner then agreed to preparation of that mapping screen. The
[Q02 r0.6 proposal](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.6/README.md)
now supplies one exact review decision for U versus N4U: four-chunk inverse
centered training scatter, with explicit uniform relative weight for groups
whose training cannot define a raw weight. All admitted detector occurrences
and all six discovery array cases stay included; fallback influence is reported.
This proposed successor does not revise the completed diagnostic's missingness,
select a policy, or turn the 20% objective into a detector cutoff.

The proposed campaign combines the nine previously specified T1 synthetic
cases (1,024 trials each, with a 242-tail pass/failure/inconclusive prescription)
and 72 T2 native/response maps on the two exact existing discovery inputs.
It binds containing pixels, c=0.1 for this isolated test, native/common support,
background/source/morphology/leakage/influence and temporal/cost measurements.
T2 is explicitly exploratory: observed-change alerts and fixed-state arithmetic
response do not certify noise uncertainty, real-data noninferiority or physical
source recovery. The old unexecuted noise-contrast selection is replaced by
reporting both fixed observations. Those method/input/gate changes and execution
are pending the single SCI-FRUIT-Q02-MAPPING-SCREEN-R0.6 decision, not inferred
from preparation approval. No new array analysis, coefficient computation,
mapping implementation or run occurred. After the screen, make a bounded
continue-to-replication-proposal or close-exploration disposition. 129081 remains
reserved, Q02-C/MAP handoff remain open, and FRUIT/T3, numerical source adoption,
Unity, production and qualification remain separately gated. All prior/frozen
packets, diagnostic products, reduction inputs and opaque archives are preserved.

The owner approved that exact screen on 2026-09-10 in the
[Q02 r0.6 approval](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/MAPPING_SCREEN_APPROVAL_2026-09-10.md).
The [completed mapping-screen report](../validation/fruit_q02_mapping_screen_2026-09-10/SCIENTIFIC_REPORT.md)
records 36,864 synthetic maps and 72 discovery maps. Two startup failures occurred
before scientific access; routine process-monitoring/metadata repairs and both
old runners are retained. The sole scientific campaign, attempt 03, completed
in 23.217 seconds at 1.46 GiB peak aggregate memory. Fifteen deterministic
checks, four analytic U checks and independent product verification passed;
both discovery input hashes are unchanged.

Uniform passes the equal-noise and shared-component synthetic cases and fails
at least one registered criterion in the other seven. N4U lowers complete-
region real-map background RMS by 21.9–54.5%, while the three complete positive
native template amplitudes change by -32.2%, -20.3% and -11.9%. Both a1400
cases lose required outer support; signed morphology is largely unavailable.
The fixed-state signed mapping checks pass, but do not establish physical
recovery or real-data covariance. Both missing-training groups remain with the
approved uniform fallback and explicit influence accounting. No region, detector
population, coefficient family or threshold was changed after results.

Close the authorized screen as evidence. The report recommends returning to the
minimum source-quantity/recovery contract question before further numerical work;
that next owner disposition remains open. No weighting policy, Q02-C/MAP handoff,
FRUIT method or qualification is approved by these outcomes. 129081 stays reserved,
and any later recommendation still needs independent replication and adequate
uncertainty evidence. All previous/frozen packets, diagnostic attempts, reductions
and opaque archives remain preserved; no Unity access or push occurred.

On 2026-09-10 the owner explicitly **closed the weighting exploration** in the
[closure decision and next-work outline](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/WEIGHTING_EXPLORATION_CLOSURE_2026-09-10.md).
No further weighting search, rerun or replication follows from its completed
execution approval. All evidence remains preserved at 349af6c33; neither U nor
N4U is selected, and Q02-C/MAP handoff remain scientifically unresolved.
The recommended next task is one short paper-only minimum reference-method
brief: intended recovered quantity, claim-specific recovery/uncertainty evidence,
exact historical and ordinary-MAP references, and only the remaining numerical
bindings under Q01–Q06. Generic Stage A/B and Q01 remain frozen; approved
residual relearning and A/B substance are not reopened. This closure does not
approve a new source estimator, method, input or numerical run. 129081 remains
reserved, all subsequent execution gates remain, and no reduction product,
prior packet, protected archive, production behavior or Unity state was changed.

The owner then authorized generating the brief. The
[minimum reference-method brief r0.1](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/MINIMUM_REFERENCE_METHOD_BRIEF_R0.1.md)
is now ready for review: proposed POINT peak/position/shape recovery, an
explicitly evaluation-only Gaussian-plus-background fit, provisional bias/loss
requirements, uniform occurrence weighting solely as an unqualified reference,
and one exact reference-record completion step under Q01–Q06. These are new
proposals, not inferred approvals or a reopening of the weighting search.
Q02-C/MAP handoff remain open; A/B substance and the frozen definitions persist.
The historical source anchor is recovered but its exact executed binding remains
unavailable; template settings and recent single-pass exports cannot fill it.
All new choices, source/profile adoption and any essential recovery experiment
retain their separate review. No data analysis, method implementation, source
adoption, run, author dispatch or 129081 scientific access occurred. Only the
brief and current navigation were added/updated; prior evidence is unchanged.

The owner then supplied the ChatGPT discussion “Reference Method Brief” for
consideration. The [r0.2 brief](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/MINIMUM_REFERENCE_METHOD_BRIEF_R0.2.md)
clarifies the pending proposal as an initial POINT reference within existing
mode-aware FRUIT. It makes pointing-offset recovery primary, separates runtime
intent policies from the shared declared recurrence and external evaluation,
and limits the Gaussian evaluator to its adequate subset. Evaluator inadequacy
leaves a claim unavailable and does not itself establish FRUIT failure. A bounded
paper OOF counterexample check finds no missing scientific role in M02/M03/M05;
no OOF numerical policy or implementation adequacy is claimed. The retrieved
three-message discussion is retained as manager review input, not owner approval
or independent-author science. POINT selection, provisional tolerances, uniform
weighting/Q02-C and source adoption remain pending; no generic requirements or
cross-mode acceptance are inferred. The r0.1 brief, weighting closure/results,
frozen definitions and protected archives remain unchanged. No numerical work,
new input access, implementation, author dispatch, run or Unity action occurred.

The owner accepted the exact r0.2 brief on 2026-09-10 in the
[POINT reference acceptance](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/POINT_REFERENCE_BRIEF_ACCEPTANCE_2026-09-10.md).
This accepts POINT for paper completion, its evaluation-only measurement direction
and provisional recovery targets, and Q02-C's uniform occurrence/rejoined MAP
scientific substance for this unqualified POINT reference. The detector-noise
adequacy question is not qualified by that choice. Exact family/QC/source/profile
adoption and intended mode applicability remain pending before numerical use;
A/B substance, all frozen science and the weighting closure persist. The next
authorized paper work is one exact reference record containing recovered control
identities and actual remaining Q01–Q06 choices. The reviewed r0.2 bytes and their
pending-at-delivery wording remain preserved under the acceptance hash. No
experimental order, case, input, implementation, run, author dispatch, Unity or
production authority follows. 129081 and all reduction/archive products remain
preserved. The local acceptance checkpoint is ready for the owner to push.

On 2026-09-10 the owner reported that checkpoint pushed and authorized the
paper continuation. The [POINT reference record r0.1](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/point_reference/r0.1/README.md)
now dispositions all 21 method-record classes, fills the accepted POINT scope,
and supplies controlled B01–B12/S1 amendment and seven candidate profile records.
The five residual PTC @2 versions, uniform QC and MAP @3 are review drafts,
not registered permissions. All existing upstream/frozen sources remain unchanged.
Read-only recovery rehashed the preserved C31 executable and four configuration/
provenance files against recorded identities. C31 remains a distinct refactor-era
reference; the exact f70701ad executable/environment/input/product tuple is still
unavailable. The historical scalar MEDRMS and OR-combined admission branches
are documented as implementation evidence, without a new noise value or
scientific-outcome analysis. Numerical S/N remains unavailable until a compatible
current scale and meaning are approved. Array-wide rank 5, coverage_cut=0.1 and
L=6 including bootstrap are explicit new paper proposals under Q03/Q04/Q06,
not defaults or selected values. Their review, complete source/mode adoption and
remaining evidence/plan bindings precede a separately authorized numerical
request. No run, implementation, new scientific data access, author dispatch,
Unity action or production change occurred. Weighting stays closed and all
prior results, frozen packets, reductions and protected archives are preserved.

On 2026-09-11 the owner replaced the paper-only next step with one bounded
[POINT coherent-source innovation experiment](../validation/fruit_point_coherent_feedback_2026-09-11/OWNER_DIRECTION.md).
The scoped authorization permitted necessary input/control/numerical definitions
and isolated implementation/execution. OG remained an operational benchmark;
reconstructing the unavailable historical environment, comprehensive noise
qualification and OOF generalization were not prerequisites. This did not adopt
upstream production source/profile successors or revise frozen science.

The [completed decision report](../validation/fruit_point_coherent_feedback_2026-09-11/SCIENTIFIC_REPORT.md)
**rejects candidate r0.2 as tested**, after its single targeted revision.
Both matched arms used immutable-parent replacement, rank-5 residual PTC
relearning and common ordinary-map controls. Coherent Gaussian feedback excluded
the fitted background and imposed no expected flux or nominal centroid/width.
Two Gaussian/null realizations and one mismatch case reran learning. The sole
revision fixed a brightest-pixel fit-start failure using coherent numerical
starts for the same free model; reference maps reproduced exactly.

The candidate met all-array Gaussian amplitude/width/centroid targets by pass
three (3.90 and 3.59 s), while the pixelwise reference missed joint targets
through seven. No source was admitted in the two synthetic nulls; direct
mismatch errors improved. Real a2000 candidate feedback nevertheless lay
46.43 arcsec from the OG published pointing estimate, with unresolved source
identification and large strong-array shape/peak changes. OG is not truth and
reference a2000 identification is also unreliable; this is not a measured true
pointing error. The conflict prevents a useful POINT recovery claim. Equal-pass
real method time was 6.59 s reference versus 7.78 s candidate (+18.0%). The
identified OG control's 205.38 s full latency has different upstream work and
threading and does not establish an end-to-end speedup.

Both frozen campaigns completed 168 passes with zero trajectory failures and
about 2.02 GB peak memory. The [integrity check](../validation/fruit_point_coherent_feedback_2026-09-11/VERIFICATION.json)
passed all product hashes, replacement/background exclusion and independent
saved-state map reconstruction. All iteration products and both attempts remain
retained at the [result-manifest paths](../validation/fruit_point_coherent_feedback_2026-09-11/RESULT_MANIFEST.json).
The revision allowance is exhausted; no sweep, further diagnostic study or
129081 evaluation follows. The bounded screen is closed with useful conditional
mechanism evidence but no recommended policy. Any later candidate requires new
bounded owner direction. Weighting remains closed; prior contracts, reductions
and protected review archives remain preserved. No Unity or production change
occurred.

Later on 2026-09-11 the owner corrected the alignment premise: a1100/a1400
are very accurately aligned, while a2000 has a small systematic relative
displacement below 2 arcsec. The owner authorized a new bounded
[relative-position prior test](../validation/fruit_point_alignment_prior_2026-09-11/OWNER_DIRECTION.md).
Its [protocol](../validation/fruit_point_alignment_prior_2026-09-11/PROTOCOL.md)
uses the reference pair to infer a free common position, then conditions a2000
on a direction-free offset disk, retaining independent amplitudes/shapes and
own-array admission. The exact-sharing and disk approximations are declared,
not measured calibration distributions. Eleven cases, two arms and seven passes
are fixed; source-absent, contaminant and offset-mismatch tests rerun learning.
The candidate was frozen at `ae758cd2f` before execution. The
[completed report](../validation/fruit_point_alignment_prior_2026-09-11/SCIENTIFIC_REPORT.md)
recommends **REVISE**: the prior reduces a2000 desired-source amplitude error
from −19.48% to −0.07% in the remote-contaminant test, reaching joint recovery
targets in two passes, while the independent-fit control misses them through
seven. Both aligned Gaussian realizations and the small-offset case pass final
recovery; null and absent-a2000 tests create no source feedback. However, the
two-component mismatch increases a1100 direct map error from 0.162 to 0.283
mJy/beam (+75.0%) and exterior error by 77.9%, failing the frozen +10% gate.
The exact causal division between shared-position inference and its numerical
fit solution is not established. No gate or candidate was changed afterward.

Real a2000 feedback is rejected at the 2-arcsec boundary throughout, preventing
reinforcement of the remote feature while leaving its output at bootstrap;
this is not recovered weak-source pointing or a measurement of true array
misalignment. Physical alignment is distinct from a processed-map fit centroid.
Real seven-pass method time is 6.57 s control versus 7.86 s candidate (+19.7%).
All 154 passes completed in 175.96 s with 2.19 GB peak memory and no trajectory
failure. The [integrity verification](../validation/fruit_point_alignment_prior_2026-09-11/VERIFICATION.json)
passes all 498 product hashes, exact reproduction of the previous six-case
control, and independent stored-state reconstruction. All products are retained
at the [manifest paths](../validation/fruit_point_alignment_prior_2026-09-11/RESULT_MANIFEST.json).

The bounded test is closed with a useful conditional association benefit and a
revise recommendation. A further numerical revision addressing morphology cost
and the relation between geometric alignment and fitted-position uncertainty
requires a new bounded owner decision. 129081 remains reserved because the
development gate failed. This authority does not reopen the rejected predecessor,
adopt production sources, or authorize a parameter sweep; frozen science,
weighting closure, prior reductions and protected archives remain unchanged.

The owner then authorized one isolated [RBF feedback experiment](../validation/fruit_point_rbf_feedback_2026-09-11/README.md)
on 2026-09-11. It compares nonnegative regularized Gaussian-basis inference with
matched pixelwise and independent-Gaussian controls, preserving rank-5 network
relearning and the accepted recurrence. Cross-array coupling is excluded.
A bounded noiseless resolution adjustment passed before the first empirical
freeze; 189 passes over nine matched cases are authorized, with at most one
targeted empirical revision. Concentration is a separate diagnostic and cannot
select reconstruction, admission or stopping. The initial 189 passes completed without solver failures, but the candidate
rejected all synthetic sources and provided no recovery improvement. The one
allowed revision strengthens only the quadratic penalty, with the original
basis, inputs, admission and recovery gates unchanged. Its noiseless fitted
models show substantial smoothing bias. The [completed report](../validation/fruit_point_rbf_feedback_2026-09-11/SCIENTIFIC_REPORT.md)
rejects both candidates after 378 passes: R1 rejects all injections, and R2
admits only two compact array/case combinations, still with −13.45%/−11.77%
amplitude bias and broad excess brightness. Neither admits the comatic source
at either brightness; all comatic output gates fail. The chosen comatic
brightnesses did not establish standalone detectability across arrays, so the
broader RBF hypothesis remains unresolved. Neither candidate admits null or
background-only feedback. Concentration is separately marked revise for
empirical use: its full-overlap algebra is verified, but noisy source estimates
are biased or unavailable. No concentration value selected the method.

The real R2 seventh map costs 7.84 seconds including 1.19-second basis setup,
versus matched P/G 5.23/6.56 seconds. Comparable-useful-recovery cost is unavailable
because the R candidates never meet the joint targets. OG remains the identified
205.38-second operational control with different processing/runtime scope, not
a matched speed ratio. Both campaigns have zero trajectory failures, about
198 seconds each, and at most 2.72 GB peak RSS. All 1,344 payload hashes and
unchanged controls verify; prior packages, 57 frozen ordinary-MAP payloads and
opaque review archives remain preserved. The one revision allowance is exhausted.
This experiment is closed; further scientific design requires a new bounded
owner decision, with no automatic sweep or numerical qualification.
The [exposure audit](../validation/fruit_point_rbf_feedback_2026-09-11/EXPOSURE_AUDIT.md)
corrects the broader 129081 description: July population-quality measurements
already characterized it. It remains reserved for new feedback comparisons,
but is not untouched holdout data. Previous campaign-scoped records are retained.

The owner subsequently supplied a [reassessment](../validation/fruit_point_rbf_admission_audit_2026-09-11/OWNER_REASSESSMENT.md).
The completed [retained-product audit](../validation/fruit_point_rbf_admission_audit_2026-09-11/REPORT.md)
preserves both rejections and corrects the causal interpretation: low pixel-parity
cosine does not uniquely diagnose overfitting. All compact rejection cases fail
the cosine gate, often despite passing both score gates. The two admitted R2
compact cases are first admitted after map 1 and applied in maps 2–7, so late
admission does not explain their final errors. Fully rejected synthetic outputs
are bitwise identical across all seven maps; repetition supplies no new evidence.

A declared oracle diagnostic uses the actual processed response and covariance
of three signed template projections estimated from the other null seed, with
coverage-compatible spatial placements. Compact source scores are 17.04–32.13
empirical scales. Bright-coma source scores are 5.96/2.58/5.27 in array order;
half-bright scores are 3.05/0.48/2.79. These use known processed morphology,
overlapping placements and only two nuisance realizations; they are not
calibrated detection significances. The comatic regime remains unestablished
across arrays. No new parent read or cleaning pass was performed; all 275 used
retained files and 49 frozen RBF packet payloads remain unchanged.

The owner-approved [preliminary starlet screen](../validation/fruit_point_starlet_preliminary_2026-09-11/SCIENTIFIC_REPORT.md)
is complete: **reject this estimator/admission candidate**. Exactly one new
rank-5 bootstrap on the frozen 4× coma established descriptive processed-template
scores of 23.425/15.037/20.130 across the arrays. The 75 map-estimator checks
admitted models in 5/6 null and 3/3 background-only cases. All six processed
compact and three new coma solves hit the 300-iteration cap; no usable model
was admitted for them. The noiseless normalization was undefined at zero outer
MAD, so required recovery/phase checks remain unavailable rather than evidence
of starlet representation failure. The null failure independently rejects the
candidate. The screen used 21.88 s and 2.21 GiB peak RSS; the 0.5 s median-array
inference bound was exceeded on 10/25 maps. Operator, mask and stored-gradient
checks passed, and prior packets/products and protected archives remain intact.
No full trajectories ran. This candidate is closed with no revision allowance;
any new estimator or changed screen requires a separate owner decision. The
next direction may return to a bounded POINT reference contract. 129081 remains
reserved for new feedback comparisons, with historical exposure disclosed.
No RBF retuning, Unity activity, production change or qualification is authorized.

On 2026-09-12 the owner directed one [matched central-domain test](../validation/fruit_point_starlet_central_domain_2026-09-12/SCIENTIFIC_REPORT.md).
It is complete: **the edge/domain hypothesis is supported, while the tested
60-arcsec reconstruction candidate is not accepted**. All 32 selected null
coefficients and 20 background coefficients lie at radii 68.41–89.31 arcsec;
selection rates account for eligible coefficient counts. The central domain
removes all null admissions (5/6 to 0/6) and background admissions (3/3 to 0/3),
using unchanged noise scales and surrounding computational/background maps.
Some old false brightness had spread inward, up to 6.72%. The normalization
repair enables noiseless solves, and all 51 positive-MAD controls reproduce the
old numerical outputs bitwise. Required synthetic solves still hit 300 iterations.
The domain excludes 11.45–11.47% of the original coma brightness and 25.38% for
the fixed offset stress case; that offset has an irreducible 29.01% image-error
floor, independently failing both declared image-error limits. The test used
174 estimator calls in 72.66 s, with zero new PTC passes or trajectories.
All prior packets/products and protected archives remain unchanged. The next
owner decision may separate central source-evidence eligibility from model
extent and resolve reconstruction fidelity/completion. No automatic radius,
threshold or solver sweep, full trajectory, new 129081 comparison, Unity,
production or qualification follows. The earlier full-domain rejection remains
preserved; its broad return-to-contract recommendation is narrowed by this
new spatial evidence.

The owner's later [POINT operational account](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/POINT_OPERATIONAL_USE_CASES_2026-09-12.md)
establishes five uses: startup flux/focus assessment, before/after OOF gain and
recentering, routine pointing, degradation monitoring, and tune/condition health.
The owner explicitly selects **peak response** as the primary same-source OOF
gain measure; integrated brightness and shape remain diagnostics. The new
[use-specific gate design](scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/POINT_USE_CASE_GATE_DESIGN_R0.1.md)
proposes routine pointing and relative peak gain as the first improvement
targets, with honest startup/degradation/health safeguards. It would make
whole-image fidelity blocking only where needed for the claimed use, while
retaining all residual, support, failure and cost evidence. A 5% peak-ratio
error target, 10% degradation sensitivity probe and eight-state/two-realization
test design are proposals, not adopted operational thresholds or execution
permission. The accepted brief and all candidate dispositions remain unchanged;
failed starlet solves do not become available. The next review concerns this
prospective applicability change and one bounded successor protocol. No new
numerical work, candidate selection, 129081 comparison or production change
was performed or authorized by the notes.

The owner then authorized **“Let's give this test run a try”** and selected
**1 arcsec** as the per-observation pointing criterion for this trial. The
[prospective protocol](../validation/fruit_point_operational_gate_trial_2026-09-12/PROTOCOL.md)
binds the eight states/two nuisance seeds, real123424, rank 5, matched pixelwise
control and seven-pass development sequences. It retains the central starlet
objective/domain and both solve-success checks, with one predeclared increase
from 300 to 3000 solver iterations. Sources and inputs are frozen before
execution; up to 238 cleaning calls, one hour, 8 GiB RSS and 4 GiB output are
the bounds. No automatic revision, 129081 comparison or qualification follows.

The [operational-gate trial is now complete](../validation/fruit_point_operational_gate_trial_2026-09-12/SCIENTIFIC_REPORT.md):
**candidate not accepted; recommend a bounded completion/evaluator revision**.
All 17 pixelwise trajectories completed; the central candidate completed the
real case and four null/background cases, but no synthetic source trajectory.
It made zero false admissions over 12 null/background array cases and all seven
passes, versus false pixel admission in every corresponding reference case;
neither arm issued a positive POINT source report there. Of 21 rejected
nonempty solves, 19 meet the relative-gradient requirement but fail optimizer
success; two fail both. All 18 reference compact/mild-case centroid errors are
0.0336–0.5888 arcsec, yet the common shape/score gate admits only ten. This
exposes a limitation of the shared measurement veto, without changing the
registered scores. The candidate's real trajectory takes 43.51 s versus 8.07 s
for the matched reference and supplies no available operational measurement
under that evaluator. No primary recovery/time improvement is established.
The run used 167 cleaning calls in 400.12 s and 1.94 GiB peak RSS. Source/input,
product, recurrence, objective/gradient and sampled learned-state checks pass;
prior products and protected archives remain unchanged. The next significant
decision is one explicit completion rule and separation of centroid usability
from peak/shape and health-warning rules. No further trial or 129081 evaluation
is authorized automatically.

The owner subsequently authorized **one bounded repair-and-retest**, handling
saved-map evaluator reassessment and numerical stopping qualification separately
before any new operational trajectories. The
[repair protocol](../validation/fruit_point_bounded_repair_2026-09-12/PROTOCOL.md)
preserves original records and uses data-only source association, fit-domain
sensitivity and separate centroid/peak availability with retained shape/support
warnings. The numerical check preserves the objective and L-BFGS-B family,
captures the declared 1e-4 relative-gradient stop and a 1e-6 point along the same
optimizer path, and compares peak/centroid stability against 0.5%/0.1-arcsec
allocations. The operational 3000-iteration cap remains; only the diagnostic
tighter comparator gets extra work. All saved problems are retained. A failed
saved-stage gate blocks the conditional 34-trajectory rerun. Neither repaired
labels nor optimizer completion alone establishes candidate benefit. No further
revision, 129081 comparison or production change is authorized by this decision.

The [bounded repair is complete](../validation/fruit_point_bounded_repair_2026-09-12/SCIENTIFIC_REPORT.md)
at the saved-problem gate: **retain the evaluator repair; reject the
completion-only repair at the declared tolerance**. The data-only evaluator
recovers all 18 reference compact/mild centroids as usable (0.0336–0.5888 arcsec
external errors), retains all seven shape warnings, and preserves null and
boundary safeguards. Peak availability remains separate and limited; only two
of six reference gain/degradation pairs are available and pass, and the usable
unchanged a2000 pair shows an 11.67% apparent peak loss. Original measurements
and registered outcomes remain unchanged.
All 60 nonempty saved problems reach relative gradient <=1e-4 within 21–611
iterations; 57 reach the tighter 1e-6 comparison point on the same optimizer
path. Only 10/60 pass the registered 0.5% peak/0.1-arcsec model-stability
allocations, including 0/27 compact/mild/response-loss problems. For those 27,
Gaussian readout changes stay within the full operational budgets (max 2.49%
peak, 0.189 arcsec), but the actual model sampled peaks differ by 0.94–67.27%.
This is a failed numerical qualification, not demonstrated operational failure
or a general rejection of wavelets. The conditional trajectory rerun was not
admitted: zero new pre-PTC reads or cleaning calls. Saved-map work took 272.11 s
and 146.28 MiB peak RSS; verification and preservation checks pass. No useful
candidate recovery/time improvement is established. Any new tolerance,
objective, support or image-selection procedure is a separate scientific owner
decision; no further numerical experiment is authorized by this completed repair.

On 2026-09-13 the owner separately authorized a
[small paired feedback-sensitivity experiment](../validation/fruit_point_feedback_sensitivity_2026-09-13/PROTOCOL.md).
It preserves the evaluator repair and failed qualification, examines all saved
nominal/tighter pairs in image, selected-coefficient and actual projected-sample
space, then performs exactly six one-step PTC-relearning branches on three
registered states. The largest source-region projected-change state is selected
before replay by a frozen data-only rule. There is no new feedback optimization,
regularizer, method, gate or full trajectory. Subspace and next-total peak/centroid
effects determine whether a later scientific change is warranted; image-peak
differences alone do not establish consequential underconstraint.

The [paired feedback-sensitivity diagnostic is complete](../validation/fruit_point_feedback_sensitivity_2026-09-13/SCIENTIFIC_REPORT.md):
141 saved pairs were screened (57 nonempty, 84 zero; three tighter solutions
remain missing), followed by exactly six one-step relearning branches on
H_20260911/pass0, T_20260911/pass1 selected by the frozen source-core projection
rule, and real123424/pass0. All nine raw next-map centroid movements are below
0.1 arcsec (max 0.04351); eight of nine fitted peak changes are below 0.5%.
T/a1400 changes by -1.195% in peak and +13.72%/-7.98% in fitted major/minor width;
the nominal peak is withheld by the unchanged evaluator and the tighter one is
available. Seven peak pairs and eight centroid pairs are jointly usable. The
earlier 67% model-peak statistic therefore overstates the next central-readout
effect, but the difference is not harmless: projected models, returned off-source
structure and learned subspaces change substantially (real/a1400 reaches an
88.02-degree principal angle in one group). Selected wavelet evidence changes
by only 0.026–0.247% of its target norm across the replayed array pairs, while
some small remaining objectives improve materially. This supports consequential
reconstruction freedom without proving an exact null space or excluding
numerical conditioning. The recommendation is to propose one explicit constraint
on unsupported fine-scale feedback, with source position, amplitude and shape
remaining free; formulation and execution require a new scientific owner
decision. No estimator or gate changed, no new feedback optimization occurred,
and the prior qualification remains failed. The run took 41.21 s and 2.59 GiB
peak RSS. All 864 learned covariance/eigen identities, 432 subspace comparisons,
product/preservation checks and focused tests pass. No full-trajectory stability,
scientific advantage, policy or production readiness is established.

On 2026-09-13 the owner superseded the fine-scale-constraint recommendation
and authorized one [fixed nominal-estimator POINT utility comparison](../validation/fruit_point_nominal_utility_2026-09-13/PROTOCOL.md).
The earlier image-stability qualification remains failed. This new question
keeps the tested nominal 1e-4 stopping path, objective, domain and work caps,
and compares complete replacement-feedback trajectories against matched P on
the existing 17 cases. Both arms are rerun for credible timing: at most 238
cleaning calls, seven passes, rank 5, four threads, unchanged resource ceilings.
The repaired data-only evaluator separates pointing, peak availability and
shape/support warnings. Required 1-arcsec/5%-ratio results, honest yield, null
safeguards and <=2x matched wall cost govern advance versus park for POINT;
standalone image/subspace/leakage differences are supporting diagnostics.
No fine-scale constraint, tighter solve, sweep, automatic revision, 129081,
Unity, OOF campaign or production change is authorized by this experiment.

The [nominal POINT comparison is complete](../validation/fruit_point_nominal_utility_2026-09-13/SCIENTIFIC_REPORT.md):
**park this candidate for POINT**, preserving its narrower raw ratio-fidelity
result. All 34 trajectories completed, with exactly 238 cleaning calls in
402.62 s and 2.07 GiB peak RSS. All 273 nonempty nominal solves finished within
746 iterations; 84 empty-support decisions correctly supplied zero models.
Both arms pass all 18 required usable 1-arcsec pointing cases (C maximum error
0.716 arcsec; P 0.589). C improves all six raw H/D gain errors to within 0.81%,
but usable gain/degradation pairs fall from P 2/6 to C 1/6. C improves usable
health ratios from 3/6 to 5/6 and unchanged-H outcomes from 0/3 to 1/3, while
the usable unchanged a2000 pair still falsely loses 6.58% peak response. Both
arms retain startup, null-source and boundary safeguards; C admits no false
null feedback. Six source trajectories exceed 2x matched wall cost; real123424
costs 13.07 s versus 8.37 s (1.56x). The successful finite cases do not establish
a consistent required peak-use improvement across the registered population.
All P controls and 51 saved nominal bootstrap models match exactly; 238
recurrences, 357 fixed problems, 672 truth scores and 864 sampled relearning
identities pass verification. No new verification cleaning/optimization was
needed. All prior products, frozen authority and opaque archives are preserved.
The earlier image-stability qualification remains failed. No automatic repair,
evaluator relaxation, new parameter, OOF campaign, 129081 evaluation or
production advancement follows this result. Further numerical work requires
its own specific scientific owner decision.

The owner's subsequent [saved-product peak diagnosis](../validation/fruit_point_peak_failure_diagnosis_2026-09-13/DIAGNOSIS.md)
preserves the parked candidate and every registered result. Its strongest
finding is cancellation of shared noise-dependent peak errors: raw H/D
within-5% counts change from matched P 5/6 and C 6/6 to crossed P 3/6 and C
2/6. These are dependent recombinations of two saved realizations, not new
validation cases. Primary H/D/T peak withholding is exact: P has six score
failures and two domain-sensitivity failures; C has four score and three domain
failures, with no shape-only veto or unfinished feedback solve. A separate
common-readout contribution is demonstrated by P's essentially exact 0.8 map
scaling but 0.817% peak-fit departure and a score crossing from 4.9838 to 5.0245.
The dominant map-versus-free-fit contribution to larger between-noise errors
remains unresolved. Exactly one next experiment is proposed, not executed:
compare the saved free Gaussian readouts with truth-shaped, unit-peak template
plus free-plane amplitude fits on the 24 terminal H/D array maps in both
existing domains (48 linear diagnostic fits). Truth assistance would remain
an evaluation-only ruler, never feedback or a deployable availability policy.
The diagnosis verifies 714 availability records, 504 ratios and 84 saved-map
scaling cases, with zero pre-PTC reads, PTC calls, refits, new thresholds,
truth-selected iterations or reserved observations. No OOF or production
advancement follows; the proposed experiment requires a separate owner decision.

The owner then authorized **“Give this a go”** for that exact saved-map
[fixed-template readout comparison](../validation/fruit_point_fixed_template_readout_2026-09-13/PROTOCOL.md).
It binds 48 unrestricted linear amplitude-plus-plane fits on the existing
24 terminal H/D array maps in the two retained domains, with a deliberately
truth-assisted source-shape ruler. The free-Gaussian comparators and all
registered availability labels remain unchanged. No PTC, feedback inference,
new observations, deployment policy or subsequent experiment is included.

The [48-fit comparison is complete](../validation/fruit_point_fixed_template_readout_2026-09-13/SCIENTIFIC_REPORT.md),
with a mixed map/readout result. At the existing 60-arcsec domain, raw absolute
peaks within 5% change from P free 6/12 to fixed-template 0/12, and from C free
4/12 to fixed-template 11/12. The 52-arcsec comparison also gives P 0/12 and C
11/12. The candidate therefore contains substantially better amplitude along
the known injected source shape in these cases; the reference free fits use
narrower profiles, which can obscure that deficit in a peak-only comparison.
Residual noise dependence remains: C fixed-template crossed H/D passes 4/6
at each domain, with maximum errors 7.27% and 8.31%. Six saved free-Gaussian
fits have higher residual cost than a known feasible fixed-template solution
in their own model family, including the four inner fits associated with all
four distinct H/D domain-sensitivity rejection states. This establishes a
common source-fit numerical solution-selection problem without relabeling
any prior result. All 48 linear fits are finite, rank four, condition number
12.14--14.61; independent residual/model checks pass with zero additional fits,
PTC or feedback work. All prior products and archive statuses remain intact.
The truth-assisted ruler is not deployable measurement authority: availability,
pointing, null, runtime and earlier qualification outcomes remain unchanged,
and starlet stays parked for POINT. No repair, new experiment, reserved pointing,
OOF or production work follows automatically.

The owner approved the exact four-file MAP author packet and Ultra Stage B
dispatch later on `2026-08-26`; this authoring approval does not approve the
revision that will be returned.
The fresh implementation-blind author returned a bounded r0.1 specification
on `2026-08-26`. Manager review retained its successful PTC-route, ownership,
response/covariance, and immutable-derivative science but did not accept it for
integration because it reinterpreted x/r identity, invented a universal PTC
availability-scope rule, standardized a JSON/SHA representation, and would
have closed OD-003 through an unapproved claim registry. The scientific owner
then directed a bounded correction of the actual canonical deliverable. The
r0.4 formal, rationale, and engineering sources now fix CAL-to-PTC-to-MAP as
the sole route; separate PTC facts from MAP admission and ALIGN/AST coordinates
from MAP projection; retain VAL only as a rule registry/evaluator; preserve
declared failure scopes; keep paired x/r upstream; and require honest response
and covariance disclosure with versioned later derivatives. All 52 requirement
IDs, 25 prediction IDs, and nine open decision IDs are retained. The durable
checker passes and all 56 rendered PDF pages passed independent Poppler visual
inspection. This is an owner-review draft, not an implementation-conformity,
validation, freeze, performance, Unity, or readiness result.

On 2026-08-27 the owner-directed targeted SCI-MAP r0.5 cross-package closure
was integrated for review without inspecting implementation, schemas, tests,
reductions, validation products, or production behavior. It binds frozen PTC
r0.5 and WP-7 closure, frozen CAL and AST authority, continuing VAL Core/
Registry/source bindings, and the exact exposure/coordinate boundaries. The
package now includes `SCI-PTC_TO_SCI-MAP v0.1/r0.1`, registered
`SCI-MAP:map_upstream_admission@1`, exact one-hot containing-pixel projection,
original-occurrence exposure carriage, typed response/null/covariance state,
and exact uniform observation-coadd/admission profiles. It renames the ordinary
quantity as calibrated-`x`-derived nonpolarimetric total-intensity-equivalent
rather than Stokes I. All 52 requirement and 25 prediction IDs remain stable;
OD-008 is resolved and eight owner decisions remain open. Frozen PTC leaves the
MAP-facing coefficient family open and MAP OD-007 leaves the admitted numerical
`coverage_cut` domain open, so no generally authorized ordinary numerical MAP
route is claimed. The durable verifier passes and all 58 PDF pages passed
Poppler visual inspection. This remains an owner-review document result only,
not implementation conformity, validation, achieved response/performance,
freeze, readiness, or production authorization.

The owner-approved SCI-MAP r0.6 targeted scientific-closure pass was completed
on `2026-08-27`. It preserves all 52 requirement IDs, 25 prediction IDs, nine
owner-decision IDs, and eight open MAP-local decisions while correcting OD-008
parity; typing all ten contribution gates; defining unique-original exposure
at each original's own AST coordinate; separating fixed-state, PTC
full-procedure, and whole-chain response; and registering source-current
`SCI-MAP:map_upstream_admission@2` plus VAL-governed aggregate
`SCI-MAP:observation_coadd_admission@1`. The exact
`SCI-PTC_TO_SCI-MAP v0.1/r0.1` artifact is byte-identical in both packets. The
durable verifier passes and all 60 rendered PDF pages passed Poppler visual
inspection. The conditional estimator is coherent, but the PTC coefficient
family and admitted numerical `coverage_cut` state still block a source-closed
numerical route. No implementation conformity, validation, response fidelity,
observational performance, freeze, readiness, or production authorization is
claimed.

On `2026-08-28` the final targeted SCI-MAP r0.7 closure packet completed its
content, mechanical, and rendered-document preflight for scientific-owner
review. It preserves all 52 requirement IDs, 25 prediction IDs, and nine
owner-decision IDs; imports one canonical shared authority into all three
views; binds the exact `SCI-PTC_TO_SCI-MAP v0.1/r0.1` source; and treats PTC
coefficient availability as structural while leaving coefficient finiteness
to MAP's numerical gate. The packet records the ordered A--E contribution DAG,
the single operator identity
`A_MAP,Pi = A_out = D_Q,out^-1 J_out G Omega`, fixed-state/PTC-full-procedure/
PTC+MAP-re-resolved response families, and original-footprint exposure at the
stable original's layered AST ALIGN-grid coordinate in the exact target WCS.
The two installed PTC/MAP boundary copies and every stable/r0.7 PDF pair are
byte-identical. The durable verifier passes, and all 65 PDF pages passed
Poppler inspection. The ordinary numerical route remains blocked by the
missing exact PTC MAP-facing coefficient family and unresolved admitted
numerical `coverage_cut` domain. This is an owner-review closure packet, not
implementation conformity, validation, achieved response or performance,
scientific-authority freeze, readiness, or production authorization.

On `2026-08-28` the owner-directed SCI-MAP r0.7.1 freeze-only errata replaced
the symbolic decision tuple with named request/applicability/eligibility/
realization fields, corrected the exact `coverage_cut` wording, canonicalized
boundary/profile identifiers, defined exact original-footprint exposure
aliases, narrowed REQ-010 structural-failure grammar, and bound the exact
Registry/source-binding records and reports through an externally SHA-bound
source manifest. All 52 requirement IDs and 25 prediction IDs remain stable;
estimator, response, exposure, support, covariance, coadd, and lifecycle
semantics are unchanged. This is an author-artifact closure record only and
makes no implementation, conformity, validation, performance, freeze,
readiness, production, or route-availability claim.

On `2026-08-28` Grant Wilson authorized the status-only freeze of the exact
SCI-MAP v0.1/r0.7.1 candidate at commit `bd010e20e`. The scientific-owner
freeze record binds the externally hashed source manifest, the 52 requirements,
25 predictions, exact boundary/profile/Registry identities, owner ledger, and
three canonical PDFs without changing their scientific or rendered bytes.
Eight MAP owner decisions and the two hard numerical gates remain explicitly
open; the source-closed numerical MAP route therefore remains unavailable.
The freeze establishes scientific authority only and makes no implementation
conformity, representation-fidelity, validation, response-achievement,
performance, readiness, or production claim. Future substantive changes
require explicit owner authority and a versioned successor or formally
reopened revision.

On `2026-08-28` the scientific owner launched SCI-JINC v0.1 as the next
downstream Stage A tranche from exact scientific-contract library authority
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`.
The package links the program charter, downstream roadmap, and frozen SCI-MAP
predecessor; recovers and classifies the frozen SCI-MAP-002 independent core,
all eight JINC owner decisions, later destination-ownership work, and later
integration/validation history; records exact revisions and content digests;
and preserved the unidentified memo behind the historical alignment note as
unavailable rather than reconstructing it. Later on `2026-08-28`, the owner
supplied F. Peter Schloerb's exact LMT OTF/JINC memo as an authoritative
generic-method reference; it is not asserted byte-identical to that unnamed
historical source. The package preserves the 42-page original,
admits only a verified pages-15--19 generic-method excerpt under an explicit
cover, and uses it to close the generic two-JINC plus envelope formula. The
memo is geared to 3-mm spectroscopic receivers: its FCRAO parameter values,
simulations, scales, optimization and performance claims are not TolTEC
authority. Exact `a1100`/`a1400`/`a2000` radial scales and per-array parameter
provenance remain open. The repaired packet also adds PTC/AST boundary
candidates, a JINC admission-profile candidate, collision-free notation,
geometry, grouping/product, response/covariance and inherited-decision tables,
while quarantining implementation, audit, repair, re-audit, Unity, validation,
achieved-performance, readiness, and production evidence. The repaired Scope
Brief, open-question ledger, and exact content-bound packet were returned for
renewed scientific-owner review. No implementation-blind Stage B rationale or
engineering conformance contract was commissioned or drafted, and frozen
SCI-MAP authority is unchanged.

Grant Wilson approved the exact repaired SCI-JINC Stage A candidate at
`6639bff3d94b92ace8faf3e407ccaefd5a38ea1f` on `2026-08-28`. The separate
approval record binds the manifest and Scope Brief hashes without changing any
approved author input. This closes only the exact-byte owner gate. The PTC
coefficient family, TolTEC per-array scale/parameter authority, VAL successor,
numerical phase/cache/error policy, outside-center edge rule, and residual
ledger gaps remain unresolved or require typed disposition. Stage B remains
unlaunched; Ultra use was not authorized; no implementation, validation,
performance, readiness, production or push claim follows.

Later on `2026-08-28`, Grant Wilson resolved `SCI-JINC-ODQ-101` for successor
architecture. PTC owns one versioned registry of positive analysis/gridding
coefficient families; every family/version explicitly permits `SCI-MAP`,
`SCI-JINC`, or both, and user or authorized versioned mode-policy selection
retains distinct requested/effective/observation-resolved/realized identities.
JINC consumes the same positive PTC-produced `omega_i` only with explicit JINC
permission and owns its signed `kappa_ip`, `w_ip`, normalization, conditioning,
support, response, covariance and product semantics. The exact post-freeze
predecessor at `54475956f6aefb839d43b2f0fb019a142cb64310`, SHA-256
`4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c`,
is admitted only under a JINC-specific cover and does not modify frozen MAP or
PTC. No family is registered by this decision, so numerical production remains
typed unavailable until an exact JINC-permitted family is selected and
realized. The bounded successor packet now awaits exact-byte approval under
`SCI-JINC-STAGE-A-Q002`; Stage B remains unlaunched. The next owner question is
`SCI-JINC-ODQ-102B`, the exact TolTEC per-array radial scale and parameter
source/value disposition.

Grant Wilson then resolved `SCI-JINC-ODQ-102B` on `2026-08-28` by a semantic/
no-numerical-route disposition. SCI-JINC preserves `r'_a=r/s_a`, with `s_a`
an explicit array-associated angular scale; the Schloerb `s=lambda/D`
realization is precedent but does not authorize current TolTEC values. The
inherited `lambda_a/(45 m)`, `(a,b,c)`, and mode-dependent `r_max` values remain
quarantined implementation evidence with partially recoverable history, not
TolTEC scientific authority, and no physical interpretation of `45 m` or
hidden default is admitted. Stage B may define parameter semantics and typed
unavailability, while a three-array numerical optimization is deferred to a
separate scientific exercise. The refreshed ODQ-101/102B successor packet
still awaits `SCI-JINC-STAGE-A-Q002`; Stage B remains unlaunched. The next
unresolved scientific-owner question is `SCI-JINC-ODQ-103`, the exact AST
coordinate-role/parent join, JINC admission/profile identity, boundary rule,
and cause policy.

Grant Wilson then resolved `SCI-JINC-ODQ-103` on `2026-08-28`. AST owns the
authoritative coordinate realization, its parent-sample association, validity/
support facts and producer causes; JINC consumes the coordinate associated
with the same processed sample realization entering its estimator. The
association is scientific authority, while any key, table join, index or
object mechanism is engineering choice. JINC owns the single profile
`SCI-JINC:jinc_map_contribution@1`, local geometry, sample-pixel support,
signed coefficient, coupled-accumulator identity and local cause policy. No
row/order/time/tolerance/detector fallback, ordinary MAP validity inheritance,
producer-owned JINC-usability decision or new per-contribution provenance
system is admitted. The successor boundaries are AST-to-JINC r0.2 and PTC-to-
JINC r0.3; frozen AST/MAP/PTC remain unchanged. The refreshed ODQ-101/102B/103
packet still awaits `SCI-JINC-STAGE-A-Q002`; Stage B remains unlaunched. The
next unresolved scientific-owner question is `SCI-JINC-ODQ-104`, whether base
v0.1 adopts only `jinc_coefficient_squared_time` and defers or authorizes a
distinct physical-exposure role.

Grant Wilson then resolved `SCI-JINC-ODQ-104` on `2026-08-28`:
`jinc_coefficient_squared_time=sum_i I_ip kappa_ip^2/f_s,i` is the sole
base-v0.1 time-support product. Its method-specific seconds meaning and
prohibited physical-exposure, precision, validity and significance
interpretations remain explicit. A separate physical-exposure product is
deferred until an identified scientific use requires and separately
authorizes exact original-occurrence lineage and semantics. The refreshed
ODQ-101/102B/103/104 packet still awaits `SCI-JINC-STAGE-A-Q002`; Stage B
remains unlaunched. The next unresolved scientific-owner question is
`SCI-JINC-ODQ-105`, whether base v0.1 is observation-only with any future JINC
coadd requiring a separately authorized boundary over complete JINC bundles.

Grant Wilson then resolved `SCI-JINC-ODQ-105` on `2026-08-28`. SCI-JINC v0.1
defines the estimator and complete product bundle for one observation and
authorizes no cross-observation combination semantics. Observation is the
scientific grouping boundary rather than a streaming, chunking, process or
memory boundary, so same-observation samples/chunks may accumulate
incrementally only under one exact array/JINC realization/bundle identity.
Any future JINC coadd requires a separately authorized boundary over complete
observation bundles and may not inherit ordinary MAP or infer accumulator-
addition or normalized-map algebra. The refreshed ODQ-101/102B/103/104/105
packet still awaits `SCI-JINC-STAGE-A-Q002`; Stage B remains unlaunched. The
next unresolved scientific-owner question is `SCI-JINC-ODQ-106`, the
independent per-array observation-bundle, missing/unrequested-array
cardinality and residual destination-identity disposition.

Grant Wilson then resolved `SCI-JINC-ODQ-106` on `2026-08-28`. For one
observation, JINC may produce zero through three independent bundles, with at
most one for each stable array admitted/requested under the exact JINC
realization and destination geometry. Missing, unavailable or unrequested
arrays create no placeholder or empty product and do not invalidate a
different produced bundle. Same-identity contributions may accumulate under
ODQ-105, while contributions with different array or destination identities
must not merge. Existing plan/bundle provenance remains sufficient; no per-
contribution provenance is added. The refreshed ODQ-101/102B/103/104/105/106
packet still awaits `SCI-JINC-STAGE-A-Q002`; Stage B remains unlaunched. The
next unresolved scientific-owner question is `SCI-JINC-ODQ-107`, the required,
conditional-required, optional and outside product-role disposition plus exact
cause vocabulary and unavailable-role representation.

Grant Wilson then resolved `SCI-JINC-ODQ-107` on `2026-08-28` by rejecting the
broad product-availability/provenance framework and fixing one closed per-
array bundle schema: required `N_p`, `C_p`, `Q_p`, derived `m_p` with its local
support/validity state, and `jinc_coefficient_squared_time`. Failure to form
any required whole-product role suppresses the complete bundle; ordinary
pixel-level invalid support does not make a role unavailable and creates no
role-availability record. No generic optional/conditional machinery, detailed
missing-product cause vocabulary, per-pixel/per-contribution provenance,
operational-reason archive, placeholder or required diagnostic is authorized.
The cancellation absolute-term sum/count/bound remains nonpersistent
construction state governed by ODQ-109. Every other role is outside/deferred;
ODQ-108 response/covariance products are deferred pending a concrete
scientific use, and their reference table is removed from author inputs. The
refreshed 16-object ODQ-101/102B/103/104/105/106/107 packet still awaits
`SCI-JINC-STAGE-A-Q002`; Stage B remains unlaunched. The next unresolved
scientific-owner question is `SCI-JINC-ODQ-109`, the exact summation/count
error bound, deterministic accumulation and phase/cache numerical policy.

Grant Wilson then resolved `SCI-JINC-ODQ-109` on `2026-08-28` by recasting it
around scientific conditioning and sufficient instrument-relevant numerical
accuracy. Finite-state requirements, `Q_p>0`, `C_p!=0`, exact-cancellation
rejection, finite-negative normalization, unit/common-scale invariance and
dimensionless `rho_p` remain. Total numerical error from arithmetic,
accumulation/reduction order, function evaluation, phase quantization and
cache/index realization must be negligible compared with the approximately
`10^-3` relative fidelity relevant to the instrument. No prescribed summation
algorithm, contributor-count/machine-epsilon bound, universal `rho` cutoff,
exact adequate tie/bin/cache choice, bitwise reproducibility or stronger
precision is a scientific requirement; adequate realization and test design
belong to later engineering conformance. The refreshed 16-object successor
packet still awaits `SCI-JINC-STAGE-A-Q002`; Stage B remains unlaunched. The
next unresolved scientific-owner question is `SCI-JINC-ODQ-110`, the rule for
a rounded sample center outside the finite map whose square support overlaps
the map.

Grant Wilson then resolved `SCI-JINC-ODQ-110` on `2026-08-28` with the center-
admission rule. The resolved rounded center used for JINC cache placement must
lie in the finite destination domain before footprint evaluation. An outside
center sets `I_ip=0` for every destination pixel and changes none of `N_p`,
`C_p`, `Q_p` or `T_p^(kappa^2)`, even when its square overlaps the map. An
admitted in-map center retains ordinary cropped square membership without
wrap, completion, renormalization or edge correction. JINC-then-crop
equivalence is not required, and no added edge cause, provenance or diagnostic
product follows. No unresolved numbered scientific-scope ODQ remains. The
next scientific-owner decision is `SCI-JINC-STAGE-A-Q002`, exact-byte approval
or revision of the 16-object successor packet; the versioned VAL registry
binding remains a separate Stage B dispatch prerequisite. Stage B remains
unlaunched.

Grant Wilson then approved `SCI-JINC-STAGE-A-Q002` on `2026-08-28` for the
complete exclusive Stage A successor packet represented by the author-packet
manifest at commit `88dcce8b0f7b1d78053b25831b39cf370afd47cc`, manifest
SHA-256 `52a8e843456a8cb033b7593d9b9f67fb83b0ee565c91c141d8e16d46b906140e`.
All sixteen allowed-object digests and the information firewall are approved;
the approved input bytes remain unchanged. Stage A is closed, no additional
JINC scientific-owner question remains, and Stage B authorship is authorized
from the controlled packet but was not dispatched in this closure increment.
The sole dispatch prerequisite is a versioned SCI-VAL source/profile registry
binding for `SCI-JINC:jinc_map_contribution@1`. Frozen SCI-VAL v0.1/r0.3
expressly preserves continuing registries, and the approved JINC profile
already supplies the owning policy, so this is an administrative/interface
successor rather than a reopened scientific choice. No Stage B normative
content, PDF, implementation-conformity, validation, performance, numerical-
readiness, production-readiness, or remote-push claim follows.

The SCI-JINC Stage B registry prerequisite was then satisfied on `2026-08-28`
through new immutable SCI-VAL continuing-registry successors. The source-
binding identity is `SCI-VAL_SOURCE_BINDING_REGISTER
v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-2026-08-28`, SHA-256
`0e7ca29ee2e9cd02fb1b76cf87cc64fce6164407a7801f9b9a105ca646317e88`;
the profile identity is `SCI-VAL_PROFILE_REGISTRY
v0.1/r0.3-map-r0.7.1-jinc-stage-a-q002-2026-08-28`, SHA-256
`4b9a1ebecfc847c83b59da772afd9b031ab1830e8febbb12d1a47f70ce5a1110`.
They register the exact JINC-owned atomic profile with neutral-preserved
`advisory` response/uncertainty roles. The existing MAP-bound registry files
remain byte-identical, so frozen MAP authority and verification are unchanged.
All SCI-JINC Stage B dispatch prerequisites are now satisfied; a fresh
implementation-blind Ultra author may launch from the exact approved manifest.
The registry binding itself is not an author input and creates no numerical
route, conformity, validation, performance, readiness or production claim.

On `2026-08-29` the implementation-blind SCI-JINC Stage B authoring and bounded
owner-review sequence completed. The two final views import one six-module
shared authority containing 44 stable requirements and 36 stable predictions.
Grant Wilson separately approved the positive-axis half-pixel center-tie rule,
which is bound with the phase-lattice disposition under stable decision
`SCI-JINC-DEC-PHASE-CENTER-001` without inferring one approval from the other.
SCI-JINC v0.1/r0.3 is frozen at commit
`a9f43877e01a661db13bd85b2e7f34ea5ac82fb7` and tag
`sci-jinc-v0.1-r0.3`; the superseding freeze-manifest SHA-256 is
`ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2`.
All source/hash, parity, identifier, reference, PDF reopen, metadata, render,
and visual checks passed. A post-freeze implementation-blind horizontal audit
over the exact frozen PTC/JINC/AST/VAL authorities found no material
incoherence and opened no successor. The verification evidence is committed at
`d6240cc5a6d3f617e60dc907f7f714c8e0212973`, and the consolidated
`codex/scientific-contract-library` branch contains that record. The numerical
TolTEC JINC route remains typed unavailable pending the separately owned
JINC-permitted PTC coefficient family, authorized TolTEC array parameter set,
and, where numerical support is claimed, exact adequacy profile with matching
certificate. No implementation candidate was inspected, and no conformity,
representation-fidelity, validation, achieved-performance, readiness,
production, or production-authorization claim is made. SCI-JINC is closed for
the time being; any later scientific correction requires an explicitly
authorized versioned successor and shall not modify the tagged bytes.

On `2026-08-29` Grant Wilson launched SCI-NOI v0.1 as a recovery-first Stage A
effort at high reasoning effort. The recovered candidate was then subjected to
the owner-directed final Stage A scope repair without further implementation,
configuration, schema, test, audit, validation, reduction, Unity, default, or
historical-behavior inspection. The repaired packet uses collision-free
`NOI-GEN`, `NOI-UNC`, and `NOI-STD` roles; exact fixed-state and relearned
operator DAGs; explicit finite assignment design, source-imprint,
target/estimator/rank/covariance, STD compatibility, and atomic lifecycle
semantics; exact sanitized MAP, JINC, and conditional pre-MAP PTC boundaries;
four NOI-owned VAL profile drafts; and a bounded FLT/Wiener/FRUIT record. One
sanitized decision artifact now separates every consequential choice, and the
exclusive author packet plus closure report bind exact bytes by SHA-256.

At that Stage A checkpoint, Stage B had not been launched. Grant Wilson
approved `SCI-NOI-ODQ-101` on
`2026-08-29`: fixed-state conditional-sign is the ordinary conditioning family;
relearned methods are separate and unavailable pending complete approved
graphs; fixed and relearned members cannot be mixed in one uncertainty
estimate. The closure repair does not treat that family as one method:
PTC-to-frozen-MAP, PTC-to-frozen-JINC, realized-MAP, realized-JINC, and
filtered routes have distinct identities and all remain unavailable. GEN owns
member completion truth; the corrected NOI profile governs only UNC member
admission; the assignment-design identity and STD unit `1` are explicit.

Grant Wilson then approved `SCI-NOI-ODQ-102A` on `2026-08-29`. The ordinary
route applies the NOI-defined realization assignment at the exact PTC-to-MAP
numerical boundary. MAP may consume the modifier inline during its frozen
ordinary accumulation, so a materialized randomized timestream is not
required. NOI owns assignment, ensemble design, realization identity, and the
resulting NOI realization map; MAP owns only conforming application. The
result is not an ordinary MAP science product. The route remains numerically
unavailable until its exact PTC coefficient and numerical `coverage_cut` gates
are realized; all other routes remain unselected and unavailable.

Grant Wilson then approved `SCI-NOI-ODQ-102B` on `2026-08-29`: once a detector
receives its member assignment, that assignment applies to all admitted samples
of that stable realized detector/channel throughout the observation. Scan,
subscan, chunk, sample/time, traversal, worker, container, and MAP accumulation
order cannot change it; the same detector identity in another observation is a
different coherence unit.

Grant Wilson then approved `SCI-NOI-ODQ-102C` on `2026-08-29`: the ordinary
route uses network-stratified, coefficient-balanced randomized detector signs.
Detector coefficient mass is derived from exact frozen MAP-admitted positive
contributions and balanced separately inside each stable readout network; no
network, array, or observation balances another. Complement-symmetric
admission/probability preserves marginal detector sign probability `1/2`;
equal detector counts and post-conditioning independence are not claimed. The
derived design coefficient is not precision, empirical NOI weight, exposure,
validity, or a replacement MAP-facing coefficient.

Grant Wilson then resolved `SCI-NOI-ODQ-102D` on `2026-08-29` as an authorship
delegation. Exact finite-design mechanics and scientific rationale belong to
the implementation-blind scientific-contract author. The proposed tolerance-
conditioned network-local construction is approved only as a nonbinding
suggestion that the author may adopt, revise, or reject within ODQ-102A/B/C.
No numerical availability or advance owner acceptance follows; the author must
supply complete exact mechanics or return one precise question.

Grant Wilson explicitly approved `SCI-NOI-ODQ-104` on `2026-08-29`. Every GEN
method must classify every scientifically consequential adjacent reduction
state as fixed, rerun/relearned, not applicable, or unavailable. A relearned
method must identify the consequential stages rerun/relearned and the resulting
state that may differ from the real-observation reduction; a generic or partial
`relearned` label is insufficient. This is scientific method definition, not a
requirement for exhaustive implementation provenance. It selects no relearned
method and does not change the ODQ-102A ordinary route or its availability.

Grant Wilson approved the scientific content of `SCI-NOI-ODQ-103` on
`2026-08-29`: randomization is intended to suppress source signal, but does not
by construction alone establish that the resulting maps are source-free. The
full source-content, target, assumption, finite-residual, consequential-
operator-effect, structured-residual, source-model-error, leakage, and
prohibited-claim disclosures remain required. Exact scientist-readable
terminology is delegated to the Stage-B scientific author; the provisional
long label is nonbinding and may not be replaced by terminology implying
guaranteed source removal. No achieved suppression or numerical availability
follows.

Grant Wilson approved `SCI-NOI-ODQ-105A` on `2026-08-29`. Candidate assignments
rejected during finite-design construction are search outcomes, not failures or
members. Once admitted, every requested realization must complete through the
declared frozen operator. Any admitted-member failure fails the entire GEN
ensemble closed for every UNC use; completed survivors cannot be silently
retained as a partial ensemble. GEN reports diagnostically sufficient failure
cause/context without exhaustive implementation provenance. Disabled remains
explicit zero-member/no-work; inability to resolve the required admitted design
is a design-resolution failure rather than an individual candidate failure.

Grant Wilson approved `SCI-NOI-ODQ-105B` on `2026-08-29`. The initial UNC
estimand is the zero-centered conditional detector-sign-randomization second
moment `V_hat_cond(p)=sum_b omega_b M_b(p)^2`, using exact normalized
finite-design weights on the common domain where every admitted realization
has a valid finite value. The ensemble mean is not subtracted and no `B-1`
correction applies. Dependence, complement structure, counts, design rank,
effective information, and estimator uncertainty/unavailability must be
reported; realization count is not independent astronomical count or exposure.
The squared-signal-unit result retains source imprint and structured nonzero
content and is not automatically physical-noise variance or MAP covariance.
Square root, projection, off-diagonal covariance, inverse, weight, and STD
transformations remain separately governed. Numerical availability remains
blocked on the existing GEN/design/domain/adequacy gates.

Grant Wilson approved `SCI-NOI-ODQ-106` on `2026-08-29`. The ODQ-105B
pointwise conditional second moment is the ordinary primary uncertainty
representation but is not covariance merely because it is pointwise or
diagonal-like. Retained ensembles and separately identified projected,
stationary/kernel, structured, full, or unavailable covariance methods are
permitted; dense full covariance is never universally required. Every
covariance method must declare exact estimator/member population, domain,
support/response, rank or rank limit, null/unresolved modes, regularization,
omissions, and uncertainty/calibration state. Unreported covariance is unknown
or unavailable rather than zero or independence. The initial method retains
its common-all-member rule and cannot use pairwise populations, survivor
subsets, or generic missing-data estimators to rescue failure. No covariance
representation implies an inverse or precision.

Grant Wilson approved `SCI-NOI-ODQ-107` on `2026-08-29`. The initial derived
inverse product is `W_hat_cond=1/V_hat_cond` on the exact finite strictly
positive parent domain. Its role is an inverse conditional second-moment scale
with inverse squared signal units, not inverse variance or precision. Zero,
negative, nonfinite, unavailable, or outside-parent-domain input yields
unavailable rather than a numerical zero; any floor, cap, clipping, epsilon,
shrinkage, or other regularization is a separate method. Marginal inverse
variance requires separately authorized marginal variance; precision requires
an authorized covariance inverse/generalized inverse on a declared subspace;
and consumer-effective weight is exact-use-specific. None is validity,
support, exposure, a PTC/MAP coefficient, or a parent-mutation instruction.
Cross-boundary use requires explicit future scientific authority.

Grant Wilson approved `SCI-NOI-ODQ-108` on `2026-08-29`. The first STD method,
`NOI-STD/MAP-CONDITIONAL-SECOND-MOMENT-SCALE@1`, divides the exact immutable
normalized real-observation MAP signal associated with the same frozen MAP
operator state by canonical `sqrt(V_hat_cond)` on their exact compatible
finite-positive valid-domain intersection. Exact estimator/generation,
parent, response, unit/beam, WCS, support, validity, lifecycle, transformation,
and numerator/scale dependence are bound. Invalid or incompatible scale is
unavailable rather than zero/infinity; interpolation, substitution, and an
implicit algebraic inverse-scale route are prohibited. The unit-`1` output
claims only MAP signal standardized by the stated conditional randomization
second-moment scale—never significance, probability, detection, completeness,
purity, or catalog authority. JINC remains a separate future method with a
JINC-specific compatible scale.

Grant Wilson approved `SCI-NOI-ODQ-109` on `2026-08-29`. SCI-NOI admits three
plan-selected modes: persisted ensemble, compact deterministic regeneration,
and streaming sufficient statistics. Requested/effective/applied/realized mode
is explicit; there is no universal default or silent fallback. Compact mode
binds exact parent/method/algorithm/operator/design/membership/key/configuration
identity and declares byte-identical or numerical reproducibility. Streaming
mode retains mathematically sufficient state for every published product/claim
and declares unsupported later reconstruction/reanalysis. ODQ-105A remains
absolute: every admitted member completes, failed ensembles yield no survivor
or partial streaming estimate, and partial accumulators carry no UNC authority.
Required persistence failure is product failure; planned transience is not.
Persistence/regeneration establishes no adequacy, covariance completeness,
calibration, significance, conformity, performance, readiness, or production
authority.

Grant Wilson approved `SCI-NOI-ODQ-110A` on `2026-08-29`. NOI does not choose
or define a deterministic filter or other scientific transformation. The
appropriate upstream/downstream scientific process owns the transformation and
the transformed scientific product. NOI must bind and apply exactly that
transformation to every admitted compatible randomization when estimating
uncertainty for that exact transformed product. Transformation authority,
version/state/parameters/order/domain/support/edge/missing-data/normalization/
unit/response/lifecycle/failure identity and parity are mandatory; omission,
substitution, relocation, inferred equivalence, and cross-product reuse are
prohibited. Member-specific transformation relearning is a separate ODQ-104
method. Every transformed route remains numerically unavailable until its owner
supplies exact content-bound authority and the NOI parity interface is
satisfied.

Grant Wilson approved `SCI-NOI-ODQ-110B` on `2026-08-29`. A Wiener
transformation learned once by its scientific owner and frozen before
application to randomizations follows ODQ-110A even when data-derived. If an
NOI product is used by that owner to learn, select, or update the
transformation, the prior UNC input, owner-learning generation, resulting
transformation, transformed science product, transformed GEN, and successor UNC
are separate immutable generations with explicit dependence. The prior UNC is
not independent evidence validating the successor and cannot be mutated.
Per-realization Wiener learning is a distinct ODQ-104 method and cannot mix
with fixed-Wiener members without an authorized mixture estimand. All numerical
Wiener routes remain unavailable pending their exact owner authority,
inference/relearning contract where applicable, and NOI boundary.

Grant Wilson approved `SCI-NOI-ODQ-110C` on `2026-08-30`. FRUIT retains
scientific authority over source modeling, subtraction/add-back, recurrence,
learning, stopping/restart/selection, response, support, validity, lifecycle,
failure, and interpretation. An exact fixed FRUIT residual or terminal
transformation may support only uncertainty conditional on frozen FRUIT state
under ODQ-110A parity. If an NOI product informs later FRUIT continuation, the
prior input, FRUIT-learning generation, transformation, science product, GEN,
and successor UNC remain separate immutable generations; the prior product is
dependent input, not independent validation. Partial or complete per-member
FRUIT replay is a distinct ODQ-104 method. Fixed and replayed members cannot
mix without an authorized mixture estimand. All numerical FRUIT routes remain
unavailable pending exact FRUIT-owner and NOI boundaries.

Grant Wilson approved `SCI-NOI-ODQ-111` on `2026-08-30`. The four exact NOI-
owned GEN-input, UNC-member, UNC-ensemble, and STD admission profile identities
and consumer actions are approved. Producer facts remain producer-owned, NOI
owns each named-use policy/action, and SCI-VAL may only bind and evaluate the
approved immutable bytes. No generic pass, implicit next-operation realization,
or cross-use veto/rescue follows. All bounded SCI-NOI Stage A owner decisions
are complete. Paired immutable SCI-VAL successors dated `2026-08-30` now bind
the exact r0.18 source packet and register the four profiles while preserving
all earlier MAP/JINC records. The process-only Registry/source prerequisite is
satisfied. Grant Wilson approved the exact r0.18 author manifest, r0.19 closure
report, paired Registry successors, and binding record on `2026-08-30`, and
authorized a fresh implementation-blind SCI-NOI Stage B author at high
reasoning effort. The author is restricted to the 17 manifest-admitted objects;
no implementation, evidence, manager record, or unlisted source is admitted.
MAP/pre-MAP numerical parents remain
unavailable pending the exact PTC MAP-facing coefficient and owner-admitted
numerical `coverage_cut`; JINC numerical parents remain unavailable under its
frozen gates. No frozen parent was modified, and no implementation conformity,
empirical calibration, physical-noise validity, significance, achieved
performance, readiness, or production claim is made.

On `2026-08-30` Grant Wilson launched
[`SCI-FLT v0.1`](scientific_contracts/packages/SCI-FLT/v0.1/README.md) as a
recovery-first Stage A scientific-contract effort from that authority line.
The initial owner-review packet included a package-specific prior-work
record, quarantined implementation-informed dossier, sanitized Scope Brief,
ownership/boundary classification, deterministic-transform science extract,
typed operator/product taxonomy, and eight bounded owner questions. It separately classifies fixed
deterministic convolution/low-pass, Wiener/noise-model-dependent,
matched/template-amplitude, source-sensitive, data-thresholded map-domain, RTC
temporal, and FRUIT feedback operations. Approved SCI-NOI Stage A controls:
FLT defines the exact transformation and NOI applies it to compatible admitted
randomizations for uncertainty of the exact transformed product; fixed-state,
successor-generation, and per-member-relearned routes remain distinct. Current
SCI-NOI Stage B draft material was excluded.

The final owner scope repair on `2026-08-30` retains `SCI-FLT` as the tranche,
selects `SCI-FLT-FIXED` as the first package, rejects `SCI-FLT-DET` because of
the detector-namespace collision, and retains `SCI-FLT-INF` only as a non-
authoritative holding tranche. Base v0.1 is strict-linear same-grid
`y=J_full L_Theta m`, fixed convolution is its concrete family, fixed low-pass
is a qualified subtype only with complete transfer facts, and full-footprint-
only is the sole edge/missing method. Affine offsets, boundary extension,
truncation, support renormalization, reprojection, inference-bearing state, and
coaddition are deferred. Exact MAP/JINC/NOI boundaries, decision tables, VAL
profile drafts, product/lifecycle roles, and a content-bound owner record now
form the repaired 17-object SHA-bound author candidate. All bounded Stage A
scope decisions are resolved. The exact bytes require owner approval and Stage
B has not begun. No algorithm or frozen authority changed, and no implementation
conformity, validation, calibration, achieved response/performance, readiness,
production, freeze, Unity, source/mode, NOI, or FRUIT action is claimed or
authorized.

The first pilot, [`SCI-CAL`](scientific_contracts/packages/SCI-CAL/v0.1/README.md),
completed its initial Stage A recovery and sanitized scope draft on 2026-08-16.
The recovery reuses the frozen implementation-independent CAL core and later
owner decisions, incorporates the layered APT-identity supersessions, and
keeps implementation, audit, repair, validation, Unity, and active ALIGN B3c
material outside authorship. Grant approved the v0.1 Scope Brief, its five
named scope decisions, and the four-item sanitized author-reference packet on
2026-08-16. The next gate is a fresh implementation-blind GPT-5.6 Ultra
scientific-author draft from that packet. Scope approval does not approve the
contract substance, establish implementation conformity, authorize
validation, or change production status.

The content-bound author packet was committed and a fresh GPT-5.6 Ultra
implementation-blind author was dispatched on 2026-08-16. The author is
restricted to the approved Scope Brief, independent core plus supersession
cover, exact passband manifest, and sanitized conventions/ownership extract.
The internal dossier, implementation, audits, repairs, tests, validation,
Unity evidence, and active ALIGN work are excluded. The next owner-facing
milestone is a manager-reviewed two-view contract draft or a precise scientific
decision that prevents one.

The implementation-blind author returned the first SCI-CAL two-view draft on
2026-08-16. The package contains one shared LaTeX authority, a 24-page
scientist-facing rationale, a 21-page engineering conformance view, 50 shared
numbered requirements, 30 shared limiting/pathological predictions, and a
complete 50-row crosswalk. The manager verified that the engineering view adds
no independent normative science and corrected one draft ambiguity so
`science-qualification-eligible` cannot be mistaken for an achieved
`science-qualified` or `calibrated-science` claim. Both PDFs passed mechanical
coverage checks and visual inspection. No scientific validation was run.

The draft is not frozen. `SCI-CAL-OWNER-Q001` remains open because the approved
packet does not contain the retained atmosphere operator's exact content-bound
nodes and ordinates, ordinate orientation, numeric support/seam rules, or
generating-model/passband provenance. Until one immutable record supplies
those facts, numeric atmosphere evaluation, calibrated numeric output,
representation-fidelity claims, and numeric science-qualification eligibility
remain unavailable. The next gate is owner scientific review; an independent
implementation-blind two-view consistency review follows only after the
science is approved.

Grant returned a complete science-team assessment of the 24-page rationale on
2026-08-16: major revision required, scientific core strong. The manager
rewrote the scientist-facing view as revision 0.2 rather than adding a summary
to the formal document. The new 14-page PDF has an approximately ten-page
physical narrative followed by formal appendices. It places the once-only
calibration equation first; adds calibration-lineage, factor-role,
uncertainty, validity, and validation explanations; moves exact hashes and
formal routing to appendices; and leaves the engineering v0.1 contract
unchanged as normative conformance authority.

The review exposed additional upstream scientific authority gaps rather than
authorizing invented answers. The rationale r0.2 register contains Q01--Q09:
physical `xs` meaning; baseline and pipeline ordering; `flxscale` derivation;
calibrator-to-target transfer; broadband photometric convention; the complete
numeric atmosphere operator (formerly Q001); opacity/segment policy rationale;
available numerical uncertainty products; and achieved-science evidence
criteria. The current durable convention maps `nw10` to `a1400`; no stronger
durable source classifying it as nonexistent or reserved was found, so the
science narrative omits the unnecessary roster and records the evidence
conflict for owner action. No implementation audit or scientific validation
was run. The package remains not frozen pending owner science decisions and a
subsequent fresh implementation-blind consistency review.

Grant then returned bounded rationale r0.3 corrections on 2026-08-16. The revised
rationale assigns source-calibration meaning to Beammap/source-APT
production, limits TolProj to target/source association and explicitly
approved child-APT transformations, limits SCI-CAL to applying the selected
child `flxscale` once plus target atmosphere, and assigns realized
mapmaker/filter response to MAP/FLT. It distinguishes scalar temporal
commutation, detector-mixing operations, and sample-dependent atmosphere;
uses an orientation-neutral atmosphere ordinate; removes unsupported `xs`
scope/layout claims; and adds the required Q01--Q09 owner-decision ledger,
r0.3 crosswalk, change log, and consistency report. No engineering
requirements, numerical science, implementation assessment, or validation
claim changed.

The final owner voice review judged r0.3 essentially finished and authorized
only a five-item cleanup: separate contract v0.1 from rationale r0.3; remove
the title-page production note; apply three sentence-level clarifications;
clarify the adopted opacity thresholds and Q06 consequence; and rename the
uncertainty table's status column. The live decision ledger now separates
resolution authority and date and records affected documents, and the PDF
decision snapshot is checked against it. The rationale architecture and
library house standard are frozen as the model for later packages, including
the v/r version-axis rule. There is no further stylistic round. SCI-CAL
scientific authority remains a draft pending owner disposition of the open
scientific questions and the final consistency gate.

After the owner directed the program to proceed, the second pilot
[`SCI-MAP`](scientific_contracts/packages/SCI-MAP/v0.1/README.md) completed
Stage A prior-work recovery on 2026-08-16. The recovery reuses the frozen
implementation-independent ordinary MAP-001 core and the later owner-approved
whole-bundle, nonprecision-coefficient, centered-integer-coadd, support,
validity, and raw-parent decisions. It also found and classified later
MAP-002 integration/ownership work and MAP-003 tracer-parent/implementation
work that postdate the initial registry. JINC and OOF residual transfer remain
separate scientific estimators and are not silently absorbed into ordinary
SCI-MAP v0.1.

The SCI-MAP package now contains the charter-linked recovery record, an
internal dossier, a sanitized Scope Brief, and author-only
supersession/convention extracts. Grant approved MAP-SCOPE-D001--D006 and the
exact three-part author packet on 2026-08-16. The approved packet reuses the
frozen MAP-001 core rather than repeating its derivation and keeps all
implementation, audit, repair, validation, Unity, MAP-002, MAP-003, and
production material outside the author channel. The next gate is a fresh
implementation-blind GPT-5.6 Ultra scientific-author draft from the exact
content-bound packet. Scope approval does not approve contract substance,
implementation conformity, validation, or production use.

The content-bound SCI-MAP author packet was committed and a fresh GPT-5.6
Ultra implementation-blind author was dispatched on 2026-08-16. The author is
restricted to the approved Scope Brief, exact independent core plus
supersession cover, and sanitized conventions/ownership extract. The internal
dossier, recovery record, implementation, audits, repairs, tests, validation,
Unity evidence, MAP-002/MAP-003 evidence, and production state are excluded.
The next owner-facing milestone is a manager-reviewed two-view contract draft
or a precise scientific decision that prevents one.

The implementation-blind author returned the SCI-MAP Stage B r0.1 draft on
2026-08-16. It contains one shared canonical LaTeX authority, a 22-page
scientist-facing rationale with a 12-page main narrative, an 18-page
engineering conformance view, 52 shared numbered requirements, 25 shared
falsifiable predictions, a complete crosswalk, and an exact seven-question
owner-decision register. The manager required explicit support-authorized row
selectors for map and coadd operators so unsupported storage rows cannot be
misread as zero-valued scientific output, and required the exact owner register
and compact traceability summary to appear in the scientist-facing PDF.

Mechanical checks pass for sequential identifiers, crosswalk and PDF coverage,
owner-ledger identity, support-row semantics, and absence of independent
normative science in the engineering view. Both PDFs compile without LaTeX
warnings, and all 40 rendered pages passed independent visual inspection. No
implementation candidate was inspected, and no validation, reduction, Unity
execution, or production decision occurred. The draft is not frozen.
`SCI-MAP-OD-001--007` remain open for threshold rationale and change authority,
response-unavailable use, covariance persistence, physical observation-map
publication, registered Pointing/OOF reuse, and the numeric domain, unit status,
boundary cases, and failure behavior of `coverage_cut`. The next gate is owner
scientific review; a fresh implementation-blind two-view consistency review
follows only after the science is approved.

The first SCI-MAP scientific editing round was integrated on 2026-08-16. The
r0.1 combined rationale/contract was preserved as a formal
scientific/engineering contract, while a separate science-team rationale r0.2
now follows the SCI-CAL house model and omits the full 52-requirement and
25-prediction inventories. The r0.2 rationale adds the fractional-projection
and complete-bundle coadd teaching figures, translates the eight map facts for
science users, and makes upstream ownership and claim layers explicit.

Dimensional review found `SCI-MAP-CI-001`: the accepted threshold equations
force `coverage_cut` to be dimensionless, while the r0.1 normative clauses
left its unit status open. On 2026-08-16 the scientific owner approved the
bounded amendment, and r0.3 incorporates it in the shared equations,
REQ-031/032, and PRED-012. OD-007 now remains open only for numerical-domain
and failure-policy questions. Existing OD-001--007 remain stable in identity; OD-008 appends the
unresolved projection-normalization/boundary authority and OD-009 appends the
unresolved canonical-grid preparation and future reprojection/mosaicking
ownership. Approved PTC D004 supplies coefficient meaning, and accepted ADR
0009 plus its 2026-08-05 amendment supplies WCS/FITS and the 0.1-arcsec
serialization authority, so neither resolved issue was reopened. The r0.3
science-team rationale is now the frozen SCI-MAP house version; it receives no
further stylistic round. No implementation inspection, validation, reduction,
Unity execution, or production decision occurred.

The contract-library package layout was normalized in the same documentation
pass after the project owner identified drift from the required bundle shape.
SCI-CAL and SCI-MAP now both expose `README.md`, `SCOPE_BRIEF.md`,
`DECISION_LOG.md`, `CROSSWALK.md`, all six canonical `src/common/` modules,
stable `src/scientific-rationale.tex` and
`src/engineering-conformance.tex` entry points, and stable versioned PDF
filenames. A small `doc/scientific_contracts/verify_layout.py` check prevents
future structural drift. The packages remain scientific contract v0.1;
filesystem normalization does not silently create v1.0 authority.

The CAL/MAP pilot lessons are now durable in
[`PILOT_PROCESS_REVIEW_2026-08-16.md`](scientific_contracts/PILOT_PROCESS_REVIEW_2026-08-16.md).
That review makes prior-work recovery, exact author-packet content binding,
Stage A/Stage B separation, one shared normative core, independent consistency
review, full rendered-PDF QA, and the four-trigger stopping rule mandatory for
later packages. It also confirms that SCI-BEAM may enter Stage A while remaining
separate from active ALIGN/AST work; this does not approve BEAM science or
physical timing/absolute-placement claims.

[`SCI-BEAM`](scientific_contracts/packages/SCI-BEAM/v0.1/README.md) completed
its Stage A recovery and sanitized scope draft on 2026-08-16. Recovery found no
dedicated approved implementation-independent BEAM core, so it preserved prior
scope, three historical dependency handoffs, current Citlali conventions,
TolAPT soft-prior ownership, and `toltec_beammap` downstream ownership without
promoting any of them into a scientific contract. Grant approved the Scope
Brief and `BEAM-SCOPE-D001--D012` on 2026-08-16. The exact three-part packet is
now content-bound: the approved Scope Brief, a sanitized conventions/ownership
extract, and a primary-reference boundary admitting only bounded context from
Bryan et al. 2018 and Golec and the TolTEC Collaboration 2024. No
analogue-instrument methodology paper, code, audit, handoff, test, validation,
current A/B or production status, or active ALIGN/AST material enters the
author channel. This approval authorizes a fresh implementation-blind author;
it does not accept the resulting contract, assess conformity, authorize
validation, or change production status.

The implementation-blind SCI-BEAM author subsequently completed document
revision `r0.1`, and the contract manager completed the bounded first review.
The draft has one shared normative core containing 46 sequential requirements
and 24 sequential falsifiable predictions, an exact 70-row crosswalk, a
13-page scientific rationale with nine substantive pages before its
appendices, and a 9-page engineering conformance view with no independent
normative science. The manager returned three bounded corrections to the same
author: exact reference-origin rather than unjustified unit-peak
normalization, modulo-pi and availability-aware convergence with separate
candidate/support/valid-detector stability, and sufficient substantive
rationale length. All were corrected without adding implementation-informed
input. The author packet hashes remained unchanged. The manager supplied the
already-approved `BEAM-SCOPE-D001--D012` ID mapping after author freeze,
closing one traceability-only question. Seven scientific-owner questions
remain open concerning later numerical policies, singular covariance,
model-inadequacy diagnostics, calibration-candidate compatibility,
response-completeness interpretation, and successor model families. The draft
is ready for scientific-owner review, then a fresh implementation-blind
consistency review; it is not accepted or frozen and makes no conformity,
validation, observational-performance, or production claim.

On 2026-08-17 the scientific owner issued a substantive SCI-BEAM r0.2
revision directive after deeper scientific review. The directive supersedes
conflicting r0.1 draft language while leaving contract version v0.1, the
original author packet, and its hashes unchanged. R0.2 defines the primary fit
as a standardized per-detector Beammap in raw fractional frequency shift
`Delta f/f`; defines the fitted tensor as the observation-local effective PSF
core rather than an intrinsic or complete beam; uses a fixed nominal-beam,
top-of-atmosphere reference-origin source amplitude with no additional
finite-source dilution factor; makes SCI-BEAM the desired authority for the
complete Beammap APT, accepted source-APT `flxscale`, and scan-domain NEFD-like
`sens`; requires full WCS metric, 2-D tensor, model Jacobian, joint covariance,
broadening, raw/horizon geometry, conventional-pivot limitations, exact
same-APT pointing transfer, independent quantity states, and empirical
map/residual companions; and deprecates `responsivity` as noncanonical.

The r0.2 authoring lane is explicitly independent of current code, APT
contracts/files, audits, repairs, tests, and production reductions. It produces
a separate science-team rationale and Formal Scientific/Engineering Contract,
retains 46 requirement and 24 prediction IDs with an exact r0.1 disposition
map, and records cross-package changes for later governed amendments rather
than editing CAL, MAP, AST, TolProj, weighting, or kernel authorities now.
Nine owner decisions remain open for sensitivity estimator details,
model-adequacy/wing policies, required science accuracy, physical pivot
registration, and kernel qualification. The planned final substantive step is
an r0.3 owner-voice/presentation pass, after which architecture freezes unless
a governed evidence or inconsistency trigger applies. No implementation,
validation, or readiness claim follows from r0.2 authorship.

The r0.2 document-consistency gate completed on 2026-08-17. The formal
contract contains all 46 requirement and 24 prediction IDs and the exact
70-row crosswalk; the separate rationale contains no formal ID inventory.
Both sources compile without warnings. The canonical PDFs are 17 pages formal
and 9 pages rationale, and all 26 Poppler-rendered pages passed visual review.
The approved three-item author packet hashes remain unchanged. R0.2 is ready
for owner review but is not accepted or frozen.

The scientific owner's final bounded SCI-BEAM review completed on 2026-08-17.
Document revision r0.3 makes NEFD-like `sens` strictly positive through
`abs(flxscale)` while preserving signed `flxscale`; separates map-fit
Jacobian/covariance from derived calibration and sensitivity propagation;
defines the centroid-to-detector sign/frame transformation; derives effective
rotation from parent-sample contribution support propagated through fit
support; requires the same immutable APT artifact for pointing/science
transfer; restores a concise soft-prior/convergence explanation; and binds the
three document-facing decision groups `SCI-BEAM-OD-001--003` to nine atomic
open ledger questions. The source flux is stated directly in TOA
mJy per fixed nominal beam without argumentative unit or duplicate-factor
language, and the rationale uses a compact contents list.

After final source/PDF consistency and rendered-page QA, SCI-BEAM v0.1/r0.3
is the frozen implementation-independent scientific authority. Its 46
requirement and 24 prediction IDs remain stable. The original author packet
and hashes remain unchanged; no implementation, APT storage, audit, repair,
test, reduction, or production behavior was inspected. Implementation
conformance, representation fidelity, observational performance,
science-impact qualification, and production readiness remain unassessed.
There is no further editorial round; only the contract-library's four governed
revision triggers may reopen the package.

After the project owner selected raw-timestream conditioning as the fourth
contract product, [`SCI-RTC`](scientific_contracts/packages/SCI-RTC/v0.1/README.md)
completed its initial Stage A recovery and sanitized scope draft on
2026-08-17. Recovery reuses the frozen implementation-independent RTC core,
approved D001--D004, the phase-zero point-selection amendment, and the
owner-approved learned-sampling design. It also reconciles those sources with
the later frozen SCI-BEAM raw `Delta f/f` boundary and conditional CAL, MAP,
and PTC interfaces rather than repeating their reasoning.

The v0.1 scope and exact three-item author packet were owner-approved on
2026-08-17 with one modification to `RTC-SCOPE-D004`. For raw donor `q` and
target `d`, valid compatible factors under `z_i = flxscale_i x_i` authorize
raw donor scale `flxscale_q / flxscale_d`; both factors must be valid for the
exact detector occurrences under the same convention/domain, and the target
factor must be nonzero. This does not restore a scientific role for legacy
`responsivity`.

The approved v0.1 scope preserves product-role-specific raw and calibrated
signal domains; exact unit-domain calibration/replacement order; transitive
synthesis/replacement ineligibility; complete-response-or-unavailable;
immutable stage identity; phase-zero sampling; fixed and optional learned
sampling; and one atomic RTC output bundle. Implementation, audits, handoffs,
repairs, re-audits, tests, validation, Unity evidence, active ALIGN work, and
production state remain quarantined. Stage B implementation-blind drafting is
now authorized from the exact content-bound packet. Eight downstream
scientific questions remain explicitly open. No RTC scientific authority,
implementation conformity, validation, or production claim has been
established.

The fresh implementation-blind author completed SCI-RTC Stage B v0.1/r0.1
from the exact approved packet on 2026-08-17. The manager-reviewed draft has
one six-file normative core, 20 definitions, 24 displayed equations, 12
bounded assumptions, 54 sequential requirements, 26 falsifiable predictions,
and an exact crosswalk. The scientist-facing rationale is 25 pages with ten
substantive pre-appendix narrative pages; the engineering view is 17 pages and
contains no independent normative mathematics. The owner-modified raw donor
rule appears literally as `flxscale_q / flxscale_d`, with exact occurrence and
domain validity, nonzero target denominator, unavailable fallbacks, and a
numerical direction falsifier. The prior eight broad question families are
decomposed into 23 open, one conditional, and four deferred owner entries.

Both PDFs compile without TeX warnings or errors. Independent manager rebuilds
match page counts and page text, and all 42 final pages passed 144-dpi Poppler
inspection. Packet hashes and author write boundaries pass. The draft is ready
for scientific-owner review but is not approved or frozen; implementation
conformity, representation fidelity, observational performance, validation,
science-impact qualification, and production readiness remain unassessed.

On 2026-08-18 the scientific owner directed an implementation-blind r0.2
revision that makes learn--resolve--immutable-apply the RTC organizing
principle. The completed revision defines operation purpose and signal model,
projected scan/beam temporal response, notch width/depth and adaptation,
low/high/band-pass meaning, constrained FIR order/taps, donor continuity and
physical limits, decimation/alias control, temporal registration, and complete
RTC-plan calibration compatibility. It preserves the approved
`flxscale_q/flxscale_d` donor convention, adds a same-Beammap circular-factor
prohibition, and imports no implementation evidence or production values.

The r0.2 shared core now contains 26 definitions, 30 equation tags, 12
assumptions, 70 requirements, and 38 predictions. The 33-page science-team
rationale has 12 substantive narrative sections and five diagrams; the
engineering view is 24 pages and contains no independent normative
mathematics. Thirty-six owner entries retain 31 open, one conditional, and
four deferred choices. Both PDFs compile without warnings; all 57 pages passed
Poppler inspection. The package remains a draft for scientific-owner review;
scientific approval, implementation conformity, validation, science
qualification, and production readiness remain unassessed.

On 2026-08-18 the scientific owner issued the targeted SCI-RTC r0.3 bounded
iterative notch-plan refinement directive. The completed implementation-blind
revision defines a finite outer sequence of immutable learn--resolve--apply
cycles, complete cumulative successor plans, default evaluation and final
replay on the original admitted input, and separately authorized cascade
semantics. Successor learning compares original, predicted, and conditioned
spectra; distinguishes hidden candidates from filter artifacts; preserves
cumulative scientific budgets; and terminates with an explicit accepted,
rejected, nonconvergent, or maximum-cycle disposition. It does not introduce
online adaptation or inspect current implementation evidence.

The r0.3 shared core appends three definitions, one equation tag, twelve
requirements, and eight predictions without renumbering prior authority. The
complete inventory is now 29 definitions, 31 equation tags, 12 assumptions,
82 requirements, and 46 predictions. Fourteen new owner decisions bring the
ledger to 50 entries: 45 open, one conditional, and four deferred. Both PDFs
compile without warnings; the rationale is 38 pages and the engineering view
29 pages, and all 67 Poppler-rendered pages passed visual inspection. The
package remains a draft for scientific-owner review; scientific approval,
implementation conformity, validation, science qualification, and production
readiness remain unassessed.

On 2026-08-18 the scientific owner approved two r0.4 boundary decisions and a
bounded correction pass. RTC now remains raw `Delta f/f` through replacement,
temporal conditioning, and phase-zero sampling; compatible
`flxscale_q/flxscale_d` is raw donor convention transfer, while absolute
`flxscale` and target-atmosphere correction belong to a later SCI-CAL handoff.
Directly selected ALIGN-synthesized or RTC-replaced occurrences are universally
excluded, while RTC preserves noncenter transitive influence and each
downstream consumer owns its eligibility policy. R0.4 also separates refinement
attempts from accepted plans, defines the initial evaluation product, corrects
phase-zero selection to the final pre-decimation stream, restores the
Learn--Resolve--Apply title and signal vocabulary, and adds a role-specific RTC
plan matrix. The 29-definition, 31-equation-tag, 12-assumption, 82-requirement,
and 46-prediction inventories retain their stable identifiers. The ledger now
contains 44 open, two resolved, and four deferred entries. Both PDFs compile
without warnings; the rationale is 39 pages and the engineering view 28 pages,
and all 67 final Poppler-rendered pages passed visual inspection. The package
remains a draft for scientific-owner review and freeze disposition; no
implementation conformity, representation fidelity, validation, science
qualification, or production-readiness claim is made.

Current SCI-RTC status supersedes the draft-stage statements above. Subsequent
bounded owner-directed revisions advanced the package through r0.12, which
Grant froze on `2026-08-21` with 52 definitions, 44 equation tags, 12
assumptions, 143 requirements, 108 predictions, and 103 owner entries. On
`2026-08-25`, the WP-7 scientific-owner disposition approved a source-resolved
explanatory correction: only the rationale's `OWNER-090--096` paraphrase
ordering now changes to match the controlling ledger. The shared normative
core, engineering source, ledger, crosswalk, numerical behavior, and retained
open states are unchanged. Both PDFs were rebuilt and rebound under
`SCIENTIFIC_OWNER_SOURCE_CORRECTION_2026-08-25.md` and
`SOURCE_MANIFEST_CORRECTED_2026-08-25.md`; package verification and all-page
Poppler QA pass. This publishes authority for a new WP-7 clean-room successor
only and establishes no implementation, validation, performance, production,
MAP, or finding-closure claim.

On `2026-08-26`, the source-commit-`170ecea9d` WP-7.1 successor audit and its
locked regression comparison completed. The independent audit found `TS-A`,
`TS-S`, and `TS-C` ready and the RTC-only terminal route contract-closed. The
comparison classified nine normalized predecessor issues as closed by the
approved successor authority, with zero regressions, recurrent findings, new
successor findings, or unresolved contradictions. Grant recorded the
[scientific-owner closure](scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_170ECEA9D/WP7_SCIENTIFIC_OWNER_CLOSURE_2026-08-26.md),
which closes WP-7 scientifically at its approved contract scope. The seven
retained limitations, including unavailable `TS-R`, `TS-U`, and `TS-T`, are
scope statements only and are not WP-7 findings, owner questions, or repair
requirements. Implementation conformity, observational validation, achieved
performance, and production readiness remain separate evidence programs; no
further WP-7 repair or audit cycle is authorized by this closure.

The previously approved `DOC-MAP-001` deliverable is the first queued user
guide. It will combine a canonical map dictionary with a compact per-reduction
rendering driven by effective and realized state, explaining what each emitted
map measures without duplicating its provenance package. MAP-, NOI-, and
filter-related work contributes stable semantics to this guide, which must
close before production expansion and Phase 5 closeout. The queue entry itself
changes no numerical behavior, output selection, or audit scope.

## 2026-08-05 SCI-MAP-001 Application Integration Candidate

The final independent re-audit at
`8fc716557ca78b0d220200a92be46fa3545797e9` and the final canonical
coordination candidate at
`c7bb0214edfd57fddf31165923f08784dfd1b8c9` accept the bounded
`SCI-MAP-001` scientific contract at exact application source
`af0c849ce59a5f80e5efc8db435bb6662863052f`. Within that scope the contract
is approved, the implementation is conformant, validation is complete within
the local plus owner-accepted bounded evidence scope, and the bounded verdict
is `accept`.

F001--F011 are closed. F012 is
`closed_bounded_owner_accepted` only for the exact-`ed28dafb` external
execution/completion, returned product/inventory, visible observation/coadd,
and SEQ/OMP claims. Its retained limitations are the absent independent raw
manifest and sample ledger, scan-farm pre-normalization planes and commit-order
trace, wrapper/Slurm/environment/collection/retrieval chain, and historical
same-case S-X observation-realization files. F013 remains `open_conditioned`
on `SCI-ALIGN-001`, `SCI-CAL-001`, `SCI-AST-001`, `SCI-PTC-001`, and
`SCI-VAL-001`.

The coordinator-directed 2026-08-05 application-integration task separately
authorizes this MAP candidate for application integration. The dedicated
`codex/integrate-sci-map-001` branch was created from exact canonical
application base `9aae0e669384c5c0c0dda93debc194d6b8dac787` and advanced only
by fast-forward through `ed28dafb3`, `1b824f138`, `02b9eb303`, `f84b9fd7d`,
and `af0c849ce`. The excluded convolve/noise candidate
`02a198cbfb379eaf6ab279c5a3d44ee73ff90435` is not in that ancestry. Before
the integration records were edited, the branch tree was exactly the
`af0c849` application tree, `47aa745554e47514398e72d579625484abdcb79e`.
The branch-tip child of `af0c849` is a documentation-only integration commit
that changes this status, the integration ledger, and the dated
[application-integration handoff](../handoff/SCI-MAP-001_APPLICATION_INTEGRATION_DECISION_2026-08-05.md);
it is not a later application-source revision.

`codex/refactor-mainline` remains unmoved at `9aae0e669`; the integration
branch is a committed candidate for owner/coordinator review and later
fast-forward. Production remains `existing_use_only`, no upstream dependency
is closed by MAP acceptance, and neither production expansion nor Conan-lane
import is authorized by this candidate.

## Historical 2026-07-31 SCI-MAP-001 Bounded Repair Lane

This section preserves the repair-lane chronology and its candidate-time gate
states. Statements below about a then-pending final re-audit are historical
evidence, not current instructions or current package status; the application-
integration disposition above supersedes them as live state.

The project owner approved a bounded repair of `SCI-MAP-001` findings
F001-F011 on `codex/repair-sci-map-001`, created directly from governing
application source `9aae0e669384c5c0c0dda93debc194d6b8dac787`. The audit and
coordination lines remain read-only authorities. The convolve/noise candidate
`02a198cbfb379eaf6ab279c5a3d44ee73ff90435` is deliberately excluded and does
not land first.

The repair scope is the accepted ordinary-naive, array-grouped Stokes-I
successor contract: contract-derived fixtures; typed exposure, count, support,
and validity state; atomic full-precision map-bundle admission followed by one
admission commit phase; centered integer common-grid embedding with `L = I`;
preservation of the existing `Q += u`, `N += u * signal`, and
`K += u * kernel` operation
order; nonprecision coefficient labeling; the eight distinct F010 products,
compatibility aliases, explicit absence rules, and lossless realized
provenance. [ADR 0009](adr/0009-science-map-bundle-admission-and-validity.md)
and [the scientific conventions](SCIENTIFIC_CONVENTIONS.md) record the durable
meaning.

The candidate records the closed pre/post observation/coadd coefficient
stages, freezes a validated raw F010 snapshot before filtering, and carries
that immutable input through filtered signal, coefficient, F010, and alias
HDUs with matching lossless `RAWPDGST` identity. Unsupported JINC,
detector-grouped, and other non-v1 profiles retain their established legacy
coadd arithmetic with explicit successor-product absence and no F009/F010
claim.

The ordinary primitive uses one detector/sample order for sequential and
requested-parallel calls. Concurrent scan commits are serialized and governed
by `within-scan-exact-scan-farm-2gamma-n-sumabs-v1`: binary64 planes are tested
against long-double per-scan sums at the pre-registered
`2 * gamma_n * sum(abs(scan_value))` bound, while integer fact planes are
exact. The raw-execution read census is advanced only for the new immutable
science identity's existing kernel and separate-polarimetry state reads; it
adds no raw configuration authority.

The local candidate snapshot passes the required implementation gates without
required-data skips or unexpected error-level records: all five requested
build targets complete; CTest executes 588/588 enabled tests successfully
(one pre-existing disabled test); the focused science-map executable passes
29/29 contract, provenance, and equation tests; its ThreadSanitizer build
passes 7/7 repaired-primitive tests; 147 baseline-tool tests pass; and the full
config preflight passes 127 unit tests, all four mode kits, all eight compact
compatibility cases, 100% compact-surface coverage, and every typed boundary
audit. The classified raw-execution census remains 45 records with zero
review-required entries at digest
`09572da976aec89d56506394420b478426a6efbd0942c864571a8f6f311da2f8`.
The successor validation epoch and product registry parse and list cleanly.
Those initial local repair-candidate results did not themselves satisfy F012
or the independent re-audit.

This lane does not authorize general reprojection, interpolation, GLS,
covariance regularization, new defaults, or changes to RTC, PTC, JINC,
noise-realization, convolve, Wiener, source-fitting, Pointing/OOF, Beammap, or
fruit-loop algorithms. Historical accepted products and profiles retain their
original versioned contracts. Contract status is `approved`; implementation
remains `nonconformant`, validation `in_progress`, production
`existing_use_only`, verdict `amend`, and re-audit `required`. F009 and F010
remain `addressed_pending_reaudit` until a fresh independent disposition.

The human evidence owner executed all seven exact-SHA
`SCI-MAP-001-UNITY-001` cases on 2026-08-03 and returned the corpus locally.
The 2026-08-05 read-only reconciliation binds every captured executable and
reduction index to candidate `ed28dafb37f9113c0d3c95297148157129a90886` and
records the exact product inventory, evidence limitations, missing `S-X-SEQ`
observation-level realization serialization, and typed-WCS/Stokes discrepancy
in the campaign closeout note. Do not repeat the campaign. The later owner
amendment accepts F012 only for bounded external product/execution/SEQ-OMP
claims and retains every missing lane as a limitation; this reconciliation
does not close findings. F013 continues to condition calibration/unit/response,
projection/WCS, coefficient/covariance, and upstream-eligibility conclusions
on `SCI-ALIGN-001`, `SCI-CAL-001`, `SCI-AST-001`, `SCI-PTC-001`, and
`SCI-VAL-001`. A fresh
`codex/reaudit-sci-map-001` worktree must assess the committed repair and the
returned external corpus before findings or production disposition can change.

The independent re-audit at
`851035e67f63bdb2bacc122b17566877a9e6db97` remains intact historical evidence.
The project-owner amendment at
`6409a36d324072c9b29145c620d01a0686275870`, reproduced byte-for-byte as
`handoff/SCI-MAP-001_OWNER_SCOPE_EVIDENCE_AMENDMENT_2026-08-05.md` with
SHA-256 `52be19700b73659ba1847012d4cb0766407399cda5899570acb79bf5b45221f3`,
authorizes a second bounded repair only for F005 aggregate/index fail-closed
safety and coadd-enabled observation-realization persistence. It also defines
the production WCS/card tests and accepts F012 only for the named external
product/execution/SEQ-OMP claims, with every missing lane retained as a
limitation and no Unity rerun required solely for those absences.

The second-cycle candidate rejects floating and signed-count aggregate
overflow and finite projected coordinates outside the representable index
domain before live bundle mutation. It persists required observation
realizations alongside coadd realizations, preserves observation/coadd
ownership and realized cardinality, and propagates missing required writer
slots before the first HDU. Production-path fixtures enforce typed/sidecar to
physical-FITS WCS separation `<= 0.1 arcsec`, exact orientation and centered
integer placement, finite/unit-bearing threshold-card identity and aliases,
sidecar agreement at `rtol=1e-12`, complete realization identity, and
unchanged-WCS atomicity. Normal finite-domain mapmaking, coadd, threshold, and
WCS policies are unchanged.

The complete local second-cycle gate set passes: `citlali_cli`, the monolithic
test executable, the safety executable, and the isolated production-FITS
executable build; 592/592 enabled CTests pass with the one pre-existing
disabled test unchanged; the focused contract/provenance/truth executable
passes 31/31; its ThreadSanitizer build passes 9/9 without a race report; the
production FITS suite passes 22/22; all 147 baseline-tool tests pass; and the
127-test config preflight passes all four mode kits, eight compact
compatibility cases, 100% compact-surface coverage, and every typed-boundary
audit. These are repair-candidate results for fresh independent review, not a
finding or conformance disposition.

The second-cycle independent re-audit at
`fc26e24e6543d1102f9fcc9bf4e849369b39dd04` proposed F005, F007, and F010
for closure, but found one remaining F004/F011 bookkeeping defect: completion
provenance multiplied both observation and coadd products by the global
filtered-stage count even though coadd-enabled filtering writes only the raw
observation stage and writes both raw and filtered coadd stages. Those proposed
closures are re-audit findings, not coordinator-integrated canonical closure.

The final bounded bookkeeping candidate now applies the already-established
observation and coadd output-stage counts separately. In the audited one
observation, one-coadd, three-map, two-realization filtered case it records the
exact 18 realization writes and 9 empirical product maps; the existing
non-coadd filtered and coadd unfiltered states remain unchanged. This alters
only realized provenance cardinality, outside the numerical and output-routing
paths.

The final-bookkeeping local gates pass: all six required build targets
(`citlali_cli`, the monolithic test executable, the safety executable, the
focused truth executable, its ThreadSanitizer build, and the isolated
production-FITS executable) complete; the exact three-state cardinality test
selection passes 3/3; 593/593 enabled CTests pass with the one pre-existing
disabled test unchanged; the focused truth, ThreadSanitizer, and production
FITS suites pass 31/31, 9/9 without a race report, and 22/22; all 147
baseline-tool tests pass; and the 127-test config preflight passes all four
mode kits, eight compact compatibility cases, 100% compact-surface coverage,
and every typed-boundary audit. These remain repair-candidate results for a
fresh independent exact-SHA re-audit, not a finding or conformance disposition.

F004 and F011 remain pending the final exact-repair-SHA re-audit. F005, F007,
and F010 retain only the second-cycle re-audit's proposed closure pending
canonical disposition. F009 and F010 remain `addressed_pending_reaudit`;
production remains `existing_use_only`. F012 is owner-accepted only in the
amendment's bounded terms. F013 remains conditioned on `SCI-ALIGN-001`,
`SCI-CAL-001`, `SCI-AST-001`, `SCI-PTC-001`, and `SCI-VAL-001`; this repair
closes none of them.

On 2026-08-01 the versioned human-run campaign package for exact candidate
`ed28dafb37f9113c0d3c95297148157129a90886` was prepared under
`validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb/`. It pins all
seven repaired-success cases, successor product contracts, explicit owner
deployment values, native TolProj/TolTECA source ordering, independent F010
reconstruction inputs, collection manifests, and frozen analysis. Package
preparation did not access Unity. The owner subsequently executed a bounded
minimal transfer of the seven cases; the external products remain in the
owner-supplied local corpus rather than this repository. The durable
`SCI-MAP-001_EXISTING_CORPUS_CLOSEOUT_2026-08-05.md` records what is present,
what is unavailable, and the exact re-audit route. The package also records
ALIGN-OD1 through ALIGN-OD8 and
ALIGN-C001 as owner-approved at record commit
`4f905f4f353e91847a303f4f3959654f3f03c302`, with canonical identity correction
at `35cc8ce246e8e70c569e650be6c1eae2c91b80ef`, and the bounded repair/re-audit
handoff at coordination commit
`0309fd48a973a6e7e136224906ac49c02f0171be`, and clean coordination-ledger HEAD
`846128c8ee6dc27851bd6c71aeecbe4739e1d24a`. The dedicated ALIGN phase-0 repair
is active from base `9aae0e669384c5c0c0dda93debc194d6b8dac787`, but no ALIGN
application-repair commit or re-audit exists. ALIGN implementation therefore
remains nonconformant, validation is in progress, and production remains
`existing_use_only`. A MAP campaign result cannot close ALIGN, CAL, AST, PTC,
or VAL; F013 remains conditioned until the ALIGN repair, exact-repair-SHA
evidence, and fresh re-audit succeed.

## 2026-07-26 Conan 2 Build Review

The previously deferred TolTECA build implementation is now available and has
received an initial architecture review. Exact evidence, requirement
dispositions, compatibility gaps, and the bounded integration sequence are
recorded in
`doc/TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md`.

The project selected the **Adapt** path. Tula CMake's typed Conan 2 feature
registry, generated-preset workflow, explicit first-party package graph, and
compiler matrix are accepted as the foundation for the successor build. The
reviewed `citlali/v4.x_conan2` target is not a drop-in application build: it
intentionally contains only a five-source static-library slice, 41 headers,
and the Gaussian-model test, with no production CLI or generated source
identity.

The full refactored application still requires 709 Citlali headers, eight
active compiled library sources, the CLI, more than 500 focused CTests,
embedded default configuration, and source/dependency provenance. Kidscpp v3
also omits the active TolTEC raw-data adapter and the presently constructed
but apparently unused sweep fitter. Direct HDF5 and Zlib ownership must be
made explicit.

Phase 5 build integration is therefore unblocked but not complete. The next
work is a bounded compatibility and target adaptation, followed by the full
local gate, a Unity point smoke run, and the frozen same-SHA four-mode matrix.
The existing build remains available until the new path proves all of those
gates. No numerical algorithm changes are part of this integration.

A 2026-07-31 isolated retest of the latest upstream revisions materially
improved this disposition: Tula, Kidscpp, and the Citlali CLI build under exact
Homebrew LLVM 20 and C++23, and their available in-tree tests pass. The
installed Citlali package-consumer test still fails because CPM-provided
NetCDF C++ headers and library metadata do not propagate through the exported
Tula package. The bundled macOS profile also resolves unversioned Homebrew
`llvm` rather than enforcing LLVM 20, one Tula CMake Python test assumes a
specific Conan launcher form, and the real TolTEC reader tests remain skipped
without fixtures. These are adaptation entry gates, not reasons to replace or
freeze the application mainline.

## Current Snapshot

- On 2026-08-22, the scientific owner froze the exact SCI-ALIGN v0.1 Stage B
  r0.3 and SCI-AST v0.1 Stage B r0.3 scientific-contract packets after their
  bounded joint revision and implementation-blind horizontal coherence audit.
  The shared boundary identity is exactly
  `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; its two installed copies are byte-identical
  with SHA-256
  `04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36`.
  The freeze preserves all stable normative IDs and the typed unresolved
  owner questions; it establishes scientific-content authority only, not
  implementation conformity, observational validation, readiness, or
  production authorization. The exact content bindings and successor rules
  are recorded in the scientific-contract library's
  [ALIGN/AST freeze record](scientific_contracts/audits/ALIGN_AST_HORIZONTAL_R0.3/SCI-ALIGN_AST_STAGE_B_R0.3_FREEZE_RECORD.md).
- A 2026-07-30 coherent raw-I/Q event investigation has produced the first
  mode-aware observe-only production slice. The current RTC/PTC learning
  path records accepted intervals per detector UID and compacts only within
  that UID, so a physical network event loses its tone-vector identity and
  fans out into many records. A versioned, fail-closed template schema,
  non-mutating classifier, alternating-half evaluation, typed configuration,
  strict template loader, all-network observation sidecar, and focused C++/
  Python tests are now in place. The extended evaluation scores all 11
  networks present in the corpus: 572
  event/network vectors and 1,210 quiet epochs. The same descriptive point
  selects 167/216 independently participating responses and 0/1,210 quiet
  epochs and surfaces 52/356 responses at shared epochs that did not
  independently trigger. Stable high-cosine but low-amplitude control modes
  show that cosine alone is not a pathology trigger. nw8 remains the positive
  benchmark and nw9 explicitly fails a single-mode stability gate, but
  neither result forms a runtime network allow-list. A catalog-time-blind
  three-state HMM trained on the first half of science observation 152431 now
  independently recovers 96.2% of catalog events in the held-out half and
  transfers with 78.3% and 81.1% recall to observations 152419 and 152433
  after unlabeled target-intrinsic shape normalization. All matched
  transitions have the expected direction and exceed 200 circular-shift null
  trials, while two quiet controls have zero catalog matches. Frozen-scale
  decoding exposes strong nonstationarity, including a 10.9-fold nw3 scale
  increase within 152431; shape evidence and absolute severity must therefore
  remain distinct. This is forensic validation, not an automatic flagger.
  See
  `handoff/SCIENCE_IQ_HELD_OUT_MODE_DETECTION_2026-07-30.md`. The opt-in
  production sidecar clusters RTC-seeded shared epochs and attempts a bounded
  raw-I/Q score for every raw network present, including networks that did not
  seed the event. It writes explicit template and compatibility status and
  changes no samples, flags, weights, learning state, or maps. The first
  bounded Unity
  smoke at `91f99bde` loaded all 11 templates and wrote a schema-valid
  sidecar, but exposed a lifecycle defect: standard RTC diagnostic output
  cleared detailed scan summaries before the observation-level sidecar read
  them, producing zero candidates. The corrected path now copies only
  threshold-passing seeds into a compact scan-keyed cache before detailed QA
  cleanup and clears that cache after sidecar publication. Its CLI build,
  14 focused tests, all 532 enabled CTests, and the full 123-test config
  preflight pass. A corrected observation-152433 Unity smoke, broader corpus
  validation, and same-input enabled/disabled output identity remain the next
  gates;
  coherent masking and subtraction remain disabled. See
  `handoff/COHERENT_RAW_IQ_MODE_OBSERVE_ONLY_ARCHITECTURE_2026-07-30.md`.
- A 2026-07-24 pointing fruit-loop investigation is active. Five controlled
  observations have exact no-feedback seeds but monotonically brighter and
  broader fitted sources through four feedback passes. Production
  subtract/add-back now has opt-in per-scan/array diagnostics; direct
  signal/kernel round-trip and controlled injected-Gaussian recurrence tests
  pass, rejecting a basic sign or unconditional double-add error. The frozen
  obsnum 133410 maps reveal that the low absolute flux cuts select a broad,
  one-sided positive model: 95--98% of active selected pixels and 57--79% of
  tapered positive model sum lie beyond 40 arcsec at the seed. Controlled
  learning, template-taper, and detector-weight Unity ablations are complete:
  learning and tapering are not material causes, and recomputing post-addback
  weights is image-array identical to the control. Fruit loops recover source
  width toward the propagated kernel width. The 13-variant follow-up matrix
  shows stable ten-iteration convergence and a strong PCA-depth response:
  cleaner strength, not broad model support or projection choice, controls the
  correction size. Cleaner-free real-source fits are farther from the
  matched-APT reference, so the reference mismatch does not establish a
  fruit-loop fault. A diagnostic-only, fail-closed full-PTC injected-source
  pair is now implemented locally: restart-matched control and injected
  branches differ only by adding a declared source through the pristine unit
  kernel before model subtraction, and the comparator fits their difference
  through every saved iteration. The first paired Unity run exposed an exact-
  restart defect: checkpoint v1 omitted retained PTC weight-validation state,
  so its control continuation diverged from the uninterrupted trajectory.
  Checkpoint v2 now stores and restores that state, rejects v1 checkpoints,
  and requires an exact uninterrupted-control gate before transfer metrics
  are interpreted. Its local CLI/test builds, all 514 enabled CTests,
  synthetic analysis-tool recovery, and complete config preflight pass. The
  corrected v2 Unity pair now passes exact continuation. The injected source
  recovers monotonically through iterations 9--13, its PSF converges to the
  realized kernel, centroids remain stable, and successive map changes shrink.
  The extended pair through iteration 18 again passes exact restart and
  converges to kernel-normalized recovery of 95.8%, 94.9%, and 98.3% for
  a1100, a1400, and a2000, with stable centroids, kernel-matched widths, and
  1.0--1.7% final map changes. The original monotonic growth is therefore
  resolved as stable recovery of cleaner-suppressed signal, not runaway
  feedback. The remaining 1.7--5.1% attenuation is a measured scientific
  limitation and not an automatic correctness fix. Production defaults
  remain unchanged. A 2026-07-26 calibration-reference assessment now
  separates the use cases: the existing products support qualified relative
  astrometry and effective processed-PSF use, do not support absolute
  photometric/transfer calibration, and cannot yet predict associated science
  response because no local pointing/science association or science-mode
  injection exists. The iteration-18 amplitude/shape plateau does not pass the
  all-array 1%, 2%, or 5% two-transition whole-map criterion; no production
  stopping policy was adopted. A minimum checkpoint-v2 Unity matrix and a
  bounded science-injection design await owner selection/approval; no new
  reductions have been requested. The follow-on 108-observation extension now
  has an independent, frozen quality baseline from all 324 RC1 array maps and
  processed kernels: 54 observations are labeled normal, 38 marginal, and 16
  stress for experiment design. The original five contain four normal, one
  marginal, and no stress observations. The 16 common-binary ten-iteration
  Stage A sentinels are now complete and downloaded: all 480 iteration metrics
  are finite, all 432 transitions are measurable or explicitly classified,
  and the predeclared Stage B gate passes with 8 normal, 5 marginal, and 2
  stress observations retaining all three source associations. At the strict
  combined endpoint gate, 7/48 array trajectories pass at 1%, 21/48 at 2%,
  36/48 at 5%, and 40/48 at 10%. One stress a2000 trajectory follows a
  cross-array-inconsistent source and eight trajectories have FWHM fits
  censored at the pointing fitter's upper bound; neither is counted as
  convergence. Astrometry is therefore qualified per source-associated stable
  trajectory, effective-PSF use is qualified only for uncensored fits,
  photometric calibration remains unsupported, and science response remains
  unmeasured. No stopping policy is adopted. The 92-observation Stage B array
  has now completed under the exact Stage A executable and unchanged policy.
  Its originally failed task 81 was rerun alone; all 16 Stage A and 92 Stage B
  jobs now pass product, log, config-checksum, and provenance audits. The
  complete analysis covers 108 observations, 324 array trajectories, and
  3,240 maps through iteration 9. It exposed a product-semantic defect in the
  historical pointing-table
  `sig2noise`: it is fitted amplitude divided by full-map RMS, so recovered
  source structure makes it a dynamic-range diagnostic rather than
  statistical significance. The population analyzer now excludes that legacy
  quantity from convergence, reports formal `amp / amp_err`, source-free
  background and roughness, and a versioned blank-sky empirical PSF S/N
  separately. A backward-compatible pointing-table v2 appends truthful
  `peak_over_full_map_rms` and `fit_sig2noise` columns while retaining the
  legacy column. The complete morphology-aware population supports a
  discussion candidate of 3% amplitude change, no evaluation before iteration
  6, and two consecutive all-array passes. Every one of 225 unresolved-source
  array trajectories resolves at 3%, with a 1.87% P90 and 3.57% maximum
  stopped-to-iteration-9 residual. Planetary disks use observation-epoch JPL
  Horizons diameters convolved with each realized kernel; only 77/99 planet
  trajectories resolve at 3%. The complete V0 multi-metric rule resolves
  57/108 observations. Of the 51 others, 23 are measurement-limited and 28
  retain measurable but unresolved trajectories appropriate for short
  checkpoint-v2 continuation. Formal and empirical point-source S/N rise
  while source-free background does not increase monotonically, confirming
  that the historical full-map dynamic-range decline is not scientific S/N
  loss. Separate PSF, centroid, map, support, learning, and noise criteria
  remain unapproved. See the
  [convergence-criteria discussion](FRUIT_LOOP_CONVERGENCE_CRITERIA_DISCUSSION_2026-07-27.md).
  The local implementation snapshot passes the `citlali_cli` build, all 517
  enabled CTests, all 135 baseline-tool tests, the complete 123-test config
  preflight and strict audits, and all 33 fruit-loop tool tests.
  Exact injected-source pairs remain reserved for one representative of each
  quality stratum. These are descriptive strata, not data rejection or
  production policy. See the
  [feedback investigation](FRUIT_LOOP_FEEDBACK_INVESTIGATION_2026-07-24.md)
  and
  [calibration-reference assessment](FRUIT_LOOP_CALIBRATION_REFERENCE_INVESTIGATION_2026-07-26.md),
  plus the
  [population extension plan](FRUIT_LOOP_POPULATION_EXTENSION_PLAN_2026-07-26.md).
- A 2026-07-24 reliability investigation is active for two long
  108-observation pointing jobs that received `SIGBUS` on the same a1400 Ceres
  solve after 45 completed observations. The fitter code is unchanged across
  the failed and current commits; 512 repeated synthetic fits and an exact
  138-fit replay from the downloaded scientific maps both pass, including
  ASan/UBSan instrumentation of the fitter translation unit. Corrected
  observation-boundary RSS is approximately 0.8--1.1 GiB rather than the
  step-wide 26/41 GiB Slurm peaks. Current work adds PID-level resource,
  executable-mapping, and robust signal diagnostics plus a low-level
  config-slicing/native-run harness. No algorithm or fit policy has changed,
  and a native Unity failure frame is still required before choosing a fix.
  TolPROJ now has a tested submission-time executable snapshot and
  checksum-verified node-local launcher on its development branch. This
  prevents future queued/running reductions from depending on a mutable
  `build/bin/citlali`, but is recorded as an operational safeguard rather than
  proof of the historical `SIGBUS` root cause.
  See the
  [investigation handoff](../handoff/CITLALI_MULTI_OBSERVATION_SIGBUS_INVESTIGATION_2026-07-24.md).
- Phase 5 validation-epoch preparation is complete as of 2026-07-24. The
  historical four-profile Phase 4 epoch remains active and immutable; a
  separate four-profile `phase5-v2.1-candidate-2026-07-24` epoch is registered
  as preparing. Its config comparison is exact except for the versioned
  `tolteca-native-project-bindings-v1` policy, which permits host/project path
  prefixes to move while preserving bound file and directory identities.
  Preparing profiles have no accepted baseline records, require an explicit
  comparator, and cannot report an accepted verdict. All four available V2.1
  suite fixtures pass config, product-contract, and product-comparison smoke
  gates; promotion remains blocked by runtime-provenance V1 in every fixture,
  missing science pointing provenance, the deferred build review, and the
  absence of a same-SHA four-mode candidate matrix. The one-command fixture
  verification matches all recorded outcomes. All 134 baseline-tool tests and
  the 123-test full config preflight pass. See the
  [successor-epoch preparation record](PHASE5_VALIDATION_EPOCH_PREPARATION_2026-07-24.md).
- Runtime resource debt D16 is closed as of 2026-07-24. TolPROJ keeps refactor
  runtime threads and generated Slurm CPUs coherent, rejects oversubscription
  before its recommended submission path, and preserves legacy defaults.
  Citlali uses an independent runtime safety net that resolves Slurm, affinity,
  and hardware
  availability, caps rather than aborts an allocated job, emits one warning,
  and writes `citlali-runtime-provenance-v2`. Local build, focused tests, full
  gates (500 CTests, 119 baseline tests, and 118 config tests), and the 147-test
  TolPROJ suite pass. The matching Unity case passed at `d339053cc` in
  `pointings_v22/redu00`: six requested threads matched six affinity-available
  and six effective OpenMP threads without adjustment; runtime provenance V2
  is valid, the run completed all 12 PTC chunks with no logged issues, and all
  non-profile scientific products are exact against `pointings_v21/redu00`.
  The intentionally mismatched direct-submission case then requested 12
  threads inside a six-CPU affinity allocation. It emitted exactly one
  resource-cap warning, continued with six effective and realized OpenMP
  threads, recorded the adjustment in valid V2 provenance, completed all 12
  PTC chunks, and again produced exact non-profile products. [The runtime
  resource contract](RUNTIME_RESOURCE_CONTRACT_2026-07-23.md) records the
  evidence.
- Refactor baseline: `376e0022`.
- Production code inspected by the external review: `84670829`.
- Latest accepted point reduction: Phase 3 exit checkpoint `redu66`, produced
  by `2a974e0dd`, is exact against full-Wiener checkpoint `redu65` across all 19
  non-profile scientific products, including complete RTC/PTC timestreams,
  with zero changed or skipped records. Their 490-leaf configs are exact. Both
  profiles contain the same multiset of 78 stage/context records; only elapsed
  values and concurrent completion order differ. The run has 12 complete PTC
  chunks, zero logged issues, all required provenance valid, and a successful
  VAST-backed exclusive output-root acquisition. `redu64` remains the accepted
  mature library-exit checkpoint and `redu63` the first compiled boundary.
  Observation-resolved astrometry
  provenance `redu61`, disabled polarimetry capability provenance
  `redu60`, external KIDs/config-source provenance
  `redu59`, post-processing authority cleanup `redu58`,
  realized provenance `redu57`, typed source-fitting `redu56`, source-finding `redu55`, map-filter `redu54`,
  enabled-filtering `redu53`,
  unfiltered `redu51`, and bounded full-noise-output `redu49` remain the
  immediate post-processing, pointing, and noise-products control fixtures.
- The Phase 4.1 self-contained point smoke `pointings_v21/redu00`, produced by
  `cfae989ce`, accepts the raw-only `phase4.1-v2.1` pointing policy. It
  completed all 12 PTC chunks in 135 log seconds with no error-level records,
  and every required provenance sidecar is valid. Requested and effective map
  filtering are disabled, realized filter-context and filtered-map counts are
  zero, and no filtered product directory exists. The raw pointing table
  contains three valid fits from three attempts, one for each array.
- Phase 3 full-Wiener point `redu65`, produced by `6dd0057f8`, is accepted
  against matched OG `redu10` at `ffc6b907`. Both use five noise realizations,
  a Gaussian template, and `lowpass_only: false`, and both execute all six
  expected Wiener core calls. The seven filtered products pass the strict
  scientific-tolerance gate with 148 compared records and no skips; the
  three-array pointing-fit table is exact. The refactor run has zero issues.
- Latest accepted OOF reduction: refactor `redu02` for observations
  152385-152387, produced by `9ea6d7f01`, is exact against accepted refactor
  `redu01`. The established OG `redu00` versus refactor relationship is
  unchanged. All 30 comparable products are
  present with no skipped records; pointing-table data and all per-observation
  ECSV/FITS dates are exact, and all scientific numeric differences pass the
  standard `2e-8 + 1e-10 * abs(reference)` tolerance. The only accepted
  differences are inactive RTC-despike config metadata recorded differently
  by the legacy and typed paths.
- Phase 4.1 OOF smoke `redu00` in the self-contained validation suite, produced
  by `e97de3fd`, intentionally enables diagnostic Gaussian fitting while
  retaining `psf_preserve` and `map_center`. All three observations completed
  in 59.4 seconds with zero logged issues, all required provenance valid, and
  nine valid array fits from nine attempts. Its matched APT differs materially
  from the older accepted OOF fixture, so this is mode-kit execution evidence,
  not a numerical replacement for accepted `redu02`.
- Latest accepted science reduction: clean single-job four-iteration sequence
  `redu28` through final `redu31`, produced by `a7a35a00`. Its 502-leaf config
  is exact against accepted `redu23`; all 12 FITS and 15 NetCDF product sets are
  complete. All 84 map layers pass the science-equivalence gate with maximum
  relative RMS `7.09e-14`, all integer diagnostics are exact, and all 1,394
  NetCDF variables pass. The final run has zero logged issues and every required
  provenance record is valid. This supersedes `redu23` as the science fixture.
- Latest accepted Beammap reduction: Phase 3 checkpoint `redu06`, produced by
  `6dd0057f8`, is exact against accepted `redu05` across all 12 comparable
  products and 16,453 comparison records, including complete detector TOD,
  diagnostic NetCDF, detector-fit tables, and six split-map FITS products. Its
  529-leaf config is exact, all required provenance is valid, and the log has
  zero issues. It accepts the fruit-loop input/feedback and mature Wiener
  failure-contract tranches for Beammap.
- The Phase 4.1 self-contained Beammap `redu00` produced by `cfae989ce`
  accepts the split-output correction. It completes 198 PTC chunks and three
  internal Beammap iterations in 3,779 log seconds with no error-level
  records, and all required provenance is valid. Its final APT marks 196, 27,
  and 38 bad detectors in a1100, a1400, and a2000; the split bad-detector FITS
  files now contain exactly those counts, while the good files contain 2,901,
  1,186, and 886 detectors. The three iteration fit counts, final
  network-position flags, and complete kernel-map diagnostic summary are exact
  against rejected predecessor `189bbf85d`, establishing that the fix changes
  output completion rather than Beammap science. Commit `e496dcb6e` catches
  the CCfits base exception at required map/header boundaries, and
  `a68bf1737` omits unavailable per-detector FITS keywords while retaining
  `NaN` in the authoritative APT table for compatibility with Unity's older
  CCfits. All 13 established Beammap product families pass. The immutable
  version-one contract continues to describe the historical 13-product
  snapshot. Successor contract `phase4.1-beammap-products-v2` classifies and
  schema-checks the required `citlali_restart_checkpoint.nc` and accepts all
  14/14 current products without changing the active V1 profile.
- `redu23` and `redu24` completed all 12 PTC chunks with zero error-level log
  records and complete TOD/diagnostic products. Their common numeric products,
  FITS maps, and pointing tables are exact; only profiling timing differs.
- `redu21` and `redu22` had exact common numeric products with complete TOD
  comparison, but both contained 12 logged NetCDF errors.
- The same YAML exposed two provenance defects in `redu22`: an effective IIR
  default appeared for a disabled filter and an extinction sentinel changed.
  `redu25` validates the intended disabled-state provenance correction with
  exact scientific products.
- Local `citlali_cli`/test builds and full config preflight pass.
- CTest discovers and passes all 490 tests. The Phase 4.1 config preflight now
  passes 117 focused tests; the checked leaf contract covers 578 leaves and the generated
  startup schema covers 728 normalized YAML nodes. The six new direct tests
  cover the production rejection of experimental maximum-likelihood
  mapmaking, analytic flux conversion, detector-specific calibration,
  detector pointing, and two source-finder safety boundaries. Eleven additional
  NGC4449 candidate tests cover cap-independent effective learning state,
  interval/penalty compaction, diagnostic decoupling, fruit-loop activation and
  realized feedback, Beammap policy isolation, and truthful standardized-map
  naming. Three additional tests cover the diagnostic-only learning/HK match
  contract and required sidecar output. Two additional Wiener tests cover
  fail-closed compensated-template normalization and unchanged
  well-conditioned convolution.

These facts are characterization evidence, not a production-equivalence claim.

### NGC4449 Full-Science Candidate Investigation (2026-07-21)

The first five-observation, ten-iteration NGC4449 run exposed four blocking
Citlali contracts: statically empty fruit-loop feedback, formal-weight
standardized maps mislabeled as S/N, a diagnostic learning cap that truncated
operational state in input order, and warning amplification that obscured QA.
The bounded investigation and candidate corrections are recorded in
`doc/NGC4449_CITLALI_INVESTIGATION_2026-07-21.md`.

Local builds, all 481 CTests, the 116-test config preflight, and 106 baseline-
tool tests pass. The changes are not an accepted scientific snapshot:
applying the full effective learned state intentionally changes the
previously cap-truncated flags and therefore requires a successor Unity science
profile and an intended-science-change ledger entry before acceptance. The
immutable phase-4 v1 product checks remain available for historical artifacts;
candidate v2 basic-map checks reserve S/N names for empirically calibrated
products. A new learning-housekeeping QA sidecar correlates deduplicated
busy-network pathologies with selected TolTEC thermometry and dilution-fridge
samples while remaining strictly outside the flagging and learning policy.

The first one-observation spatial-feedback control then exposed a separate
configuration-authority defect: science YAML requested
`pointing.source_strategy.fruitloops_center_mode: map_center`, but the science
execution path never loads the pointing plan and therefore continued with
automatic off-center peaks. A successor typed control,
`timestream.fruit_loops.source_center_mode`, is now owned by the processed-
timestream request, serialized in its snapshot and NetCDF config record, and
adapted directly to the fruit-loop processor. The additive `auto` default
preserves prior behavior. The NGC4449 successor requests `map_center`; a Unity
run must confirm the realized log before this becomes accepted science
evidence.

The project owner then approved state-complete cross-job fruit-loop
continuation so NGC4449 iterations can be extended without discarding learned
state. ADR 0006 defines the new required atomic
`citlali_restart_checkpoint.nc` artifact and explicit
`timestream.fruit_loops.restart_path`. The checkpoint stores compacted
operational masks and detector penalties, absolute iteration identity, ordered
observations, map type, creator version, and the complete learning-policy
snapshot; bounded diagnostic event history is intentionally excluded. Loading
is fail-closed for the stored compatibility contract, `path` and
`restart_path` are mutually exclusive, and `max_iters` is the absolute
exclusive stop. Local split-run tests show five completed synthetic learning
iterations plus a two-iteration restart exactly match seven uninterrupted
iterations. Local CLI/test builds, 488 CTests, the 116-test complete config
preflight, and all 108 baseline-tool tests (including 60 reduction-audit
tests) pass. A matched Unity split versus
uninterrupted science run is still required; the already-running older binary
cannot create this new checkpoint.

The first real exact-restart control, performed for the full-PTC
injected-source experiment on 2026-07-25, invalidated checkpoint schema v1.
Restarted absolute iteration 9 differed from uninterrupted iteration 9 by
4.6--26.6% relative RMS in signal maps and 8.3--49.7% in weight maps. The
missing state was the validated PTC weighting accumulator and finalized
detector-factor vectors retained in `PTCProc` across in-process iterations.
Schema v2 now stores those vectors and their phase, records a canonical
processed-timestream policy snapshot, and rejects v1 checkpoints. Focused
restart tests, the 500-test `citlali_test` binary, the local CLI build, and the
complete 123-test config preflight pass. A new uninterrupted v2 trajectory
plus exact restarted control is required before the injected-source transfer
experiment can be interpreted.

The completed older-binary continuation then exposed an invalid-success
boundary in lowpass-only kernel filtering. The latest radial `a1400` template
has `abs(sum)/sum(abs)` approximately `0.004`; unit-sum convolution therefore
amplified a compensated transfer kernel by a cancellation condition number of
about 251 and produced filtered signal/kernel/weight products with no valid
flux interpretation. A shared serial/OpenMP runtime contract now rejects a
non-finite, zero-L1, or cancellation-conditioned unit-sum template below a
`0.05` DC fraction, limiting normalization L1 gain to 20 and reporting all
conditioning values plus the full-Wiener corrective route. Well-conditioned
convolution behavior is unchanged. Local serial and OpenMP CLI/test builds,
all 490 CTests, the 117-test complete config preflight, and all 108 baseline-
tool tests pass. An NGC4449 full-Wiener Unity successor remains required
before acceptance; all filtered `a1400` products from the lowpass-only
NGC4449 series remain quarantined.

The follow-up raw-readout investigation on 2026-07-29 established the
producer-confirmed schema of `Header.Toltec.AdcSnapData`: shape `[2,4096]`,
with index `0` representing the beginning of the raw data file, index `1` the
end, and signed 12-bit ADC counts in `[-2048,2047]` stored in a NetCDF
`short`. The current input reader now names that boundary ordering and count
domain without changing numerical behavior. In the eight downloaded NGC4449
pointings, nw9 reaches both rails at both file boundaries in every
observation; nw3/nw4 have low headroom with sparse rail contact; and nw1,
nw2, and nw8 exhibit late map pathology without ADC saturation. ADC
utilization does not acquire the broader pathology's 152420 onset, so clipping
is a real nw9 validity problem but not its common cause. Retained debt D17 now
requires a cold-boundary saturation validator and explicit persisted
severity, while leaving warning, network-exclusion, and reduction-failure
thresholds unapproved pending representative validation.

## Active Phase

**Phase 4.1 - TolTECA operator config structure** is complete as of 2026-07-23,
and **Phase 4.2 - technique and performance review** is complete as of
2026-07-17. The project owner added both stages between the adopted Phase 4
evidence package and final Phase 5 integration. Phase 3 library/session work
is complete: local gates pass and Unity point `redu66` accepts the output-root
ownership repair and exact scientific behavior at the first compiled
boundary. Phase 2 config authority and provenance remains complete at Unity
point `redu62`. Formal Phase 5 integration remains blocked on the deferred
TolTECA build review; compilation-independent validation-contract and
integration-packet preparation may continue.

The TolTECA build owner is preparing a Citlali v4.x build approach intended to
apply here but has not yet provided an implementation for review. The
[`build integration requirements`](TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md)
define the outcome and evidence expected from that work without prescribing
its tools or creating a competing build-system rewrite. This tree retains its
current working CMake path until the implementation can be evaluated and
adopted, boundedly adapted, or explicitly deferred.

Compilation-independent Phase 5 readiness is current as of 2026-07-24. The
existing local path builds `citlali_cli`; its previously established CTest
  gate remains valid, and the current 123-test config preflight and 134
baseline-tool tests pass. The 60-record validation ledger, three-entry
intended-science-change ledger, eight-profile/two-epoch registry, and
session-exit audit are valid, with zero supported library exits or growth. The
[`integration record template`](PHASE5_INTEGRATION_RECORD_TEMPLATE.md) now
captures the frozen-SHA build decision, local gates, four-mode Unity matrix,
science/capability disposition, and integration authorization without
pre-filling unavailable evidence. This readiness does not freeze a candidate
or close the deferred build criteria.

TolPROJ commit `e0754af` supersedes the custom suite installer and staging
layers introduced by commits `39f724d` and `8310c24`. The canonical
`suite.yaml` now contains only path-free observation selections and validation
metadata. `tolproj validation-suite init` queries TolPROJ's metadata database
and uses the existing native project builders to create ordinary `point`,
`oof`, `beammaps`, and `science` projects. The science project contains its own
complete pointing support, while the selected-observation Beammap builder
discovers source-matched pointing support from metadata. Raw discovery and
copying, tune reduction, cohort construction, APT seed selection and matching,
Beammap flux estimation, and `--refactor` reduction setup all remain owned by
the established TolPROJ commands. Native `project.yaml` is deliberately live
workflow state rather than a hashed immutable artifact. Verification guards
the portable selection and required native structure without objecting to
normal status, cohort, or APT updates. The workflow does not provision an OG
tree, submit jobs, or run Citlali. All 104 TolPROJ tests pass. Freshly created
Unity projects subsequently completed point, OOF, Beammap, and science smoke
reductions with their requested products and no unexpected errors, closing
the Phase 4.1 execution gate.

TolPROJ follow-up commit `d2c90f3` applies the first Unity setup corrections.
The native skeleton is mode-minimal: point and OOF omit an unused nested
`pointings/` directory; Beammap omits both unused `pointings/` and `apts/`;
science retains its pointing-support structure. All generated Citlali
`02_redu.sh` scripts now request the configured partition (`toltec-cpu` on
Unity). TolPROJ file logging is action-specific (`copyraw.log`,
`reducetunes.log`, `matchapts.log`, `pointings.log`, `science.log`, and
`flxscale.log`) instead of funneling independent steps into `tolproj.log`;
SLURM stdout remains separately named `<jobname>-%j.out`. All 105 TolPROJ tests,
full Ruff, and byte-compilation pass.

TolPROJ commit `9fb4c80` closes a science setup ordering hole found during the
first suite attempt. Science setup now verifies that every configured
`cal_objs` pointing-product directory exists before writing reduction configs,
and accepts `--pointing-reduction reduNN` so setup uses the same accepted run as
pointing flux calibration. This converts a late TolTECA `invalid calobj path`
failure into an actionable TolPROJ preflight error. All 106 TolPROJ tests pass.

The populated Unity tree then showed that existence alone is insufficient:
TolTECA recursively requires exactly one `ppt_*.ecsv` under each `cal_objs`
path, while an observation root can contain both raw and filtered tables.
TolPROJ commit `704b486` makes the established raw pointing product explicit,
validates exactly one table under `<obsnum>/raw`, and emits that directory for
science and Beammap pointing references. All 106 TolPROJ tests, full Ruff, and
byte-compilation pass.

Citlali commits `95b7b7f57` and `f59c663f8` establish config kit
`phase4.1-v2.1`. The self-contained OOF smoke showed that the prior default
produced vacuous all-zero pointing-fit tables, so OOF now enables diagnostic
Gaussian fitting while preserving PSF-preserving mapmaking and map-centered
fruit-loop support. Routine standalone, science-support, and Beammap-support
pointings now default to raw products; Wiener settings remain visible for an
explicit validation overlay but filtering is disabled. TolPROJ commit
`16ebe69` vendors the complete 29-file canonical kit byte-for-byte, retains
`phase4.1-v2` for historical reproducibility, and selects V2.1 only for fresh
`--refactor` setups. All 139 TolPROJ tests and Ruff pass. No Citlali
compilation is required for these numbered-YAML integration changes.

The four native-project smoke gates are complete. Point
`pointings_v21/redu00` verifies the V2.1 raw-only policy with three valid array
fits and no filtered outputs. OOF `redu00` verifies diagnostic Gaussian fits
under PSF-preserving mapmaking. Science `redu03` verifies the unchanged science
V2/V2.1 policy across two observations, four fruit-loop iterations, raw and
filtered coadds, noise products, and learning outputs. Beammap `redu00`
verifies all 198 PTC chunks, three internal iterations, APT good/bad
classification, and complete good/bad split FITS products after the required
split-output correction. All four runs completed without unexpected
error-level records. This closes Phase 4.1 and retained-debt item D09.

Compilation-side Phase 4 work is explicitly deferred as of 2026-07-16 pending
review of the TolTECA developer's revised C++ build and integration approach.
Do not change Citlali CMake structure, presets, dependency management, CI build
lanes, install/export rules, or cluster build helpers until that direction is
understood. This is a sequencing decision, not acceptance of the current build
as the final reproducible-build solution. Phase 4 continues meanwhile through
strict validation, current baseline/ledger work, controlled performance
evidence, and scientific-contract documentation that is independent of the
eventual compilation strategy.

The first compilation-independent Phase 4 tranche establishes a versioned
validation epoch. Four named profiles pin the current point, OOF, science, and
Beammap snapshots to their required provenance, exact low-level configuration,
and mode-appropriate product comparator. Point `redu66` is the zero-tolerance
structural-closeout snapshot; clean science `redu31` is the current
scientific-tolerance snapshot. One profile-driven command now performs the run
audit, config comparison, and product comparison without duplicating the
existing scientific comparator logic. Accepted snapshots are immutable.
Future intentional algorithm, default, schema, or product changes create a
successor epoch with a predecessor comparison and explicit scientific
rationale instead of silently replacing a baseline or loosening its policy.

The profile command now includes a fourth, versioned scientific-product
contract gate. `validation/product_contracts.json` classifies all accepted
FITS, NetCDF, ECSV, and CSV products: point 21/21, OOF 31/31, science 28/28,
and Beammap 13/13. Configuration-controlled families are evaluated against the
generated merged low-level YAML in both directions: requested output must be
present and disabled output must be absent. Products without an independent
switch remain required companions of their parent output; operational timing
and bounded learning records remain optional diagnostics. The contract records
scientific identity, coordinate frame, axes, units, indexing, missing-value,
and fatal required-write policies while naming existing metadata debt rather
than inventing semantics. See the
[scientific product contract](PHASE4_SCIENTIFIC_PRODUCT_CONTRACT_2026-07-16.md).

The intended post-baseline science-change census is now machine-readable in
`validation/intended_science_changes.json`. It identifies three accepted
imports from `gw_dev`: the RTC/PTC parallel determinism fix, the three-commit
Wiener optimization, and the active-detector PCA optimization. Full source and
integration commits are recorded. Four patch-equivalent cherry picks are
verified mechanically by stable Git patch identity; the manually transplanted
determinism fix is tied to its direct two-run OMP evidence. Every entry records
its expected behavior and numerical/schema effect, affected modes and product
families, accepted-run evidence, and limitations. Scientific behavior already
present at baseline `376e0022` is explicitly inherited rather than relabeled,
and later OG `ffc6b907` is recorded as a comparator rather than an imported
commit. Future non-structural changes require a ledger entry before acceptance.

The canonical human scientific contract is now
[`SCIENTIFIC_CONVENTIONS.md`](SCIENTIFIC_CONVENTIONS.md). It consolidates the
validated identity distinctions, sample/detector/map shapes, coordinate frames,
units, indexing, missing-data rules, requested/effective/observation/realized
semantics, output-failure policy, and active numerical gates. It explicitly
keeps enabled polarimetry and R-channel execution outside the validated
capability boundary and collects unresolved scientific-owner decisions without
inventing answers. During that census, the product registry's RTC/PTC
`output_scan_index` description was corrected to match the writer and NetCDF
metadata: it is the one-based original scan number, while dimension positions
remain zero-based. No product data or schema changed.

The canonical current software map is now
[`ARCHITECTURE.md`](ARCHITECTURE.md). It records the active target and CLI
entry, session/result boundary, runtime and scientific data flow, configuration
state transitions, lifecycle owners, failure and product contracts, and the
allowed direction for new dependencies. It explicitly distinguishes active,
transitional, unbuilt legacy/experimental, and deferred paths. In particular,
`Engine` remains an active but frozen compatibility aggregate, header-defined
mode and numerical code remains transitional, and the four unbuilt historical
main programs are not supported entry points. The document does not disguise
the still header-dominant physical build or authorize the deferred CMake and
dependency work.

The broader section-F.2 exit criteria are now mapped in the
[`Phase 4 closeout census`](PHASE4_CLOSEOUT_CENSUS_2026-07-16.md). Of the 15
criteria, ten are closed by implementation and evidence, two are closed by an
explicit owner scope decision or proportionality exception, and three are
compilation-dependent and remain deliberately deferred. The five focused ADRs
are indexed in `doc/adr/README.md`; root `CODEX.md` is now a concise canonical-
document redirect; and [`RETAINED_DEBT.md`](RETAINED_DEBT.md) records each
deliberate limitation with role owner, reopening trigger, and exit condition.
The adopted external-review Phase 4 compilation-independent criteria are
complete. The project owner subsequently added two explicit pre-integration
stages: the
[`four-mode TolTECA authoring structure`](PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md)
and the
[`whole-code technique/performance review`](PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md).
These additions do not change the census counts or waive the three deferred
build criteria. They do replace the previous instruction to remain idle until
the TolTECA build direction is available.

The TolTECA build owner is unavailable until the week following 2026-07-16.
The project owner authorized a bounded
[`Phase 5 preparation lane`](PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md)
so closeout planning does not sit idle. This is not formal Phase 5 integration
and does not waive criteria 6, 7, or 10. The source disposition, final
same-SHA validation matrix, and integration packet can be prepared now; CMake
changes, placeholder deletion, final candidate tagging, and integration remain
blocked on the build review. Phase 4.1 and Phase 4.2 are the only additional
tranches authorized by this scheduling workaround.

The controlled-performance evidence path is now specified without touching
deferred compilation infrastructure. A Unity-side wrapper records GNU Time
wall/RSS/I/O data together with Citlali log time, exact config leaves, bounded
input hashes, runtime policy, binary identity, serious log counts, and
profile-stage totals. An offline campaign analyzer requires same-node warmups,
at least three alternating measured Beammap pairs, matched config/input/runtime
policy, complete measurements, an explicit runtime budget, and required RSS
measurement; it reports paired ratios plus median and IQR. The checked-in
campaign is a diagnostic template rather than a mandatory Phase 4 run. If used,
the 5% wall-time ceiling applies and peak RSS remains required evidence with an
evidence-driven limit.

The GNU Time wrapper passed its first live Unity exercise in point `redu67` at
`7ca0be50c`. It captured matching retained and attached evidence, binary and
dependency identities, host/storage/runtime policy, 131.08 seconds external
wall time, 110.477 seconds Citlali time, and 908,316 KB peak RSS. The active
point profile accepted `redu67` against immutable baseline `redu66`: zero
logged issues, zero differences across 490 config leaves, and zero changes in
2,064 records from 19 products. This qualifies the wrapper integration but does
not constitute a Beammap performance conclusion.

The project owner accepted a proportionality exception to a dedicated Beammap
campaign on 2026-07-16. Twelve accepted refactor checkpoints range from
3,397.522 to 4,215.296 seconds with a median of 3,594.693 seconds, move in both
directions, and end with a 1.9% adjacent increase. A prior 13.0% total-time
increase coincided with 1.3% faster mapmaking and was concentrated in
VAST-sensitive PTC and diagnostics I/O. This history and repeated scientific
validation show no sustained regression signal. Serializing Citlali jobs would
not control unrelated VAST traffic, so eight dedicated hour-scale reductions
are not justified now.

Future naturally required Beammap validation should use the wrapper to collect
peak RSS and full provenance. A controlled campaign becomes mandatory only for
a sustained runtime regression, unexplained stage slowdown, memory failure,
peak RSS near node capacity, or a material hot-path change. Profiling overhead
is likewise investigated when a performance signal warrants adding an explicit
control. See the
[controlled performance protocol](PHASE4_PERFORMANCE_PROTOCOL_2026-07-16.md).

A planned post-refactor re-reduction of approximately 50 historical Beammap
observations will provide the broader operational performance census. The
[`corpus plan`](BEAMMAP_CORPUS_PERFORMANCE_CENSUS_PLAN_2026-07-23.md),
manifest template, and offline analyzer are collection-ready as of 2026-07-23.
They reuse the existing GNU Time evidence record, verify observation identity
against Beammap provenance, require one current record per expected
observation, extract workload and output-volume evidence, report population
distributions and workload relationships, and preserve only explicit
same-observation comparisons. Unlike observations are not treated as repeated
trials, and ranked observations are never silently excluded. Eight focused
tests cover completeness, identity, pairing, grouping, workload relationships,
conflicting overrides, and failure behavior. The census itself remains a
future release baseline, not a Phase 4 closeout prerequisite.

Retained-debt item D15 now has a bounded offline evidence path. The
[`fruit-loop convergence study`](FRUIT_LOOP_CONVERGENCE_STUDY_2026-07-23.md)
and manifest-driven analyzer compare consecutive raw coadds for every array,
check support, weights, aperture and peak behavior, and require stable effective
learning state before simulating an explicitly non-production stopping rule.
The five active NGC4449 spatial-feedback iterations pass the study protocol
but do not demonstrate an early stop: the sequence changes non-monotonically
when learned state takes effect, and only the final transition passes the
exploratory all-array rule. The older map-path continuation resets its learning
lifecycle and is not appended as false state-continuous evidence. Production
retains `max_iters`; representative checkpoint-complete sequences and
scientific-owner threshold approval are still required to close D15.

Phase 1 safety stabilization is complete for point, Beammap, science, and OOF.
OOF refactor `redu01` closes the multi-observation date-header gate and is the
accepted comparison against OG `redu00`. Do not reopen typed analysis-control
migration during Phase 4; validation and reproducibility are now the priority.

Operational config migration must proceed one authority domain at a time with
the one-way requested-to-effective-to-realized contract, focused tests, and the
existing mode gates. Compact-config production rollout and open-ended file
splitting remain out of scope.

The initial Phase 3 session boundary is implemented locally. A non-copyable
`citlali::session::ReductionSession` owns sequential run state and returns a
structured `ReductionResult` containing status, diagnostics, product roots,
and published provenance artifacts. Standard reduction loading and processor
selection now execute inside that session. The CLI remains the only layer that
prints result diagnostics and translates success to a process exit code.
Focused tests cover success, exception conversion, failure recovery, two
sequential runs, nested-run rejection, CLI policy separation, independent
header compilation, and multi-translation-unit linkage. Both local test
targets build, all 448 CTests pass, and full config preflight passes. This is
the facade checkpoint, not the Phase 3 exit gate: reachable library exits,
complete internal failure classification, remaining lifecycle ownership cuts,
and validation of the first `.cpp` boundary remain open. The
[bounded ownership plan](PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md) records the
sequence and stop rules.

The first failure-boundary and exit-census slice is also complete locally.
`ReductionSession` classifies canonical config, I/O, output, runtime, and
internal errors without terminating the process. Eight direct setup exits are
retired without touching numerical loops. An independent scan-context test
found and repaired a real include-order dependency on typed runtime policy.
The new conservative session audit follows 667 project-header dependencies
from the reusable entry and freezes a no-growth baseline of 94 direct library
exits across 22 files, with no CLI exits in the graph. The
[exit census](PHASE3_SESSION_EXIT_CENSUS_2026-07-15.md) defines the bounded
retirement order and separates low-risk setup/output work from mature
timestream and Wiener kernels. The first post-baseline cluster removes all six
TOD output-selection config exits. Invalid strings, empty or nonpositive chunk
lists, negative counts, and impossible selection modes now accumulate atomic,
path-aware config diagnostics. The adjacent effective row-selection boundary
now converts invalid effective modes, empty source-crossing selections, and
out-of-range chunks to canonical errors while preserving valid row assignment.
The current dependency-reachable count is 85. Its isolated test also
characterized the remaining ambient named-logger dependency in the legacy
`get_config_value` helper for later ownership work.

The first observation-input tranche centralizes the three duplicated KIDs
matrix validity checks used by direct, loaded, and gap-aligned RTC input.
Finite matrices retain the same path; NaN and infinite values now become
canonical I/O errors that a `ReductionSession` can report without terminating
the process. Three focused tests cover the contract, and the session audit is
down to 79 dependency-reachable library exits.

The observation/input setup census group is now complete. Detector-count and
cross-network sample-rate mismatches, invalid gap-alignment sample rates,
negative derived extinction, missing polarization calibration groups, invalid
IIR/Nyquist combinations, and Beammap fit-map shape mismatches all use explicit
canonical failure categories. Existing valid setup, metadata reads, and
numerical work are unchanged; the sample-rate path retains one metadata read
per network. Eight focused contract tests pass, and the session audit is down
to 71 dependency-reachable library exits.

Required FITS image and PHDU output-slot validation is now session-safe. Nine
map, Stokes, array, noise-map, and PHDU cardinality exits route through one
canonical required-output failure helper: the library logs the concrete slot
diagnostic and throws an output error, while only the CLI selects a process
exit code. Valid slot lookup and map writing are unchanged. Focused success
and failure tests pass, including every retired branch, and the session audit
is down to 62 dependency-reachable library exits.

The FITS/ECSV adapter tranche completes the output census group. CCfits'
nonstandard `FitsException` hierarchy is caught at operation boundaries and
classified as input I/O or required-output failure; ECSV input and atomic
publication use the same categories. Negative-path tests found and fixed the
distinction between `FitsError` and its sibling open/create exceptions. The
last apparent exit in this group was inside a fully commented, unused Gaussian
transfer-function prototype, which was removed as dead code. The audit reached
57 exits after this tranche.

The final three non-kernel mapmaking preconditions are also session-safe.
Unsupported polarization/grouping combinations, non-altaz Beammap requests,
and missing Wiener template FWHM values now throw canonical config errors;
successful policy and template setup are unchanged. The audit now reports 54
dependency-reachable library exits, all confined to mature RTC, PTC,
timestream, and Wiener implementations. Further retirement must proceed by
measured algorithm-boundary tranche with corresponding mode validation, not by
mechanical replacement.

Run-owned profiling migration is complete locally without changing production
timing records. `ReductionSession` owns and resets a non-copyable
`StageProfileCollector`; the explicit owner now crosses loading, processor
selection, reduction, iteration, observation, generic output, engine setup and
pipeline, Pointing ordered-output, and Beammap internal and specialized-output
boundaries. Output-directory configuration, every production timing scope,
and sidecar publication use that owner. The process-static collector and the
legacy implicit adapter are deleted, and the collector is not stored in
`Engine`.

Tests prove sequential-run reset behavior and verify representative reduction,
observation, and map-output records in the supplied collector. Both local build
targets pass, all 451 CTests pass, and full config preflight passes after the
atomic cutover. Unity point `redu63` confirms unchanged products and profile-
sidecar behavior. Its profile contains the same 76 stage/context records as
accepted `redu62`; only elapsed values and the natural completion order of
concurrent chunk writes differ.

The first concrete lifecycle cut after profiling removes a duplicate collector
reset from `run_reduction_pipeline`. Reset policy now belongs only to
`ReductionSession`, and a regression test proves that records created before
scientific-pipeline entry survive in the same run-owned collector. This is the
bounded stale-state repair required by Phase 3 step 4; no observation or scan
state was moved without a demonstrated hazard.

The first real compiled implementation boundary is accepted.
Timestream enum name tables and parse/format definitions now compile once in
`src/citlali/core/config/timestream_enums.cpp`; the public header retains enum
declarations and small predicates. The header shrank from 946 to 712 lines and
the new source is linked through `citlali`. One immediate before/after CLI
compile pair was 62.4 versus 63.7 seconds, so this slice demonstrates neither a
build-time win nor a material regression. All three local targets build, all
451 CTests pass, and full config preflight passes. Unity compile and point
`redu63` accept the boundary with zero product differences and no runtime
regression attributable to the extraction.

The first bounded mature-implementation exit tranche is accepted for its point
coverage.
Two PTC weighting exits now classify non-contiguous network grouping as input
I/O failure and impossible counters as an internal failure. RTC kernel setup
classifies mismatched kernel-image cardinality as invalid configuration. Valid
paths and numerical loops are unchanged; focused contracts cover each error
class. The dependency audit now reports 51 library exits and zero CLI exits.
Point `redu63` exercises the unchanged valid PTC weighting path exactly. The
next production tranche is fruit-loop map ingestion and requires matched
science and Beammap validation after its local checkpoint.

The fruit-loop map-ingestion tranche is locally complete. All 37 exits in
`TCProc::load_mb` now become canonical config or input-I/O failures at the
session boundary. Required file discovery, FITS header/schema, grouping and map
identity, WCS, and cardinality diagnostics retain their concrete context.
Optional `GROUPING` and `RADESYS` handling ignores only missing-key exceptions,
preventing real schema failures from being swallowed after the move to
exceptions. Valid loading and numerical processing are unchanged. All three
local targets build, all 453 CTests pass, full config preflight passes, and the
session audit is down to 14 library exits with zero CLI exits. Matched science
and Beammap fruit-loop validation is pending.

The three adjacent fruit-loop feedback exits are also retired locally behind a
header-isolated invariant boundary. Non-contiguous calibration grouping,
unknown detector-array identity, and out-of-range map indices now become
session-owned input-I/O failures before the affected map access. Interpolation
and map-to-TOD loops are unchanged. All three local targets build, all 454
CTests pass, and the session audit is down to 11 library exits, all in serial
or OpenMP Wiener filtering. This change shares the pending science and Beammap
fruit-loop acceptance runs.

The Wiener failure-boundary tranche is locally complete. Shared runtime
contracts cover serial and OpenMP template geometry, pixel spacing,
kernel/weight identity and shape, finite kernel peaks, and FFTW resource
creation. The OpenMP allocation path captures exceptions inside each worker,
synchronizes before worksharing, and rethrows only after leaving the parallel
region; partial FFTW resources are reset before failure. Valid filtering and
denominator arithmetic are unchanged. All three local targets build, all 455
CTests pass, full config preflight passes, and the conservative session audit
now reports zero dependency-reachable library or CLI exits. Standard point and
focused full-Wiener point are accepted. Fruit-loop science and fruit-loop
Beammap validation remain required before accepting these final mature
tranches.

The exit audit now also scans every implementation source under
`src/citlali/core`, closing a blind spot in the original header-reachability
census. The wider scan found and retired three invalid APT-table exits and one
invalid Lissajous chunk exit. Manual review confines the remaining textual
exits to successful CLI help/version handling and two legacy main programs that
CMake does not build. No supported non-CLI path retains explicit process
termination.

Unity point `redu64` accepts the standard point path at `6dd0057f8`. Its merged
configuration is byte-identical to `redu63`; the strict complete-product gate
opens every RTC/PTC array and reports 21 common products, 2,041 comparison
records, zero changed records, and zero skipped records. The audit reports 56
files, 22 stable comparable products, 12 PTC chunks, no logged issues, and all
required provenance valid. Total log time is 174.880 seconds versus 169.728
seconds and PTC chunk spacing differs by 0.5%, so no performance regression is
attributed. The queued science config enables fruit loops but retains
`wiener_filter.lowpass_only: true`, so it exercises convolution rather than
Wiener denominator construction. It remains the fruit-loop science gate. A
focused point run with noise maps enabled and `lowpass_only: false` supplies the
full-Wiener denominator gate. The fruit-loop science and Beammap runs remain the
mode-specific acceptance gates for map ingestion and feedback.

The full-Wiener gate is accepted on matched OG `redu10` and refactor `redu65`.
Their 490 low-level leaves differ only in the OG/refactor output directory and
the corresponding telescope-file path; the two telescope inputs have identical
SHA-256 hashes. Both runs execute six Wiener core calls with five noise maps and
`lowpass_only: false`. The strict filtered-product comparison reads seven
products and 148 records with zero changed or skipped records under the
established `2e-8 + 1e-10 * abs(reference)` profile. The pointing-fit table is
exact across all columns. Maximum signal and kernel absolute differences are
`6.34e-9` and `7.99e-10`. Refactor non-uniform denominator work totals 38.4
seconds versus 42.9 seconds for OG; the uncontrolled pair shows no performance
regression. The refactor run has no logged issues and valid required
provenance. OG's twelve `NetCDF: Not a valid ID` records are its known legacy
limitation and are not accepted as refactor behavior.

Beammap `redu06` accepts the remaining mature Phase 3 tranches for that mode.
Its low-level config is byte-identical to accepted `redu05`; both runs complete
198 PTC chunks and expose the same valid provenance and product inventory. The
strict zero-tolerance comparison reads every comparable FITS, NetCDF, and ECSV
product, including complete detector TOD, and reports 12 common products,
16,453 records, zero changes, and zero skips. Total log time is 4,215.296
seconds versus 4,136.440 seconds, a 1.9% uncontrolled difference with no
performance attribution.

The matched science attempt at `6dd0057f8` stopped during configuration before
creating a `reduNN` directory. TolTECA emitted the historical
`timestream.output.rtcdiag.enabled` leaf, which the new complete startup schema
did not recognize. Because that diagnostic prevented installation of the raw
execution adapter, the later kernel-template check reported the misleading
secondary error `wiener filter kernel template requires kernel`. Commit
`7ef43ef93` explicitly classifies the historical switch as an ignored
compatibility spelling: RTC diagnostics remain required and are always
written. The Wiener prerequisite now reads typed raw policy instead of mutable
`rtcproc` state, reducing the checked legacy access census from 44 to 43. The
exact failed science YAML now passes local configuration and reaches the raw
data boundary; all 456 CTests and full config preflight pass. A Unity science
rerun is required before closing the science fruit-loop gate.

The first repaired science rerun was invalidated by two Citlali jobs sharing
the same output root while `fruit_loops.save_all_iters=true`. One job advanced
to `redu26` and attempted to read `redu25/coadded/raw` while the other job was
still writing observation products into `redu25`; the resulting missing-map
diagnostic correctly exposed the incomplete input. This is an output-directory
ownership failure, not evidence of a numerical or fruit-loop ingestion change.
Production session execution now holds a nonblocking filesystem lease on the
configured output root from successful runtime setup through final provenance
publication. A competing Citlali process fails immediately with a required-
output diagnostic, while reductions using distinct output roots remain
independent. Focused tests cover contention, automatic release, independent
roots, and public-header linkage. The CLI build, all 460 CTests, and full config
preflight pass locally. Clean single-job science sequence `redu28` through
`redu31` then completed normally at pre-lease commit `a7a35a00`: every
iteration consumed its immediately preceding complete map directory, the final
run logged no issues, and exact-config scientific equivalence against accepted
`redu23` passed. This closes the fruit-loop map-input repair gate. The output-
root lease then passed its first Unity/VAST exercise in point `redu66`: the
parent log records successful exclusive acquisition, the run completed without
issues, and all non-timing products are exact against `redu65`. This closes the
Phase 3 output-ownership and compiled-boundary gates.

The runtime domain is the first operational Phase 2 migration. Requested,
effective, and realized runtime state are now separate in memory, and execution
consumes the effective thread and runtime policy. Remaining direct mutable
runtime reads are confined to config construction. The required, atomically
published `runtime_provenance.yaml` sidecar uses the stable
`citlali-runtime-provenance-v1` schema. Unity `redu27` validates the sidecar,
zero serious log issues, and exact pre-existing point products. The runtime
domain is complete; the next operational domain is timestream output selection
and chunking.

The timestream-output domain routes RTC/PTC output shape, outer-buffer
allocation, NetCDF serialization mode, metadata, selection, and scan-index
construction through typed configuration. The required, atomically published
per-observation `timestream_output_provenance.yaml` carries the versioned
requested/effective/realized output record. Unity `redu28` validates all 12
selected and realized RTC/PTC chunks, both registered TOD files, zero serious
log issues, and exact existing products. The former processor output-mode and
telescope chunking mirrors are removed; parser and writer boundaries receive
typed values explicitly. The local CLI/test build, all 229 tests, and full
config preflight pass. This domain is complete.

Work has started on the `raw-timestream` domain. Downsample enablement,
requested factor/frequency, anti-alias validation, and effective sample-rate
preflight now use typed raw-time-chunk configuration. Frequency-derived factors
are synchronized into the RTC downsampler only as an execution adapter. A
divergence test proves typed policy wins over stale processor mirrors. Typed
policy also controls FIR/notch/IIR setup, kernel-dependent allocation and
products, flux-unit selection, and extinction setup; processor objects retain
the corresponding numerical state. All 231 tests pass. Remaining RTC flagging,
source-protection, line-audit, and diagnostics boundaries are being migrated in
bounded clusters.

Raw source-protection activation now flows requested typed policy to realized
typed state and then to the RTC execution adapter. Learned-mask application,
FITS event-mask provenance, and RTC diagnostic impulsive-product shape consume
typed policy directly. The shared processed source-protection activation follows
the same direction. All 232 tests pass; line-audit and remaining diagnostic
configuration are the next raw-timestream clusters.

The line-audit cluster now uses typed policy for model-protected PTC audit
activation, model-subtraction requirements, notch-family selection, iteration
counts, frequency overrides, and dynamic edge-guard decisions. RTC diagnostic
sidecars, TOD headers, and chunk summaries serialize requested raw settings from
typed config. Existing RTC notch methods still consume the processor options
object as a numerical adapter, and realized edge-context/guard sample counts
remain processor state. The CLI build, all 232 tests, and full config preflight
pass.

The processed migration now has its first explicit one-way adapter:
`TimestreamFruitLoopsConfig` synchronizes the numerical `PTCProc` fields after
loading. A focused divergence test proves typed values overwrite adapter state.
This enables direct typed parsing to replace legacy parsing incrementally. All
236 tests and full preflight pass.

Direct typed parsing now owns the core fruit-loop lifecycle and model-selection
fields before the one-way processor adapter runs. This is a staged extraction:
expert fruit-loop numerical fields still arrive through the legacy parser and
typed mirror until their cohesive reader is moved. All eight real config
profiles, all 236 tests, and full preflight pass.

The direct fruit-loop reader now covers the complete typed fruit-loop surface,
including expert local-noise, adaptive-support, feedback, interpolation, and
post-addback controls. The legacy combined PTC parser remains temporarily for
other processed domains; fruit-loop execution state is overwritten only from
typed policy through the adapter. All 236 tests and full preflight pass.

Processed cleaning now has a complete one-way typed adapter covering all four
cleaner modes and correlation grouping. The local build caught and corrected
an `int` versus `Eigen::Index` boundary conversion before Unity. All 236 tests,
the CLI build, and full preflight pass.

The cleaning reader now directly owns core activation and mode-selection
policy before the one-way cleaner adapter runs. Expert mode parameters and
eigen-count padding remain in the compatibility parser for the next slices.
All 236 tests, all eight real config profiles, and full preflight pass.

Direct cleaning parsing now includes standard-PCA eigen-count normalization
and both current and legacy key aliases. Empty and short vectors receive the
same defaulting and padding behavior before the one-way adapter runs. All 236
tests, all eight real profiles, and full preflight pass.

Direct cleaning parsing now covers correlation grouping and null-model scalar
policy. Group-name canonicalization remains deliberately mirrored because it
still depends on cleaner-specific helpers. All 236 tests, all eight real config
profiles, and full preflight pass.

Direct cleaning parsing now covers Marchenko-Pastur and adaptive-selector
numerical policy, including adaptive frequency-band validation. The remaining
cleaning-parser dependency is cleaner-specific grouping-name canonicalization.
All 236 tests, all eight real profiles, and full preflight pass.

Raw input and metadata boundaries now use typed policy for duplicate-tone
frequency separation, RTC diagnostic FIR/source-bandwidth ratios, and whether
FITS/TOD tau metadata is calculated. The atmospheric calibration object remains
processor-owned numerical state. The CLI build, all 232 tests, and full config
preflight pass.

Learning collection and learned-mask/exclusion orchestration now reads typed
second-pass source-protection activation, radius, and score thresholds. This
removes another execution-facing dependency on `PTCProc` policy mirrors while
leaving its numerical implementation unchanged. All 235 tests and preflight
pass.

RTC diagnostic and RTC TOD diagnostic schema construction now receives typed
downsample and impulsive-capture policy explicitly. External raw product-shape
decisions no longer depend on `RTCProc` mirrors. Remaining raw-timestream work
is concentrated in numerical-method adapters internal to `RTCProc`; polarimetry
is tracked as a separate authority domain.

The first processed-timestream authority slice now routes fruit-loop
enablement, effective iteration count, retained-iteration output layout,
initial/previous model-map paths, and learning source-model availability
through typed fruit-loop config. Beammap and disabled-loop normalization is
recorded in typed effective state and copied into `PTCProc` only as an execution
adapter. All 232 tests and the full config preflight pass.

Processed-timestream orchestration now also uses typed fruit-loop policy for
model subtraction/add-back, source-subtracted weight retention, final noise-map
population, and beammap adaptive-gate setup. Processor state remains the home
of runtime model buffers and numerical kernels, but no longer decides whether
these operations are enabled. All 233 tests and the full config preflight pass.
Interpolation override selection and fruit-loop runtime-policy logging now use
the same typed authority; the processor retains only the realized interpolation
mode required by map-to-TOD execution. All 234 tests pass.

TOD, PTC-diagnostic, and FITS-map fruit-loop metadata now serializes typed
effective configuration, including array flux limits and pointing source-center
policy. Pointing warnings use the same authority. Runtime detector fit vectors
remain in `PTCProc`. The CLI build, all 235 tests, and full preflight pass.

Compact PTC-diagnostic `CONFIG.*` metadata now also reads typed cleaning,
weight-penalty, busy-row, and second-pass policy. This establishes a consistent
boundary: typed configuration is serialized as policy, while processor-owned
arrays remain realized diagnostics. All 235 tests and full preflight pass.

TOD NetCDF and map FITS cleaning metadata now uses typed processed-timestream
policy throughout. The only retained cleaner value at this output boundary is
the per-array removed-eigenmode count, which is a realized result rather than
configuration. The CLI build, all 235 tests, and full preflight pass.

Weighting metadata now uses typed raw and processed policy for scheme,
cutoffs, hybrid correction, and validation settings. The PTC diagnostic
sampling-window duration remains an explicit realized processor input pending
a typed representation. The CLI build, all 235 tests, and preflight pass.

Optional PTC TOD diagnostic block selection now reads typed processed policy.
Second-pass, correlation, busy-row, and adaptive-cleaner schema decisions no
longer depend on processor mirrors. The CLI build, all 235 tests, and full
preflight pass.

Processed effective-policy resolution is now being separated from YAML
parsing. Pure result types preserve requested values while recording cleaner
group canonicalization, weighting source-mask inheritance, validated-weighting
and busy-row dependency decisions, and disabled/beammap fruit-loop iteration
normalization. Cleaner-mode precedence and fruit-loop interpolation defaults,
overrides, and JINC fallback now use the same pattern; source-protection
activation has an explicit realized-state result. Existing mutating calls
remain thin compatibility adapters with unchanged warnings and processor
values. A non-wired `ProcessedTimestreamExecutionPlan` now provides separate
requested, effective, effective-resolution, and realized storage without
claiming complete output provenance. The CLI and test builds, all 243 tests,
all eight config profiles, and the frozen 171-path PTC boundary audit pass.
The boundary audit now also routes all 171 paths to their declared typed
reader, requires each leaf key in that source, and fails preflight on uncovered
paths or stale compatibility aliases. This mechanically satisfies the path-
coverage prerequisite for removing the legacy parser; the provenance and
cross-mode validation prerequisites remain open.
Focused adapter tests now assign and verify every field copied from typed
fruit-loop, cleaning, weighting, validation, correlation-penalty, busy-row,
and second-pass configuration into the processor compatibility targets. The
full C++ suite passes all 244 tests. The concrete `PTCProc` header remains a
contextual include rather than an isolated test dependency; that existing
header-boundary defect belongs to Phase 3 and was not expanded in this phase.
The non-wired execution plan now has an atomic repeated-run reset operation.
Disabled sections retain their requested parameter values while remaining
inactive, and reset clears all prior effective-resolution and realized state.
All 245 C++ tests pass. Current legacy reader objects are not reset piecemeal;
the contract became operational only with the complete Engine wiring described
below.
Pure YAML component serializers now cover the complete requested/effective
processed snapshot surface. The boundary audit enforces serialization of all
171 frozen legacy paths as well as typed-reader coverage. There is deliberately
no final provenance schema version, output filename, or writer yet; effective-
resolution and realized-state component serialization now also use explicit
availability records. Beammap `redu14` (`4b0126e7`) completed cleanly and
exactly reproduces accepted refactor `redu11` across all 5,234 detector maps.
It also passes the versioned OG scientific-equivalence profile with exact
detector identities, flags, and product sets. The matched beammap gate is
therefore closed. `Engine` now owns and initializes the processed execution
plan, processed runtime accessors select its effective snapshot, and cleaner,
weighting, source-protection, interpolation, iteration-policy, and completed-
iteration decisions populate explicit resolution or realized records. The
legacy parser remains only as the compatibility seed and no provenance file is
published yet. Unity point `redu34` (`86c47fa7`) passes the strict complete-
product gate against accepted `redu33`: its 489-leaf config is exact, all 13
scientific product families are present, and every RTC/PTC timestream and map
record is exact with zero skipped records. The Engine authority change is
accepted; the versioned provenance root and atomic writer are next.
The v1 processed provenance sidecar is now implemented at the CLI success
boundary. It writes the authoritative plan only after completed iterations,
uses the shared atomic YAML writer, and fails the reduction on uninitialized
state or filesystem failure. Local CLI/test builds, all 252 tests, all eight
config profiles, and the 171/171 boundary audit pass. Unity output validation
of the new required sidecar passes at `81020d46` point `redu35`. The sidecar
contains all five effective-resolution and all three realized-state records;
its schema and hash are recorded in the validation ledger. Against accepted
`redu34`, the merged 489-leaf config and all 13 scientific product families,
including complete RTC/PTC timestreams, are exact with zero skipped records.
Point, beammap, and science processed provenance are accepted. The documented
compatibility-parser removal gate is closed.
Parser-removal preparation now includes a complete default-snapshot parity
test using a real value-initialized `PTCProc`. Typed defaults and the legacy
compatibility snapshot are identical across every serialized processed field.
Together with 171/171 reader coverage and exhaustive one-way adapter tests,
this closes the deterministic omitted-default prerequisite without changing
production parsing. All 253 C++ tests pass.
The six PTC-to-typed mirror calls are now consolidated behind
`seed_processed_timestream_config_from_legacy(...)`. Production still performs
the same compatibility seeding, typed reads, resolution, and one-way adapter
steps, but the legacy parser exit is now one named boundary. Local CLI/test
builds, all 253 C++ tests, config preflight, and provenance-audit tests pass.
Unity beammap `redu15` at `50235fd6` closes the beammap processed-provenance
gate. Its 529-leaf config is exact against accepted `redu14`; all 12 comparable
FITS, NetCDF, and ECSV products are exact with no skipped records; the required
sidecar passes semantic audit; and wall time improved from 3576.607 to 3458.917
seconds. The final matched science pair is OG `ffc6b907` `redu27` and refactor
`50235fd6` `redu24`; the intermediate `reduNN` directories are retained
fruit-loop iterations, not independent runs. Their 502-leaf configs differ only
in input/output path strings. Science-equivalence profile v2 preserves the
`1e-8` raw-map bound and separately enforces the owner-approved 1.5% filtered-
map bound. All 63 raw layers remain within `2.33e-11`; the 21 Wiener-filtered
layers peak at 0.986%; product sets and integer diagnostics are exact; all
other numerical bounds pass. Refactor wall time is 2686.252 seconds versus
2754.146 seconds for OG. The science processed-provenance gate is accepted and
recorded in the validation ledger.
The processed authority migration is now operationally complete in production:
`Engine::get_ptc_config` starts from typed defaults, reads all 171 paths through
typed readers, resolves the effective plan, and populates `PTCProc` only through
one-way execution adapters. The legacy parser call, compatibility seed, and all
processed PTC-to-typed mirrors are removed. The retired-boundary audit rejects
their reintroduction while preserving 171/171 reader and serializer coverage.
Local CLI/test builds, all 252 C++ tests, all eight config profiles, and 13
focused Python tests pass.
Unity point `redu36` at `c22bc127` closes the production parser-retirement
gate. Its merged config is an exact 489-leaf match to accepted `redu35`; all 13
scientific product families, including every RTC/PTC array, are exact with zero
missing, extra, changed, or skipped records. Processed and runtime provenance
are byte-identical; timestream-output provenance differs only in the expected
`redu35`/`redu36` paths. The run completed without serious log issues in 53.277
seconds versus 60.159 seconds for the baseline. This acceptance is recorded in
the validation ledger. The frozen 171-path inventory now lives in the versioned
`processed_timestream_legacy_paths.json` manifest. Boundary-audit schema v5
validates its canonical ordering, declared count, and digest before checking
171/171 typed-reader and serializer coverage. The unreachable
`PTCProc::get_config` declaration and roughly 1,190-line body are deleted.
Local CLI/test builds, all 252 C++ tests, all eight config profiles, and eight
focused boundary-audit tests pass after deletion. The processed-timestream
authority migration and its legacy-parser cleanup are complete.
Raw-timestream characterization is the next bounded Phase 2 domain. The frozen
RTC boundary contains 169 raw paths plus two adjacent polarimetry paths,
originally 14 direct parser exits, one production parser call, and ten
legacy-to-typed mirror helpers. The authority inventory now labels raw execution as legacy-authoritative
instead of incorrectly claiming a typed-to-legacy adapter. The finite transition
contract is `doc/raw_timestream_config_transition.md`. No RTC execution behavior
has changed. The non-wired preparation checkpoint now has 169/169 direct typed-
reader and request-serializer coverage. A 40-record external RTC access census
classifies 22 executor operations, six observation-state accesses, seven
output/realized-state accesses, one raw policy read, and four separate-domain
polarimetry accesses, with zero unreviewed records. An unwired execution plan
separates requested, context-free effective, per-observation, and realized
state and resets observation state between runs. All 260 C++ tests, 21 focused
config-tool tests, and all eight config profiles pass. Production remains
legacy-authoritative. The complete unwired typed-to-RTC adapter now covers all
169 raw paths, with a real-`RTCProc` request round trip, disabled-sentinel
checks, and a separate observation-state overlay for sample rate, downsampling,
edge context, source protection, and extinction. The frozen audit enforces
169/169 adapter coverage. All 264 C++ tests, 22 focused config-tool tests, and
all eight config profiles pass. Pure observation resolution now covers native
and effective sample rate, derived downsample factor and anti-alias checks,
filter edge guard/context contributions, source-protection activation, and
extinction-model selection. Filter transient estimates and extinction-model
selection are shared by the typed resolver and legacy processors rather than
duplicated. Focused tests prove edge-guard parity for sum/max policies and
extinction parity across representative tau values. All 271 C++ tests and full
preflight pass. Constructing the typed plan as a non-authoritative production
shadow is the next gate before the authority flip. That context-free shadow is
now active: the Engine directly reads an isolated typed request, constructs the
raw execution plan, adapts into a temporary RTC policy object, and requires its
deterministic 169-path snapshot to equal the legacy parser/mirror snapshot.
Legacy `rtcproc` still drives execution. The frozen audit requires one typed
read before the parser and one comparison after all ten mirrors. The generated
default config, disabled expert semantics, and injected divergence behavior are
covered by focused tests. The per-observation shadow is now active at the
existing lifecycle boundaries: input preparation records and compares native
and effective sample rate, downsample factor, edge guard/context, and raw source
protection; observation setup records and compares extinction activation and
model. Legacy `rtcproc` remains the execution authority. A second observation
resets the first observation's state and realized counters. Frequency-derived
downsampling exposes a pre-existing ordering gap because legacy configures its
edge guard before deriving the factor; that single comparison is explicitly
marked deferred rather than changing numerical behavior. All other divergence
fails with field-level diagnostics. The external RTC census is frozen at 44
classified records with zero review-required entries. Local CLI/test builds,
all 277 C++ tests, 23 focused config-tool tests, all eight profiles, and full
preflight pass. Unity validation of this shadow checkpoint is pending; no raw
authority flip or parser/mirror retirement is permitted before that gate.
The versioned `citlali-raw-timestream-provenance-v1` schema was prepared but not
yet wired at this checkpoint. It serializes the complete requested/effective
config, context-free resolutions, explicit observation-field availability and
edge-guard deferral, an execution-completed marker, and realized counters. Its
atomic writer rejects uninitialized plans and propagates publication failures.
All 281 C++ tests and full preflight pass. Production publication remained
deferred so required-output placement and lifecycle completion could be reviewed
with the Unity shadow checkpoint rather than introduced without mode evidence.
The remaining 14 direct exits in `RTCProc::get_config` are removed. Legacy
cross-field checks now append exact invalid-key paths to the existing config
diagnostics and continue safely through malformed notch vector shapes; the CLI
validation boundary remains responsible for rejecting the reduction. Valid
configuration behavior is unchanged. The frozen raw boundary now requires zero
direct parser exits. Local builds, all 282 C++ tests, and full preflight pass.
Unity point `redu37` accepts the complete raw-shadow checkpoint at `cd8da24f`.
The run used the same 489-leaf merged config hash as accepted `redu36`, completed
all 12 PTC chunks with zero logged issues, and retained the exact 36-file/14
stable-product inventory. Strict comparison including complete RTC/PTC
timestreams found 13 common product families, zero missing or extra products,
zero changed records, and zero skipped records. Runtime and processed
provenance are byte-identical; output provenance differs only in expected
`redu36`/`redu37` paths. Logged runtime was 51.723 seconds versus 53.277 seconds
for `redu36`. This closes the Unity point gate for observation shadowing,
prepared raw provenance, propagated parser diagnostics, and yaml-cpp 0.7
compatibility. Beammap/science evidence remains required before raw authority
flip and parser/mirror retirement.
The accepted point shadow gate now permits required production raw provenance.
Each successfully completed observation atomically publishes
`raw_timestream_provenance.yaml` in its observation directory after required TOD
writers and observation products have completed. The observation lifecycle owns
the completed-scan count and expected required TOD-write count; flagged-sample
and dynamic-notch counts remain explicitly unavailable rather than being
guessed from mutable RTC state. Publication failure propagates and fails the
reduction, and the writer rejects observation, completion, or realized-count
state that is incomplete. Repeated-observation tests prove state reset and
independent sidecars, while a filesystem-failure test proves required-output
propagation. The run-audit tooling can require and semantically validate every
observation's sidecar, including science reductions. It pairs setup-time output
provenance with completion-time raw provenance, rejects missing observation
sidecars, cross-checks scan counts, and validates resolved sample-rate state.
Local CLI/test builds, all 287 C++ tests, 11 provenance-audit tests, and full
config preflight pass. Unity point `redu38` accepts the required raw provenance
at `6bbc12ce`. It has the identical merged config and stable 14-product inventory
as accepted `redu37`, zero serious log issues, and a valid observation sidecar
recording 12 completed scans and 48 required writes. Strict comparison opened
all RTC/PTC arrays across 13 common product families and found zero missing,
extra, changed, or skipped records. Logged runtime was 51.459 seconds versus
51.723 seconds for `redu37`. The point publication gate is closed and recorded
in the validation ledger. Beammap and science acceptance remain pending; raw
execution therefore remains legacy-authoritative.
A science cross-mode attempt at `5d403887` stopped before observation numerical
processing because the shadow compared typed physical downsample factor 1 with
legacy RTC's disabled value 0. Legacy initializes and reads that factor only
when downsampling is enabled, so inspecting it while disabled is outside the
legacy contract and can read inactive state. Observation parity now always
compares enablement and compares factor only when enabled; typed observation
state still records the physical identity factor 1 and unchanged sample rate.
A focused science-style test preserves enabled-factor divergence detection and
accepts the disabled legacy sentinel. Local CLI/test builds, all 288 C++ tests,
and full config preflight pass. The science and Beammap gates must be rerun.
The repaired `2d6f80a3` candidate closes both cross-mode publication gates.
Beammap `redu17` has one complete raw sidecar with 198 scans and 198 required
writes; all 12 complete product families are exact against accepted `redu15`,
with zero skipped records and runtime 3397.522 versus 3458.917 seconds. Science
final iteration `redu29` has two complete raw sidecars, each with 124 scans and
248 required writes; all 27 complete product families pass the strict gate
against accepted `redu24` with zero changed or skipped records. Its largest
absolute difference is `4.452e-10`, within established tolerance, and runtime
is 697.572 versus 705.784 seconds. Both runs have zero serious log issues and
log each published sidecar path. The validation ledger records both accepted
checkpoints. Point, Beammap, and science prerequisites are now satisfied for
the bounded raw execution-authority cutover; OOF reuses the accepted pointing
execution gate, and polarimetry remains outside this authority claim.
The bounded raw execution-authority cutover is now implemented locally. Direct
typed parsing initializes requested/effective plan state and the one-way
production `RTCProc` adapter. The legacy parser and ten mirrors remain only as
a temporary read-only oracle whose deterministic snapshot must match the
production RTC before execution. Focused tests prove stale processor state is
overwritten, disabled requested values remain intact, and divergence fails.
The CLI build, all 291 C++ tests, all eight real config profiles, the complete
169-path boundary audit, and the frozen 44-record execution-read census pass.
Unity point, Beammap, and science cutover validation is the next gate; parser
and mirror retirement is prohibited until it passes.
The first Unity point cutover attempt at `475bf8e22` reached map output but
failed because the production `RTCProc` no longer received the adjacent legacy
polarimetry initialization. For an unpolarized run, that parser side effect
creates the mandatory Stokes-I entry; without it, `stokes_params` was empty and
map indexing read invalid state. A narrow legacy-polarimetry runtime adapter now
copies only enablement, grouping, and Stokes labels from the temporary parser
object. Polarimetry remains outside the raw authority claim. A focused
regression test and the boundary audit require this transfer. The repaired
candidate builds locally, all 292 C++ tests and all eight profiles pass, and
full preflight has zero drift. Unity point cutover validation must be rerun.
The repaired point run `redu40` completes with zero serious issues, all required
provenance valid, and exact scientific products and complete timestream arrays
against accepted `redu38`. The strict gate nevertheless rejects two metadata
records: disabled `CONFIG.TODIIRHP.FREQ_HZ` changed from the established
processor-effective sentinel `0.0` to the preserved inactive request `0.1` in
the RTC and PTC NetCDF products. Raw provenance correctly retains the request
and explicit disabled resolution, so the fix is a pure FITS/NetCDF metadata
projection rather than a plan mutation or processor readback. Disabled IIR
metadata now resolves to frequency `0.0`, order `1`, and zero-phase `false`;
enabled values pass through. All 293 C++ tests and full preflight pass locally.
One final point rerun is required before starting the expensive Beammap and
science cutover gates.
The raw execution-authority cutover validation gate is closed. Point `redu42`
at `880869b3` passes the complete strict comparison against accepted `redu38`:
13 common product families, zero changed or skipped records, valid byte-stable
raw/processed/runtime provenance, zero serious issues, and runtime 54.412 versus
51.459 seconds. Beammap `redu18` at `398d5127` has exact numerical products and
all 5,234 detector results against `redu17`, zero skipped records, valid
byte-stable provenance, and zero serious issues. Its six accepted rtcdiag
metadata changes expose configured values beneath a disabled local-residual
section instead of legacy processor defaults. Science final iteration `redu33`
at `398d5127` passes against `redu29` with 27 common products, zero changed or
skipped records, maximum absolute difference `3.746e-10`, byte-stable
provenance, zero serious issues, and runtime 704.234 versus 697.572 seconds.
The validation ledger records all three accepted gates. OOF reuses the point
execution gate; polarimetry remains separate. The temporary 169-path raw parser
and ten oracle mirrors may now be retired as the next bounded change while
retaining the narrow adjacent polarimetry compatibility boundary.

The authorized raw-parser retirement is complete locally. The declaration and
roughly 1,080-line `RTCProc::get_config` implementation, all ten raw reverse
mirrors, and the context-free parity oracle are removed. The versioned
`raw_timestream_legacy_paths.json` manifest preserves the canonical 171-path
historical surface and digest. The boundary audit now rejects reintroduction of
the parser, a raw mirror, or the parity comparison while continuing to enforce
169/169 direct-reader, serializer, and typed-to-RTC adapter coverage. The two
adjacent polarimetry keys use a dedicated compatibility reader and one-way
runtime adapter; they do not repopulate raw typed state. A forward TOD output-
context helper formerly hidden in the mirror umbrella now has its own named
header. Fresh local CLI, primary-test, and safety-test builds pass all 285 C++
tests; 12 focused raw-boundary audit tests, the unchanged 44-record execution
census, all config profiles, full preflight, and the validation ledger pass.
Unity point `redu43` at `11afd6f6` closes the retirement gate against accepted
`redu42`. The merged 489-leaf config is exact, all 13 product families and
complete RTC/PTC arrays are exact with zero changed or skipped records, all
required provenance is valid, and raw, processed, and runtime sidecars are
byte-identical. Output provenance differs only in the expected reduction-number
file paths. The run has zero serious issues and completed in 54.182 seconds
versus 54.412 seconds. The validation ledger records the acceptance. The raw-
timestream authority migration, including legacy parser/oracle cleanup, is now
complete; polarimetry remains a separate compatibility domain.

The mapmaking authority migration has passed its first Unity mode gates. All
22 frozen `mapmaking.*` leaves now enter typed request state through
one boundary. `MapBuffer`, JINC, maximum-likelihood, observation-map, and
coadd-map configuration no longer parse YAML. One-way adapters construct the
legacy numerical mapmakers and WCS buffers from typed state. The immutable
execution plan preserves the requested grouping while exposing the resolved
effective grouping to downstream accessors; the transitional root request is
no longer mutated by map-count setup. Successful reductions must atomically
publish versioned `mapmaking_provenance.yaml`, and write failures propagate.
The effective plan also records the uncalibrated TOD-type unit substitution
without changing the requested `cunit`. Version-2 provenance now records one
identified observation per input in the final fruit-loop iteration, each
observation's map count, effective pixel size, required logical map-product
count, optional coadd cardinality, and completion state. Lifecycle counters
reset between fruit-loop iterations and advance only after required output
stages return successfully; CLI completion rejects incomplete or inconsistent
counts. The audit accepts historical version-1 sidecars but applies strict
cardinality semantics to version 2. The boundary preflight freezes the
22-path digest, enforces 22/22 reader
coverage, rejects retired parser symbols, and checks the production authority
sequence and provenance writer. Local CLI/test/safety builds, all 305 C++
tests, all eight config profiles, and the full preflight pass. A strict point
run is required first to validate the lifecycle wiring and new sidecar;
Beammap and science runs then validate their mode-specific output cardinality.
This Unity validation is the last mapmaking provenance sub-gate. Unity point
`redu44`, final science
iteration `redu03`, and Beammap `redu00` all embed `5c8f5eb4`; their merged
configs are exact against accepted `redu43`, `redu33`, and `redu18`
respectively. All three runs have zero serious log issues and valid mapmaking,
raw, processed, output, and runtime provenance. Point has 13 exact complete
product families including RTC/PTC TOD. Science has all 27 products with zero
skips and passes the scientific-equivalence profile; its largest map
RMS-relative difference is `5.87e-14`. Beammap has exact non-map products,
exact identity and flags for all 5,234 detectors, and zero RMS difference in
every accepted good/bad signal, weight, and kernel map. Point, science, and
Beammap runtimes are 55.341, 699.904, and 3483.362 seconds, respectively,
versus 54.182, 704.234, and 3580.078 seconds for their baselines.

Version-2 cardinality validation is accepted at `e8e42945`. Point
`redu45` is exact against `redu44`: its 489-leaf merged config is unchanged,
all 13 product families including complete RTC/PTC arrays compare exactly, the
strict audit reports zero issues, and runtime is 56.176 seconds versus 55.341
seconds. Final science iteration `redu07` is accepted against `redu03`: its
502-leaf merged config is unchanged, all 27 products are present with no
skips, the dedicated science-equivalence profile reports a maximum map RMS-
relative difference of `6.23e-14`, and runtime is 709.597 seconds versus
699.904 seconds. Both version-2 sidecars report complete, internally
consistent observation/coadd cardinality. Beammap `redu01` is exact against
`redu00`: its 529-leaf merged config is unchanged, all non-map ECSV/NetCDF
products compare exactly, all 5,234 detector identities and flags are exact,
and every accepted good/bad signal, weight, and kernel map has zero RMS
difference. Its strict audit reports zero issues, 198 completed PTC chunks,
and one completed 5,234-map observation with no coadd; runtime is 3449.262
seconds versus 3483.362 seconds. The validation ledger records all three
accepted runs. The mapmaking authority and provenance domain is complete.

The bounded coadd authority domain is implemented locally without changing
coaddition numerics. Its frozen one-path reader owns `coadd.enabled` and
preserves the requested value. `CoaddExecutionPlan` resolves effective
activation from the mapmaking plan without mutating that request. Successful
CLI reductions require atomic `coadd_provenance.yaml` using schema
`citlali-coadd-provenance-v1`; its realized map and required-write cardinality
is a one-way snapshot of the already validated mapmaking coadd lifecycle, and
the reduction audit rejects disagreement between the two sidecars. The legacy
coadd reader and reverse mutation helper are removed. Local CLI/test builds,
all 314 C++ tests, all 38 focused config tests, 24 reduction-audit tests, all
eight config profiles and full preflight pass. Unity point `redu46` at
`c2e053b3` closes the disabled-coadd gate against accepted `redu45`: all 489
config leaves and all 13 complete scientific product families, including RTC
and PTC timestream arrays, are exact with zero skipped records or serious log
issues. The new coadd sidecar records requested/effective disabled activation,
no execution or cardinality, and agrees with the unchanged mapmaking sidecar.
All prior provenance is byte-identical except the expected reduction-number
TOD paths. Runtime is 53.804 seconds versus 56.176 seconds. Final science
iteration `redu11` at `c2e053b3` closes the enabled-coadd gate against accepted
`redu07`: all 502 config leaves match, all 27 products are present with zero
skipped records or serious log issues, and the science-equivalence profile
accepts a maximum map RMS-relative difference of `7.65e-14`. Coadd provenance
records requested/effective enabled, successful execution, three maps, six
required logical writes, and completed outputs; every value agrees with
mapmaking provenance. Runtime is 719.154 seconds versus 709.597 seconds. The
33-record validation ledger passes. The coadd authority and provenance domain
is complete.

The bounded `noise-products` implementation checkpoint is complete.
The six frozen `noise_maps.*` inputs now have one direct typed reader, a
requested/effective/realized `NoiseExecutionPlan`, and a one-way adapter into
the mature observation/coadd map buffers. The existing deterministic Boost
MT19937 identity is now explicit and versioned as fixed internal seed `5489`;
no user-facing seed knob was added. Required atomic
`noise_products_provenance.yaml` records activation/count resolution, final-
iteration observation/coadd realization cardinality, empirical-product count,
realization-image count, and completion. The reduction auditor validates those
semantics and cross-checks scientific-map cardinality against mapmaking v2
provenance. The legacy noise readers and reverse request mutations are retired.
The CLI/test build, all 328 CTest cases, all eight config profiles, the frozen
six-path audit, 48 config-boundary tests, and full preflight pass. No noise-
generation or product algorithm changed.

Unity point `redu47` at `1faec7cc` closes the disabled-noise path against
accepted `redu46`: all 489 config leaves and all 13 complete product families
are exact, with no skipped records or serious log issues. Point `redu49`
closes the bounded full-output fixture with ten realizations per scientific
map, three empirical-product maps, and 30 realization-image writes. Its
realization, empirical-variance, and empirical-weight outputs agree with the
matching OG fixture at maximum RMS-relative differences of `7.65e-14`,
`8.84e-14`, and `6.42e-14`, respectively. The final science iteration
`redu15` closes the generation-only coadd path: six observation maps produce
60 realizations and three coadd maps produce 30, for exactly 90 total with no
optional empirical products or realization files. Its 502-leaf config is
exact against accepted `redu11`; all 27 scientific products are present with
no skips, and the science-equivalence profile accepts a maximum map RMS-
relative difference of `6.93e-14`. Against the matching OG science run, the
profile accepts the previously approved filtered-map differences with maximum
map RMS-relative difference `0.00986`. All three candidate runs have valid
version-1 noise provenance and zero serious log issues. The noise-products
authority and provenance domain is complete.

The bounded pointing implementation is locally complete. Its frozen five-key
surface now has a direct typed request reader, a separate effective execution
plan, and a one-way adapter for the three mature PTC source-center fields.
Effective fit activation preserves the request and depends only on availability
of normalized observation maps from mapmaking. Optional filtering and coaddition
occur downstream and do not disable raw pointing fits. Required atomic
`pointing_provenance.yaml` records the request, resolution decisions,
per-observation map/fit cardinality, and realized completion. The reduction
auditor validates those semantics and cross-checks observation identity and
map counts against mapmaking v2 provenance. The CLI/test builds, all 336 CTest
cases, the frozen boundary audit, all eight compact profiles, and full config
preflight pass. Gaussian fitting, Ceres use, source finding, and map numerics
are unchanged. Unity point validation remains the sole exit gate before this
domain is complete.

The first Unity candidate, point `redu50` at `98d2a5d2`, correctly exposed an
effective-policy error. Its 489-leaf config, maps, timestreams, diagnostics, and
all non-fit products are exact against disabled-noise `redu47`, and it has zero
serious log issues. However, the new plan incorrectly treated disabled map
filtering as making pointing fits unavailable. The resulting three-row pointing
table zeroed all 11 fitted columns instead of preserving the accepted fits. The
gate is failed. Pointing fit availability now follows mapmaking alone; the
semantic auditor rejects the invalid `redu50` sidecar, and focused tests cover
both filter-independent fitting and mapmaking-disabled fitting. Local builds,
all 336 CTests, 43 baseline-tool tests, 54 config tests, all eight profiles, and
full preflight pass. A corrected Unity point run remains required.

Corrected Unity point `redu51` at `a9d17fa1` closes the pointing gate. Its
489-leaf merged config is exact against accepted disabled-noise `redu47`; all
13 scientific product families, including every RTC/PTC timestream record and
all pointing-fit columns, are exact with zero changed or skipped records. The
candidate has zero serious log issues and valid pointing provenance recording
one observation, three scientific maps, three fit attempts, and three valid
fits. Runtime is 59.971 seconds versus 58.627 seconds. The validation ledger
records the accepted checkpoint. The pointing authority and provenance domain
is complete; post-processing is now the active bounded domain.

Post-processing characterization freezes 35 supported leaves: 24 under
`post_processing.*` and 11 under the historical top-level `wiener_filter.*`
prefix. The latter controls filter template construction and convergence and
therefore belongs to the same authority domain. The starting boundary is
intentionally mixed: the legacy Wiener parser still reads 21 leaves and
reverse-mirrors most of them into typed state, while direct typed readers cover
13 other leaves. The initial typed-request gaps,
`post_processing.source_fitting.model` and
`wiener_filter.kernel_template_tail_mode`, now have closed-enum representation
in a complete 35-leaf direct request reader. That reader now runs during
`Engine` config loading as a fail-fast, read-only shadow. Activation and
histogram always compare; detail fields compare only when the legacy path
loads them, so disabled requested values are preserved without false mismatch
reports. The legacy parser and reverse mirrors still drive execution. Focused
shadow tests cover inactive science policy, pointing fit values, active filter
values, and mismatch diagnostics. The CLI/test builds, all 342 CTests, 60
config tests, all eight compatibility profiles, and full preflight pass. See
`doc/POST_PROCESSING_CONFIG_AUTHORITY.md`.

Unity point `redu52` at `d9db1183`, the first enabled-filtering overlay,
reached both the raw and filtered
pointing-fit stages, then failed during lifecycle recording with `pointing fit
results already recorded`. This exposed a provenance-model defect rather than
a fitting or mapmaking failure: version 1 represented only one fit event per
observation even though filtered pointing output deliberately fits the maps a
second time. The execution plan now names the raw and filtered fit stages,
enforces exactly one result per expected stage, and records their cardinalities
separately in `citlali-pointing-provenance-v2`. The reduction auditor accepts
both historical v1 and current v2 sidecars and validates stage expectations
against filtering/coadd policy. Numerical fitting and product-writing order are
unchanged. Local `citlali_cli`/test builds, all 344 CTests, 45 provenance-tool
tests, 60 config tests, all eight compact profiles, and full preflight pass. The
same enabled-filtering point overlay must pass on Unity before post-processing
authority migration proceeds.

Unity point `redu53` at `c75f079b` closes that repair and enabled-filtering
gate. It completes in 59.772 seconds with zero serious log issues. Its v2
pointing sidecar records one observation, three raw and three filtered fit
attempts, all valid, and completed output. All 13 products shared with accepted
unfiltered refactor `redu51` are exact, proving the overlay and lifecycle repair
did not alter the raw path. Against matching OG point `redu09`, all eight
filtered products are present with no skipped or changed records under the
standard numerical gate; the three-row pointing-fit table and 195-row source
table are exact. Maximum filtered signal absolute difference is `2.97e-11`.
The 490-leaf merged configs differ only in their two expected output paths.
The validation ledger records the accepted checkpoint. Post-processing may now
advance from request shadowing to a separate effective execution plan.

The first post-processing authority checkpoint is complete locally. A
`PostProcessingExecutionPlan` now owns the immutable 35-leaf request, a
separate effective snapshot, explicit resolution reasons, and reset realized
state. Effective map filtering and source finding are suppressed only when
mapmaking is unavailable; pointing and Beammap source fitting remains required
whenever mapmaking is available, independent of optional filtering. The plan
is constructed once during config loading and the legacy state is still
compared against its request. Production filtering, finding, fitting, and
output consumers have not been switched yet, so this checkpoint changes no
numerical or output behavior. Focused plan and frozen-boundary tests, all 347
CTest cases, all eight compatibility profiles, and full config preflight pass.
The next bounded cutover at that checkpoint was the one-way typed map-filter
adapter, followed by source finding and source fitting; accepted `redu53` is
the validation baseline after a consumer cutover, not for plan construction
alone.

The map-filter consumer cutover is complete and accepted. The duplicate serial and
OpenMP Wiener YAML parsers and the reverse Wiener-to-typed mirror are removed.
A single one-way adapter copies the effective typed filter snapshot into the
mature numerical target while preserving conditional Gaussian/Airy FWHM
loading and arcsecond-to-radian conversion. Filter activation, runtime noise/
kernel dependency checks, required filtered-output policy, and map-diagnostic
edge-guard metadata now consume effective typed policy. The Wiener algorithms,
map arrays, and output ordering are unchanged. The frozen audit rejects parser,
reverse-mirror, output-policy, or adapter drift. Local CLI/test builds, all 347
CTest cases, 60 config tests, all eight compatibility profiles, and full
preflight pass. Unity point `redu54` at `a89e0ee5` reruns the unchanged
enabled-filtering overlay with zero serious log issues, all required provenance
valid, and the same 21-product inventory as `redu53`. Its 490-leaf merged
low-level config is byte-identical to `redu53`; all 2,041 compared records pass
the established tolerance with no skips, and all 639 non-PTC records compared
against matching OG `redu09` pass as well. The 16 non-bitwise records are
confined to three filtered a1400 products, have no finite-mask mismatch, and
have maximum absolute difference `8.73e-11`.

The source-finding consumer cutover is complete locally. Its duplicate YAML
parser and observation-to-coadd reverse mirror are removed. One adapter writes
`source_sigma`, the arcsecond-to-radian source window, and finder mode directly
from the effective typed plan to the observation map buffer and, when enabled,
the coadd map buffer. Source-finding execution and output activation now use
the same effective authority. The legacy shadow retains activation parity but
no longer compares details that legacy state does not own. Detection, fitting,
map arrays, source tables, and output order are unchanged. Both local targets
build, all 349 CTests and 61 config-boundary tests pass, all eight compatibility
profiles pass, and full preflight is clean. Unity point `redu55` at `aa593a2b`
closes this gate with zero serious log issues, all required provenance valid,
and bit-for-bit identity across all 2,041 records in the 21 common products
against `redu54`, including full RTC/PTC timestreams, 195 source rows, and both
pointing tables. The 490-leaf config is byte-identical to `redu54`; all 639
non-PTC records also pass against matching OG `redu09`. Source fitting is now
the active bounded consumer cutover.

The source-fitting consumer cutover is complete and accepted. The mixed
YAML-to-`mapFitter` parser is removed. A standalone
one-way adapter now projects the effective typed fitting request into the
mature fitter target, preserving arcsecond-to-pixel conversion, fit-angle
policy, two-element amplitude/FWHM vectors, and the historical rule that a
nonpositive limit factor retains the fitter's established default. The
Gaussian fitting implementation and its numerical inputs are otherwise
unchanged. Source-fitting details are no longer copied into or compared
against legacy config state; the temporary legacy shadow now covers only the
remaining activation and histogram values it actually owns. Both local
targets build, all 350 CTests and 62 config-boundary tests pass, all eight
compatibility profiles pass, and full preflight is clean. Unity point `redu56`
at `9f8ad50e` closes the gate with zero serious log issues, all required
provenance valid, the same 50-file inventory, byte-identical 490-leaf merged
config, and bit-for-bit identity across all 2,041 records in the 21 common
products against `redu55`, including the 195-row source table and complete
RTC/PTC timestreams. Realized post-processing state and required provenance are
now the active bounded work.

The realized post-processing implementation is complete and accepted for the
point workflow. Per-iteration state records observation and coadd filter
contexts and map counts; source-finding contexts, detected candidates, catalog
fit attempts/valid fits, and successfully written source-table rows; raw and
filtered pointing fit contexts; and Beammap fit contexts. These fitter families
remain separate by project-owner decision rather than being collapsed into one
ambiguous total. Completion rejects missing or inconsistent cardinality and is
cross-checked against completed mapmaking. Source finding without map filtering
is now a fail-fast configuration error because the supported execution path
operates only on filtered maps.

The CLI publishes required atomic `post_processing_provenance.yaml` using
`citlali-post-processing-provenance-v1` only after successful pipeline output
and realized-state completion; write or lifecycle failures fail the reduction.
The reduction auditor validates internal cardinality and activation semantics,
cross-checks filter map counts with mapmaking v2, and cross-checks raw/filtered
pointing fit totals with pointing v2. The frozen source-boundary audit requires
the lifecycle hooks, schema, atomic writer, and single CLI completion/write
calls. Local `citlali_cli` and `citlali_test` builds pass, all 357 CTests pass,
43 reduction-auditor tests pass, all eight compact profiles pass, and full
config preflight is clean. No filter, source-detection, Gaussian-fit, or map
numerical algorithm was changed.

Unity point `redu57` at `f8a4a596` closes the point gate. It has zero serious
log issues, a valid required sidecar with one observation filter/source/table
context, 195 source rows, and separate three-map raw/filtered pointing fit
contexts. Its 490-leaf merged config is byte-identical to `redu56`, and all
2,041 records in the 21 common products, including full RTC/PTC timestreams,
are exact. Science must still exercise coadd-only filtering/source routing and
Beammap must exercise iterative detector-fit cardinality. Those expensive mode
gates are intentionally batched until after the remaining activation-only
legacy shadow is retired locally; the domain is not complete until both pass.

The activation-only compatibility shadow is now retired locally. The complete
typed post-processing request is loaded once before mapmaking setup, owns the
histogram setting that map buffers consume through the existing one-way
adapter, and initializes the effective execution plan without a second YAML
activation pass. Disabling mapmaking no longer mutates requested filtering,
finding, or fitting policy; effective suppression remains the execution plan's
responsibility. The established no-map Beammap single-iteration optimization
is preserved separately. Both local targets build, all 355 CTests and 63
config-boundary tests pass, all eight compact profiles pass, and full preflight
is clean. This cleanup still requires a point run after Unity compilation; it
is not covered by the preceding `redu57` acceptance. Unity point `redu58` now
closes that gate with the same config and post-processing provenance hashes,
zero serious log issues, and exact identity across all 2,041 records in the 21
common products, including full RTC/PTC timestreams.

The next Beammap authority domain is characterized without changing runtime
behavior. A versioned manifest freezes all 74 `beammap.*` leaves; there are no
known typed-model gaps, and config literals remain confined to the declared
loading and validation boundary. One typed-to-legacy adapter copies only the
fit support radius into the mature `map_fitter`. Dedicated requested/effective/
realized Beammap provenance is explicitly missing. The six-test static audit
is part of the full preflight and will reject surface, reader-boundary,
authority, or adapter drift.

The final post-processing mode gates are accepted. Science final iteration
`redu19` at `342a021c` has zero serious log records and valid required
provenance. Its realized record contains no observation filter contexts and
exactly one coadd filter context with three filtered maps. Against accepted
science `redu15`, the low-level config is byte-identical and the strict full-
depth comparison finds 27 common products, no missing or extra products, no
skipped records, and no changed records outside the standard tolerance.

Beammap `redu02` at the same commit has zero serious log records and valid
required provenance. Its realized record contains exactly three detector-fit
contexts with 15,407 attempts and 15,407 valid fits. Against accepted Beammap
`redu01`, the low-level config is byte-identical and the strict full-depth
comparison, including complete detector TOD and split FITS maps, finds 12
common products with no missing, extra, skipped, or changed records. The
profiling sidecar differs only in elapsed timing and is excluded from the
scientific gate. Post-processing authority and provenance are complete.

Project-owner decision (2026-07-10): every output explicitly enabled in the
configuration is required. RTC TOD, PTC TOD, `rtcdiag`, and `ptcdiag` write
failures must fail the reduction. There are no best-effort enabled products.

Immediate work order:

1. Begin the bounded Beammap effective-plan and provenance migration using the
   accepted sequence in the Beammap authority review.
2. Ask only the owner questions needed by the first Beammap implementation
   cut; do not silently change phase, prior, split, reference, or source-flux
   behavior.
3. Preserve Gaussian fitting, prior matching, detector flagging, RTC/PTC,
   mapmaking, and all other mature numerical algorithms.
4. Keep compact-config rollout, polarimetry expansion, and Phase 3 compiled-
   boundary work paused.

### Parallel Review Synthesis - 2026-07-14

Three read-only reviews were completed and adopted as advisory detail under
this living roadmap:

- [Phase 2 completion census](../handoff/PHASE2_COMPLETION_CENSUS_2026-07-14.md)
- [Beammap authority design review](../handoff/BEAMMAP_AUTHORITY_DESIGN_REVIEW_2026-07-14.md)
- [compact configuration and TolTECA usability review](../handoff/CONFIG_USABILITY_TOLTECA_REVIEW_2026-07-14.md)

They agree with the active sequence and expose no reason to reopen the nine
completed authority domains. Phase 2 remains incomplete: Beammap and the
minimal KIDs external boundary are implementation-ready; polarimetry and atomic
astrometry/photometry still require scientific-policy decisions. Domain-level
completion must not be mistaken for the global Phase 2 exit gate.

After the post-processing gates close, the adopted shortest sequence is:

1. Complete the bounded Beammap effective-plan and provenance migration,
   preserving all mature numerical algorithms.
2. Complete atomic Beammap photometry observation configuration, including
   replacement rather than merging of per-observation calibrator flux. Keep
   source identity in telescope data and leave flux estimation to TolProj.
3. Record the minimal external KIDs schema/config identity and the durable
   ordered configuration-source manifest.
4. Mechanically disposition polarimetry as either supported and validated or
   rejected as an unavailable capability.
5. Run current matched point, OOF, Beammap, and science snapshots on the final
   Phase 2 candidate before beginning Phase 3.

The frozen 74-leaf `beammap.*` manifest is the correct Beammap policy boundary,
not a claim to contain every scientific input used by a Beammap reduction.
`beammap_source.fluxes` remains an adjacent photometry input. The
review identified a concrete stale-state risk there: a later observation can
inherit a per-array source flux omitted from its own input. The Beammap work
must therefore reference an atomically constructed observation photometry value;
it must not absorb that adjacent domain or preserve merge semantics.

For Phase 2, "reviewed overlay fixtures" means retained matched low-level mode
overlays plus durable ordered-source evidence. Compact-config production
deployment and its full hermetic TolTECA numbered-overlay acceptance suite are
explicitly deferred rollout blockers, not Phase 2 exit requirements. Current
`*_standard` compact profiles remain translation prototypes and must not be
presented as approved operational defaults. Normal compact controls must also
be audited in both directions: user-facing low-level paths must be reachable,
and ordinary compact fields must not write expert-only policy.

Open scientific and operational choices listed in the reviews will be asked
only when the next implementation depends on them. They must not be inferred
silently. In particular, Beammap source-flux failure behavior, phase/prior/
split/reference fallbacks, HWPR and polarimetry capability, astrometry frame
and time rules, supported KIDs types, and ownership of ordered TolTECA source
provenance remain owner decisions.

### Phase 1 Progress

- The 12 `NetCDF: Not a valid ID` errors in `redu21`/`redu22` were traced to
  the PTC TOD stream, one error per requested output scan. The schema omitted
  four second-pass rejection/source-protection variables that the append path
  wrote unconditionally. Signal, flags, weights, and earlier diagnostics had
  already been written before each exception, which explains why pairwise
  numeric comparison passed despite incomplete diagnostics.
- The PTC TOD schema now includes all four fields. A focused NetCDF schema test
  creates the file layout and checks their presence. Local `citlali_cli` build
  and `citlali::safety::ptc_tod_schema.includes_all_second_pass_summary_fields`
  pass. Unity reduction validation is pending.
- CTest is now enabled at the project boundary and the focused safety target is
  discoverable from the normal top-level build directory.
- Parsed enum failures now enter the authoritative invalid-key diagnostics
  instead of silently retaining their typed default. Legacy authoritative
  range parsing and typed validation reject NaN and infinity for ordinary
  numeric fields. The four documented line-frequency inheritance fields retain
  their explicit NaN sentinel but reject either infinity. Focused parser and
  finite-value tests pass locally.
- Required RTC TOD, PTC TOD, `rtcdiag`, and `ptcdiag` NetCDF failures now retain
  the failing path in an error diagnostic and propagate out of the reduction.
  Ordered writers cancel as one output domain, so a failure wakes workers
  waiting on the same or another product stream instead of deadlocking. Focused
  serialization, cancellation, and cross-stream cancellation tests pass.
- A real fixed-size NetCDF failure test now writes the first row, injects an
  out-of-range second write, verifies that a waiting third writer is cancelled
  and the partial product is explicit, confirms a nonzero CLI result, then
  recreates and completes the product with a fresh writer domain in the same
  process.
- The owner-thread failure state now lets Pointing, Lali, and Beammap rethrow
  required output failures after GrPPI worker drainage, so the normal CLI error
  boundary can report them without an exception escaping a worker thread.
- Disabled IIR and extinction mirrors now preserve legacy effective provenance:
  IIR uses frequency `0`/order `1`/zero-phase `false`, and extinction uses
  `N/A`. Enabled values are unchanged. Four focused mirror tests pass.
- Reduction audit comparison now treats any error-level log record as blocking;
  `redu22` correctly fails the audit with 12 errors while the clean `redu23` to
  `redu24` comparison passes.
- Reduction product comparison now has an explicit strict mode. It fails on
  product-set differences, skipped items, or changed records. A complete TOD
  comparison of `redu23` and `redu24` passes with zero changes/skips when the
  volatile profile sidecar is explicitly excluded; retaining that sidecar or
  the default large-array cap correctly fails the gate.
- The pre-existing `citlali_test` target was found to have substantial test
  infrastructure and source decay. It has now been decoupled from the obsolete
  Google Benchmark runner, modernized for typed config and explicit alignment
  and output-path ownership, and reactivated with all 201 declared legacy tests
  passing. The seven utility tests that had remained inside a block comment now
  exercise the current Tula APIs with assertions. Together with the 18 focused
  safety tests, CTest discovers and passes 219 tests with none skipped or
  disabled. The local CLI build and complete config preflight continue to pass.
- Enabled timestream products now carry mode- and config-derived expected write
  counts. Pointing, Lali, and Beammap verify RTC TOD, PTC TOD, `rtcdiag`, and
  `ptcdiag` cardinality after worker drainage and before map finalization, so a
  silently omitted required chunk fails even when no individual write throws.
- Main timestream scan generators now own their cursors per pipeline invocation
  instead of sharing function-local static counters. Focused tests prove exact
  enumeration and a clean scan-zero start after an earlier cursor is abandoned.
- `redu25` (`c2ec8ae5`) finished with zero serious log issues and the same
  complete 33-file/14 stable-product inventory as `redu24`. Scientific arrays,
  maps, and tables are exact. The only strict-comparison differences are the
  intended disabled-IIR effective-provenance changes in RTC/PTC metadata.
- Beammap detector-specific TOD now obeys the required-output policy. Config
  preflight rejects enabled output with no slots or non-detector map grouping;
  unavailable scans, PTC samples, or pointing fail at runtime instead of
  silently skipping the declared product.
- Enabled learning diagnostics now fail on open, write, flush, or close errors.
  Required Beammap PTC TOD metadata updates likewise fail when the file or
  `FRUITLOOPS_ITER` variable is unavailable.
- ECSV table output is now published atomically through a temporary file.
  Failure removes the temporary product and propagates instead of silently
  substituting a differently named ASCII table.
- `validation/accepted_runs.json` is the checked-in machine-readable validation
  ledger. Its first record captures the accepted `redu25` point checkpoint,
  including explicit unavailable provenance and the two intended metadata
  differences. A standard-library validator enforces its core consistency
  rules.
- `redu26` validates the full current Phase 1 checkpoint at `9ef7da8a`. It has
  zero serious log records, the same complete 33-file product inventory and
  merged-config hash as `redu25`, and zero changed or skipped records in the
  strict comparison including every TOD array. Total logged runtime was 59.25
  seconds versus 61.51 seconds for `redu25`; this is recorded as run variation,
  not a performance conclusion.
- Phase 2 preparation now has a checked authority inventory covering 13 config
  domains. It enforces the one-way requested-YAML to typed-config to legacy
  adapter contract and records a concrete exit gate for each domain. Seven
  domains remain materially mixed, four are typed-authoritative without an
  adapter, Beammap is typed-authoritative with one fitting adapter, and KIDs is
  an explicit external boundary. This checkpoint changes no runtime behavior;
  operational authority migration remains gated on the remaining Phase 1
  validation decisions.
- Phase 1 science validation at refactor `redu12` (`59c35e60`) completed both
  observations with 248 PTC chunks, zero logged issues, and the expected 25
  stable products. Against same-config refactor `redu10` (`9ef7da8a`), all 24
  compared FITS/NetCDF products have zero changed or skipped records. Against
  deterministic OG science `redu15`, all nine FITS products remain within the
  current tolerance, while 30 RTC/PTC diagnostic records differ under the
  generic pointwise comparator. Scientific-owner review accepted those
  differences on 2026-07-11: all integer diagnostics are exact, map RMS drift
  is at most `2.31e-11`, PTC weight RMS drift is `2.14e-12`, and the largest
  near-zero detector-median difference is `2.85e-5` absolute and `2.42e-4`
  fractional. The versioned `science-scientific-equivalence-v1` gate enforces
  the accepted bounds and the validation ledger records the checkpoint.
- The intervening science `redu11` failed after observation 0 when observation
  1 metadata loading raised an unqualified NetCDF `No such file or directory`.
  Its merged config was identical to the successful runs. Metadata-load
  failures now report observation index, name, and telescope filepath; all 220
  local tests pass. The successful `redu12` shows this was not a persistent
  numerical or lifecycle failure.
- Beammap refactor `redu10` (`f278bd32`) and `redu11` (`9ef7da8a`) use identical
  merged configs and are numerically repeatable: all six large split FITS
  products, both APT tables, RTC/PTC diagnostics, and the complete detector-TOD
  `signal`/`flags` arrays have zero changed records. The matched OG Beammap pair
  is also deterministic. Scientific-owner review accepted the bounded OG to
  refactor differences on 2026-07-11: detector identities and flags are exact;
  the worst good-detector signal and weight RMS-relative differences are
  0.625% and 0.308%; sensitivity differs by at most 0.255%; and positional and
  FWHM differences are sub-microarcsecond. The versioned
  `beammap-scientific-equivalence-v1` gate now enforces these limits and the
  validation ledger records the accepted checkpoint. Any future threshold
  breach is numerical creep and requires investigation rather than automatic
  tolerance relaxation.

The first Beammap authority preparation checkpoint is complete locally without
changing production execution. Mechanical boundary checks expand 59 typed
reader roots and 59 serializer roots to exact 74/74 frozen-path coverage. A
pure, production-unwired `BeammapExecutionPlan` preserves requested values and
separately characterizes current phase correction, prior inheritance and
missing-path disablement, split-flag normalization, convergence availability,
and mapmaking-disabled iteration policy. Cold-boundary validation now rejects
non-finite Beammap vector and scalar values and enforces reader-established
vector cardinality. The existing typed request and one-way fitting adapter
remain production authority, and dedicated Beammap provenance remains missing;
this is preparation for a later bounded consumer cutover, not a completed
migration claim.
The local CLI and test targets build, all 363 CTests pass, and full config
preflight passes 74 boundary tests, all eight compatibility profiles, and the
complete authority audit suite. Because the plan and serializer are explicitly
unwired, this checkpoint does not require a Unity reduction.

## Beammap Effective-Plan Boundary Activated

The next bounded checkpoint constructs `BeammapExecutionPlan` in production
from one raw 74-leaf request plus explicit-key presence. Policy correction no
longer mutates values inside the family YAML readers. The immutable request is
preserved while a separate effective snapshot records phase correction, prior
inheritance and missing-path disablement, split-flag normalization, convergence
availability, and mapmaking-disabled iteration behavior.

Existing mature Beammap algorithms temporarily consume a one-way copy of the
effective snapshot through `ReductionConfig::beammap`, preserving their current
inputs without creating reverse synchronization. The existing map-fitter
radius adapter is the first bounded consumer to read effective plan policy
directly. The boundary audit enforces the ordered read/resolve/install/adapt
sequence and rejects reintroduction of the retired reader-side mutation
helpers. Dedicated Beammap realized lifecycle and provenance remain missing,
and the component serializer remains unpublished.

Local verification is clean: `citlali_cli` and `citlali_test` build, all 364
CTest cases pass, and full config preflight passes 74 tests, all eight compact
compatibility profiles, 100% compact-surface coverage, and every authority
audit. This changes production configuration construction, so the eventual
Beammap provenance checkpoint requires a Unity compile and matched Beammap
reduction before the domain can be accepted. The next local work is realized
iteration/output state and required atomic provenance; do not spend a Beammap
run on this intermediate commit alone.

## Beammap Realized Lifecycle And Provenance Prepared

The next local checkpoint adds an explicit Beammap observation and internal-
iteration lifecycle around the established execution without changing its
numerical control flow. Each enabled-mapmaking observation records identity,
detector/map/scan counts, contiguous iteration indices and phases, active map
counts, one or two completed mapmaking passes, the source-aware RTC decision,
fit completion, newly/total converged maps, and maximum-iteration or all-maps-
converged termination. Disabled mapmaking records a successful zero-product
execution instead of manufacturing observations or fit contexts.

Completion requires every internal stage and observation output to finish. It
then cross-checks Beammap observation identity/map counts against the completed
mapmaking plan and requires the post-processing Beammap fit-context count to
equal the exact number of completed internal iterations. Map write counts and
fit attempt/valid aggregates remain owned by their existing plans rather than
being copied into Beammap state.

Successful Beammap reductions now require atomically published
`beammap_provenance.yaml` with schema `citlali-beammap-provenance-v1`. The file
contains the complete requested and effective 74-leaf snapshots, effective-
resolution reasons, observation/iteration lifecycle, and terminal realized
state. Incomplete lifecycle and publication failures propagate to the CLI.
The strengthened boundary audit requires all lifecycle hooks, exact 74/74
reader and config-serializer coverage, and one ordered CLI completion/write
path.

Local verification is clean: both build targets pass, all 372 CTests pass,
and full preflight passes 75 Python tests, all eight compatibility profiles,
100% compact-surface coverage, and every authority audit. The authority
inventory deliberately remains `partial` until a matched Unity Beammap run
accepts this sidecar and scientific products. Observation-resolved prior and
reference decisions, adjacent atomic `beammap_source.fluxes` state, and any
additional Beammap-specific optional-product cardinalities required by the
design review remain bounded follow-up work; this checkpoint does not claim
the Beammap domain complete.

Enabled detector-specific Beammap PTC TOD is now an explicit required
observation product in the realized plan. The record is updated only after the
existing atomic NetCDF writer returns and captures the output iteration plus
detector, slot, and maximum-sample dimensions. Observation completion requires
exactly one such write when `beammap.detector_tod_output.enabled=true`, rejects
duplicates, and requires zero writes when disabled. This implements the
project-wide enabled-output decision without selecting new scan slots or
changing the detector-TOD numerical content.

The CLI and test targets build, all 373 CTests pass, and full preflight remains
clean with 75 Python tests and all eight compatibility profiles. This is part
of the pending Beammap Unity validation candidate, not a separately accepted
domain gate. Prior/reference and split-output fallback policies remain
unchanged and unresolved owner decisions are not inferred.

## Beammap Lifecycle Gate Accepted

Unity Beammap `redu03` was produced by `v4.0.0-3486-gb530e838` from the same
low-level configuration as accepted `redu02` (SHA-256
`aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d`).
The run completed 198 PTC chunks in 3,609.307 seconds with zero error-,
critical-, or fatal-level log records. The required
`citlali-beammap-provenance-v1` sidecar records one 5,234-detector/map
observation, three contiguous completed Beammap iterations, one mapmaking pass
per iteration, the expected source-aware RTC rerun on iteration one,
maximum-iteration termination, and exactly one required detector-TOD write at
iteration two with shape 5,234 detectors by 20 slots and 788 maximum samples.

Against `redu02`, the merged configuration is byte-identical. The accepted
Beammap profile reports exact detector identity, flags, APT quantities, and
all good/bad signal, weight, and kernel maps. The strict full-depth comparison
excludes only volatile `citlali_profile.ecsv` timing, reads all 12 scientific
products including detector TOD and six split FITS files, and finds no missing,
extra, skipped, or changed records.

The standard reduction audit now recognizes and can require Beammap provenance.
It validates observation/iteration lifecycle, terminal state, convergence
accounting, detector-TOD cardinality and shape, and cross-checks observation
identity/map count against mapmaking plus iteration count against
post-processing fit contexts. This closes the pending lifecycle/provenance
validation checkpoint, but the Beammap authority domain remains partial until
observation-resolved prior/reference state and adjacent atomic
`beammap_source.*` handling are completed. No unresolved fallback policy is
inferred by this gate.

## Atomic Beammap Photometry State Accepted

The adjacent photometry safety cut removes the concrete
cross-observation source-flux hazard without changing successful numerical
behavior. `beammap_source.*` is parsed into a temporary observation value and
all required runtime-array fluxes are validated before any Engine state is
mutated. Successful installation replaces typed photometry and the legacy
mJy/beam map and clears the derived MJy/sr map; it never merges with an
earlier observation. Missing or invalid required flux retains the established
fatal reduction outcome, but now throws a typed invalid-config error instead
of calling `exit()` inside `Engine::get_photometry_config`.

Project-owner clarification (2026-07-15): source identity belongs to telescope
data and TolProj owns calibrator selection and flux estimation. Citlali must
not mirror source name or coordinates into this config domain. Beammap
provenance therefore advances to `citlali-beammap-provenance-v2` with
`telescope_data` named as the source-identity authority and only the installed
per-array flux/uncertainty recorded as Citlali photometry input. The reduction
audit accepts historical v1 sidecars and requires this ownership record for
v2.

Project-owner decision (2026-07-15): every runtime array requires a positive,
finite calibrator flux; missing or invalid required flux fails the reduction.
No fallback is permitted.

Unity Beammap `redu04` was produced by `v4.0.0-3489-g7e577c81` from the same
byte-identical low-level config as accepted `redu03` (SHA-256
`aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d`).
The run completed all 198 PTC chunks with zero error-level messages. Its valid
`citlali-beammap-provenance-v2` sidecar names telescope data and TolProj as the
respective source-identity and calibrator-flux authorities and records the
three required installed array fluxes. The strict full-depth comparison reads
all 12 scientific products, including detector TOD and six split FITS files,
and finds no missing, extra, skipped, or changed records. The dedicated
Beammap profile also reports exact detector identity, flags, APT quantities,
and good/bad signal, weight, and kernel maps.

The total log interval is 3,661.793 seconds versus 3,609.307 seconds for
`redu03` (+1.45%). The dominant mapmaking interval is 0.53% faster; the
variation is concentrated in PTC chunk and diagnostics timing. This is within
the provisional 3-5% runtime budget and does not indicate a provenance-path
regression. Peak RSS remains unmeasured.

Both local targets build; all 24 focused Beammap/photometry tests, all 377
CTests, and all 49 reduction-audit tests pass. Full config preflight passes 75
tests, all eight compatibility profiles, 100% compact coverage, and every
authority audit.

## External KIDs And Config-Source Provenance Prepared

The bounded external KIDs checkpoint preserves Kidscpp as the numerical
execution authority while recording the exact bridge identity Citlali uses.
All four solved TOD representations (`xs`, `rs`, `is`, and `qs`) are supported.
The requested fitter/solver values, effective values, selected TOD type,
TolTEC data schema, and Kidscpp build version are separate fields in the
required atomic `citlali-kids-external-provenance-v1` sidecar. Historical
`solver.extra_output` behavior remains disabled and is now recorded explicitly
instead of being controlled by a header-level global.

The same successful CLI boundary now requires
`citlali-config-source-manifest-v1`. It records the ordered files actually
passed to Citlali, collision-safe copies, byte sizes, SHA-256 digests, and the
canonical merged YAML snapshot. TolTECA remains the owner of numbered
`NN*.yaml` discovery and upstream merge provenance; the record explicitly says
that TolTECA's complete ordered authoring-source list is not currently passed
to Citlali. Citlali does not guess or duplicate that merge.

Local CLI and test builds, all 382 CTests, 52 reduction-audit tests, and the
full 78-test config preflight pass. Unity point `redu59` identifies
`d016e1a64`, has zero serious log records, and passes semantic and digest
audits for both new records. Its low-level config is byte-identical to accepted
`redu58`; the strict full-depth comparison reads all 21 scientific products,
including complete RTC/PTC timestreams, with zero changed, skipped, missing, or
extra records. The external KIDs and Citlali CLI config-source checkpoint is
accepted. Complete upstream `NN*.yaml` provenance remains a future TolTECA
interface responsibility rather than a Citlali reconstruction task.

## Polarimetry Capability Disposition Accepted

The project owner intends Citlali to become the center of polarimetry
reductions, but not in the present refactor and not without an enabled
validation dataset. Phase 2 therefore preserves polarimetry as a planned
capability while mechanically rejecting `timestream.polarimetry.enabled: true`
before reduction execution. The exit condition is an approved polarimetry/HWPR
scientific contract plus an enabled end-to-end reference gate.

The frozen three-leaf request now has one direct typed reader, one immutable
request/effective capability plan, and one forward adapter into `RTCProc` and
`Calib`. The temporary legacy compatibility reader and reverse mirror are
removed. There is no separate `calibration.ignore_hwpr` YAML input; that name
was stale inventory text referring to the legacy adapter target. Disabled
reductions retain Stokes-I initialization and the established default values.

Successful reductions now require atomic
`citlali-polarimetry-provenance-v1`, recording the capability disposition,
requested/effective policy, accepted resolution, and realized non-execution.
The dedicated static audit freezes the boundary, while the reduction auditor
semantically rejects enabled or executed polarimetry in a successful run.
Local CLI and test builds, all 386 CTests, 54 reduction-audit tests, and the
full 82-test config preflight pass.

Unity point `redu60` identifies `db22bca1f`, completes all 12 PTC chunks in a
67.032-second total log interval, and has zero error-, critical-, or fatal-level
records. Its required v1 sidecar records the planned-unavailable capability,
an accepted disabled request, a disabled effective plan, completed reduction,
and no polarimetry or HWPR execution. The low-level input is byte-identical to
accepted `redu59`; the strict zero-tolerance comparison reads all 21 stable
scientific products, including complete RTC/PTC timestreams, with no changed,
skipped, missing, or extra records. The disabled capability boundary is
accepted. Enabled polarimetry remains planned but unavailable until its
scientific/HWPR contract and enabled reference gate are approved.

## Observation-Resolved Astrometry Candidate

The astrometry calibration-item loader now constructs the complete typed
pointing-offset request before touching observation runtime state. Structural
and finite-value validation runs on that temporary value, and a single forward
adapter then replaces both the typed request and the legacy Eigen vectors.
Invalid input throws the normal typed invalid-config error; the loader no
longer calls `exit()` or builds typed policy by mirroring partially mutated
runtime state. Legacy named axes, positional axes, one/two-value shapes, and
non-positive MJD sentinel normalization are preserved. The interpolation
kernel and its existing no-extrapolation behavior are unchanged. The remaining
interpolation failures now propagate as typed exceptions rather than terminating
the process from library code; successful numerical behavior is unchanged.

The project owner approved the legacy application contract. TolTECA selects
pointing support: two bracketing pointing observations produce interpolated
offsets, one pointing produces constant offsets, and no pointing observations
leave the explicitly configured offsets in force. Citlali applies the supplied
values. Positive MJD endpoints must remain strictly increasing, bracket the
whole observation, and are never extrapolated. Citlali does not receive the
upstream support-selection metadata, so it records that origin as unspecified
rather than inferring whether a constant came from one pointing or direct
configuration.

An observation-indexed execution plan now retains each immutable request,
effective application mode (`constant`, `observation-span-linear`, or
`explicit-mjd-linear`), observation number, installation/application counts,
and telescope sample count. Successful CLI completion requires atomic
`citlali-astrometry-provenance-v1`. Its authority record names TolTECA for
calibration selection and Citlali for application. A semantic reduction audit
and a static config-boundary audit reject incomplete lifecycle, malformed
offsets, inconsistent modes, authority drift, reverse mirrors, process exits,
or a missing required write.

The CLI and test targets build; all 398 CTests, 60 reduction-audit tests, and the
full 84-test config preflight pass. The combined astrometry/photometry domain is
still marked partial until Unity validates the new required sidecar and
scientific equivalence. The next gate should include a point reduction, then a
multi-observation OOF reduction because that fixture exercises observation
identity and stale-state isolation most directly. Beammap should follow before
the combined domain is marked complete.

## Astrometry Point Gate Accepted

Unity point `redu61` was produced by `v4.0.0-3496-g9ea6d7f0` from the same
byte-identical low-level and canonical merged configuration as accepted
`redu60`. It completed all 12 PTC chunks in a 63.741-second total log interval
with zero error-, critical-, or fatal-level records. Every applicable required
provenance record passes semantic audit.

The new `citlali-astrometry-provenance-v1` sidecar records TolTECA as calibration-
selection authority and Citlali as application authority without claiming
unavailable support-origin metadata. Observation 152389 has one requested and
effective zero-valued az/alt correction, constant application mode, one atomic
installation, one application, and 7,697 telescope samples. The reduction is
complete.

The strict zero-tolerance comparison against `redu60` reads all 21 scientific
products and 2,041 records, including complete RTC/PTC timestreams, with zero
changed, skipped, missing, or extra records. The point checkpoint is accepted.
The combined astrometry/photometry domain remains partial until a multi-
observation OOF run validates observation identity and stale-state isolation,
followed by a Beammap run validating the adjacent accepted photometry contract.

## Astrometry Multi-Observation OOF Gate Accepted

Unity OOF `redu02` was produced by `v4.0.0-3496-g9ea6d7f0` from the same byte-
identical low-level configuration as accepted refactor `redu01`. It completed
all 18 PTC chunks for observations 152385-152387 in a 40.667-second total log
interval with zero error-, critical-, or fatal-level records. All applicable
required provenance records pass semantic audit.

The astrometry sidecar contains three contiguous observation identities. Each
was installed and applied twice, once during initial geometry and once during
the reduction iteration, with stable per-observation telescope sample counts.
This closes the multi-observation replacement and stale-state-isolation gate.
TolTECA supplied a constant zero-offset request for each observation, so this
fixture does not provide an end-to-end positive-MJD interpolation test; that
limitation is retained explicitly rather than overstating the evidence.

The strict zero-tolerance comparison against accepted refactor `redu01` reads
all 30 configured products and 1,941 records with zero changed, skipped,
missing, or extra records. Direct comparison against OG `redu00` reproduces the
same nine previously accepted inactive RTC-despike metadata differences; all
scientific numeric differences remain within the standard OOF tolerance. The
OOF checkpoint is accepted. Beammap remains the combined astrometry/photometry
gate, and science remains required for the final Phase 2 snapshot matrix.

## Astrometry Science Interpolation Gate Accepted

Unity science `redu20` through `redu23` was produced by
`v4.0.0-3496-g9ea6d7f0` from the same byte-identical low-level configuration as
accepted `redu16` through `redu19`. Final `redu23` completed 248 PTC chunks in a
711.330-second total log interval with zero error-, critical-, or fatal-level
records. Every science-applicable required provenance record passes semantic
audit.

The astrometry sidecar records observations 152390 and 152392 with distinct,
strictly increasing positive-MJD support pairs and `explicit-mjd-linear`
effective mode. Each observation was installed and applied five times, once
during initial geometry and once in each of four fruit-loop iterations, with
stable telescope sample counts of 151,535 and 151,941. Successful completion
also proves that each support pair bracketed its complete telescope timestream;
the unchanged application kernel forbids extrapolation.

Every retained fruit-loop iteration passes the standard strict science gate:
`redu16`-`redu19` versus `redu20`-`redu23` each has 27 common products and 1,478
comparison records, with zero missing, extra, skipped, or out-of-tolerance
records at `2e-8 + 1e-10 * abs(reference)`. A zero-tolerance probe sees only the
expected tiny OMP run-to-run drift. The science and explicit-MJD interpolation
checkpoint is accepted. Beammap is the final mode gate for the combined
astrometry/photometry authority domain.

## Astrometry And Photometry Beammap Gate Accepted

Unity Beammap `redu05` was produced by `v4.0.0-3496-g9ea6d7f0` from the same
byte-identical low-level configuration as accepted `redu04`. It completed all
198 PTC chunks with zero error-, critical-, or fatal-level records. Every
Beammap-applicable required provenance record passes semantic audit.

The version-two Beammap provenance is identical to `redu04`: one 5,234-map
observation, three completed iterations, 15,407 valid detector fits, exact
telescope-data source identity and TolProj flux authority, three atomically
installed array fluxes, and one required 5,234-detector by 20-slot TOD write.
The added astrometry record captures one constant zero-offset application over
383,699 telescope samples without changing that accepted photometry contract.

The zero-tolerance full-depth comparison reads all 12 products and 16,453
records, including complete detector TOD and six split FITS cubes, with zero
changed, skipped, missing, or extra records. The dedicated Beammap scientific-
equivalence profile reports exact detector identities, flags, APT quantities,
and signal/weight/kernel maps for all 4,980 good and 254 bad detectors.

The 4,136.440-second total interval is 13.0% slower than `redu04`, but the
dominant mapmaking interval is 1.3% faster. The increase is concentrated in PTC
and diagnostics I/O before mapmaking, outside the astrometry change. Record the
variation without attributing it or treating one uncontrolled Unity comparison
as a performance conclusion; controlled performance/RSS certification remains
Phase 4 work.

The combined astrometry/photometry authority domain is complete. All 13 domains
in the original operational migration matrix have complete migration and
provenance disposition. The global F.1 leaf census and document/ledger
reconciliation remain before changing the active phase to Phase 3.

## F.1 Leaf Census Checkpoint

The owner approved the generated low-level Citlali YAML as Citlali's immutable
configuration/provenance boundary. TolTECA owns discovery, ordering, and merge
semantics for upstream `NN*.yaml` authoring files and must eventually record
that upstream provenance; Citlali records exact source bytes and ordered paths
from the generated low-level input onward. This is an explicit boundary
decision, not an inference that Citlali received unavailable source metadata.

The checked F.1 leaf contract resolves the union of `data/config.yaml` and the
four retained point, OOF, Beammap, and science low-level fixtures. It records
573 unique leaves, including 572 executable leaves and one explicitly ignored
deprecated leaf. Every record has a machine-readable authority, typed or
external owner, unit, allowed value-domain class, mode applicability, lifecycle
classification, resolution stage, and validation source. The preflight fails
on an uncovered leaf or drift from the resolved manifest.

This census exposed two real closeout omissions hidden by the earlier broad
subsystem grouping: 28 `timestream.learning` leaves executed from a legacy
options object populated in parallel with the typed request, and 14
`interface_sync_offset` leaves executed from an untyped mutable map with
permissive duplicate handling. They are now explicit `learning` and
`interface-sync` authority domains. Both are locally migrated through immutable
typed request, one-way adapter, validation, and versioned provenance. No
scientific algorithm or reduction behavior changed in either migration.

The learning omission is now locally migrated. All 28 leaves parse directly
into immutable `TimestreamLearningConfig`; one one-way adapter constructs the
unchanged `ReductionLearningState::Options` numerical input. The processed-
timestream requested/effective snapshots and versioned provenance now include
the complete learning policy. A frozen 28-path audit rejects reverse mirrors,
reader drift, incomplete adapter coverage, or missing serialization. Local CLI
and test builds plus focused reader/adapter tests pass. Because the standard
point fixture enables learning, an exact point Unity gate is the remaining
condition before marking this closeout domain complete.

The interface-sync omission is also locally migrated. All 14 TolTEC/HWPR
offsets parse atomically into immutable typed request state. Duplicate,
unknown, malformed, and non-finite entries are fatal; omitted interfaces retain
the established zero-second default with an explicit warning. One adapter
populates the unchanged alignment map. Raw-timestream provenance version 2
records requested and effective offsets with seconds as the explicit unit. A
frozen 14-path audit rejects reader, adapter, or provenance drift.

The F.1 startup gate is now operational rather than documentary. A generated
allowlist covers every normalized node in the checked 573-leaf contract and
the retained default configuration. Unknown nodes, including unknown empty
containers, enter fatal config diagnostics before execution. The `inputs`
subtree is deliberately excluded because its schema is owned by TolTECA; all
other low-level nodes are Citlali-owned. Typed validation errors now enter the
same fatal diagnostics instead of being logged as advisory mirror warnings.
The existing observation-scoped astrometry and photometry gates remain atomic.

The detailed [Phase 2 F.1 closeout](../handoff/PHASE2_F1_CLOSEOUT_2026-07-15.md)
maps every adopted checklist item to code, audit, and reduction evidence. Local
`citlali_cli` and test builds, all 410 CTests, all 96 config tests, eight compact
compatibility fixtures, 100% compact-surface coverage, and every boundary audit
pass. Unity point `redu62` closes the final gate as recorded below.

## Phase 2 Final Point Gate Accepted

Unity point `redu62` identifies `v4.0.0-3503-g9a3901e9` and the expected commit
`9a3901e91`. Its generated low-level input is byte-identical to accepted
`redu61`. It completed all 12 PTC chunks with zero error-, critical-, or
fatal-level records. Every required provenance sidecar passes semantic audit.

Processed-timestream provenance contains the complete 28-leaf requested and
effective learning policy exercised by the standard point fixture. Raw-
timestream provenance v2 contains all 13 TolTEC interface offsets plus HWPR in
requested and effective state, with unit seconds and exact equality. The
configuration-source manifest and canonical merged input are valid.

The strict zero-tolerance full-depth comparison reads all 21 stable products
and 2,041 records, including every RTC/PTC array. It reports zero changed,
skipped, missing, or extra records. The final F.1 gate is accepted; all 15
authority domains now have complete disposition and Phase 2 is complete.

The run took 176.435 seconds versus 63.671 seconds for `redu61`. The difference
is isolated to filesystem-facing stages: observation file setup increased from
1.758 to 28.723 seconds, raw/filtered output from 6.136 to 52.767 seconds, and
the 48 chunk-write calls averaged 4.172 rather than 2.482 seconds. Map
filtering, diagnostics, fitting, and other computational stages remained near
their prior timings. Treat this as an uncontrolled Unity/VAST I/O observation,
not a Phase 2 code-performance regression. A same-SHA rerun may characterize
the storage variance but is not required for scientific acceptance.

## Roadmap With Owner-Added Bridge Stages

### Phase 1 - Safety Stabilization

Repair output and run-success contracts, config parsing and finite-value
validation, output schema/cardinality checks, and ordered-writer cancellation.
Add injected failure and repeated-run tests without rewriting mature numerical
algorithms.

Exit gates:

- An injected required write failure returns a nonzero CLI status.
- Ordered output cannot deadlock after failure, and partial products have an
  explicit diagnosed disposition.
- A subsequent reduction in the same process starts with clean state.
- Invalid enums, NaN, and infinity fail with actionable config paths.
- The current point run has zero unexpected error-level messages and passes a
  strict complete-TOD and metadata comparison.

### Phase 2 - Config Authority And Provenance

Build the one-way flow from immutable requested config to effective execution
plan to realized observation metadata, with a temporary one-way legacy adapter.
Fix disabled-option provenance, atomic observation config, stale beammap flux
state, and typed/legacy parity checks. Validate real TolTECA overlay behavior
before compact config becomes operational.

Exit gates are the complete current-config definition of done in section F.1 of
the external review, including one authority per migrated field, no fallback to
raw YAML in migrated execution paths, correct provenance, and reviewed overlay
fixtures for each supported reduction mode.

### Phase 3 - Library, Session, And First Compiled Boundary

Introduce a minimal non-CLI reduction session/result boundary, remove reachable
library exits, give run/observation/scan state explicit owners, and freeze
`Engine` as a compatibility adapter. Add header-isolation and multi-translation-
unit checks, repair ODR hazards, and move one measured, coherent declaration and
validation tranche into `.cpp` files.

Exit gates:

- CLI policy is outside the library boundary.
- Sequential reductions in one process are clean and supported.
- Lifecycle state is reset by ownership rather than scattered cleanup.
- The first compiled boundary reduces dependency exposure without a material
  build or runtime regression.
- Further extraction has a named ownership or contract benefit; textual
  subdivision alone is not sufficient.

### Phase 4 - Validation, Performance, And Reproducible Build

Make strict comparison and active tests pinned CI gates. Add hermetic fixtures,
version/dependency provenance, current matched mode baselines, and controlled
performance diagnostics when triggered. Continue collecting timing and
peak-memory evidence during naturally required Beammap validation. Establish
polarimetry support or an explicit capability policy before release claims.

Exit gates are the broader structural definition of done in section F.2 of the
external review, with the project-owner performance proportionality exception:
strict scientific equivalence, zero unexpected errors, reproducible builds,
operational performance evidence with triggered controlled diagnostics, and
documented scientific conventions.

### Phase 4.1 - Four-Mode TolTECA Config Structure

Create a consistent numbered-YAML authoring kit for point, OOF, Beammap, and
science. Separate stable mode policy, site/runtime values,
observation/calibration selection, product choices, and user overrides. Keep
TolTECA as the merge owner and Citlali's generated low-level YAML as the
execution boundary.

Exit gates are defined in
[`PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md`](PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md):
all four kits exist, overlay semantics are hermetically tested, accepted
low-level equivalence is explicit, and one TolTECA smoke run per mode passes.

The Citlali-owned kit and validation tranche is complete as of 2026-07-16.
`config/tolteca/` contains four hash-pinned five-file kits derived from the
accepted point, OOF, Beammap, and science snapshots. The hermetic merge tool
implements TolTECA/Tollan list semantics, reports effective authority and
override provenance, and participates in the full config preflight. That
preflight passes 107 tests and all four accepted policy hashes. It also exposed
and closed two existing science cleaner-grouping gaps in the resolved leaf
contract, which now covers 576 leaves. TolPROJ commit `a33d26a` vendors this
exact kit behind an opt-in `--refactor` setup path for pointing, automatically
selected OOF/science, and Beammap project setup. Its default commands retain
the established `70_reduce.yaml`/`72_reduce.yaml` behavior. The refactor path
hash-verifies every vendored file, rejects mixed numbered-config families,
generates `72_observation.yaml`, preserves operator-owned runtime and expert
overrides on same-kit reruns, and rejects in-place mode or kit changes. All 96
TolPROJ tests, Ruff, byte-compilation, and tracked-file audits pass. Phase 4.1
does not proceed to smoke reductions yet. Project-owner review found that the
V1 files still expose the full machine policy under generic names and do not
materially separate routine, advanced, and expert authoring. V1 remains a
mechanically exact reference and the TolPROJ path remains opt-in, but it is not
the accepted operator interface.

The V2 authoring structure, first reviewed through science, is now generalized
under `config/tolteca/v2/` for point, OOF, Beammap, and science. Every mode uses
seven mode-named files: generated internal policy, site runtime,
TolPROJ-generated observation binding, routine analysis defaults, product
choices, advanced overrides, and expert overrides. The ordinary surfaces are
bounded to 4 runtime leaves, 27-44 analysis leaves, and 5-30 product leaves.
Mode-inapplicable controls are excluded, fruit-loop controls are consolidated,
and source finding is visible but explicitly experimental and disabled.

All four unchanged V2 kits merge exactly to their accepted V1 policy hashes.
The preflight enforces classification, file-size, ownership-disjointness,
mode-scope, data-binding, and byte-for-byte regeneration gates; it passes 116
focused tests and every config-authority audit. Citlali commit `6b6be9f57` is
the canonical source. TolPROJ commit `8490f09` vendors that snapshot
byte-for-byte for all four modes and selects it only under `--refactor`; every
non-refactor command retains the legacy path. Its manifest-driven installer
generates mode-named observation files, preserves all five operator files on a
same-kit rerun, rejects mixed or in-place kit changes, and passes all 100
TolPROJ tests. A fresh Unity smoke reduction for each mode now completes Phase
4.1; no Citlali compilation is required for this YAML-only integration.

Project layout review found that TolPROJ science and OOF reductions live under
`<root>/<user>/<source>` while shared data live under `<root>/data`. Data input
and KIDs fit-report paths therefore belong to the generated observation/data
binding, not the reducer-edited runtime file. The canonical V2 generator places
those paths in the mode-specific generated observation file; TolPROJ supplies
`../../data` for nested science and OOF projects.

### Phase 4.2 - Technique And Performance Review

Review every active subsystem for scientific/numerical appropriateness and
real-workload efficiency. Produce evidence-labeled dispositions and a finite
backlog before broad remediation. Intentional science changes use successor
validation evidence rather than being forced to match OG.

Exit gates are defined in
[`PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md`](PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md):
all active components are covered, no unowned P0/P1 finding remains, dominant
runtime/memory contributors have evidence-backed dispositions, and accepted
changes receive proportionate tests and mode validation.

The comprehensive component census is complete as of 2026-07-16. The
[`technique and performance evaluation`](PHASE4_2_TECHNIQUE_PERFORMANCE_EVALUATION_2026-07-16.md)
and machine-readable
[`component review`](../validation/phase4_2_component_review.json) assign every
active component to one of 13 review units, reconcile the earlier correctness
and performance audits with current code, and record evidence-labeled
dispositions. The production RTC, PTC, naive/JINC mapmaking, fruit-loop,
point/OOF, Beammap, coadd, and Wiener techniques are retained. No wholesale
numerical rewrite is justified by the evidence.

The census found one P0 capability defect: experimental
`maximum_likelihood` mapmaking remained selectable even though it was not a
validated global noise-aware mapmaker and Beammap did not populate that method.
Typed validation now rejects it for production while preserving the research
implementation for an explicit future decision. It also found one P1 output
contract defect: required pointing and Beammap FITS metadata could silently
fall back to zero or omission. Those catches are removed so required write
failures propagate. Subsequent accepted point and Beammap reductions exercised
the supported output paths without unexpected errors, closing those
behavior-preserving production guards.

Current profiles identify three measured Beammap costs: PTC cleaning consumed
1,565.923 seconds, map population 1,250.498 seconds, and the PTC diagnostic
sidecar 344.554 seconds in accepted `redu06`. Science `redu28` spent 2,799.461
seconds in the aggregate TOD pipeline. These measurements provide bounded
targets if a future performance trigger occurs; the accepted run history does
not establish a sustained regression that justifies speculative changes to the
mature numerical or output paths.

Phase 4.2 closed on 2026-07-17. The production P0/P1 findings are owned and
repaired, later accepted point and Beammap runs exercise the supported paths,
and every active component has an evidence-backed disposition. Candidate
Beammap NetCDF lock narrowing and finer science-stage attribution remain
responses to a measured slowdown rather than mandatory optimizations. Dedicated
Beammap/noise-heavy peak-RSS and profiler-overhead campaigning is
trigger-deferred under retained-debt item D13. Source finding is explicitly
experimental and disabled in the accepted operator kits; enabling it requires
a scientifically owned injection/recovery matrix. The
[`evaluation completion addendum`](PHASE4_2_TECHNIQUE_PERFORMANCE_EVALUATION_2026-07-16.md#completion-addendum---2026-07-17)
records the final disposition.

Compilation-boundary and build-system work remains deferred pending the
TolTECA developer's current build design. Header changes still demonstrate the
cost of that debt: rebuilding the CLI translation unit and link took 60.02
seconds locally during this review.

### Phase 5 - Integration And Closeout

Consolidate canonical architecture and scientific-convention documentation,
the validation ledger, and the intended-science-change manifest. Mark or remove
legacy/stub paths, tag the forensic refactor branch, and integrate the exact
validated tree. Add install/export support only if external library consumption
is an accepted project goal.

Phase 4.2 may recommend bounded RTC/PTC or other algorithm work, but review does
not authorize a wholesale rewrite. R execution remains a follow-up until its
measured-channel prerequisites are explicitly approved.

## Stop And Defer Rules

- Stop splitting files when a split has no clear owner, contract, test seam, or
  dependency benefit.
- Do not broadly rewrite RTC/PTC, JINC, or Wiener-filter numerical kernels in
  this refactor.
- Do not make compact config authoritative before TolTECA overlay acceptance.
- Do not implement R execution before a measured-channel data contract exists.
- Do not add concurrent reductions as a requirement unless the project owner
  explicitly needs them; sequential same-process reentrancy is required.
- After the refactor, replace the flat fruit-loop `reduNN` iteration sequence
  with one atomically claimed run directory containing explicit nested
  iteration identities, for example `redu01/iterations/iter00` through
  `iterNN`. Treat `redu01` as the stable identity of one user-invoked reduction,
  not as an iteration number. Add a run manifest that records a stable execution
  ID, each child iteration ID, the selected final iteration, Citlali version and
  git revision, and the effective-config digest. Preserve TolTECA-facing final-
  product compatibility during migration. This is the preferred long-term
  replacement for coarse output-root exclusion, but it is not part of the
  bounded Phase 3 repair.
- Do not squash or rewrite the only validated branch history.

## Decisions Requiring Scientific Ownership

Ask the project owner when implementation first depends on an answer. Do not
silently choose among these:

- Which output products are required versus optional in each reduction mode.
- How disabled filters and extinction states appear in requested, effective,
  and realized provenance.
- The future scientific meaning of hardware-polarization controls and the
  contract required to make enabled polarimetry a supported capability.
- Allowed calibration or analysis fallbacks and their required diagnostics.
- Canonical detector/network/array identities, coordinate frames, units,
  missing-value sentinels, and table schemas.
- OOF scientific intent and the acceptance tolerances for each mode.
- Whether any future caller needs concurrent reductions in one process.
- The measured-channel contract and missing-data policy for future R analysis.
- Whether Citlali must be installable and consumable as an external library.

## Durable Evidence

`validation/accepted_runs.json` is the machine-readable validation ledger. New
accepted checkpoints must record commit, binary version, mode, input/config
identity, comparator version, tolerances, error count, timing, available memory
evidence, and disposition. Run
`tools/baseline/validate_validation_ledger.py` after editing it.
`validation/intended_science_changes.json` is the separate source-to-evidence
ledger for intentional post-baseline scientific changes; validate commit
ancestry, patch identity, and evidence links with
`tools/baseline/validate_science_change_ledger.py`.
`doc/SCIENTIFIC_CONVENTIONS.md` is the canonical human reference for identity,
units, frames, indexing, validity, provenance states, and change-to-validation
routing. Product-specific executable requirements remain in
`validation/product_contracts.json`.
`doc/ARCHITECTURE.md` is the canonical human reference for the active software
entry, component and dependency direction, lifecycle ownership, compatibility
boundaries, failure flow, source classification, and extension routing.
`doc/PHASE4_CLOSEOUT_CENSUS_2026-07-16.md` maps every adopted F.2 completion
criterion to evidence, an approved exception, a deliberate deferral, or a
finite remaining action.
`doc/RETAINED_DEBT.md` is the canonical owner/trigger/exit register for
deliberately retained limitations, and `doc/adr/README.md` indexes durable
architecture decisions.
`validation/validation_profiles.json` identifies the active immutable
validation epoch and one profile per supported reduction family; validate it
with `tools/baseline/validation_profiles.py --list`. Continue to update this
document and the dated handoff note at phase gates and material validation
checkpoints.
