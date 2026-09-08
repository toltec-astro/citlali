# TIMESTREAM-SUCCESSOR-RTC-EVENT-BACKGROUND-001

Status: bounded implementation locally verified; source review,
representative Unity gate, acceptance and integration pending.

## Work order / preflight

- Owner: Grant Wilson. Latest instruction: "I like this. Let's give it a go."
  This accepts the proposed joint cubic with an optional additive post-event
  offset, following the event-extent decisions below. It authorizes this
  numerical Learn increment, not unstated significance or treatment policy.
- Purpose: the next bounded part of S2 in
  `doc/WP7_TIMESTREAM_SUCCESSOR_IMPLEMENTATION_BASELINE.md`, RTC event-background
  learning after the accepted original-pair spike-candidate learner.
- Risk tier 2; one scientific-module slot, one branch/worktree. Prior D2, VAL
  and RTC candidate work is accepted/closed; retained topic refs are evidence.
  NOI is paused; MAP/FRUIT contract work is outside this application increment.
- Effective governance read: `doc/governance/ENGINEERING_GOVERNANCE.md`,
  `TIMESTREAM_SUCCESSOR_GOVERNANCE.md`, `REVIEW_AND_CONFORMANCE.md`; respective
  SHA-256 values `70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76`,
  `29fae6f789bb6133c1f5bcdaf0f15437f2eb8c4110f338d3a8de9a4d98ba88dc`,
  `691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1`.
  Accepted `06a3ade51c1b3f38887295433d913811bf25cd14` and effectiveness
  `77507836325eff9f469062d5884481ea37599594` remain on canonical ancestry.
  AGENTS.md and toltec-context routing were followed. Status/ledger govern
  sequencing; ADRs 0017--0023 and the accepted native/VAL/RTC interfaces govern
  architecture. No frozen contract bytes change.
- Exact canonical base: `b675bb64a7054f7b24403c79898965e8765cfd02`, tree
  `b60bb3ab4287aa9f2f6565ee1a4dd0d70d657e27`, verified with explicit SSH
  GitHub `refs/heads/codex/refactor-mainline` on 2026-09-08. Its first parent
  is accepted RTC closure `2d41532ce3493ef329e35f438fe433a69c7c93a2`; its
  second parent is `b972d2850c2521662af59bba31d71a3ee525033c`. Moving-base
  reassessment: only documentation changed since that RTC closure; preserve
  the entire merge literally and continue this bounded RTC increment.
- Branch: `codex/timestream-successor-rtc-event-background-001`.
  Worktree: `/private/tmp/citlali-timestream-successor-rtc-event-background-001`.
  Initial staged/unstaged/untracked state empty. The unrelated dirty
  `/Users/gwilson/GitHub/citlali-refactor` checkout stays untouched.
- Scientific requirements: frozen SCI-RTC v0.1/r0.12 REQ-013/095 (original
  pair, spike exclusion before shift learning), 026/027 (numerical/context
  failures), 043--045/055 (honest support, selection and uncertainty),
  056/057 (no accepted plan from missing predicates), 094/096/098 (event,
  finite transition and plateau evidence), 118--125 (separate x/r evidence,
  protection, accepted typed classification and complete plan before Apply).
  OWNER-075 already selects additive baseline change and an unmodeled finite
  transition. This work binds that rule; it does not reopen it as gain change.
- Runtime Learn consumes immutable `RtcSpikeEvidence`, original x/r, its
  initial VAL snapshot and source-protection authority. It produces immutable
  `RtcEventBackgroundEvidence` for one named candidate and explicit trial
  fitting exclusion, with separate x/r fits and causes. Bounded Consider
  checks exact evidence/snapshot identity and reports numerical unavailability
  or the remaining policy/protection prerequisites. These constraints feed
  later complete RTC planning. No Apply treatment is delivered or implied.
  Consider also retains the accepted `RtcSpikeLearningDecision` for all
  preceding noise blocks; new fits cannot clear its paired exclusions.
- Engineering Learn/Consider/Apply remains recovery, implementation,
  conformance evidence and independent review followed by owner disposition;
  following that workflow does not satisfy the separate runtime boundaries.
- Scope: one RTC numerical interface, focused/header-isolation tests, CMake
  registration and necessary decision/status/ledger records. No Engine state,
  generic framework, AST/VAL mutation, route/config/default change or donor
  substitution. D2/VAL/native and accepted spike-learning source stays intact.
- Memory/performance: retain exact shared evidence; two small model summaries
  per coordinate; scratch limited to the two selected local flanks. No copied
  observation, noise/validity plane or per-cell provenance. Measure synthetic
  execution and logical scratch size; do not claim observation throughput.
- Gates: analytic/injected cubic/offset/spike cases, separate coordinates,
  guard sensitivity, context boundaries and real gaps, invalid input,
  insufficient support, zero scale, arithmetic/rank/convergence failures,
  source protection, immutable identity and partition invariance; header
  isolation, prior successor regressions, local CLI/safety/full CTest, config
  preflight and baseline tools. Local AppleClang/Homebrew/cached dependencies
  are supplemental. A new owner-run exact-source GCC13/Spack Unity gate and
  independent fresh-context exact-SHA three-axis review precede admission.
- 152390 remains a timestream fixture. Synthetic conformance here does not
  establish observational classification performance. No map comparison,
  separate validation program, CAL/PTC/filter/downsampling/production work.
- Stop/reassess for scientific ambiguity affecting the implemented numerical
  method, outside ownership, additional active module, changed canonical,
  unexpected numerical/resource behavior, new covariance or acceptance claims.
  Missing classification predicates block classification, not the explicitly
  selected descriptive fit calculation. Record and repair within this scope.
- Integration/push/activation/cleanup are not authorized. Owner performs all
  GitHub pushes. Reverify canonical before proposing admission, preserve the
  literal implementation, reconcile moving ancestry, review exact admission
  and closure SHAs independently. Earlier Unity gates do not cover this code.

## Recovered owner decisions and the selected increment

The discussion records were preserved under
`/private/tmp/citlali-rtc-event-assessment-owner-decisions-2026-09-08`.
This repository record makes the following selections durable:

1. Event extent requires demonstrated return to continuing local background;
   subthreshold parts belong to the excursion. Further jumps before recovery
   remain unresolved together. Missing recovery is not proof of a short spike.
2. Cubic background; at most two seconds of context on each side of the
   excluded event. Ten-second candidate-noise blocks are a different object.
3. Continuous observation context crosses internal processing scan/chunk
   boundaries. True observation ends can leave incomplete evidence. Actual
   acquisition gaps are not invented data or ordinary scan edges.
4. Recovery confirmation is 50 ms sustained agreement within +/-4 sigma.
   Confirmation samples are not automatically part of affected support.
   These recovery decisions are retained, not implemented by this fit alone.
5. Sigma is 1.4826 times the median absolute deviation from the median of
   pre-event cubic residuals; x/r separately. Freeze the pre-event estimate;
   post-event data cannot widen it. Huber IRLS tuning is 1.345. At least 64
   usable samples per flank per coordinate; no expanded-window or lower-order
   fallback. Numerical computability is not scientific model adequacy.
6. The proposal requiring two independent cubics to agree across a gap was
   challenged and withdrawn, never accepted. The derivative proposal was
   brainstorming: smooth drift can produce or cancel a fitted derivative.
7. Selected replacement: jointly fit M0 = p3(t) and M1 = p3(t) + Delta I_after,
   with one additive offset only on the observed post-exclusion flank. The
   central interval remains unmodeled. Retain a pre-only robust cubic to
   estimate the frozen scale; both joint fits use that same scale and support.
   Joint fitting supersedes separate post-cubic extrapolation for comparison.
8. The suggested +/-50 ms mask is an initial fitting trial, not established
   event extent, finite-transition containment, or the separate 50 ms recovery
   confirmation. This interface accepts an explicit native-row fitting
   exclusion containing both candidate endpoints, records its physical
   support, and never claims containment. Wider exclusions can be supplied
   as distinct evidence attempts; no automatic grouping/search policy is set.
9. Protected-region optical incompatibility remains required for spike
   treatment. The accepted readout assumption and beam/motion authorities
   remain intact. A fitted offset or small residual is not optical proof.

## Numerical binding and remaining decisions

The two flanks contain only original valid samples whose complete integration
support lies outside the trial exclusion and within its adjacent two-second
windows, in the candidate's same native acquisition run. Their midpoint
times define a centered/scaled polynomial basis; no resampling occurs.
Counts, excluded-invalid counts and actual first/last used rows/time coverage
are recorded. Observation/gap truncation is recorded separately from numerical
computability. No continuation across a physical gap is attempted. If another
known candidate touches either fit flank, that coordinate stays unavailable
for this isolated trial; callers must resolve/exclude that excursion explicitly.

Solver details are numerical controls, not scientific significance: scaled
time basis; column-pivoted QR with explicit floating-point rank tolerance;
ordinary least-squares initialization followed by Huber IRLS; finite iteration
limit and prediction/scale convergence criterion. Pre-only residual MAD is
updated during fitting; both joint models use its frozen final scale. An
unusable scale/rank/convergence/arithmetic result stays typed unavailable,
with NaN numerical fields. No covariance, standard error, p-value or effective
sample-size estimate is fabricated. Robust loss is descriptive and is not a
likelihood-ratio statistic or acceptance rule.

The pre-exclusion side defines the baseline reference for the signed post-side
offset. The fits are inert evidence: neither side is certified a stable
plateau, and no offset is applied. x/r fits use separate amplitudes and scales
but share the seed candidate and trial support; the selector's originating
coordinate is retained. No statistically independent estimates or genuinely
joint Exr statistic are claimed. Candidate selection, correlated atmosphere,
background-model inadequacy and trial-support uncertainty remain unavailable
uncertainty components. Numerical convergence does not establish their size.

Runtime cost includes native-run discovery and inspection of the existing
candidate catalogue for competing events, as well as the bounded local fits.
The synthetic timing witness is not a claim that batched observation-wide
event assessment has been optimized. No full-observation numerical planes
are copied; original state and deterministic selection reconstruct exact
coordinate-local fit populations without another per-cell mask plane.

Remaining OWNER-059--065 / REQ-043--045, 055/057/098 decision: how to account
for time-correlated residuals and data-selected windows when quantifying
offset uncertainty, and what evidence accepts a persistent shift versus
inadequate cubic continuation. This blocks hard shift classification and an
Apply correction, not the selected fit evidence. Adequate temporal coverage,
exclusion containment, event grouping/search extent, completed recovery and
protected-source optical acceptance likewise remain explicit downstream
requirements. No generic finite-fit or residual threshold silently closes them.

## Completion / conformance

- Disposition: implementation candidate, not accepted integration. One RTC
  header (independently compilable), one header-isolation translation unit,
  one focused test file with 16 tests, CMake registration and these necessary
  work-order/status/ledger records. No existing application implementation,
  frozen package or accepted D2/VAL/RTC source was edited.
- Local CLI/safety build and complete CTest pass: 933 registered, 932 runnable,
  only established `MapFitterLifecycle.ExactProductSequence` disabled. New
  RTC tests 16/16, focused successor regressions 83/83, full config preflight
  four modes, baseline tools 207/207, validation ledger 60 records and
  science-change ledger 3 changes / 5 integration records all pass.
- First broad attempt was not passing evidence: the safety executable had
  not yet been explicitly built, and the newly added 64-sample fixture
  contradicted its payload-finiteness flags. Built the required safety target
  and repaired fixture data; no scientific algorithm/policy change. Full
  rerun passed. Failed-stage logs are preserved rather than overwritten.
- Actual environment: AppleClang 21 / arm64, Release, Homebrew and existing
  disconnected dependency sources. tula `f30f81d97c44bd79618273bb842302ef839c6ab1`
  has six preexisting dirty headers; kidscpp
  `04088da182622c3e879f04314974a7c0d60ee2d6` has three. All nine paths/digests
  are inventoried; none edited. This is supplemental compilation/regression
  evidence. Representative Spack/Unity: not performed for this increment.
- Synthetic numerical cases demonstrate separate signed x/r offsets,
  an excluded transient on cubic drift, drift invariance, frozen pre-scale,
  minimum support, physical gaps versus processing partitions, invalid
  exclusion, competing candidates, zero/rank/nonfinite failures, exact
  parent/snapshot identity, source protection across the trial interval and
  retention of preceding screening exclusions. No false-positive probability,
  observational classification efficiency or calibrated uncertainty is claimed.
- Local synthetic witness: 16,384 input rows, 488 local scratch rows and a
  736-byte logical retained summary on this compiler (shared parent products
  excluded). Execution approximately 0.6 ms; precise run output is retained.
  This is not a batched observation benchmark. 152390 remains the timestream
  fixture; no real-data classification or map-product comparison was performed.
- Intentional scientific implementation: descriptive joint cubic and additive
  offset estimation with the owner-selected robust method. It changes no
  ordinary application mode or numerical output; no affected-mode reduction
  gate is claimed. No unexpected error-level output in final passing gates.
- Three-axis review disposition: writer supplies conformance evidence and
  ownership/ancestry inventories, not independent approval. Exact source SHA,
  tree and independent verdict must be bound by the external review receipt.
  The evidence directory is
  `/private/tmp/citlali-rtc-event-background-evidence-2026-09-08`.
- Next gates: independent exact-source review; owner-run new Unity gate;
  owner acceptance; reverified moving-base canonical-admission proposal and
  exact admission/closure reviews. Publication remains owner-controlled.
- Next scientific decision: the uncertainty/acceptance rule identified above
  before any hard shift classification. Final event extent/recovery and
  protected optical admission likewise remain unresolved; these are not
  encoded as hidden defaults.
