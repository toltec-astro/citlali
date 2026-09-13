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


## Owner-directed observational census continuation — 2026-09-09

The owner requests testing this implementation on as many timestreams as possible
for a census of the data. This authorizes a bounded continuation of this same
RTC work order: an inert real-input test driver and census evidence. It does
not accept the pending candidate or authorize integration, push or activation.
The original numerical headers, accepted D2/VAL/spike implementation, sealed
source/Unity evidence and receipt remain unchanged.

Literal continuation parent is receipt `b5a7411bd09a95849b36df52c037971cad47089f`
(tree `20f2ddaa1e2b4841467bd77c82bab533f5383bfd`). Live canonical was reverified
by explicit SSH GitHub query at `3efd079bb5c28f28f6f2d49296a29e966bde050b`.
Its intervening documentation is preserved; no admission operation occurs.
The existing branch/worktree remains the sole RTC work-order location and was
clean at census preflight. The unrelated SCI-ALIGN checkout is unchanged.
All three effective governance documents were reread; their identities and
subject authorities above continue to apply. This is risk tier 2 test/evidence
work, not a second scientific module or framework.

Expected changes are one excluded-from-default-build executable under
`tools/timestream_successor`, its CMake registration, a bounded local runner
and necessary status/evidence records. The driver calls the actual approved
RTC Learn and bounded Consider interfaces and existing raw/Tune/compact-v2
adapters. No numerical algorithm is reimplemented. It consumes verified
original I/Q, the raw-header-named Tune report, exact detector relations and
producer timing. It records separate original x/r candidate and fit facts,
all detector denominators, unavailable noise blocks, fitting failures and
unresolved Consider predicates. It never constructs or executes an Apply plan.
Engineering conformance remains distinct from these runtime responsibilities.

Input preflight found 27 observation headers / 148,494 channel timestreams,
298 unique filenames in 441 local copies; filename deduplication is discovery,
not hash identity. Thirteen observations / 71,734 channel timestreams have
local compact-v2 manifests for strict verification. Missing bindings remain
explicit input unavailability, not a newly missing scientific decision. No
legacy APT or already filtered/despiked/downsampled RTC product is substituted.
152390 remains included; maps and their comparisons remain excluded.

The explicit initial fitting trial uses +/-50 ms around each candidate edge,
clipped only at actual observation/acquisition-run boundaries. It includes both
candidate endpoints and labels its exact native rows. This reuses the owner's
initial-mask suggestion, not physical event extent or a recovery declaration.
The two-second flanks, cubic, frozen pre-scale, Huber tuning, minimum support
and candidate rejection rules remain those in the unchanged implementation.
No thresholds are tuned from census outcomes. The existing provisional uniform
readout averaging assumption is retained; midpoint assignment is explicitly a
within-network trial convention. Telescope epoch synchronization, cross-network
coincidence, source membership and optical incompatibility are not inferred.
Source protection is typed unavailable, so no protected-region treatment can
be authorized by these results.

Gates: verify raw/Tune/APT binding before invocation; exercise the driver on real
inputs; reconcile counts and typed failures; check required unresolved predicates;
run existing focused successor tests; preserve exact input/source/executable
hashes, actual local build identity and time/memory evidence; independently
review the exact census candidate. A new Unity build is not required for this
inert local test driver unless new evidence triggers that reassessment. Prior
Unity gates remain bound to their exact source. Unexpected input, numerical,
resource or performance behavior is recorded and reassessed; it is not silently
converted to a successful census. Missing detector bindings block those inputs;
uncertainty/classification/extent/source-policy decisions continue to block
classification and Apply, not descriptive fitting on admissible inputs.


Census implementation checks: local CMake target build, real raw/Tune/APT input
execution, all 83 focused successor regressions, wrong-Tune rejection, invalid
trial-width rejection, and existing-output preservation pass. The real driver
requires the Tune report's exact digest/size to match the compact-v2 KMP source,
in addition to its raw-header calibration relation. Original-input identities
are therefore checked before fitting. Six explicitly selected display examples
per network are hard bounded to at most 2,048 original rows each; selection is
for review only and does not change census populations. The all-detector and
candidate-summary records retain exact input/parent/binding identities and
unavailable causes without publishing a detector-by-sample mask history.
The external completion binds the final observational outcome and independent
exact-SHA verdict; no candidate is approved by the writer. No application mode
is changed, so ordinary-mode reduction and a repeated full application build
are not triggered by this test-driver addition. The earlier 932-test Unity
result remains evidence for the unchanged scientific implementation, not a
Spack test of this new local driver.


## Completed census evidence — 2026-09-09

The owner-directed continuation completed on exact source
`bff55895310c415a2924273514146cee3c9ad30c`, tree
`c5c2d27cd27d7e0d3ac27ffa1dcba04cb612598f`, parent reviewed Unity receipt
`b5a7411bd09a95849b36df52c037971cad47089f`. The compiled source-file digests
match this commit despite the explicitly disclosed precommit build Git label.
Executable SHA-256 is
`8d10a8c397893463c2666e81ce3b58dedad0a9504ffec4a6cba626c66dfb2af2`.
Independent exact-source review passes with recorded limitations and no findings
(report SHA-256 `a4ac8ed5d9bf41c52555c1b7fc3ee9c243e9e9f9ce30a7aa0f8ab5c890b5ae4d`).
Its fixed snapshot covered 123 completed networks; it does not itself approve
later aggregate/figures or this documentation receipt.

Final census: 143/143 network invocations pass, 71,734 channel-observation
instances (the same 5,518-channel inventory in 13 observations), 2,148,911,461
original x/r detector-sample pairs. All selected observations are from the
2026-02-19 night; no all-weather or distinct-device generalization is claimed.
There are 70,024 Tune-valid channel-observation instances. All input denominators
retain the invalid remainder and typed unavailable screening consequences.
The code finds 608,223 x candidate edges and 302,018 r candidate edges.
Each of the total 910,241 seeds receives one fitting attempt per coordinate.
These are seed/attempt counts, not unique physical events or affected-data
occupancy. No event grouping or physical extent is inferred.

| Coordinate outcome | x attempts | r attempts |
| --- | ---: | ---: |
| Numerically available | 130791 | 338458 |
| Other candidate in fitting flank | 775929 | 568299 |
| Insufficient fitting samples | 3416 | 3416 |
| Explicit iteration limit | 105 | 68 |

All records and per-network output checksums reconcile; the local runner exited
zero. Serial network wall time sums to 862.170958 seconds; peak per-process
resident memory is 5,328,666,624 bytes (macOS `wait4` resource accounting).
No unexpected producer error/critical-level message occurred. Existing legacy
`kindvar=0` raw-reader warnings remain in logs; they are not suppressed or
promoted into a different producer authority. The initial `/usr/bin/time` pilot
could not query sandboxed clock-rate metadata; final resource evidence uses
per-child kernel accounting instead. Expected negative-input tests and the
first harness compile failure are preserved separately, not called passes.

The test uses unchanged runtime Learn and bounded Consider. The principal
observational limitation is the existing isolated-candidate rule, which leaves
nearby-event cases unavailable. Numerical convergence or offset/pre-scale ratio
is not scientific acceptance. Correlated uncertainty, model adequacy, physical
extent/recovery and protected optical admission remain unresolved. No threshold
or policy is retuned from these data, and no Apply plan is constructed.

The discovery inventory also contains 155 network files / 76,760 channel streams
in 14 other observations lacking local canonical detector-binding manifests.
They are untested and are not counted as passed; 123424 also lacks a Tune report
in its discovered raw directory. Existing producer/binding preparation is the
needed prerequisite for expansion, not a new scientific classification decision.
No raw download, legacy APT substitution, MAP comparison or alternate framework
is added to address that gap.

Evidence root: `/private/tmp/citlali-rtc-event-census-2026-09-09`.
The final report includes all-observation totals, unavailable outcomes, offset
amplitude distributions and bounded actual-data displays. Saved examples are
illustrations from explicit selections, not a random population or optical
classification. The original 52-member source evidence and nine-member Unity
receipt seal both remain intact. This continuation does not integrate or accept
the still-pending RTC increment; the final documentation-only receipt and its
independent evidence review/seal bind the final disposition externally.

- `campaign-01/summary.json` SHA-256: `111b2f31b5349007fd6caefc02c4394afee013271da19d9660d27958852eadae`.

- `report/metrics.json` SHA-256: `43f8abf8d7309b82dd4a33ece7d9184e26f6233711cc2ce951f144beffcfcbba`.

- `report/README.md` SHA-256: `a5b66213dd72344bf91dab72ec6735a6d4617b18f7c62aa718187d4ce73d05d4`.

- `report/census-overview.png` SHA-256: `a198a7c6b1736a49eae0898d6799cebcdbe3b989d8afe8885f30a0bd2d2af1a8`.

- `report/census-examples.png` SHA-256: `efb56474611794c00843b483caa5831f95767eccddc40d908adc12925eed71b3`.


## Approved event-assessment continuation — 2026-09-09

Owner instruction: "I like these ideas. Let's do this and rerun the same test
suite." This accepts the six-case proposal preserved in this task: A isolated
recovery, B same-network context, C multiple fitting exclusions, D shift
assessment with neighboring spike excluded, E spectral context without automatic
periodicity rejection, F observation-time health review. This is one Tier-2 RTC
module continuation on the existing owned clean worktree/branch. Literal source
base is `be5636087af3c662655779eb99a032156799f04b`; earlier accepted D2/VAL/spike
and joint-fit implementations remain unchanged. No new integration branch.

Preflight: AGENTS, toltec-context, current status, architecture/scientific
contracts and all three effective governance documents read. Their digests and
accepted/effectiveness identities above remain exact. Live canonical verified
read-only on 2026-09-09 at `86c20b31f7300ba4063be044380b61cd0baf25eb`. Its common
ancestor with this module is `b675bb64a7054f7b24403c79898965e8765cfd02`; subsequent
canonical differences are documentation-only. Continue the approved module on
its literal source; preserve both histories and require independently reviewed
exact merge/closure identities before later admission. No merge or push here.

New owner selections: two-second bounded forward recovery search; 50 ms
continuous recovery within inclusive +/-4 frozen pre-scale sigma; preserve the
joint cubic background and two-second flanks. Starting +/-50 ms exclusions are
sample-support rounded and remain distinct from measured affected support.
Known candidates in either coordinate receive paired fitting exclusions;
remaining original valid samples define the fit. Confirmed recovery separates
events; further candidates before recovery stay compound/unresolved. Combining
fitting masks does not merge physical identities by itself. The pre-reference
polynomial part of the existing joint cubic-plus-offset fit defines continuing
background; its fitted offset is not subtracted when checking return. Lack of
recovery cannot establish a short event, gain change, or accepted level shift.

Trial health rule (review only): per coordinate, first-difference scale strictly
above 10 times the contemporaneous median of other explicitly eligible same-
network detectors AND candidate edges strictly above 1% of admitted differences,
in at least 80% of complete ten-second blocks, with at least six such blocks.
Insufficient/unknown comparison evidence remains unavailable; it is not healthy.
An either-coordinate finding raises a paired review concern, never a Tune rewrite
or automatic scientific exclusion. Peer eligibility is an explicit exact-parent
input, not an internal positional/flag inference. The census adapter binds it to
verified original Tune and existing clear APT quality fields; missing fields
exclude a detector from the comparison population, not from the census.

B compares original paired samples on exact within-network native rows. Level
correlation and first-difference correlation are distinct evidence; deterministic
strongest-peer summaries retain counts, support and timing. No atmospheric
subtraction, source diagnosis, cross-network association or hard coherence cut.
E's native spectral-use adoption gap was recovered from the prior VAL/D2 handoff;
a narrow owner decision is pending while independent implementation proceeds.
Neither legacy mask behavior nor diagnostic naming supplies that missing policy.

Runtime: RTC Learn consumes immutable original-pair spike evidence, the exact
initial VAL/protection handles and explicit peer eligibility; produces immutable
background, recovery, event membership and health/context facts. RTC Consider
consumes those facts and the same snapshot, applies the selected review policy,
retains prior noise-screening exclusions and reports unmet protection/adequacy/
shift-acceptance requirements. No Apply plan, replacement or offset correction.
Engineering recovery/testing/review is separate from these runtime boundaries.

Scope: new RTC event-assessment interface, tests/header isolation, inert census
adapter/runner and conformance records. Bounded original-sample scratch and
compact event/block summaries; no duplicate full observation or per-cell
provenance. Same exact 143 network inputs / 13 observations; A-F fixed review
cases and a spike injected on E's original oscillating background. Measure time,
memory, candidate preservation, unavailable dispositions and all caught health
channels. Preserve original census and all failed attempts. Tests: analytic and
injected recovery/extent/multiple events, shifts, health boundary comparisons,
peer isolation, gaps/endpoints/invalid support, partition determinism, original
immutability, source-protection honesty; prior focused tests, local full CTest,
config preflight and baseline tools. Independent fresh-context exact-SHA
three-axis review is required. Local AppleClang/Homebrew/cached dependencies are
supplemental; representative Spack/Unity admission remains separately gated.
No filtering, factor selection, downsampling, substantive PTC/AST/MAP/CAL,
activation, production, cleanup or user-controlled push is included.

Owner disposition of recovered E prerequisite: "Keep spectral context unavailable
for this rerun." No native PSD-use profile is adopted or inferred. Every event
retains explicit spectral-context unavailability. The E original-background
injection remains a candidate/recovery check, not periodicity classification.

Implementation binding: every original candidate receives a centered same-network
peer comparison; identical x/r edges reuse the same calculation. Event seed
context is separately referenced by the grouped record. Recovery-support
refinement expands fitting exclusions monotonically for at most eight numerical
attempts; exhaustion is explicitly unresolved and never accepted recovery.
A candidate that never leaves the fitted band receives `no_resolved_excursion`,
not `recovered_candidate`. Health assessment availability is distinct from a
negative review concern. Public evidence/decision construction is factory-only.

Local pre-candidate checks: 20 new focused tests, isolated public header, original
E waveform plus explicitly separate synthetic injection, four-mode config
preflight, 207 baseline tests / 137 subtests pass. Six real-input pilot networks
complete (A-F) with all candidate memberships accounted for. Pilot source bytes
and exact executable are recorded externally; later per-candidate-context and
disposition refinements require a complete new exact-source census. The E crop
has its own shorter noise population and does not claim bitwise reproduction of
the full-observation noise blocks. Original sample values remain exact.

First exact candidate `3b277caaa018f74624d979e5c3d5932aea9cdda6` passed
952 runnable local CTests (one established disabled test). Independent review
required repair: later grouped x/r candidates could inherit recovery preceding
their edge; required buffered output close failures were unchecked; candidate
peer records omitted available population/support counts. Preserve that review
and all test attempts in the external assessment evidence directory.

Repair retains the fixed original two-second search deadline, anchors onset to
the first member in each coordinate, and requires confirmation after its latest
member. New edges during the confirmation interval interrupt recovery; membership
is recomputed after each support refinement. A never-departing coordinate still
has no invented affected cells. Two staggered/alternating-coordinate regressions
join the existing suite (22 focused tests pass). Every required output and the
receipt now explicitly close and check errors; every candidate serializes peer
eligibility and shared-support counts. No scientific policy or scope expands.
A new exact-SHA independent review and complete census are still required.

Second exact review on `71cf11f43d28d14a7a709e1dc77930b6777759e5`
closed the three findings but found one boundary error: membership used the
pre-edge row, merging a new spike immediately after completed recovery. The
comparison now uses the post-edge sample against the half-open confirmation
end. A regression verifies two separate pulses with seven quiet native cells
between them (57.344 ms). All 23 focused tests pass. The intervening full-build
attempt is retained as interim evidence only because this final boundary repair
began before that build/test command ended; the final exact candidate must be
rebuilt and rechecked before the full census.


### 2026-09-09 A–F implementation and same-corpus completion

Disposition: **candidate**, ready for owner review; no accepted canonical
integration or production disposition. Literal base
`be5636087af3c662655779eb99a032156799f04b`; final implementation
`17c1bf0e6a33648d2ea0aede03fa111012ad4bed`, tree
`c7d12ea2a45c15e9fb6202818024575cea38a2a7`, immediate parent
`71cf11f43d28d14a7a709e1dc77930b6777759e5`. The earlier repair-required source
and reviews remain preserved. Independent review-03 binds the final full SHA:
**PASS with recorded limitations, no remaining findings**, all three axes.
Review SHA256 `7a1eccec8bc7a21ac6141c0cebc475c9bb9bee31ae37c00402b5c2dc03e1a7b6`.

External completion evidence:
`/private/tmp/citlali-rtc-event-assessment-2026-09-09`.
`EVIDENCE_SHA256SUMS` seals 1,282 files, SHA256
`a084be76119a32e35df89e8dbd997b53047d2542169102b98fdbb23906914af5`.
Changed-source digests, exact executable/CLI binding, complete source-state and
cached-dependency checks are in `source-binding.json` and
`source-state-after.json`. The census executable SHA256 is
`1eb2c2cbaa7d03daacbd96929f2979ea1f58122a2c82e3cf2bfacd57a6bba377`.
All accepted D2/VAL/spike and original joint-fit source remains byte unchanged.
Only this documentation receipt follows the tested implementation; its exact-SHA
review and final completion seal are added externally after commit.

Focused and broader local gates: final build/CLI match the implementation SHA;
955/955 runnable CTests pass, including 23/23 new focused tests and public-header
isolation. The established MapFitterLifecycle test is the sole disabled test.
Four config-preflight modes and 207 baseline-tool tests / 137 subtests passed;
subsequent repairs touch neither config nor baseline behavior. Both governance
ledgers validate unchanged (60 validation records; three scientific changes,
five integration commits). Wrong-Tune and existing-output negative checks pass.
All required data streams and receipt explicitly close/check failures. Expected
negative-test failures and intermediate build/review attempts are preserved;
all 143 successful census invocations require zero unexpected producer
error/critical messages.

The complete rerun uses the same 143 exact network inputs, 71,734 detector-
observation streams, 13 observations, and 2,148,911,461 original x/r sample pairs.
Every one of 910,241 original candidate detector/coordinate/row/score records,
all raw/Tune/manifest digests, original Tune validity and paired screening
exclusions match. Every candidate belongs to exactly one review assessment;
confirmed coordinate recovery never predates any member in that coordinate.
All A–F original displayed sample values compare exactly. Previous consumed
census/quality/case files are checked against their original sealed manifests.

Results: 242,014 grouped assessments, comprising 123,715 measured returns,
28,704 persistent/compound unresolved, 51,386 unavailable backgrounds, and
38,209 with no resolved excursion. Three explicitly retain a support-refinement
limit. These are review dispositions, not accepted physical event counts or
an affected-data percentage. Member-candidate-weighted background availability
is x 697,858 and r 698,761 of 910,241; earlier independent per-edge attempts had
x 130,791 and r 338,458 available. Grouped fits are shared by their members, so
this measures available fitting evidence, not classification accuracy.

A remains a one-native-cell paired excursion. B has strong same-network shared
variation: x channel 273 gives level correlation 0.994154 and difference
correlation 0.473555 over 488 original shared samples, with a 16.384 ms largest-
peer-edge delay; no causal atmospheric/electronic diagnosis follows. C separates
into three brief assessments with recovery between them and both backgrounds
available. D retains a large persistent shift assessment while excluding the
earlier small spike from its fit. E has no resolved departure from the return
band; spectral context remains explicitly owner-deferred. Its saved-background
injection test verifies candidate preservation, not periodicity classification
or equivalence to the full-observation noise population. F raises the health
review concern through r in all 124 complete ten-second blocks.

The health table retains all 17 caught detector-observation occurrences with
existing quality evidence: five clear, one already APT-flagged, eleven with
unavailable APT quality fields, and valid Tune flags in all seventeen. No new
scientific rejection or Tune/APT rewrite follows. Original samples remain
available for inspection. The full figures include fitting exclusions, measured
support, recovery confirmation, frozen pre-reference +/-4-sigma bands, and F's
block-by-block health comparison. See external `README.md`, `reconciliation.json`,
`cases.json`, `health-review.csv`, and full-identity `health-review.jsonl`.

Runtime architectural outcome: RTC Learn produces the immutable fit/recovery/
membership/context/health evidence from exact native parents; RTC Consider
applies the selected review policy while retaining original noise-screening
constraints. Apply produces no correction, replacement, offset subtraction or
new rejection plan in this increment. Engineering Learn/Consider/Apply was
separately followed through implementation, testing and independent review;
that workflow is not credited as the runtime architecture.

Resource result: 1,853.260 seconds summed serialized network process wall time
(30.89 minutes), peak child RSS 5,347,835,904 bytes (4.98 GiB). Local AppleClang 21,
arm64, Release C++20/Homebrew, with the same disclosed cached tula/kidscpp heads
and all nine pre-existing dirty source-file digests as the original census.
Representative Spack/Unity: **not performed for this candidate**; earlier job
64109974 applies only to its earlier exact source. No deployed affected-mode
route is exercised: this is the inert native test driver. Fourteen additional
observations (155 discovered network records) remain unavailable for this test
because required local detector-binding manifests are missing.

Retained limits: source protection is unavailable in this corpus; native spectra
are owner-deferred. Hard event acceptance, optical incompatibility, offset
uncertainty/significance and Apply remain separate contracted prerequisites.
No filtering, downsampling, factor selection, PTC/AST/common-grid/MAP/CAL,
production route, map comparison, canonical integration, push or cleanup.
Live canonical was reverified at `86c20b31f7300ba4063be044380b61cd0baf25eb`;
preserve moving-base ancestry and require independent exact-merge review for
later admission. Implementation worktree was clean throughout the final build
and census. The next owner decision is review of A–F and the health records;
this receipt does not silently approve further scientific policy.

## Owner decision: level-0 full-scan detector flagging — 2026-09-09

The owner now selects:

> The problem with shorter scans is that we're using a fixed number of
> eigenmodes in the PCA projection, so shorter scans would arguably be cleaned
> differently from longer scans. I think the level-0 approach should be to flag
> the detector with the jump for the full scan.

This supersedes the discussion's proposed level-0 retention of separately
processed pre/post pieces or preference for releveling when available. A confirmed
jump flags the affected detector for the full existing scan used by the
requested PCA processing. Other detectors retain that scan; this decision does
not split the PCA group into shorter intervals or change its configured rank.
Equal rank alone does not assert an identical learned projection after detector
exclusion; remaining-group support and rank feasibility still have to pass.

The event learner continues to use original native acquisition support, the
selected cubic and two-second flanks, paired fitting exclusions, and the
accepted recovery controls. Those fitting windows, ten-second noise blocks,
and detector-local stable segments do not define new telescope/PCA scans.
Confirmed event support and the full-scan treatment extent remain distinct.
The event-to-existing-scan binding, including a transition touching a scan
boundary, must be explicit before execution; this decision does not invent a
new scan definition or settle an ambiguous boundary-assignment rule.

The detector-scan flag is an intended exclusion from the selected PCA science
use, not merely an instruction to omit the detector from mode learning and
then retain its untreated stepped output. Implementation must express separate
learning, application and output-use decisions under their named owner. RTC
retains original event evidence, pair-coherent causes/support and the selected
treatment fact; PTC owns PCA-use admission and shared VAL evaluates that policy.
This does not rewrite Tune/APT producer flags, mark the detector permanently
bad, zero-fill excluded samples, or reject the rest of the detector group.
It selects no offset subtraction, gain correction, independent segment
centering, donor reconstruction or adaptive PCA rank. An unrealizable retained
group follows the existing PTC failure contract rather than a reduced rank.

Recovered authority: SCI-RTC REQ-094--099/102/106/114/118--125 separates event
evidence, transition support, treatment and consumer-owned admission;
SCI-PTC REQ-011--015/023/029/089/092--096 separates causes and named uses and
preserves configured rank, group support and fixed-plan application. These
contracts remain byte-unchanged. The selection closes the level-0 treatment
choice only; unselected recovery alternatives remain future proposals.

Runtime increment still required: Learn supplies accepted-method event and
quality/protection evidence; Consider must apply an explicitly selected jump
admission rule and bind an accepted event to the existing scan and detector;
Apply must execute the exact resolved detector-scan exclusion with realized
causes and support. No candidate, persistent/compound review disposition,
health concern, or unavailable source-protection state becomes an accepted
jump merely because this treatment is now selected. Adequate discrimination
from atmosphere/optical signal and background-model failure remains required.
The next scientific decision is that admission evidence and false-rejection
control. Precise additive-offset estimation is not a prerequisite for this
treatment; amplitude and its uncertainty may still inform event evidence.

Documentation-only receipt preflight: owner Grant Wilson; tier 2 because this
records scientific and cross-stage policy; local parent
`fefba770ce80f0073ce5987745373c74a00e66ff`, tree
`fef1be12b08ada7c010674f4071b74ea1e72d8cf`; the existing branch/worktree named
above was clean. AGENTS.md, toltec-context routing and all three governance
documents were read; their digests match the effective ledger bindings listed
in this work order. Only this work order and `doc/REFACTOR_STATUS.md` change.
Verification is documentation diff/whitespace, scope and immutable-source
preservation plus independent fresh-context exact-SHA three-axis review; no
compilation, census rerun or Unity gate is triggered by this receipt. Exact
review and completion evidence will be recorded outside the sealed census at
`/private/tmp/citlali-rtc-level0-policy-2026-09-09`.
The last canonical observation above remains historical, not a fresh live-ref
claim; this receipt performs no canonical admission or ancestry reconciliation.
No application source, frozen contract, route, integration, push, activation,
sealed evidence or cleanup changes are authorized or performed here.

## Owner decision: initial 5-sigma jump amplitude cut — 2026-09-10

Following the level-0 full-scan detector treatment, the owner suggested the same
threshold as candidate detection and then explicitly agreed to the proposed
definition: "I agree with this. What's next?"

Selected initial rule: `abs(A_jump) >= 5 * sigma_delta`. `A_jump` is the signed
additive offset from the existing cubic-plus-step fit in the candidate's
coordinate and original units. `sigma_delta` is the existing positive, finite,
available robust scale of adjacent-sample differences for that same detector
and coordinate in the candidate's exact ten-second noise block:
`1.4826 * MAD(difference)`. Preserve its original admitted population, native
run/block identity, exclusions and parent evidence. Equality passes. Use no
cross-coordinate scale substitution, sample-count reduction, square-root-of-two
conversion or newly estimated plateau-error denominator. The fitted background
and candidate/block parents must be explicitly compatible; grouping candidates
does not authorize silently choosing a more permissive block or combining
coordinate evidence under an unstated pair-admission rule.

This is an empirical noise-relative amplitude cut. It does not estimate the
standard error of the offset or assert a Gaussian false-detection probability.
The frozen pre-event residual scale used by the existing +/-4-sigma recovery
band is a different quantity and remains unchanged. A missing fit, invalid
scale or incompatible parent makes the amplitude criterion unavailable, not
passed. No unavailable uncertainty component becomes zero.

The cut supplies one RTC Consider predicate. It does not alone distinguish a
persistent additive jump from an isolated spike, atmospheric/optical variation,
compound event or inadequate cubic continuation. Existing confirmed-recovery
rules remain intact; failure to recover within the two-second search is not
hard shift acceptance. Required source protection and quality evidence remain
required. A candidate below this cut has not been proved harmless; its other
causes and selected screening rules remain visible. Level-0 full-scan flagging
requires a separately accepted jump and an exact existing-scan binding.

Next bounded runtime work is implementation of this amplitude predicate and
its exact evidence linkage, with threshold-boundary/unavailable/identity tests.
Hard jump admission additionally requires the still-open persistent-shift and
background-adequacy policy; it must not be inferred from successful fits or
good-looking census examples. Subsequent checks should challenge known shifts,
isolated spikes, atmospheric curvature and oscillations using the established
examples and controlled injections. No releveling uncertainty program or new
spectral-use policy is selected by this decision.

Documentation-only receipt: tier 2, same owned branch/worktree, initially clean;
parent `07cb9105c05ceab554098ccc0b990f5cc04c96b3`, tree
`21fb8a40e326014b1be467766c8ffa8e359ce427`. The previously read AGENTS,
toltec-context routing and three effective governance documents apply; their
digests were rechecked unchanged against the recorded accepted bindings.
SCI-RTC REQ-043--045/055--057/094--098/118--125 and the preceding treatment
decision retain their separate uncertainty, admission, ownership and lifecycle
requirements. Only this work order and `doc/REFACTOR_STATUS.md` change; all
application code, frozen contracts and sealed evidence remain unchanged.
Checks are diff/whitespace, scope and source preservation, plus independent
fresh-context exact-SHA three-axis review. Build, census and Unity gates are
not triggered by this documentation-only receipt. Its exact review/completion
record belongs at
`/private/tmp/citlali-rtc-jump-threshold-policy-2026-09-10`.
No fresh canonical-ref observation, admission, ancestry reconciliation, push,
route activation, application implementation or cleanup occurs in this receipt.

## Jump consistency and timing continuation — 2026-09-10

Owner Grant Wilson agreed to the fixed consistency check: "Well this makes
sense. We'll need to do some compute timing estimates to make sure this isn't
dominating the processing time." This accepts the proposed inner-one-second
check, same-sign and two five-sigma amplitude tests, and inclusive two-sigma
offset-agreement tolerance as an initial trial. It additionally requires
measured computation cost, rather than assuming that one extra fit is cheap.

Bounded work: `TIMESTREAM-SUCCESSOR-RTC-JUMP-CONSISTENCY-001`, within the existing
S2 RTC module/work order and its owned branch/worktree. Local literal parent
`ee30f449f8766104be23d3729a3ab9409fb5b720`, tree
`a5e53deb924b22d8d4162c53ee2a394651bf489c`, initially clean. Canonical was verified
read-only via explicit SSH GitHub URL on 2026-09-10 at unchanged
`86c20b31f7300ba4063be044380b61cd0baf25eb`. The previously recorded common
ancestor and documentation-only canonical delta still apply. Preserve the
literal module and canonical histories for later exact-admission review;
no merge, ref movement, push or canonical admission occurs here.

Preflight: tier 2; AGENTS, toltec-context, current status, application/program
baselines, architecture/conventions, implementation baseline S2, authority router,
SCI-RTC/SCI-PTC requirements and all three effective governance documents read.
The recorded governance digests and accepted/effectiveness commits remain exact.
This binds selected owner decisions within the frozen scientific contracts;
it does not alter those contracts or confer observational classification or
production authority. Prior accepted implementation/evidence stays preserved.

Numerical policy: keep the primary two-second flanks, original paired exclusions,
native run/timing, cubic degree and robust-fit method. For each seeded coordinate
of the existing grouped assessment, bind its first original candidate in the
existing ordered membership (the coordinate's existing onset seed), with that
candidate's exact same-coordinate ten-second noise block. No choice among noise
blocks by score, no pooling, and no unseeded-coordinate surrogate is allowed.
The primary offset must pass `abs(A2) >= 5*sigma_delta` before the shorter fit
is requested. Then fit the inner one-second pre/post flanks immediately outside
the unchanged trial exclusion, using a subset of the original valid primary-fit
support and the same neighbor exclusions. Preserve the primary coordinate/time
basis and reference side. Re-estimate the shorter pre-flank residual scale with
the existing pre-cubic procedure and freeze it for its cubic-plus-offset fit;
this is distinct from the unchanged `sigma_delta` in all admission comparisons.
There is no need to compute an unused shorter plain joint-cubic comparator.
The method therefore invokes at most one shorter pre-scale fit and one shorter
joint offset fit per qualifying coordinate, each retaining the existing finite
numerical iteration bound and 64-valid-samples-per-flank minimum. Count actual
fit calls and exposed iteration counts, distinguishing unavailable results.
Counts include IRLS loop entries on successful and failed fits; zero denotes
failure before entering the loop.

The shorter offset A1 must satisfy `abs(A1) >= 5*sigma_delta`, have the same
nonzero sign as A2, and satisfy `abs(A1-A2) <= 2*sigma_delta`. Equality passes.
All comparisons use original coordinate units and the one exact noise scale.
These overlapping fits are dependent empirical checks, not independent
uncertainty estimates. Missing support, numerical failure or nonfinite comparison
remains unavailable. Existing confirmed recovery excludes persistent-shift
treatment; no recovery alone still does not establish it. Preserve source
protection, model-adequacy, compound-event and scan-binding constraints.
Consistency success never alone authorizes the level-0 detector-scan flag.

Runtime architecture: prior Learn/Consider assessment -> immutable RTC amplitude
Consider decision -> bounded short-fit Learn evidence for the requested subset
-> immutable RTC consistency Consider result. The gate is explicit policy in
Consider, not a hidden selection inside a producer. These concrete products
retain exact earlier evidence handles, coordinate origins and original VAL
snapshot. They feed later complete planning; no Apply correction, replacement,
full-scan rejection, PCA change, route or other stage is implemented here.
This fixed diagnostic sequence does not add a repeat-until-satisfied loop.

Changed areas: one coherent RTC header and isolated/focused tests; test CMake;
the existing inert census driver/runner for extra evidence and stage timing;
necessary work-order/status updates. No Engine state, generic framework,
source-sample mutation, per-cell provenance plane or scientific optimization.
Scientific change is the explicit amplitude/consistency predicates; primary
measurements, candidate identities, screening and earlier review output must
remain exact. Frozen D2/VAL/spike/background/assessment source stays unchanged.

Gates: threshold equality and signed/invalid arithmetic; exact parent/VAL and
noise-block binding; deterministic ordering; short-fit original-sample subset,
neighbor exclusions, gap/edge/insufficient support; injected positive/negative
steps, nearby spikes, curved backgrounds and unstable offsets; recovery and
protection preservation; header isolation, prior focused and full local CTest,
config preflight and baseline-tool regressions. Rerun the same exact 143-file
corpus and A-F windows, with source/input/output identities and candidate/
screening/earlier-assessment reconciliation. Include shorter-window residuals
in the report as model diagnostics, not an unapproved extra numerical gate.
Independent fresh-context exact-SHA three-axis review is required before close.

Timing: separate input/producer, spike learning, original assessment, original
Consider, new amplitude gate, short-fit learning, final Consider and output
cost in the inert driver. Record candidate/coordinate gate counts, pre/joint
fit calls and iterations, cumulative wall time, maximum per-network cost and
memory; show incremental cost relative to the original stages in the same run.
Use representative quiet and event-rich inputs, then the complete same corpus.
No arbitrary performance budget or hardware-independent throughput is assumed.
Stage/corpus timings are local AppleClang/Homebrew supplemental measurements;
they are not full production RTC/PTC runtime or a Unity result. No direct Unity
access is authorized. Preserve the earlier sealed evidence and all failed
attempts; new artifacts belong under
`/private/tmp/citlali-rtc-jump-consistency-2026-09-10`.

Stop/reassess for a scientific-policy ambiguity, changed authority or source,
new ownership, nonfinite/incorrect behavior, incompatible parents, unexpected
resource growth or an apparent need to change the primary estimator. Timings
may motivate a separately recorded optimization; they cannot silently change
the accepted fit or threshold. Missing source protection in the current corpus
remains explicit; no MAP comparison, native PSD use, filtering, downsampling,
PTC/CAL/AST work, production, integration, push or cleanup is included.

Pre-candidate verification: 16 focused tests and public-header compilation pass,
including signed/equality thresholds, short-support insufficiency, real packet
gaps, noise-block boundary ownership and nonlinear-background offset instability.
The second six-network A-F pilot completes. The first pilot stopped on a runner
assertion that incorrectly required both endpoints inside a noise block; the
original producer correctly owns a crossing edge by its later endpoint. The
runner and a focused boundary test now preserve that rule. Build setup/type
errors and the initial gap-fixture correction are retained in external logs.
No primary scientific source changed. Full exact-source gates/timing and
independent review remain pending at this implementation commit.

Independent source review of `5f0492722d3b05b944c222604ea0a3bcf999ba1a`
found one minor evidence-label error, RTC-JC-R01: the preserved solver does
retain iteration counts after failure. This continuation corrects the labels
and adds assertions for failure before/inside the loop, without changing the
solver, accumulated counts or scientific behavior. All 971 runnable local
tests, configuration preflight and 207 baseline-tool tests passed on that
candidate. Final repaired-source build, corpus timing and review follow.

## Jump consistency and timing completion — 2026-09-10

`TIMESTREAM-SUCCESSOR-RTC-JUMP-CONSISTENCY-001` completes its bounded local
implementation and measurement gates, ready for owner acceptance. Tested
implementation `0d04f442f70fb221694355cba20c301bccb1008c`, tree
`a75ba2e8a955a0f7500ebdc0bc00fa4b1bad4648`, has parent
`5f0492722d3b05b944c222604ea0a3bcf999ba1a` and literal increment base
`ee30f449f8766104be23d3729a3ab9409fb5b720`. The earlier implementation and
review-label repair remain separate commits; no scientific solver or prior
accepted implementation was rewritten. Live canonical was rechecked read-only
during the final corpus observation at unchanged
`86c20b31f7300ba4063be044380b61cd0baf25eb`.
Preserve the recorded moving-base ancestry for later admission. No merge,
canonical ref movement or push occurs in this module closure.

Runtime products now implement amplitude Consider -> requested shorter-fit
Learn -> consistency Consider, each retaining the exact prior evidence and
original VAL snapshot. Both offsets use the same original candidate-coordinate
noise block for the selected comparisons. The shorter fit uses original inner
support and preserved exclusions, with at most one pre-scale and one joint
offset fit, keeping the 256-entry numerical bound. Recovery and protection
constraints remain visible. These products are not an Apply plan and do not
execute the chosen full-existing-scan detector flag. Development workflow
conformance is recorded separately from these runtime responsibilities.

Exact-source local gates: 971/971 runnable CTests pass, including 16 new focused
tests and isolated-header compilation; one established MapFitterLifecycle
test remains disabled. Four config modes and 207 baseline-tool tests passed
on the implementation candidate before the comment/evidence-label repair;
their source was untouched by that repair. Repaired exact-source full CTest
and the corpus then pass. AppleClang 21 arm64 Release C++20, CLI version
`g0d04f442f`, and both executable digests are bound. All cached dependency
revisions and nine pre-existing dirty dependency file digests are unchanged.
No GCC13/Spack or Unity result is claimed.

The exact prior 143/143 inputs complete: 13 observations, 71,734 channel
timestreams, 2,148,911,461 original x/r sample pairs. All raw/Tune/APT digests
match. All 715 original candidate/event/health/detector/example JSONL outputs
match byte for byte, preserving 910,241 original candidate edges and 242,014
grouped assessments. The runner and independent reconciliation check exact
noise-block links, comparisons, fit/iteration counts and per-network hashes.
Source, binaries and dependency state are unchanged after the run. Fourteen
other discovered observations, 155 files, still lack detector-binding manifests
and remain untested. The tested corpus covers one observing night.

Of 484,028 assessment-coordinate records, 51,280 pass the primary amplitude
gate; 40,246 shorter pre-scale and 40,220 shorter joint fits are attempted,
with 40,125 available results. The remaining 11,155 requested results are
unavailable: 195 truncated contexts, 10,839 insufficient supports, 26 pre-fit
iteration limits and 95 joint-fit iteration limits. The final consistency
counts are 6,558 shorter offsets below threshold, 2,630 sign disagreements,
16,144 offset disagreements and 14,793 passes. Of those passes, 12,815 lack
confirmed recovery; that is not an accepted physical-jump or detector-scan-loss
count. Missing source protection and other admission predicates remain missing.

A/C retain the selected pulse recovery evidence; their evaluated persistent
offsets fail the amplitude cut. B's x offsets are 26.450 and 50.895 sigma_delta,
disagreeing by 24.445 sigma_delta. D's x/r agreement differences are
0.0146/0.1816 sigma_delta,
both passing. E's selected x offset stays below the primary cut; spectra remain
unavailable, without a blanket oscillation-rejection claim. F retains its
separate detector-health concern. Original sample windows are unchanged.
The shorter residual plots use recorded row support; primary residual display
bounds are explicitly reconstructed from sample timing. Residuals are model
diagnostics, not an extra admission threshold or fitted-offset significance.

Measured on Apple M4 Pro with 48 GiB RAM, serial networks, no competing agent
build/test/profiler workload: new amplitude/short-fit/final-Consider stages
total 6.533370 seconds. Existing event assessment takes 1,469.529250 seconds:
the increment is 0.444589%. Relative to prior spike Learn + event assessment
and Consider it is 0.422710%, and relative to all original driver stages,
including input/output, 0.369606%. New output costs another 0.865588 seconds;
all additions are 0.418574% of original stages. Stage total 1,775.057590 seconds
excludes final receipt/hash and shutdown; summed child wall time is
1,847.726008 seconds (30.80 minutes), excluding Python orchestration between
children. These are same-run stage comparisons, not a cross-run speedup or
production fraction. The normal desktop was not exclusively reserved.

Amortized numerical cost is 0.1274 ms per requested coordinate, not individual
fit latency. Pre/joint IRLS loop entries are 653,664/1,084,518, including failed
fits. Maximum added numerical time is 1.251178 seconds at 152392/network11;
maximum relative increment is 1.627980% at 152390/network11. Whole-child peak
RSS is 5,343,887,360 bytes (4.977 GiB), not isolated incremental memory.
Maximum shorter support is 244 logical sample rows, not a byte allocation.
These local results do not trigger a solver optimization or policy change.

Independent exact-source and full evidence three-axis reviews pass with
recorded limitations and no remaining findings. Source finding RTC-JC-R01 is
closed; failed/preliminary build, test, pilot and source-binding attempts are
preserved. The optional hardware-read sandbox denial was resolved by a scoped
read-only metadata query and did not affect the source or census. All prior
sealed evidence remains untouched.

External evidence root: `/private/tmp/citlali-rtc-jump-consistency-2026-09-10`.
Source review SHA-256: `51f8ce89fb3744c98fbd57862709af5dabbe21f7554c621f527b77639df07c94`.
Full evidence review SHA-256: `2c5f52bc386b3617e47e8e2cf08269d7fecd4cce61741782603c9b3b23177726`.
`EVIDENCE_SHA256SUMS`: `382f906777ab25d2c114ec2bde591210a4021b5a9eb7450d2b355fecaa8dbaef`.
The documentation-only closure must have the tested implementation as its
direct parent;
its exact commit/tree, independent exact-SHA review and final complete manifest
are recorded receipt-last in external `completion.json` and `SHA256SUMS`.
This avoids a self-referential commit identity inside the closure itself.

No Apply correction/replacement/rejection, native PSD binding, filtering,
downsampling, PTC/AST/CAL/MAP implementation, production route, canonical
integration, Unity operation, push or cleanup is included. The next executable
treatment must consume complete contracted event/protection evidence and exact
scan binding; this consistency result alone cannot authorize detector loss.

## Owner decision: every scan intersected by an accepted transition bound — 2026-09-10

Owner Grant Wilson reports, "Pushed. And I agree with this starting
recommendation", accepting exclusion of the affected detector from every
existing scan intersected by an accepted conservative physical transition
bound. This completes the scan-selection choice left open by the level-0
treatment discussion: a transition that spans two existing scans selects both,
rather than assigning the event only to a candidate-center scan. SCI-RTC
notation and SCI-ALIGN REQ-031/053 already select half-open intervals; preserve
that representation and its ordinary endpoint intersection semantics. The
new instruction selects no numerical transition-bounding method.
The existing scan authority must supply its exact support and timing identity;
no event window, noise block or detector-local segment becomes a new PCA scan.

The full-existing-scan treatment, shared scan definitions, configured PCA rank,
remaining-group feasibility, and separate learning/application/output-use
dispositions remain as selected above. RTC retains pair-coherent physical
event and treatment facts; PTC owns PCA-use admission and VAL evaluates the
named consumer policy. A full-scan treatment extent is not the physical
transition extent or an operator guard. No offset subtraction, gain correction,
source substitution, scan splitting, Tune/APT rewriting or permanent detector
flag is selected.

Recovered settled authority: SCI-RTC DEF-036 and REQ-094/096 require a finite
physical transition interval or conservative bound, with affected pair cells
derived from the native timing vector and timing uncertainty retained.
REQ-118/119/126 already preserve distinct coordinate evidence and form the
pair-coherent union after class-specific hard-event admission. This decision
does not reopen that union or require both coordinates to independently pass.
REQ-121 continues to require causal/source-protection predicates; unavailable
required authority does not become a hard event. REQ-124/125 require the
complete immutable plan before Apply, with no new detection during Apply.

Narrow unresolved prerequisite: OWNER-060/061 leave the physical transition
localization or bounding estimator, stable-side support/quality criterion, and
the binding of physical support and timing uncertainty into native pair cells
and existing scan supports to explicit selection. The general half-open
interval convention is already resolved; these open methodological details do
not reopen it or select a new rounding or uncertainty-expansion rule.
The current +/-50 ms trial exclusion was selected to protect background fitting;
it is not measured or accepted physical transition support. Likewise, the
existing +/-4-sigma/50 ms return-to-pre-event-baseline test answers recovery,
not establishment of the post-jump plateau. Reusing those numerical settings
for transition localization would require an explicit new use binding.
Existing two-second primary and one-second consistency fits, same-coordinate
five-sigma amplitude cut and two-sigma agreement remain accepted evidence;
they are not silently promoted into a complete physical-jump classifier.
Required background/causal/protection and compound-event dispositions remain
prerequisites for hard admission; no missing predicate is replaced by this
scan-selection rule.

Next executable responsibility remains inside implementation-baseline S2 RTC:
runtime Learn must produce the selected original-sample transition bound and
its evidence/availability; runtime Consider must consume it with the existing
amplitude, consistency, recovery and protection evidence to resolve an event.
The accepted event's exact detector, pair support and existing scan identities
then support the selected scan-use rule under RTC/PTC/VAL ownership. Apply
executes only the completed resolved plan and records realized treatment.
Following the engineering Learn/Consider/Apply workflow does not implement
these runtime products. This receipt introduces no runtime framework or code.

Publication receipt: read-only `ls-remote` using the explicit Mac SSH GitHub URL
on 2026-09-10 verifies
`refs/heads/codex/timestream-successor-rtc-event-background-001` at
`d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`, tree
`b5fa1e1242b1f7b508082ac4330a72cbf355ecf4`, whose direct parent is tested
implementation `0d04f442f70fb221694355cba20c301bccb1008c`. Canonical
`refs/heads/codex/refactor-mainline` remains
`86c20b31f7300ba4063be044380b61cd0baf25eb`. Common ancestor is
`b675bb64a7054f7b24403c79898965e8765cfd02`; preserve both literal histories.
This is verified feature publication, not canonical admission. Any eventual
admission must reverify the then-live canonical base, preserve the accepted
implementation, and receive independent review of the exact admission SHA;
neither the old source review nor this receipt's review transfers to a merge.

Documentation-only preflight: owner Grant Wilson; tier 2 scientific/cross-stage
policy receipt; same owned branch/worktree, initially clean at the pushed
closure above. AGENTS, toltec-context routing, current status, existing RTC
work order and all three effective governance documents were read. Governance
digests match the accepted integration-ledger bindings; governance acceptance
`06a3ade51c1b3f38887295433d913811bf25cd14` and effectiveness record
`77507836325eff9f469062d5884481ea37599594` remain on canonical ancestry.
Only this work order and `doc/REFACTOR_STATUS.md` change. Gates are exact
identity/ancestry, diff/whitespace, unchanged application/frozen-contract bytes
and independent fresh-context exact-SHA three-axis review. No build, config,
census, timing or Unity rerun is triggered by documentation alone. Evidence,
review, final identities and conformance belong in the new external receipt at
`/private/tmp/citlali-rtc-scan-intersection-policy-2026-09-10`; the earlier
sealed roots remain unchanged. Stop before selecting any unresolved scientific
method or new ownership. No integration, push by the agent, route activation,
automatic flagging, PTC implementation, native PSD use, filtering, downsampling,
AST/CAL/MAP work, production or cleanup occurs here. Observation 152390 remains
a timestream fixture.

## Jump transition Learn continuation — 2026-09-10

Owner Grant Wilson says "Let's try this", selecting the proposed original-sample
transition bracket: compare against the primary fitted cubic before the jump,
and that same cubic plus its signed fitted offset afterward. Require inclusive
+/-4-sigma agreement sustained for at least 50 ms, with the unchanged primary
pre-event residual scale, separately in each originating coordinate. Search
within two seconds either side of the existing event seed in its native
physical run. Use the last confirmed pre-jump support and first confirmed
post-jump support to bracket the transition. Ambiguous or missing support
remains unresolved. This explicitly binds a new use of the accepted settings;
it neither changes the recovery test nor promotes the +/-50 ms fitting mask.

`TIMESTREAM-SUCCESSOR-RTC-JUMP-TRANSITION-001` is a bounded continuation of S2,
in the same owned RTC module branch/worktree. Literal parent
`8755461edb7e4060ee09c9e1262e8b00b8266026`, tree
`3f2ec2e0afef46a8ed0a7669d597ab7ef7eb15da`, was clean. Live canonical was
reverified read-only at `86c20b31f7300ba4063be044380b61cd0baf25eb`; published
module closure remains `d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`.
The recorded common ancestor and moving-base review obligations remain intact.
No merge, history rewrite, new branch or canonical admission occurs here.

Tier-2 preflight: AGENTS, toltec-context routing, status/program/application
baselines, S2 work order, architecture/conventions, authority router, SCI-RTC
REQ-094--098/118--126 and OWNER-060/061/075, and all three effective governance
documents apply. Governance digests/effectiveness ancestry were reverified.
The selected physical-support measurement advances OWNER-060/061 within the
frozen contracts; complete event admission and uncertainty qualification stay
separate. RTC/PTC/VAL ownership and settled half-open intervals are unchanged.

Runtime Consider explicitly requests bounds for coordinates passing the prior
amplitude/consistency check without confirmed recovery. Runtime Learn retains
that request, original parent/VAL and distinct coordinate evidence, then
measures confirmations and the intervening physical/native support. It uses
original producer-valid samples and preserves paired candidate/neighbor
exclusions; no donor samples or refitted model enter the search. Candidate
edge endpoints cannot certify stable support. Confirmation measures covered
integration time, not a fixed row count, and cannot bridge invalid/excluded
cells or physical gaps. Its support is distinct from affected transition
cells and the original fitting mask. Recorded model/support ambiguity, missing
context and unquantified timing/source protection remain visible to subsequent
Consider. No new hard-class pair union is performed before hard admission.
No automatic refit, repeat-until-clean loop or Apply is implemented.

Expected changes: one coherent RTC header; isolated-header and focused tests;
test CMake; new diagnostic output/timing in the existing inert census driver
and runner; these status/work-order records. Prior D2/VAL/native/spike,
background, assessment and consistency application headers remain byte-exact.
Retain only compact confirmations/bounds/counts per event/coordinate and shared
parent handles; no observation copy, per-cell provenance plane, generic
framework, Engine state or cross-stage reach-through.

Gates: threshold/duration equality, signed steps and finite transitions,
original-sample masks, missing/nonfinite/overlapping support, gaps and true
observation edges, internal scan/chunk invariance, compound/ambiguous cases,
exact identity, deterministic repeat and immutability; header isolation,
prior/full local CTest, all config modes and baseline-tool tests. Rerun the
same real-input census and A-F windows, preserving prior outputs and exact
input identities. Measure added request/search/output time and bounded memory.
These are local AppleClang/Homebrew/cached-dependency supplemental results;
no representative Spack/Unity gate or production-runtime claim is implied.
Independent fresh-context read-only exact-SHA review precedes closure.

External evidence belongs under
`/private/tmp/citlali-rtc-jump-transition-2026-09-10`; preserve every earlier
sealed root and failed attempt. Stop/reassess for an actual scientific gap,
outside ownership, changed source/authority, unexpected numerical/resources
or required primary-estimator changes. No source-protection policy, spectral
binding, correction, automatic flagging, filtering, downsampling, PTC/AST/CAL/
MAP work, production, route activation, integration, agent push or cleanup.

Implementation binding/reassessment: the earlier `neighbor_exclusions` is a
merged fitting-mask union that includes the current event's own trial. It
cannot be used unchanged for localization, or subtracted blindly where another
candidate overlaps it. The new learner reconstructs other candidates' unchanged
paired masks from exact membership, using a temporary index of candidate IDs
for requested detectors. Only the current group's fitting trial is omitted
from the confirmation search; its original candidate edge endpoints still
cannot certify a stable side. No primary-fit population changes. Index entries
and peak local neighbor-range counts are measured; sample payload is never
copied. This is a bounded local prerequisite, not a new mask-selection policy.

The recorded last pre-confirmation is the latest sustained agreeing run ending
before the group's earliest candidate endpoint; the post-confirmation is the
first run after its latest endpoint to reach 50 ms. Their intervening half-open
integration support defines the measured bracket and original cell range.
Agreement with both reference models throughout a selected confirmation is
explicitly ambiguous. Multiple candidate edges do not by themselves establish
multiple physical jumps or invalidate a measured group bracket. Preserve their
exact membership and multiplicity separately; physical-event identity stays
unresolved for later Consider, without an invented split/merge rule. Invalid or competing masked support inside the bracket,
nonfinite arithmetic, or non-contiguous/overlapping integration geometry makes
the measurement unavailable. Floating-point adjacency uses the existing native
recovery comparison tolerance; it is not a scientific gap allowance.
Bounds outside the primary fitting exclusion are explicitly marked for later
fit-containment reassessment; no automatic refit or acceptance follows.

Independent source review of `2e24ca315ee5696b1c025e9fb67c6f8f732c5e11`
identified RTC-JT-R01: an added single-edge veto conflated multiple difference
candidates with multiple physical jumps. The repair removes that extra gate,
retains a conditional group bracket and a separate multiplicity/unresolved-
identity fact, and tests both adjacent-edge finite transitions and separated
jumps. No physical split/merge, coherence or acceptance policy is selected.
Initial 18 focused and 989 runnable CTests, four config modes and 207 baseline
tests passed; a later make jobserver-pipe failure affected the next census
build target after `check` completed. The separate repaired-source build,
focused/full regressions, source review and exact real-data census follow.

## Jump transition implementation and measurement completion — 2026-09-10

`TIMESTREAM-SUCCESSOR-RTC-JUMP-TRANSITION-001` completes its bounded local
implementation/measurement gates, ready for owner acceptance. Exact tested
implementation `14c93a4ad4793a554eb98360c7dd9fe9aabe59e7`, tree
`bddc3b6325ea2da57735188c382f5189e03bb8f8`, has parent
`2e24ca315ee5696b1c025e9fb67c6f8f732c5e11` and literal increment base
`8755461edb7e4060ee09c9e1262e8b00b8266026`. Initial implementation and
RTC-JT-R01 repair remain separate preserved commits. Independent repaired-source
and full-evidence three-axis reviews pass with recorded limitations and no
remaining findings. The review history and preliminary failed attempts remain
in the external evidence root; no prior accepted implementation is rewritten.

Runtime Consider requests bounds only for prior consistency passes without
confirmed recovery, retaining exact original VAL and evidence. Runtime Learn
compares original producer-valid samples with the unchanged primary cubic before
and cubic plus signed fitted offset afterward. The inclusive 4-sigma limit uses
the frozen primary pre-event residual scale, with 50 ms continuous confirmation
inside the selected two-second context and physical native run. It reconstructs
other candidates' paired masks by exact membership without changing any primary
fit population. Temporary indices contain candidate IDs, not sample copies.
Last pre and first post confirmations bracket original half-open integration
support. Multiple candidate edges remain an explicit fact, not a single-edge veto
or resolved physical split/merge decision. Exact support/invalid/nonfinite/mask
and fit-containment limits remain explicit. No automatic refit or outer loop.

Later complete Consider must resolve physical identity, timing uncertainty,
source protection, hard-event admission and exact scan-use binding under the
approved owners. Apply must consume the completed plan for the selected
every-intersected-existing-scan detector treatment; no new Apply action is
implemented here. Matching D x/r brackets do not select a coherence tolerance
or hard-class union policy. The development Learn/Consider/Apply workflow is
recorded separately from these runtime responsibilities.

Exact repaired-source local check passes 990/990 runnable tests, including 19
new focused tests and isolated-header compilation; one established
MapFitterLifecycle test remains disabled. Four config modes and 207 baseline-tool
tests passed on the initial implementation; the transition-only repair left
their behavior untouched. CLI version g14c93a4ad and both executable hashes bind
the repaired source. Prior scientific headers are byte-exact to the literal
base; dependency revisions and nine pre-existing cached dirty files match the
preceding census. AppleClang 21 arm64 Release C++20/Homebrew results are local
supplemental evidence; no GCC13/Spack/Unity or production gate is claimed.

Six pilot networks pass before the complete same-corpus run. All 143/143 files,
13 observations, 71,734 channel timestreams and 2,148,911,461 original sample pairs
complete. Raw/Tune/APT identity digests match. All 858 prior candidate, event,
health, detector, example and consistency outputs are byte-identical, preserving
910,241 candidate edges and 242,014 group assessments. Per-network manifests,
request/cause/support accounting and source/executable/dependency post-run
identity checks pass. Fixed A-F sample audits directly verify all measured
confirmations there (D x/r) against the original samples and raw-header duration;
this is not a sample-by-sample independent audit of every corpus bracket.
Fourteen other observations / 155 files still lack binding manifests and remain
untested; the tested observations cover one night. Spectra remain owner-deferred
and source protection unavailable.

Of 484,028 coordinate assessments, 12,815 request this Learn work. Measured
conditional brackets: 11,830; unavailable: 985.
Cause counts in enum order (not requested, measured, background unavailable,
nonfinite, pre missing, post missing, ambiguous reference, invalid support,
competing exclusion, support geometry): [471213, 11830, 0, 0, 0, 599, 2, 0, 384, 0].
Measured records with multiple candidate edges: 8,005;
measured brackets outside the original fitting exclusion: 4,464.
These retain later identity/fit-containment reassessment; no hard-event count,
detector-scan loss fraction or false-positive rate is inferred. D x/r both
produce original cells [6253,6256), a conditional 0.02457594871520996-second
bracket. The duration is sampled support, not a measured intrinsic jump duration.
A/C, B, E and F retain preceding pulse/consistency/health decisions.

Serial local timing on Apple M4 Pro, 48 GiB RAM, with no competing agent
build/test/profiler workload: new request/Learn 0.134633861 s,
new diagnostic output 0.699774168 s. Numerical work is
0.007592719% of preceding same-run driver stages and
0.009170065% of existing event assessment.
All additions including output are 0.047056701% of preceding stages.
Stage total 1774.031461 s; summed child wall time
1845.709842 s, excluding Python reconciliation between
children. Normal desktop scheduling/cache variation remains; no production
fraction or cross-run speedup is claimed. Learner examines 3,796,349 rows.
Peak whole-driver RSS 5339938816 bytes (4.973 GiB) is not
incremental memory. Peak temporary index 48,450 candidate IDs;
peak local mask list 8 ranges; largest logical result
payload 5,267,328 bytes excludes capacity, request/map/allocator
overhead and parent-owned evidence. These measurements do not trigger optimization.

External evidence: `/private/tmp/citlali-rtc-jump-transition-2026-09-10`.
Repaired source review SHA-256: `c9201edda688de2bc93ffc9af17227337df97813702d62870cce1e742f1cf89d`.
Full evidence review SHA-256: `eefc9920fd79956623014ab1596070e96377c0b90851a786c04d9a8d2518e69d`.
`EVIDENCE_SHA256SUMS`: `027269c719c192f130b4049a038b36495c649e10a9920d1435f457a2d8bed515`.
The documentation-only closure has the tested implementation as its direct
parent; its exact commit/tree, independent exact-SHA review and final complete
manifest are recorded receipt-last in external `completion.json` and
`SHA256SUMS`, avoiding a self-referential commit identity inside the closure.

Canonical was reverified at the recorded current authority; both literal
histories and common ancestor remain intact. No merge or canonical admission
occurs; later admission requires the then-live authority and an independent
exact-result-SHA review. Prior sealed evidence remains untouched. No scientific
policy selection beyond the owner-approved transition method, full-scan flag
execution, correction, spectral binding, filtering, downsampling, PTC/AST/CAL/MAP,
production, route activation, Unity operation, agent push or cleanup occurs.

## Owner-approved single reassessment and validation — 2026-09-11

`TIMESTREAM-SUCCESSOR-RTC-JUMP-REASSESSMENT-001` continues the existing S2 slot
from clean closure `6de4d5914c3598eb2d93f22b0bdc0aa32d9ee97f`. The owner's exact
pasted instruction and Tier-2 preflight are preserved at
`/private/tmp/citlali-rtc-jump-reassessment-2026-09-11`. Live read-only verification
finds canonical `86c20b31f7300ba4063be044380b61cd0baf25eb` and remote module
`d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`; closure 6de remains local. No push is
needed to continue locally. Canonical admission is separate and must reconcile
the moving base, preserve accepted implementation ancestry, and receive fresh
independent review of the exact resulting SHA. Previous reviews do not transfer.
All three effective governance digests and their ancestry are verified.

The original 4,464 coordinate brackets outside the fitting exclusion are an
audit population, not proof that a transition influenced its fit. Runtime Learn
recovers the actual original admitted finite, producer-valid fitting cells under
the existing flank, exclusion and support predicates. Runtime Consider requests
one additional paired refit only where a measured transition intersects admitted
fit samples. Its immutable request masks the union of original fitting exclusion,
measured x/r support and existing neighboring-event exclusions. This diagnostic
mask union is not an accepted hard-event union or a physical-event identity.

The successor Learn uses original samples and unchanged two-second primary and
one-second shorter contexts, original polynomial basis, cubic/cubic-plus-offset
models, and minimum 64 samples per flank. The primary pre-fit residual scale,
the shorter pre-fit residual scale, and the original candidate difference-noise
scale remain fixed in their respective uses; they are not pooled or substituted.
The old pre-fit is retained as scale authority, not represented as a new fit to
the successor support. Missing old scales remain unavailable. Consider reuses
the 5-sigma amplitude and 2-sigma agreement checks and recomputed existing
recovery predicate; Learn reuses the existing 4-residual-sigma, 50-ms transition
measurement. One pass only. Any new overlap, inadequate support, ambiguity or
failed consistency remains unresolved. No iterative acceptance or new threshold.

This implements SCI-RTC REQ-013/095 original-pair ordering, REQ-026/027 explicit
failure, REQ-043--045/055--057 truthful support and complete lifecycle,
REQ-094/096/098 finite transitions excluded from plateau estimates, and
REQ-118--125 preserved coordinate origins/protection/immutable attempts. The
settled additive model and physical half-open interval convention remain intact.
Engineering Learn/Consider/Apply governs development and verification separately;
it does not substitute for these runtime evidence and decision boundaries.
No runtime Apply plan or automatic full-scan flag is authorized by this increment.

Validation preserves the previous sealed corpus as control and compares all
seven preceding scientific output files on the same 143 inputs. Retain A-F and
select 24 distinct groups reproducibly across inside/outside/unresolved strata
and observations/detectors before rerunning. Review original x/r, actual fit
samples, both fitted states, bounds, confirmations and residuals. A fixed small
injection catalog includes sharp/finite steps, neighboring spikes and no-jump
controls, including matched original/injected real backgrounds, through candidate
finding. Detection, known-support coverage, excess and unresolved are observations,
not new qualification cuts. D's 24.576 ms is a sampled conditional bracket, not
an intrinsic settling time or calibrated timing-confidence interval.

The every-intersected-existing-scan rule is unchanged. Conditional reporting
keeps coordinate records, detector event groups and unique detector-scan pairs
separate. The current census explicitly lacks a native-to-telescope scan timing
binding; absent an accepted relation, real scan assignments and their changes
remain unavailable. Existing scan metadata must not silently become authority.
Known scan-boundary fixtures verify the intersection arithmetic without
inventing real scan assignments. Protection, harmfulness, physical identity,
timing uncertainty and complete admission remain required for action.

Focused/header, full local CTest, config/baseline gates, fixed corpus preservation,
plots/injections, separate refit/output cost and fresh independent exact-SHA
three-axis review govern completion. The local cached dependency realization is
supplemental, not Unity/Spack or production. No spectra, filters, stitching, new
classifier, MAP/CAL/PTC/AST implementation, canonical admission, push or cleanup.

## Single reassessment completion and conformance — 2026-09-11

Tested source: `aaa86ea1b006cb11aa740adce2429375b13e5026`; tree `0cceff728976b7ee8ced918ab49d91460105482c`;
literal direct parent/control closure: `6de4d5914c3598eb2d93f22b0bdc0aa32d9ee97f`. The same S2 slot
and branch are retained. There is no rebase, redesign or canonical admission.
Final read-only GitHub verification still finds canonical
`86c20b31f7300ba4063be044380b61cd0baf25eb` and remote module
`d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`. The common ancestor remains
`b675bb64a7054f7b24403c79898965e8765cfd02`; publication and exact-result-SHA
canonical admission review are separate. No agent push occurred.

Runtime Learn audits original admitted rows, then learns refits and transition
measurements from the original immutable pair. Explicit Consider products select
one paired refit on actual overlap, reuse the existing amplitude/consistency and
recovery predicates, and leave renewed overlap unresolved. Original two-second
and shorter one-second contexts, polynomial basis, 64-sample flank minimum,
primary/shorter residual scales and candidate sigma_delta roles are unchanged.
The intentional numerical change is removal of newly measured transition cells
from the successor fitting population. It is diagnostic evidence, not a hard
event, donor/offset correction, full-scan flag, or Apply plan. Development
Learn/Consider/Apply is separately satisfied by preflight, bounded implementation,
verification and independent review; it does not substitute for runtime phases.

### Counts and observational limits

The identical 143 files / 13 observations / 71,734 channel streams /
2,148,911,461 sample pairs complete. All 1,001 preceding scientific files compare
byte-for-byte. Original 910,241 candidate edges, 242,014 groups and 484,028
coordinate assessments are unchanged. Of 4,464 outside-exclusion coordinate
brackets, 3,785 overlap their own primary fitting samples; 679 do not. Paired
mask consequences request one refit on 3,226 groups, touching 4,021 originally
measured coordinates. Original measured-coordinate dispositions are:

| Disposition | Count |
|---|---:|
| No refit needed, original measurement retained | 7,809 |
| Measurement survives one refit | 1,263 |
| Required refit unavailable | 2,001 |
| Existing consistency fails | 420 |
| Remaining successor support overlap | 332 |
| Confirmed return to original state | 2 |
| New transition unavailable | 3 |

Thus 9,072 of 11,830 original coordinate measurements remain numerically supported;
2,758 stay unresolved. Another 28 previously unmeasured partner coordinates become
measurable; total remaining coordinate bounds are 9,100 in 6,758 candidate groups
versus 8,980 groups with original bounds. None are counts of accepted physical
events or independent x/r exposures. Unique real detector-scan pairs and their
before/after changes remain unavailable, not zero, because the corpus has no
accepted native-to-existing-scan timing relation. The every-intersected-existing-
scan treatment and fixed shared scan/PCA definition remain settled. Four controlled
half-open boundary fixtures verify arithmetic/deduplication without supplying a
real relation or selecting other detectors.

A-F and 24 deterministic distinct detector groups across the three requested
strata and all 13 observations have detailed x/r, actual fit support, two fitted
states, confirmations, bounds and residual plots. Unavailable fits' retained
populations are labelled eligible-only. All 60 coordinate sample audits pass,
including independent support reconstruction, frozen-scale Huber losses,
4-sigma confirmations lasting at least 50 ms, raw-header integration cells, and
byte-exact A-F control values. D retains its 24.576-ms sampled conditional bracket;
it is not a physical settling time or calibrated confidence interval.

Eighteen fixed trials use a synthetic background and matched unmodified/injected
first-20-second real backgrounds from 152385/4/61 and 152430/8/253. The twelve-cell
ramps are missed at candidate finding in all three backgrounds. Sharp/three-cell
candidates on 152385/4/61 produce original brackets, but none remain measured
after reassessment. Other successful/unsuccessful
outcomes, known physical and affected-cell support coverage, excess support and
unresolved causes are all retained. This is bounded conformance evidence, not
observational qualification or justification to tune thresholds or add a classifier.

### Gates, cost and review

All 1,000 runnable CTests pass (1,001 registered; the established
MapFitterLifecycle.ExactProductSequence remains disabled), including 10 focused
new tests and header isolation. Four config modes and all 207 baseline-tool tests
pass. The same 155 unbound input files remain explicitly deferred. Required failures
propagate; there are no unexpected error-level messages in completed invocations.
The initial fixture-count/exporter-identity corrections and gate attempts are
preserved in the evidence; no prior scientific header or contract bytes changed.

Added numerical time: 2.267548661 s, of which refitting is 2.045849748 s across
12,069 fit calls. This is 0.125414% of 1,808.052299 s in preceding stages of the
same run; maximum per-network ratio is 0.845651%. Additional diagnostic output
and sample exports cost 0.411773753 s. Peak per-fit scratch is 487 rows. Peak
whole-process RSS is 5,343,215,616 bytes (4.976 GiB), not incremental memory.
The source-bound local AppleClang/Homebrew/cached dependency realization is
supplemental; there is no Unity/Spack or production claim. The original nine
cached-dependency dirty files are unchanged, and all 1,783 sealed predecessor
files have been reverified.

Independent source and final evidence reviews pass with recorded limitations and
no remaining findings. Evidence: `/private/tmp/citlali-rtc-jump-reassessment-2026-09-11`.
Scientific evidence manifest SHA-256: `8af212434016221b22c88c2f5716a4055e07b50ff8394a27afc6e2be8dfa8ef5`.
Source review SHA-256: `f4d412ecf1dfc4c7a49aa7572e6af7dc169f70e1548615fb195bf8483add5ae6`.
Final evidence review SHA-256: `0c7c9d61277c56c1fa3f0285bc12df25067e70eeb9ad469f7d58031b92281049`.
The documentation-only closure must have the tested source as its direct parent;
its exact SHA/tree and fresh independent review are recorded in the external completion
receipt and final manifest after that review, not guessed in its own contents.

Conformance: scientific/behavioral PASS for the bounded implementation with the
recorded qualification limits; architecture/ownership PASS; repository/evidence
PASS. Source protection, harmfulness/complete admission, physical identity, timing
uncertainty and real scan binding remain required before action. Spectra remain
owner-deferred. No filtering, stitching, new classifier, automatic flags, MAP/CAL/
PTC/AST implementation, route activation, production, canonical admission, push,
cleanup or subsequent work is performed. Ready for owner acceptance.


## Owner-authorized jump-loss diagnosis — 2026-09-11

`TIMESTREAM-SUCCESSOR-RTC-JUMP-LOSS-DIAGNOSIS-001` continues the same S2 module
slot, owned branch and worktree. Owner Grant Wilson requests a bounded comparison
against known injected truth and saved census breakdown, preserving the current
candidate method. Literal clean base is closure
`8a66bc0203383995a2e6bdcf4261245003a20237`, tree
`f1253f7621a09ebdeeb57306ac1cbbf12c8b617a`; tested control source remains
`aaa86ea1b006cb11aa740adce2429375b13e5026`. All 2,281 sealed control files were
verified before work. New evidence and the full owner directive are at
`/private/tmp/citlali-rtc-jump-loss-diagnosis-2026-09-11`.

Tier-2 preflight reads effective AGENTS, toltec-context routing, three governance
documents and their accepted digests/ancestry, current status, S2/router,
architecture/conventions, SCI-RTC REQ-043--045/055/094/098/118 and the recorded
amplitude/consistency/transition/reassessment decisions. Read-only GitHub lookup
this turn still finds canonical `86c20b31f7300ba4063be044380b61cd0baf25eb` and
published module `d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`. Common ancestor
`b675bb64a7054f7b24403c79898965e8765cfd02` and moving-base/exact-admission-review
obligations are unchanged. No canonical admission or agent push occurs.

The selected sigma_delta is the fixed empirical adjacent-difference scale,
not a parameter standard error. The 5-sigma amplitude and 2-sigma agreement
rules explicitly forbid inventing a plateau-error denominator. Primary and
short pre-event residual scales remain distinct and frozen for reassessment.
This task diagnoses those decisions; it does not replace their scientific policy.
Support/design changes do not by themselves establish an offset covariance.

Included work: read saved 143-file results for mutually exclusive original-
coordinate terminal causes, additional evaluated conditions, distinct paired-
candidate-group counts and health/concentration context. Replay the same 18 fixed
injection cases and add two predetermined synthetic finite-pulse controls (3 and
12 cells), keeping original samples, validity, amplitudes, times and noise.
Compare original measurement, current single reassessment, and a test-only fit
whose exclusion covers known transition/pulse support. The latter preserves the
original outer context, time basis, frozen scale roles and other candidate
exclusions, while replacing the target's inferred exclusion; it cannot become a
production mask. Missing candidates stay in end-to-end denominators. A supplied
location for missed/no-candidate cases is explicitly separate diagnostic evidence.
An ungated numerical transition probe may expose a later decision's effect but
never counts as an end-to-end retained measurement.

Expected changes are the inert injection executable and private diagnostic
helpers, saved-result analysis/report tooling, this handoff and status. Original
runtime headers and frozen contracts remain byte-identical. Test-only explicit-
anchor adapters copy the exact existing recovery/transition numerical bodies,
replacing candidate lookup only; parity with production functions is checked for
every detected replay group/coordinate/model. They cannot fabricate immutable RTC
producer evidence or mutate a parent. Review must inspect this copied-body scope
and parity as a specific provenance risk. Original typed runtime Learn/Consider
ownership is preserved; engineering learn/consider/apply is a separate workflow.

Gates: unchanged sealed control and production source, exact replay of all 18
old injection outputs, explicit denominators and sample/fit/truth/decision audits,
focused and broad local tests, config and baseline-tool gates as applicable,
headless visual inspection and fresh independent exact-SHA three-axis review.
Preserve failure attempts and inherited cached-dependency bytes. Local AppleClang/
Homebrew/cached dependencies are supplemental; no Unity/Spack reproduction or
production qualification is required or claimed. Do not rerun the full corpus.

Stop/reassess before a new scientific rule, production algorithm change, changed
owner or runtime interface, altered original input, broadened sample context,
new uncertainty assumption or other scope expansion. The final compact report
must recommend at most one smallest next change, or retention unchanged; a
recommendation is not authorization to implement it. No automatic flags, offset
correction, filtering, spectra, new candidate policy, Apply, PTC/CAL/AST/MAP,
route activation, production, canonical integration, push or cleanup occurs.


## Jump-loss diagnosis completion — 2026-09-11

Exact diagnostic source `8ef77477823a5393f9cc7860d8f963509aee0fc8`, tree
`5d6d10ac9ef5d0068898ad3f0053f7e559062fc2`, has direct parent/literal base
`8a66bc0203383995a2e6bdcf4261245003a20237`. The changed executable is inert
injection tooling, with private test-only anchors and saved-result/report scripts.
Original runtime headers, tests/build configuration, governance and frozen
contracts are unchanged. Candidate method, 2-second/1-second contexts, cubic and
offset models, 64-sample support minimum, original validity, three fixed scale
roles, amplitude/agreement/4-sigma/50-ms rules and single-pass limit are preserved.

The saved 143-file census gives 7,809 original coordinate measurements retained
without a refit, 1,263 retained after a refit and 2,758 unresolved, from 11,830.
There are 3,226 requested paired refits touching 4,021 original coordinates;
28 new partner measurements are counted separately. Terminal causes are disjoint:
1,994 insufficient fitting support, 7 numerical fit failures, 343 short/long
offset disagreements, 1 sign disagreement, 44 primary-amplitude failures, 32
short-amplitude failures, 2 confirmed recoveries, 3 unavailable transitions and
332 remaining paired overlaps. Support takes reporting precedence within the
combined availability OR when a numerical failure also occurs; all other
actually evaluated conditions remain recorded. Recovery changes persistent-jump
interpretation; it does not assert no disturbance occurred.

The 8,980 original paired candidate groups split into 5,754 no-refit retained,
942 retaining all original coordinate measurements after refit, 57 losing some
and 2,227 losing all. Including new partner evidence, 6,758 groups retain at least
one bound. These are candidate groups, not accepted physical-event identities.
Losses occur across 1,039 observation/network/channel combinations; top ten supply
434 (15.7%). Of 17 saved health-concern streams, none supplied an original measured
bound; absence of this marker is not health certification. Full observation and
channel populations and additional evaluated conditions remain in the report's
linked machine-readable census. No full data-corpus rerun was performed.

All 18 original injection records reproduce byte-for-byte. Two fixed synthetic
finite-pulse controls extend the catalog to 20 cases / 40 coordinate comparisons.
Across 12 injected jumps /24 coordinates, 18 coordinates have candidates at the
injection. Original retention is 16/24 and current retention 9/24; known-truth
fitting is test-only and does not bypass subsequent decisions. All 18 detected
coordinates have numerical primary offset fits, including withheld cases. Every
x/r/version retains offset availability, raw added-offset difference, incremental
error against the same-support uninjected background, signed physical boundary
error in native cells and seconds, and first decisive disposition. Real controls
are not assumed event-free. Synthetic unmodified/spike/3-cell-pulse/12-cell-pulse
controls retain zero persistent-jump measurements in eight coordinate cases.

The failed sharp injection on 152385/network 4/channel 61 is grouped with later
background candidate edges. The existing endpoint-span rule expands its bound to
[1220,1324), 0.851968050 seconds, around instantaneous truth at cell 1221's
midpoint. The reassessment mask [1214,1330) leaves only 20 post samples for the
short fit (previously 104, or 89 with the neighboring spike), below 64. That support
failure precedes the 2-sigma consistency test. Truth exclusion restores 111 (or 96)
short post samples and passes existing x/r amplitude/sign/agreement checks, but
the unchanged grouping-based bound still overlaps correct fitting samples and
therefore remains withheld. A distinct 152430/network 8/channel 253 x-coordinate
case fails offset agreement even with truth exclusion; r survives current
reassessment. Its uninjected comparison exposes background contribution, without
claiming a real-event truth or a calibrated covariance.

The twelve-cell ramps last about 98.304 ms and miss candidate finding in x/r on
all three backgrounds. Exact durations and amplitudes are in the report. Supplied-
location current probes retain both coordinates on synthetic and 152385 backgrounds;
152430 remains unresolved for overlap. These probes never count as end-to-end
recovery. The unchanged sigma_delta is the approved empirical adjacent-difference
comparison tolerance, not the standard error of revised offset estimates. Frozen
residual noise does not establish parameter uncertainty under changed support,
design and shared-data dependence; no new uncertainty assumption was adopted.

Recommendation only: separate the onset transition's support from later candidate
edges already grouped with it, retaining those later edges as visible/masked
context rather than forcing intervening stable plateau into the onset transition.
Do not implement this recommendation without the next bounded owner decision.
No new split/merge classifier, threshold, model or iterative pass was selected,
and this evidence does not show all 2,758 lost measurements are safe to retain.

Exact-source local CLI/injection builds, all 1,000 runnable CTests (one established
disabled test), 102 focused tests, four config modes and 207 baseline-tool tests
pass. Independent data audits cover 97,680 sample values, 384 available fit losses
and 132 production/explicit-anchor parity comparisons. Pilot/final output is
deterministic. The independent ramp formula differs by at most 1 floating-point
ULP on a few cells; a retained failed strict-equality audit was narrowed to a
2-ULP arithmetic tolerance, separate from exact legacy reproduction. No input or
scientific rule changed. All 2,281 control files and three copied executables are
preserved; nine inherited cached-dependency dirty files remain unchanged.

Independent fresh-context review of this exact source and sealed evidence passes
with recorded limitations and no findings on scientific/behavioral conformance,
architecture/ownership and repository/evidence hygiene. Source/evidence review
SHA-256: `7314b57211849965d0f614060f93cc3b75bdee71ae7acffcf7771bffd6eb2237`. The 181-file scientific evidence
manifest SHA-256 is `83d9ffc6fa9fce3fa8734bcf716a85ca51e5c895aaca88a6f46d7c7af986c93b`. Evidence root:
`/private/tmp/citlali-rtc-jump-loss-diagnosis-2026-09-11`. All four final figures were viewed headlessly.
Local AppleClang/Homebrew/cached dependencies remain supplemental; no Unity/Spack
or production qualification is required or claimed. No unexpected final replay
stderr is present. Live canonical/module refs remain respectively
`86c20b31f7300ba4063be044380b61cd0baf25eb` and
`d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`; future admission must reverify moving
canonical authority and receive its own exact-SHA review.

The documentation-only closure must have this tested diagnostic source as its
direct parent. Its exact SHA/tree and fresh independent review will be recorded in the
external completion receipt and final manifest after review, not guessed within
its own contents. No automatic flags, Apply, correction, filtering, spectra,
new candidate policy, PTC/CAL/AST/MAP, route activation, production, canonical
integration, agent push, cleanup or later work occurs. Ready for owner acceptance.


## Owner-approved onset-support trial — 2026-09-12

`TIMESTREAM-SUCCESSOR-RTC-JUMP-ONSET-TRIAL-001`: the owner says “Let's try the
test with your recommendation.” Continue the owned S2 module at clean literal
base `019a0ee3fe710c1a62da3107286cf3c92794a5ca`, tree
`418e5dfd05447eedcbfece7aba257c75e8dd79c5`, preserving the reviewed diagnostic
and single-pass controls. This is approval for a bounded implementation trial,
not general scientific acceptance, canonical admission, activation or production.
Read-only live refs remain canonical `86c20b31f7300ba4063be044380b61cd0baf25eb`
and module `d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`. Three effective governance
digests and accepted ancestry were reverified; owned AGENTS, toltec-context,
status, SCI-RTC REQ-055/094/096/098/118 and recorded owner rules were read.

The trial defines the onset anchor as the connected component of native candidate
edge cells containing the original seed: edges sharing a native endpoint cell
connect, with no new temporal threshold. Parent assessment members are already
chronological and seed-first; membership and physical-event identity remain
unchanged. Disconnected later members retain their existing +/-50ms neighbor
guards. This removes the implicit rule that their entire intervening span must
belong to the onset. Stable confirmation still uses original samples and the
unchanged 4-sigma/50ms rule; masks still interrupt confirmation and a competing
exclusion inside the inferred bound still makes it unavailable. A second genuine
jump is not silently combined or assigned a corrected offset.

Runtime Consider requests the existing RTC Learn transition product; Learn
produces conditional x/r physical bounds and causes; existing Consider audits
actual fitting support and permits the same single reassessment. Engineering
learn/consider/apply remains a separate development workflow. No runtime Apply
plan or accepted event is produced. Candidate finding, parent grouping, recovery,
initial fits, 2s/1s contexts, frozen scale roles, amplitude/consistency thresholds,
validity, protection and timing uncertainty policy remain unchanged.

Preserve all 194 files and both source-bound binaries of the previous diagnosis.
Use the same 20 fixed trials, including the synthetic no-jump/pulse controls,
same two real backgrounds, x/r truth errors and all missed/unresolved cases.
The 12-cell misses stay in end-to-end denominators. Do not rerun the 143-file
corpus. Focused regressions must cover connected edges, later disturbances,
competing guards and genuine later steps; broad local tests/config/baseline
and fresh independent exact-SHA source/evidence and closure reviews remain
required. Local evidence is supplemental, without a Unity/Spack/production
claim. Onset selection scans existing ordered members without an allocation; the
existing neighbor-mask vector may contain more guards. Targeted timing cannot
establish full-corpus performance.

Scope: transition learner, focused tests, inert diagnostic adapters/comparison
reports and these existing documentation records. No filter, flag, correction,
spectral context, new candidate/grouping policy, MAP/CAL/PTC/AST, route,
production, canonical admission, push, cleanup or later increment. Record and
reassess any unexpected outcome; do not tune further policy to obtain a pass.


## Onset-support trial completion — 2026-09-12

Exact tested source `d929070dbdfb8d8262bad109126d000b5c941462`, tree
`10a02bdb8ff39a7f647eed13d14a5b288a4a1dd0`, has direct parent/literal base
`019a0ee3fe710c1a62da3107286cf3c92794a5ca`. The intentional numerical change
is confined to RTC Learn's conditional transition support: start with the original
seed edge cells, extend through members sharing an endpoint cell, and retain
disconnected members under existing neighbor masks. The immutable assessment
producer supplies chronological seed-first membership. No new time threshold,
candidate split/merge rule or physical-event identity is introduced. The policy
identity is `rtc-jump-transition-onset-2026-09-12-v2`; the diagnostic schema records
seed/onset cells and every neighbor exclusion. Private explicit-location adapters
remain test-only and are audited against production calculations.

Runtime Consider still requests the same original-sample Learn evidence and
audits actual fit support under the same maximum single reassessment. Original
finding/grouping, initial cubic/offset fits, 2s/1s context, 64-sample minimum,
fixed scale roles, amplitude/sign/consistency checks, +/-4-sigma/50ms confirmation,
recovery, validity and source-protection rules are unchanged. This delivers a
bounded Learn correction consumed by existing Consider; the engineering workflow
does not substitute for those runtime responsibilities. No Apply plan, accepted
event, flag, correction or scan assignment is produced.

The preserved diagnosis executable reproduces all its targeted outputs exactly.
The same 20 trial cases /40 coordinate comparisons use unchanged synthetic and
two real background samples and added signals. Of 24 injected-jump coordinates,
18 have candidates; retained measurements improve from 9 to 15, with six gains
and zero losses. The gains are x/r for sharp, three-cell and sharp-plus-spike
injections on 152385/network 4/channel 61. The sharp and sharp-plus-spike bounds
change from the original broad [1220,1324) interval (0.851968050 seconds) to
[1220,1223), about 24.576 ms. The instantaneous truth is enclosed with signed
endpoint errors of -1.5/+1.5 native cells. The three-cell injection bound is
[1221,1224), matching its injected extent. Short post fitting support is now
104 samples, or 89 with the neighboring spike, rather than 20; these recovered
cases require no additional refit. All actual masks, fits, offsets and signed
truth errors remain in the report's JSON/TSV, including unavailable outcomes.

Three detected x-coordinate injections on 152430/network 8/channel 253 remain
unresolved: sharp and three-cell cases fail the unchanged initial offset
consistency rule; sharp-plus-spike is withheld by a competing neighbor exclusion
while measuring transition support. Its r measurement survives with a shorter
bound. The six twelve-cell-ramp coordinate misses remain end-to-end misses at
the original duration/amplitude; supplied-location probes remain separately
labelled diagnostic results. Synthetic unmodified/spike/three-cell-pulse/
twelve-cell-pulse controls retain 0/8 persistent-jump coordinates. Real unmodified
backgrounds are not assumed event-free. No evidence here establishes that every
loss in the earlier 143-file census is recoverable; the full corpus was not rerun.

Known-support fits may still report support conflict because conservative learned
bounds include adjacent native cells outside the exact injected support. Their
availability is not a production-retention gate or a calibrated parameter
uncertainty. The approved sigma_delta remains a fixed empirical comparison
tolerance. Event identity, harmfulness, protection, timing uncertainty and the
real native-to-existing-scan relation remain separate admission requirements.

Exact-source CLI and injection builds, all 1,002 runnable local CTests (one
established disabled test), 31 focused tests, all four require-all config modes
and 207 baseline-tool tests pass. Added regressions cover later grouped x/r
spikes, an overlapping disconnected guard and a genuine later step. Data audits
verify 97,680 original values, 396 available model losses, 112 explicit-anchor/
production parity comparisons, 40 unchanged upstream coordinate chains and
20 independent onset/neighbor-mask reconstructions. Eighteen same-source
legacy/diagnostic cases agree; this is not a claim that the changed trial matches
the old control's transition output. Pilot/final replay bytes are deterministic.
All four final plots were inspected headlessly.

All 194 sealed predecessor files and two copied control binaries are preserved,
as are 37 other timestream headers, contracts/governance/validation/build
configuration and nine inherited cached-dependency dirty files. Retained initial
attempts record a misspelled pilot test target, an audit-only correction separating
forced-location full-mask probes from production-local masks, and a report-label
clarification for candidate misses. None caused additional scientific tuning.
The onset scan uses a constant-size result without allocation; the existing
neighbor-mask vector may grow. Concurrent whole-process diagnostic timings are
not an incremental cost measurement, and no full-corpus performance inference is
made. AppleClang/Homebrew/cached-dependency results are supplemental local
evidence; Unity/Spack or production qualification is neither required nor claimed.

Fresh independent exact-source/evidence review passes with recorded limitations
and no findings on scientific/behavioral conformance, architecture/ownership
and repository/evidence hygiene. Source review SHA-256:
`a53f8ec4f625a2685a22e5504093df2bcdc279523b877a8181022acaeb58aaed`.
The 189-file scientific evidence manifest SHA-256 is
`7b73ef2af1b824bfc603ed67341f7b30a630f33c6a8ee1166092a9e37f73f63f`.
Evidence root: `/private/tmp/citlali-rtc-jump-onset-trial-2026-09-12`.
The report is `report-final-01/README.md`; source and gate bindings are alongside it.
No unexpected final replay stderr is present. Live canonical/module refs were
verified read-only as `86c20b31f7300ba4063be044380b61cd0baf25eb` and
`d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`; future admission must reverify
moving canonical authority and receive its own independent exact-SHA review.

The documentation-only closure must have this tested source as its direct parent.
Its exact SHA/tree and fresh independent review are recorded in the external
completion receipt and final manifest after review. Ready for owner acceptance
of this bounded trial; no full-corpus rerun, additional scientific policy,
filtering, flags, correction, spectra, Apply, PTC/CAL/AST/MAP, route activation,
production, canonical integration, agent push, cleanup or later increment occurs.


## Owner-approved onset-support corpus test — 2026-09-12

`TIMESTREAM-SUCCESSOR-RTC-JUMP-ONSET-CORPUS-001`: the owner requests “Let's run
this on the full corpus unless you have another idea.” Continue the owned S2
module at clean literal base `f0deca48083d6b7a20931a1aa09d4b706063dd8c`, tree
`e11d52a8fb058e68c625dbb19fa6ea23e1322239`. This authorizes the unchanged
reviewed onset method's full observational test, not a new numerical increment.
The three effective governance documents, owned AGENTS/status, toltec-context,
scientific and recorded owner requirements were recovered. Read-only live refs
remain canonical `86c20b31f7300ba4063be044380b61cd0baf25eb` and module
`d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`; future admission must reverify both
moving ancestry and its independent exact-SHA candidate review.

Preserve all 196 sealed onset-trial files and 2,281 preceding reassessment files,
their executable identities and inherited cached dependencies. Use the same
143 eligible inputs across 13 observations, 71,734 channel timestreams and
2,148,911,461 paired samples; retain the same 155 explicit missing-binding
deferrals without admitting new files. Keep 152390 as a timestream fixture.
Runtime Learn/Consider and the original maximum one reassessment are unchanged;
the engineering workflow is separate from these runtime responsibilities.

Only the inert Python census verifier and existing documentation need changes.
The verifier independently checks the reviewed seed-connected onset component
and explicitly allows transition output differences only for the named v1-to-v2
policy pair. All six earlier outputs remain byte-identical; input raw/Tune/APT
hashes, candidate identities, original fits and scales remain exact. Compare
the full original and final coordinate populations, paired groups, gains/losses,
support, bounds, unresolved causes and observation/channel concentration.
Retain all 30 previously selected examples; inspect gains/losses with fixed
deterministic selections if needed. No numerical policy is tuned to improve
retention. Tests and source review remain inherited where runtime bytes are
unchanged; build and CLI binding must identify this exact run source, and new
verifier behavior and evidence receive focused and independent exact-SHA review.

Run networks serially locally, preserving full native runs and bounded memory.
Use existing numerical stage timers to separate transition/reassessment costs
from original fitting, ingress and diagnostic output. Process RSS is not
incremental memory; previous-run timing is not a controlled speed benchmark.
The same supplemental AppleClang/Homebrew/cached-dependency environment is used;
no Unity access or Spack/application-generation qualification is claimed.
Unexpected behavior, provenance, numerical, logging or resource differences
trigger recorded reassessment; diagnose within this test, preserve failures and
stop before any new scientific choice. No filters, flags, correction, spectral
context, Apply, PTC/CAL/AST/MAP, new data download, canonical integration, push,
route activation, production or cleanup is included.

## Onset-support full-corpus completion — 2026-09-12

The authorized test completed on exact source
`e83815a91fc87d658fd63f1da40ecac82d602105`, tree
`6d91ff288f53ce5c65e8bb0ca2211af32eeccf09`, with literal base/direct parent
`f0deca48083d6b7a20931a1aa09d4b706063dd8c`. Its only changed source file is
`tools/timestream_successor/run_rtc_event_assessment_census.py`; existing status
and this handoff record scope. All 38 timestream headers, C++ tools, tests,
build configuration, contracts and governance remain unchanged. The rebuilt
local CLI identifies `e83815a91`; the source/evidence receipt binds the exact
CLI, census executable and inherited nine cached dependency file changes.

All 143 eligible files complete with unchanged raw/Tune/APT identities:
13 observations, 71,734 channel timestreams, 2,148,911,461 paired samples,
910,241 original candidate edges, 242,014 groups and 484,028 coordinate
assessments. The same 12,815 initial transition requests are made. All 858
pre-transition output files, 143 selected-sample exports and two injection
backgrounds are byte-identical to the preceding corpus. The same 155
missing-binding inventory entries remain deferred, not passed or admitted.

| Evidence population | Preserved control | Onset method |
| --- | ---: | ---: |
| Initial available coordinate bounds | 11,830 | 10,318 |
| Paired refit requests | 3,226 | 1,314 |
| Final retained coordinates | 9,100 | 9,667 |
| Groups with at least one retained coordinate | 6,758 | 7,169 |

The final comparison is 7,910 retained by both, 1,757 gained, 1,190 lost and
473,171 retained by neither. Gains comprise 771 x and 986 r coordinates;
losses comprise 637 x and 553 r coordinates. The control total explicitly
includes its 9,072 retained original measurements and 28 new partners; the
onset total includes 9,652 retained onset-initial measurements and 15 partners.
The report also keeps the original 11,830-bound and 12,815-request denominators
fixed and records paired 0/1/2-coordinate changes.

Of the gains, 1,350 were formerly unavailable through fitting-support loss,
329 lacked a post confirmation and 78 follow other recorded paths. Of the
losses, 1,170 are initial competing exclusions, 19 lose the earlier
partner-refit opportunity and one lacks post confirmation. The earlier
107-file partial result (387 gains, 500 losses) is preserved as partial evidence;
the full population has a net gain of 567. Gains concentrate in 152390/152392,
while several shorter observations have net losses. These are conditional
coordinate measurements, not accepted physical events or truth-accuracy rates.

The original A-F and 24 fixed selections are preserved, with four gained and
four lost groups selected by a fixed SHA-256 rank and unique detector rule.
All 38 comparison figures were inspected through contact sheets with individual
detail views. In G01 (152390/network 11/channel 481), 70 more post fitting
samples survive and both coordinates are retained while the nearby spike guard
remains. Some gains occur among several large level changes with substantial
fit residuals; a retained onset does not certify the whole disturbance model.
No thresholds or fit rules were tuned to improve results.

A reproduced loss at 152432/network 2/channel 292/event 366/r has exactly the
same original fit and fitting samples. Seed cells 5978–5979 are disconnected
from later original-group candidate edges starting at 5981, about 24.576 ms
after the seed edge. Existing ±50 ms guards remove confirmation samples and
overlap the resulting proposed bound, causing competing-exclusion withholding.
The independent reconstruction accounts for all 14 counted excluded cells.
This explains one step-like example without assigning physical-event truth,
merging edges or authorizing a new guard exception.

All-event support/decision audits, 11,693 available-bound checks, the 60-coordinate
fixed-sample audit and 152 coordinate/version comparison checks pass. Replays
preserve 118 same-source scientific files. Independent tests exercise 1,016
graph/member-order cases and 12 mocked actual-driver policy/preservation gates.
Exact local CLI/census builds pass. Prior 1,002 runnable CTests, 31 focused tests,
four config modes and 207 baseline tests remain inherited from reviewed
`d929070dbdfb8d8262bad109126d000b5c941462`; these suites were not rerun at
`e83815a91`. No unexpected runtime error-level output occurs; 143 inherited
raw-kind warning lines remain explicit. External plotting/checker failures and
their bounded repairs are retained separately; scientific outputs were unchanged.

Existing numerical stage clocks measure 1.793314045 seconds for the complete
transition/reassessment sequence, 0.0950023% of preceding input/numerical work
(1,887.652991 seconds). Transition learning costs 0.146556333 seconds and
refitting 1.416161129 seconds. Diagnostic output totals 15.757960 seconds;
summed child wall time is 1,982.896693 seconds and peak whole-process RSS
5,348,720,640 bytes (4.981 GiB). Concurrent local analysis and targeted replays
prevent controlled speed comparisons; RSS is not incremental memory.
The actual environment remains supplemental AppleClang/Homebrew with bound
cached dependencies, not Unity/Spack, V2 application reproduction or production.

Runtime RTC Learn still supplies conditional original-sample evidence and
unchanged Consider still audits support and requests at most one reassessment.
No runtime Apply plan is produced. Separately, the engineering workflow uses
these conformance measurements for the next owner decision; it does not replace
the required runtime phase boundaries. Preserved D2/VAL, source protection and
spectral limits remain unchanged. Real scan admission and downstream work remain
outside this test.

Evidence root:
`/private/tmp/citlali-rtc-jump-onset-corpus-2026-09-12`.
The sealed 2,401-file `EVIDENCE_SHA256SUMS` has digest
`61fd2c4343d669025329698fba60137abe01abd98a054b8ec496c2174798a5e4`.
All 196 prior onset-trial files and 2,281 preceding reassessment files, control
binaries and dependency bytes were reverified. Independent exact-source/evidence
review passes with recorded limitations and no findings on the three governance
axes; the review identity/digest is recorded below. This documents conformance
of the bounded test and its evidence, not automatic adoption of the rule.

Recommend reviewing nearby-edge membership/guard treatment before promoting
the onset method unchanged. No new connection interval, grouping rule,
classifier, threshold, correction or Apply choice is made here. The separate
two-document closure must have the tested source as its direct parent; its exact SHA/tree,
fresh independent exact-SHA review and final completion binding are recorded
externally after review. Future canonical admission must reverify live moving
ancestry and review the exact integration candidate independently. No merge,
push, route activation, filtering, flags, spectra, PTC/CAL/AST/MAP, new data,
production or cleanup occurs. Ready for owner review of this completed test.

Source/evidence review: `review-01.md`, SHA-256
`d5d48a9e9e504c1e980433d60464eee4ec2a0d0d42ec2d5d8709b883333f93bb`,
reviewer `/root/rtc_onset_corpus_exact_review`.

## Approved plateau boundary increment — 2026-09-12

Following the preserved full-corpus comparison and read-only characterization,
the owner approved the proposed rule: retain nearby original-group edges as
an unresolved transition until original samples demonstrate the fitted
post-jump level continuously for 50 ms within ±4 frozen residual sigma.
Original candidate cells cannot certify the plateau. Members after it retain
the existing ±50 ms guards; if a later guard reaches into the first selected
confirmation or bracket, withhold rather than absorb that later disturbance or
search again. Other original groups keep their guards throughout. x/r retain
separate model, scale, support, parent and outcome identities. This is a
provisional support role, not new candidate grouping or resolved physical-event
identity. Frozen contract REQ-096/118 and owner decisions 060/061/075 remain
authorities; this owner binding resolves only the narrow separator decision.

Work order TIMESTREAM-SUCCESSOR-RTC-JUMP-PLATEAU-BOUNDARY-001 is Tier 2,
based directly on reviewed closure bed75cd828a2db9655f311545cc739c0cd3b32c6,
in the existing codex/timestream-successor-rtc-event-background-001 slot.
Governance remains effective 06a3ade51c1b3f38887295433d913811bf25cd14 as
incorporated by 77507836325eff9f469062d5884481ea37599594; current status
governs sequencing. The three governance document digests remain unchanged.
Expected paths are the transition header, private injection parity adapter and
output, inert corpus verifier, focused transition tests and these two records.
Runtime Learn owns the original-sample separator measurement; existing
Consider consumes it and limits reassessment to one. Runtime Apply remains
unimplemented in this increment. The separate engineering Learn/Consider/Apply
workflow requires source-bound local tests and independent conformance review.

Focused tests cover nearby disconnected edges, later separated disturbances,
overlapping later guards, native physical duration, invalidity/gaps, x/r
origins and unchanged threshold/identity semantics. Broader gates are local
CTest, config and baseline tools; preserved injections and the same eligible
143-file corpus demonstrate affected behavior, all gains/losses, original-sample
boundaries, determinism and timing. Cached local dependency bytes are bound
and preserved. This is supplemental local evidence, not Unity/Spack or
production qualification. Independent fresh-context exact-source/evidence and
separate documentation-only closure reviews are required. A new scientific gap
or broadened responsibility triggers reassessment; no unapproved policy tuning.

External root: /private/tmp/citlali-rtc-jump-guard-boundary-2026-09-12.
The sealed preparation remains historical; approved-preflight.json supersedes
only its pending-owner status. No integration, push, route, flags, correction,
spectra, MAP/CAL/PTC/AST, new data, Unity access or cleanup is authorized. Future
canonical admission must verify the live moving authority and independently
review the exact integration SHA.

## Plateau-boundary implementation completion — 2026-09-12

Work order TIMESTREAM-SUCCESSOR-RTC-JUMP-PLATEAU-BOUNDARY-001 completed on
exact source `81a35ced25d45eeccafabc3243b5aa3010893625`, tree
`cf70c8b7f35c825f85ca37260026ba0299bb9802`, with literal base/direct parent
`bed75cd828a2db9655f311545cc739c0cd3b32c6`. It changes the transition header,
focused transition tests, private injection parity adapter/output, inert Python
census verifier and the two existing scope records. The other 37 timestream
headers, frozen contracts, effective governance, D2/VAL, original event/background learner and
Consider/reassessment implementation are unchanged. The worktree remains the
existing `codex/timestream-successor-rtc-event-background-001` slot.

The approved support binding is executable: original-group edge cells remain
unresolved until the first original-sample post-level confirmation lasts at
least 50 ms within ±4 frozen residual sigma. Candidate edge cells in either
coordinate cannot certify either coordinate's plateau. Seed-connected onset
cells remain mandatory. Other groups retain their guards throughout; members
after the first post confirmation retain their original ±50 ms guards. If one
of these later guards overlaps the selected pre confirmation, bracket or post
confirmation, the measurement is unavailable without another separator search.
Original membership remains immutable and does not become physical-event
identity. No threshold, fit, grouping, window, scale or refit-count rule changes.

Runtime Learn owns this conditional coordinate-local support measurement and
consumes original native x/r, initial VAL/source-protection parents, frozen
coordinate model and residual scale. Existing Consider audits the output and
may request the same one paired reassessment; the remeasurement uses the same
helper and preserves separate x/r support and origins. No runtime Apply plan
or sample mutation is produced. Separately, engineering Learn/Consider/Apply
uses the source-bound tests and measurements below to establish conformance;
that workflow does not substitute for the runtime boundaries.

All 143 eligible network files complete, preserving the exact 13-observation,
71,734-channel, 2,148,911,461-paired-sample population. All 910,241 candidate
edges, 242,014 groups, 484,028 coordinate assessments and 12,815 initial
transition requests remain fixed. The same 155 missing-binding inventory
entries remain deferred, not admitted. All 858 pre-transition output files,
143 original-sample exports and both injection backgrounds are byte-identical
to the preserved control. Selected replays preserve 81 same-source scientific
files across nine network invocations; both fixed injection replays are
numerically identical. No unexpected runtime error-level output occurs;
143 inherited raw-kind warning lines remain explicit.

| Evidence population | Preserved onset control | Plateau boundary |
| --- | ---: | ---: |
| Initial available coordinate bounds | 10,318 | 12,219 |
| Paired refit requests | 1,314 | 2,030 |
| Final retained coordinates | 9,667 | 11,150 |
| Groups with at least one retained coordinate | 7,169 | 8,389 |

The full final comparison is 9,656 retained by both, 1,494 gained, 11 lost and
472,867 retained by neither. Gains comprise 773 x and 721 r coordinates;
losses comprise five x and six r coordinates. The control total includes 9,652
retained own-initial measurements plus 15 partner recoveries; the new total
includes 11,121 retained own-initial measurements plus 29 partner recoveries.
All coordinate and paired 0/1/2-coordinate consequences are recorded.

Of the gains, 1,478 recover competing-exclusion losses, 14 are partner-coordinate
recoveries and two formerly lacked post confirmation. Of the losses, seven
fail fixed fitting support, two fail short/long offset consistency and two
retain paired overlap. Every earlier changed coordinate is followed: 1,746
of the preceding 1,757 gains remain retained, 1,091 of 1,170 prior guard losses
are recovered, and nine of the other 20 prior losses are recovered. The 79
remaining prior guard losses are 77 competing exclusions, one unavailable
transition and one remaining overlap. None are silently dropped from counts.

The motivating previous L04 (now PL04), 152432/network 2/channel 292/group 366/r,
retains native cells [5976,5985), 73.728 ms, with post confirmation [5985,5992),
57.344 ms. Original fitting rows and cubic-plus-offset fit remain identical.
Nearby original-group edges now belong to unresolved support; the x coordinate
remains withheld independently. Representative new losses preserve the support
rule: 152418/network 11/channel 11/group 288 leaves zero short-fit post samples
instead of 85, and 152430/network 2/channel 359/group 300 leaves 37 instead of
87, below the unchanged minimum 64. Some gained bounds (G01/G03) occur among
several levels and large residuals. Conditional support does not certify the
entire disturbance model, physical truth, or an Apply disposition.

The same 20 fixed injection trials retain 15/24 injected-jump coordinate
measurements, with zero final gains/losses relative to the onset method.
Detection remains 18/24; all six twelve-cell gradual-ramp misses remain in the
denominator. Synthetic no-jump/spike/pulse controls retain zero persistent
jumps in eight coordinates. One new initial x bound on the complex real
background reaches reassessment but fails the unchanged offset-consistency
check. Supplied-location diagnostics remain separate from candidate detection
and cannot convert a missed injection into a retained runtime candidate.
Original candidates, groups, fitting values, losses, scales and test truth
support remain preserved; private adapter/runtime parity passes.

All 1,004 runnable CTests out of 1,005 registered, 33 focused transition/
reassessment tests, four config modes and 207 baseline-tool tests pass at this
exact source. The sole established disabled test remains
`citlali::MapFitterLifecycle.ExactProductSequence`. All 14,164 available bounds
(12,219 initial plus 1,945 remeasured) are independently checked against raw
producer clocks, physical cells, membership and all relevant guards. The 60
fixed-coordinate audit, 184 comparison coordinate/version original-sample
checks and 50 first-post confirmations pass. All 46 labeled figures were
inspected through contact sheets, with individual gain/loss/motivating views.
The fixed 30, preceding eight and new eight selections are preserved; two
previous/new selections overlap and do not imply 46 distinct physical groups.

Existing stage clocks measure 2.451022743 seconds for the complete transition/
reassessment chain, 0.135191% of preceding input/numerical work. Transition
Learn costs 0.137657917 seconds, refitting 2.10128191 seconds; original event
assessment remains 1,517.717125126 seconds. Summed child wall time is
1,903.999702742 seconds; peak whole-process RSS is 5,336,137,728 bytes
(4.970 GiB). Concurrent build/test/diagnostic work prevents a controlled speed
comparison; RSS does not measure incremental memory. The recorded Apple M4 Pro,
AppleClang 21 arm64 Release C++20, Homebrew and cached dependencies remain
supplemental local evidence, not Unity/Spack or production qualification. All
nine existing dirty dependency file digests and the three control binaries
remain unchanged. The exact CLI, census and injection executables are bound
in `source-binding.json`; the CLI reports the tested `81a35ced2` source.

Evidence root: `/private/tmp/citlali-rtc-jump-guard-boundary-2026-09-12`.
The sealed 2,448-file `EVIDENCE_SHA256SUMS` has digest
`0f3380179d9121793b126b25a5671aa0a1fb2b7a422e049d0b6d12b46cdb156a`. All 2,412 previous
full-corpus, 196 onset-trial and 2,281 reassessment evidence files were
reverified unchanged. Historical preparation remains sealed and is superseded
only in its pending-owner status by `approved-preflight.json`. Effective
scientific/engineering/governance authority and file digests are recorded in
`effective-authority.json`. No contract was reopened or inferred from legacy
behavior. Source/evidence review `review-01.md`, by
`/root/rtc_plateau_boundary_exact_review`, passes on all three governance axes
with recorded limitations and no findings; its SHA-256 is
`f285a395a122fbceb85ad2bb26819b49b5c385ec0d7cb87f11ff9fffec5c9d62`.

This completes the approved implementation and its conformance measurements,
ready for owner acceptance. Physical-event identity, full timing uncertainty,
harmfulness/protection admission, real native-to-existing-scan relations and
Apply authority remain outside this support increment. Spectral context and
source membership remain unavailable in this corpus; sigma_delta remains the
settled empirical comparison tolerance, not a fitted-parameter standard error.
The separate two-document closure must have the exact tested source as its
direct parent; its SHA/tree and fresh independent exact-SHA review are bound externally after
review. No rebuilding at documentation HEAD may replace the tested identity.
Preflight remote observations are historical; future canonical admission must
freshly verify live moving ancestry and independently review the exact
integration SHA. No integration, push, route activation, flags, correction,
filtering, spectra, MAP/CAL/PTC/AST, Unity access, new data, production or cleanup
occurs. All GitHub pushes remain owner-controlled.


## RTC disturbance-burden census approved preflight — 2026-09-13

Work order **RTC-DISTURBANCE-BURDEN-CENSUS-001**, owned by the project owner,
is approved for the native-time census proposed in the design review. Risk
Tier 2: inert persisted evidence accounting; no runtime scientific change.
Engineering governance, Timestream Successor governance and Review And
Conformance are effective through accepted `06a3ade51c1b3f38887295433d913811bf25cd14`
and incorporation `77507836325eff9f469062d5884481ea37599594`; their digests remain
those recorded in the integration ledger. SCI-RTC r0.12 REQ-094–099, 102, 106,
114, 118–125 and 138 govern support distinctions, evidence origin, source
protection, phase separation and downstream ownership. D2/VAL and all frozen
contracts remain preserved. Architecture and scientific conventions retain
their existing authority.

Literal implementation base is reviewed documentation closure
`8fff882dd0e8891512b01e6065cdb442981b1e5d`, tree
`8b3167e4d477fbb7aa515339306a432257cd2c62`, directly after tested source
`81a35ced25d45eeccafabc3243b5aa3010893625`. Live canonical, read by explicit
SSH GitHub URL on 2026-09-13, is `86c20b31f7300ba4063be044380b61cd0baf25eb`;
live former RTC branch is `d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`. This
work is not canonical admission. The old temporary checkout is incomplete
and lacks its .git pointer; its files and registered ref are left untouched.
The unrelated dirty citlali-refactor checkout is likewise preserved.

The one clean owned worktree is
`/private/tmp/citlali-rtc-disturbance-burden-census-001`, branch
`codex/rtc-disturbance-burden-census-001`, initially empty staged/unstaged/
untracked state. This bounded evidence tool does not reopen the completed RTC
module or occupy a runtime spine slot. Expected changes are one local Python
accounting tool, its analytical tests, and these existing handoff/status records.
The tool consumes saved Learn evidence and final Consider dispositions; it
produces engineering decision evidence, not a runtime Consider or Apply plan.

The fixed input cohort is 143 eligible network files, 13 observations and 155
deferred inventory entries. Initial VAL and producer Tune-valid finite-pair
support define the fixed detector-time denominator before RTC disturbance
exclusions. Saved noise-screening-required support is reported separately;
peer-only APT cuts and review-only health concerns do not remove denominator
support. Failed fits and unresolved bounds never shrink it. Pairs count once.
All durations use interval unions. Physical transition bounds, bounded recovery
support, original fitting guards, additional hypothetical guards and unresolved
extents remain distinct. Final retained coordinates must reconcile to 11,150.
No producer group becomes a physical-event identity.

Timing is reconstructed from the recorded producer clock and integration-center
support, with the already provisional uniform averaging assumption unchanged.
Accounting uses integer microseconds relative to the first native center;
rounding error is measured, no more than half a microsecond per endpoint, and
checked against saved bounds. This avoids artificial nanosecond clock-rounding
holes without hiding acquisition-run breaks. Deferred exposure is unavailable
unless its missing binding can actually be supplied. Verified metadata only
may name program intent/array; missing labels remain unavailable. Cross-network
concurrency, actual full-scan rejection, longest/all segments per scan and
releveling-opportunity exposure remain unavailable without authoritative
native-to-PCA-scan and clock relations. Acquisition ScanNum is never a proxy.
Existing full-detector/full-intersected-PCA-scan level-0 policy remains settled.

Approved sensitivity assumptions are minimum native interval durations of
1, 5, 10 s and additional boundary exclusions of 0, 50, 100 ms. These show
native duration qualification only, not scientific adequacy or runtime defaults.
Localized recovered transients are exclusions, not new level-shift identities.
Candidate-inclusive edge/guard footprints are sensitivities, not prevalence
bounds or a completeness correction. Source protection and spectral context
remain unavailable; no harmfulness threshold is introduced.

Focused gates cover overlap/union, pair coherence, gaps, invalidity, differing
cadence, missing bounds, phase selection and analytic scenario costs; corpus
hierarchical sums and preserved population reconciliation are required.
Determinism, timings, memory, source/input/output digests and independent fresh
read-only exact-SHA review are required. The prior 1,004 CTests, 33 focused RTC
tests, four config modes and 207 baseline tests remain inherited evidence for
unchanged C++; local Python accounting tests are the fresh gate. A new build,
Unity/Spack campaign or reduction is not required for this non-runtime tool.
New authority needs, absent input bindings, unexpected support, schema, memory
or numerical outcomes trigger recorded reassessment. Keep bounded or stop;
do not broaden into scan reconstruction or detector development.

No integration, push, cleanup, route activation, flags, replacement, correction,
filtering, spectra, maps, production, new data or subsequent runtime capability
is authorized. Owner performs all GitHub pushes. Completion records exact
source and evidence plus the three independent review dispositions; canonical
admission would separately require fresh moving-base and exact integration-SHA
review. The same original evidence remains immutable.


Census review reassessment: exact preliminary source
`b4872da9d75a1cbc429a0b65bf91f8599aa87135` passed 19 analytical tests and
completed the 143-file census twice. Its 152 deterministic artifacts matched
byte-for-byte. Independent review found two original producer groups with
`refinement_limited=true` that were absent from group-level unresolved counts
because their coordinate recovery bounds were available. Repair preserves the
producer refinement-limit cause and unions it with coordinate-level unresolved
states; it keeps the measured support and all scientific decisions unchanged.
Unresolved group count changes 75,743 to 75,745; both detector streams already
had unresolved evidence, so stream/exposure totals do not change. A regression
fixture covers this distinction. The new persisted reporting field is scoped
to this inert census and introduces no runtime schema or authority.

The same bounded reporting completion adds coordinate-local amplitude and
transition-duration distribution summaries/figure plus concentration and
verified program coverage summaries. Reassessed recoveries without exported
new support remain explicitly unavailable (703 coordinate records in the
preliminary corpus); no extra solver or event-analysis pass is needed to
report the limit. The final source must receive a new full-corpus accounting
run and independent exact-SHA review. Original source/evidence remains sealed.


## RTC disturbance-burden census completion — 2026-09-13

Disposition: implementation/evidence candidate complete for owner acceptance;
no canonical integration, production admission or Apply treatment selection.
Tested source `5973de4eb696ad211be4eceeeabcc20e25435b6f`, tree
`ccdd4a8d1edca3c6c2b64c46d01ec392fe8a07f8`, direct parent
`b4872da9d75a1cbc429a0b65bf91f8599aa87135`, literal approved base
`8fff882dd0e8891512b01e6065cdb442981b1e5d`. Scope is the single accounting
Python tool, 21 analytical tests and existing handoff/status records. The new
branch/worktree and effective authority are the approved preflight above;
prior partial temporary worktrees, unrelated dirty checkout and all preserved
refs remain untouched. No second runtime probe, spine or integration branch
is created. The engineering evidence consumer does not satisfy or change any
runtime Learn/Consider/Apply responsibility.

The original source `81a35ced25d45eeccafabc3243b5aa3010893625` and its 2,457-file
closure manifest are preserved and verified. Final accounting binds original
raw/Tune/APT input SHA-256 values, program-only telescope metadata and every
native timing cell. Original candidate and final coordinate populations
reconcile exactly. The 21 tests cover paired overlap, acquisition gaps, initial
invalidity, unequal cadence, hypothetical scan boundaries, missing bounds,
final bound selection, original group uncertainty, guards, qualification and
byte-stable serialization. Full corpus checks conserve integer-microsecond
exposure across detector, observation, network, array, program and concurrency
partitions. Maximum endpoint rounding is 0.0078125 microseconds; the inherited
provisional uniform-averaging and absent absolute/cross-network clock authority
remain explicit.

| Final accounting quantity | Value |
|---|---:|
| Assessed files / observations | 143 / 13 |
| Initially eligible detector streams | 70,024 |
| Initial producer/VAL paired exposure | 17,216,191,725,568 us |
| Initial excluded exposure | 387,690,962,944 us |
| Union of available direct support | 22,075,703,296 us / 0.1282264% |
| Union of retained transition support | 617,742,336 us / 0.0035881% |
| Union of bounded finite recoveries | 21,503,729,664 us / 0.1249041% |
| Original fitting exclusions | 53,427,945,472 us / 0.3103354% |
| Direct plus recorded candidate-edge sensitivity | 28,116,566,016 us / 0.1633147% |
| Existing noise-screening-required support | 7,489,978,368 us / 0.0435054% |
| Final retained coordinate measurements / original groups | 11,150 / 8,389 |
| Unresolved original groups / detector streams | 75,745 / 7,020 |
| Exposure of streams containing unresolved evidence | 2,569,939,558,400 us / 14.92746% |
| Reassessed recovered coordinates lacking new extent export | 703 |
| Deferred inventory entries / exposure | 155 / unavailable |

These support rows overlap and must not be summed. Unresolved-stream exposure
is a population denominator, not unknown-event duration or a prevalence bound.
Final bounds comprise 9,678 unchanged initial and 1,472 remeasured coordinates;
this phase distinction is separate from prior own-initial/partner-recovery
categories. There are 170,088 bounded unreassessed recovery coordinates.
Original refinement limits remain group-level unresolved causes, including
when local coordinate support is measurable. One-coordinate bounds do not
resolve unavailable partner extents. Failures never become zero-duration or
clean-data assertions; rejected jump checks do not classify spikes.

The descriptive concentration result is 74.77575% of direct support in the top
5% of eligible detector streams. Scope-verified producer metadata covers three
OOF, eight pointing and two science observations, all labeled Lissajous. The
result does not establish coverage of other programs/conditions or operation-
wide prevalence. x/r amplitudes, empirical sigma_delta and physical transition
durations remain coordinate-local; sigma_delta is not fitted-offset uncertainty.
Available-bound native duration qualification ranges from approximately 0.12825%
to 0.21254% exclusion across approved min-duration 1/5/10 s and additional
transition guard 0/50/100 ms assumptions. These are not scientifically qualified
plateaus, selected runtime defaults or actual scan-treatment costs. Transient
holes do not become level-shift boundaries. All-segment retention cannot be less
than longest-segment retention under identical declared fixture definitions.

Actual full-detector/full-PCA-scan loss, longest/all segments per scan and
releveling opportunity remain unavailable. The exact native-cell/PCA-scan
relation is the next prerequisite for comparing those costs; acquisition ScanNum
and network files are not substitutes. Source protection, spectra, cross-network
concurrency, physical-event identity/completeness and downstream scientific
adequacy retain their existing limits. No new harmfulness or small-amplitude
threshold, donor/replacement scheme, releveling policy or correction is selected.
The owner-settled full-affected-detector/full-intersected-existing-scan level-0
policy remains unchanged. No native-PSD use profile was activated.

Fresh source-bound run: `/private/tmp/citlali-rtc-disturbance-burden-2026-09-13/census-02`; manifest SHA-256 `e8156f63a96b21f045e571ff784d1c246994acd83f8e1cf0647e4c4ef33b95b4` (158 files).
Initial exact-source replay proof: `/private/tmp/citlali-rtc-disturbance-burden-2026-09-13/determinism-01.json`, 152 identical
artifacts. Final invariance proof: `/private/tmp/citlali-rtc-disturbance-burden-2026-09-13/invariant-final.json`, 149 identical
timing/interval/metadata/original-plot artifacts, 71,734 unchanged detector
support/duration records and 242,014 unchanged original event-field records;
only the two reviewed unresolved group counts and explicit new reporting fields
change. Distribution aggregation was independently replayed exactly. Four
figures were visually inspected. Source review by `/root/rtc_burden_exact_review`
is `/private/tmp/citlali-rtc-disturbance-burden-2026-09-13/review-source.md`, SHA-256 `659b7f01e859aee62f594b50a9610117b83576c2272666db9d7890d7c275924f`:
scientific/behavioral PASS with recorded limitations, architecture/ownership
PASS, repository/evidence PASS, no remaining findings.

Final local environment: macOS arm64, Python 3.13.2, NumPy 2.2.3, netCDF4 1.7.2,
Matplotlib 3.10.0. Full census wall time 67.1844426 s, peak process RSS
293,289,984 bytes, zero unexpected errors. This includes input hashing and
artifact generation, not another solver/event-analysis pass or future runtime
Apply cost. Prior unchanged C++/config/baseline gates are inherited evidence,
not newly run gates; representative Spack/Unity and affected-mode reductions
are not required for this inert tool and were not performed. New science,
route activation, maps, flags, correction, filtering, spectra, PTC/AST/CAL,
production, remote writes and cleanup are outside scope.

The documentation-only closure must directly follow tested source `5973de4eb`
and receive independent exact-SHA review. Its exact identity/tree/review are
recorded externally after commit; no run at documentation HEAD replaces the
tested identity. Owner acceptance and any owner-run push remain separate.
Future canonical admission must freshly verify live ancestry and review its
exact integration SHA. The next decision is the bounded scan-association
prerequisite, not an inference that low direct support makes an Apply treatment
safe or inexpensive at scan scale.


## RTC observation recurrence assessment approved preflight — 2026-09-13

Owner approval: “sounds good. Proceed” authorizes the proposed comparison of
repetition counts, rates, temporal spread and exposure costs using the same
sealed census, plus read-only investigation of the native-to-existing-PCA-scan
prerequisite. Work order **RTC-OBSERVATION-RECURRENCE-001**, Tier 2, is an inert
evidence-tool continuation on the same owned clean worktree and branch
`/private/tmp/citlali-rtc-disturbance-burden-census-001`,
`codex/rtc-disturbance-burden-census-001`. It occupies no runtime spine/module
slot and opens no second branch. Literal base is accepted census closure
`4eb5de86ce3f4fe071053e0c2381bf77fdc14e1c`, tree
`aaa784903ef594e280c9d90aaf796ec8e207c43b`; its exact tested source remains
`5973de4eb696ad211be4eceeeabcc20e25435b6f`. Initial index/worktree/untracked
state is empty. Live GitHub canonical verified by explicit SSH URL is
`86c20b31f7300ba4063be044380b61cd0baf25eb`; former RTC remote remains
`d1ed875f7aa3ccbadbe783b571f635d7ba96b2e4`; this census branch is absent
remotely. There is no canonical admission or remote mutation.

AGENTS, toltec-context, current status, scientific conventions, architectural
invariants and all three effective governance documents were read. Governance
effectiveness/digests remain those above (accepted 06a3ade51, incorporation
775078363). SCI-RTC r0.12 REQ-094–099, 102, 106, 114, 118–125 and 138,
SCI-ALIGN half-open support requirements, and the preserved September 9/10
owner decisions govern identities, paired evidence, support, health review and
full-intersected-existing-scan treatment. Prior D2/VAL and RTC source are immutable.

The tool consumes the exact 158-file sealed census generation, retaining initial
producer/VAL exposure, original producer-group identities and separate retained
transition, recovery, unresolved, noise and health evidence. A group counts
once even when x and r both contain retained measurements. Original groups
are not physical-event identities. Rates use eligible detector-minutes; elapsed
temporal spread is separately described and acquisition gaps are not exposure.
Four equal elapsed-time divisions are descriptive bins, never processing scans;
a group's earliest retained-bound start places one marker, without regrouping.
Counts 2/3/5/10, rates 0.25/0.5/1/2 per minute and occupied divisions 2/3/4
are exploratory sensitivity coordinates, not scientific defaults or a chosen
exclusion policy. Whole-observation cost is unioned initially eligible exposure;
additional loss beyond direct measured support is not additional loss beyond
full-scan treatment. The latter remains unavailable without exact scan support.

The existing health rule (10x peer scale, >1% edges, >=80% of >=6 complete
10-second blocks) remains review-only. No observation-wide bad-detector policy
is selected. Any future exclusion has observation/named-use scope, not persistent
hardware identity or Tune/APT rewrite. This engineering assessment consumes
runtime Learn and Consider products; it produces no runtime Consider/Apply plan.

Expected edits: one Python recurrence consumer and analytical tests; existing
status/handoff only. New artifacts live outside the sealed census root. Fresh
gates: pair/group and identity reconciliation, eligible-time/gap/rate boundary
checks, analytic costs, monotonic threshold sensitivity, all-corpus accounting,
deterministic replay, timing/RSS, and independent fresh-context exact-SHA review
with all three dispositions. C++/config/Spack gates remain inherited because
no corresponding source changes; no build or Unity run is required. A docs-only
closure directly follows the tested source and receives its own exact review.

Read-only metadata/log investigation may recover a previously recorded relation,
but must not reverse-engineer missing alignment or scan rows into authority.
Missing native clock/run/config relations, unexpected schemas or counts, new
policy needs, changes to runtime/corpus/ownership trigger reassessment; preserve
and report missing bindings rather than inventing them. No map data, spectral
work, new observations, runtime flags/corrections, activation, integration, push
or cleanup. GitHub pushes remain owner-controlled.
