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
rather than assigning the event only to a candidate-center scan. It does not
select an endpoint-touch convention or a numerical transition-bounding method.
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
localization or bounding estimator, stable-side support/quality criterion,
timing inclusion/equality and uncertainty treatment to explicit selection.
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
