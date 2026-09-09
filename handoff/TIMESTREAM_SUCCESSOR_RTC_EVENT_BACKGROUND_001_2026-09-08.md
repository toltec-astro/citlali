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
