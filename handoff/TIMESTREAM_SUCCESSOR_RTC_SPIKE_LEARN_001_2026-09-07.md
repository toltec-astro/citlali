# TIMESTREAM-SUCCESSOR-RTC-SPIKE-LEARN-001

Status: bounded implementation locally verified; exact-SHA review and new
owner-run Unity gate pending; no integration or activation.

## Work order / preflight

- Purpose: implement the next bounded runtime Learn responsibility in S2 of
  the existing Timestream Successor implementation order: paired native
  difference/noise and spike-candidate evidence before level-shift learning.
- Owner: Grant Wilson. The owner approved this component, selected its
  numerical choices one at a time, authorized a provisional readout model,
  and directed: "great. Let's proceed".
- Risk tier: 2. One scientific-module implementation slot; no second spine
  or module branch is opened.
- Applicable governance read: `ENGINEERING_GOVERNANCE.md`,
  `TIMESTREAM_SUCCESSOR_GOVERNANCE.md`, `REVIEW_AND_CONFORMANCE.md` under
  `doc/governance/`. Their accepted SHA-256 values are respectively
  `70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76`,
  `29fae6f789bb6133c1f5bcdaf0f15437f2eb8c4110f338d3a8de9a4d98ba88dc`,
  `691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1`.
  Accepted commit `06a3ade51c1b3f38887295433d913811bf25cd14` and effectiveness
  `77507836325eff9f469062d5884481ea37599594` remain on canonical ancestry.
- Current sequencing: `doc/REFACTOR_STATUS.md`, `doc/INTEGRATION_LEDGER.md`.
  D2 and VAL native-target units are accepted and closed. The historical
  authority JSON and S2 grouping are retained without rewriting their history.
- Exact canonical base: `8eccf8990027b8f65febb5f20d9a415829152322`, tree
  `4e2afa6b4e05a71d5e46e71d23548f57f7068f99`, parent
  `74a2727539a5d1812892042a59f4ca9b14ff9c2e`. Explicit SSH GitHub
  `refs/heads/codex/refactor-mainline` was verified at that SHA before work.
- Branch: `codex/timestream-successor-rtc-spike-learn-001`.
- Worktree: `/private/tmp/citlali-timestream-successor-rtc-spike-learn-001`.
  Initial staged/unstaged/untracked state: empty. The unrelated dirty
  `/Users/gwilson/GitHub/citlali-refactor` checkout was inventoried and preserved.
- Scientific authority: frozen SCI-RTC v0.1/r0.12, particularly REQ-013/017,
  025/026, 043--047, 055--058, 095/097 and 115--125; SCI-VAL REQ-001--005,
  023--025 and 027--030; the owner decisions recorded below. These are a
  bounded owner disposition of parts of OWNER-003/006/009/029, not closure
  of every remaining despike/donor/level-shift policy.
- Architectural authority: ADRs 0017--0023 and the accepted Paired-D1,
  identity Learn/Consider/Apply, D2, and exact native VAL target interfaces.
- Runtime boundary: Learn owns numerical evidence and compact original-input
  locators; Consider consumes immutable evidence to resolve the selected
  screening-failure consequence and preserve unresolved candidates. Later
  complete RTC plans consume those decisions before Apply. This increment
  does not implement replacement, level-shift resolution, or pretend that an
  identity Apply has executed spike treatment.
- Engineering workflow: recover authority, implement, test conformance,
  independently review an exact SHA, then present for owner acceptance.
  That workflow is distinct from the runtime phases above.
- Expected paths: RTC-owned spike-learning and optical-reference interfaces,
  focused tests/header isolation/CMake registration; necessary owner-decision,
  status, ledger, and conformance records. Existing D2/VAL/native/identity
  implementations and frozen scientific package bytes are preserved.
- Memory: reference heavy immutable parents and snapshots; retain compact
  block summaries and sparse coordinate-specific edge candidates. Reuse
  bounded per-block scratch. Do not allocate copied x/r, full difference,
  full noise-scale, or generic validity planes.
- Gates: focused numerical, boundary, gap, source protection, exact identity,
  stale-snapshot, nonfinite, immutability and partition-invariance tests;
  header isolation; existing successor regressions; local CLI/safety/full
  CTest, config preflight, baseline tools and ledgers. Local AppleClang /
  Homebrew / cached-source evidence is supplemental, with dependency dirt
  disclosed. A new owner-run exact-source Unity GCC13/Spack gate is required
  before admission. Prior D2/VAL jobs do not cover new source.
- Fixture: 152390 remains a timestream fixture. Synthetic/injected events
  test the implementation; no map-product comparison or separate validation
  program is introduced. Timing/response assumptions remain visible.
- Review triggers/stop conditions: absent scientific predicate, inappropriate
  cause promotion, source protection or AST authority loss, new cross-stage
  reach-through, unexpected resource use, changing canonical base, or scope
  expansion. Reassess explicitly; never borrow a numerical default from the
  legacy despiker as scientific authority.
- Integration/push/activation/cleanup: not authorized. User performs all
  GitHub pushes. Reverify live canonical before any proposed admission;
  preserve the implementation literally, reconcile moving ancestry explicitly,
  and independently review the exact admission SHA as well as the source SHA.

## Owner decisions and precise implementation binding

1. The statistic is adjacent-sample differences of original native x and r,
   independently centered by their median and scaled by MAD. No x/r amplitude
   mixing or inferred physical-origin attribution is authorized.
2. Noise-estimation blocks last 10 seconds, anchored to a physical native
   acquisition run. Computational partitions do not redefine blocks. A
   difference belongs to the block of its later endpoint and retains both
   endpoint identities. It may cross an estimation-block boundary, never a
   native gap. Half-open block intervals and exact binary64 comparisons are
   used. A partial computational view cannot masquerade as a complete run.
3. Candidate threshold is **inclusive** `abs(d - median) >= 5 * scale`, with
   `scale = 1.4826 * MAD`. The owner's later `>= 5s` wording, understood and
   discussed as `>= 5 sigma`, supersedes the earlier strict comparison.
   This is an empirical difference score, not a calibrated tail probability.
4. At least 256 admitted adjacent differences are required for a block.
   Insufficient population, zero/undefined scale and admitted nonfinite or
   arithmetic failure produce typed unavailable evidence. No zero filling,
   re-admission of excluded inputs, sigma floor, or scan-wide fallback is
   authorized. Declared-invalid inputs are excluded before reading payloads.
   Population is frozen from original input; candidate discovery does not
   iteratively change that population.
5. Owner: "if we can't make a noise estimate to spike find, we have no
   business keeping those samples for mapmaking." Failure in either x or r
   requires paired exclusion for the affected detector's block, with each
   coordinate's reason retained. Owner expects this to be exceptional. This
   is a named-use consequence, not a claim of a detected spike or universal
   raw invalidity. Other successfully screened blocks/detectors remain local.
6. The additional optical-response check applies **only in source-protected
   regions**. There a >=5-sigma candidate requires demonstrated incompatibility
   with the fastest permitted sampled optical response before spike treatment.
   Outside, ordinary candidate/event checks apply. A persistent jump remains
   a possible level shift. An edge or count of above-threshold samples does
   not establish event duration or rule out an optical crossing.
7. Reuse the approved array Airy beam and AST/ALIGN motion authorities in the
   optical comparison. The example 5 arcsec/500 arcsec-per-second is not a new
   universal instrument limit. The shortest accepted Airy FWHM is about
   4.679 arcsec. Required source membership is an explicit, exact-parent input;
   unknown protection is never silently outside-source. Protected samples are
   not automatically omitted from the robust difference population: the owner
   selected source-aware event assessment instead of blanket source exclusion.
8. Owner could not supply the readout weighting and authorized an assumption.
   `rtc-native-readout-uniform-average-assumption-v1` models each sample as
   a unit-normalized uniform average over its recorded integration interval,
   with the accepted midpoint time. No additional electronics filter is
   modeled. The fastest detector limit is instantaneous; the stated physical
   expectation remains a time constant below 1 ms. Optical crossing phase is
   free relative to the sample boundaries. Classifications depending on this
   model are conditional; the model identity accompanies evidence/decisions.
   Correct hardware information triggers a new model version and replay from
   unchanged original data, not retroactive edits to prior evidence.
9. The historical 1--3% affected fraction is context, not a clipping quota,
   prior probability, acceptance target, or authorization for forced masking.

## Increment boundary and remaining scientific decisions

This component publishes edge candidates, noise evidence and optical-response
reference evidence for later event assessment. It does not equate a large
difference with an isolated spike, or a mismatch to one optical template with
proof against every allowed optical explanation. A numerical optical
incompatibility acceptance metric and its uncertainty/tolerance, event
grouping/duration, source-region geometry for an operational route, and
donor/level-shift policies have not been selected by the owner conversation.
The implementation must retain the corresponding candidate/required-review
state rather than invent those decisions. This limit does not prevent the
selected numerical Learn responsibility or the explicit screening-failure
decision from being implemented and tested now.

## Executable responsibility and contract binding

`timestream_rtc_spike_learn.h` implements the selected original-pair numerical
Learn responsibility. Immutable evidence references its exact original native
view, source-protection binding, initial VAL snapshot and nonzero RTC attempt.
The parent occurrence axis must cover its complete declared timing authority;
a full computational view of a clipped occurrence axis is rejected. Upstream
still owns the truth of the declared acquisition support. Candidates retain
both endpoint rows, detector/network identity, coordinate, signed difference,
centered difference, score and source-protection state. Block summaries retain
population counts, estimator values, time anchoring and typed failures.

The bounded `RtcSpikeLearningDecision` consumes that exact evidence/snapshot.
It requires the owner-selected paired exclusion when either coordinate's
noise estimation fails; it does not claim that samples have already been
excluded. Candidate dispositions require ordinary event assessment, protected
optical assessment, or resolution of missing protection. Successful noise
estimation is not a general eligibility decision. A future complete RTC plan
must consume these constraints with the remaining event/shift/donor decisions;
only then can Apply execute fixed actions. No Apply treatment is delivered here.

The explicit optional VAL export publishes only coordinate-local unavailable
noise-estimation facts on caller-supplied exact **original-input** native
realizations. Within this versioned RTC evidence product, fact code 1 means
noise estimation unavailable, state 1 means unavailable, and the cause is the
documented `RtcSpikeNoiseCause` bitset. The caller allocates unique RTC product
attempt identities; author identity and numerical subject identity are separate.
No candidate fact is promoted to a hard spike or generic invalid flag. Dense
per-cell export is caller-requested; normal retention remains compact by block.

This original-input learner accepts initial VAL snapshots only. It cannot
silently ignore previously committed producer facts or invent a profile for
them. Supporting later snapshots requires an explicit binding to the approved
operation-specific validity rules; it does not authorize a new scientific
policy or reopen those rules. The accepted VAL carrier remains unchanged.
Native admission already rejects a nonfinite payload falsely declared finite;
RTC defensively checks admitted values and handles finite-input arithmetic
overflow. Declared-invalid payloads are excluded before arithmetic.

`timestream_rtc_spike_optical_reference.h` consumes protected-candidate evidence,
the exact existing ALIGN-mapped AST motion handle and the original detector
association record for an explicitly supplied array. It exposes the retained
WP7 Airy beam scale and **local reference** motion scale, plus the analytic
uniform-average sampling of a harmonic with arbitrary phase over each recorded
integration interval. The versioned readout assumption is always visible.
Missing AST support stays unavailable. A local measured speed is not promoted
to an event-wide upper bound; FWHM is not confused with strict optical frequency
support. No numerical optical-impossibility predicate or template-fit acceptance
tolerance is introduced. Those missing predicates block hard protected-event
classification, not this original candidate learner or sampling reference.

## Completion / conformance

- Added two isolated RTC public interfaces, two header-isolation translation
  units, 17 focused tests and one isolated CMake test target. No existing
  D2/VAL/native/identity implementation or frozen scientific package is edited.
- Local CLI and safety builds pass. Full CTest passes 916/916 runnable tests
  (917 registered; the established `MapFitterLifecycle.ExactProductSequence`
  remains disabled). Config preflight passes all four modes. Baseline tools
  pass 207/207; validation and science-change ledgers report 60 valid records
  and 3 valid changes / 5 integrations respectively.
- New tests cover inclusive thresholds, coordinate separation, linear drift,
  fluctuating background with injected events above the historical 3% fraction,
  source protection, gaps, block-crossing edges, partition invariance, incomplete
  physical context, minimum support, zero MAD, invalid-before-arithmetic,
  ingress nonfinite rejection, arithmetic overflow, exact original VAL subject
  publication, stale snapshots, conditional averaging/phase and AST identity.
- Synthetic storage/time witness: 16,384 rows x 16 detectors, about 0.00293 s
  for Learn on the local Mac, 28,672 logical owned evidence bytes and at most
  1,221 scratch differences (9,768 logical bytes). This is a bounded synthetic
  measurement, not a full-observation benchmark or accuracy qualification.
- Real 152390 x/r execution and observational spike-classification qualification
  are not claimed. 152390 remains the timestream fixture for subsequent
  conformance measurements; no map products are used. No ordinary application
  mode changes, so there is no claimed change to existing mode outputs.
- Actual local environment is AppleClang 21 / arm64, CMake 4.3, Homebrew and
  existing cached sources. The three preexisting dirty kidscpp-associated
  headers and their digests match the earlier disclosed environment. This is
  supplemental evidence, not the authoritative GCC13/Spack gate.
- Development failures are retained externally: the first standalone header
  accessor error, premature PCH/CLI-dependent checks, initial make target
  registration, and incorrect synthetic MAD/finiteness fixtures. Fixtures were
  corrected to the native contract; numerical policy was not weakened. Final
  gates have no unexpected error-level messages.
- External evidence root:
  `/private/tmp/citlali-rtc-spike-learn-evidence-2026-09-07`. Final commit/tree,
  clean-source binding, complete manifest, independent three-axis exact-SHA
  review and owner Unity transport bind externally after commit, avoiding a
  self-referential source hash. No prior result is relabeled as a run here.
- Source acceptance/admission remain pending that review, the new owner-run
  Unity gate and owner acceptance. Reverify live canonical and independently
  review any later admission SHA. No push, canonical movement, route activation,
  sample treatment, filtering/downsampling, subsequent module, production or
  cleanup is performed.
