# POINT 152389 network rescue — owner-authorized follow-on

## Work order / preflight

Owner: Grant Wilson, approval “I agree. Let's do this” following the bounded
RTC boundary/VAL repair proposal. Tier 2. ENGINEERING_GOVERNANCE,
TIMESTREAM_SUCCESSOR_GOVERNANCE, and REVIEW_AND_CONFORMANCE read; the accepted
ledger bindings remain governing. This resumes the existing MAP spine slot and
branch `codex/map-successor-first-forward-001` at
`131ef891537292c43b7eb35ea3f2080fe26e32af`, worktree
`/private/tmp/citlali-map-successor-first-forward-001`. Tracked clean initially;
only the existing untracked build symlink. Original work-order/canonical base
identities remain in MAP_SUCCESSOR_FIRST_FORWARD_001_2026-09-21.md. This is an
explicit bounded prerequisite repair under that owner, not a second branch or
canonical integration. The shared dirty ALIGN checkout remains untouched.

Purpose: complete RTC/CAL/PTC for POINT networks 2 and 11 without converting
known outside-science jumps into a whole-network abort; rebuild MAP with any
newly completed eligible publications. Preserve original failures and maps.

Authority: owner-approved repair, SCI-RTC r0.12 identity/support/causal-influence
requirements, SCI-VAL 001/002/004/007/008 and existing named consumer uses,
rtc-jump-exclusion-2026-09-13-v1 (including three-group detector-observation
exclusion), existing RTC/PTC partial-completion and observed-entry bindings.
No new detection, donor, filtering, fit, APT, calibration or MAP coefficient
method. Context cannot be declared harmless merely because science overlap is
absent. Scope excludes nw3 ALS convergence repair, FRUIT, Unity, integration,
push, production activation, ref movement and cleanup.

Owners: existing processing-scan projection produces exact science/context and
outside-science native-row relations. RTC admits that typed relation, retains
per-event support facts, freezes paired exclusions and preserves full filter
influence. VAL retains the immutable RTC result and evaluates named consumer
uses under existing profiles. The application adapter parses/serializes and
sequences these owners; no Engine growth or scientific policy in CLI.

Expected paths: processing-scan and RTC jump/support headers; existing RTC
application serialization; focused jump/recovery tests; evidence tools only as
needed for reproducible real-product verification; this handoff and status.
Relations bind exact native parents/networks and half-open native rows. Science,
context, known outside science, and unavailable relation remain distinct.
Memory: sparse per-event/per-scan facts only; no new sample×detector plane.

Implementation detail to validate: the existing RTC event/filter path already
makes native outside-science support unavailable and splits finite-filter runs
at those boundaries. The repair must carry that exact relation into the jump
planner and prove its use through filtering. Tail transitions stay paired causal
exclusions/facts and still count toward the unchanged detector-wide rule; they
do not select a nonexistent science scan. Context overlap is recorded separately.
An absent or contradictory processing relation still cannot admit data.

Gates: direct science/context/outside/unknown jump tests, complete-domain relation
validation and header isolation; meaningful filter-boundary/VAL checks; focused
and full CTest, baseline and successor Python tests, required config preflight;
real nw2/nw11 RTC/CAL/PTC rerun plus an unaffected-network regression control;
actual availability/exclusion/context evidence; rebuilt MAP exact parent/operator
replay and source description; runtime memory/time and independent fresh-context
exact-SHA review. Local AppleClang/Homebrew is supplemental; no representative
Unity/Spack result is implied. Any new scientific ambiguity or different upstream
failure requires recorded reassessment before repair or claims of completion.

## Implemented candidate / local gates

The typed scan binding optionally carries a validated complete processing domain;
old callers lacking that authority retain their strict missing-relation behavior.
The binding checks science/context compatibility and exact nonoverlapping
science-plus-outside coverage of every represented native network. The projection
now supplies the complete relation. RTC stores each admitted coordinate's science
scans, context scans, and outside-science transition support. Tail transitions
remain paired exclusions on that support and count toward the unchanged
three-group rule. The event/filter boundary requires the same outside-domain
restrictions, preventing a caller from bypassing guards after resolving a tail.
`jump-exclusions.yaml` is digest-bound in the final receipt and exposes the exact
producer support decisions; existing VAL output facts retain the owning plan.

All 1,289 runnable CTests pass after rebuilding all test targets; one established
MapFitterLifecycle case remains disabled. This includes five added focused tests
covering known tail/context, half-open/straddling science boundaries, unchanged
whole-detector counting, invalid/unknown relations, and a large outside-science
injection that leaves every admitted filtered output bit-identical. The first
injection fixture failed because it supplied literal cadence rather than the
measured cadence required by the existing transfer interface; corrected fixture
uses the measured evidence. No algorithm was relaxed for that failure.
175 successor Python tests, 207 baseline Python tests, and required full config
preflight pass. Real-network recovery and final independent review remain pending.

## Independent review repair

Fresh-context review of `fa94788f7ef411c12c149d3e5a290a2693801229`
found one typed-boundary defect: event treatment compared merged unavailable
ranges but retained the original ordering, while downstream membership uses
binary search. Normalize the stored ranges before comparison and use. The
filter injection test now covers canonical, reversed, and overlapping equivalent
ranges at both science boundaries. All 1,289 runnable CTests pass again.
The initial actual run completed nw0/nw2/nw11, with all six PTC segments per
network converged; it remains preliminary evidence under `upstream`. Final
runtime and maps will bind the repaired committed source under `upstream-r2`.

## Final observation evidence

Runtime source `e93e8861be3b9d84049cc72d431fd72ab35d8959` on the
existing isolated MAP branch; clean rebuild version
`v2-ngc4449-memory-repair-187df04b-450-ge93e8861b`. Initial candidate
`fa94788f7ef411c12c149d3e5a290a2693801229` is superseded by the reviewed
interval canonicalization repair; its preliminary run is retained separately.

Durable root:
`~/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-network-rescue-20260922`.
Final runs: `upstream-r2` and `map-pass`; initial run: `upstream`.
`RESULTS.md` links the FITS maps and source preview. `FILES_SHA256SUMS`
indexes 39,983 files, SHA-256
`731c7c0ef9914d0238c31fa2fdc7cd64d5d32e2d5243d52170403d589a37b9d2`.
The original MAP run and corrected offset audit remain immutable references.

| Network | RTC output pairs | CAL / PTC retained samples | Converged segments | Whole-detector jump exclusions |
|---|---:|---:|---:|---:|
| nw0 control | 4,396,784 | 3,856,977 | 6 / 6 | 0 |
| nw2 | 3,343,413 | 2,936,552 | 6 / 6 | 0 |
| nw11 | 1,531,932 | 1,379,512 | 6 / 6 | 15 |

All 7,616 original nw0 numerical binaries are bit-identical, including RTC,
CAL and PTC. Original event evidence and native time remain byte-identical for
all three networks. Admitted jump counts remain 0, 19, 74 respectively.
nw2 preserves both tail groups, with channel 405 overlapping scan 5 context;
nw11 preserves all 15 tail groups. The 15 whole-detector exclusions in nw11
follow the existing >=3-group rule, including science and outside events.

`evidence/upstream-r2-verification.json` checks all 1,626 detectors' emitted
finite-filter footprints against admitted runs and outside-science boundaries,
all outside inputs unavailable, every tail jump transition retaining its paired
transient cause, unchanged jump counts and whole-detector policy, exact event/time
identity, and CAL membership after mapping each scheduled slot to native row
using its retained factor and phase. No unavailable tail sample becomes a
science contribution. The control also verifies values and masks bit for bit.

MAP combines original successful networks 0,1,3,4,5,7,8,9,12 with recovered
2 and 11. All three exact PTC numerator/operator replays, checksums, support
arithmetic, WCS representation checks and source inspections pass; both saved
parent identity mutation checks reject the inconsistent metadata. All a1400
numerical planes are bit-identical. Source image was visually inspected.

| Array | Contributions | J2000 east / north centroid (arcsec) | Shift from previous (arcsec) | Measured FWHM major / minor (arcsec) |
|---|---:|---:|---:|---:|
| a1100 | 16,389,673 | +4.668 / -5.171 | 0.272 | 6.057 / 5.093 |
| a1400 | 6,719,154 | +4.378 / -5.405 | 0.000 | 7.165 / 6.282 |
| a2000 | 2,781,761 | +4.697 / -5.204 | 0.410 | 9.434 / 8.615 |

Nominal calibration beams remain 5, 7.5 and 10 arcsec. The selected APT,
boresight, CAL, filters, ALS rank 10, uniform gamma=1, J2000, 2 arcsec pixels,
coverage cut 0.5 and no spatial mask remain unchanged. Fits describe the source;
they do not establish uncertainty or full-chain source response.

Measured single-worker upstream: 80.167 s, sampled peak RSS 1,445,904,384 bytes.
MAP: 66.575 s, sampled peak RSS 1,744,420,864 bytes. Both exit 0 with zero
unexpected error-level messages, no concurrent computational diagnostics during
timing, and complete executable/thread/sampling records. This is supplemental
local AppleClang/Homebrew evidence, not Unity/Spack production qualification.
The existing unresolved RTC/CAL scientific qualification remains unchanged.

nw3 scan 5 ALS failure remains separately unavailable; raw nw6/nw10 remain
absent. No canonical integration, push, production activation or cleanup was
performed. Final independent exact-evidence-candidate review follows; runtime
code identity stays explicitly bound to e93e8861 rather than a later docs SHA.
