# POINT 152430: repaired route and completed contamination comparison

The two-stage comparison completed: unchanged B and all four authorized Q roles
reached T5 and their POINT endpoints, with **no failed launch or retry**. Removing
one frozen detector/physical run gives a localized residual improvement, small
changes in global residuals and POINT measurements, and mixed broad-source
response. **Keep B as the development reference.** Q remains experimental
evidence; this observation does not establish a general veto or justify promotion.

## What was repaired and reproduced

The prior failure at `da94a43795cde31c80b0f642ddde258d65905d3a` remains a failed
campaign. Native FRUIT recognized the four-network route under
`rtc-step-consequences-v1` but not `rtc-retained-residuals-v1`, sending the new
request through another route's a2000 guard. Commit
`ff92aa69d73aee9441e3269adf4ff4b9892ae509` unifies that scope interpretation across
array, prospective, cycle and realized gates. Identity, network, cycle and binding
checks remain enforced. A validation-only CLI submits the actual native request
and returns before numerical processing. All five cycles, foreign identifiers,
invalid arrays/networks, unsupported cycles and previous routes were exercised.
No Stage 1 scientific settings changed. See [Stage 1](STAGE1.md) and its bound
request/test receipts.

B ran at `9d28b2c6e134375b6f377ae59acc4e9f967ac3d4`; all **25,584 native
fingerprints, 60 map/candidate planes, 37 discrete/frozen records, and the endpoint**
agree exactly with the sealed reference. Maximum plane difference is zero.
Original input, tune, APT, calibration, reference geometry and historical source
controls were checked before launch. Historical B controls therefore remain the
comparator, with no new B injection runs or relaxed tolerance. Q ran at
`a7351335f791c995e39d7cda4546412e3f4a5638`. Warm fully re-solved rank-10 ALS and
exactly five FRUIT applications were fixed throughout.

## Population audit and frozen selection

This is POINT 152430/0/2, **a1100 networks 1–4, a partial array**. All 2,069
physical detector runs were audited at CAL, M0 and T5 under the same rules.
There is one physical run per detector; six numerical scans do not create six
physical runs. Unavailable samples remain unavailable, never measured zeros.
The T5 native residual/operator reconstruction agrees with the saved map to
2.85e-13 mJy/beam, with exact support, HITS and normalization.

| Network | Detector runs | Retained at T5 in B | Unavailable |
|---|---:|---:|---:|
| 1 | 494 | 361 | 133 |
| 2 | 510 | 399 | 111 |
| 3 | 533 | 249 | 284 |
| 4 | 532 | 377 | 155 |
| Total | 2,069 | 1,386 | 683 |

The audit preserves all 32 earlier windows, pairs the 12 newer windows with T5,
and uses only four additional T5-selected windows (16 new windows total).
Large extrema and adjacent-difference MAD are descriptive diagnostics, not
uncertainty on a step contrast or evidence of consequential map bias by themselves.

[QUARANTINE.json](QUARANTINE.json), frozen before any Q run, selects only
**network 2, detector/channel 88, physical run 0, native interval [0,7821)**.
Its SHA256 is `a8edbfecdc297e06fd8625fbd1b435ff1b43bae60de852043cac0fcda995f1c5`.
The candidate is a finite, opposite-coordinate x/r excursion around 37.2–37.45 s.
The complete inspected 36.23–39.23 s window is originally paired-valid and
62.69–88.70 arcsec from the source; the event is at least 87.35 arcsec away.
During-minus-before x/r contrasts lie 29.15/49.45 cohort MAD from the same-network
background median. These are descriptive cohort scales, not detection significance.
Only 2/439 source-excluded peers have absolute waveform correlation at least 0.5
in both coordinates; their r amplitudes are 0.029/0.066 of the target, and displayed
adjacent readout correlations are below 0.07.

The frozen rules require complete-window validity/source separation, opposite
contrasts above 10 cohort MAD, fewer than 1% paired-coherent peers, adjacent
correlations below 0.3, and actual T5 science-pixel projection. They support one
bounded non-astronomical-disturbance hypothesis, not a proven cause or exhaustive
classifier. Network 1 D451 was excluded because 118/331 background peers share
its waveform; apparent isolation did not survive the full cohort check. Network 2
D309's full window approaches 0.77 arcsec and includes existing unavailable
support. Its plateau alone is off-source; the exclusion does not establish an
astronomical origin. No Q result was used to select or retune membership.

## What survives and where Q changes the result

The D88 inspected excerpt supplies 367 B MAP occurrences, 362 in science support,
with maximum direct selected contribution 144.84 mJy/beam. None enters the 40 arcsec
fit disk. This is not an isolated contaminant bias: other times in the whole run
contribute elsewhere, and whole-run quarantine changes PTC/FRUIT learning.

B's selected excerpt footprint contains 59 science pixels. Q retains 58; the lost
pixel at row 104, column 115 had B fit residual **248.28 mJy/beam** (HITS 522→462).
It is **unavailable in Q**, not zero or improved. On the **same 58 retained pixels**,
residual scatter falls **39.62→30.46 mJy/beam (23.1%)**, and robust scale
29.19→22.62. Actual Q−B scatter there is 23.05 mJy/beam and includes end-to-end
relearning. The all-59 B scatter is 50.72; comparing that with the 58-pixel Q
scatter would overstate improvement by hiding the removed worst pixel.

| T5 region | Explicit support cost | B→Q residual scatter (mJy/beam) |
|---|---|---:|
| Selected footprint | 59 B, 58 common, 1 unavailable in Q | 39.62→30.46 on common 58 |
| Source-excluded background | 7,923 B, 7,919 common, 4 unavailable in Q | 24.35→24.24 on common 7,919 |
| 40 arcsec fit disk | 1,257 common, none lost | 19.98→19.88 |

Background robust scale slightly worsens, 22.31→22.58 on the common domain.
The local improvement does not demonstrate broad map cleanliness or a reduction
in systematic POINT error. Full B support, missing Q pixels and normalization
changes are retained in [comparison maps](generated/B-Q-maps.png) and the packed
arrays, rather than restricting the report silently to Q's domain.

Q removes 6,844 originally available paired RTC samples (56.066 detector-seconds)
and 6,765 RTC map centres, distinct support counts. Final MAP contributions fall
9,805,999→9,799,155, a **0.069794%** loss. Science pixels fall **8,345→8,341**,
with four lost and none gained in Q REF, and normalization changes in 1,060 pixels.
Q has 1,385 retained detector runs and 684 unavailable. Only D88 loses native
paired support; no detector gains it. REF Learn events, decisions and support
classification records are unchanged. Existing native RTC rejection ownership
excludes paired/filter/downstream support, without releveling or donor recovery.

## POINT, uncertainty and source response

The fit exchanges axis labels between B and Q. Compare canonical major/minor
FWHM and major-axis angle modulo pi, not raw sigma-x/sigma-y. The B angle is
constrained at a bound while Q's is interior; B's conditional angle scale is not
a trustworthy boundary-aware confidence interval.

| T5 measurement | B | Q | Change / B conditional scale |
|---|---:|---:|---:|
| Peak (mJy/beam) | 429.451 | 430.581 | +0.074 |
| Centroid x (arcsec) | 4.14807 | 4.14038 | −0.067 |
| Centroid y (arcsec) | 7.25366 | 7.26883 | +0.219 |
| Major FWHM (arcsec) | 6.20277 | 6.29083 | +0.319 |
| Minor FWHM (arcsec) | 4.66378 | 4.64139 | −0.110 |
| Template amplitude (mJy/beam) | 471.051 | 473.867 | +0.161 |
| Largest aperture (mJy/beam arcsec²) | 15,276.8 | 15,796.2 | +0.244 |

Major-axis angle changes about −8.03 degrees. Fitted constant changes
−0.646→−1.425 mJy/beam (−0.360 conditional scale). Peak conditional uncertainty
falls **15.329→15.228 mJy/beam (0.66%)**. These use the existing own-reference
noise/operator package; they are conditional single-map scales, **not paired-test
significance**, and omit learned-model and policy-selection uncertainty. Remaining
network 2 detectors have a modest median T5 adjacent-difference MAD change
453.47→447.52 mJy/beam; other networks are essentially unchanged. That descriptive
noise statistic does not establish a step-contrast uncertainty.

Each Q perturbation uses the exact historical compact ±10% or broad +5% raster,
original r and native validity, its own normal active Learn–Consider–Apply path,
and Q REF geometry/support/noise. Fixed quarantine membership is identical in all
four roles. The finite-response gain and morphology mismatch at T5 are:

| Perturbation | B gain | Q gain | B mismatch | Q mismatch |
|---|---:|---:|---:|---:|
| Compact +10% | 0.990904 | 0.990961 | 0.082994 | 0.083182 |
| Compact −10% | 0.989508 | 0.989530 | 0.021492 | 0.021434 |
| Broad +5% | 0.848598 | 0.875547 | 0.240023 | 0.274994 |

Compact response is essentially unchanged. Broad gain moves closer to unity but
shape mismatch increases: **a mixed result**, not unqualified improvement.
Ideal-addition diagnostics are saved calculations, not extra reduction runs or
new uncertainty ensembles. See [response figure](generated/B-Q-response-and-measurements.png)
and `compact/report/response-Q/` for detailed response, fit and support records.

Active injections reproduce the historical decision-change pattern: positive
compact/broad introduce a network 1 D446 event (1,188 RTC samples / 1,170 map
centres lost), compact-minus changes one network 3 boundary decision without
support loss, and broad gains 94 network 2 D202 samples/centres through normal
active learning. These are separate from fixed D88 quarantine. Compact-plus has
21 gained science pixels despite unchanged normalization in those pixels: its
reference normalization moves 979→978, lowering the 0.5 threshold 489.5→489.
Those pixels each have normalization 489. This is a mask-threshold effect, not
recovered native exposure. Relative to matching historical B branches, all Q
roles lose four B science pixels; compact-plus additionally gains these 21.
`compact/report/branch-support-domains.json` preserves both sides explicitly.

## Execution, validation and preservation

| Role | Wall time (s) | Peak sampled process-tree RSS (GB, decimal) |
|---|---:|---:|
| B REF | 165.435 | 2.782 |
| Q REF | 162.878 | 3.105 |
| Q compact +10% | 162.644 | 2.924 |
| Q compact −10% | 162.709 | 2.900 |
| Q broad +5% | 177.065 | 2.580 |

Counts: **5 original parents, 5 trajectories, 25 attempted/completed feedback
applications, 5 endpoints, 0 failed launches, 0 replacements**. Twenty initial
network RTC Apply completions, 120 initial segment solves and 600 feedback segment
solves are internal work, not additional trajectories. All source admissions
completed, with zero unexpected error-level science messages. Stage 1 contract
calls and saved report calculations are not observational launches. Wall times
are single-run observations, not a statistically established speedup. Shared
conservative storage reservation peaked at 21,743,321,581 bytes, below the carried
30 GB limit including previous failed scratch. Execution remained sequential
with two network workers and one numerical thread/feedback worker.

Repair validation passed 1,315 CTests; the Q binding added one, for **1,316 passing
CTest tests**, with one pre-existing disabled MapFitterLifecycle test. POINT's
204 tests, baseline tools' 207 tests and full required config preflight passed;
exact-source native contract and quarantine checks passed. Logs distinguish an
initial sandbox-cache config failure and a mistaken test-directory invocation
from the successful corrected gates. Supplemental local AppleClang/Homebrew
build evidence includes the longstanding dirty KIDs dependency; no Unity build
or production certification is claimed.

[Reproduction](REPRODUCTION.md), compact manifests, the saved-report test receipt
and OS-denied report-isolation receipt provide the preservation checks. Stage 1
had an independent fresh-context reviewer. Exact Q launch review used a separate
existing reviewer context after the fresh-agent limit was reached; that limitation
is explicit in [Q_REVIEW.json](Q_REVIEW.json). Final independent review is recorded
separately in the closeout.

**No scratch was deleted.** The previous 2.27 GB proposal remains unapproved.
`NATIVE_SCRATCH_PROPOSAL.json.gz` enumerates the new campaign's potential cleanup
only and has `applied=false`, `owner_approval=false`; it is not a deletion program.
Preservation retains 2,336 compact files (113,373,454 bytes); all 7,347 protected
inputs were rehashed unchanged. The new proposal lists 127,920 native files
(13,771,984,323 bytes), all still present. Eleven saved-evidence tests pass, and
all ten report artifacts (eight figures) reproduce byte-for-byte under OS denial
of original/protected input and scratch reads. No push,
merge, production-default change, fixed11 revival or automatic follow-on occurred.

This is a completed bounded comparison, not an execution-blocked result. Its
in-sample selection on one partial-array observation, conditional uncertainties,
unavailable worst pixel and mixed broad response limit generalization. The local
benefit is worth preserving as evidence while **B remains the reference**.
