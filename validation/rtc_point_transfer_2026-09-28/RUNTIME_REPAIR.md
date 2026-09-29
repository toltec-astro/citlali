# Owner-approved runtime repair — 2026-09-29

Owner requests: “Please investigate, repair, and let me know the command to
relaunch.” Subsequent explicit decision: **“Approve both bounded changes”** to
(1) padding eligible 2–4 second RTC conditioned diagnostic stretches onto the
existing fixed 4-second grid without requiring one full 4-second stretch, and
(2) PTC iteration cap100→500 while retaining rank10 and convergence tolerance1e-5.
The question stated both are intentional policy changes, not implementation bug
fixes. The exact approval is recorded here before the revised observation runs.

## Preflight and authority

Base6eb15ca337604298074f574152a01a8a7f697647; clean at investigation start.
Same codex/rtc-point-transfer-20260928 branch and ptc-rank-response worktree.
One existing module probe; no additional branch or integration lane. Tier2
engineering/successor/review governance and exact accepted identities in PREFLIGHT
remain applicable. Scientific authority is unchanged except the two explicit owner
choices above. Partial-completion authority remains
`doc/SCI_RTC_PTC_PARTIAL_COMPLETION_OWNER_BINDING_2026-09-18.md`.
Original inputs, source/paired validity, physical runs, filters, selector, rank,
fit convergence criteria and five FRUIT applications remain under existing owners.
No Unity, production activation, push, merge, cleanup or automatic retry.

RUNTIME_DIAGNOSIS.json and independent RUNTIME_DIAGNOSIS_REVIEW.json bind the
failed evidence. Network4 channel155 has4146 conditioned review samples in25
stretches, longest481 versus FFT488; seven stretches meet minimum244. Eighty-three
other detector pairs have zero conditioned review support. Channel155 definitely
blocks current completion; the saved evidence does not guarantee other gates will
pass after repair. Network1 scan4 had260 fit detectors and failed after100 steps;
last relative decrease1.530095e-5 exceeds1e-5. Neither was a confirmed deviation
from existing implementation policy. Changing padding or stopping criteria silently
would have been incorrect; owner approval precedes implementation.

## Owned changes and limits

`RtcNativeSpectralEvidence` accepts an explicit typed conditioned grid policy.
Default/original Learn retains the exact old behavior. The opt-in keeps the fixed
FFT size, cadence, Hann normalization, both centering operations, minimum2s,
minimum2windows, physical-gap boundaries, and per-coordinate evidence admission.
It permits padding when all surviving stretches are short; it never compacts across
masks/gaps. Matched before/after evidence uses the identical policy and footprints;
original broad spectra remain unchanged. Policy identity is persisted in conditioned
outcomes and reassessment. No filter result or original sample is modified by Learn.
Existing empty-detector completion, donor restrictions and unavailable arithmetic
remain enforced. No missing outcome is passed as zero power.

Private CLI enables `point-transfer-runtime-2026-09-29-v1` only for the existing
152420/0/2 a1100 networks1–4 source-probe route. PTC receives cap500 at upstream
Learn; FRUIT validates the same profile and inherits the exact cap on every solve,
including warm initialization. Other scopes remain at100; missing/foreign profile
or mismatched caps fail. Unconverged results remain unavailable and fatal to this
POINT campaign. Rank10, tolerances1e-5/1e-10, means and solver algebra are unchanged.
Both B and Q use the revised runtime profile; Q alone adds the previously frozen
original-domain rejection rule. Comparisons must label this intentional new profile.

The existing saved-input PTC cost probe gains an optional iteration limit, default
100. A single checksummed replay of only the failed preserved segment at500 is
repair validation, not an observational reduction or a parameter sweep. Its result
is diagnostic and cannot replace a campaign role, parent receipt or fit state.

## Execution, storage, gates and review

Fresh root: successor-rtc-point-transfer-runtime-repair-20260929 under the existing
local citlali-validation directory. Both prior attempt roots remain untouched and
count in PRIOR. Owner-requested relaunch authorizes a replacement for the failed
B-REF; the old charge remains visible. New-root maximum remains8 parents/trajectories,
40feedback and8endpoints; total historical attempted parents/trajectories may be9.
No automatic further retry, second target, threshold sweep or extra role.
The failed science root measured2092905987bytes. Reduce completed-role admission
cap3.025GB→2.7GB to preserve that evidence within30GB; active reservation unchanged.
Capacity is recomputed before any numerical role; overruns stop without deletion.

Focused gates: native short-conditioned-window regression and retained old-default
controls; original/matched identity and support; native all-five-cycle profile gates
with foreign/missing/mismatched rejection; runner fixtures; single saved failed PTC
segment replay at approved500. Broader actual configure/build, CTest, baseline,
POINT, configrequire-all and preflight-only must pass at final exact source. Local
AppleClang/Homebrew cached realization is supplemental, not Unity Spack evidence.
Exact-SHA independent implementation review and external evidence receipt precede
READY. No new scientific trajectory runs during this repair turn. Owner launch
rechecks all gates; final scientific assessment follows only after owner resume.
Review must separately disposition science/behavior, architecture/ownership, and
repository/evidence hygiene. Further behavioral surprises require reassessment.

## Local implementation verification

Initial local build and1320 CTest cases passed; the added minimum-duration/window
count and both-policy nonfinite controls passed after their compilation. Native
FRUIT request validation passed9 tests and runner preparation passed11. The
prospective PTC invocation remains method/rank only; cap500 is owned by the exact
runtime profile and recorded in scientific_recipe.PTC.max_iterations, preserving
the strict native request boundary. Final exact-source preflight, the one saved
segment replay, and independent review are published in the new root's
preflight-only/<source-SHA>/ and PREFLIGHT_REVIEW.json before READY. These external
receipts bind the final commit and binary without self-referential source hashes.
The historical RUNTIME_DIAGNOSIS.json retains its pre-approval proposal state;
the explicit subsequent approval above supersedes that pending decision.
