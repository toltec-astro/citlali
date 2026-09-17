# RTC matched-support treatment outcomes — 2026-09-17

Owner-directed runtime increment after accepted common-mode closure
`e9ff1fb0ce7acd34d0093d9b90e37244eb43ad00`. The detector-response science is
separate: [verified handoff](DETECTOR_RESPONSE_INVESTIGATION_2026-09-17.md).
Tune dependence is an untested hypothesis and not an RTC prerequisite.

Runtime candidate: `cae0ebdc0de91e1d952593e2a32526e7a74a0602`;
tree `4d96b9d639801607bbc9be2a4e932232c6166517`.
Existing branch `codex/timestream-successor-rtc-notch-recovery-001`, worktree
`/private/tmp/citlali-timestream-successor-rtc-notch-recovery-001`; initially
clean. Read-only live canonical `0f52e1a421416eadaf7d8d99d64f537132395c0c`
and topic `becccce85816379e578ac1a325e8c9fe156defbe` were verified. No ref
reconciliation, push, production activation or cleanup is included.

## Runtime responsibility

`RtcTreatmentOutcomeEvidence::learn` owns the comparison of original and
conditioned native spectra on intersected eligible support, using the existing
D2 numerical owner and identical realized windows. Its immutable parents retain
exact input, stage, attempt, VAL, support, source and replacement history.
`RtcPipelineReassessment::consider` checks the comparison against the exact
original reference and prior Apply's conditioned evidence. Ordinary spectral
relearning remains available alongside these restricted comparisons.

Full admitted-run nonfinite checks precede comparison restriction. A missing
comparison is unavailable, never a zero residual. Physical gaps stay separate;
window overlap is not counted twice in support duration. Coordinate-local r
absence and admitted x replacement influence remain distinct. Powers are
integrals of stored D2 PSD, not background-subtracted line power, independent
noise, sensitivity gain, event classification or a stopping criterion.

Runtime Apply and complete plans are unchanged. Any revised plan executes
again from original x/r. This implements a runtime Learn→Consider boundary;
following the engineering learn/consider/apply workflow alone would not do so.

## Result and evidence

Evidence root: `/private/tmp/citlali-rtc-treatment-outcome-2026-09-17`.
`REPORT.md` is the compact result; `outcomes.csv` includes all 96 coordinate /
stage / arm outcomes, with exact runtime windows and PSDs under `replay`.
The same twelve 152390/network12 detectors and existing filters were used.
Eight notched x coordinates retain 1.634–2.893% of original stored power in the
previous 10.380900–11.631611 Hz band after notch+low-pass; four low-pass controls
retain approximately 100%. Source and atmosphere remain included. Stage
comparisons use their own matched originals, not unequal-window denominators.

All 370 prior science/support/decision/injection files are byte-identical and
both frozen-plan arm receipts match apart from timings/new metadata. Originals
are unchanged. Existing 851,646/909,228 paired-output retention remains; the
spectral-review support is smaller under the unchanged adaptive rules and is
not a new output loss. No natural donor occurs in this replay; controlled tests
cover x donor continuity and local r unavailability.

Added comparison time is about 1.3 s per complete twelve-detector arm, with
roughly 4 MB logical retained outcomes across two stages. Whole caller time
145.18 s, peak RSS 932 MB. Exact source/binary/input and per-artifact identities
are in `SOURCE.json`, `REPLAY.json`, `GATES.json`, `verification.json` and the
final external closure/review/seal records. The latter bind this continuity
commit without circular self-identification. Prior evidence remains untouched.

Nine focused tests and isolated header compile, 1,189 runnable CTests, 207
baseline Python tests, 82 RTC Python tests and required config preflight pass.
One existing MapFitterLifecycle test remains disabled. Independent fresh-context
review of the exact runtime SHA passes all three axes with no actionable
findings; it independently repeats 25 outcome/pipeline tests and confirms
370-file parity and all 96 power integrals. Final documentation has its own
exact review binding. Local AppleClang21/Homebrew/C++20, existing kids04088da-dirty,
is supplemental evidence, not Unity/Spack qualification. Successful replay has
no unexpected error-level messages. Initial fixture mistakes and their bounded
corrections remain in `REASSESSMENT.md`.

## Authority and continuation

Tier2 governance: effective ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE
and REVIEW_AND_CONFORMANCE, owner acceptance
`06a3ade51c1b3f38887295433d913811bf25cd14`, incorporation
`77507836325eff9f469062d5884481ea37599594`; repository AGENTS and toltec-context.
SCI-RTC r0.12 requirements 010–013,018–027,055,118,121,125,135,140,
accepted initial spectral use and 2026-09-16 fixed-filter speed-support authority
remain governing. Complete preflight, risks and gate evidence are in the root.

This completes the matched-support outcome-evidence connection within S2,
not all S2. The next RTC implementation responsibility is an explicit complete-
plan retain/revise/unavailable disposition through the existing attempt sequence,
using original evidence and these outcomes, followed by fresh-original replay.
Automatic admission/stopping thresholds remain unselected. Common-mode science,
natural-event admission and donor expansion are not the next task. Filtering,
support, paired validity, coefficients, motion, calibration and weights remain
unchanged; FRUIT, map work and production remain deferred.
