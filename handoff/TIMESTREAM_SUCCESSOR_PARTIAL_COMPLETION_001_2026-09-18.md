# TIMESTREAM-SUCCESSOR-PARTIAL-COMPLETION-001 — 2026-09-18

The owner directed completion with good detectors, without rescuing the 51.
The [owner binding](../doc/SCI_RTC_PTC_PARTIAL_COMPLETION_OWNER_BINDING_2026-09-18.md)
permits RTC completion when missing assessments belong to entirely empty output
with no active donor dependency. PTC omits zero-eligible CAL columns before
fitting, then freezes membership. All requested output identities remain.
Thresholds, filters, paired validity, masks and numerical controls are unchanged.

The ordinary invocation for observation 152390/0/2, network 12,
Science/Lissajous NGC4449, completed RTC → CAL → PTC → VAL with exit 0.
All 124 rank-10 iterative fits converged. All 27,157,906 eligible samples from
363 detectors were retained. The same 51 detectors remain unavailable in the
414-detector product. Segment membership is 363 in 121 segments, 362 in two,
and 319 in one; zero segment support is not a new observation-wide rejection.
Partially observed columns remain admitted, and response/VAL preserve the full
requested output mapping.

All 3,734 prior RTC binary files are identical. Every admitted PTC value was
checked against exact CAL bits. Both methods succeeded in all 124 matched
saved-input replays; iterative outputs match the default bit-for-bit.
The full run took 618.69 seconds. PTC took 32.96 seconds: preparation 15.12,
fitting 3.62, Apply 0.104, and publication 13.86. Whole-process peak RSS was
14.06 GB. Matched standalone fit totals were 4.04 seconds iterative and 3.41
seconds pairwise. Fixed segment 13 (610 times × 363 detectors) rank 5/10/15
median fit times were 26.32/25.86/30.31 ms iterative and 22.68/23.29/23.22 ms
pairwise. Three iterative ranks cost about 3.2 times one rank-10 fit plus Apply.
The tighter rank-10 check changed cleaned values by a relative norm of
4.856e-5 with identical retention. No scientific acceptance or rank choice is
inferred from these measurements.

Executed pipeline: `96247c94311edc53a8e26607a138b7de9aae7bd2`, tree
`085df30969889c62de5d70d683abfac9a1f1ffa8`.
Reviewed implementation including the replay-reader repair:
`4690f97471c9492bb2c935b01753ff467fe431c6`, tree
`ed9d009705a4cec9e83256c8e2a0359a34bca9e1`.
Only the separate comparison-tool reader differs; pipeline source is identical.
The independent v2-reader finding is resolved. Exact-SHA review passes with
recorded limitations on all three axes. Canonical base is pushed
`cb5361f7425ba7d211e284ee57a9a3a67cf393b5`; no moving-base merge was needed.

Gates: 1,258 runnable CTests pass, with one established disabled test; 17 focused
and eight numerical tests were independently reproduced. Full configuration
preflight, 207 baseline, 86 timestream, four source-graph and 12 default-CLI tests
pass. Local AppleClang/Homebrew evidence does not extend Unity qualification.
The preexisting `kids 04088da-dirty` dependency banner remains recorded.
No production, MAP/FRUIT, rank-selection or complete-response claim is made.

Durable evidence:
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-partial-completion-4690f9747-20260918`.
`RESULT.md` contains the comparison table and invocation. Full products are in
`default-network/donor-continuity/cal` and `default-network/donor-continuity/ptc`.
Working evidence origin:
`/private/tmp/citlali-successor-partial-completion-20260918`.
The full run binds 96247c943; exact comparison and rank probes bind 4690f9747.
The prior failed run, population audit and frozen old-route baseline are intact.

The active default stops after PTC/VAL. There is now actual full-network output
to assess. The next downstream implementation owner remains typed MAP
consumption under its own use profile; further rescue of the 51 is not a
prerequisite. The owner performs all pushes. Unrelated worktrees are unchanged.
