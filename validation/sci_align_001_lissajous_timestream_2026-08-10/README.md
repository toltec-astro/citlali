# SCI-ALIGN-001 Lissajous timestream fit

This package is the frozen entry point for the complementary direct-PTC
timestream timing diagnostic approved on 2026-08-10.  Its exact base is
commit `6ec08656fd5c12607e806f55389cc094aa4b6a2d` on branch
`codex/sci-align-001-lissajous-timestream-fit`.

The observation selection remains the checksum-bound set of 22 3C273
pointings in the adjacent map-space validation package.  The new diagnostic
may not add, remove, replace, or quality-select observations after timing
inspection.

`frozen_protocol.json` was written before inspecting any new real-data
timestream fit.  It fixes the lag bounds, common support, interpolation,
source and nuisance models, sensitivities, blocked prediction, scan-block
bootstrap, paired map/timestream bootstrap, corpus pairing, anchor discipline,
and stop conditions.  In particular, the map estimate is not treated as
uncertainty-free: the paired analysis applies identical deterministic scan
draws to both estimators and forms `tau_timestream - tau_map` per replicate.

This diagnostic tests the relative registration of the delivered PTC signal
and telescope-coordinate trajectory.  It is independent of map binning and
velocity-sector centroid compression, but not of upstream PTC processing.  It
does not identify an FPGA, PPS, detector-clock, telescope-servo, encoder,
secondary-mirror, or raw-data cause and cannot authorize a correction.

No Unity access, reduction regeneration, production Citlali edit, production
configuration edit, merge, rebase, push, or correction is in scope.

## Current disposition

The run stopped at the frozen persistent-multimodality gate after opening 9
of the 22 selected pointings. A successor read-only audit then proved that 361
of the nominally successful ObsNum 136280 bootstrap fits had actually stopped
abnormally at iteration zero and been accepted only because their objective
was finite. The historical 1,500-draw distribution and its paired statistics
are therefore contaminated, not evidence for a second physical mode.

The diagnostic-only optimizer repair was refrozen at protocol revision 6 after
all 11 synthetic tests and the full config preflight passed. Its first fresh
ObsNum 136280 attempt was owner-stopped after 66 minutes because the runner had
no stage-level progress, fallback census, or enforceable runtime bound. The
process was actively computing and left no partial artifact, but that was not
enough to establish normal progress. Runtime instrumentation and a generalized
authenticated fit-visualization tool were then installed without changing the
numerical protocol. Two bounded ObsNum 150818 lifecycle attempts showed that
the full model fits take about seven minutes, but the held-out refits dominate
the downstream cost. The 3,600-second attempt completed held-out and network
diagnostics before stopping in bootstrap; it published no scientific result.

Direct full execution is now disabled. Every new observation must first stop
after a checksum-bound `fit-gate` package containing model, optimizer,
per-scan, and visual diagnostics. Resume requires explicit owner review and
does not repeat the full fits. Every later expensive deterministic stage is
also atomically checkpointed and reused after interruption. The fitted tau is
reported at the gate but is never an acceptance criterion. ObsNum 136280
remains deferred; no further real observation has been opened. The other 13
pointings remain unopened, and no all-22 or high-S/N corpus inference has been
computed. See `REAL_RUNTIME_OBSERVABILITY_STOP_03.md` and
`FIT_GATE_CHECKPOINT_RUNTIME_AUDIT_04.md`.

The new 66-pointing high-S/N campaign completed its isolated ObsNum 150818
pilot fit gate in 1,868.66 seconds. All machine structural checks passed, but
owner inspection identified that the fit-gate PDF's page-3 source profiles
again averaged focal-plane detectors at common timestamps even though their
source crossings occur at different times. The numerical objective keeps
detectors separate, so the fit remains immutable while its compact-source
adequacy visualization is rejected. A supplementary renderer is frozen to
align each detector crossing by signed distance along its own local trajectory
and show individual crossings. The remaining 65 gates remain blocked pending
owner review of that artifact. See
`FIT_GATE_SOURCE_PROFILE_VISUALIZATION_FAILURE_05.md`.

The bounded result is documented in `REPORT.md`. Compact authenticated values
are in `partial_observation_results.ecsv` and
`partial_observation_results.json`; exact input and result identities are in
`partial_input_identities.json`; and the machine-readable stop disposition is
in `partial_stop_summary.json`. `representative_anchor_o150818.pdf` and
`representative_stop_o136280.pdf` retain the point-objective, residual, and
paired-bootstrap diagnostics for the anchor and stopping observation.

`PREEXISTING_WORKTREE_STATE.txt` records the complete 32-file owner-owned
untracked state seen before this work. Those files remain untouched, so only
the scoped diagnostic diff—not the entire worktree—is described as clean.
