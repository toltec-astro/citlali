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

## Final disposition

The run stopped at the frozen persistent-multimodality gate after opening 9
of the 22 selected pointings. ObsNum 136280 remained bimodal after the full
1,500 successful exact whole-scan bootstrap realizations. The other 13
pointings were not opened, and no all-22 or high-S/N corpus inference was
computed.

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
