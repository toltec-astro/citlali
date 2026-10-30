# Source-offset AltAz OOF JINC launch

> **Historical launch record — held for replay (2026-10-30).** This selected
> history contains pre-repair OOF replay helpers. The referenced external
> passing review/preparation binds `a8c1f313f8ef688538e2aec0e1f20eb57ce1a58c`,
> which is outside the approved pointing/engineering input history. Its
> manifest-location and comparison-import fixes must be recovered and reviewed
> under a separate OOF work scope before these instructions are runnable.
> The former worktree path below is also retired. Preserve the historical
> command and evidence; do not execute it from this candidate. No new OOF
> repair, launch, optical qualification or budget is authorized here.
> See [the candidate scope](../../doc/POINTING_ENGINEERING_CANDIDATE_2026-10-30.md).

This work prepares three new map-only applications,152385/0/1–152387/0/1,
a2000. Exact runtime readiness is recorded externally in
`/Users/gwilson/work_toltec/local_data/citlali-validation/successor-oof-a2000-altaz-jinc-20261002/{GATES,PREPARED,review,FREEZE}.json`.
Do not launch without those passing exact-source records.

All previously admitted samples, values, CAL/APT identity, uniform coefficients,
processed frequencies, kernel parameters and numerical limits remain fixed.
Native AST regenerates sample coordinates from the bound telescope source
azimuth/elevation, corrections, APT geometry and pointing offsets. New FITS
uses AZOFFSET/ELOFFSET in arcsec, zero reference, matching the historical R1″
WCS exactly; preparation rejects any retained center outside that grid. There
is no post-map rotation, source fitting, registration, feedback or optical solver.
Full signed-denominator edge artifacts remain possible; numerical support is
not scientific coverage. Historical R cleaning and weighting still differ.

The shared derived storage ceiling stays30GB, free reserve2GB, new campaign
reservation1GB. One serial numerical worker. Old FK5 JINC executables are
preserved under its executed-jinc-binaries directory before rebuilding.
No upstream RTC/CAL/PTC starts, POINT campaigns, cleanup, push or integration.
The existing POINT feedback reader explicitly rejects new AltAz parents; it
must receive its own frame-aware validation before POINT feedback resumes.

Owner foreground launch once, from any directory:

```bash
PYTHONDONTWRITEBYTECODE=1 /Users/gwilson/tolteca/bin/python /Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor/validation/oof_altaz_2026-10-02/altaz_campaign.py run
```

After it returns, say “done”. The saved-only `compare_saved.py` then authenticates
the outputs and renders common-coordinate raw/normalized100″ and normalized32″
panels. No automatic retry or agent monitoring. This owner launch/manual-resume
boundary is required by TIMESTREAM_SUCCESSOR_GOVERNANCE, standing2026-09-30
amendment; it is not a new scientific approval request.
