# Saved feedback inspection — 2026-10-30

Owner request: inspect saved a1100/a1400 feedback masks and source profiles,
including 2 arcsec sampling; retain noise weighting; no new reductions.

Preflight: TolTEC context, Engineering Governance, Timestream Successor Governance
and Review and Conformance consulted. Sequencing: REFACTOR_STATUS.md and explicit
owner direction. Clean base ea61862a5f4bdc9c3dad0443a7ca295be1ff7e93;
branch codex/configurable-point-20261028, current configurable-point worktree.
Tier 2 evidence-only, no application WIP. Adopt the previous paired-assessment
products and arithmetic; scientific interpretation remains provisional.

Read authenticated noise-arm M0–M9 masks/maps/fits and their next-cycle requests.
Measure selection, radial/sector profiles and exclusion without refitting data.
Evaluate the already-saved Gaussian shapes on fine grids to isolate idealized
2 versus 1 arcsec containing-pixel representation error. This is numerical
quadrature of saved analytic parameters, not a native processing run, injected
source experiment, new fit or detector-time propagation. Do not equate toy error
with observed flux loss or accept a new scientific method.

Owners unchanged: configured POINT coordinator invokes source selector; native
FRUIT owns subtraction/restoration and containing-pixel evaluation; MAP owns
projection. No application/configuration/threshold/noise-policy changes, Unity,
push, calibration, cleanup or reductions. Preserve all saved products. New scope
or missing evidence triggers a recorded limitation rather than reconstruction.

Focused gates: exact model/map/request/next-plan associations, mask/value identity,
finite arithmetic, deterministic saved-only reproduction, quadrature refinement,
visual review and fresh-context exact-SHA independent review. No build/CTest/config
gate or Unity qualification is triggered. Outputs: scoped script, metrics, report,
figures and status/closure. Final integration and operational choices stay with Grant.
