# POINT152420 completion repair — 2026-09-30

Owner: Grant Wilson; accepted directive is COMPLETION_AUTHORITY.md, copied
verbatim from the owner attachment. This implements its explicit authority,
not instructions inferred from an unrelated document. Tier 2, existing module
probe and worktree, branch codex/rtc-point-transfer-20260928. Initial source
`dbc7d47fcd537a2ca01e650f9db7fc031861ec32`, clean. Canonical authority and
scientific bindings remain those in PREFLIGHT.md; original user checkout remains
untouched. Read ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE and
REVIEW_AND_CONFORMANCE; governing incorporation and digests in PREFLIGHT.md.
The owner explicitly authorizes the standing governance amendment.

Scope: independent auxiliary fixed-template support, directly dependent
measurements and uncertainty bindings, compact executed M0/T1–T5 fit metrics,
a fresh completion batch, failure accounting and authorized reproducible
scratch cleanup. No native scientific changes: RTC/CAL, rank10 observed-entry
ALS warm initializer/full solve, cap500/tolerance1e-5, five applications,
2 arcsec ordinary MAP, 40 arcsec fit/feedback and threshold3, original Q policy
and eight role design remain frozen. T5 consumes m4; m5 is not applied.

Ownership: point_auxiliary owns geometry/physical-grid extraction and unmasked
normalization; point_reference owns full-map noise contexts/conditional
uncertainty; point_guarded_feedback owns the executed fit description and
availability; orchestration supplies actual full-map samples and lifecycle.
B/M0 owns matched geometry and injection rasters, copied unchanged into Q.
Q keeps its own noise reference/learned states. No Engine/native interface changes.
Geometry schema and exact source bindings invalidate old cached context. Memory
is a small map window; no new timestream arrays or numerical workers.

The auxiliary rectangle is the deterministic union of the old fit/constant
operator rectangle and the complete three-major-FWHM circle. No interpolation,
data padding, clipping, recentering or support renormalization. Both supported
and missing template pixels are recorded. The full-map C0/BASE background masks,
coverage model and noise learning remain unchanged. If an auxiliary rectangle
exceeds the old 49-pixel covariance-domain minimum, only its auxiliary PSD is
rebuilt from the same empirical pairs with the same taper/projection procedure;
guard covariance stays unchanged. Linear fit/constant operators can be embedded
with zero coefficients; this is not padding missing observed data. Unsupported
noise transfer withholds its uncertainty.

Continuation determination: the failed B has a map and native upstream products,
but no successful UPSTREAM publication, complete context, realized manifest or
ready feedback state. The ordinary runner has no validated resume entry at this
partial boundary. Do not create a checkpoint system or transplant source-bound
state. Rerun ordinary B first under new bindings, then remaining original roles.
Zero roles completed; the new directive's two additional replacement starts
remain unused. Historical failures stay in RECOVERY_ACCOUNTING.json and their
compact campaign records. Any future replacement is manually prepared during
owner-resumed work with failure/repair/evidence/invalidation and consumed counts;
the unattended batch cannot retry or spend that allowance.

Focused gates: complete/contained/enlarged/missing/invalid auxiliary tests,
physical coordinates, normalization, nuisance propagation, unchanged guard and
feedback; saved current M0, earlier V2 and redu09 tests; full saved-M0 initialize
and conditional endpoint (no native trajectory). Broader: all enabled CTest,
baseline/POINT/native FRUIT/runner tests and full config. One independent focused
milestone review of substantive implementation/conclusions at exact SHA.
Actual environment: local AppleClang/Homebrew fallback and owner Python venv;
supplemental evidence, not accepted Unity Spack reproduction. No Unity access.

Launch: no established persistent process mechanism. Prepare owner-terminal
foreground command after validation/review, then end turn. No automatic numerical
launch, model polling or log-watching. Final scientific comparison/adoption is
pending actual batch output and the same focused milestone review's conclusion
stage; no activation, integration or push. Changes can be committed locally.

## Local precommit evidence

Six focused auxiliary tests passed; all 209 then-current POINT tests and 11
runner fixtures passed (full exact-source gates follow commit). Saved M0 source
guard is exactly unchanged; current template has five nonzero pixels outside
40 arcsec, in a 41x41 window. Native redu09 needs 87x93 pixels, including 431
outside the feedback disk. Both full intended footprints are supported and
normalized to unit Gaussian response. The contained earlier V2 weights are
bitwise identical. Saved-M0 endpoint yields all conditional uncertainties.
Six-cycle export/historical comparison rendered in a synthetic repeated-M0
fixture only; not an observation. Detailed evidence is in completion-results/.

The saved endpoint check exposed spurious Apple BLAS floating-point warnings
in the existing small SVD matrix products. Explicit NumPy contractions now
perform the same linear algebra without those flags; no estimator, rank
threshold or scientific tolerance changes. The saved check is warning-free.

Authorized cleanup removed 44934 reproducible upstream binary files
(3760231210 bytes) from the two failed B attempts after checking
process state. All original dependencies, saved maps, compact diagnosis,
receipts, bindings and logs remain. The external removal manifest records every
path/size/SHA. Source-bound deleted payloads are intentionally unavailable,
not claimed preserved; authoritative inputs plus recorded recipe permit rerun.
Updated full campaign peak admission bound: 28460015015 bytes,
within unchanged 30000000000 bytes. No trajectory launched.
