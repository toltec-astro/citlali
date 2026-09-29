# Saved-map comparison — owner request 2026-09-29

Owner: “Go ahead and carry out that comparison.” Compare prior successful and
current POINT152420/0/2 a1100 saved maps with consistent descriptive fitting.
Base a84eb2456fef4a8c2e768144dd75a47f21a83654, clean at start, existing branch
codex/rtc-point-transfer-20260928 and ptc-rank-response worktree. No new branch,
application implementation slot, reduction, feedback, cleanup, Unity or push.

Tier2 evidence analysis. Read ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE,
REVIEW_AND_CONFORMANCE; accepted identities/digests in ../rtc_point_transfer_2026-09-28/PREFLIGHT.md.
Scientific authority is descriptive saved-map inspection and the owner's comparison
request; no policy or production acceptance changes. Existing inspect_point_source
owns the numerical Gaussian solver; this local analysis consumes immutable maps.
Current sequencing: REFACTOR_STATUS and failed runtime-repair campaign status.
Units/frame/validity follow SCIENTIFIC_CONVENTIONS and each FITS product header.

Primary method fixed before running: same unweighted seven-parameter elliptical
Gaussian plus constant, same 40-arcsec origin-centered disk, nominal5arcsec
initial width and sigma floor1.5arcsec, same native solver and bounds. Require
finite supported pixels throughout each fit disk. No covariance/SNR acceptance
is recomputed. Fit on native physical coordinates: old AZOFFSET/ELOFFSET in
arcsec, new local FK5 tangent offsets in arcsec. Compare rotation-invariant major/
minor widths and radial centroid offset; no assertion that orientation or pixel
coordinates are the same sky frame. No guessed rotation or pixel subtraction.

Two fixed diagnostic checks: (1) average old adjacent2x2 pixel blocks to2arcsec,
with exact block-center coordinates and all-four support required; (2) one
15arcsec source-centered fit on each original map to check sensitivity to distant
structure, initialized from its primary fit. Neither changes campaign acceptance.
No optimizer initialization/radius sweep or reduction rerun. Plot raw maps,
fitted half-maximum ellipses, residuals, and normalized principal-axis profiles.

Gate plan: synthetic rotated-Gaussian recovery and invalid-support checks;
reproduce current saved guard's geometry; FITS identity/checksum and input hashes
before/after; validate block coordinate/value arithmetic; inspect saved plot;
independent fresh-context exact-SHA review of comparison and conclusion. Broader
CTest/build/config gates are not triggered: no application/tool owner/config edits,
only isolated diagnostic script and evidence/docs. Local tolteca Python with
Agg and temporary writable caches; no representative Unity execution required.
Stop/reassess on missing units/support/identity, failed fit, active bound, materially
changed source/inputs or unexplained inconsistency. Do not infer a specific RTC,
PTC, pointing or detector-selection cause from these two differently reduced maps.
