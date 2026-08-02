# SCI-CAL-001 atmosphere-confirmation and passband decision — 2026-08-02

Status: owner passband choice approved; bounded EL25 confirmation authorized;
no atmospheric operator, application repair, or production use authorized

Package: `SCI-CAL-001`

Authority: project owner acting as calibration and atmosphere scientist for
`BAND-001`; scientific-audit coordinator for the bounded evidence-execution
policies `DOMAIN-001` and `WARN-001`

Evidence branch: `codex/sci-cal-001-atmosphere-operator`

Evidence head: `f4014d3669b94b1eceb8158da7993737efc908f2`

Evidence parent: `742a3c263faf68c2de7d5b8db0d3423127f60480`

Evidence-package `SHA256SUMS` SHA-256:
`bafd34e4a3d5bffb95b3af1fdbcfb7c993146248b2bccd1d0333bae91fd3caad`

## Owner decision: `BAND-001`

The project owner selected `select_tolteca_v1_ecsv`. The exact immutable
passband-set provenance identity is:

- passband-set ID:
  `toltec-passband-set-v1:sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433`;
- audit authority-record ID: `SCI-CAL-001-PASSBAND-AUTHORITY-001`;
- human alias: `tolteca-v1-ecsv`;
- coordination manifest: `SCI-CAL-001_PASSBAND_AUTHORITY_001.json`;
- manifest SHA-256:
  `2756908181cc466550399ec0a869e6671de7912bd3a935f9aeebf63e3e826617`;
- TolTECA source commit:
  `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`.

The passband-set ID binds the TolTECA index and all three ECSV files. The
array-file SHA-256 values are `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72`
for a1100, `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e`
for a1400, and `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff`
for a2000. The index SHA-256 is
`74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5`.

For the bounded confirmation, use the exact ECSV frequency nodes and
throughput values as supplied, composite-trapezoid quadrature, no spectral
extrapolation, and top-of-atmosphere `S_nu` proportional to `nu^alpha` for the
frozen test values `alpha = {-1, 0, 2, 4}`. This energy-weighted integration is
the confirmation-study convention, not a universal production color-
correction decision.

The selected curves are array-named modeled responses. The surviving evidence
does not establish them as detector- or network-weighted telescope
measurements, does not establish uncertainties or covariance, and does not
establish the physical photon-versus-energy response convention. The curves
are used as supplied and are neither peak- nor area-normalized. These unknowns
are retained as provenance limits rather than inferred from numerical
agreement.

## Coordinator evidence policies

`DOMAIN-001` is resolved for this confirmation only:

- closed zenith-opacity support:
  `0 <= tau225 <= 0.158313198574890929`;
- closed elevation support: `25 deg <= EL <= 80 deg`;
- q95 excluded;
- full eligible-sample modified-secant airmass and top-of-atmosphere pivot
  `X_ref = 0`;
- fail closed outside support or when the required SCI-ALIGN-001 sample
  identity, aligned elevation, interpolation origin, duration, or eligibility
  state is absent or ineligible.

This is a confirmation-study domain, not an operational-domain authorization.

`WARN-001` is resolved as
`accept_bounded_status_1_warning_bearing_evidence`. An AM status 1 result is
admissible only when it has the complete expected 50,001-row output; only the
preregistered unresolved-narrow-line warning records and corresponding summary;
the canonical unresolved-line summary count of 86, 87, or 88; and no unknown
warnings, cache mutation, or errors. It must always be labeled warning-bearing
evidence, never clean success. Every other nonzero status fails closed. This is
an evidence-generation rule, not a Citlali runtime warning policy.

## Future telescope-passband measurement plan

Plan ID: `SCI-CAL-001-PASSBAND-MEASUREMENT-PLAN-001`

Status: `planned_nonblocking_awaiting_telescope_measurements`

The future process keeps acquisition, processed candidates, and adopted sets
as different objects:

1. Ingest each raw campaign without altering its bytes and assign
   `SCI-CAL-001-PASSBAND-MEAS-YYYYMMDD-NNN`.
2. Preserve raw-file digests and produce detector/network-level responses as
   derived artifacts. Assign each reduction a distinct immutable candidate ID,
   `SCI-CAL-001-PASSBAND-CANDIDATE-NNN`.
3. Preregister quality cuts, exclusions, detector/network-to-array weighting,
   aggregation, normalization, frequency-grid handling, uncertainty treatment,
   and out-of-band policy before inspecting the adoption comparison.
4. Compare repeatability and scientific impact with the v1 content-bound
   passband-set ID, including the CAL atmosphere operator, Beammap calibration,
   and representative source spectra.
5. Require an explicit calibration-owner amendment before adoption. An
   accepted candidate receives both a new immutable audit adoption-record ID
   and a new content-bound
   `toltec-passband-set-<release>:sha256:<digest>` identity with a declared
   validity interval; it never overwrites the v1 set.
6. Change the active TolTECA/configuration reference only in a separately
   authorized repository change, then rerun the affected numerical and
   observational CAL/Beammap gates. Preserve every prior set and adoption
   interval so older reductions remain reproducible.

Before promotion the owner must designate one upstream calibration custodian
and one authoritative release manifest. TolTECA is the current selection and
passband-byte authority; this coordination record only indexes that upstream
authority and does not create a second copy. A future raw archive or derived
candidate may live elsewhere, but it does not become authoritative merely by
existing.

Every campaign and candidate manifest must bind observation/file IDs,
acquisition UTC, telescope/instrument/tuning configuration, detector/network/
array and APT identity, frequency frame/grid/units, response quantity and
units, raw and derived artifact digests, processing code commit and execution
environment, baseline/deglitch/negative-value processing, QC and exclusions,
uncertainty representation, aggregation weights, normalization, quadrature,
out-of-band handling, accountable reviewer, parent/supersedes relations, and
status (`candidate`, `validated`, `adopted`, or `retired`). Unknown facts remain
explicitly `unknown`.

Later requested, effective, observation-resolved, and realized calibration
provenance should carry the compact content-bound passband-set ID and the
authoritative release-manifest digest. Dense per-sample copies of the passband
are neither required nor desired.

## Immediate authorization and stop boundary

This decision authorizes the existing CAL evidence task to preregister and run
one independent EL25 numerical confirmation against the exact passband and
domain above. The post-hoc EL25 slice remains diagnostic only until that
confirmation completes and is independently reviewed.

Do not adopt an atmospheric operator, authorize a production calibration or
operational domain, modify Citlali application code, request Unity evidence,
begin CAL repair, or launch the CAL re-audit. Stop after committing the
confirmation evidence and return it to the coordinator for review.
