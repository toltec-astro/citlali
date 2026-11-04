# Configured POINT 152389: owner-run result assessment

Date: 2026-11-04. Disposition: **successful representative configured POINT
execution for this observation and recipe; no actionable issue found in the
reviewed evidence. Scientific method/use and production acceptance are not
established by this record.**

Grant executed the version check and reduction. Codex and the independent
reviewer performed static file/Git/report inspection only. No builds, tests,
probes, reductions, refits, plotting or validation runners were launched by
either reviewer. No Unity access, push, runtime replacement or cleanup.

## Identity and authority

- Application: `19b6fc462930af6f0a7bf2ca5b5350dd3c72af6e`; tree
  `3b1d9e0ae067e502faa1e02895be3695f5166a66`. Preserved stage-3 checkout
  remains clean at this identity.
- Assessment documentation base: clean `4404034e12bfa92f913ed5fcc263ec439d63282b`
  on `codex/refactor-mainline`.
- Native SHA-256: `40cf99ab50f3d65f3a37b878ea08e28d39d1f535e9ca8c339ebcf356b0c7358a`.
  Mac Spack, Clang 20.1.8, DAG `kqk3pl4f6jgsb5yksobnynlx2gmnqks6`;
  unmanaged deployment. This does not reproduce the Unity V2 campaign.
- Scope: [selected 152389 work order](../../doc/POINT_QUALIFICATION_152389_2026-11-04.md),
  [qualification sequence](../../doc/POINT_SCIENCE_QUALIFICATION_PLAN_2026-11-03.md),
  [POINT method/use boundaries](../../doc/POINT_METHOD_AND_USE_ACCEPTANCE_2026-11-04.md),
  scientific conventions and accepted engineering/successor/review governance.

The packet is
`/Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-point-qualification-20261104`.
Owner attempt: `run-152389-20261104T150728.304089Z`.
Reduction: `outputs/152389/redu00`; final products: `152389/raw` within it.

| Evidence | SHA-256 |
| --- | --- |
| Packet `BINDING.json` and attempt copy | `fe94f367dbdc12c71f97dc18fb22561a83da6537e58beb0553be13ebb1ab2dd6` |
| Received `configs/152389.yaml` | `9fe9d1e1a528a2916f4eb4206580e81595d80cfd8060138313dc576a3cb45b56` |
| Reduction `RUN.json` | `d91e6a7733fac8091e8686c8021117925f0a28d2186b108e78a97aeba6adc18e` |
| Attempt `point.log` | `78b4879d1c28c07d5676329bd18622b76635f02b18ba4a2ceedb44777762eafb` |
| Published pointing ECSV | `b38dc5e1b95ca12ebd4ba8470d21eb29634c9a602ccc5d0e6b2964cfa4819c03` |

## Observed execution and publication

The configured application records 540.55 seconds, approximately nine minutes.
Version and reduction commands exited 0. All 13 native calls exited 0 with
empty unexpected-error lists: CAL, optical-health preparation, and eleven
weighted map cycles M0–M10. The observation receipt records all 11 networks
completed, with three requested/started workers. Top-level and native logs
contain no warning/error/critical messages or tracebacks in the inspected logs.
Existing RTC messages about unresolved reassessment/natural-isolated policy
remain explicit retained limitations, not new scientific-policy acceptance.

Input identity is observation 152389, subobservation 0, scan 2, source 1146+399.
The 39 preflight external input path/hash records match the realized input
ledger. The owner runner also rehashed those inputs before and after reduction.
Settings remain the selected weighted recipe: all three arrays, 2 arcsec
pixels, coverage 0.15, ranks 3/4/6, optical threshold 0.10, zero supplied az/alt
offsets, three workers and a ten-feedback-application budget. No coadd or
equal-weight comparison arm was requested.

Feedback completed all ten applications after M0. The recorded reason is
`fixed_budget_completed`, with `convergence_claim=false`. This is completion
of the requested budget, not proof of scientific convergence.

Final FITS and bundle hashes match the product/fit/table receipts. Inspected
primary headers on all three maps show delta-source Alt/Az, `AZOFFSET` /
`ELOFFSET`, arcseconds, zero reference offsets, pixel steps (-2,+2) arcsec and
`mJy/beam`. The sidecars retain nominal-beam meaning and empirical inverse-noise
coefficient semantics; normalization is not claimed as map precision.

All final `POINT-EXPECTED-v1` guards are accepted. The published ECSV rows
match the final diagnostic amplitudes and centroids. Signed measured source
displacements are:

| Array | Tangent azimuth (arcsec) | Elevation (arcsec) | Existing fit guard |
| --- | ---: | ---: | --- |
| a1100 | +7.589926 | -0.222293 | accepted |
| a1400 | +7.425521 | -0.477918 | accepted |
| a2000 | +7.533495 | -0.629206 | accepted |

These are measurements; downstream correction construction remains separate.
The pointing table is published and `pointing_eligible=true` under the existing
all-array adapter. Errors remain `NaN`, uncertainty status remains unavailable,
`measurements_are_corrections=false`, and `calibration_eligible=false`. This
publication does not implement the newly approved array-selection strategy.

No detectors were newly dropped by the disturbance policy. Optical-health
publication separately records retained/weak/reversed/unresolved/prior-unavailable
populations. Retained counts are 2018/765/550 for a1100/a1400/a2000; unresolved
counts are 343/166/258. These exclusions remain explicit; zero disturbance
drops does not imply universal detector eligibility. The conditional-bootstrap
coverage and false-rejection limitations remain in `detector-health.json`.

## Historical comparison and limitations

Read-only text comparison with the completed weighted 152389 fixture at
`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-weighted-152389-20261029/reduced/redu00/152389/raw`
shows each final `point-fit-<array>.json` differs only in `map_sha256`. All
saved fitted parameters, uncertainty/guard diagnostics and other JSON values
are identical. This is useful repeatability evidence against that recorded
case; the historical source was `331ff29ea` in a different non-Spack realization.
It is not a same-source/environment acceptance campaign.

Full FITS pixel-plane equivalence, fresh image statistics, fits, noise errors,
raw-sample numerical behavior and absolute sky-reference accuracy were not
recomputed. No new accuracy or broad-source-recovery threshold was applied.
The normal requested intermediate-retention policy released newly generated
transport payloads; receipts/final products remain. No reviewer restored or
replayed those payloads. The new array-selection/beam policy, approximately
1.5 arcsec accuracy target, flux calibration, SCIENCE/coadds/known-offset check,
Unity and production remain outside this result's claims.

## Review dispositions

- **Scientific/behavioral:** successful existing configured development behavior
  for the selected observation/settings, with the above limits. This is not
  frozen SCI-POINT method conformance or scientific-profile registration.
- **Architecture/ownership:** no application changes. Inspected ownership of
  final-map fitting, publication and the one-way table adapter is preserved;
  no extra fit, correction or fabricated error is introduced by publication.
- **Repository/evidence:** exact source/native/config and inspected receipt/
  product identities agree; preflight and realized input ledgers reconcile;
  original inputs and historical results are preserved. No push or activation.

[Independent fresh-context review](INDEPENDENT_REVIEW.md) found no actionable
publication, fit/table or evidence inconsistency within its static coverage.
No repair or rerun is indicated merely to record this bounded result. Owner
acceptance of any scientific use remains a separate explicit disposition.
Observation 152418 is not authorized by completion of this one-observation batch.
