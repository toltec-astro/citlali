# SCI-CAL-001 atmosphere regeneration and operator-decision package

This directory preserves the 2026-08-01 reproducible evidence package bound to repair base `9aae0e669384c5c0c0dda93debc194d6b8dac787` and repair-line evidence head `ae99be1cef8c390d0e7490835ffca1f31da7ebc0`.

The outcome has advanced from a previously unresolved two-path choice to an owner-selected, separately versioned AM 12.2 successor evaluation. The generic q25/q50/q75 raw grids and exact legacy fit are recovered. A separately named copied AM 12.2 annual/seasonal suite is completely inventoried, and a distinct native build reproduces all 180 annual reference grids exactly. The R1 frequency-resolution diagnostic passes, and P1 directly evaluates all 100 copied-profile/H2O-scale hypotheses. These results establish copied-suite software and numerical reproducibility. The copied files are distinct registered products from the generic-q files; their historical generic-generator association is not established. Generic q95 and exact generic custody remain unresolved historical/diagnostic provenance, not successor-evaluation gates. No successor operator or operational domain is selected or authorized.

## Read first

- `OWNER_DECISION_BRIEF.md`: concise decision, recommendation, remaining choices, and stop boundary.
- `REGENERATION_SPEC.md`: versioned raw-generation, legacy-fit, operator, and validation contract.
- `LOCAL_PROVENANCE_INVENTORY.md`: read-only search record, positive findings, negative result, and digests.
- `owner_input_request.json`: exact machine-readable request limited to unresolved facts.
- `owner_supplied_manifest.schema.json`: compact schema for partial or complete owner responses.
- `regeneration_manifest.json`: partial-recovery state conforming to `atmosphere_regeneration_manifest.schema.json`.
- `NATIVE_REGENERATION_REPORT.md`: exact 180-case copied-suite native reproduction.
- `FREQUENCY_RESOLUTION_REPORT.md`: R1 10/5/2/1-MHz convergence result.
- `COPIED_AM_FOLLOWUP_REPORT.md`: copied-suite custody, identity, and C1 stress evidence.
- `H2O_SCALE_HYPOTHESIS_REPORT.md`: canonical all-direct P1 ranks and limitations.

## Decision state

The owner direction is evaluation, not adoption or operator selection:

1. **Selected study direction -- separately versioned AM 12.2 successor:** evaluate whether the reproducible AM 12.2 family warrants adoption under an explicit profile rule, H2O-scale construction, grid, spectral convention, unresolved-line warning policy, independent validation design, and support-backed q95-excluding study domain.
2. **Historical provenance retained, nonblocking:** generic q95 datafile ID 461 and exact generic generator/profile/grid custody remain unresolved. No copied product is substituted for a generic product, and the successor must not be represented as a historical regeneration.

The machine state is `evaluation_only_not_adopted`; successor-study results are pending and intentionally unbound.

Retain piecewise-linear line-of-sight optical depth as the baseline and PCHIP as the challenger only. Neither is authorized. Selection requires exact study anchors, finite positive transmission, continuity, opacity monotonicity, fail-closed support, and no more than one-percent fractional extinction-correction error against independent model truth over the declared successor study domain. Elevation monotonicity must pass within that domain or receive an explicit owner scientific disposition supported by independent model evidence. The legacy q95/a2000 `0.839827%` feature and the full-q0--q95 C1 maxima remain historical diagnostics, not successor release gates. Exact operational opacity/elevation endpoints remain unapproved.

Zenith `tau225` must be applied with the full airmass of every eligible sample and a top-of-atmosphere pivot, `X_ref=0`. Software correctness, numerical model-representation fidelity, and observational performance are separate gates; the target 5--10% absolute flux accuracy and provisional approximately 5% observation-to-observation repeatability are not established here. The open SCI-ALIGN-001 handoff additionally requires ordered sample identity, timing-gap/interpolation origin, duration, and original-versus-synthesized eligibility before aligned elevation can be consumed.

## Generated evidence

`recover_legacy_raw_grids.py` verifies complete q25/q50/q75 source NPZ digests and generates:

- `recovered_raw_grid_manifest.json`;
- `recovered_raw_nominal_grid.csv`;
- `recovered_fit_coefficients.csv`;
- `raw_anchor_fit_metrics.csv`;
- `raw_anchor_operator_metrics.csv`;
- `raw_q50_holdout_metrics.csv`;
- `raw_q50_operator_holdout_metrics.csv`;
- `raw_grid_physical_metrics.csv`;
- `RAW_GRID_RECOVERY_REPORT.md`.

`generate_operator_analysis.py` verifies the frozen repair-base and phase-0 inputs and generates:

- `legacy_anchor_manifest.json`;
- `legacy_anchor_surface.csv`;
- `legacy_anchor_metrics.csv`;
- `candidate_surface_metrics.csv`;
- `leave_one_anchor_out_metrics.csv`;
- `candidate_disagreement_metrics.csv`;
- `CONTINUOUS_OPERATOR_EVALUATION.md`.

The copied-suite follow-up tools and their frozen protocol records generate deterministic inventory, C1, native-regeneration, R1, and P1 reports, tables, manifests, execution-context identities, and digests. Canonical P1 contains 100 scale rows, 1,200 metric rows, and 1,050 coefficient rows. All 100 direct full grids reproduce their parsed T225 anchor exactly. Its 13,667 referenced AM runs comprise 9,792 status-0 anchor runs and 3,875 complete status-1 full-grid runs with only the retained unresolved-line warning class; other warnings, errors, and failed canonical attempts are zero. The execution context is SHA-256 `05148050e96e73577ec75be525b026b5bf37bbd2a8753f8e3702fc0b6dfb2bee`, and the frozen runner is SHA-256 `caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c`.

P1 finds a full-grid provenance mismatch even after exact T225 matching: no copied-profile hypothesis passes the one-percent maximum correction diagnostic over all 0--500 GHz samples for q25, q50, or q75. The smallest maxima are `97.968871%` (q25/MAM25), `99.845844%` (q50/MAM25), and `98.987223%` (q75/annual50). This is distinct from the legacy nominal-frequency diagnostic: at 272.73, 214.29, and 150.00 GHz, all 25 hypotheses pass one percent for q25/q50/q75, with a worst result of `0.665829%` at q75/a1100/JJA5. The q95 raw grid is absent; its weaker 93-point ratio-surface comparison has no one-percent pass, with best maximum error `1.117452%` (annual25) and transmission-ratio RMS rank-one maximum error `1.190949%` (DJF25). For q25/q50/q75, direct copied-AM `atmTaun` is authoritative on the candidate side, while P1 reconstructs generic truth-side line-of-sight tau as `-log(atmTtx)` because those NPZs contain transmission but not tau. q95 is necessarily ratio-only and reconstructs both sides as `-log(T_band/T_225)`. The frozen P1 report and manifest overgeneralize the candidate-side tau authority; this package-level clarification supersedes that wording without changing the frozen artifacts. P1 is post hoc, not custody proof or an independent operator holdout. The frozen addendum's descriptive phrase "closest same-percentile family is DJF" is not a registered ranking; final transmission and Rayleigh-Jeans ranks are separate and use the all-direct P1 definitions.

`SHA256SUMS` covers every regular file in this package except itself. The full raw NPZ inputs are not duplicated into this repository; their immutable SHA-256 values, TolTECA MD5 identities, repository lineage, and observed read-only paths are recorded. Generated recovery artifacts locate inputs by filename and digest rather than embedding the runtime directory, so identical bytes may be staged elsewhere and passed with `--source-dir`.

## Reproduce and verify

Run from the Citlali repository root with the required local venv:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/recover_legacy_raw_grids.py \
  --source-dir /Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/generate_operator_analysis.py
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/generate_package_digests.py
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/verify_package.py
```

The verifier checks both JSON Schemas, exact unresolved-request IDs and owner-path semantics, CSV structure, regenerated byte identity, package digests, phase-0/coordination records, raw NPZs, supporting sources, and dissertation digest. It performs no network or Unity access and makes no sibling-repository changes.

The canonical P1 artifacts can be reconstructed from the external cache without launching AM by running its exact cache-only check:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/probe_am12_h2o_scale_hypotheses.py \
  --am-executable /private/tmp/sci_cal_001_am12_2_native_build_20260801_root/am \
  --am-root /Users/gwilson/work_toltec/local_data/AM \
  --legacy-source-dir /Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity \
  --cache-dir /private/tmp/sci_cal_001_h2o_scale_p1_context_v3_lightweight_final_20260801_root \
  --jobs 8 --omp-threads 1 \
  --compiler-executable /opt/homebrew/Cellar/gcc/15.2.0_1/bin/gcc-15 \
  --native-build-command 'make -j8 gcc-omp COMPILER_GCC=gcc-15' \
  --check
```

The full local verifier can additionally validate both canonical external caches in cache-only mode:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/verify_package.py \
  --h2o-cache-dir /private/tmp/sci_cal_001_h2o_scale_p1_context_v3_lightweight_final_20260801_root \
  --native-cache-dir /private/tmp/sci_cal_001_am12_2_native_matrix_context_v2_final_20260801_root
```

On another host, `--skip-external` omits unavailable machine-local coordination and AM-tree checks. If `--raw-source-dir` is explicitly supplied, its digest-identical q25/q50/q75 and supporting files are still checked; they are not skipped:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/verify_package.py \
  --skip-external --raw-source-dir /path/to/staged/toltec_sensitivity
```

For a check-only rerun after generation:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/recover_legacy_raw_grids.py \
  --source-dir /Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity --check
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/generate_operator_analysis.py --check
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/generate_package_digests.py --check
```

## Scope boundary

This package does not modify Citlali application code, implement a repair, launch a re-audit, access Unity, fetch q95, or edit the coordination registry. The SCI-ALIGN-001 handoff remains an explicit downstream eligibility dependency for any future aligned-elevation consumer.
