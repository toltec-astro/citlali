# SCI-CAL-001 atmosphere regeneration and operator-decision package

This directory preserves the 2026-08-01 reproducible evidence package bound to repair base `9aae0e669384c5c0c0dda93debc194d6b8dac787` and repair-line evidence head `ae99be1cef8c390d0e7490835ffca1f31da7ebc0`.

The owner-selected, separately versioned AM 12.2 successor evaluation is now
complete.  Its provenance and structural gates pass, but all four frozen
lane/operator candidates miss the preregistered one-percent representation
gate over q0--q75/EL20--80 at the same low-opacity a2000 point.  The worst
error is a `1.159949%` flux under-correction.  Machine status is
`numerical_adoption_evidence_fail`; no successor operator or operational
domain is selected or authorized.  Generic q95 and exact generic custody
remain unresolved historical/diagnostic provenance and do not enter this
decision.

On 2026-08-02 the owner resolved `LOW-001` by selecting EL25 as the proposed
minimum for a separately preregistered confirmation.  Coordination decision
commit `f513f410b` and binding commit `8fc9263a2` subsequently resolve
`BAND-001`, `DOMAIN-001`, and `WARN-001` for that confirmation only: exact
TolTECA v1 ECSV bytes, closed q0--q75/EL25--80 support, fail-closed ALIGN
eligibility, and the bounded AM status-1 warning rule.  The independent
confirmation protocol was frozen at preregistration commit `fe3b3a1f7`, exact
parent `f4014d366`.  Its one authorized execution stopped fail-closed after 12
of 16 cases and 672 of 896 full grids, before any durable candidate metric or
maximum existed.  The stop was a two-binary64-ULP disagreement in an
unregistered `5e-17` runner consistency guard, not an AM target mismatch.
Machine status is `confirmation_invalid`; no numerical pass/fail, operator,
or domain is available.  The stopped-cache decision and full SHA-256 inventory
are recorded separately, and any corrected fresh execution requires new
owner/coordinator authority.  The completed v2 failure, EL20--80 study domain,
and 13 deterministic artifact identities remain unchanged; `OBS-001` remains
a later production-only gate.

## Read first

- `OWNER_DECISION_BRIEF.md`: concise decision, recommendation, remaining choices, and stop boundary.
- `OWNER_EL25_CONFIRMATION_DIRECTION_2026-08-02.md`: owner resolution of
  `LOW-001` and its authorization boundary.
- `AM12_EL25_CONFIRMATION_PROTOCOL.md`: immutable independent confirmation
  tuples, primary candidate, gates, schemas, and stop boundary.
- `am12_el25_confirmation_preregistration.json`: exact machine-readable
  authority, opacity/profile/elevation selection, passband, warning, coverage,
  and output contract frozen before execution.
- `AM12_EL25_CONFIRMATION_FAILURE_REPORT.md`: concise invalid-execution stop,
  preserved coverage, exact two-ULP guard discrepancy, and authority boundary.
- `am12_el25_confirmation_failure_decision.json`: complete machine-readable
  cache inventory, warning evidence, coverage, and failure provenance; it
  intentionally contains no partial candidate metric.
- `AM12_SUCCESSOR_ADOPTION_STUDY_REPORT.md`: frozen v2 numerical result.
- `AM12_SUCCESSOR_ADOPTION_STUDY_EXECUTION_RECORD_2026-08-01.md`: P1 check,
  excluded-v1, canonical-v2, and deterministic-replay lineage.
- `REGENERATION_SPEC.md`: versioned raw-generation, legacy-fit, operator, and validation contract.
- `LOCAL_PROVENANCE_INVENTORY.md`: read-only search record, positive findings, negative result, and digests.
- `owner_input_request.json`: exact machine-readable request limited to unresolved facts.
- `owner_supplied_manifest.schema.json`: fail-closed schema for the complete owner decision response.
- `governance_manifest.json`: owner authorization boundary, bound v2 result, and decision-contract identities.
- `regeneration_manifest.json`: completed-study/no-adoption state conforming to `atmosphere_regeneration_manifest.schema.json`.
- `NATIVE_REGENERATION_REPORT.md`: exact 180-case copied-suite native reproduction.
- `FREQUENCY_RESOLUTION_REPORT.md`: R1 10/5/2/1-MHz convergence result.
- `COPIED_AM_FOLLOWUP_REPORT.md`: copied-suite custody, identity, and C1 stress evidence.
- `H2O_SCALE_HYPOTHESIS_REPORT.md`: canonical all-direct P1 ranks and limitations.

## Decision state

The owner direction remains evaluation, not adoption or operator selection:

1. **Completed study -- separately versioned AM 12.2 successor:** 155 direct
   P1 training grids and 240 new full-grid holdouts produced 23,040
   band-integrated comparison rows across two lanes, two operators, primary
   TolTECA ECSV passbands, representative FTS challengers, and four source
   indices.
2. **EL25 confirmation preregistered on 2026-08-02:** `LOW-001` is resolved as
   `confirm_el25_minimum_with_preregistered_evidence`.  The 25-degree floor is
   confirmation-only support, not an operational domain.  The later exact
   passband/domain/warning decision authorizes one independent study.  Its
   uniform integer-degree EL25--80 lattice and two exact opacity trisections
   per q interval produce 896 new direct grids; the post-result v2 EL25 slice
   is not reused as confirmation truth.  The authorized execution stopped at
   672 grids before complete evidence, so it is `confirmation_invalid` rather
   than a numerical pass or fail.  A replacement execution is not authorized.
3. **Historical provenance retained, nonblocking:** generic q95 datafile ID 461 and exact generic generator/profile/grid custody remain unresolved. No copied product is substituted for a generic product, and the successor must not be represented as a historical regeneration.

The 13 deterministic study artifacts are bound.  All structural and coverage
gates pass, but the numerical result is
`numerical_adoption_evidence_fail`: primary maximum `1.159949%`, challenger
maximum `1.101467%`, no eligible candidate, empty conditional ranking, and
null recommendation.  Above q25 the simplest fixed-DJF25/piecewise-linear
candidate is within `0.288111%` of direct AM truth.  A post-result EL25--80
slice falls below one percent but cannot retroactively narrow the frozen
domain.  It motivates the now-owner-selected EL25 confirmation path only.  The
FTS-versus-ECSV direct-truth difference reaches `3.474613%`.  The owner has
selected the exact content-bound TolTECA v1 ECSV set for the bounded
confirmation; the selection is not a universal production color-correction
contract.  Exact operational opacity/elevation endpoints remain unapproved,
and `OBS-001` remains production-only.

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

`run_am12_successor_adoption_study.py` preserves the final successor evidence:

- `am12_successor_adoption_manifest.json` binds all 12 child artifacts;
- `am12_successor_holdout_execution_context.json` binds exact runner,
  protocol, clarification, erratum, P1, AM executable, profiles, passbands,
  host, grid, and execution parameters;
- `am12_successor_p1_run_inventory.csv` contains 155 validated training grids;
- `am12_successor_holdout_run_inventory.csv` contains 785 scale-search and 240
  full-grid executions;
- `am12_successor_holdout_rows.csv` contains all 23,040 expanded truth rows;
- `am12_successor_decision.json` records the failed numerical gate and null
  recommendation.

The v2 execution context SHA-256 is
`05dd063ca433b79ab3e2c2fa469e0976802a69502232aae2ddc58121d1a7ccff`.
Cache-only replay reproduced all 13 artifacts byte-for-byte without launching
AM.  The predecessor v1 context
`f0acb32cd43fd0bd128a06ab8d7e354bc6a6c1389d6d0794db716753d03f85c8`
is retained and excluded after the documentary P1-stage lookup failure.

Earlier copied-AM, legacy-anchor, and P1 manifests remain byte-frozen with
their pre-execution `pending_results`/`unbound_pending_study_results` fields.
Those fields identify the state when each sub-study was frozen.  The f401
`governance_manifest.json`, `regeneration_manifest.json`, and
`owner_input_request.json` are now also retained as predecessor decision-state
records: their `BAND-001`/`DOMAIN-001`/`WARN-001` pending fields describe the
state before coordination decision `f513f410b` and are not rewritten.  The
content-bound coordination authority, confirmation preregistration, and
invalid-execution record supersede those pending fields for the bounded study
only without relabeling the predecessor lineage.

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

The verifier checks both JSON Schemas, the exact completed-v2 owner-decision
contract, all successor artifact and scientific invariants, CSV structure,
regenerated byte identity, package digests, phase-0/coordination records, raw
NPZs, supporting sources, and dissertation digest. It performs no network or
Unity access and makes no sibling-repository changes.

The canonical P1 cache was independently replayed without launching AM using
the exact checker bytes exported from commit `3ccf4c44b`.  The living P1
wrapper intentionally binds a later documentary deviation log and therefore
stops before cache access if pointed at that older cache.  The exact snapshot,
hashes, outcome, and reason for retaining the old bytes are recorded in
`AM12_SUCCESSOR_ADOPTION_STUDY_EXECUTION_RECORD_2026-08-01.md`.  The successor
runner's cache-only check separately validates the 155 P1 grids it actually
consumes.

The full local verifier can additionally validate the canonical native cache
and the paired P1/successor caches in cache-only mode.  The paired flags feed
P1 directly to the completed-v2 runner and do not invoke the incompatible
standalone living P1 wrapper:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/verify_package.py \
  --native-cache-dir /private/tmp/sci_cal_001_am12_2_native_matrix_context_v2_final_20260801_root \
  --adoption-p1-cache-dir /private/tmp/sci_cal_001_h2o_scale_p1_context_v3_lightweight_final_20260801_root \
  --adoption-cache-dir /private/tmp/sci_cal_001_am12_successor_adoption_v2_20260801_root
```

The successor cache can also be replayed directly:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/run_am12_successor_adoption_study.py \
  --check \
  --p1-cache-dir /private/tmp/sci_cal_001_h2o_scale_p1_context_v3_lightweight_final_20260801_root \
  --holdout-cache-dir /private/tmp/sci_cal_001_am12_successor_adoption_v2_20260801_root \
  --am-root /Users/gwilson/work_toltec/local_data/AM \
  --am-executable /private/tmp/sci_cal_001_am12_2_native_build_20260801_root/am \
  --tolteca-repo /Users/gwilson/GitHub/tolteca \
  --beammap-repo /Users/gwilson/GitHub/toltec_beammap \
  --output-dir validation/sci_cal_001_atmosphere_operator_2026-08-01 \
  --jobs 8 --omp-threads 1
```

The stopped EL25 cache is preserved and must not be resumed or reused. Its
failure decision and full digest inventory can be reproduced without AM,
passband integration, or candidate evaluation:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/record_am12_el25_confirmation_failure.py \
  --cache-dir /private/tmp/sci_cal_001_am12_el25_confirmation_v1_20260802_root \
  --check
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
