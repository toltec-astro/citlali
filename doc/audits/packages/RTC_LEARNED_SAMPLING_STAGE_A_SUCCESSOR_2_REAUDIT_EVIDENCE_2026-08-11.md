# SCI-RTC-001 Stage A successor-2 independent re-audit evidence

Date: 2026-08-11

Disposition supported: **RETURN FOR REPAIR**

Companion report:
[RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_2026-08-11.md](./RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_2026-08-11.md)

Finding register:
[RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_FINDINGS_2026-08-11.csv](./RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_FINDINGS_2026-08-11.csv)

## 1. Audit protocol and immutable identities

The mandatory READY checkpoint was completed before substantive application
source, diff, and test exposure. The coordinator accepted the checkpoint before
the audit branch or any artifact was created.

The auditor used the fresh worktree
`/Users/gwilson/.codex/worktrees/fce6/citlali-refactor`, distinct from repair
task `019ff09e-e42f-7630-9535-0bc048afb773` and its worktree
`/Users/gwilson/.codex/worktrees/24fb/citlali-refactor`. The approved audit
branch is
`codex/reaudit-rtc-learned-sampling-stage-a-successor-2-20260811`.

### 1.1 Candidate identity

| Identity | Value | Independent check |
| --- | --- | --- |
| Live origin ref | `refs/heads/codex/repair-rtc-learned-sampling-stage-a-successor-2` | `git ls-remote origin ...` returned the candidate SHA on 2026-08-11 |
| Local candidate/HEAD before audit artifacts | `cbb2fd767e0676906d1413ae84022270bee1a667` | `git rev-parse HEAD` |
| Parent | `66c96757164af2c83ee1449d00fea30d131a7e3f` | `git rev-parse HEAD^` |
| Candidate tree | `4727864c7ca4f078649fcf6473a7225d5d3aa9f8` | `git rev-parse HEAD^{tree}` |
| Binary patch SHA-256 | `d1521fbc0a5afdfcfa61b41c57ba483b1d69969a45115829d2f8d973a51c9c39` | `git diff --binary <parent> <candidate> \| shasum -a 256` |

The final live-origin read returned:

```text
cbb2fd767e0676906d1413ae84022270bee1a667 refs/heads/codex/repair-rtc-learned-sampling-stage-a-successor-2
```

The candidate's 23 changed paths are:

```text
CMakeLists.txt
doc/REFACTOR_STATUS.md
doc/RTC_LEARNED_SAMPLING_STAGE_A_METRICS_2026-08-09.md
include/citlali/core/engine/detail/rtcdiag_output_impl.h
include/citlali/core/pipeline/initial_observation_setup.h
include/citlali/core/pipeline/reduction_observation_inputs.h
include/citlali/core/pipeline/reduction_observation_pipeline.h
include/citlali/core/pipeline/rtc_learned_sampling_metrics.h
include/citlali/core/pipeline/rtcdiag_netcdf.h
include/citlali/core/pipeline/rtcdiag_scan_summary.h
include/citlali/core/pipeline/telescope_pointing.h
include/citlali/core/pipeline/timestream_alignment_helpers.h
include/citlali/core/pipeline/timestream_alignment_state.h
include/citlali/core/timestream/rtc/rtcproc.h
include/citlali/core/utils/netcdf_io.h
tests/test_config_scaffold.cpp
tests/test_ordered_writer.cpp
tests/test_rtc_learned_sampling_metrics.cpp
tools/baseline/test_validate_product_contract.py
tools/baseline/validate_product_contract.py
validation/product_contracts.json
validation/rtc_learned_sampling_stage_a_example_2026-08-09/candidate_metrics.csv
validation/validation_profiles.json
```

This is an application-code candidate. Audit edits were restricted to the three
approved `handoff/` documentation artifacts.

### 1.2 Frozen authority objects

The following bytes were read directly from the named Git objects and hashed:

| Object | SHA-256 |
| --- | --- |
| `3fe0aa30eaa0d8848dbb39eb720457326c0b43ba:doc/audits/packages/SCI-RTC-001_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REPAIR_HANDOFF_2026-08-11.md` | `60c3237752ba6195b80223e05f3b41e536a6e13263ed2c5a138c97575a1572e5` |
| `3fe0aa30eaa0d8848dbb39eb720457326c0b43ba:doc/audits/proposals/SCI-RTC-001_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REPAIR_FINDING_LEDGER_2026-08-11.yaml` | `7721535ef8f6af34a6f587add155f5de6f22ce322c8f981a36aa2a32abbd7a50` |
| `923eca42a8f892a18774df8f483dd78404807d59:handoff/RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REAUDIT_2026-08-10.md` | `a4f5a662be4b4a7d9569152dd6220444b45eb490a523a74cf5367e639c987214` |
| `923eca42a8f892a18774df8f483dd78404807d59:handoff/RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REAUDIT_EVIDENCE_2026-08-10.md` | `d368c9f3f699c60c1f27a33d53db1f64b2b932aaf03319969646abb6ffe6133f` |
| `923eca42a8f892a18774df8f483dd78404807d59:handoff/RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REAUDIT_FINDINGS_2026-08-10.csv` | `ef42bb65c1763f8453e9a26c0452cfcc3665234767f1b5176f8f1a8788459b9b` |

No frozen identity mismatch was found.

## 2. SRA-001/SRA-002 lifecycle and total-intensity boundary

### 2.1 The source carrier is captured after interpolation

The relevant production sequence is:

1. `include/citlali/core/pipeline/reduction_observation_inputs.h:71-94`
   resets the carrier, calls `load_and_point_telescope_data`, and only then calls
   `capture_telescope_sampling_motion_carrier`.
2. `include/citlali/core/pipeline/telescope_pointing.h:32-44` copies the current
   `engine.telescope.tel_data` into the carrier.
3. `include/citlali/core/pipeline/telescope_data_loading.h:14-24` routes loading
   through telescope alignment.
4. `include/citlali/core/engine/detail/todproc_alignment_impl.h:139-143` and
   `:256-257` interpolate telescope data to the common detector grid.
5. `include/citlali/core/pipeline/timestream_alignment_helpers.h:365-387`
   replaces every telescope column, including `TelTime`, with the interpolated
   result.

The carrier authority string at
`include/citlali/core/pipeline/timestream_alignment_state.h:21-24,37-41` claims
`source-telescope-rows-before-detector-grid-interpolation`. That identity is
false for the production capture. The candidate diff specifically removed the
earlier capture from `initial_observation_setup.h` and performs capture after
alignment in both initial and subsequent-observation paths.

Consequences include loss of source-row gaps, source-native scan boundaries,
and the approved empirical source-row motion/v95 authority.

### 2.2 Supported total-intensity preparation retains stale HWPR dependence

The production rtcdiag request explicitly sets total-intensity analysis in
`include/citlali/core/engine/detail/rtcdiag_output_impl.h:38-41`, and enabled
polarimetry is rejected through the typed capability boundary in
`include/citlali/core/engine/detail/rtc_config_impl.h:26-44`. Those are positive
controls.

However, the aligned telescope data that becomes the source carrier is produced
through branches on legacy `engine().calib.run_hwpr` at
`include/citlali/core/engine/detail/todproc_alignment_impl.h:85-99,113-149,259-265`.
Total-intensity HWPR loading intentionally does nothing at
`include/citlali/core/pipeline/hwpr_loading.h:16-34`, while the field is not
initialized at declaration in `include/citlali/core/engine/calib.h:40-41`.
The existing test at `tests/test_config_scaffold.cpp:7188-7196` explicitly
expects a retained true value to remain true for a non-polarized observation.

Thus the supported path indirectly depends on precisely the stale legacy state
prohibited by owner decision `RTC-STAGE-A-HWPR-UNSUPPORTED`.

The explicit HWPR-dependent diagnostic fallback also occurs after cadence, FIR,
and beam checks at
`include/citlali/core/pipeline/rtcdiag_scan_summary.h:542-625`. An explicitly
HWPR-dependent request with another invalid prerequisite does not deterministically
return `unsupported_hwpr`. The focused test covers only valid prerequisite
values.

### 2.3 Same-observation identity is not fully consumed

`include/citlali/core/pipeline/timestream_alignment_state.h:52-81` stores and can
check observation index, external `obsnum`, and telescope path. Production
creation at `include/citlali/core/engine/detail/rtcdiag_output_impl.h:98-103`
checks only availability and `obsnum`; it does not bind index/path or call the
complete helper. Helper-only reset/match tests do not exercise production
consumption or stale-context permutations.

## 3. SRA-003 exact validity and exclusion accounting

### 3.1 Exact context is evaluated on the realized output grid

`include/citlali/core/timestream/rtc/rtcproc.h:1184-1245` applies the realized
RTC downsampling factor to samples, flags, telescope data, and guard state. The
pointing, lali, and Beammap output paths pass that already-processed PTC object
to `append_diag_to_netcdf`:

- `include/citlali/core/engine/detail/pointing_timestream_output_impl.h:22-32`;
- `include/citlali/core/engine/detail/lali_timestream_output_impl.h:19-25`;
- `include/citlali/core/engine/detail/beammap_timestream_pipeline_impl.h:119-145`.

The final exact writer at
`include/citlali/core/timestream/rtc/rtcproc.h:5491-5619` derives time, flags,
guard, rows, and detector cells from that PTC object. At `:5689-5719` it sets
the hypothetical domain size to the PTC output row count and applies every
candidate factor again while still labeling the cadence native.

For an applied production factor `R > 1`, candidate factor 1 therefore has
roughly `N/R` inputs instead of `N`; candidate factor `M` has roughly
`N/(R*M)` outputs instead of `N/M`. The production OR-reduced flags and FIR
context cannot reconstruct native counterfactual validity. This final write
also replaces the earlier native-grid summary values constructed by
`include/citlali/core/pipeline/rtcdiag_scan_summary.h:788-800`.

### 3.2 Scan guards and complete categories are missing

Per-scan guarded source rows and `eligible_grid_by_scan` are constructed at
`include/citlali/core/pipeline/rtcdiag_scan_summary.h:221-252`. The final writer
does not consume them. Instead it performs only time-to-interval lookup at
`include/citlali/core/timestream/rtc/rtcproc.h:5554-5573` and serializes input
boundary context as zero at
`include/citlali/core/pipeline/rtcdiag_scan_summary.h:1108-1123`.

Source scan-boundary exclusions can therefore become supported or internal
gaps. A nonfinite `TelTime` also fails interval lookup first and is labeled
`internal_gap`, before the nonfinite-input check at `rtcproc.h:5605-5609`.

Only category counts are defined and written at
`rtcdiag_scan_summary.h:1108-1123,1213-1229` and
`rtcproc.h:5532-5549,5636-5647`. The frozen required per-category fractions do
not exist; only the aggregate `rtc_sampling_full_fraction` is present.

Guard-mask and residual science-flag categories remain separate, and detector
cell sums plus temporal “at least one production-valid detector” logic at
`rtc_learned_sampling_metrics.h:868-973` are internally consistent. They are
computed on the wrong and incompletely guarded domain.

In isolation, the motion helper truthfully reports
`insufficient_source_motion_rows` for fewer than two rows,
`no_valid_source_motion_intervals` when all intervals are unusable,
`no_guarded_source_motion_support` when a scan is too short after its one-row
guards, and `scan_unusable_for_applied_rtc_operator`/`no_complete_context` when
no applied output has complete context. These are the intended short-scan and
unusable-data semantics, but the mislabeled post-interpolation carrier and final
PTC-domain recomputation prevent them from establishing the frozen production
outcome.

## 4. SRA-004 resource and exact enumeration evidence

### 4.1 Work undercount counterexample

`rtc_sampling_phase_zero_coherent_response_at` computes one unaliased response
and then all `M` folded images at
`include/citlali/core/pipeline/rtc_learned_sampling_metrics.h:642-667`.
Consequently the numerical term for factor `M` is

```text
L * ((M + 1) * (6Q + 1) + 2Q + 4)
```

but preflight at `rtc_learned_sampling_metrics.h:1113-1135` charges

```text
L * (M * (6Q + 1) + 2Q + 4).
```

An exact-header probe compiled this call:

```cpp
auto r = citlali::pipeline::rtc_sampling_resource_preflight(
    {2}, {1}, {100001}, {1}, {1});
```

with C++20 and the repository header. Its output fields were
`table_available`, integer table status, integer table reason,
`estimated_actual_work_units`, and `estimated_rtcdiag_bytes`:

```text
1 16 0 360703607 801016
```

For `Q=257`, `6Q+1=1543`, `2Q+4=518`, `L=100001`, `D=N=1`, the candidate
charges factor-1 numerical/context plus factor-2 numerical/context as

```text
3607 * 100001 = 360,703,607.
```

The actually invoked path is

```text
5150 * 100001 = 515,005,150,
```

which exceeds the configured `500,000,000` work limit while the table remains
available.

### 4.2 Other unbounded work and storage

- `rtc_sampling_source_interval_at_time` linearly scans intervals from the
  beginning at `timestream_alignment_helpers.h:22-39`; production calls it once
  for every row at `rtcproc.h:5554-5573`. The potentially
  `O(N_rows * N_intervals)` work is not charged.
- `cell_categories` allocation for rows times detectors at `rtcproc.h:5550-5552`
  is not preflighted.
- Storage preflight at `rtc_learned_sampling_metrics.h:1248-1274` charges
  rectangular candidate cells and FIR bytes, but not nine full source-interval
  arrays serialized at `rtcdiag_scan_summary.h:971-1040`, runtime category
  state, or raw-manifest bytes.
- `rtcdiag_netcdf.h:146-154,307-327` reads and embeds the entire supplied
  manifest without a prior size bound. The actual file-size guard is evaluated
  only after write and close at `:337-342`.

### 4.3 Exact `M_max` overflow identity

`rtc_sampling_candidate_mmax` returns `-1` when a finite floor exceeds
`INT_MAX` at `rtc_learned_sampling_metrics.h:1042-1056`; production maps this to
`invalid_cadence` at `rtcdiag_scan_summary.h:634-642`. A probe with
FWHM `8.48 arcsec`, native cadence `1e10 Hz`, and speed `1 arcsec/s` returns
`-1`; the exact authority formula gives `84,800,000,000`. The exact identity
and resource-limit status are lost.

### 4.4 Positive numerical controls

- Stored fixed FWHM values are 4.66, 5.94, and 8.48 arcsec. Independent
  diffraction calculations yield 4.664884858, 5.937126183, and 8.481608833.
- Isolated motion thresholds and empirical eligible-interval v95 are correct.
- Factor 1 is always included for ordinary finite ranges. At 488.28125 Hz and
  1 arcsec/s, exact candidate maxima are 2275, 2900, and 4140.
- Realized RTC coefficients are copied; disabled filtering binds `[1]`. No
  candidate-specific FIR synthesis or suitability assertion was found.
- Independent digests are:
  - `[1]`:
    `sha256:deba79ae42e24ae0ec753e347d299187cb8a4f0cf2ef58c646846237c1fc45df`;
  - `[0.25, 0.75]`:
    `sha256:926a87b0ac0b131a6e100f3c8d2e426433844827e7560939f34d79bcd67efa33`.
- An independent identity-filter factor-2 check confirms the analytical values
  alias `0.9518498073692735`, relative amplitude 2, power 4, distortion 1,
  phase 0, and stopband rejection 0 dB are enclosed by the implementation.
- Factor 1 has zero alias and `not_applicable_no_decimation` stopband status.
- No ranking, recommendation, acceptance threshold, candidate selection, or
  Stage A use of `epsilon_alias` was found.

## 5. SRA-005/SRA-006 product, contract, and build identity

### 5.1 Conditional schema validator accepts malformed complete products

The Python validator checks only the trailing candidate dimension at
`tools/baseline/validate_product_contract.py:1046-1102`. It skips
dimension-size-to-count equality when the dimension is absent at `:1015-1026`,
and the unavailable branch does not require declared count zero. The C++ staging
validator similarly checks only a trailing candidate dimension at
`include/citlali/core/pipeline/rtcdiag_netcdf.h:205-218`.

Two complete adversarial NetCDF fixtures were built in a task-owned temporary
directory against the exact successor check:

1. Available: `n_scans=2`, `n_arrays=3`, candidate count/dimension 2; all 81
   conditional candidate variables are present but have only shape
   `[n_rtc_sampling_candidates]`, not production
   `[n_scans,n_arrays,n_rtc_sampling_candidates]`; identity bindings are bogus.
2. Unavailable: no candidate dimension or candidate arrays, but declared
   candidate count is 7.

The exact validator invocation loaded
`checks["rtc_diagnostics_stage_a_successor_v3"]` from the committed registry and
called `validate_netcdf` for each file with the three array IDs. Output:

```text
available_wrong_full_shapes_and_bindings.nc []
unavailable_nonzero_declared_count.nc []
```

Thus the executable contract accepts both contradictions.

Mechanical writer/contract comparison found 35 compact noncandidate successor
fields written by production but not required by the executable contract, plus
nine per-interval arrays that may be omitted under the frozen no-exhaustive-
replay decision. The 35 include effective/realized native cadence, all 15
requested/effective/realized filter fields, fixed beam FWHM and temporal sigma,
valid maximum speed, category precedence, resource row-byte state, and compact
source-support status/count state written at `rtcdiag_scan_summary.h:902-1066`.
Of the fields that are required, 32 identity/method/cadence scalars have no
exact value/type constraint, and nine joins have only presence/format checks.
FIR digest is not recomputed from coefficients, and raw-manifest SHA is not
recomputed from canonical bytes.

### 5.2 Embedded IDs do not identify the executable mode contract/profile

Every product embeds the generic contract ID
`sci-rtc-001-stage-a-successor-products-v1` and the point profile
`sci-rtc-001-stage-a-successor-v1` at
`include/citlali/core/pipeline/rtcdiag_netcdf.h:38-43,317-322`.

The generic contract ID is absent from the contract registry. The actual wrapper
IDs at `validation/product_contracts.json:2811-2848` are mode-specific point,
OOF, Beammap, and science contracts; corresponding profiles are mode-specific
at `validation/validation_profiles.json:483-648`. OOF, Beammap, and science
products therefore claim the point profile and no executable contract matching
their embedded contract scalar.

The schema and algorithm strings themselves are consistently `rtcdiag-v3` and
`rtc-learned-sampling-stage-a-v3`. Static inventory confirmed 81/81/81
candidate-variable names are set-equal across writer, finalizer, and contract.
Four preparing profiles exist, and their wrappers inherit the existing native
mode contracts.

The implementation also defines a finite numeric status/reason enum and a
serialized vocabulary. It is not deterministically truthful end to end:
unsupported-HWPR may be hidden by another prerequisite, nonfinite time can be
reported as an internal gap, finite large `M_max` is mapped to invalid cadence,
and the executable contract requires the vocabulary variable but does not
enforce its exact value or relationships.

### 5.3 Build/source identity can be stale

`CMakeLists.txt:67-96` snapshots full commit and dirty state only at configure
time. No build dependency tracks Git HEAD, index, or source dirty state.
Configure-clean followed by a tracked source change and an ordinary build can
therefore compile different source with the earlier SHA and `DIRTY=0`.
Runtime checks at `rtcdiag_output_impl.h:104-108` validate only those potentially
stale constants.

The audited fresh build did receive the correct macros:

```text
CITLALI_GIT_COMMIT_FULL="cbb2fd767e0676906d1413ae84022270bee1a667"
CITLALI_GIT_WORKTREE_DIRTY=0
```

That single configure/build does not prove the required committed source
identity invariant under ordinary rebuilds.

## 6. SRA-008 cadence provenance counterexample

`rtcdiag_scan_summary.h:61-95` calculates epoch-valued adjacent differences but
uses tolerance `max(1e-9, 64*epsilon*median)`, which does not account for the
representational ULP of a large Unix epoch. `TelTime` is Unix seconds by
`include/citlali/core/pipeline/observation_date.h:18-21` and
`timestream_alignment_helpers.h:414-434`.

The independent venv probe was:

```text
t = 1.7e9 + arange(32) / 488.0
```

and returned:

```text
increments [0.002048969268798828, 0.0020492076873779297]
median 0.0020492076873779297
max_deviation 2.384185791015625e-07
tolerance 1e-09
would_reject True
```

An ideal regular native grid is therefore marked
`irregular_realized_cadence`, making all dependent candidate metrics
unavailable.

Requested/effective consistency is computed only after valid realized cadence
at `rtcdiag_scan_summary.h:61-117`; an invalid realized grid hides an
independently knowable requested/effective mismatch. Frequency-derived valid
requests retain nonpositive `requested_factor` at
`rtcdiag_output_impl.h:43-51`, even though an effective factor is derived at
`raw_timestream_observation_resolution.h:76-107`; the consistency helper then
returns `unavailable_nonpositive`. Repeated calls do not fully reset realized
values and consistency strings.

## 7. SRA-009 atomic lifecycle and raw-input join

The pipeline passes fixed-name `raw_timestream_provenance.yaml` as the canonical
raw-input manifest at
`include/citlali/core/pipeline/reduction_observation_pipeline.h:30-40`.
The serialized node at
`include/citlali/core/pipeline/raw_timestream_provenance.h:147-166` contains
requested/effective configuration, observation state, and realized counts, but
no raw-input membership, paths, roles, or content hashes. It is not the frozen
canonical raw-input manifest.

`include/citlali/core/pipeline/raw_timestream_provenance_lifecycle.h:80-96`
atomically replaces that fixed sidecar before rtcdiag finalization. If later
manifest embedding, validation, sync, close, or rtcdiag publish fails, the prior
rtcdiag file is preserved but the prior manifest it references has already been
replaced. This violates prior-good-generation preservation and the required
atomic provenance join.

The failure tests at `tests/test_rtc_learned_sampling_metrics.cpp:1063-1078`
verify only prior NetCDF preservation; they do not exercise pipeline ordering,
sidecar rollback, scan-append failure, or the complete each-stage absent/prior
matrix.

Positive mechanics include same-directory unique `O_EXCL` staging, no
destructive pre-delete, sync/close before rename, cleanup of task-created
temporaries, ordered-writer completion, and refusal to append after publication
at `rtcproc.h:5803`. Those mechanics cannot provide transactionality across the
already-published fixed sidecar.

## 8. SRA-007 observe-only and independent-evidence assessment

Static diff review found no direct Stage A mutation of production science
samples, flags, timestamps, RTC/PTC/map inputs, weights, maps, cadence, FIR,
factor, or non-rtcdiag products. Optional guard capture preserves the preexisting
flag write and additional flag downsampling is a separate OR reduction.

The frozen closure requires production-boundary evidence, not only static
inspection:

- The coherent-fold expectation at
  `tests/test_rtc_learned_sampling_metrics.cpp:325-339` uses the candidate's own
  `rtc_sampling_composed_transfer` helper.
- The generic bound test at `:391-434` does not independently exercise the
  production calculation.
- The A/B test at `:756-789` captures only telescope columns; its science, RTC,
  PTC, and map vectors are unrelated locals that the production path cannot
  mutate.
- No production A/B then B/A, failed/partial observation, repeated observation,
  or permitted parallel/OpenMP permutation was found.

The green tests therefore do not demonstrate observe-only behavior at the
required boundary.

## 9. Contract and `ptcdiag` invariance controls

The pre-Stage-A existing `rtc_diagnostics` check was restored exactly: canonical
JSON SHA-256 is
`a2df72164c175d59b9875feaa21ca9d7f902ff247c287a500b1a875d7b57cc16`
in both authority commit `6cbe119a59f8915c5aecf5eaf333425dd592993d`
and the candidate. The `ptc_diagnostics` check SHA-256 is unchanged at
`df8e8169375a0baf35e7e135415d948d6b9b2e959bfd1e13be54bc711c2db061`.

Parent and candidate blob IDs are identical for the inspected `ptcdiag` source
surface:

| Path | Parent blob | Candidate blob |
| --- | --- | --- |
| `include/citlali/core/timestream/ptc/ptcproc.h` | `0436bfa814409a10d944a557be83286287e798cd` | `0436bfa814409a10d944a557be83286287e798cd` |
| `include/citlali/core/engine/detail/ptcdiag_output_impl.h` | `98a56daf629395f6265c26499c6bdae4597d8a52` | `98a56daf629395f6265c26499c6bdae4597d8a52` |
| `include/citlali/core/pipeline/ptcdiag_netcdf.h` | `f9124321508152fcd8eef0cd6f41095908b646a5` | `f9124321508152fcd8eef0cd6f41095908b646a5` |

No `ptcdiag` finding or change was introduced by this audit.

## 10. Exact deterministic gates run

### 10.1 Configure and C++ gates

| Command | Result |
| --- | --- |
| `env BUILD_TESTS=ON BUILD_TYPE=Release BUILD_DIR=/Users/gwilson/.codex/worktrees/fce6/citlali-refactor/build CMAKE_GENERATOR='Unix Makefiles' tools/macos/configure-build-dir.sh` | Pass; fresh Release test build configured. |
| `cmake --build build --target citlali_cli citlali_test citlali_safety_test -j 8` | Pass. |
| `cmake --build build --target citlali_science_map_truth_test citlali_science_map_fits_products_test -j 8` | Pass. |
| `build/tests/citlali_test --gtest_filter='RtcLearnedSamplingMetrics.*:config_scaffold.rejects_enabled_polarimetry_capability:config_scaffold.serializes_versioned_raw_timestream_provenance:config_scaffold.atomically_writes_raw_timestream_provenance:config_scaffold.raw_timestream_provenance_failure_propagates:config_scaffold.rejects_incomplete_raw_timestream_provenance'` | Pass, 34/34. |
| `build/tests/citlali_safety_test --gtest_filter='ordered_writer.*'` | Pass, 7/7. |
| `ctest --test-dir build --output-on-failure -j 8` | Pass: 653 executed, 653 passed, 0 failed; 1 disabled (`MapFitterLifecycle.ExactProductSequence`) of 654 enumerated. |

### 10.2 Python, configuration, and repository gates

| Command | Result |
| --- | --- |
| `$HOME/tolteca/bin/python -m unittest tools.baseline.test_validate_product_contract tools.baseline.test_validation_profiles` | Pass, 38/38. |
| `$HOME/tolteca/bin/python -m unittest discover -s tools/baseline -p 'test_*.py'` | Pass, 177/177. |
| `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all` | Pass: 127/127 config tests; point/oof/beammap/science mode kits; 8/8 compact compatibility; 261 covered + 17 profile-owned compact leaves, 0 gaps; all boundary audits pass. |
| `$HOME/tolteca/bin/python tools/config/validate_config_authority_inventory.py` | Pass: 15 domains valid (`external-boundary=1`, `typed-authoritative=3`, `typed-authoritative-with-adapter=11`). |
| `$HOME/tolteca/bin/python tools/config/audit_raw_timestream_execution_reads.py --json-out /private/tmp/codex_successor2_reaudit_raw_execution.json --markdown-out /private/tmp/codex_successor2_reaudit_raw_execution.md --fail-on-drift --fail-on-review` | Pass: 45 records, digest `09572da976aec89d56506394420b478426a6efbd0942c864571a8f6f311da2f8`, 0 review, no drift. |
| `$HOME/tolteca/bin/python tools/baseline/validate_validation_ledger.py` | Pass: 60 records valid. |
| `$HOME/tolteca/bin/python tools/baseline/validate_science_change_ledger.py` | Pass: 3 changes and 5 integration commits valid. |
| `$HOME/tolteca/bin/python tools/baseline/validation_profiles.py --list` | Pass: registry valid, 4 active and 12 preparing profiles. |
| `$HOME/tolteca/bin/python tools/baseline/phase5_readiness.py` | Pass; truthfully reports phase5 preparing, promotion ready false, one promotion-candidate SHA false. |
| `$HOME/tolteca/bin/python tools/refactor/audit_session_exits.py --fail-on-growth` | Pass: 710 dependencies, 0 library exits, 0 CLI exits, 0 growth. |
| `git diff --check 66c96757164af2c83ee1449d00fea30d131a7e3f cbb2fd767e0676906d1413ae84022270bee1a667` | Pass. |

Config mode-kit hashes were:

```text
point   f2d124d40ac7ad9e6351a647253050e5146659c666feed07262125e6fa5415c8
oof     414a5d16ceba8b6f9163851c139affa486f50397e1ead56fc480ed53475b76f4
beammap 75eaf79fb5ce45b383f48bbb6a4715209fbb25cb29a1ac595a3afb2a7df4e0b0
science 10095418b09100f15c90af173ee34ea7bfcf12260cec41d80f43f6f50473a347
```

All ordinary deterministic gates pass. They do not invalidate the independently
reproduced resource, epoch-cadence, schema, lifecycle, and publication
counterexamples.

## 11. Authority and stop-condition accounting

No scientific-choice, owner-interpretation, frozen-authority contradiction, or
architecture conflict requiring an audit stop was encountered. Each open issue
is a technical nonconformance to an existing decision.

The audit did not repair any defect and did not modify application,
configuration, test, build-system, validation-product, canonical coordination,
or production code. It did not push, merge, integrate, contact external parties,
access/request Unity, run a science reduction, authorize production, launch
Stage B or downstream work, or create a repair or re-audit task.
