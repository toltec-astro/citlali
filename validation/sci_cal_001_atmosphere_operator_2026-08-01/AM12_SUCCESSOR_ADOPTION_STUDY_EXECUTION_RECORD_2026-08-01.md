# SCI-CAL-001 AM12 successor adoption-study execution record

Date: 2026-08-01
Disposition: v2 completed and reproduced cache-only; v1 excluded.

## Prerequisite P1 validation

The current package wrapper intentionally differs from the wrapper frozen by
canonical P1.  Running the P1 checker directly from the living package first
stopped before touching the cache with:

```text
frozen protocol size mismatch for FOLLOWUP_STUDY_DEVIATION_LOG.md: 2405 != 2066
```

This was a documentary-version mismatch, not a cache or numerical failure.
Commit `3ccf4c44b` was exported read-only to
`/private/tmp/sci_cal_001_p1_check_snapshot_20260801_root.Q88xUK` and the
canonical checker was run there against
`/private/tmp/sci_cal_001_h2o_scale_p1_context_v3_lightweight_final_20260801_root`.
The snapshot runner SHA-256 was
`caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c`;
the snapshot deviation-log SHA-256 was
`a3df86366c7869579b3255d9ea8f95cf6827e78018e0a2a83a1640360be1b036`.
The checker recomputed all 25 constructed and all 25 direct profiles and
ended with:

```text
verified SCI-CAL-001 H2O-scale hypothesis artifacts
```

No AM process was launched and the P1 cache was not modified.

## Excluded v1 launch

The first successor launch used committed runner `b9f4f48b9`, SHA-256
`f61a8f94edb0fe0e71c96f76cff528f3aaf0cdaab3d14733999ac934c606e96f`,
and fresh cache
`/private/tmp/sci_cal_001_am12_successor_adoption_v1_20260801_root`.
It completed the raw holdout matrix but stopped while loading P1 training
evidence because the runner requested the wrong documentary P1 stage.  The
failure and exact correction are frozen in
`AM12_SUCCESSOR_ADOPTION_STUDY_EXECUTION_ERRATUM_2026-08-01.md`.  The v1 cache
is retained, excluded, and never reused by v2.

## Canonical v2 execution

The bounded correction is commit `6532252f0`.  An independent pre-launch
audit approved exact runner SHA-256
`ace8e08a037535260b6b1d889f83dbf722ffc932e05bc1f7f83f0565ef0ff47c`
after all 155 P1 training grids loaded as exactly 93 general-stage and 62
selected-stage records.

The v2 cache path was initially absent.  The executed command was:

```text
/Users/gwilson/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/run_am12_successor_adoption_study.py --run-holdouts --p1-cache-dir /private/tmp/sci_cal_001_h2o_scale_p1_context_v3_lightweight_final_20260801_root --holdout-cache-dir /private/tmp/sci_cal_001_am12_successor_adoption_v2_20260801_root --am-root /Users/gwilson/work_toltec/local_data/AM --am-executable /private/tmp/sci_cal_001_am12_2_native_build_20260801_root/am --tolteca-repo /Users/gwilson/GitHub/tolteca --beammap-repo /Users/gwilson/GitHub/toltec_beammap --output-dir /Users/gwilson/.codex/worktrees/cdd5/citlali-refactor/validation/sci_cal_001_atmosphere_operator_2026-08-01 --jobs 8 --omp-threads 1
```

It ended successfully:

```text
Wrote 13 deterministic artifacts with 240 newly executed holdouts.
```

The v2 execution context is 17,531 bytes with SHA-256
`05dd063ca433b79ab3e2c2fa469e0976802a69502232aae2ddc58121d1a7ccff`.
One frozen context key is historically misnamed:
`imported_canonical_p1_runner.canonical_p1_context_sha256` contains
`caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c`,
which is the imported canonical P1 **runner** SHA-256.  The actual canonical P1
execution-context SHA-256 is separately and correctly recorded by
`p1_execution_context_sha256` as
`05148050e96e73577ec75be525b026b5bf37bbd2a8753f8e3702fc0b6dfb2bee`.
The frozen context bytes are preserved; the misnamed key must never be treated
as P1 context identity.
It was created at `2026-08-01T22:19:44-0400`; the last raw-output-directory
modification was `2026-08-01T22:35:54-0400`.  The cache contains eight scale
traces, 1,025 raw outputs, 1,025 execution sidecars, 21,416 AM spectral-cache
files, and zero failed-attempt files.  The run inventory resolves those 1,025
executions as 785 midpoint scale-search anchors and 240 full truth grids.

## Mandatory deterministic replay

The identical command was rerun with `--check` in place of
`--run-holdouts`.  It ended:

```text
Validated 13 deterministic artifacts cache-only; no AM process executed.
```

As an additional excluded-predecessor diagnostic, v1 and v2 contain the same
1,025 physical requests and argv.  All 1,025 numeric-text SHA-256 values and
all 1,025 normalized-output SHA-256 values agree.  Raw combined-output hashes
differ in 935 cases because raw AM output retains volatile runtime/cache
lines; these fields are removed only by the frozen normalization algorithm.
This comparison does not admit v1 into the decision evidence.

No Unity access, network access, Citlali application edit, sibling-repository
edit, repair implementation, re-audit, or coordination-registry edit occurred.
