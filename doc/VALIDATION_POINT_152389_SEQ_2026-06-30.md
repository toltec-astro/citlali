# Point 152389 Deterministic Refactor Validation - 2026-06-30

## Purpose

Record the first behavior-preserving validation gate for the structural
refactor branch. This case uses deterministic `seq` execution so refactor
changes can be compared against the protected `gw_dev` Citlali checkout without
OpenMP run-to-run drift.

## Local Data Layout

- Downloaded validation root:
  `/Users/gwilson/work_toltec/local_data/2026-refactor/point`
- Protected baseline outputs:
  `citlali/reduced/redu02` and `citlali/reduced/redu03`
- Refactor outputs:
  `refactor/reduced/redu02` and `refactor/reduced/redu03`
- Shared data:
  `data`
- Shared APTs:
  `apts`

The matching Unity validation root was `${HOME}/c2025t/2026-refactor`.

## Execution Mode

- `parallel_policy: seq`
- `n_threads: 1`
- Baseline executable: protected `citlali_dev/citlali/build` comparison build
- Refactor executable: `citlali_dev/citlali_refactor/build`
- Refactor version checked before validation:
  `v4.0.0-422-ga47705f2`
- Refactor commit checked before validation:
  `a47705f2`
- kids dependency reported by executable:
  `04088da`

## Run-to-Run Determinism

Baseline `redu02` and `redu03` matched for generated configs, learning-table
counters, ECSV values, FITS arrays, and diagnostic netCDF variables. The
learning counters were:

- `sample_masks=2944`
- `detector_penalties=4`
- `high_weight_detectors=120`
- `map_pixel_outliers=24`
- `busy_network_summaries=132`

Refactor `redu02` and `redu03` showed the same deterministic behavior.

## Baseline vs Refactor Result

Baseline `redu02` and refactor `redu02` matched under the deterministic
manifest policy with `atol=2e-8` and `rtol=1e-10`.

Observed differences were floating roundoff only:

- FITS `a1100` signal max absolute difference:
  `7.816973734975363e-10`
- FITS `a1400` signal max absolute difference:
  `4.052935764775612e-11`
- FITS `a2000` signal max absolute difference:
  `6.556888365594205e-11`
- Largest diagnostic netCDF max difference:
  `1.674743543844670e-08`

The point-source ECSV and learning counters were identical.

## Comparator Command

```bash
tools/baseline/compare_deterministic_manifests.sh \
  /private/tmp/point_152389_citlali_seq_redu02_manifest.json \
  /private/tmp/point_152389_refactor_seq_redu02_manifest.json
```

Expected result:

```text
manifests match
```

## OMP Note

The earlier OMP/14 runs did not provide a deterministic gate. Baseline and
refactor both showed run-to-run drift in learning-table row counts, sample mask
counts, and FITS products. OMP nondeterminism should be fixed in a separate
branch; this structural refactor should use the `seq`/one-thread case as its
first correctness gate until that work lands.
