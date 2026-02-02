Citlali Debug Notes (2026-02-02)
================================

Context
-------
Working on PCA (processed_time_chunk clean) removal counts and several recent segfaults.
Target data: pointing reduction in `~/work_toltec/local_data/citlali_dev/ss_point`.
Latest `pointings.out` shows mostly "removing 0 largest eigenvalue(s)" with occasional huge removals
(~145–236), which is inconsistent with expected 5–15 removals.

Key Findings
------------
1) PCA stddev cut behavior:
   - With `stddev_limit: 1.5` and `n_calc: 0` (full spectrum), the stddev cut is unstable.
   - It can remove 0 for many groups and then wipe hundreds for others.
   - This happens because the cut is computed over the *entire* eigen spectrum (long tail).

2) When eigenvalues were inspected for chunk 0:
   - File: `~/work_toltec/local_data/citlali_dev/ss_point/redu03/130191/raw/toltec_commissioning_pointing_130191_stats.nc`
   - Variable: `evals_nw_0_chunk_0` (shape `(12, 128)`)
   - Eigenvalues look reasonable; threshold too high at stddev_limit=2.5 yielded 0 removals.
   - Empirically, stddev_limit ~1.4–1.6 gives ~9–11 removals *if* the spectrum is limited to top N.

Recommended Next Step
---------------------
Config-only fix (no code changes):
  - Set `n_calc: 128` (or 256) and keep `stddev_limit: 1.5`.
  - This caps the spectrum used for the stddev cut and yields ~5–15 removals for most groups.

If you want `n_calc: 0`, code should be changed to:
  - compute the stddev cut only on the top N eigenvalues (e.g., 128)
  - or enforce min/max removal count per group

Commands used to inspect eigenvalues
------------------------------------
```
source ~/toltec/bin/activate
python - <<'PY'
import netCDF4 as nc
import numpy as np
path = "/Users/wilson/work_toltec/local_data/citlali_dev/ss_point/redu03/130191/raw/toltec_commissioning_pointing_130191_stats.nc"
f = nc.Dataset(path)
arr = f["evals_nw_0_chunk_0"][:]  # (groups, n_eigs)
for g in range(arr.shape[0]):
    row = arr[g]
    finite = row[np.isfinite(row)]
    print(g, finite.min(), np.median(finite), finite.max(), finite[:10])
f.close()
PY
```

Relevant code changes already made (summary)
-------------------------------------------
1) PCA logging safety:
   - In `include/citlali/core/timestream/ptc/ptcproc.h`, debug logs no longer dump full evec matrices.
     Now logs only eigenvalue head and evec shape to prevent segfaults.

2) Guard against empty APT header:
   - In `include/citlali/core/engine/engine.h`, APT header write now handles empty `apt_filepath`.

3) Guard against MEAN_TAU issues:
   - In `include/citlali/core/engine/engine.h`, MEAN_TAU now checks `TelElAct` and tau map safety.

4) Prevent stats netCDF segfault:
   - In `include/citlali/core/engine/engine.h`, evals output is skipped unless `n_calc > 0`
     and evals are non-empty. Also pads/truncates safely when writing.

5) Hard stop when `timestream.enabled` is false:
   - In `include/citlali/core/engine/engine.h`, added explicit error+exit for `timestream.enabled: false`.
     This avoids confusing runs where mapmaking/output proceeds without TOD processing.

Notes on config
---------------
In the current `70_reduce.yaml` under ss_point:
  - `timestream.enabled: false` is invalid for a full pipeline; now triggers error and exit.
  - Use `"null"` (string) rather than YAML `null` for paths if needed by code comparisons.

What to check next
------------------
1) Re-run with:
   - `processed_time_chunk.clean.n_calc: 128`
   - `processed_time_chunk.clean.stddev_limit: 1.5`
2) Confirm log shows consistent removals (5–15).
3) If still unstable, implement per-group adaptive threshold or min/max removal rule.
