# SCI-FRUIT EL-F2 — Input-Binding Correction r0.2

Decision candidate: `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.2`

Status: **owner-review correction; no replacement run is authorized**

## What went wrong

The r0.1 packet said the KIDs fit-report path was unused. That was incorrect.
The first trajectory stopped before iteration 0 because Citlali requires one
processed tune fit report for every detector network. No map or checkpoint was
created, so there is no scientific result.

The failed output and log are preserved. This consumes the first of the two
allowed environmental replacements.

## The correction

The user retrieved the matching 12 fit reports. Every file identifies
observation 123424, sub-observation 0, tune scan 1, and its expected network.
Their exact hashes are in `FITREPORT_INPUT_INVENTORY_R0.2.md`.

One new overlay changes only:

```yaml
kids:
  solver:
    fitreportdir: /Users/gwilson/work_toltec/local_data/fruit-development/point-123424/input/fitreports/
```

The overlay is applied after the r0.1 `COMMON_LOCAL.yaml`, replacing only the
bad local directory path. It does not change the KIDs model, weights, raw
data, APT, telescope data, FRUIT recurrence, alpha values, injections,
terminal iterations, scientific metrics, thresholds, run order, or resource
limits.

The executable and analysis frozen before the failed attempt remain unchanged.
The 10 focused analyzer tests still pass. Before a replacement run, the new
overlay and 12 fit reports must be copied or referenced from the setup,
rehashed, and recorded in a new r0.2 freeze record.

## What approval permits

Choice A permits the failed first trajectory to be replaced and the original
four valid primary trajectories to proceed in the same BAAB order. One
environmental replacement remains. Any unfavorable scientific outcome is
retained and may not be rerun. The conditional exact restart replay remains
allowed only after a promising primary result.

Nothing else in the original EL-F2 question changes. The r0.1 owner-review
proposal and analysis manifest remain the authority for the scientific and
performance comparison.

## Owner choices

### Choice A — Correct the path and resume (recommended)

Approve `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.2` exactly as
bound by `EL_F2_BUNDLE_MANIFEST_R0.2.md`.

### Choice B — Stop EL-F2

Retain the failed attempt and do not run observation 123424.

Neither choice qualifies a method or APT, changes production defaults, starts
Gate D or Stage B, establishes historical superiority, or authorizes Unity
work.
