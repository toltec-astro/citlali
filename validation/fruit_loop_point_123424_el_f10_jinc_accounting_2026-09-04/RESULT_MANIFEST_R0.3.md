# SCI-FRUIT EL-F10-R2 result manifest r0.3

Test ID: `SCI-FRUIT-EL-F10-R2-NETCDF-SCALAR-READER-REPAIR-R0.1`

Status: **stopped; no scientific result products written**

| File | Bytes | SHA-256 | Role |
|---|---:|---|---|
| `REGISTRATION_R0.3.yaml` | 10,392 | `957b277518f3267b61c47edacfeaca69b4fc16ba72b8e23fb254ce7ef3573435` | exact pre-analysis registration |
| `REGISTRATION_MANIFEST_R0.3.md` | 1,775 | `58bdca679664380b54923019752f753bde2ffa6c1dcdd96c07832fc9c122a09e` | registration explanation |
| `R2_ANALYSIS_ABORT_R0.3.json` | 1,362 | `800cb374b4e0ba9f5ce1701fe41f58cfec104ac5326893f29cfb260d465a287a` | machine-readable gate stop and orientation diagnosis |
| `EXECUTION_RESULT_R0.3.md` | 2,101 | `0db2e6f87760309441693532ddb6366530bf864f415171cf7cccefdb1112da33` | owner-facing stopped result |

The registered analyzer passed scalar parsing, all registered-file identities,
ordinary-map neutrality, and checkpoint compatibility. It stopped at the exact
total-accumulator closure gate because it compared internal-orientation
receipt planes directly with column-reversed FITS output planes.

Read-only diagnosis demonstrated bitwise equality for signal, finalized and
captured formal coefficient, captured empirical coefficient, and support after
applying the FITS writer's documented column reversal. The delegated R3 repair
is governed by
`EL_F10_R3_ROUTINE_FITS_ORIENTATION_REPAIR_2026-09-04.md` and the standing
routine-defect direction. No Citlali replay or scientific-gate change occurred.
