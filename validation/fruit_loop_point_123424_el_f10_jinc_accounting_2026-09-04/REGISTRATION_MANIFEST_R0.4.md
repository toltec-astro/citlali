# SCI-FRUIT EL-F10-R3 registration manifest r0.4

Test ID: `SCI-FRUIT-EL-F10-R3-FITS-ORIENTATION-READER-REPAIR-R0.1`

Status: **frozen after the replay and before continued accounting analysis**

The standing routine-defect direction and
`EL_F10_R3_ROUTINE_FITS_ORIENTATION_REPAIR_2026-09-04.md` authorize the local
reader repair and continuation without a new scientific-owner decision.

`REGISTRATION_R0.4.yaml` is 11,447 bytes with SHA-256
`822224575e56d125db3034033e5927ab3ccec0ae4e2d8d09ee131fc91d0318e5`.
All 26 files registered there passed size and SHA-256 validation before this
manifest was written. They include the immutable R0.3 registration and stopped
result, unchanged scientific test definition, twice-repaired analyzer and
focused tests, governing authorization records, retained receipt and target
ledger, replay checkpoint and map products, log, and historical comparison
products.

No additional Citlali replay is authorized or required. The exact retained
receipt, target ledger, and comparison products remain unchanged.

## Sole R3 reader repair

Every two-dimensional receipt plane is reversed once along its column axis on
read, matching the explicit internal-matrix-to-FITS transformation in
`include/citlali/core/utils/fits_io.h:add_typed_hdu`. Before the repair, a
read-only diagnosis established exact equality after this transform for the
finalized signal, finalized formal coefficient, captured formal coefficient,
captured empirical coefficient, and normalization support.

A focused regression test covers the orientation transform. The focused file
passed 8 tests, and the complete baseline plus FRUIT-loop Python suite passed
all 252 tests. Ruff, byte compilation, and `git diff --check` also passed.

Every scientific equation, gate, bound, support rule, region, trigger, input,
and claim limit remains unchanged from R0.3.
