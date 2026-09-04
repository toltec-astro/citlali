# SCI-FRUIT EL-F10-R2 registration manifest r0.3

Test ID: `SCI-FRUIT-EL-F10-R2-NETCDF-SCALAR-READER-REPAIR-R0.1`

Status: **frozen after the replay and before accounting values were opened**

The scientific owner authorized the exact no-replay scalar-reader repair in
`EL_F10_R2_BUNDLE_MANIFEST_R0.1.md`. The authorization is recorded in
`SCIENTIFIC_OWNER_EL_F10_R2_AUTHORIZATION_2026-09-04.md`.

`REGISTRATION_R0.3.yaml` is 10,392 bytes with SHA-256
`957b277518f3267b61c47edacfeaca69b4fc16ba72b8e23fb254ce7ef3573435`.
All 24 files registered there passed size and SHA-256 validation before this
manifest was written. They include the immutable r0.2 registration and result,
unchanged scientific test definition, repaired analyzer and focused tests,
R2 authorization packet, retained receipt and target ledger, replay checkpoint
and map products, log, and historical comparison products.

No additional Citlali replay is authorized or required. The exact retained
receipt, target ledger, and comparison products remain unchanged from the
hashes recorded before approval.

## Sole reader repair

The scalar helper now:

- decodes a byte string;
- calls `.item()` on a NumPy scalar; and
- returns an ordinary Python scalar unchanged.

Three focused tests cover a native string, bytes, and a NumPy numeric scalar.
The focused file passed 7 tests, and the complete baseline plus FRUIT-loop
Python suite passed all 251 tests. Ruff, byte compilation, and `git diff
--check` also passed.

The checkpoint normalization, required checkpoint-difference set, ordinary-map
bitwise comparison, total-accumulator closure, 305/34/271 sample ledger,
forward-error formula, safety factor, support rule, regions, trigger pixels,
descriptive outputs, and claim limits are unchanged from r0.2.
