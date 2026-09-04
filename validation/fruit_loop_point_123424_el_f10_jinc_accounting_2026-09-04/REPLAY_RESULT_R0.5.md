# SCI-FRUIT EL-F10-R4 replacement replay result r0.5

Test ID: `SCI-FRUIT-EL-F10-R4-NOISE-PASS-LEDGER-REPAIR-R0.1`

Status: **replacement replay complete; output binding required before analysis**

The isolated local replacement replay completed successfully on 2026-09-04:

- wall time: 33.17 seconds;
- maximum resident set size: 871,006,208 bytes;
- retained root size: 78,832 KiB;
- configured threads: 1;
- `--grppiex seq`;
- no error- or critical-level log records; and
- completed absolute FRUIT iteration 5 from the same registered iteration-4
  checkpoint.

The output root is
`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.2`.
The defective `r0.1` replay remains intact.

The repaired target ledger has exactly 305 rows and 305 unique sample indices:
271 `admitted` and 34 `final_flagged`. It contains no noise-only-pass
`center_outside_map` records. The new diagnostic NetCDF receipt has the same
size and SHA-256 as the defective replay's receipt, demonstrating that total
and target accumulator content was unchanged by the ledger-only repair. The
replacement checkpoint likewise has the same size and SHA-256 as the defective
replay checkpoint.

Only ledger shape, sample identity, admission flags, and reason counts were
read for this pre-analysis verification. JINC accumulator values, response
metrics, target contribution values, and scientific summaries remain unopened
until a new output-bound registration is frozen.
