# SCI-FRUIT EL-F10-R3 result manifest r0.4

Test ID: `SCI-FRUIT-EL-F10-R3-FITS-ORIENTATION-READER-REPAIR-R0.1`

Status: **stopped; no scientific result products written**

| File | Bytes | SHA-256 | Role |
|---|---:|---|---|
| `REGISTRATION_R0.4.yaml` | 11,447 | `822224575e56d125db3034033e5927ab3ccec0ae4e2d8d09ee131fc91d0318e5` | exact pre-analysis registration |
| `REGISTRATION_MANIFEST_R0.4.md` | 1,861 | `3cd4f6e56b7d70af7c97fd2f78cbe16df9899dbce737296738670bb310b2c795` | registration explanation |
| `R3_ANALYSIS_ABORT_R0.4.json` | 1,404 | `2242c7adfeda1c60d381e163ebf84d69238f176c8030a79ab116f2f1ed24b826` | machine-readable gate stop and duplicate-pass diagnosis |
| `EXECUTION_RESULT_R0.4.md` | 1,748 | `8cc5849fd328f9a90786c9fde6b29da9cb1304b699e10acbd08289055d31b7e7` | owner-facing stopped result |

The R0.4 analyzer passed registered-file identity, map neutrality, checkpoint
compatibility, and all six exact total-accumulator closure checks. It stopped
at the target-ledger gate before reconstruction or scientific interpretation.

The ledger contains one complete noise-only-pass sequence and one complete
final-map-pass sequence. The second sequence has exactly the registered
305/34/271 final-PTC accounting. The source defect and bounded replacement are
recorded in `EL_F10_R4_ROUTINE_NOISE_PASS_LEDGER_REPAIR_2026-09-04.md` under
the standing routine-defect direction. The defective replay remains retained.
