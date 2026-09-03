# FRUIT EL-F8 registration manifest r0.2

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **frozen before R0.2 external staging or execution**

R0.2 supersedes the pre-execution R0.1 registration only to incorporate the
bounded historical-checkpoint compatibility repair recorded in
`PRE_EXECUTION_ABORT_R0.1.md`.  The owner-approved intervention, trajectories,
order, measurements, stopping rules, and claim limits are unchanged.

| Object | Bytes | SHA-256 |
|---|---:|---|
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F8_AUTHORIZATION_2026-09-03.md` | 854 | `38fcf2ee4a0835af9accf100f13570c2930b511f3a724cd75b1b2e338cace8c0` |
| `TEST_DEFINITION.md` | 3118 | `8016f629444a32a746f5b2df916aec98045fd0e0393736184125fa11259d68e0` |
| `PRE_EXECUTION_ABORT_R0.1.md` | 2080 | `6e2a41d2637bf4d45115b9dd1014b4a5d3075ce2da373a2fe8cbb869f82e60b5` |
| `REGISTRATION_R0.2.yaml` | 3398 | `d47700a2972d8efaf25f1bc60c7bca84e6b67346d9270e841a68d1d7ec7d70aa` |
| `ANALYSIS_MANIFEST_R0.2.yaml` | 3040 | `abcc039c8f7501385949a64fae9d45de95902903cd87bdc8f60edffec613dd79` |
| `EL_F8_C5_CURRENT_R0.2.yaml` | 450 | `6210b49a41921c153886e86f1e900f00a12700741e7f7bfd553e613101eda6c0` |
| `EL_F8_A5_CURRENT_R0.2.yaml` | 450 | `6791b0923c4d5403735bc82e7305c4597b065502348ca80abfd5c0a361b2c859` |
| `EL_F8_C5_MAP_R0.2.yaml` | 443 | `01049ca9a59303ec47ecb640411560e48b4509537c154e40b7ad3a4c4afdb0b8` |
| `EL_F8_A5_MAP_R0.2.yaml` | 443 | `657bb1dc259cc169d1624bbe26892dd8a99f3ea6180ed56dc467e8671d5d5665` |
| `tools/fruit_loops/analyze_penalty_placement.py` | 33585 | `02b5748cc854ecab6759777e4209d7aa43d33d9fbbc78d4734bf1f245279309c` |
| `tools/fruit_loops/test_analyze_penalty_placement.py` | 3547 | `906bb4e0049dbf0d3d2a553f014efc7a979d9195e520a14b224cad4a16fccd21` |

The executable is built from exact repair commit
`eba17addabf4beb32dd886b0482e406ae2faaef6` and has SHA-256
`952e331856cfc72b498c77aca34572a5f9f784c2f65604d9339ac6130d71cabb`.
All 624 enabled CTest cases passed; the one pre-existing unrelated test
remained disabled.  All 233 baseline and FRUIT-loop Python tests and the
complete configuration preflight passed.  Every R0.2 YAML file parsed and the
repository whitespace check passed.  The R0.2 external root did not exist
when this manifest was written.
