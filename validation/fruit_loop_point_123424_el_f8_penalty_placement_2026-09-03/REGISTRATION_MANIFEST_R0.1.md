# FRUIT EL-F8 registration manifest r0.1

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **frozen before external staging or execution**

| Object | Bytes | SHA-256 |
|---|---:|---|
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F8_AUTHORIZATION_2026-09-03.md` | 854 | `38fcf2ee4a0835af9accf100f13570c2930b511f3a724cd75b1b2e338cace8c0` |
| `TEST_DEFINITION.md` | 3118 | `8016f629444a32a746f5b2df916aec98045fd0e0393736184125fa11259d68e0` |
| `REGISTRATION_R0.1.yaml` | 3139 | `bc587195879cdc2edfd8c88f5c63715b61c5d68c9b7800a8721f4e1c2c4700c2` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | 3010 | `6a8750e52d6c3df4a360cc6f0d93ece176298cdf44fdb9fbac09c1db46166374` |
| `EL_F8_C5_CURRENT.yaml` | 440 | `7ed3c35250863e2f49f8c499f12041141a81f775b903e9a7dd74d5b41d57a7ce` |
| `EL_F8_A5_CURRENT.yaml` | 440 | `2bc3b4eef6db9469ddb6ee33118d7c9decb3ef0c0f75bb0912a529ddbe5e9b79` |
| `EL_F8_C5_MAP.yaml` | 433 | `1bb1372269a85dfede393063a19b87388e58f5da2a5f124d0cdcf7fda5fdd7bb` |
| `EL_F8_A5_MAP.yaml` | 433 | `cc0412be6b07561106af8ed5cfe73952599cecfedd85bb2529886eb0c1dd867f` |
| `tools/fruit_loops/analyze_penalty_placement.py` | 33585 | `02b5748cc854ecab6759777e4209d7aa43d33d9fbbc78d4734bf1f245279309c` |
| `tools/fruit_loops/test_analyze_penalty_placement.py` | 3547 | `906bb4e0049dbf0d3d2a553f014efc7a979d9195e520a14b224cad4a16fccd21` |

The executable is built from exact implementation commit
`ccb67a99257fc9fba82d25346e85503363673651` and has SHA-256
`7190abe12c092cc11314a89673a2840f810fd906915e2636e41ffe196b8754a0`.
The source C4 and A4 checkpoint hashes match the approved packet exactly.

Before freeze, all 623 enabled CTest cases passed (one unrelated test remained
disabled), all 233 baseline and FRUIT-loop Python tests passed, the complete
configuration preflight passed, the new analyzer passed Ruff and byte
compilation, every registered YAML file parsed, and the repository whitespace
check passed. No EL-F8 external root, replay output, log, or analysis artifact
existed when this manifest was written.
