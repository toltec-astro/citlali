# FRUIT EL-F8 registration manifest r0.3

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **frozen before R0.3 external staging or execution**

R0.3 supersedes the incomplete R0.2 execution only to state the validated
injected-source restart convention explicitly.  The owner-approved
intervention, four fresh trajectories, order, measurements, stopping rules,
and claim limits are unchanged.  No R0.2 product is reused.

| Object | Bytes | SHA-256 |
|---|---:|---|
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F8_AUTHORIZATION_2026-09-03.md` | 854 | `38fcf2ee4a0835af9accf100f13570c2930b511f3a724cd75b1b2e338cace8c0` |
| `TEST_DEFINITION.md` | 3118 | `8016f629444a32a746f5b2df916aec98045fd0e0393736184125fa11259d68e0` |
| `PRE_EXECUTION_ABORT_R0.1.md` | 2080 | `6e2a41d2637bf4d45115b9dd1014b4a5d3075ce2da373a2fe8cbb869f82e60b5` |
| `PARTIAL_EXECUTION_R0.2.md` | 1905 | `79a9fb7a3762234f10a7e9691c8bdfa0f99e8858fbdf2acfcabd3e8eebedc5df` |
| `REGISTRATION_R0.3.yaml` | 3532 | `bb22e777c5a9a28a9c33155ef16739b2cf430ef9e8d45a57591c9c7606e42dc9` |
| `ANALYSIS_MANIFEST_R0.3.yaml` | 3040 | `6f19a4cfb3da06533304f89a0a12b448d27e590acf095440311e88f47572b8c6` |
| `EL_F8_C5_CURRENT_R0.3.yaml` | 450 | `136fcbdfbe5e4b89895b8c1ab29860bd7c933535b2a4589f55bfa50e7d37d5c3` |
| `EL_F8_A5_CURRENT_R0.3.yaml` | 575 | `bb55a337471f34d4dc8aa2ac9ee22520b515d64b60ca8d4555c59fd61c80ab09` |
| `EL_F8_C5_MAP_R0.3.yaml` | 443 | `766988f5ecaa37b5a77092d9916d5d307783c215d4545789fdc850fef513be33` |
| `EL_F8_A5_MAP_R0.3.yaml` | 568 | `6ee5707a93bbbfe7efc299026ee5715f2734a014ea01f4c6efbf161c6ab56f57` |
| `tools/fruit_loops/analyze_penalty_placement.py` | 33585 | `02b5748cc854ecab6759777e4209d7aa43d33d9fbbc78d4734bf1f245279309c` |
| `tools/fruit_loops/test_analyze_penalty_placement.py` | 3547 | `906bb4e0049dbf0d3d2a553f014efc7a979d9195e520a14b224cad4a16fccd21` |

The executable remains the verified build from exact repair commit
`eba17addabf4beb32dd886b0482e406ae2faaef6`, SHA-256
`952e331856cfc72b498c77aca34572a5f9f784c2f65604d9339ac6130d71cabb`.
All 624 enabled CTest cases passed; the one pre-existing unrelated test
remained disabled.  All 233 baseline and FRUIT-loop Python tests and the
complete configuration preflight passed.  Every R0.3 YAML file parsed and the
repository whitespace check passed.  The R0.3 external root did not exist
when this manifest was written.
