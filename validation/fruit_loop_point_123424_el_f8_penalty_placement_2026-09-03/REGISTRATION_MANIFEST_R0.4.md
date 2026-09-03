# FRUIT EL-F8 registration manifest r0.4

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **frozen before R0.4 external staging or execution**

R0.4 supersedes the incomplete R0.3 execution by representing the approved
placement intervention in the two copied map-branch checkpoints as well as in
their run configuration.  Exact-restart validation remains strict.  Four
fresh trajectories will run; no earlier product is reused.

| Object | Bytes | SHA-256 |
|---|---:|---|
| `doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane/SCIENTIFIC_OWNER_EL_F8_AUTHORIZATION_2026-09-03.md` | 854 | `38fcf2ee4a0835af9accf100f13570c2930b511f3a724cd75b1b2e338cace8c0` |
| `TEST_DEFINITION.md` | 3118 | `8016f629444a32a746f5b2df916aec98045fd0e0393736184125fa11259d68e0` |
| `PARTIAL_EXECUTION_R0.3.md` | 2487 | `f1866c83ef56aeb0a23be1f053d2a808c737a0e1a789b51c9d83babbb6d23cb2` |
| `REGISTRATION_R0.4.yaml` | 3975 | `107f47bd3a616b7add5a3163288605936e334d86345824454c539b2ef6234446` |
| `ANALYSIS_MANIFEST_R0.4.yaml` | 3747 | `a4e8c9de7edab65839760bd0762286da4ec389447e94b48027445fbfd37f8b6d` |
| `EL_F8_C5_CURRENT_R0.4.yaml` | 450 | `a0f5e80e8301bfcf163efe6d3f4f95667832f08a291484b6b4801348fbe8e028` |
| `EL_F8_A5_CURRENT_R0.4.yaml` | 575 | `03ecf76ab9c356e0a244bf23c91e2b847cb1f73788e1fd3bd5d0033b76914c83` |
| `EL_F8_C5_MAP_R0.4.yaml` | 443 | `e9eb7dcaa29969848f97180228826a896a280404362112d656c49dca08db1808` |
| `EL_F8_A5_MAP_R0.4.yaml` | 568 | `72e6ca0a7d6268ece0d33d000a2bdd2d5942ba42f6176705f76904461a82eb23` |
| `tools/fruit_loops/edit_restart_checkpoint_learning_policy.py` | 8300 | `1a8c56014b209d10ecf590aaaa6b3adf2d0619ed34525cbbb65dbb735a0f1f5f` |
| `tools/fruit_loops/test_edit_restart_checkpoint_learning_policy.py` | 2868 | `5f0f77ca852f9c4321a5c6ceccd8dca5e7e1ca42f34e9c8b76513d4814d1ee17` |
| `tools/fruit_loops/analyze_penalty_placement.py` | 36294 | `e19510b5e8b7585aebd22456b9b0a733a12fc5e99d72d7988771078d14da979d` |
| `tools/fruit_loops/test_analyze_penalty_placement.py` | 4965 | `9e1ff9566a990d4335f7d77ab62089f22a5ebc227838b90b3e366c7084aebef6` |

The executable remains the verified build from exact repair commit
`eba17addabf4beb32dd886b0482e406ae2faaef6`, SHA-256
`952e331856cfc72b498c77aca34572a5f9f784c2f65604d9339ac6130d71cabb`.
The intervention and analysis tooling is exact commit
`b6d7c893728e8b86a19395bee98ec2ff4110e5c5`.

All 624 enabled CTest cases passed; the one pre-existing unrelated test
remained disabled.  All 237 baseline and FRUIT-loop Python tests, focused
Ruff checks, and the complete configuration preflight passed.  Every R0.4
YAML file parsed and the repository whitespace check passed.  The R0.4
external root did not exist when this manifest was written.  The transformed
checkpoint hashes and machine audits will be recorded in
`FROZEN_INPUTS_R0.4.md` after staging and before any replay.
