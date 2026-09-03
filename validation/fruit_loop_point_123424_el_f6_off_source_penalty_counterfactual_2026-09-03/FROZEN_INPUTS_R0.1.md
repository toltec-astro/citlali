# FRUIT EL-F6 frozen executable, inputs, and intervention

Frozen before either replay on 2026-09-03.

Test ID: `SCI-FRUIT-EL-F6-OFF-SOURCE-PENALTY-COUNTERFACTUAL-R0.1`

Repository preparation commit:
`e68a4ae75222931259a82046c73ccaab63826b39`

Branch: `codex/sci-fruit-v0.1-empirical-lane`

## Execution software

| Object | SHA-256 |
|---|---|
| `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f6-off-source-penalty-counterfactual-r0.1/setup/citlali-el-f5` | `6431c6653ed46ff6e1dfa5512cd27e8169525f7a110207b0b24505786f39dbbe` |

This is an exact copy of the frozen EL-F5 executable. No rebuild is permitted
within EL-F6.

## Fixed configuration stack

Each replay applies the first six files in order, followed by its own final
overlay and `--grppiex seq`.

| Order | File | SHA-256 |
|---:|---|---|
| 1 | `POINT_123424_BASE.yaml` | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |
| 2 | `POINT_123424_INPUTS_LOCAL.yaml` | `d79e22e1fdfdc67e9235829f1fc2b904a82894c4fbfc10fbfa7b713409b9bc02` |
| 3 | `POINT_123424_COMMON_LOCAL.yaml` | `716c952133ee37b51d8ca4edb783741279aa8f9117149810ab94f987c5d4e269` |
| 4 | `POINT_123424_FITREPORTS_LOCAL.yaml` | `df63463a1a3e83ed5dc2969525b9600c60d88b3226f79af82fda6cba4750d629` |
| 5 | `POINT_123424_ALPHA_1P25_INJECTED.yaml` | `e140f68eb4d445393e7bab590f1901a7f732869d699e23b7608e4ffd3ef0f8c0` |
| 6 | `POINT_123424_OFF_SOURCE_INJECTED.yaml` | `a7ac14987b8f71e2ee5bb4dc5ae61901d182d19a0eabb293b523444e3a0f3c3f` |
| 7a | `UNTOUCHED_INJECTED_SHAM.yaml` | `265a60da45f9adea4e3cf810f1f9d24d6c88294fe13918c16d3132e28f21b0e6` |
| 7b | `INJECTED_WITHOUT_UID4460.yaml` | `6dbb17601d2bf46aafd4490da95555cdc1840285225b9168db1d0e5364fa0cac` |

All files are retained under the EL-F6 external `setup` directory. Both final
overlays request exactly one absolute transition, iteration 4 to 5, with one
thread. They enable the already injected source from restarted iteration 5;
the preceding off-source overlay fixes its map-world position at `(0, -60)`
arcsec.

## Source and copied restart state

The source is the complete EL-F5 off-source injected `redu04` directory:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f5-off-source-injection-r0.1/point-123424/off-source-injected/reduced/redu04`

Both isolated copies were recursively equal to this source before the
registered intervention. The source checkpoint and both pre-intervention
copies had SHA-256
`2d600fde6b642ea053bc49d357bed16c800bb1dd689c0ee5ae084e115970fb7c`.
The source and all EL-F5 reduction products remain unchanged.

The copied iteration-4 signal products retain these identities:

| Array | SHA-256 |
|---|---|
| a1100 | `f423fbb19ea53b4c83d0bd1fb899216ee9973bc8e690b43f4ad4bfa771c64aa6` |
| a1400 | `96cea6952608d0ea7605873e97604ecdc70c6af3ee5bfa410ed7028ed3797dbc` |
| a2000 | `0e9512a3ac1c6c730b2f1b9f8fe90e20318b516a275a8c786663af7b07d4fe09` |

## Registered intervention

The counterfactual copy retains the original checkpoint as
`citlali_restart_checkpoint.provenance-source.nc`. The fail-closed editor
removed the one registered UID 4460 record and wrote a new restart checkpoint.

| Object | SHA-256 |
|---|---|
| untouched sham checkpoint | `2d600fde6b642ea053bc49d357bed16c800bb1dd689c0ee5ae084e115970fb7c` |
| retained counterfactual source checkpoint | `2d600fde6b642ea053bc49d357bed16c800bb1dd689c0ee5ae084e115970fb7c` |
| counterfactual checkpoint | `9f8faf73fc759202258ba58109ba499bd73d8f513d93ea763df75069ae78f942` |
| intervention audit | `6894a356b889d15bc0641ba0d66c220ed3de7a888c87b94c6f7128eca790892f` |

The audit records source count four, output count three, removed index one,
and the exact registered identity. It verifies all other values and all types,
dimensions, and attributes equal.

## Analysis identities and verification

| Object | SHA-256 |
|---|---|
| `REGISTRATION_R0.1.yaml` | `ef503cb651bac1be20bd60515d3be49ca25f55cae714f7f16581922c4b5c90f8` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `cfaf035ee9fa3ef9266dc0a423063ab3d5ec124bb8ab8749627382173b7ae183` |
| `edit_restart_checkpoint_penalty.py` | `40afc57ba07e5d7e2415b5d0967fb3cb73b831560258aebfac1e3a4e0c9961e0` |
| `test_edit_restart_checkpoint_penalty.py` | `0cea07ae72fa4e093b1980e41c46cb7310e712da0231bd3dae2d485faf558799` |
| `analyze_off_source_penalty_counterfactual.py` | `1fdc15d657b45551813fe16bc7038b86ee9bd860aca78ef4af9a81c362db44b2` |
| `test_analyze_off_source_penalty_counterfactual.py` | `ae42f1e5d45787d9d92cabb473ef4c9d189d24a301bdbe6b8d2c475e33fd2c56` |

Before the freeze:

- all 222 baseline and FRUIT-loop Python tests passed;
- Ruff and Python byte-compilation checks passed;
- a temporary trial intervention against the actual source removed exactly
  one row and verified every other checkpoint property equal;
- the new analyzer reproduced all eight frozen EL-F5 a1400 values exactly;
- all YAML files parsed and the repository whitespace check passed; and
- 311,740,552 KiB was available before external preparation.

The external root retained 136,964 KiB at freeze. No replay product or result
had been created.

## Fixed execution order

1. Run the untouched injected sham from iteration 4 to 5.
2. Require exact all-array image planes and complete checkpoint equality with
   the original EL-F5 injected iteration 5.
3. Only if that gate passes, run the UID-4460-removed counterfactual from
   iteration 4 to 5.
4. Apply the frozen analysis without adding a variant, iteration, threshold,
   or tuning choice.

