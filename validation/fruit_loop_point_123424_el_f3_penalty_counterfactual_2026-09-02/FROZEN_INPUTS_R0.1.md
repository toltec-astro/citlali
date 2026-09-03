# FRUIT EL-F3 frozen inputs

Status: **frozen before checkpoint intervention or execution**

Test ID: `SCI-FRUIT-EL-F3-LATE-PENALTY-COUNTERFACTUAL-R0.1`

Freeze date: 2026-09-02

Repository commit: `d3dc9e0f4d0f3649827c70f3a6509b6f2bdc4f1d`

Branch: `codex/sci-fruit-v0.1-empirical-lane`

## Execution software

| Object | SHA-256 |
|---|---|
| `/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1/setup/citlali-el-f2` | `a49082dde8f71d6f50edd8c378ad94195496b5eb0e0855b746e189f3442acbcc` |

The preserved EL-F2 executable is used without rebuilding it.

## Configuration stack

The arguments are applied in the order shown, followed by the appropriate
EL-F3 trajectory overlay and `--grppiex seq`.

| Order | Object | SHA-256 |
|---:|---|---|
| 1 | `BASE_POINT_123424.yaml` | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |
| 2 | `INPUTS_LOCAL.yaml` | `d79e22e1fdfdc67e9235829f1fc2b904a82894c4fbfc10fbfa7b713409b9bc02` |
| 3 | `COMMON_LOCAL.yaml` | `716c952133ee37b51d8ca4edb783741279aa8f9117149810ab94f987c5d4e269` |
| 4 | `FITREPORTS_LOCAL_R0.3.yaml` | `df63463a1a3e83ed5dc2969525b9600c60d88b3226f79af82fda6cba4750d629` |
| 5a | `ALPHA_1P25_CONTROL.yaml` | `e0e84e693d75033e02e2af0b097f171bf84e1751d7aacff4a40a51bb82eafd56` |
| 5b | `ALPHA_1P25_INJECTED.yaml` | `e140f68eb4d445393e7bab590f1901a7f732869d699e23b7608e4ffd3ef0f8c0` |
| 6a | `CONTROL_REPLAY.yaml` | `464d3fe20d283b20c1125ae37a9d8d27bda0d75ce6d9fdbe263dd65a4f1dcb7c` |
| 6b | `INJECTED_WITHOUT_UID4460.yaml` | `2890684061deca323f3a21bce59ec3d42ccc50f677fc08a24574b7e59a45cb15` |

The first five configuration files reside under
`/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1/setup`.
The two EL-F3 overlays reside beside this record.

## Iteration-4 source state

| Trajectory | Source directory | Checkpoint SHA-256 |
|---|---|---|
| control sham | `/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1/alpha-1.25/control/reduced/redu04` | `0eb7a0e9d8b35a4168f542c07142f34dff048244a92dc6fa718cd8812e2cd351` |
| injected counterfactual | `/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1/alpha-1.25/injected/reduced/redu04` | `c9eee5fada65fe7d9172d39ba84fb275b4124eea635933c95c29b101e6c2192f` |

The copied control state remains unmodified. The copied injected source
checkpoint is retained under a provenance-only name before the registered
one-row intervention produces the checkpoint that Citlali will load.

The source iteration-4 map-product hashes are:

| Trajectory | Array | SHA-256 |
|---|---|---|
| control | a1100 | `565e93babf441aec46ee176329c9da900c0e821b6197f1a0709b8c6cc715be5b` |
| control | a1400 | `c29458d75580bf7cc8c499d9048bdb078ded2fecc3c2c5c28e0733bb5bee3db1` |
| control | a2000 | `922f8242b0bc7e4a40cf0f77ca9bab12ceaf6d49720b9359386590252204980a` |
| injected | a1100 | `ec3e6420d26ca342247aa2f0ab5a973cb460b60554a06f3e5b6bc1a21c060095` |
| injected | a1400 | `cf760ce68e4d82d35b0bd949bfa5689447ddc251cbe6b216d5007b9d09e0678e` |
| injected | a2000 | `4cf6f6708c5479521f9f07ced5f20019a072d5f4884f79456a14f710208c5351` |

## Registered analysis and intervention tools

| Object | SHA-256 |
|---|---|
| `tools/fruit_loops/edit_restart_checkpoint_penalty.py` | `40afc57ba07e5d7e2415b5d0967fb3cb73b831560258aebfac1e3a4e0c9961e0` |
| `tools/fruit_loops/test_edit_restart_checkpoint_penalty.py` | `0cea07ae72fa4e093b1980e41c46cb7310e712da0231bd3dae2d485faf558799` |
| `tools/fruit_loops/analyze_penalty_counterfactual.py` | `b96b0f5ee9e6bc68ccce7ecc1b64dea8b5182d74745f9f9084859adf4f4fba1b` |
| `tools/fruit_loops/test_analyze_penalty_counterfactual.py` | `71be295e389f56841589e46735c65f8b1d14d32a4e5579d804d109b1992bd085` |
| `COUNTERFACTUAL_REGISTRATION_R0.1.yaml` | `c6f7bcaa1cc6f291684e2f78b13fdde4677053466fb7f8492a9901b315af0fac` |
| `TEST_DEFINITION.md` | `7f766a9f12d2f313f44ad2e57e0acb6af7f028bdb96051a66fa9bb26c250df84` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `062c2a5918789a452d34756ab8e8c0698b9ec2499e09a05d30827e860b24289f` |

## Pre-execution verification

- checkpoint editor focused tests: 3 passed;
- counterfactual analyzer focused tests: 6 passed;
- complete fruit-loop Python tests: 60 passed;
- Ruff and Python byte-compilation checks: passed;
- trial transformation of the actual source checkpoint: exactly one target row
  removed, with all remaining dimensions, values, types, and attributes equal;
- repository whitespace check: passed; and
- available filesystem capacity before copying: 317,536,948 KiB.

The trial transformation was performed in temporary storage and is not an
experimental result. It did not modify either source checkpoint.

## Fixed execution order

1. Copy and recursively compare both iteration-4 source directories.
2. Apply and audit the one-row intervention only to the injected copy.
3. Run the control sham for absolute iteration 5.
4. Require bitwise equality of every control signal, kernel, and weight map,
   plus value equality of the complete iteration-5 checkpoint.
5. Only after that gate passes, run the injected counterfactual for absolute
   iteration 5.
6. Apply the prospectively registered analysis without adding iterations,
   variants, thresholds, or tuning.

The output root and all stop rules remain those in `TEST_DEFINITION.md` and
`COUNTERFACTUAL_REGISTRATION_R0.1.yaml`.
