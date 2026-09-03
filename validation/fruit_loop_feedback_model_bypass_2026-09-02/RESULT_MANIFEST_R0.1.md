# FRUIT EL-F4 result manifest r0.1

Generated after all eight registered trajectories completed and the
prospective screen was evaluated on 2026-09-02.

The registered packet is commit `8135ad384`. The tested implementation is
commit `ad4b73e421fcfa63c895d6a1005f86580b65d203`; its frozen executable and
input record is commit `f4ca7c99b`. The frozen executable SHA-256 is
`7ece87c148787ed5b38c484b0931e2066a0a979f27d2286246749471382cae00`.

## Repository result artifacts

| Artifact | SHA-256 |
|---|---|
| `tools/fruit_loops/analyze_feedback_model_bypass.py` | `961642594c136f3779e9b22a55485147fed36378ddd5baa69430ad6598049f9e` |
| `tools/fruit_loops/test_feedback_model_bypass.py` | `8e584ccb67bc82a6781d5806532a43888fabe7bbc75dcc7a50afa89fb57a9663` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `9bf59ab9a6762a3305d14fab9508cd598fb8ed48026c51fa1e44427af2704967` |
| `EXECUTION_RESULT_R0.1.md` | `9945aeb6a0662b3b422a0cc7c87c0cb000c7b62d15d2dd2c42efa45095991b8e` |
| `ITERATION_METRICS_R0.1.csv` | `5ca65b7289652217f89790ef19f8a344d56ee44662c98448e5efe77c5fda4d55` |
| `PENALTY_INVENTORY_R0.1.csv` | `1b4e68d1ff0f163fa343b6633f38c2735309ecc45427325a7a9f50807f72b78a` |
| `PENALTY_COMPARISON_R0.1.csv` | `0042f4b0850cb9c23bb0b623e3491ff8156573f4b7a630d73230bfa39e65650b` |
| `PRIMARY_EXECUTION_R0.1.csv` | `f6534840294f5903156b744754cca9c26ab4be6f553183a6110c42ed8af3318b` |
| `SCREEN_RESULT_R0.1.json` | `3b8bed14d76b1c70408416112c241f9eb74a2f9c72aa0b335cebc9d2a28d8d53` |

## External execution logs

All logs are under
`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f4-feedback-model-bypass-r0.1/logs/`.

| Log | SHA-256 |
|---|---|
| `01-point-123424-complete-map-control.log` | `2c9ab66a61135a8535d565cfb4769f7ddf79da08d849608fecaca344c807ef0b` |
| `02-point-123424-feedback-excluded-control.log` | `b5fa3a72cca51c66780786ac7fc82a9464943de98e3c6768f0ea479cd433bbaf` |
| `03-point-123424-feedback-excluded-injected.log` | `5d484e9361010bb48b7b64aca7cbc121a53805f2e8d62424d7334deafa6a38ff` |
| `04-point-123424-complete-map-injected.log` | `bc2dd52cd62adfd8cb3a2b41928afd7c51e5238889ca0e73609c04c24db0a3b1` |
| `05-point-152389-complete-map-control.log` | `a57808a5ffd6a84dafdeb0a4daf63e30ec42b11aed8d83076e3cb68262c5ec8c` |
| `06-point-152389-feedback-excluded-control.log` | `e2101a1c70aed742ff0546c5fe31eb065146d6db11afc0ec85adeb621e1da0d0` |
| `07-point-152389-feedback-excluded-injected.log` | `b7ef6181190ca5c1e40bc1091fe2e2ebb4c8754c42dfa82cf54e0c7950d35e14` |
| `08-point-152389-complete-map-injected.log` | `092c5e9082066ab44744adc4950d8642b2c9c2ca858f3b33827d48ef7797fd9b` |

The complete reductions remain in the registered external output root. They
are development products, not qualification or production artifacts.
