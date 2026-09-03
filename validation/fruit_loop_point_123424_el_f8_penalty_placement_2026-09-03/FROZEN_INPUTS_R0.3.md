# FRUIT EL-F8 R0.3 frozen executable and inputs

Frozen after R0.3 registration and external staging, before any R0.3 replay,
on 2026-09-03.

## Source and executable

- implementation and compatibility-repair commit:
  `eba17addabf4beb32dd886b0482e406ae2faaef6`;
- R0.3 registration commit:
  `f5e2f04f7`;
- staged executable:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f8-penalty-placement-r0.1/r0.3/setup/citlali-el-f8`;
- executable bytes: `14,813,208`;
- executable SHA-256:
  `952e331856cfc72b498c77aca34572a5f9f784c2f65604d9339ac6130d71cabb`.

All 624 enabled CTest cases passed; the one pre-existing unrelated test
remained disabled.  The complete configuration preflight and all 233 baseline
and FRUIT-loop Python tests passed.  Every R0.3 YAML file parsed and the
repository whitespace check passed.

## Frozen starting states

Both C4 copies recursively match the untouched EL-F5 control iteration-4
directory.  Their checkpoints have SHA-256
`a77505ab0637c1f257016ee0d9e801b3bba17ed52ab88d52f417a5c1513b451f`.

Both A4 copies recursively match the untouched EL-F5 injected iteration-4
directory.  Their checkpoints have SHA-256
`2d600fde6b642ea053bc49d357bed16c800bb1dd689c0ee5ae084e115970fb7c`.

## Setup identities

| File | SHA-256 |
|---|---|
| `POINT_123424_BASE.yaml` | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |
| `POINT_123424_INPUTS_LOCAL.yaml` | `d79e22e1fdfdc67e9235829f1fc2b904a82894c4fbfc10fbfa7b713409b9bc02` |
| `POINT_123424_COMMON_LOCAL.yaml` | `716c952133ee37b51d8ca4edb783741279aa8f9117149810ab94f987c5d4e269` |
| `POINT_123424_FITREPORTS_LOCAL.yaml` | `df63463a1a3e83ed5dc2969525b9600c60d88b3226f79af82fda6cba4750d629` |
| `POINT_123424_ALPHA_1P25_CONTROL.yaml` | `e0e84e693d75033e02e2af0b097f171bf84e1751d7aacff4a40a51bb82eafd56` |
| `POINT_123424_ALPHA_1P25_INJECTED.yaml` | `e140f68eb4d445393e7bab590f1901a7f732869d699e23b7608e4ffd3ef0f8c0` |
| `POINT_123424_OFF_SOURCE_CONTROL.yaml` | `da215704b9becf1b941bc7ccdfede6aed924967e4d8f8f7d59693a2e7a6ea3ca` |
| `POINT_123424_OFF_SOURCE_INJECTED.yaml` | `a7ac14987b8f71e2ee5bb4dc5ae61901d182d19a0eabb293b523444e3a0f3c3f` |
| `EL_F8_C5_CURRENT_R0.3.yaml` | `136fcbdfbe5e4b89895b8c1ab29860bd7c933535b2a4589f55bfa50e7d37d5c3` |
| `EL_F8_A5_CURRENT_R0.3.yaml` | `bb55a337471f34d4dc8aa2ac9ee22520b515d64b60ca8d4555c59fd61c80ab09` |
| `EL_F8_C5_MAP_R0.3.yaml` | `766988f5ecaa37b5a77092d9916d5d307783c215d4545789fdc850fef513be33` |
| `EL_F8_A5_MAP_R0.3.yaml` | `6ee5707a93bbbfe7efc299026ee5715f2734a014ea01f4c6efbf161c6ab56f57` |

## Run boundary

The fixed order is four fresh trajectories: C5-current, A5-current, mandatory
compatibility-only analysis, C5-map, A5-map, and the registered full analysis.
Every replay uses one thread, `--grppiex seq`, and one absolute transition
from iteration 4 to 5.  Both injected overlays explicitly use the validated
restart convention `start_iteration: 5` with unchanged source amplitudes.
Before R0.3 execution the R0.3 tree retained 246,744 KiB and the containing
filesystem reported 315,649,480 KiB available.  No R0.3 replay output,
execution log, or analysis artifact existed at freeze.

No R0.1 or R0.2 product is modified or reused.
