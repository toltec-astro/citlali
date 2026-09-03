# FRUIT EL-F8 R0.4 frozen executable and inputs

Frozen after R0.4 registration, external staging, and audited checkpoint
intervention, before any R0.4 replay, on 2026-09-03.

## Source and executable

- executable implementation commit:
  `eba17addabf4beb32dd886b0482e406ae2faaef6`;
- intervention and analysis tooling commit:
  `b6d7c893728e8b86a19395bee98ec2ff4110e5c5`;
- R0.4 registration commit: `d7f91b394`;
- staged executable:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f8-penalty-placement-r0.1/r0.4/setup/citlali-el-f8`;
- executable SHA-256:
  `952e331856cfc72b498c77aca34572a5f9f784c2f65604d9339ac6130d71cabb`.

All 624 enabled CTest cases passed; the one pre-existing unrelated test
remained disabled.  The complete configuration preflight, all 237 baseline
and FRUIT-loop Python tests, and focused Ruff checks passed.

## Frozen starting states and policy intervention

All four iteration-4 reduction directories recursively matched their
untouched EL-F5 sources before intervention.  The current-placement copies
remain unchanged:

- C4 checkpoint SHA-256:
  `a77505ab0637c1f257016ee0d9e801b3bba17ed52ab88d52f417a5c1513b451f`;
- A4 checkpoint SHA-256:
  `2d600fde6b642ea053bc49d357bed16c800bb1dd689c0ee5ae084e115970fb7c`.

The two original map-branch checkpoints are retained under `interventions/`
with those same hashes.  The fail-closed editor changed only
`learning_policy_yaml.map_pixel_outlier_detector_exclusion_application` from
its normalized historical value `pre_cleaning` to `pre_mapmaking`:

| Branch | Transformed checkpoint SHA-256 | Audit SHA-256 |
|---|---|---|
| control | `8d9fe5dfa4b90e21ff352315caec6e6566228e659c496bdb80b3a5e05018872a` | `a0624dfd798645e7f32c24bcc0601217a4dc55a762f00f27537c4783ca5ae679` |
| injected | `b357a9a8b2a94759e86163722783711494df1bc46d5eeeaff119f1e82fdd0217` | `6d302958304235a2089acad9aab40e3aaa86c80a68773243eb076ec9e6b49844` |

Both machine audits and an independent analyzer re-audit verify that all
other values, types, dimensions, and attributes are unchanged.  In
particular, every feedback value and every learned detector-penalty record is
identical to its source checkpoint.

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
| `EL_F8_C5_CURRENT_R0.4.yaml` | `a0f5e80e8301bfcf163efe6d3f4f95667832f08a291484b6b4801348fbe8e028` |
| `EL_F8_A5_CURRENT_R0.4.yaml` | `03ecf76ab9c356e0a244bf23c91e2b847cb1f73788e1fd3bd5d0033b76914c83` |
| `EL_F8_C5_MAP_R0.4.yaml` | `e9eb7dcaa29969848f97180228826a896a280404362112d656c49dca08db1808` |
| `EL_F8_A5_MAP_R0.4.yaml` | `72e6ca0a7d6268ece0d33d000a2bdd2d5942ba42f6176705f76904461a82eb23` |

## Run boundary

The fixed order is four fresh trajectories: C5-current, A5-current, mandatory
compatibility-only analysis, C5-map, A5-map, and the registered full analysis.
Every replay uses one thread, `--grppiex seq`, and one absolute transition
from iteration 4 to 5.  Before R0.4 execution the R0.4 tree retained 259,464
KiB and the containing filesystem reported 315,289,528 KiB available.  No
R0.4 replay output, execution log, or analysis artifact existed at freeze.

No R0.1, R0.2, or R0.3 product is modified or reused.
