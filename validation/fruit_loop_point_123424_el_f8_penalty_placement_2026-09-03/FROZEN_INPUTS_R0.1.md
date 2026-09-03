# FRUIT EL-F8 frozen executable and inputs

Frozen after registration and external staging, before any EL-F8 replay, on
2026-09-03.

## Source and executable

- implementation commit:
  `ccb67a99257fc9fba82d25346e85503363673651`;
- registration commit:
  `fd34e33c3`;
- staged executable:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f8-penalty-placement-r0.1/setup/citlali-el-f8`;
- executable bytes: `14794808`;
- executable SHA-256:
  `7190abe12c092cc11314a89673a2840f810fd906915e2636e41ffe196b8754a0`.

The implementation passed all 623 enabled CTest cases; the one pre-existing
unrelated test remained disabled. The complete configuration preflight, five
focused configuration tests, six focused application-path tests, all 233
baseline and FRUIT-loop Python tests, analyzer Ruff and byte-compilation
checks, YAML parsing, and the repository whitespace check passed.

## Frozen starting states

The two C4 copies recursively match the untouched EL-F5 control iteration-4
directory. Their checkpoints both have SHA-256
`a77505ab0637c1f257016ee0d9e801b3bba17ed52ab88d52f417a5c1513b451f`.

The two A4 copies recursively match the untouched EL-F5 injected iteration-4
directory. Their checkpoints both have SHA-256
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
| `EL_F8_C5_CURRENT.yaml` | `7ed3c35250863e2f49f8c499f12041141a81f775b903e9a7dd74d5b41d57a7ce` |
| `EL_F8_A5_CURRENT.yaml` | `2bc3b4eef6db9469ddb6ee33118d7c9decb3ef0c0f75bb0912a529ddbe5e9b79` |
| `EL_F8_C5_MAP.yaml` | `1bb1372269a85dfede393063a19b87388e58f5da2a5f124d0cdcf7fda5fdd7bb` |
| `EL_F8_A5_MAP.yaml` | `cc0412be6b07561106af8ed5cfe73952599cecfedd85bb2529886eb0c1dd867f` |

## Run boundary

The fixed order is C5-current, A5-current, the mandatory compatibility-only
analysis, C5-map, A5-map, and the registered full analysis. Every replay uses
one thread, `--grppiex seq`, and one absolute transition from iteration 4 to
5. Before execution the external tree retained 246,728 KiB and the containing
filesystem reported 316,487,280 KiB available. No replay output, execution
log, or analysis artifact existed at freeze.
