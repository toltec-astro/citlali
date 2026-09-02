# SCI-FRUIT EL-F2 r0.2 approved-input freeze

Frozen after exact r0.2 owner approval and before the authorized replacement
trajectory on 2026-09-02.

## Approval identity

- decision: `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.2`;
- manifest: `EL_F2_BUNDLE_MANIFEST_R0.2.md`;
- manifest bytes: 2529; and
- manifest SHA-256:
  `0ffc2446568b4e70696291c2c46aad2545e0e99be73cb3bd439ddfbfaf8acb88`.

All seven manifest members were reverified against their recorded sizes and
hashes immediately after approval.

## Preserved r0.1 execution identities

The executable, analysis, original configuration overlays, analysis manifest,
scientific question, thresholds, terminal iterations, and BAAB order remain
exactly those frozen in `FROZEN_INPUTS_R0.1.md`.

| Object | SHA-256 |
| --- | --- |
| `setup/citlali-el-f2` | `a49082dde8f71d6f50edd8c378ad94195496b5eb0e0855b746e189f3442acbcc` |
| `analyze_early_stop_screen.py` | `6ec845afb77da71cc1033c26a49b4ba44168adee1981b6976b723958cd182aa4` |
| `analyze_compact_relaxation_screen.py` | `07ed91932b5bc297ee26cdc73f665840174f2f3402814c7d93ecce7742dede8e` |
| `compare_injected_source_pair.py` | `74b94aac7f21fe13b82e21d4578056a0bbb2b44f80b89f729fbf8db7a33e3280` |
| `BASE_POINT_123424.yaml` | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |
| `INPUTS_LOCAL.yaml` | `d79e22e1fdfdc67e9235829f1fc2b904a82894c4fbfc10fbfa7b713409b9bc02` |
| `COMMON_LOCAL.yaml` | `716c952133ee37b51d8ca4edb783741279aa8f9117149810ab94f987c5d4e269` |
| `ALPHA_1P00_CONTROL.yaml` | `c5fc9f5c4ea86de468a0e939e07b6d60d91ec40dd50cae6a67d858983906e3a7` |
| `ALPHA_1P00_INJECTED.yaml` | `74f0e27c320951552cfb23093fa8e116672ddd27570c7c2cefabd69bb731e603` |
| `ALPHA_1P25_CONTROL.yaml` | `e0e84e693d75033e02e2af0b097f171bf84e1751d7aacff4a40a51bb82eafd56` |
| `ALPHA_1P25_INJECTED.yaml` | `e140f68eb4d445393e7bab590f1901a7f732869d699e23b7608e4ffd3ef0f8c0` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `b566e6a301f5d3677be92753faef9ed8754382befd227209637ce68a56deafc1` |

Each external frozen copy was rehashed after approval and matched these
identities. The 15 entries in `INPUT_INVENTORY_R0.1.md` were also reverified
against their recorded sizes and hashes.

## Corrected fit-report binding

The only new configuration object is:

| Object | Bytes | SHA-256 |
| --- | ---: | --- |
| `setup/FITREPORTS_LOCAL_R0.2.yaml` | 121 | `8caa3f8827b56eb3b716b469ce67904588d8e4dbd632c089fa926877d78f94ff` |

It was byte-compared with the approved repository overlay. It changes only
`kids.solver.fitreportdir` and is merged after `COMMON_LOCAL.yaml` and before
the trajectory-specific overlay.

All 12 files in `FITREPORT_INPUT_INVENTORY_R0.2.md` were reverified by size and
SHA-256. Their NetCDF headers were rechecked: every file has 177 sweeps,
observation 123424, sub-observation 0, scan 1, and the expected one of networks
0--9, 11, or 12.

The subsequent replacement run established that this exact freeze was not a
valid executable input correction. The files are processed tune NetCDFs, not
the ECSV/ASCII text fit reports selected by the executable's per-network
`cal_file` regular expressions. The approved bytes and checks remain preserved
as evidence, but their r0.2 classification as loadable fit reports is
superseded by the second-attempt failure record.

## Replacement accounting

The original and r0.2 attempts both remain excluded from scientific and
performance analysis. Each stopped before iteration 0, and together they
consume both environmental replacements authorized by r0.1. No further
trajectory may begin without a new exact owner decision; an unfavorable
scientific outcome may not be rerun.
