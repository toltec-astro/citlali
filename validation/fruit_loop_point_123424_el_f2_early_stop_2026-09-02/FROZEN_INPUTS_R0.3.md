# SCI-FRUIT EL-F2 r0.3 final-attempt freeze

Frozen after exact r0.3 owner approval and before the final authorized
input-related attempt on 2026-09-02.

## Approval identity

- decision: `SCI-FRUIT-EL-F2-INDEPENDENT-POINTING-EARLY-STOP-R0.3`;
- manifest: `EL_F2_BUNDLE_MANIFEST_R0.3.md`;
- manifest bytes: 2728; and
- manifest SHA-256:
  `440fbb18a3190563061b0dad9c4a156a5a7e7c6699b7df0bec48fec5ae5bc579`.

All seven manifest members were reverified against their recorded sizes and
hashes immediately after approval.

## Corrected text fit-report binding

The repository and external frozen copies of `FITREPORTS_LOCAL_R0.3.yaml` are
byte-identical, 119 bytes, and have SHA-256
`df63463a1a3e83ed5dc2969525b9600c60d88b3226f79af82fda6cba4750d629`.
The overlay changes only `kids.solver.fitreportdir` and is merged after the
preserved `COMMON_LOCAL.yaml` and before the trajectory overlay.

All 12 entries in `TEXT_FITREPORT_INPUT_INVENTORY_R0.3.md` were reverified.
For each network, exactly one filename matches the executable's observed
pattern; the file retained its recorded size and hash; it parsed as ECSV; its
metadata identified observation 123424, sub-observation 0, tune scan 1, and
the expected network; and its row count equaled the corresponding processed
tune NetCDF `ntones` dimension. The 12 files contain 5,905 rows and 1,220,172
bytes in total.

## Preserved execution identities

| Object | SHA-256 |
| --- | --- |
| `setup/citlali-el-f2` | `a49082dde8f71d6f50edd8c378ad94195496b5eb0e0855b746e189f3442acbcc` |
| `BASE_POINT_123424.yaml` | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |
| `INPUTS_LOCAL.yaml` | `d79e22e1fdfdc67e9235829f1fc2b904a82894c4fbfc10fbfa7b713409b9bc02` |
| `COMMON_LOCAL.yaml` | `716c952133ee37b51d8ca4edb783741279aa8f9117149810ab94f987c5d4e269` |
| `ALPHA_1P00_CONTROL.yaml` | `c5fc9f5c4ea86de468a0e939e07b6d60d91ec40dd50cae6a67d858983906e3a7` |
| `ALPHA_1P00_INJECTED.yaml` | `74f0e27c320951552cfb23093fa8e116672ddd27570c7c2cefabd69bb731e603` |
| `ALPHA_1P25_CONTROL.yaml` | `e0e84e693d75033e02e2af0b097f171bf84e1751d7aacff4a40a51bb82eafd56` |
| `ALPHA_1P25_INJECTED.yaml` | `e140f68eb4d445393e7bab590f1901a7f732869d699e23b7608e4ffd3ef0f8c0` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `b566e6a301f5d3677be92753faef9ed8754382befd227209637ce68a56deafc1` |
| `analyze_early_stop_screen.py` | `6ec845afb77da71cc1033c26a49b4ba44168adee1981b6976b723958cd182aa4` |
| `analyze_compact_relaxation_screen.py` | `07ed91932b5bc297ee26cdc73f665840174f2f3402814c7d93ecce7742dede8e` |
| `compare_injected_source_pair.py` | `74b94aac7f21fe13b82e21d4578056a0bbb2b44f80b89f729fbf8db7a33e3280` |

Each external frozen object was rehashed after approval and matched its prior
identity. The scientific question, recurrence, raw/telescope/APT inputs,
alpha values, injections, terminal iterations, metrics, thresholds, BAAB
order, single-thread rule, and conditional restart rule remain unchanged.

## Final-attempt accounting

The first two stopped attempts remain excluded from scientific and performance
analysis. The next `alpha = 1.25` control start is the one final environmental
replacement authorized by r0.3. If it fails before iteration 0, EL-F2 ends as
invalid. If it succeeds, the other three primary trajectories may proceed in
their frozen order. An unfavorable scientific outcome may not be rerun.
