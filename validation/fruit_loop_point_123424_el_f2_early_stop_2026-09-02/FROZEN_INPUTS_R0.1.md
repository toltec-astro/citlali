# SCI-FRUIT EL-F2 frozen executable, analysis, and inputs

Frozen before any primary comparison run on 2026-09-02.

## Executable

- Frozen path:
  `/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1/setup/citlali-el-f2`
- SHA-256: `a49082dde8f71d6f50edd8c378ad94195496b5eb0e0855b746e189f3442acbcc`
- Embedded Citlali version:
  `sci-noi-v0.1-stage-a-27-g2b59ad642 (2026-09-02T10:36:22)`
- Embedded kids version: `04088da-dirty (2026-09-02T10:36:22)`

The required local build completed successfully and reported the target up to
date. The exact executable is byte-identical to the executable used for the
valid EL-F1 r1 screen. No C++ source differs from proposal commit `cb1b24f25`;
the only pre-run source change is the approved Python analysis-tool extension.
The executable hash, rather than its incomplete embedded version string, is
the execution identity.

## Frozen analysis

Byte-identical copies were placed under `setup/frozen_analysis` before any new
map product was opened.

| File | SHA-256 |
| --- | --- |
| `analyze_early_stop_screen.py` | `6ec845afb77da71cc1033c26a49b4ba44168adee1981b6976b723958cd182aa4` |
| `analyze_compact_relaxation_screen.py` | `07ed91932b5bc297ee26cdc73f665840174f2f3402814c7d93ecce7742dede8e` |
| `compare_injected_source_pair.py` | `74b94aac7f21fe13b82e21d4578056a0bbb2b44f80b89f729fbf8db7a33e3280` |

Ten focused tests passed before the freeze. They include exact use of
reference iteration 6 and candidate iteration 5 and rejection of missing or
extra iterations. Ruff passed on every touched Python file.

## Frozen configuration

| File | SHA-256 |
| --- | --- |
| `BASE_POINT_123424.yaml` | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |
| `INPUTS_LOCAL.yaml` | `d79e22e1fdfdc67e9235829f1fc2b904a82894c4fbfc10fbfa7b713409b9bc02` |
| `COMMON_LOCAL.yaml` | `716c952133ee37b51d8ca4edb783741279aa8f9117149810ab94f987c5d4e269` |
| `ALPHA_1P00_CONTROL.yaml` | `c5fc9f5c4ea86de468a0e939e07b6d60d91ec40dd50cae6a67d858983906e3a7` |
| `ALPHA_1P00_INJECTED.yaml` | `74f0e27c320951552cfb23093fa8e116672ddd27570c7c2cefabd69bb731e603` |
| `ALPHA_1P25_CONTROL.yaml` | `e0e84e693d75033e02e2af0b097f171bf84e1751d7aacff4a40a51bb82eafd56` |
| `ALPHA_1P25_INJECTED.yaml` | `e140f68eb4d445393e7bab590f1901a7f732869d699e23b7608e4ffd3ef0f8c0` |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `b566e6a301f5d3677be92753faef9ed8754382befd227209637ce68a56deafc1` |

Every frozen copy was byte-compared with its approved repository source. The
merged configurations contain no Unity path, resolve exactly one observation,
and reference all 14 locally verified input files.

The external raw, telescope, APT, and source-configuration hashes remain those
in `INPUT_INVENTORY_R0.1.md` and were reverified after owner approval. The
legacy APT remains an immutable, common development input.

## Predeclared run order

Before opening any new product, the sequential run order was fixed as:

1. `alpha = 1.25`, injection disabled;
2. `alpha = 1.00`, injection disabled;
3. `alpha = 1.00`, injection enabled; and
4. `alpha = 1.25`, injection enabled.

This BAAB order places the candidate in the cold-start and final positions,
so the candidate does not receive the simplest possible warm-cache advantage.
Only the pair-mean wall-time result has the predeclared performance role.
