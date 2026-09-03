# FRUIT EL-F5 frozen executable and inputs

Frozen after implementation verification and before either EL-F5 trajectory
on 2026-09-03.

## Executable

- path:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f5-off-source-injection-r0.1/setup/citlali-el-f5`;
- SHA-256:
  `6431c6653ed46ff6e1dfa5512cd27e8169525f7a110207b0b24505786f39dbbe`;
- exact Citlali source commit:
  `fd760cdbf59940f803ab38323088b35682f342cd`;
- source worktree status at freeze: clean except for the two pre-existing,
  unrelated owner-review archives at repository root; and
- embedded Citlali version:
  `sci-noi-v0.1-stage-a-27-g2b59ad642 (2026-09-02T10:36:22)`.

The build configuration retained an older embedded version string. The
executable hash and exact clean source commit above are the execution identity.
No rebuild is permitted within the registered matrix.

Before it was copied, the executable passed all 618 enabled CTest cases (one
unrelated test remained disabled), 13 focused Python tests, the focused C++
injection tests, and the complete required configuration preflight.

## Frozen setup files

All files below are copies under the new EL-F5 `setup` directory. The EL-F4
setup files and all earlier reduction products remain unchanged.

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

## Fixed merge order

The control command uses, in order: `POINT_123424_BASE.yaml`,
`POINT_123424_INPUTS_LOCAL.yaml`, `POINT_123424_COMMON_LOCAL.yaml`,
`POINT_123424_FITREPORTS_LOCAL.yaml`,
`POINT_123424_ALPHA_1P25_CONTROL.yaml`, and
`POINT_123424_OFF_SOURCE_CONTROL.yaml`.

The injected command uses the same first four files, then
`POINT_123424_ALPHA_1P25_INJECTED.yaml` and
`POINT_123424_OFF_SOURCE_INJECTED.yaml`. Both commands use `--grppiex seq` and
the frozen executable above. The resulting merged configurations may differ
only in output directory and injection-enabled state.

