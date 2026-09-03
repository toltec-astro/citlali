# FRUIT EL-F4 frozen executable and inputs

Frozen after implementation verification and before any EL-F4 trajectory on
2026-09-02.

## Executable

- path:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f4-feedback-model-bypass-r0.1/setup/citlali-el-f4`;
- SHA-256:
  `7ece87c148787ed5b38c484b0931e2066a0a979f27d2286246749471382cae00`;
- exact Citlali source commit:
  `ad4b73e421fcfa63c895d6a1005f86580b65d203`;
- source worktree status at freeze: clean except for the two pre-existing,
  unrelated owner-review archives at repository root; and
- embedded Citlali version:
  `sci-noi-v0.1-stage-a-27-g2b59ad642 (2026-09-02T10:36:22)`.

The embedded version was fixed by the existing build configuration before the
EL-F4 commit. The executable hash and exact clean source commit above are the
execution identity. No rebuild is permitted within the registered matrix.

The executable passed the 613 enabled CTest cases, the focused EL-F4 tests,
and the complete required configuration preflight before it was copied.

## Frozen setup files

All paths below are copies under the new EL-F4 `setup` directory. Their source
files and all earlier reduction products remain unchanged.

| File | SHA-256 |
|---|---|
| `POINT_123424_BASE.yaml` | `f710d1c172b5655b136ef4d8ebbff918083d37cb4974036c7af8725978d51491` |
| `POINT_123424_INPUTS_LOCAL.yaml` | `d79e22e1fdfdc67e9235829f1fc2b904a82894c4fbfc10fbfa7b713409b9bc02` |
| `POINT_123424_COMMON_LOCAL.yaml` | `716c952133ee37b51d8ca4edb783741279aa8f9117149810ab94f987c5d4e269` |
| `POINT_123424_FITREPORTS_LOCAL.yaml` | `df63463a1a3e83ed5dc2969525b9600c60d88b3226f79af82fda6cba4750d629` |
| `POINT_123424_ALPHA_1P25_CONTROL.yaml` | `e0e84e693d75033e02e2af0b097f171bf84e1751d7aacff4a40a51bb82eafd56` |
| `POINT_123424_ALPHA_1P25_INJECTED.yaml` | `e140f68eb4d445393e7bab590f1901a7f732869d699e23b7608e4ffd3ef0f8c0` |
| `POINT_152389_BASE.yaml` | `dc0df89b706f1af9f32d747861f8c23975ded7cb0cf5c706110e7a96126d5909` |
| `POINT_152389_COMMON_LOCAL.yaml` | `3a3dce72481a27352ff1d6764cfc7d9071360a211f533720a8be73698f811ae3` |
| `POINT_152389_ALPHA_1P25_CONTROL.yaml` | `75edc6a168381b032067b44068ed36c35eb9f71abe3df8d882ffb810ab494b64` |
| `POINT_152389_ALPHA_1P25_INJECTED.yaml` | `73b96dc8c9ca50d3fc31cb87771c00f2322552d79485c3b76f861908fa1bc288` |
| `POINT_123424_COMPLETE_MAP_CONTROL.yaml` | `53b8dd228c43a332c78280556ec043cfec86343fa01f0a17a9578b11309c863c` |
| `POINT_123424_COMPLETE_MAP_INJECTED.yaml` | `0a51a6d8ab64577c1fde0c9946f2c34731b2609b237651b148dac2cbdd977f9d` |
| `POINT_123424_FEEDBACK_EXCLUDED_CONTROL.yaml` | `bd3c6b556d6386aea3ac82337637d042ff085723c5d40751ba952f1aac20cfd1` |
| `POINT_123424_FEEDBACK_EXCLUDED_INJECTED.yaml` | `cd8afdd78c23b2ddfa0bfddce2e1b82cb73962ed3220bb8bc90bd126538e99c9` |
| `POINT_152389_COMPLETE_MAP_CONTROL.yaml` | `31371aaabfffca76a518d5a1ef927f1020091832c9b786ad1f6e5597f65f2826` |
| `POINT_152389_COMPLETE_MAP_INJECTED.yaml` | `3ebc1fde15e49b2a25666396ab2d706551c0ec8809cb83ae2a74b4509e2e5dde` |
| `POINT_152389_FEEDBACK_EXCLUDED_CONTROL.yaml` | `6fb4046b9fd476fc397bb8f957536a645db705bffe26b1204b0b9eda2a8d490f` |
| `POINT_152389_FEEDBACK_EXCLUDED_INJECTED.yaml` | `7ef0193b64d90fad35bdc992d52f8d449faa1c6fd1cad54f6776bbd5e860d818` |

## Fixed merge order

Every observation-123424 command uses, in order: its base, local inputs,
common local settings, local fit-report directory, the matching inherited
alpha-1.25 control or injected overlay, and the registered EL-F4 trajectory
overlay. Every observation-152389 command uses its base, common local
settings, matching inherited alpha-1.25 overlay, and registered EL-F4
trajectory overlay. Every command uses `--grppiex seq` and the frozen
executable above.
