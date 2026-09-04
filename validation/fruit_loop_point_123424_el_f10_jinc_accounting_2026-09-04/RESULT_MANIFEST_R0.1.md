# FRUIT EL-F10 result manifest r0.1

Test ID: `SCI-FRUIT-EL-F10-TARGETED-JINC-ACCOUNTING-R0.1`

Status: **valid compatibility-failure stop; no target accounting result**

## Authorization and frozen method

| Object | Bytes | SHA-256 |
|---|---:|---|
| `SCIENTIFIC_OWNER_EL_F10_AUTHORIZATION_2026-09-04.md` | 996 | `e013eba659aa192dcdd3295dece3f41aaff4a4dbb4aef6f64f0fd7a1f78fbbbc` |
| `EL_F10_BUNDLE_MANIFEST_R0.1.md` | 3,923 | `c5b2a485b9f2e02862b3ec0449f80f87f73a8eb22553aa41b363d0a14c639d12` |
| `TEST_DEFINITION.md` | 5,806 | `0ca64854ffcd25b9309e4217b8413f53f00616185cccf94a921585cda94a902c` |
| `EL_F10_JINC_ACCOUNTING.yaml` | 560 | `21e2d5126c3aaa93cda2145a681631bd4a35a79b5f47df62430595e23ec26b32` |
| `REGISTRATION_R0.1.yaml` | 8,419 | `5853f6250a3190315521f6f3206f9dbe93e6b2e6b9d583afb3c5b928c9964835` |
| `REGISTRATION_MANIFEST_R0.1.md` | 3,336 | `c10548b15c01463d5da7c9d74fc2f3fd3f2a0b8d9c7afcab56a31f168ac1d2af` |
| `tools/fruit_loops/analyze_jinc_accounting.py` | 27,141 | `f78d033af1d7fb68b8c5a73197cbcf1b1d936b1cecaa92b4241dc76693da31e4` |

The owner authorization and bundle are in
`doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane`; the other
unqualified names are in this validation directory. Source commit
`38bf8b68379e9fe5e0f361883e2ce2b1c05b0933` produced the registered
executable. Registration commit `6dba0857b` froze the replay before execution.

## Retained execution evidence

| External object | Bytes | SHA-256 |
|---|---:|---|
| `logs/replay.log` | 494,796 | `a24b3bd1df953f2869f90bbf4f066136bca7656fca6a8b01d54b9728f607b254` |
| diagnostic a1100 FITS | 7,191,360 | `75a0735c42006d2c7f55204fee051b546e6cb1d24aa8a571ee980584be33d77a` |
| diagnostic a1400 FITS | 7,191,360 | `aa7c9aaa7cabd420969f3674660f650a1b401a8c2ce9298bce4832366b2e6c7d` |
| diagnostic a2000 FITS | 7,191,360 | `1f5ce1d757eae7a018ca64fd648340d931cb12db834adffb149757df957d3206` |
| diagnostic checkpoint | 6,506,466 | `d1879db047230e1768069276bef4670d9b4826ea6f2376be59e24b46c3d9638f` |
| diagnostic mapdiag | 63,962 | `72b9fe7473f89a4a9a2040b42ab42236829294986630c9c7074d4a52d7f4ef57` |
| diagnostic JINC receipt | 5,485,571 | `010d8cbac8b4031223b84b3ef4e6a2e77d52a5a9d0b8e673a735e0e97e1c9cfc` |
| diagnostic target ledger | 100,681 | `8e9e94178bbe1099344ca9be9e1e3e7ba8048dbc016ebd2ace83c885cca5d08c` |

The external root is
`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1`.
These products are preserved in place. The receipt and ledger hashes were
recorded without opening their accounting values.

## Result records

| Object | Bytes | SHA-256 |
|---|---:|---|
| `COMPATIBILITY_ABORT_R0.1.json` | 1,436 | `b05008fb25b73f3fbf9fc1ffca66af87c29eb7004ecdce303dbdf28452f3e179` |
| `EXECUTION_RESULT_R0.1.md` | 4,775 | `ae9b01cfd9e7e5ec478448934d29d5e1558fa53ef29a1d964afce609a3815a3d` |

## Verification and disposition

- the authorized implementation passed all 632 enabled CTest cases; one
  unrelated test remains intentionally disabled;
- all 248 baseline and FRUIT-loop Python tests passed;
- the complete configuration preflight passed, including 127 unit tests,
  eight compact-compatibility fixtures, and every boundary audit;
- Ruff, Python byte compilation, generated-schema checks, and repository
  whitespace checks passed;
- all 19 registered replay inputs passed size and SHA-256 validation;
- the replay completed exactly once with exit code zero, no error or critical
  log records, 33.29 seconds wall time, and 922,501,120 bytes peak resident
  memory;
- nine ordinary science planes and three formal-coefficient planes reproduce
  EL-F6 N5 bitwise with matching grids;
- checkpoint structure is identical and its only value differences are
  `creator_version` and `learning_policy_yaml`;
- the latter differs only by the newly explicit historical-default
  `map_pixel_outlier_detector_exclusion_application: pre_cleaning` line;
- because r0.1 registered only `creator_version` as allowable, checkpoint
  neutrality failed and target accounting interpretation stopped; and
- no external input or prior product was modified, no replacement replay was
  run, and no Unity activity occurred.

This result is not evidence for UID 4460 leverage, a detector judgment, a
generic mechanism, a safeguard, a penalty or threshold change, a recurrence
choice, qualification, production use, Gate D, or Stage B.
