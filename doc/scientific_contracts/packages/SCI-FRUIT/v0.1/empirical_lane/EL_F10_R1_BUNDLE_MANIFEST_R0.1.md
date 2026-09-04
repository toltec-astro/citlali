# SCI-FRUIT v0.1 — EL-F10-R1 bundle manifest r0.1

Artifact identity:
`SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1`

Manifest date: `2026-09-04`

Status: **exact owner-review payload; EL-F10-R1 is not approved**

Package paths are relative to this manifest; validation, tool, and source
paths are repository-relative.

| File | Bytes | SHA-256 | Role |
|---|---:|---|---|
| `EL_F10_R1_COMPATIBILITY_NORMALIZATION_OWNER_REVIEW_R0.1.md` | 4,555 | `2c141326a46c0fbac84a48010ac5c9aa14697b61c20398f70881cb8997d1881e` | exact proposed no-replay repair and owner choices |
| `validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/EXECUTION_RESULT_R0.1.md` | 4,775 | `ae9b01cfd9e7e5ec478448934d29d5e1558fa53ef29a1d964afce609a3815a3d` | preserved compatibility-failure result and next recommendation |
| `validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/RESULT_MANIFEST_R0.1.md` | 4,376 | `6fae331b38023280a9b5833f65370a8dc42c39df87bd1ba86db25256e08431f1` | result and retained-output identities |
| `validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/COMPATIBILITY_ABORT_R0.1.json` | 1,436 | `b05008fb25b73f3fbf9fc1ffca66af87c29eb7004ecdce303dbdf28452f3e179` | machine-readable stop record proving accounting values remained unopened |
| `validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/REGISTRATION_R0.1.yaml` | 8,419 | `5853f6250a3190315521f6f3206f9dbe93e6b2e6b9d583afb3c5b928c9964835` | immutable failed registration and unchanged scientific gates |
| `validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/TEST_DEFINITION.md` | 5,806 | `0ca64854ffcd25b9309e4217b8413f53f00616185cccf94a921585cda94a902c` | exact finalization, error bounds, measurements, and claim limits |
| `tools/fruit_loops/analyze_jinc_accounting.py` | 27,141 | `f78d033af1d7fb68b8c5a73197cbcf1b1d936b1cecaa92b4241dc76693da31e4` | frozen analysis and exact one-key compatibility normalization |
| `validation/fruit_loop_point_123424_el_f8_penalty_placement_2026-09-03/PRE_EXECUTION_ABORT_R0.1.md` | 2,080 | `6e2a41d2637bf4d45115b9dd1014b4a5d3075ce2da373a2fe8cbb869f82e60b5` | earlier independent record of the identical missing-default issue |
| `src/citlali/core/pipeline/reduction_restart_checkpoint.cpp` | 60,902 | `6b00c726b539b7607d32e2d54723d1707be04514e179002cd98b4174ddb8f7cf` | checked-in bounded restart normalization and rejection rule |

The exact retained, still-uninterpreted accounting files are:

- receipt: 5,485,571 bytes, SHA-256
  `010d8cbac8b4031223b84b3ef4e6a2e77d52a5a9d0b8e673a735e0e97e1c9cfc`;
- target ledger: 100,681 bytes, SHA-256
  `8e9e94178bbe1099344ca9be9e1e3e7ba8048dbc016ebd2ace83c885cca5d08c`;
- replay checkpoint: 6,506,466 bytes, SHA-256
  `d1879db047230e1768069276bef4670d9b4826ea6f2376be59e24b46c3d9638f`;
- a1100 FITS: 7,191,360 bytes, SHA-256
  `75a0735c42006d2c7f55204fee051b546e6cb1d24aa8a571ee980584be33d77a`;
- a1400 FITS: 7,191,360 bytes, SHA-256
  `aa7c9aaa7cabd420969f3674660f650a1b401a8c2ce9298bce4832366b2e6c7d`;
- a2000 FITS: 7,191,360 bytes, SHA-256
  `1f5ce1d757eae7a018ca64fd648340d931cb12db834adffb149757df957d3206`;
  and
- replay log: 494,796 bytes, SHA-256
  `a24b3bd1df953f2869f90bbf4f066136bca7656fca6a8b01d54b9728f607b254`.

Their external root is
`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1`.
Approval authorizes binding these exact files in a new registration before
opening the receipt or ledger. It does not authorize replacing or regenerating
any of them.

## Approval semantics

The exact affirmative statement is:

> I approve `SCI-FRUIT-EL-F10-R1-COMPATIBILITY-NORMALIZATION-R0.1` against the exact `EL_F10_R1_BUNDLE_MANIFEST_R0.1.md`.

That statement selects Choice A in the owner-review file. It authorizes only
an owner-authorization record, an exact output-bound R1 registration, and one
run of the frozen analysis on the existing EL-F10 products using the one-key
absence-to-`pre_cleaning` normalization. Every other r0.1 gate remains fixed.

It does not authorize another Citlali replay, an algorithm or configuration
change, a detector judgment, a safeguard or penalty decision, a recurrence
change, production use, qualification, Gate D, Stage B, or Unity activity.
