# SCI-FRUIT EL-F10-R4 registration manifest r0.5

Test ID: `SCI-FRUIT-EL-F10-R4-NOISE-PASS-LEDGER-REPAIR-R0.1`

Status: **frozen before isolated local replacement replay**

The standing routine-defect direction and
`EL_F10_R4_ROUTINE_NOISE_PASS_LEDGER_REPAIR_2026-09-04.md` authorize this
diagnostic-only replacement without a new scientific-owner decision. The
defective replay remains retained at its original root.

`REGISTRATION_R0.5.yaml` is 10,603 bytes with SHA-256
`018ce09c634789d011a1f73843301fcd7f9b2fda38e8c9d29c00e954a76e34c8`.
All 24 registered files passed size and SHA-256 validation before this manifest
was written.

The staged executable is 14,858,136 bytes with SHA-256
`71911a6768b7ecfff0d165a17d498adf5c0e8e0219e733e72d633d8b545c7636`,
built from source commit `c68b97f30`. The only source change affecting the
replay requires an observation-map pass before a target sample is retained in
the diagnostic ledger. Full local verification passed: the CLI built, all 632
enabled CTest cases passed with one disabled test, all 252 baseline and
FRUIT-loop Python tests passed, and the complete configuration preflight
passed.

## Frozen launch

Exactly one isolated local replacement replay may be launched with this
argument order:

```text
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.2/setup/citlali-el-f10-r4
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_BASE.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_INPUTS_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_COMMON_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_FITREPORTS_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_ALPHA_1P25_INJECTED.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_OFF_SOURCE_INJECTED.yaml
/Users/gwilson/.codex/worktrees/4c31/citlali-refactor/validation/fruit_loop_point_123424_el_f10_jinc_accounting_2026-09-04/EL_F10_R4_JINC_ACCOUNTING.yaml
--grppiex seq
```

The output root is the new isolated
`fruit-el-f10-jinc-accounting-r0.2/reduced` directory. The same registered
iteration-4 checkpoint, observation inputs, injection, science configuration,
target, gates, bounds, and comparison products are used. No Unity activity is
authorized.
