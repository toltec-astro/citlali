# SCI-FRUIT EL-F11 replay registration manifest r0.1

Test ID:
`SCI-FRUIT-EL-F11-PROSPECTIVE-INFLUENCE-PERSISTENCE-R0.1`

Status: **frozen before the one authorized local replay**

The exact owner approval is recorded in
`SCIENTIFIC_OWNER_EL_F11_AUTHORIZATION_2026-09-04.md` and bound to
`EL_F11_BUNDLE_MANIFEST_R0.1.md`.

`REGISTRATION_R0.1.yaml` is 11,413 bytes with SHA-256
`3e20076162e0b28b7ab25ee0c8f4a2d9bd7a2d0c1bc212c5eee7d000e820e116`.
All 23 registered files passed their exact size and SHA-256 checks before this
manifest was written.

The complete copied iteration-3 restart directory passed a recursive
file-content comparison against the preserved EL-F5 source. Its checkpoint is
6,508,010 bytes with SHA-256
`a20558aaed4ddf1c34ab343770002d15882964c1fa22b616277146bb54e5c00e`.
The isolated destination existed only as the verified restart-source copy;
the `reduced` and `logs` output directories did not exist when the method was
frozen.

The frozen EL-F10-R4 executable is 14,858,136 bytes with SHA-256
`71911a6768b7ecfff0d165a17d498adf5c0e8e0219e733e72d633d8b545c7636`.
No executable or science-code change is part of EL-F11.

The new analysis tool and its focused tests pass Ruff and byte compilation.
All 110 tests under `tools/fruit_loops` pass. The complete configuration
preflight passes, including 127 unit tests, eight compact-compatibility
fixtures, and all authority/boundary audits. No C++ source changed, and the
already frozen executable retains the complete EL-F10 build/CTest evidence.

## Frozen launch

Exactly one local replay may be launched with this argument order:

```text
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.2/setup/citlali-el-f10-r4
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_BASE.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_INPUTS_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_COMMON_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_FITREPORTS_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_ALPHA_1P25_INJECTED.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_OFF_SOURCE_INJECTED.yaml
/Users/gwilson/.codex/worktrees/4c31/citlali-refactor/validation/fruit_loop_point_123424_el_f11_prospective_influence_2026-09-04/EL_F11_JINC_ACCOUNTING.yaml
--grppiex seq
```

The final override fixes one configured thread, the isolated output root, the
copied iteration-3 restart path, exclusive `max_iters: 5`, and the existing
a1400/UID-4460/scan-5 JINC accounting target. All other science, injection,
learning, mapmaking, mask, weight, and processing settings remain inherited
unchanged from the registered stack.

The run is bounded to one hour, 64 GiB peak memory, and 8 GiB retained output.
No value from the new accounting receipt may be opened until replay outputs
are hash-bound in a successor registration. No Unity activity is authorized.
