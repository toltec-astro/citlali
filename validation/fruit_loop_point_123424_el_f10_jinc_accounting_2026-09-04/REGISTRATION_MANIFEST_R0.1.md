# SCI-FRUIT EL-F10 registration manifest r0.1

Test ID: `SCI-FRUIT-EL-F10-TARGETED-JINC-ACCOUNTING-R0.1`

Status: **frozen before execution; no result exists at registration**

## Registered test

`REGISTRATION_R0.1.yaml` is 8,419 bytes with SHA-256
`5853f6250a3190315521f6f3206f9dbe93e6b2e6b9d583afb3c5b928c9964835`.
Its 19 registered input and method files were re-read successfully after the
isolated workspace was staged.

The replay uses source commit
`38bf8b68379e9fe5e0f361883e2ce2b1c05b0933`. The staged executable is
14,858,136 bytes with SHA-256
`bbc4a4ba83135cce1629b6431a3f5f3c8e38b65879f3d9dd080590a3aca97d31`.
Its reported Citlali version is
`sci-noi-v0.1-stage-a-84-g38bf8b683`.

The copied EL-F6 no-record iteration-4 checkpoint is 6,508,179 bytes with
SHA-256
`9f8faf73fc759202258ba58109ba499bd73d8f513d93ea763df75069ae78f942`,
identical to the registered source checkpoint. No prior reduction product is
an output target.

## Frozen launch

Exactly one local replay may be launched with this argument order:

```text
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/citlali-el-f10
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_BASE.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_INPUTS_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_COMMON_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_FITREPORTS_LOCAL.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_ALPHA_1P25_INJECTED.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/POINT_123424_OFF_SOURCE_INJECTED.yaml
/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f10-jinc-accounting-r0.1/setup/EL_F10_JINC_ACCOUNTING.yaml
--grppiex seq
```

The last overlay fixes one configured thread, the copied restart path,
exclusive `max_iters: 6`, and the disabled-by-default JINC diagnostic target
to a1400, UID 4460, zero-based scan 5. The output root is the new isolated
`fruit-el-f10-jinc-accounting-r0.1/reduced` directory.

## Gates frozen before the replay

The controlling order, exact finalization, sample ledger, forward-error
formulas, safety factor 16, regions, trigger pixels, summaries, resource
limits, and claim limits are in `TEST_DEFINITION.md` and
`REGISTRATION_R0.1.yaml`. They cannot be relaxed after seeing the output.

Scientific interpretation stops unless the diagnostic replay first matches
EL-F6 N5 bitwise in all nine ordinary science planes and all three formal
coefficient planes, and its scientific checkpoint differs only in the
registered creator-version field. The captured total accumulators must then
reproduce the ordinary diagnostic a1400 signal and formal coefficient
bitwise. Only after those neutrality and closure gates pass may the exact
target subtraction be compared with EL-F8 A5-map under the frozen binary64
bound.

This registration authorizes no second scientific replay. The sole allowed
replacement is for an environmental interruption, and no Unity activity is
authorized.
