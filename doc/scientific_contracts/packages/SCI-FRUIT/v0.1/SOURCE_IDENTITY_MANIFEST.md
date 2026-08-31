# SCI-FRUIT v0.1 — Stage A Source Identity Manifest

Manifest identity: `SCI-FRUIT-STAGE-A-SOURCE-IDENTITY v0.1/r0.1`

Status: exact recovery-source binding; not an author-packet or scientific
authority manifest

## Launch Identity

| Object | Exact identity |
| --- | --- |
| Launch commit | `7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5` |
| Launch tree | `03b77c9187eb5421488641d2ea1fe4dcb572a9a9` |
| Dedicated branch created from launch commit | `codex/sci-fruit-v0.1-stage-a` |
| Conditional SCI-FLT-FIXED freeze target | `43f4fe59ab23a591c1c9e17a2ac4b1fed0a9e613` |
| Conditional SCI-FLT-FIXED freeze record commit | launch commit above |

The manifest binds launch-base files by `7f9307ff...:<path>` even when the
current Stage A branch later updates a program index or roadmap.

## Governing And Adjacent Authority Sources At Launch

| Role | Exact ref:path | Git blob | SHA-256 | Disposition |
| --- | --- | --- | --- | --- |
| Program charter | `7f9307ff...:doc/scientific_contracts/README.md` | `29bca574c554f04b84a18248c2359a1d5da55d83` | `351e9b7775b0bf78cba01bf4cd2fafd9591c4b43931b0dc23d82d97f0dfe82d2` | govern |
| Launch-time roadmap | `7f9307ff...:doc/scientific_contracts/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md` | `48c189776bfa491fdeedaa705c93ace478e90dd2` | `0c0a7551689523ac16c72569834a687fd598a647d3f3c7dca3cd81cf5609a691` | cite; owner-directed order updated separately |
| Prior-work registry seed | `7f9307ff...:doc/scientific_contracts/PRIOR_WORK_REGISTRY.md` | `0b532abf8d0b3d88f07674cea160266e2dae15a2` | `8cd9868879a3fe7d5caaa2e6468d886d4b769242e65cc19968c6e6bb28cf8897` | discovery seed |
| Scientific conventions | `7f9307ff...:doc/SCIENTIFIC_CONVENTIONS.md` | `cd89bf353b48caeaa025f5d11875aec79b7776b7` | `24c8397b130de0fb1c0dcfcd87c057c06e4f095ee6a54472759a6ef276bb5add` | adopt exact iteration/state distinctions |
| Restart ADR 0006 | `7f9307ff...:doc/adr/0006-fruit-loop-restart-checkpoint.md` | `0f367980e073afd4b08cb9a1597d2502436532ac` | `c6abf5d5c0f9edd1e68cf080ed6d76d2cac93dc59e3bc398162ce92d6bb8ca2b` | adopt lifecycle identity; implementation state remains evidence |
| Frozen PTC r0.5 record | `7f9307ff...:doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` | `532c61bc4fe74b90970cefb49b03744b8097b0bf` | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` | cite boundary; freeze target `8f0ecccfacbdce0543141c4289ec06c702065f5e` |
| Frozen MAP r0.7.1 record | `7f9307ff...:doc/scientific_contracts/packages/SCI-MAP/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.7.1.md` | `bbbbc48c7e01dda30cc052f9974e5cc8f7dd311e` | `91801005ba2f2bce6471a9f6f4ed0b79806c893f498b4f3cca9e81e26df39ce1` | cite route boundary; freeze target `bd010e20eb8a7901aa677810aa7a5c982a436e07` |
| Frozen JINC r0.3 manifest | `7f9307ff...:doc/scientific_contracts/packages/SCI-JINC/v0.1/FREEZE_AUTHORITY_MANIFEST_R0.3.md` | `6fecb2083b9551946a0f9ca59b97451a0c61eae3` | `ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2` | cite route boundary; freeze commit/tag `a9f43877e01a661db13bd85b2e7f34ea5ac82fb7` / `sci-jinc-v0.1-r0.3` |
| Approved NOI Stage A launch record | `7f9307ff...:doc/scientific_contracts/packages/SCI-NOI/v0.1/SCIENTIFIC_OWNER_STAGE_A_FINAL_APPROVAL_2026-08-30.md` | `6284aee433b5a9a56bf34279fe6508d4f83d92f3` | `49377d1596c9e47a6e2328e890ebcd6b25f42af3781b533b5bd8c2cded08fa6b` | cite exact approved Stage A boundary; do not claim launch-base freeze |
| Approved NOI FRUIT scope | `7f9307ff...:doc/scientific_contracts/packages/SCI-NOI/v0.1/FILTER_AND_FRUIT_SCOPE.md` | `71013fd53c2399aaa3cce33ead20a39b51a360f8` | `08eba55f840e8f8aa265e1d2f1a981e16351a1c2460e74907cb4beb5ccb7df77` | adopt cross-package ownership/generation boundary |
| Conditional FLT-FIXED freeze record | `7f9307ff...:doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/SCIENTIFIC_OWNER_FREEZE_RECORD.md` | `db8a1f9803695a1dbd8598125df5a8502b07c943` | `ad00d2895982c2d26000fffb33a2dc73c716ea425f03a2a96b684563b2dfaf39` | cite exact conditional scope/unavailable states |
| Conditional FLT-FIXED machine binding | `7f9307ff...:doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/SCIENTIFIC_OWNER_FREEZE_BINDING.json` | `ce0283516d6346fb3239ef23f15cda2bbfad3a47` | `17298df3328bf812de8f896a1f931c41754bdee73b8dfddf40d48b03652e89cd` | exact binding |

## Exact Provisional FLT-MATCHED Snapshot

Snapshot commit: `faff97565ee27e375e1337febe5a0a6681507c3b`

Snapshot tree: `0dfa3cdfa8a261bd00878cafd593aafb87394163`

| Exact ref:path | Git blob | SHA-256 | Use |
| --- | --- | --- | --- |
| `faff97565...:doc/scientific_contracts/studies/SCI-FLT-INF_STAGE_A_2026-08-30/README.md` | `074ab040ff77e12cfdfa64215332b09cadd46ffc` | `d70873561f5e6c408fd64dc0ddc6e92827e29a5b023b324efbf64c7b9a7dcd34` | identify status, estimand, parent roles, open next question |
| `faff97565...:.../CROSS_PACKAGE_AND_NOI_BOUNDARIES.md` | `e5f598af5de6622a20f0fe634d41ab8a7f19e2cf` | `af862ffb29f690a94945fa6122e6858492a99aca5c8caae66e9963f5740a6929` | classify provisional FRUIT boundary only |
| `faff97565...:.../SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `79e7367244ad717eecae8c95b0e5d9d06659479a` | `bbcae3582eb7db058d1681a6be85e895aff251a1f544d49cfacab1f33e70dc16` | preserve open learned-state/lifecycle decisions |
| `faff97565...:.../STAGE_A_SOURCE_MANIFEST.md` | `b37ee64e0fd9f8ef69dd80e42b0f31c773efa77a` | `b12e01a7dbb25ed4351d8bdc902d742be8ecc42b71e93afa060a8086109161e1` | bind exact holding study sources |

These objects remain provisional evidence. This manifest does not approve or
promote them and does not follow a mutable branch name.

## Historical FRUIT Evidence At Launch

| Exact ref:path | Git blob | SHA-256 | Disposition |
| --- | --- | --- | --- |
| `7f9307ff...:doc/FRUIT_LOOP_FEEDBACK_INVESTIGATION_2026-07-24.md` | `f80a9c2627dacabd07f6db63838d4e705d52db3d` | `8ec6b4949e259eb1c2a07f45fae175955ffdc356297cdd616a2689feae150326` | abstract questions; exclude from author packet |
| `7f9307ff...:doc/FRUIT_LOOP_CONVERGENCE_STUDY_2026-07-23.md` | `5e73d43f1618e84a260cd76e36f4c4b04da94150` | `ab350193c78047bda5fe49cc8ead02b7e8d05ca1d73ec1cd9b4a40fd40d81178` | abstract metric taxonomy; exclude behavior |
| `7f9307ff...:doc/FRUIT_LOOP_CONVERGENCE_CRITERIA_DISCUSSION_2026-07-27.md` | `46fb3004f3d232911bbc954bae4ccb73d5d34408` | `e1d1a20988d4c5a8340b6b4c0519b2d353a3f79fcf363e885ddcd7617db86a55` | abstract separations; defer thresholds |
| `7f9307ff...:doc/FRUIT_LOOP_CALIBRATION_REFERENCE_INVESTIGATION_2026-07-26.md` | `7349ec5c1b16b4abd62891530163e851757bb56f` | `26fe065c6f4b0fc32cff05349cbf00f8cc8656705d6cb52a55dcc5296f0501f8` | retain limitations; exclude calibration inference |
| `7f9307ff...:doc/FRUIT_LOOP_POPULATION_EXTENSION_PLAN_2026-07-26.md` | `36a69f0fbf4ebbe3103c96b0232632ada291f400` | `cd726b6141a479db6fe6ae9c5656f4249fcd17186627ae224da183cf33584c7d` | abstract validation strata; exclude operations |

Historical fruit-calibration reference commit:
`f70701ad488444f3e2528c6bbe3e798863c9e301`. It is an ancestor/evidence
identity, not a scientific authority ref.

## Historical Coordination/Audit Snapshot

Snapshot commit: `8c581bfb26f01b187f4f1e0565f4457bcc25f099`

| Exact ref:path | Git blob | SHA-256 | Disposition |
| --- | --- | --- | --- |
| `8c581bfb...:doc/audits/audit-ledger.yaml` | `39b31962d371970c0ba51997b0155eca9bc8b4c3` | `91636d50ebea9f4502ed2dbccde22e981850bef9a79b3cf7301d8b90c616c906` | inventory evidence only |
| `8c581bfb...:doc/audits/handoffs/SCI-FRUIT-001/SCI-FRUIT-001-XAUD-001.yaml` | `517c49a9834934347a13ce0a61b26d481bbdbb83` | `8b0919fcfda18e338dbf3f1a8538d86dd29660e9fb0d605e0620373d93b1dd18` | terminal/restart dependency evidence only |
| `8c581bfb...:doc/audits/handoffs/SCI-MAP-003/SCI-MAP-003-XAUD-007.yaml` | `85d0466ecd0a35e04410739f126cfc816771b790` | `801928c460745b748163027191d20f5873110db937aa508e95542206e5623498` | open terminal-parent relation evidence only |

## Implementation-Evidence Binding

All current code/configuration evidence in `INTERNAL_DOSSIER.md` is bound to
the launch commit and tree, not to mutable working-tree behavior. Exact paths
include the FRUIT engine detail implementations, iteration/map-loading and
feedback-validation pipeline headers, learning lifecycle, weight-validation
state, and restart checkpoint. This grouped binding is intentionally not an
author reference.

## Verification Rule

The package verifier checks the launch and provisional commit/tree identities,
the exact SHA-256 values above, local links, Stage B placeholders, route labels,
and the absence of an owner-approved/dispatchable Stage B claim. A later source
change requires a new manifest revision; it must not silently reuse `r0.1`.
