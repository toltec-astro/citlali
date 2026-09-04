# MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001 Source Authority Manifest

Status: completed audit under recorded owner disposition; shared-source repair outstanding

Recommended disposition: `ACCEPT WITH BOUNDED CONTRACT REPAIR`

This manifest binds the read-only authority set used by the audit and the
owner disposition that resolved the four disputed scientific meanings in
favor of frozen SCI-MAP r0.7.1 and SCI-JINC r0.3.  The disposition resolves
scientific interpretation but does not amend the still-conflicting shared
source.  This audit makes no implementation, validation, performance,
readiness, production, activation, or Unity claim.

## Audit identity and safety preflight

| Field | Exact value |
| --- | --- |
| Audited commit | `5f0fc20042b88fb6cd883c92d1b59b7f22832901` |
| Audited tree | `97a4d908061e51418f93afc1d97d27433af441b8` |
| Sole parent | `9a2780aa3bd8343fea87ac0b28b390384118c883` |
| Checkout | detached HEAD |
| Initial worktree | clean |
| Local `refs/heads/codex/refactor-mainline` at preflight | `5f0fc20042b88fb6cd883c92d1b59b7f22832901` |
| Current permitted changes | untracked files only under this audit directory |

The branch name is recorded only as a safety check; the commit and tree above
are the audit identity.  No ref was moved.

## Work order and governance controls

| ID | Role | Path or exact object | SHA-256 |
| --- | --- | --- | --- |
| CTRL-001 | complete consolidated owner-approved work order and incorporated owner disposition | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/WORK_ORDER.md` | `b5cfdc0d2e9b72984b48bbe46e6d5750699828e47370e36996f72fc0b7196d4f` |
| CTRL-002 | exact source attachment preserved as the first 28,899 bytes of CTRL-001 | `/Users/gwilson/.codex/attachments/cb832940-341b-4419-8bae-5a5f87b7f968/pasted-text.txt` | `400388f1172bd155866f770debbd5754c0cf86ee364e31b5a6d2bdadc2c82713` |
| CTRL-003 | effective engineering governance | `doc/governance/ENGINEERING_GOVERNANCE.md` | `70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76` |
| CTRL-004 | effective review and conformance governance | `doc/governance/REVIEW_AND_CONFORMANCE.md` | `691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1` |
| CTRL-005 | scientific-contract program instructions | `doc/scientific_contracts/README.md` | `351e9b7775b0bf78cba01bf4cd2fafd9591c4b43931b0dc23d82d97f0dfe82d2` |
| CTRL-006 | `MSP-OD-001` owner resolution and resume authority, incorporated in the final section of CTRL-001 | `doc/scientific_contracts/audits/MAP_SPACE_HORIZONTAL_AUDIT_001/WORK_ORDER.md` | `b5cfdc0d2e9b72984b48bbe46e6d5750699828e47370e36996f72fc0b7196d4f` |

The accepted governance digests are recorded on integration authority at
commit `06a3ade51...`; CTRL-003 and CTRL-004 match those accepted digests.
This is Tier 2 scientific review.  It opens no application work-in-progress
slot.

## Authority precedence and classifications

The work-order precedence is applied without changing source bytes: exact
frozen package manifests and owner freeze records; current frozen package
sources; exact admitted predecessor boundaries; manager records for routing
only.  The source classes used below are:

- `NORMATIVE`: frozen package scientific authority.
- `REPRESENTATION`: formal-core wrapper, scientist rationale, or engineering
  conformance view; the last is inspected only for representation fidelity.
- `BOUNDARY_ONLY`: admitted only for the exact shared/boundary role named.
- `PROCESS_LOCK`: exact Registry/source identity and evaluation state; it adds
  no package-local science.
- `MANAGER_ONLY`: inventory or sequencing evidence, not science.

`doc/SCIENTIFIC_CONVENTIONS.md` is admitted only as a boundary/shared-
convention source.  It is not allowed to reopen or override frozen package
science.  `MSP-OD-001` resolves its four conflicting meanings in favor of the
frozen MAP/JINC authorities.  The conflicting clauses remain unrepaired and
are recorded as MAJOR `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`
findings in `FINDINGS_REPAIRS_AND_OWNER_DECISIONS.md`.

## Frozen package authority inventory

| Package | Frozen identity | Exact frozen revision | Controlling binding | Set digest | Verification status |
| --- | --- | --- | --- | --- | --- |
| SCI-MAP | `v0.1/r0.7.1` | candidate promoted at `bd010e20eb8a7901aa677810aa7a5c982a436e07` | `SOURCE_MANIFEST_R0.7.md`; owner freeze `SCIENTIFIC_OWNER_FREEZE_R0.7.1.md` | `bd3f172f8bb1e17bf95fde034ad76632439c766ef89b0a992361b9e6d79ada0a` | local read-only verifier exit 0: 52 REQs, 25 PREDs, shared-view and exact boundary/Registry checks PASS |
| SCI-JINC | `v0.1/r0.3` | commit `a9f43877e01a661db13bd85b2e7f34ea5ac82fb7`, tree `70c750b1...d7d28`, tag `sci-jinc-v0.1-r0.3` | `FREEZE_AUTHORITY_MANIFEST_R0.3.md` | `ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2` | frozen verification record PASS: 44 REQs, 36 PREDs, 16/16 inputs; no separate executable in the frozen set was rerun |
| SCI-FLT-FIXED | `v0.1`, conditionally frozen | target `43f4fe59ab23a591c1c9e17a2ac4b1fed0a9e613`; approval-record tip `7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5` | `stage_b/AUTHORITY_MANIFEST.json`; external owner binding and record | `69e6766f26396ba843ee29cfb89a48efd91b7e1b517ed90d3d93c87a63e55778` | frozen verification record PASS; rerun not completed because the mandated local environment lacks `reportlab`; no contrary result |
| SCI-FLT-MATCHED | `v0.1/r0.6` | exact 46-object frozen set at audit base | `STAGE_B_DRAFT_MANIFEST.md`; `SCIENTIFIC_OWNER_FREEZE_2026-09-01.md` | `6b0231a7e9d34f028eda9cce48f62de1fc9e594348aa1448a2d182d732f78688` | local read-only verifier exit 0: 46 objects, owner dispositions complete, frozen authority PASS |
| SCI-NOI | `v0.1/r0.5` | snapshot `2303daf7061a19945a6333099d33dd559cf2abf8`; approval-record tip `f28d7a2617160febca85c1c40e6f7ba7494e266e` | `stage_b/r0.5/PROPOSED_FREEZE_MANIFEST.json` plus post-snapshot approval | `b6915186424dd52d7c94fb0df47db91654d3c20cf4b3fa6ab98c3554626d8bfc` | local package verifier exit 0: 51 REQs, 26 PREDs and deterministic identity PASS; temporary verifier products were confined to `/private/tmp` and retained no audit PDF |
| SCI-POINT | `v0.1/r0.4` presentation freeze over unchanged accepted r0.3 science | r0.4 introduction `c7582052d48c991e0caec6f2b56ab63d2d44afcd`; subtree `e96ee5c369eba288e20b7169842c272083e5194b` unchanged through audit base | `STAGE_B_SOURCE_MANIFEST.json`; r0.4 view-separation directive/record | `76811b925834c7572b422aba3b23820b041307348bdde3da2fb1300263bf1828` | local read-only verifier exit 0: 38 REQs, 32 PREDs, 23 UNAV states, view/delivery identity PASS |

Verifier output retained outside the repository:

- SCI-MAP: `/private/tmp/map-space-horizontal-audit-001.imoOOX/verify_sci_map.out`
- SCI-FLT-MATCHED: `/private/tmp/map-space-horizontal-audit-001.imoOOX/verify_sci_flt_matched.out`
- SCI-POINT: `/private/tmp/map-space-point-audit.FwgKyG/verify_stage_b.out`
- FLT-FIXED/NOI task scratch root: `/private/tmp/map-space-horizontal-audit-001-filter-noi`

These results establish identity, completeness, and package-internal
consistency only.  They do not establish horizontal coherence.

## Formal core and audience-view bindings

| Package | Shared formal core | Scientist-facing rationale | Engineering-conformance view | Fidelity result |
| --- | --- | --- | --- | --- |
| SCI-MAP | wrapper `src/SCI-MAP-v0.1_SHARED_AUTHORITY_r0.7.1.tex` `08fcc9782cfba806d33dc07652a2363c8bd6540084f54e752e1fa91a5336b6bb`; ordered six-module aggregate `649e1694b2a6353b0e5e8cb42ab73d3f03b1bc5cad671810361e802a87278e9b` | `src/scientific-rationale.tex` `652c94b65e9742f6232c4a97027335536ffa584c2b30222edca1947c80a10cd0` | `src/engineering-conformance.tex` `3e67d0b0984278c3a2f16f6b1d001cbd228639e9c3456e40e2987f5046ce2c42` | PASS: both views import the same wrapper; no ECS strengthening found |
| SCI-JINC | ordered six-module aggregate `ca6650743af30e34940b7360a92c66f6638e993e07648b329e05f107b3b9e657` | `src/scientific-rationale.tex` `7cabea85eaa5ad9afbb0914c585d2fe7917806c9919964a465c0d9742fdb55e2` | `src/engineering-conformance.tex` `a8cc9b66d22f1c4c0e9dc53c46724721f38fa2b2d267f74e7341b359874c19aa` | PASS: identical ordered common imports; shared modules control |
| SCI-FLT-FIXED | `stage_b/source/SHARED_NORMATIVE_CORE.md` `7147d242f54d64ca80f6d3a17d309c65f75180457629fed62ce676da93b11089` | `stage_b/source/SCIENTIST_RATIONALE.md` `9609a22e7ff4db1c1413f57e5110405b1a74d9af7c0f3d2256462e5a61700235` | `stage_b/source/ENGINEERING_CONFORMANCE.md` `6796518dc6fe3f34558cff01c62bb78deed6e152ee2e01b5f89ae9d0b12c56a9` | PASS: ECS is an indexed representation of the bound core |
| SCI-FLT-MATCHED | ordered six-module set bound by manifest | `src/scientific-rationale.tex` `8c1aa33d9785e8fdda074afaa4119265b91cd3e4a3c0b798cfef69f8b86e9ae4` | `src/engineering-conformance.tex` `4f34b8d4f058117b592445ca5ad315913919c9b4a7a59ca10a682d53cb9b02a8` | PASS: identical common imports; no independent ECS science found |
| SCI-NOI | six modules bound by `NORMATIVE_MODULE_BINDING.json` `aa59ecaaaa149e2990d07623563d90af76c7b3084ee37c497a06e17ebf0fe213` | `stage_b/r0.5/SCIENTIFIC_RATIONALE.md` `d160c60410939750bf0779fbbbe1310a1ca1c4d83cac5388430acfc0854fd9d7` | `stage_b/r0.5/ENGINEERING_CONFORMANCE_SPECIFICATION.md` `de49c2c94cd59eca76d11157e964d73ff745e29b3d75aa134bdfd14f2349ad95` | PASS: view hashes and module binding close; no ECS strengthening found |
| SCI-POINT | ordered six-module aggregate `c0ca71bd457b8e6d37a425eb3ead76400dba3a5e29c869420807928201cdcdbd`; `src/normative-core.tex` `9c1e0e2fa73e393691660333f26e7bd9c34cc5c3e4cc8dbd8adbd22da71b69cb` | `src/scientific-rationale.tex` `3a11b4997531d5ace1a9f8a373abf531da55baeea5de1e1df0897119d0808adf` | `src/engineering-conformance.tex` `48918f10085168f326c0dde92c853666ee30ce9d7d902f52d81d10e113c3df39` | PASS: r0.4 separation preserves byte-exact r0.3 core; ECS is indexed |

## Admitted source ledger

Every row between the markers is admitted in exactly the class shown.  A
package manifest transitively binds additional frozen objects, but those
objects were not independently broadened by this audit.

<!-- BEGIN-ADMITTED-SOURCES -->
| Source ID | Class | Repository-relative path | SHA-256 |
| --- | --- | --- | --- |
| SRC-001 | BOUNDARY_ONLY | `doc/SCIENTIFIC_CONVENTIONS.md` | `affe9c5fa144fd2fe196b8cccaf4dc9bc9ec9970634ef7db9386ac9c5e2a1f53` |
| SRC-002 | NORMATIVE | `doc/scientific_contracts/packages/SCI-MAP/v0.1/SOURCE_MANIFEST_R0.7.md` | `bd3f172f8bb1e17bf95fde034ad76632439c766ef89b0a992361b9e6d79ada0a` |
| SRC-003 | NORMATIVE | `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.7.1.md` | `91801005ba2f2bce6471a9f6f4ed0b79806c893f498b4f3cca9e81e26df39ce1` |
| SRC-004 | NORMATIVE | `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/requirements.tex` | `68acf81d6c27788495cc680a1819da5f269616493b9aba5a6e0a1d1058ba5fa7` |
| SRC-005 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/formal-scientific-engineering-contract.tex` | `99fdb64e42c005afb37740170aa69ba41c50e9c53424dbe936a76ee2a7664229` |
| SRC-006 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/scientific-rationale.tex` | `652c94b65e9742f6232c4a97027335536ffa584c2b30222edca1947c80a10cd0` |
| SRC-007 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/engineering-conformance.tex` | `3e67d0b0984278c3a2f16f6b1d001cbd228639e9c3456e40e2987f5046ce2c42` |
| SRC-008 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY.md` | `a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7` |
| SRC-009 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY.md` | `a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7` |
| SRC-010 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE_BOUNDARY.md` | `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4` |
| SRC-011 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-MAP_UPSTREAM_ADMISSION_PROFILE.md` | `0717476c0a1d177074ee8702c18308f093d45a4913b22933f3fda3d33090a883` |
| SRC-012 | NORMATIVE | `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-MAP_COADD_PROFILES_R0.7.md` | `d93c04488925931676b02dff433774ff2cda9846fdd1d3f34bff29d76efdd702` |
| SRC-013 | NORMATIVE | `doc/scientific_contracts/packages/SCI-JINC/v0.1/FREEZE_AUTHORITY_MANIFEST_R0.3.md` | `ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2` |
| SRC-014 | NORMATIVE | `doc/scientific_contracts/packages/SCI-JINC/v0.1/src/common/requirements.tex` | `207a85acb31a4f381b289781706c9f14058d330ff847e99023e9e5714c4d4dff` |
| SRC-015 | NORMATIVE | `doc/scientific_contracts/packages/SCI-JINC/v0.1/src/common/assumptions.tex` | `15b811ab6ace92aa2d1713ae19b92454cb865e8862b82a599f94eca1003a1765` |
| SRC-016 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-JINC/v0.1/src/scientific-rationale.tex` | `7cabea85eaa5ad9afbb0914c585d2fe7917806c9919964a465c0d9742fdb55e2` |
| SRC-017 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-JINC/v0.1/src/engineering-conformance.tex` | `a8cc9b66d22f1c4c0e9dc53c46724721f38fa2b2d267f74e7341b359874c19aa` |
| SRC-018 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-JINC/v0.1/SCI-PTC_TO_SCI-JINC_BOUNDARY.md` | `5769d413460e931745e0d401ea432b12d1077c15466247c49caa71b997d4ab1e` |
| SRC-019 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-JINC/v0.1/SCI-AST_TO_SCI-JINC_BOUNDARY.md` | `efffa7059b59c89793fa1d523fb3bb48235f1ab55f7d55060af1600cbfd470a5` |
| SRC-020 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-JINC/v0.1/SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md` | `2db95da7e5d1b980df79993907d45ac0ababc3aa05c189bfb62dcf04ff2c2e8a` |
| SRC-021 | PROCESS_LOCK | `doc/scientific_contracts/packages/SCI-JINC/v0.1/SCI_VAL_REGISTRY_BINDING_2026-08-28.md` | `ee8f20db5febdb51e39f7157449d6c2d03a0d17058605dd5531e9ab5ca439e30` |
| SRC-022 | NORMATIVE | `doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/AUTHORITY_MANIFEST.json` | `69e6766f26396ba843ee29cfb89a48efd91b7e1b517ed90d3d93c87a63e55778` |
| SRC-023 | NORMATIVE | `doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/SCIENTIFIC_OWNER_FREEZE_RECORD.md` | `ad00d2895982c2d26000fffb33a2dc73c716ea425f03a2a96b684563b2dfaf39` |
| SRC-024 | NORMATIVE | `doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/source/SHARED_NORMATIVE_CORE.md` | `7147d242f54d64ca80f6d3a17d309c65f75180457629fed62ce676da93b11089` |
| SRC-025 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/source/SCIENTIST_RATIONALE.md` | `9609a22e7ff4db1c1413f57e5110405b1a74d9af7c0f3d2256462e5a61700235` |
| SRC-026 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-FLT/v0.1/stage_b/source/ENGINEERING_CONFORMANCE.md` | `6796518dc6fe3f34558cff01c62bb78deed6e152ee2e01b5f89ae9d0b12c56a9` |
| SRC-027 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-FLT/v0.1/SCI-MAP_TO_SCI-FLT-FIXED_BOUNDARY.md` | `2c04689734359a6fa8139b502a691238a118002ae27d8cd58fe82c3d0dddfbca` |
| SRC-028 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-FLT/v0.1/SCI-JINC_TO_SCI-FLT-FIXED_BOUNDARY.md` | `8c9cffe3641311ece334827136eafd47752a13750df1dcf2f55107ecc115892f` |
| SRC-029 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-FLT/v0.1/SCI-FLT-FIXED_TO_SCI-NOI_BOUNDARY.md` | `a349064e5bd0711eec54cd4f63ab02f934a60c2b1b6d5eccae0b64c02b47acd8` |
| SRC-030 | NORMATIVE | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/STAGE_B_DRAFT_MANIFEST.md` | `6b0231a7e9d34f028eda9cce48f62de1fc9e594348aa1448a2d182d732f78688` |
| SRC-031 | NORMATIVE | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/SCIENTIFIC_OWNER_FREEZE_2026-09-01.md` | `7ca33ede8e3a88102c02ff84de3b3858ce7f582f70aec500c5a3cbfa1b1f7746` |
| SRC-032 | NORMATIVE | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/src/common/definitions.tex` | `252e39fe965a798526a45ee0ed7d19af3148e23d03b7a0e358a6d41d29f634e4` |
| SRC-033 | NORMATIVE | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/src/common/equations.tex` | `b3a7eef087abd95121edc1f47b84427c35549667bf4207facc7046d3cac63ec6` |
| SRC-034 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/src/scientific-rationale.tex` | `8c1aa33d9785e8fdda074afaa4119265b91cd3e4a3c0b798cfef69f8b86e9ae4` |
| SRC-035 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/src/engineering-conformance.tex` | `4f34b8d4f058117b592445ca5ad315913919c9b4a7a59ca10a682d53cb9b02a8` |
| SRC-036 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/SCI-MAP_TO_SCI-FLT-MATCHED-v0.1-r0.6.md` | `fc49930b728320ecf6bc2710fc83dbb56a4055070fe7b579ff0b2a27dc271fa8` |
| SRC-037 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/SCI-TEMPLATE_TO_SCI-FLT-MATCHED-v0.1-r0.6.md` | `b6e11b0c0d98a88f9921bb63349a4273f445f6a40abee417a1b3623c0546363a` |
| SRC-038 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/SCI-FLT-MATCHED_TO_SCI-NOI-v0.1-r0.6.md` | `3ef4525c0fabc4cba83c04c4b5feb601778a8d701aa1e8c4922e17090a797273` |
| SRC-039 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-FLT-MATCHED/v0.1/SCI-FLT-MATCHED_TO_SCI-FRUIT-v0.1-r0.6.md` | `ecc130f1275e795cd877da5088f3096bc3b36422b0dbdd485236303f3eadd190` |
| SRC-040 | NORMATIVE | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/PROPOSED_FREEZE_MANIFEST.json` | `b6915186424dd52d7c94fb0df47db91654d3c20cf4b3fa6ab98c3554626d8bfc` |
| SRC-041 | NORMATIVE | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/POST_SNAPSHOT_SCIENTIFIC_OWNER_FREEZE_APPROVAL.md` | `dba66966ed7082ea55b756c23f4cc9de6205022f1362ff0a69da63bc85190d2c` |
| SRC-042 | NORMATIVE | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/NORMATIVE_MODULE_BINDING.json` | `aa59ecaaaa149e2990d07623563d90af76c7b3084ee37c497a06e17ebf0fe213` |
| SRC-043 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/NORMATIVE_CORE.md` | `a46244c2b295c0efbbc0d2a6161ba4a6835080946356e239a34307cb2fcf2288` |
| SRC-044 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/SCIENTIFIC_RATIONALE.md` | `d160c60410939750bf0779fbbbe1310a1ca1c4d83cac5388430acfc0854fd9d7` |
| SRC-045 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/ENGINEERING_CONFORMANCE_SPECIFICATION.md` | `de49c2c94cd59eca76d11157e964d73ff745e29b3d75aa134bdfd14f2349ad95` |
| SRC-046 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-NOI/v0.1/SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md` | `0a6484058569930cee62e80e04ca2045c107fde67603f662473ae471406f905c` |
| SRC-047 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-NOI/v0.1/SCI-MAP_TO_SCI-NOI_BOUNDARY.md` | `4273c5a75ff10d00506e5aa8732690cd3f398ff5afbaa561af8f1434ec467e29` |
| SRC-048 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-NOI/v0.1/SCI-JINC_TO_SCI-NOI_BOUNDARY.md` | `7bf0ff489957943cee5abcd581b6b6b1fea0840969d62ced4d73072cff8b51f8` |
| SRC-049 | NORMATIVE | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/PROFILE_AND_REGISTRY_SUCCESSOR_TABLE.md` | `38d4e66613a9c290d470948a2e9b550384338b9362f4aca76fa1bd38cb29cec7` |
| SRC-050 | NORMATIVE | `doc/scientific_contracts/packages/SCI-NOI/v0.1/stage_b/r0.5/PROPOSED_PROFILE_SUCCESSORS.md` | `aa1c53e7400f8b38804bc0002f23731b6d4b2d04a87b663a67151e4c07065f75` |
| SRC-051 | PROCESS_LOCK | `doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md` | `95b2ddeca2039aa2b3614e3dfede530cb7e024e6db33ca22baed867f19d2709d` |
| SRC-052 | PROCESS_LOCK | `doc/scientific_contracts/packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md` | `7b91a324f35196a8c8a6e23c8abbbf5322fc601798e36d4ac821907a6090eadf` |
| SRC-053 | PROCESS_LOCK | `doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY_JINC_STAGE_A_Q002_2026-08-28.md` | `4b9a1ebecfc847c83b59da772afd9b031ab1830e8febbb12d1a47f70ce5a1110` |
| SRC-054 | PROCESS_LOCK | `doc/scientific_contracts/packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER_JINC_STAGE_A_Q002_2026-08-28.md` | `0e7ca29ee2e9cd02fb1b76cf87cc64fce6164407a7801f9b9a105ca646317e88` |
| SRC-055 | PROCESS_LOCK | `doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY_NOI_STAGE_A_R0_18_2026-08-30.md` | `5994f4dff49dff3a9c9da6fbb494671b14a2f926f325f1c7c4a9603a6c2a38c1` |
| SRC-056 | PROCESS_LOCK | `doc/scientific_contracts/packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER_NOI_STAGE_A_R0_18_2026-08-30.md` | `04eca2da9ce76afacf18ae90dc2dbcb702fedbf55e03acb28e14e7dbc459a7c3` |
| SRC-057 | NORMATIVE | `doc/scientific_contracts/packages/SCI-POINT/v0.1/STAGE_B_SOURCE_MANIFEST.json` | `76811b925834c7572b422aba3b23820b041307348bdde3da2fb1300263bf1828` |
| SRC-058 | NORMATIVE | `doc/scientific_contracts/packages/SCI-POINT/v0.1/SCIENTIFIC_OWNER_R0_4_VIEW_SEPARATION_DIRECTIVE_2026-09-03.md` | `3f2331d4dc2a926ebd840cddb9bd85c9bc7d2a88be28718074313e262c259193` |
| SRC-059 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-POINT/v0.1/src/normative-core.tex` | `9c1e0e2fa73e393691660333f26e7bd9c34cc5c3e4cc8dbd8adbd22da71b69cb` |
| SRC-060 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-POINT/v0.1/src/scientific-rationale.tex` | `3a11b4997531d5ace1a9f8a373abf531da55baeea5de1e1df0897119d0808adf` |
| SRC-061 | REPRESENTATION | `doc/scientific_contracts/packages/SCI-POINT/v0.1/src/engineering-conformance.tex` | `48918f10085168f326c0dde92c853666ee30ce9d7d902f52d81d10e113c3df39` |
| SRC-062 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-POINT/v0.1/AUTHOR_MAP_TO_POINT_BOUNDARY.md` | `0c6c47f6674fb14f895707bbf5ed174ed519dd6d61b087c6b53951b9daa6b998` |
| SRC-063 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-POINT/v0.1/AUTHOR_JINC_TO_POINT_BOUNDARY.md` | `e52bc99f20e5bd01145150ac2efc3e4cd050533b835f01c84a4a750d4beba95a` |
| SRC-064 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-POINT/v0.1/AUTHOR_FLT_FIXED_TO_POINT_BOUNDARY.md` | `e2419d2bb09c8db06aa70d251bedc3457f34bfc7b3559f4ddace17a5c9f5ec86` |
| SRC-065 | BOUNDARY_ONLY | `doc/scientific_contracts/packages/SCI-POINT/v0.1/AUTHOR_FLT_MATCHED_TO_POINT_BOUNDARY.md` | `456c940d81c3f2270978ebb31ad84f12d64a4ddce0137b72cdf45e30a2217356` |
| SRC-066 | NORMATIVE | `doc/scientific_contracts/packages/SCI-POINT/v0.1/AUTHOR_ROUTE_SPECIFIC_COMPATIBILITY_TABLE.md` | `b19b1c3f19d592e32fbbb44ba688fcfc469347735cd66424bffa0f4bfb53c86f` |
| SRC-067 | NORMATIVE | `doc/scientific_contracts/packages/SCI-POINT/v0.1/AUTHOR_POLICY_PROFILE_RECORDS.md` | `bc731f720d46e9e55f0fdc95be2f6b15a8492c4bb1c3ef6e5f753ef6dfcef231` |
| SRC-068 | NORMATIVE | `doc/scientific_contracts/packages/SCI-POINT/v0.1/STAGE_B_R0_3_RECORDS.json` | `b80094ec2af23dac256a4ecac488183229c6ea22dc2c24d7660e48bcc8ff57c4` |
| SRC-069 | MANAGER_ONLY | `doc/scientific_contracts/INDEX.md` | `4c8eda81517fbd8575827835ea3d60e9802bcc9541d3270672abc0a821bbfcb2` |
| SRC-070 | MANAGER_ONLY | `doc/scientific_contracts/POINT_NOI_FLT_FIXED_INTEGRATION_CANDIDATE_2026-09-03.md` | `e11be4b7bb2fde72c4d439ac82505021b53039e2615d1f7627d12ef809a42d85` |
| SRC-071 | MANAGER_ONLY | `doc/REFACTOR_STATUS.md` | `2356da2b30e3420848779d041751a6b05027cbcdb4efdcb6e8c0b4e5105ce94a` |
<!-- END-ADMITTED-SOURCES -->

The MAP producer and consumer copies in SRC-008 and SRC-009 are byte equal.
The complete ordered module hashes for each package remain reproducible from
its controlling manifest; no historical or recovery source was used to fill a
gap.

## Explicit exclusions

- Active SCI-FRUIT science, worktree, branch, artifacts, and validation were
  not inspected.  Only the exact deferred FLT-MATCHED boundary file SRC-039 was
  admitted as an excluded attachment envelope.
- Historical ALIGN worktrees/branches and ALIGN package-local science were not
  inspected.
- PTC, AST, VAL, RTC, CAL, and ALIGN package-local science was not reopened.
- Application source, application tests, runtime behavior, validation outcomes,
  prior-work/recovery dossiers, active development branches, Unity, and remotes
  were not used.
- No PDF is an audit deliverable.  Existing frozen PDFs were read only by
  package identity checks; no PDF is present in this audit directory.

## Owner-resolved conflict and outstanding repair

SRC-001 conflicts with frozen MAP/JINC sources on product identity,
normalization, exposure/support, response, uncertainty, and product roles.
Those conflicts invoked the stop rule.  The owner then supplied `MSP-OD-001`,
which preserves the frozen package meanings and authorizes completion of the
audit without editing SRC-001.  No additional consequential conflict appeared.

Scientific/package coherence is therefore established under the recorded
owner disposition, subject to the explicit typed unavailable states.  The
repository documentation remains inconsistent until a separate, narrowly
scoped work order repairs only the affected clauses of SRC-001, updates their
cross-references as needed, and independently verifies that no unrelated
shared convention or frozen package meaning changed.  This outstanding
shared-source repair prevents an unqualified `PASS`.
