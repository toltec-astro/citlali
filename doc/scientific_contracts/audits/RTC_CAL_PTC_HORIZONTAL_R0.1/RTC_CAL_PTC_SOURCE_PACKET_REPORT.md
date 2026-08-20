# RTC–CAL–PTC Source-Packet Report

## Pin and clean-room boundary

- Branch: `codex/scientific-contract-library`
- Commit: `9564bcca0323dacb8bea13a5ec4bbbf3b908de8f`
- Ancestor condition: exact requested commit, therefore satisfies “commit 9564bcca0 or descendant containing it.”
- Audit worktree: `/private/tmp/citlali-scientific-contract-library`
- Package source state: no modifications under `packages/SCI-RTC/v0.1`, `packages/SCI-CAL/v0.1`, or `packages/SCI-PTC/v0.1`.
- Pre-existing unrelated worktree state was not used: modified working-tree `INDEX.md` and `PRIOR_WORK_REGISTRY.md`; pre-existing draft chain profile; untracked `SCI-VAL` package.
- The committed `INDEX.md` at the pinned commit was read first as navigation only. Its SHA-256 is `89aa8440845e6e81f6a010db0a3b0aa7ccb87951b303fd802a0ca8f7b6954e3b`.

No Citlali implementation source, tests, configuration, schemas, generated data products, implementation documentation, audit/repair history, Git history beyond the exact source pin, earlier integration notes, external literature, or web material was used as scientific evidence.

## Package preflight

| Package | Version/revision and status | Normative source | Amendments/ledger | Pair disposition |
|---|---|---|---|---|
| SCI-RTC | `v0.1/r0.9`; scientific authority frozen 2026-08-20; conformity unassessed | Six-file shared core imported by ECS r0.9 | R0.8 decision, R0.9 decisions, freeze, active owner ledger | Matched rationale/ECS/core; canonical PDFs 14 + 43 pages |
| SCI-CAL | Contract `v0.1`; active rationale `r0.3`; formal ECS `v0.1`; scientific authority not frozen | Six-file shared formal core; ECS draft | Supersession cover, V0.2 owner questions, active Q01–Q09 ledger, r0.3 changelog/crosswalk | Explicit package-controlled revision asymmetry, not an unidentified mismatch; remains draft |
| SCI-PTC | `v0.1/r0.4`; scientific authority frozen 2026-08-20; conformity unassessed | Six-file shared core imported by ECS r0.4 | R0.1/R0.2 owner reviews, detailed decisions, freeze, active owner ledger | Matched rationale/ECS/core; canonical PDFs 11 + 22 pages |

The committed global `INDEX.md` is stale for RTC: it describes an earlier RTC status, while the controlling clean package README at the same pin identifies `v0.1/r0.9` as frozen. The user-directed rule that each package README determines status and canonical artifacts resolves navigation, but the stale global index is recorded as a library-navigation deficiency. It does not make RTC package authority ambiguous.

## Source manifest

Role codes: `NAV` package navigation/status; `AMD` binding amendment/supersession/freeze; `LED` open/decided owner authority; `XWK` navigation crosswalk only; `RAT` scientist-facing rationale; `ECS` engineering view importing formal authority; `NORM` shared normative core; `PDF-M` PDF identity/QA manifest; `PDF` canonical rendered view; `ALT` compared alternate artifact, not independent authority.

All hashes are SHA-256 of the exact files in the pinned worktree.

### SCI-RTC v0.1/r0.9 — frozen

| Role | File | SHA-256 |
|---|---|---|
| NAV | `README.md` | `17c26be008a5a79e694666a8e17623633939fb04db7b7969264d266927ac6d4d` |
| AMD | `AUTHOR_SUPERSESSION_COVER.md` | `f183c8fb083c3a851fda5d77a0944405cc41650ced29bd0162cffba832f25575` |
| AMD | `SCIENTIFIC_OWNER_DECISION_R0.8.md` | `8862e3d4caf3fdd695fa66cbc0af58d40725375444f145525c4393f3859095b1` |
| AMD | `SCIENTIFIC_OWNER_DECISIONS_R0.9.md` | `90cad00151d975e0bb2a432c907f4a2198a1f3645f52c645c7e71cfa58ac57cb` |
| LED | `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `1bbe5f3f1c4b4b88e65e89f6f44ae1fe77ee5832a549b0d1541cc5a1eae00135` |
| AMD | `SCIENTIFIC_OWNER_FREEZE_R0.9.md` | `e64e8686a25ce4b1ab436442f4a7a27584a3c077f0be096a9f89ef08a8d66815` |
| XWK | `CROSSWALK.md` | `4a2645353a2cd3057c388644b2555c8a131dec4cc3c1f450c5bccf5386aaa668` |
| RAT | `src/scientific-rationale.tex` | `68453be8b3178a5d67b3e7c3499959e301f3b37b27438fc2e2786bcbb8c2c42f` |
| ECS | `src/engineering-conformance.tex` | `2237fdc377fb18a68be2eb2ffded85213f2816a00c03ec55579aeb595c247a23` |
| NORM | `src/common/notation.tex` | `3c3276f52712c932585f4447d52739b0863182bb61875f1bc00f1e5452530441` |
| NORM | `src/common/definitions.tex` | `b18a0394bfd74cf97e1e6f59f5ae7ec8d9e2dc6012160ad24cb250d3b10241b9` |
| NORM | `src/common/equations.tex` | `ac0de25324cc5575a3e61ab92296af58a9375eb0268f5b9e12150885c8e0afd5` |
| NORM | `src/common/assumptions.tex` | `8249d5ccefbae97a8e597bb44235173985e1dcb2dc1b9f5e8f1cee5d09aaa3ee` |
| NORM | `src/common/requirements.tex` | `5d3e2e4177df9f80a8dc7cb8bf077ff813005f609cd85bd01a7222169587d344` |
| NORM | `src/common/edge_cases.tex` | `e3f68625f090658c1ab2cb83363c1b0ff23b224f5f5d8f2e21cd600f88856535` |
| PDF-M | `pdf/README.md` | `3265679f5b75de11bc41883103ec96708eba00140319c6f7a9931369b04184a7` |
| PDF | `pdf/SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `0d397cbcf3eb5df19aa684c84efc317e95fcef7e404f3954a1356336ce09629e` |
| PDF | `pdf/SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `8ff6eb431f18ac64659f864d9fbd3f40c2349892fcc5154bc51ab3a9fc598805` |

### SCI-CAL v0.1 / rationale r0.3 — draft, not frozen

| Role | File | SHA-256 |
|---|---|---|
| NAV | `README.md` | `87a98577e746b24ffb59a28dcfc551a8ee3c033749dbe01455c0ab9cd92d9090` |
| AMD | `AUTHOR_SUPERSESSION_COVER.md` | `57dba2d9fdc837902cf0768a20a9680462929e647a6649c1cb51676fad4638b2` |
| LED | `SCIENTIFIC_OWNER_DECISIONS_V0.2.md` | `1b522fa4c61908e7291a4fa5af77cfa7a6f0c3eb3067844cfee7e2c60ba8f485` |
| LED | `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `bb3da56c7eea8150429610d787a2c92a7f599b23bace9d76024006b9c974a007` |
| AMD | `SCIENTIFIC_RATIONALE_R0.3_CHANGELOG.md` | `f45c7afa79bb8ce09a2b4bdb7c0ddd0336f7eba2986a8fad5aeae1cfd5050935` |
| XWK | `SCIENTIST_CROSSWALK_R0.3.md` | `9ee2edae235cce5d0ab61f9cbb78662b3d39e1f99542134b0dd38fbb99492266` |
| XWK | `CROSSWALK.md` | `98a3039f8fb716439fa68c637829cbfb27b3fa2f1e3c68830ca3cb71d1665976` |
| RAT | `src/scientific-rationale.tex` | `ea63e1260af3f897c9572639ed0c418561d35e14781e42019a8cea71b58f5374` |
| ECS | `src/engineering-conformance.tex` | `685b814a04ed85d2698aac157ba31bb0cf5a015af968133a6a4c17b597ec97ff` |
| NORM | `src/common/notation.tex` | `d7738c2eb0fd7791a4c717183e62dcab792bd55b76410ccc2ca3b51e94957fb2` |
| NORM | `src/common/definitions.tex` | `35ed1f896c2c42aa8127c1afbfbebc64b67253dd324a87ad50657eed313a4627` |
| NORM | `src/common/equations.tex` | `1291b578e1b95ca98044add1de71df36653767279c117b4b086df0da7e9e0510` |
| NORM | `src/common/assumptions.tex` | `74158d7ebccf9e929f220943d6c1b6dc462ca3d1722754803d7759bcc640699d` |
| NORM | `src/common/requirements.tex` | `cea59387a30daf26fb237f636502e09b8718699e628fdd10baf2b97e88c92c44` |
| NORM | `src/common/edge_cases.tex` | `fee215db9bce94bcfe8ae5d386c6a5ce2af6d17ddbbb52deaae764afcdb978a4` |
| PDF | `pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf` | `075efafcbe4f0f3897be3bb88604e00a575d5d623a2eaf78a11d25ed7c3284d3` |
| ALT | `pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1-r0.3-DRAFT.pdf` | `6a1322a275b308c23fc9c3fcf3020338f049bbe787b26b4555b00bdb1b5a600f` |
| PDF | `pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf` | `1a5cd02e7844c9c57b22bef5927fe159ed89577891ae68d9fc757dda2030a326` |

The stable canonical CAL rationale and the explicitly named r0.3 draft have different binary digests and creation times, but both identify themselves as the r0.3 draft and all 14 pages are pixel-identical when rendered at 72 dpi. The README designates the stable filename as the active canonical PDF; that file and the canonical source govern. The alternate file supplies no separate authority.

### SCI-PTC v0.1/r0.4 — frozen

| Role | File | SHA-256 |
|---|---|---|
| NAV | `README.md` | `055785ed1092067c32e1f0d14bbbc0dbcb1865dff205a729122497371851d111` |
| AMD | `AUTHOR_SUPERSESSION_COVER.md` | `2a13d3984c2334ccd1886021d2d869bb71363abd3a06bb7f9fbf536614d9ee3e` |
| AMD | `SCIENTIFIC_OWNER_REVIEW_R0.1.md` | `0adcd0816d190a1319ea34c0331fc520949492f87b54f4feb4cca9e37c0ebee7` |
| AMD | `SCIENTIFIC_OWNER_REVIEW_R0.2.md` | `cb757d7777163cc008f58e1afeaeda76e6e845ba814d56c0f4827b352961c782` |
| LED | `AUTHOR_DRAFT_DECISIONS.md` | `ea42dea6c88d22458fef85ec7d46e92bb8d487e9901eeb13e3b1ff4804d7c54c` |
| LED | `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `adb4c5fa53a1dd2e0b6c863bac95143eadf579b3c7db80bab8a9296761f6ff0e` |
| AMD | `SCIENTIFIC_OWNER_FREEZE_R0.4.md` | `90334ea7853e1ab274f6858fad66078356c06326438625c7fe294e41c07fbcc4` |
| XWK | `CROSS_PACKAGE_FOLLOWUP.md` | `1c2db81ecc4ad09f42d4d7448b49264796a11b7dca86f20afe4040f4afd51ccb` |
| XWK | `CROSSWALK.md` | `2211e823f045d78bc2afa491996667e7d931fc59c1f0e51deb896af2bf734125` |
| RAT | `src/scientific-rationale.tex` | `a2c7301448bcbf5402abce61182930a7f65475b5f7e206d4fded2f76b189d272` |
| ECS | `src/engineering-conformance.tex` | `ef219eed260722a503592c022d6e07e5789fc09d4f132f1bf491eaa9af0c6fac` |
| NORM | `src/common/notation.tex` | `108022499a5179bc8bbf44060bdc00680ec89c56486c3f833564e63d2e700df7` |
| NORM | `src/common/definitions.tex` | `38770c599c8e7b56357577114e799462368f745b703c6690c43d601b5ab4fe6f` |
| NORM | `src/common/equations.tex` | `4d56ab506f88d26a7af061dcc3b7a8a1e852255999dd4a975cc5cb3517ed3d14` |
| NORM | `src/common/assumptions.tex` | `f4f4ec3593419917071714a9586f22d31019487fb37c75f0dad836578d63e80e` |
| NORM | `src/common/requirements.tex` | `74a077b631bdbcfbdf72306d5dff1693ba93f66d86b9dc5384d63997e3268d62` |
| NORM | `src/common/edge_cases.tex` | `d6df04f82a219f1804e41e638095197c1d139d428c0bd693c6a794233d29c493` |
| PDF-M | `pdf/README.md` | `6bbd9cd24072fca5a9b96d069cbb4233bdadd83c0634c9df59f44cac579242aa` |
| PDF | `pdf/SCI-PTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `7cb358eec6633e06ca2559741d4f32ca2cf62607fac2fe6efb73365863832fd0` |
| PDF | `pdf/SCI-PTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `1e73d3e001dafce4dd6a9025553af95da58075fb49ea2b4eb41222431d658b85` |

## PDF identity and visual verification

`pdfinfo` confirmed the six README-designated canonical PDFs are unencrypted US Letter files with 14/43 RTC pages, 14/21 CAL pages, and 11/22 PTC pages. All 125 canonical pages were rendered with Poppler at 72 dpi and visually inspected in contact sheets. Titles, contents, formal registers, tables, equations, and final pages were present; no blank, clipped, rotated, corrupt, or obviously unreadable page was observed. This verification checks packet identity and readability only, not scientific correctness or implementation conformity.

## Source sufficiency and exclusions

The packet is sufficient to perform the requested horizontal audit because each package supplies an identifiable formal core/ECS, rationale, owner records, ledger, crosswalk, and canonical rendered views. It is **not sufficient to freeze a chain profile or authorize a numerical chain**, for substantive reasons recorded in the findings:

1. CAL is draft/not frozen.
2. CAL `Q01/Q02` leave the exact RTC input semantics and full chain order open.
3. CAL `ASM-011/Q06` blocks numerical calibrated output.
4. CAL does not guarantee the complete upstream response ending on the CAL grid or close RTC-filter/CAL-atmosphere noncommutation.
5. CAL does not fully guarantee RTC cause/full-parent carriage required by PTC.
6. CAL formal peak wording conflicts with its active rationale’s point-source-equivalent limitation.
7. PTC’s fixed-nominal-beam precondition is stronger than CAL’s originating response-basis guarantee.

Recorded RTC and PTC open, known-but-not-supplied, conditional, and deferred states remain open. No missing authority was synthesized from rationale wording, crosswalks, general scientific knowledge, or presumed practice.
