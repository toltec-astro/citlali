# Six-Package Source Packet Report

## Immutable audit source

| Field | Exact value |
| --- | --- |
| Repository branch | `codex/scientific-contract-library` |
| Pinned commit | `55efd8a54464636a24e621f6d1b60486d235b20e` |
| Git object type | `commit` |
| Branch resolution at preflight | exactly `55efd8a54464636a24e621f6d1b60486d235b20e` |
| Commit subject | `docs: consolidate scientific contract library` |
| Commit timestamp | `2026-08-22 12:48:50 -0400` |
| Audit worktree | clean dedicated worktree at the pinned branch/commit before deliverables were added |
| Scientific owner named by program | Grant Wilson |

The branch name is navigation only; every finding is bound to the immutable SHA above. No moving branch-tip claim is made.

## Authority and clean-room method

Sources were admitted in the required order: shared normative core/ECS; explicit owner freeze, amendments, and decisions; exact approved boundary; rationale; then crosswalk/README/manifest as navigation and status evidence. Separate isolated extraction passes were completed for ALIGN, AST, RTC, CAL, PTC, and VAL before cross-package vocabulary was composed. MAP was extracted separately and used only as a downstream-consumer reference.

The following were excluded as scientific evidence: Citlali implementation; tests; configuration; schemas; generated products; validation results; external literature or web searches; prior horizontal audits; prior repair directives; chat history; and undocumented practice. `PRIOR_WORK.md`, `INTERNAL_DOSSIER.md`, implementation-derived review material, and old validation artifacts were not used to establish scientific meaning. Historical revision files were consulted only when an active supersession/freeze record required status interpretation; no superseded clause was promoted over the active core.

## Program-level navigation

| Exact path | SHA-256 | Status and admitted role |
| --- | --- | --- |
| `doc/scientific_contracts/README.md` | `80ffb1da1bf34e1b0c40d830a39715ab66452eacfda5728014358811c3e552d2` | Governing program charter and authority-order/process rule; not a package scientific authority. |
| `doc/scientific_contracts/INDEX.md` | `336a2b00dc1b83e1f793ba5900132e2f9fade309f12392f714b4f7a27a26e118` | Current package-status navigation. |
| `doc/scientific_contracts/CONSOLIDATION_LEDGER_2026-08-22.md` | `04710e4fc58471e6d934a3142c9ddbac9a0645237c1ecb06412773548d9681de` | Consolidation inventory only; expressly not package scientific authority. |

## Package status summary

| Package | Version/revision admitted | Status at pinned commit | Matching rationale/ECS/core disposition | Owner and active decision state |
| --- | --- | --- | --- | --- |
| SCI-ALIGN | v0.1/r0.3 | **Frozen** 2026-08-22; implementation conformity unassessed | One six-file r0.3 core imported by both canonical views; exact PDFs source-bound | Grant Wilson; ODQ-101--105 and 110 open, 109 deferred; 106--108 decided |
| SCI-AST | v0.1/r0.3 | **Frozen** 2026-08-22; implementation conformity unassessed | One six-file r0.3 core imported by both canonical views; exact PDFs source-bound | Grant Wilson; Q001--004, Q006--007 open; Q005 deferred; Q008 closed |
| SCI-RTC | v0.1/r0.12 | **Frozen** 2026-08-21; implementation conformity unassessed | One six-file r0.12 core; canonical rationale/ECS and PDFs; r0.12 freeze verification | Grant Wilson; ledger retains 63 open, one conditional, 34 resolved, five deferred entries |
| SCI-CAL | v0.1; rationale r0.5 / ECS r0.4 | Active, owner decisions Q01--Q09 incorporated; final scientific acceptance and freeze pending | Shared core reflects the active r0.5/r0.4 pair; consistency report says matched after bounded repair | Grant Wilson; no Q01--Q09 science disposition open, but several numerical uncertainty/evidence products unavailable |
| SCI-PTC | v0.1/r0.4 | **Frozen** 2026-08-20; implementation conformity unassessed | One six-file r0.4 core imported by canonical rationale/ECS; exact PDFs bound | Grant Wilson; 12 open/known/deferred items remain in named roles; this audit finds two frozen internal conflicts |
| SCI-VAL | v0.1/r0.3 | Owner-approved targeted revision and manager-reviewed; scientific-owner review/freeze pending | One six-file r0.3 core imported once by rationale/ECS; PDFs manager-reviewed | Grant Wilson; no general Core question open; profile authoring/source rebinding remains separate; serialization/equivalence details deferred |
| SCI-MAP reference | v0.1; rationale/formal r0.3, ECS r0.2 | Rationale house version frozen; scientific authority **not frozen** | r0.3 shared formal source and rationale, r0.2 ECS; downstream reference only | Grant Wilson; OD-001--009 open, CI-001 resolved |

## SCI-ALIGN admitted sources

All rows below are under frozen v0.1/r0.3. README and manifest administratively bind the exact preserved bytes, including embedded pre-freeze draft labels; those labels do not supersede the explicit freeze status.

| Exact path | SHA-256 | Normative/status role |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/README.md` | `be51b2347f04237ed5ae5773efb6978405f76666b3a92647721a482d25f7f9e0` | Frozen status, active artifact navigation, stale-label disposition |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SOURCE_MANIFEST.md` | `26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac` | Exact frozen source/PDF binding |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `cc9aa8df6d6e4258661554844f2e850b04ce5a1678c953f8f3ae08b2091cdb1f` | Active owner decisions/open questions |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/OWNER_DECISION_REGISTER.md` | `cc9aa8df6d6e4258661554844f2e850b04ce5a1678c953f8f3ae08b2091cdb1f` | Byte-identical owner register copy |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/CROSSWALK.md` | `8d66b0ff0aba3da541efac3b7587bbb3fdb42132e789c7195a57d4c485bb9974` | Rationale/ECS/core routing |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` | `04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36` | Approved transfer `SCI-ALIGN_TO_SCI-AST v0.1/r0.1` |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/BOUNDARY_IDENTITY_AND_EQUALITY_REPORT.md` | `3031afc67b7d9c87b59ef0f2ecd565db9dde81795b7787cd8c9709743fa76a5d` | Boundary-copy equality/status evidence |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/scientific-rationale.tex` | `88931d5c9d3e41b5986ff4d57cf414126b00c1799ffc0cb55aef161d22cc3e38` | Scientist-facing frozen view source |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/engineering-conformance.tex` | `173a2cf1dd8256149015e98a7c13cba8494e1153e5004700bee41e29d009842a` | Engineering-facing frozen view source |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/common/notation.tex` | `dbf8bd4d6124ea90a45007d0fcc2ad120b604667c930ab808c0da41dd2371baf` | Shared normative core |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/common/definitions.tex` | `9dd8e7dc50651a38a8f3dd8de10627dea9123a904b23acc9cd1bd9f4aa10f2db` | Shared normative core |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/common/equations.tex` | `da73a88cfc9ec4c794bdad47fe3d0c191de3bb06fb35a6d7c2a6804a15f00dde` | Shared normative core |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/common/assumptions.tex` | `326f698bbbe353669f7e84ca782bbe78f6c1464cf83407b74319d52009d47ea1` | Shared normative core |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/common/requirements.tex` | `667e4e96251a7d3f416aa9148f6429f23fc8793fccd722fe0d63852085c48e63` | Shared normative core; 55 requirements |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/src/common/edge_cases.tex` | `60dcca05d4d822b408c8388c2b90e35c8e6c2edbeb0110ed026b129b4be191c6` | Shared normative predictions; 26 predictions |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/pdf/README.md` | `41a416544f07b5072c286006bd6a71a27dd4477788040c43015df42c76aed243` | Canonical PDF/source identity |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/pdf/SCI-ALIGN-SCIENTIFIC-RATIONALE-v0.1.pdf` | `3ff4de1c6a487e14285c7c4f37771c8106e78d94f4299cd3d92604ee0b0c4538` | Canonical frozen rationale PDF |
| `doc/scientific_contracts/packages/SCI-ALIGN/v0.1/pdf/SCI-ALIGN-ENGINEERING-CONFORMANCE-v0.1.pdf` | `800f13a4133eac3e293533541f0e58fe90d0fd0a75afbe7f1068c0321de3b2a8` | Canonical frozen ECS PDF |

## SCI-AST admitted sources

All rows below are under frozen v0.1/r0.3. As with ALIGN, the exact frozen bytes retain some pre-review labels; README and manifest supply the explicit later freeze disposition without claiming byte changes.

| Exact path | SHA-256 | Normative/status role |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/README.md` | `f722589fb39df1d75c12c6f5a99797ee9bd1f304088edada8cf4788311b8b257` | Frozen status and active navigation |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/SOURCE_MANIFEST.md` | `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601` | Exact frozen source/PDF binding |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `55daf66c43de34963eb3f986bec283e8358b71ff896984480d900e2da6a773c0` | Active owner decisions/open questions |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/OWNER_DECISION_REGISTER.md` | `55daf66c43de34963eb3f986bec283e8358b71ff896984480d900e2da6a773c0` | Byte-identical owner register copy |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/CROSSWALK.md` | `322c5b700cbf25599dd44e66cfc059670b7d3a6a1ea0c440aff289489ddfac50` | Rationale/ECS/core routing |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` | `04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36` | Approved byte-identical ALIGN transfer copy |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/ROLE_FACTORED_PARENTAGE_MAP.md` | `cd181110cbbc6b4834bfd0ce1d150db79eb9c3946a9c8fd52676509ea5ae8bf2` | Approved role/parent map |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/BOUNDARY_IDENTITY_PROOF.md` | `223dae35be88be0dbf6ff55dd8eba65632c98727e45c27a26886316117148833` | Boundary identity proof/status evidence |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/src/scientific-rationale.tex` | `b8835be124e57245e1f9da850fb3a94f8d9407af10b4f569a3ec6f35b93a8ea9` | Scientist-facing frozen view source |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/src/engineering-conformance.tex` | `1b5cc58cd6d098bf7d4b289dd06dd1d155f13f36473cfae6c5d3947b401ece96` | Engineering-facing frozen view source |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/notation.tex` | `ba0bd3a366416860d612d1d94b723c4a72553290990c0663294901c2dc1586d7` | Shared normative core |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/definitions.tex` | `d03d04fb35026091e32e6071e53cbb40087dda8429d59c5a3c43f8151c07e5c3` | Shared normative core |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/equations.tex` | `ed3379f897f8d33a05efe61c11e4b3e27aecd02adada9eecc8dbf7ca7ed0ee83` | Shared normative core |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/assumptions.tex` | `1744f17660d707bbb3fc7b1316dcdc48246124005cfb0443b4565a82f2156593` | Shared normative core |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/requirements.tex` | `47b357dd79136fb3d019f45b1092a2efd88fbd1ed16ab038fc6bf51beaf06f01` | Shared normative core; 90 requirements |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/edge_cases.tex` | `9fd99ee4aa12e3e1f8d878614bfe70f9e95c4d0c3ad9296df83247abac91fbbf` | Shared normative predictions; 50 predictions |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/pdf/README.md` | `ad298df95791f375e71037df21e1e34795729b591bce72bd76e0fd03364a578c` | Canonical PDF/source identity |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/pdf/SCI-AST-SCIENTIFIC-RATIONALE-v0.1.pdf` | `40b1b1759715365722fbff778e75859d4095dbd227f79c6990866003015efeba` | Canonical frozen rationale PDF |
| `doc/scientific_contracts/packages/SCI-AST/v0.1/pdf/SCI-AST-ENGINEERING-CONFORMANCE-v0.1.pdf` | `08fb199054e2c79c226c78d76b7ecca5e00f70288db20ab435715baf83edc5c9` | Canonical frozen ECS PDF |

## SCI-RTC admitted sources

The active scientific authority is the r0.12 shared core and its canonical views. Older change/review material is superseded history except where the r0.12 freeze or ledger incorporates a decision by reference.

| Exact path | SHA-256 | Normative/status role |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/README.md` | `eeb5b4e51fb6725190bb9956715738bd0a18b844ba1af94ce7e9d0d23ef0331a` | Frozen r0.12 status and active navigation |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/AUTHOR_PACKET_MANIFEST.md` | `931de6e1045dd8527b52ea7897fe4bbe00c630639e59ac09b687c2fad0c397f0` | Approved author-input binding; not by itself final output manifest |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/AUTHOR_SUPERSESSION_COVER.md` | `f183c8fb083c3a851fda5d77a0944405cc41650ced29bd0162cffba832f25575` | Stage-B supersession/firewall record |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.12.md` | `0cac4396df225c1f2808ee1055e063c9a4e72a02549557c5e997f54d72dac0bf` | Binding scientific-owner freeze |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/SCIENTIFIC_OWNER_REVISION_DIRECTIVE_R0.11.md` | `4178f6b3f7de50265d9c7de1f40da222c2f0e0c539f54015372f66375c368ea0` | Active r0.11 amendments inherited into r0.12 |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/SCIENTIFIC_OWNER_REVISION_DIRECTIVE_R0.12.md` | `48cc2878d47daf324e63d0ac60b975f2a433371bc542fd25ba4d0e85ad7c4339` | Active bounded r0.12 correction authority |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `3d4ac36773cc4a0e0d116f86ea9af0013cbcc520cff4e9bfd5a4c4d4f6870956` | Active 103-entry owner ledger |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/CROSSWALK.md` | `68b71422e4f7f46d75a52360a571b25434228a06c82ee0f7798985f1ae1062fe` | Canonical requirement/prediction routing |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/RATIONALE_TO_CONTRACT_CROSSWALK_R0.12.md` | `8946e0811b1ed57e454af56bac7806dc8a6fa4ba4cffe366a0ac2bb3436545c6` | r0.12 rationale-to-core routing |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/FREEZE_VERIFICATION_R0.12.md` | `e9bba3f833317dbe6a6597dcb50a33f5833551e6a8e86aabb11d07eafb894229` | Source/structure/freeze identity record; not implementation conformance |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/scientific-rationale.tex` | `cfce4f3c3919b09f09e200a20abd278845f40f59c05a4388ec7fdba7160d8552` | Scientist-facing frozen view source |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/engineering-conformance.tex` | `b5cc1256df031ac46700c0b4f140e67bd37831c8299f5d07e8b2c23625f0a12b` | Engineering-facing frozen view source |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/common/notation.tex` | `8e252c7da6289f5fe206e457146ea48094295aa1b6c8a6796d9971141bc2064a` | Shared normative r0.12 core |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/common/definitions.tex` | `8b744d8dca7c7187f7c4a6d3560bcdb6f699cb54090e5083ce883fc9f6196233` | Shared normative r0.12 core; 52 definitions |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/common/equations.tex` | `b83f0d41efc98a0085cc7548ce9b8fbb1c10f342f0f6fcd52d2c02bf6c1802bf` | Shared normative r0.12 core; 44 displayed equations |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/common/assumptions.tex` | `016064c558d02db6dd43054003e2584cde91e2d1ccc936f07cba2161856b2dc6` | Shared normative r0.12 core; 12 assumptions |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/common/requirements.tex` | `e3c8838ae661046394dd6dfd6a2f0ef1409cdd798df1419ae4ad29757b3f6627` | Shared normative r0.12 core; 143 requirements |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/src/common/edge_cases.tex` | `ff31d46c137fc259962775b2e5c28f566605d2c6727da22210d93649628045e5` | Shared normative r0.12 predictions; 108 predictions |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/pdf/README.md` | `3094da0c23cee5a590004935a8a8d9f79b27ba4399e70a8440faf6bb71e083cb` | Canonical frozen PDF/source binding |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/pdf/SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `b0060b28253906f83f2f106d9df761864d8277317ebd5e3742ff963e11e30b3d` | Canonical frozen rationale PDF |
| `doc/scientific_contracts/packages/SCI-RTC/v0.1/pdf/SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `9211091e71830295a8fe5febb102704c95f8397b017584cbeb4575728081da42` | Canonical frozen ECS PDF |

No exact standalone RTC-to-AST sample-grid boundary artifact, RTC-to-CAL boundary artifact, or package-local VAL profile is present. The RTC core itself contains detailed handoff clauses; absence of a standalone boundary is recorded separately rather than filled by inference.

## SCI-CAL admitted sources

The active pair is science rationale r0.5 and ECS r0.4. Q01--Q09 are owner-decided, but the package is not scientifically frozen. The author manifest binds inputs; no final current source/PDF manifest equivalent to frozen ALIGN/AST/RTC/PTC exists.

| Exact path | SHA-256 | Normative/status role |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/README.md` | `a2f0eb27e88056affecf5bb9bfe83c6556372a1d508b694c7ef3d19b7b2e6f51` | Active r0.5/r0.4 status and navigation |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/AUTHOR_PACKET_MANIFEST.md` | `fbb4868d46d2ba117d8c866efa827c50cba5eeececd76777693182d082ff1262` | Approved author-input binding |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/AUTHOR_SUPERSESSION_COVER.md` | `57dba2d9fdc837902cf0768a20a9680462929e647a6649c1cb51676fad4638b2` | Stage-B supersession/firewall record |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `7e9a630fd183ca04bc3d8bbd21b5e801776b9aad7bd084d48fa7f2c572766520` | Original author conventions; later approved decisions supersede conflicting bounded opacity policy text |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_DECISIONS_R0.5.md` | `9124c28ee9dab3b8499b7c89055aaa70ebc0684463a5b3a7d4ab678cbb175e4e` | Active approved Q01--Q09 dispositions and external content identities |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `21a58f718e99c1eec7454a30b9b3cdf3ceacdb6f7ce01e26f4d0c38ddf1435e6` | Active owner ledger and unavailable evidence/products |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/CROSSWALK.md` | `c4c413310fc387accc76224e1a3c562a52f58508a7f4322498da1a96bd81fd18` | Canonical routing/navigation |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIST_CROSSWALK_R0.4.md` | `0d4bd4fff398458218e52c8f92fce53fcfe37462dddcc751a7a2e5f9912a333a` | Active grouped rationale/ECS routing aid |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_ENGINEERING_CONSISTENCY_R0.5.md` | `78c25f1e9978dec5326b78cedd1a21e0a03dcbce26c25d9b9f9ca91f3f70d290` | Active r0.5/r0.4 consistency/status review; not implementation conformance |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/scientific-rationale.tex` | `edeed1dbdd2706b9a518abc95f3369db7c1c83002fb7cb34ba02b8054020ca24` | Active scientist-facing r0.5 source |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/engineering-conformance.tex` | `3dc656da692bfaa5802f1578d5ae022c45576c22bc4d13c0f5419257449b1ef3` | Active engineering-facing r0.4 source |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/notation.tex` | `c5d4ec103d6a01eaec15bcb816d019d78b6aaf8700998e563e0849122421f4db` | Shared normative active core |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/definitions.tex` | `2a9c91f485ea7d41ba6d5b13c77f77b8314612da4bc4b59eb1228235374b71b5` | Shared normative active core |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/equations.tex` | `b8027f5e0b787a95708be6cc51018bb993d32f4863c34c4b1b55dd71bd2d3322` | Shared normative active core |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/assumptions.tex` | `6da85c4a44d5b20b222f5796dae8922594f1b1d043a9ac993f5fb6f12059eea9` | Shared normative active core; 11 assumptions |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/requirements.tex` | `ff4b4f924ecd0c21e7a131ca823396b578d80ffde6bd91ab9e7e63bf946e6218` | Shared normative active core; 50 requirements |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/edge_cases.tex` | `a0c6b6be73cd0ef8e3b0655dc842eef83b2ca3037c48b8b2651101570f645556` | Shared normative active predictions; 30 predictions |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/pdf/SCI-CAL-SCIENTIFIC-RATIONALE-v0.1.pdf` | `d4024db374f361854060ef4939796ae8c2fec910a33935852f832384f7d692a3` | Canonical active rationale PDF |
| `doc/scientific_contracts/packages/SCI-CAL/v0.1/pdf/SCI-CAL-ENGINEERING-CONFORMANCE-v0.1.pdf` | `994a641b21c0f4af0701c3eb5c09d86669bb7943b0be02d2080024d00331ac0d` | Canonical active ECS PDF |

### CAL imported content identities

The following are exact identities admitted by CAL's owner decision and core; their scientific correctness was not independently re-audited outside the supplied package:

| Imported object | Exact identity | Disposition |
| --- | --- | --- |
| Independent CAL core reference | SHA-256 `106755520b048f601bc60fd04e7b6020e6fa470480ac3105fa7ba269c730a4fe` | Bound by `AUTHOR_PACKET_MANIFEST.md`; content not promoted beyond CAL's adopted authority |
| TolTECA v1 passband authority | SHA-256 `2756908181cc466550399ec0a869e6671de7912bd3a935f9aeebf63e3e826617` | Selected modeled array-average passband identity; detector/network variation and uncertainty unavailable |
| Atmosphere authority commit | `7156881bd1a47e8cece97b8c541a013c93ac03e1` | Exact external content commit named by owner decision |
| Atmosphere operator contract | SHA-256 `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` | Exact operator-contract content identity |
| Atmosphere operator nodes | SHA-256 `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` | Exact ordered-node content identity |

No exact standalone RTC-to-CAL or CAL-to-PTC boundary artifact is present. Their active normative cores do state compatible quantity/order/lineage clauses; the absence is not silently repaired by inventing a boundary version.

## SCI-PTC admitted sources

The v0.1/r0.4 packet is frozen. The method-reference boundary was admitted only as a content-bound author-packet identity; no external literature claim was imported into this audit. Two internal normative conflicts found by this audit are recorded without amending the frozen source.

| Exact path | SHA-256 | Normative/status role |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/README.md` | `055785ed1092067c32e1f0d14bbbc0dbcb1865dff205a729122497371851d111` | Frozen r0.4 status and navigation |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/AUTHOR_PACKET_MANIFEST.md` | `f8ef424d0b13b73d344446b02ce088e065433f9088a12da6b0aa399addc95769` | Approved author-input binding |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/AUTHOR_SUPERSESSION_COVER.md` | `2a13d3984c2334ccd1886021d2d869bb71363abd3a06bb7f9fbf536614d9ee3e` | Stage-B supersession/firewall record |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `568b35ff3da16c8ed6902d3bb0d845e01eec38e5374c6e89e75823f1f8ecabe6` | Approved author conventions and ownership |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/AUTHOR_METHOD_REFERENCE_BOUNDARY.md` | `d5d33180c9e40958237916ec6dd98ba655d161bc984a3b694197a1a90d78be61` | Sanitized reference boundary identity only |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/AUTHOR_DRAFT_DECISIONS.md` | `ea42dea6c88d22458fef85ec7d46e92bb8d487e9901eeb13e3b1ff4804d7c54c` | Detailed adopted author/owner decisions including PTC-AUTH-D027 |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `adb4c5fa53a1dd2e0b6c863bac95143eadf579b3c7db80bab8a9296761f6ff0e` | Active summary ledger/open-adjacent/deferred states |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.4.md` | `90334ea7853e1ab274f6858fad66078356c06326438625c7fe294e41c07fbcc4` | Binding scientific-owner freeze |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/CROSSWALK.md` | `2211e823f045d78bc2afa491996667e7d931fc59c1f0e51deb896af2bf734125` | Rationale/ECS/core routing |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/scientific-rationale.tex` | `a2c7301448bcbf5402abce61182930a7f65475b5f7e206d4fded2f76b189d272` | Scientist-facing frozen r0.4 source |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/engineering-conformance.tex` | `ef219eed260722a503592c022d6e07e5789fc09d4f132f1bf491eaa9af0c6fac` | Engineering-facing frozen r0.4 source |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/common/notation.tex` | `108022499a5179bc8bbf44060bdc00680ec89c56486c3f833564e63d2e700df7` | Shared normative r0.4 core |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/common/definitions.tex` | `38770c599c8e7b56357577114e799462368f745b703c6690c43d601b5ab4fe6f` | Shared normative r0.4 core; 41 definitions |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/common/equations.tex` | `4d56ab506f88d26a7af061dcc3b7a8a1e852255999dd4a975cc5cb3517ed3d14` | Shared normative r0.4 core; 25 equations; contains F-001/F-002/F-009/F-011 |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/common/assumptions.tex` | `f4f4ec3593419917071714a9586f22d31019487fb37c75f0dad836578d63e80e` | Shared normative r0.4 core; 29 assumptions |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/common/requirements.tex` | `74a077b631bdbcfbdf72306d5dff1693ba93f66d86b9dc5384d63997e3268d62` | Shared normative r0.4 core; 89 requirements |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/src/common/edge_cases.tex` | `d6df04f82a219f1804e41e638095197c1d139d428c0bd693c6a794233d29c493` | Shared normative r0.4 predictions; 50 predictions |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/pdf/README.md` | `6bbd9cd24072fca5a9b96d069cbb4233bdadd83c0634c9df59f44cac579242aa` | Canonical frozen PDF/source identity |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/pdf/SCI-PTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | `7cb358eec6633e06ca2559741d4f32ca2cf62607fac2fe6efb73365863832fd0` | Canonical frozen rationale PDF |
| `doc/scientific_contracts/packages/SCI-PTC/v0.1/pdf/SCI-PTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | `1e73d3e001dafce4dd6a9025553af95da58075fb49ea2b4eb41222431d658b85` | Canonical frozen ECS PDF |

The active ledger keeps Q001/Q002 resolved, six owner policy choices open, four adjacent-owner inputs known but not supplied, and two evidence choices deferred. In particular, the concrete MAP-facing analysis/gridding coefficient remains open (`PTC-OD-010`). No exact package-local PTC profile is registered in VAL.

## SCI-VAL admitted sources

SCI-VAL r0.3 is manager-reviewed and owner-approved as a targeted revision but not scientifically frozen. Core evaluation semantics, Registry bindings, source bindings, and consumer action remain separate authorities.

| Exact path | SHA-256 | Normative/status role |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/README.md` | `2b3a11b4e94e5bbe531ab80272b0fa52b0ccf5cd26fea317e0e4a83682019b20` | r0.3 status and active navigation |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/AUTHOR_PACKET_MANIFEST.md` | `464fb2665ef17803bf7cd801ceb7c7ccec1599da6a5c58873dc116a51fd0fd54` | Approved author-input binding; not a final frozen source manifest |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `32dc62160dff5dcb15e4af83d0df3311024494f30de075784603d4b4bfb4a52c` | Approved Core/Registry/consumer ownership conventions |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md` | `7296112f48fd1edc8eb4b4527883aad86b3dbade19509ab8268e9c6f8b7e4964` | Author-packet cross-package boundary input; subordinate to current exact bindings |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/REVISION_DIRECTIVE_R0.2.md` | `5b8f36288917bb12c342ada192d2dee0b87bb40f8f9868acdcc11eff489d8ef0` | Owner-approved r0.2 revision authority |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/REVISION_DIRECTIVE_R0.3.md` | `c33e07121dcb2979a28463eecbfe61025e4bd4b9c310b733f8f8e5ebe5c9da0e` | Owner-approved r0.3 revision authority |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/SCOPE_OWNERSHIP_DECISION_R0.2.md` | `fd73969c57a5a35f60ca1f0eb1a965a199459692ca2f72fefff8e2c388dc532a` | Binding ownership split/no-rescue decision |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `249685df554c2879f8ebc4737c81f9cf37dfcc3e4e8a00e2a9e99d54c0788d49` | Active Core decisions/deferred representation details |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md` | `d552499fe04309213e05cef11006755ab301e5186399e9515db94b8a81e79d3f` | Current profile registry; one nominal canonical row, reserved names otherwise |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md` | `63743e7a50524ce85255d7938b4e1dee1f94f728ea93f9cbbd7fc0cf0fac030f` | Current but stale adjacent-source binding register |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/MANAGER_REVIEW_R0.3.md` | `8902276659274c076d2f5c43615cfd49785076619a2c41f1c05f036d9dc19e89` | r0.3 manager review/source-PDF identity; not scientific freeze |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/CROSSWALK.md` | `2fcbb48ed152b325065b81bd48c4ee13528e982f5c7316d385e3b16ef6b28ed0` | Rationale/ECS/core routing; 73 rows |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/scientific-rationale.tex` | `dc13fe49e623552ae12dbe8a66c801d0fbb3204f5fb3a33f32254cea84271138` | Active scientist-facing r0.3 source |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/engineering-conformance.tex` | `548eed664f387da6931ac4173a68b24473dbb9a15d1f4a8819ecbd32271de130` | Active engineering-facing r0.3 source |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/common/notation.tex` | `b2101d551100fce6afb79153c3bd0e3419c945f874fa6663a31ee4d693516f44` | Shared normative r0.3 core |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/common/definitions.tex` | `30b86d7fc888b21d21794799f2ef5c77f869ca1a9506cb49b6349af27feabdd1` | Shared normative r0.3 core |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/common/equations.tex` | `fc2b07567bad39314776fda9453b010b482ba67525d5b14dc093df4cc459046e` | Shared normative r0.3 core |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/common/assumptions.tex` | `e616b2c28ea2052b7a0af39a0ca5a320e0b2be95cbe65393d51408e696c16b9b` | Shared normative r0.3 core |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/common/requirements.tex` | `8c518ea1ffba9142d70a5982ce6f403dcb462ecf7de772047bffc7d24bad99d6` | Shared normative r0.3 core; 49 requirements |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/src/common/edge_cases.tex` | `13381205d78f7b69b6e80f3705c9c74351ad2028bffa010dac4f2ae6ea7bb579` | Shared normative predictions; 24 predictions |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/pdf/SCI-VAL-SCIENTIFIC-RATIONALE-v0.1.pdf` | `6cc42e5802fa0bab938613ace377be5ce87f37a96df921996e4c9914600f9bfd` | Canonical active rationale PDF |
| `doc/scientific_contracts/packages/SCI-VAL/v0.1/pdf/SCI-VAL-ENGINEERING-CONFORMANCE-v0.1.pdf` | `07c35ea33b11a8375b28428b3e09973d5402df2c2c9943c2f88d34af0bf141c0` | Canonical active ECS PDF |

## SCI-MAP downstream-reference sources

MAP was not counted as a seventh primary authority and was not allowed to repair upstream gaps. Its formal requirements were used to test whether a complete logical handoff exists.

| Exact path | SHA-256 | Normative/status role |
| --- | --- | --- |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/README.md` | `36ec97653be8e72596bb2341129cf37176cf643f1741786f3e61c8c71c84932a` | r0.3/r0.2 downstream-reference status |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/AUTHOR_PACKET_MANIFEST.md` | `c154cb5b1d2417fff210154ec6a49527ff423a8b4ddb59ed69e173092904e62a` | Approved author-input binding; not final frozen source manifest |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/AUTHOR_SUPERSESSION_COVER.md` | `8ea283525f18199d9760c3f672d145d71f0db87b320ab2e88a5c6635ef3d4aa0` | Supersession/firewall record |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `2d478cb6c5e897308d19614b8b01663318744971850c67459f84c7ddcd57c5c9` | Approved ownership/convention input |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `ba9b32ed5ccee6302942d3c748a32a6e72016ae0d0205df639962662eb1c8728` | Active OD-001--009; CI-001 resolved |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/CROSSWALK.md` | `bab8a2d52c953f17d6bc2d468ab2117f077c0b0982deb34306dc264aa83ef766` | Canonical routing/navigation |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCIENTIST_CROSSWALK_R0.3.md` | `a8faa63ba588652389356085d946d3b45aaa4cde686c318a0b8d6444e2c21883` | r0.3 grouped routing |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCIENTIFIC_FORMAL_CONSISTENCY_R0.3.md` | `360bc2ee4a3ef5df8834d0297abfa232a252dc27a7792efb38c502b4e835fe56` | r0.3 internal consistency/status review |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/scientific-rationale.tex` | `aef5eecfa2e9a8adbf27b7cdf1848f3f3602f3efd704d253d04d4b76d64e6b65` | r0.3 house rationale source |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/engineering-conformance.tex` | `827125961d7f6f1a8d17352ad1e59113712eaa909b665b7388b59d9d51529b7f` | r0.2 engineering-facing source |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/formal-scientific-engineering-contract.tex` | `c0b83e4fea49b46f97c101a91a6eca0d745f18b9561e6c7df8751c0f95b85637` | r0.3 formal shared-authority view source |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/notation.tex` | `722d4b80fb118b419508578877bc21aa46c26d47e80a0d55c09b955edc1dca99` | Shared formal r0.3 core |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/definitions.tex` | `42c8a3bfeb09d67068e6f145d3517fcd8bf87ebdaac883acf9b1b6b9657e50c8` | Shared formal r0.3 core |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/equations.tex` | `03c8177a14251f7154e7084d4392ea2446b369cc1d02c3d4a02c7dcc058c4890` | Shared formal r0.3 core |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/assumptions.tex` | `2b2e37b16e1a483a211396051845167247f27644542021633b79bf92dda2bf38` | Shared formal r0.3 core |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/requirements.tex` | `da49405f2702b9a658c63bb9a3ce33f801947ab532b05bff6edf76c9b792393b` | Shared formal core; 52 requirements |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/edge_cases.tex` | `fd967cbb941d1a22135ff15010d3e48cc530fe845da7ab2a62e6e7ef18b5d3dd` | Shared formal predictions; 25 predictions |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/pdf/SCI-MAP-v0.1_SCIENCE-TEAM-RATIONALE_r0.3-DRAFT.pdf` | `f04339872de3d1fe0863be6e39c27411e916af898de86761e58ac7971d8e552f` | r0.3 house rationale PDF; scientific authority not frozen |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/pdf/SCI-MAP-v0.1_FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT_r0.3-DRAFT.pdf` | `24a0194433a1b57d1501f5a424753244f9a042afe3df361ae7a8cda741a9e2f1` | r0.3 formal contract PDF; scientific authority not frozen |
| `doc/scientific_contracts/packages/SCI-MAP/v0.1/pdf/SCI-MAP-v0.1_ENGINEERING-CONFORMANCE_r0.2-DRAFT.pdf` | `33aa0115cc8b1df5b8e28692a4d618a2bca7e7ccb5553db8dac46cbb9bbcb70a` | r0.2 ECS PDF; scientific authority not frozen |

## Boundary-artifact inventory

| Boundary or dependency | Exact artifact identity at pinned commit | Status | Audit disposition |
| --- | --- | --- | --- |
| ALIGN to AST | Two copies at `packages/SCI-ALIGN/v0.1/SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` and `packages/SCI-AST/v0.1/SCI-ALIGN_TO_SCI-AST_BOUNDARY.md`; each SHA-256 `04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36` | Approved `v0.1/r0.1`, byte-identical | Exact boundary admitted |
| Detector geometry / field rotation to AST | No exact file named `DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md` and no explicitly equivalent approved artifact discovered in the admitted packet | Absent | F-007; external authority and role-specific AST coordinate blocked |
| RTC output grid to AST | No exact file named `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md` and no explicitly equivalent digest-bound body discovered | Absent | F-006; cores compose conceptually, exact versioned transfer does not |
| RTC to CAL | No standalone exact boundary artifact | Absent as artifact; clauses exist | RTC `REQ-103/137` and CAL `REQ-001/003/015--016` are compatible, but no boundary version may be invented |
| CAL to PTC | No standalone exact boundary artifact | Absent as artifact; clauses exist | CAL `REQ-002/039` and PTC `REQ-001/010/061--062` are compatible, but no boundary version may be invented |
| VAL adjacent sources | `packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md`, SHA-256 `63743e7a50524ce85255d7938b4e1dee1f94f728ea93f9cbbd7fc0cf0fac030f` | Continuing authority but stale for consolidated state | F-003; affected decisions unavailable |
| VAL profiles | `packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md`, SHA-256 `d552499fe04309213e05cef11006755ab301e5186399e9515db94b8a81e79d3f` | One nominal record, otherwise reserved/unbound; no aggregate | F-004, F-005, F-015 |
| CAL atmosphere/passband | commit and content digests listed above | Exact imported identities; external observational truth/uncertainty not established | Admitted only to CAL's declared operator/convention scope |
| AST to MAP projection | AST `REQ-080--083` and MAP `OD-008`; no exact resolved MAP plan or materialized `G_pi` parent | Owner-open | F-014; numerical MAP route blocked |

The file-absence checks were performed against the pinned Git tree, not only the working directory. A missing standalone boundary is not automatically an equation conflict: where both cores state compatible obligations, the audit labels the source-closure/identity gap and preserves the compatible clauses.

## VAL source-binding freshness

The continuing binding register says that it must not silently substitute current adjacent meaning. Applying that rule gives:

| Producer/use owner | Binding recorded by VAL | Exact audited package state | Freshness and consequence |
| --- | --- | --- | --- |
| SCI-RTC | v0.1/r0.9 frozen | v0.1/r0.12 frozen; freeze SHA-256 `0cac4396df225c1f2808ee1055e063c9a4e72a02549557c5e997f54d72dac0bf`; requirements SHA-256 `e3c8838ae661046394dd6dfd6a2f0ef1409cdd798df1419ae4ad29757b3f6627` | **Stale.** New representative/origin/influence compatibility binding required; independent-exposure evaluation unavailable meanwhile. |
| SCI-CAL | v0.1/r0.3 architecture-frozen rationale, not frozen | active rationale r0.5 / ECS r0.4, not frozen; requirements SHA-256 `ff4b4f924ecd0c21e7a131ca823396b578d80ffde6bd91ab9e7e63bf946e6218` | **Stale.** Dependent calibration/availability facts cannot be imported as the r0.3 meaning. |
| SCI-PTC | v0.1/r0.4 frozen | v0.1/r0.4 frozen; freeze SHA-256 `90334ea7853e1ab274f6858fad66078356c06326438625c7fe294e41c07fbcc4` | **Version match**, but all operational PTC profiles remain unbound and F-001/F-002 affect the source itself. |
| SCI-MAP | v0.1/r0.3 house rationale, scientific authority not frozen | same named r0.3 reference, scientific authority not frozen; requirements SHA-256 `da49405f2702b9a658c63bb9a3ce33f801947ab532b05bff6edf76c9b792393b` | **Named revision match, authority conditional.** No MAP profile exists and no freeze-strength claim is permitted. |
| ALIGN/AST/TEL | no standalone frozen package/version supplied | ALIGN and AST each frozen v0.1/r0.3; source-manifest SHA-256 values `26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac` and `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`; TEL remains external | **Stale/incomplete.** ALIGN and AST require exact new rows; TEL remains an external dependency rather than a combined alias. |

## Profile source/status preflight

| Profile identity | Exact source/status at pinned commit | Source-binding status | Usability |
| --- | --- | --- | --- |
| `SCI-VAL:independent_exposure@1` | Canonical row in `PROFILE_REGISTRY.md`; owner Grant Wilson; `VAL-R03-D001`, `VAL-R02-D003`, `SCI-VAL-REQ-019/043/045` | RTC binding stale; row omits explicit aggregation/propagation compatibility or `not_applicable` required by Registry Rule item 9 | Not demonstrated usable at this source state |
| `SCI-PTC:basis_fit_admission` | Reserved name only | PTC r0.4 source matches but no immutable policy source/profile | Unbound/unavailable |
| `SCI-PTC:loading_fit_admission` | Reserved name only | Same | Unbound/unavailable |
| `SCI-PTC:operator_application` | Reserved name only | Same; also affected by F-001/F-002 | Unbound/unavailable |
| `SCI-PTC:output_retention` | Reserved name only | Same | Unbound/unavailable |
| `SCI-PTC:coefficient_qc_population` | Reserved name only | Same; concrete MAP coefficient owner decision open | Unbound/unavailable |
| `SCI-PTC:response_companion` | Reserved name only | Same; complete chain response may be unavailable | Unbound/unavailable |
| `SCI-PTC:empirical_or_simulation_population` | Reserved name only | Same | Unbound/unavailable |
| `SCI-MAP:map_upstream_admission` | Reserved name only | MAP r0.3 is nonfrozen and no predicate/source supplied | Unbound/unavailable |
| `<PACKAGE>:diagnostic_display` | Namespace template only | No concrete owner-bound source | Unbound/unavailable |
| Any aggregate or observation-coadd profile | No registered record | No atomic-source compatibility or aggregate policy | Missing/unavailable |

## Active amendments, supersessions, and open ledgers

- **ALIGN:** the exact r0.3 packet is frozen without altering preserved draft-labeled bytes. ODQ-101--105 and 110 remain open, 109 deferred. Dependent timing-field/synthesis/response claims remain unavailable at their exact scopes.
- **AST:** the exact r0.3 packet is similarly frozen. Q001--004, Q006, and Q007 remain open; Q005 deferred; Q008 closes the ordinary RTC-grid role. Missing geometry and RTC-grid boundary bodies are not supplied by that closure.
- **RTC:** r0.11 canonical-pair decisions and the bounded r0.12 correction are incorporated into the r0.12 freeze. Older revisions are superseded history. The active ledger deliberately retains its open/conditional/deferred states.
- **CAL:** `SCIENTIFIC_OWNER_DECISIONS_R0.5.md` supersedes the earlier author-convention restriction that would have rejected the now-approved supported engineering-only opacity range. This is a later-authority resolution, not a current core conflict. Final scientific freeze and unavailable uncertainty/evidence products remain open.
- **PTC:** r0.4 support ownership/composition and nonrestoring centering are frozen. The audit does not use older wording to repair F-001 or F-002. Q001/Q002 are resolved; OD-010 and other named policy/adjacent/evidence items remain open/known/deferred.
- **VAL:** r0.2/r0.3 directives establish four axes, knowledge semantics, Core/Registry/consumer split, aggregation, and source binding. The continuing registers must be updated separately and nonretroactively. Scientific-owner freeze remains pending.
- **MAP reference:** CI-001 resolves dimensional identity of `coverage_cut`; OD-001--009 remain open. In particular OD-003, OD-004, OD-008, and OD-009 affect the downstream handoff/route.

## Explicit exclusions and their consequences

| Excluded material | Reason | Consequence for this report |
| --- | --- | --- |
| Prior RTC–CAL–PTC horizontal audit and any ALIGN–AST audit | Clean-room independence | No finding, vocabulary, repair, or disposition was imported or regression-compared. |
| Citlali implementation, tests, configs, schemas, products, and production behavior | Audit is implementation-blind | No implementation conformity, runtime behavior, or readiness claim. |
| Validation result content | Contract audit only | Contract predictions/requirements are treated as authority; no “test passed” claim. CAL's exact operator content identities were verified as bindings without using validation outcomes. |
| External literature/web | Explicitly prohibited | No external scientific correctness claim; unsupplied packages remain external authorities. |
| SCI-BEAM, NOI, FLT, SRC/MODE, Pointing, OOF, FRUIT contracts | Not included as allowed downstream packets | Only generic dependency/consumer envelopes; no acceptance claim for those packages. |
| Historical package dossiers/prior-work inventories | Mixed implementation/history and lower authority | Used neither to supply scientific predicates nor to cure active gaps. |
| Superseded PDFs/revisions | Lower active authority | Retained as history only; current explicit supersession/freeze controls. |

## Source-preflight disposition

The preflight is complete enough to conduct the audit because each admitted claim is pinned to an exact source and the missing/stale items are explicit. It does **not** satisfy the stronger acceptance condition that every source and VAL binding be current: VAL bindings are stale; most ordinary-chain profiles are unbound; aggregate/coadd policy is absent; RTC-to-AST and detector-geometry boundary bodies are absent; CAL, VAL, and MAP are not frozen; and PTC's frozen core contains material internal conflicts. Consequently, the only honest output is a non-authoritative draft handoff plus a finite blocker ledger, not a frozen or numerical MAP route.
