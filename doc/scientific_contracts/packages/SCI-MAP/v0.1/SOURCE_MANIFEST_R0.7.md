# SCI-MAP v0.1/r0.7 Exact Source And Digest Manifest

Manifest identity: `SCI-MAP_SOURCE_MANIFEST v0.1/r0.7`

Status: deterministic scientific source binding; no implementation conformity,
validation, response fidelity, numerical-route availability, scientific
freeze, readiness, or production claim

Prepared: `2026-08-28`; source base commit
`75ab92fcaa3ef1b2de47070ba5088359de68e37a`.

The manifest does not hash itself. Its immutable identity is the exact path,
contract version, semantic revision, and bytes committed with this packet.

## Governing and adjacent authorities

| Authority | Exact path and version/revision | SHA-256 | Compatibility and supersession |
| --- | --- | --- | --- |
| r0.7 owner directive | Codex attachment `7ecbf4ca-489e-4e68-adad-e477d762b629/pasted-text.txt`; SCI-MAP v0.1/r0.7 | `f7747eea28710d524e12c818b872ac3fcc49f413271f83c0644ae129949a8c8c` | Governs r0.7; supersedes the r0.6 targeted directive for this packet |
| SCI-PTC freeze | `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md`; v0.1/r0.5 | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` | Frozen and compatible only under exact boundary/profile rules; `PTC-OD-010` remains open |
| SCI-PTC decision ledger | `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md`; v0.1/r0.5 | `d899d93bfe433762d4b3b06a9fa7aeff08f7dcbfd6f170d614e69cff58d35c6f` | Exact coefficient-owner authority; not superseded by MAP |
| SCI-CAL freeze | `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md`; science r0.5 / engineering r0.4 | `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22` | Compatible quantity/beam/response authority through frozen PTC route only |
| SCI-CAL requirements | `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/requirements.tex`; frozen authority | `80054fbd526d6a0878f6724c620024955062d41fca1273b85836ead3ee9b5f74` | Exact quantity, beam, availability, response, and uncertainty meanings |
| SCI-AST manifest | `doc/scientific_contracts/packages/SCI-AST/v0.1/SOURCE_MANIFEST.md`; v0.1/r0.3 | `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601` | Frozen source for both coordinate roles below |
| SCI-AST parent equations | `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/notation.tex`; Equations `align-parent`, `tangent-parent`, `pixel-parent`, and `rtc-parent` | `ba0bd3a366416860d612d1d94b723c4a72553290990c0663294901c2dc1586d7` | ALIGN-grid layered parent supplies original-footprint role; RTC parent supplies signal role; neither substitutes for the other |
| SCI-AST requirements | `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/requirements.tex`; REQ-073--081 | `47b357dd79136fb3d019f45b1092a2efd88fbd1ed16ab038fc6bf51beaf06f01` | Exact stable-slot, RTC-grid, projection-ownership, and pre-MAP geometry authority |
| SCI-VAL Core freeze | `doc/scientific_contracts/packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.3.md`; v0.1/r0.3 | `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6` | Core evaluator mechanics unchanged; does not author MAP policy |
| SCI-VAL Registry | `doc/scientific_contracts/packages/SCI-VAL/v0.1/PROFILE_REGISTRY.md`; continuing r0.3 Registry with r0.7 MAP records | `aceace5cbc34f76d9bbc914cfe693837c5283f5ec2839f118fc65971cf0952f7` | `map_upstream_admission@1` historical; exact `@2` and coadd `@1` source-bound; no alias substitution |
| SCI-VAL Source Binding | `doc/scientific_contracts/packages/SCI-VAL/v0.1/SOURCE_BINDING_REGISTER.md`; continuing r0.3 register, 2026-08-28 | `59e4510a9df54a964b0b1ab2f4898e3231ad790a981a903ca14fd1c52f546a22` | Binds exact r0.7 source generation; all required rows are compatible |

## WP-7 admitted scientific-owner authority

The WP-7 audit is not scientific authority. Only these exact owner-approved
or authority-publication artifacts are admitted:

| Exact path | Role | SHA-256 | Supersession state |
| --- | --- | --- | --- |
| `doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_170ECEA9D/SOURCE_MANIFEST.md` | Exact published source packet | `6c55ea528c5a646d34f9ee0b1d7eed3b5d3de7ce53e31c622dd992302c4c0890` | Admitted packet manifest |
| `doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_170ECEA9D/WP7_SCIENTIFIC_OWNER_CLOSURE_2026-08-26.md` | Owner closure | `18133f105fd790ab12f04ed14dabd3d40bcc0e3479c39ebe2831422d02640d14` | Owner action retained |
| `doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F/REPAIR_AND_CLOSURE/WP7_SCIENTIFIC_OWNER_DISPOSITION_2026-08-25.md` | Exact D001--D004 disposition | `cee7f2445dd0bbad9b1925e82e6d9f757ed158237a481fbccee1e83263e72833` | Owner decisions retained |
| `doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F/REPAIR_AND_CLOSURE/WP7_REPAIR_AUTHORITY_MANIFEST_2026-08-25.md` | Repair-authority manifest | `f5f9f903e52a979339c04cd741686d267c739e5530e364ab49e3bb612b37b26f` | Exact repair generation |
| `doc/scientific_contracts/audits/WP7_TIMESTREAM_CLEAN_ROOM_F01E22F5F/REPAIR_AND_CLOSURE/WP7_APPROVED_SCIENTIFIC_AUTHORITY_ADDENDUM_2026-08-25.md` | Sanitized approved scientific authority | `7a0a92a411f4d93f321257fba5cdbc561a4249f5d49cc49b8fe974f87e77d577` | Readable owner-approved authority |

## Boundaries and profiles

| Identity and exact path | SHA-256 | Compatibility and supersession |
| --- | --- | --- |
| `SCI-PTC_TO_SCI-MAP v0.1/r0.1`, `packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY.md` and byte-identical PTC copy | `db0eae0aeeb63a61ce1fdbc71914a8cb424e94cc6ae34e64f1b0ccbfe714e52d` | Both copies must remain byte-identical; any mismatch makes the route unavailable |
| `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1`, MAP boundary path | `77c5f6c0f0056fa7e4b2c3a62d82114f0e87a6ad7afb833b344681fa88e19390` | Binds existing AST ALIGN-grid equations/requirements to exact target MAP WCS; not an RTC-coordinate alias |
| `SCI-MAP:map_upstream_admission@2`, `SCI-MAP_UPSTREAM_ADMISSION_PROFILE.md` | `29a4ca004b3d2672ece104b148a2f88a0e71ebcf3e01e52cd1b9132bb879935c` | Source-bound r0.7 occurrence record; `@1` is historical and incompatible as a substitute |
| `SCI-MAP:observation_coadd_admission@1` and `SCI-MAP:uniform_observation_coadd_coefficient@1`, `SCI-MAP_COADD_PROFILES_R0.7.md` | `4546ba5e021dcc2e0255fc7a1d8a68b1f6fdce1fb7dd43b9fe2546bde4e9357b` | Exact aggregate and coefficient roles; a changed source, role, or restriction requires a successor profile |
| `SCI-MAP:one_hot_containing_pixel@1`, shared r0.7 authority below | shared digest below | Sole base-v0.1 projection; fractional projection is not a compatible substitute |

## Shared MAP authority and views

The shared-authority SHA-256 is computed over raw bytes of the wrapper followed
in the listed module order: `275cd4fa296b690011dd54fa326724573a8d854e7047734bdd8bc075e3f170d5`.

| Exact r0.7 MAP source | SHA-256 |
| --- | --- |
| `src/SCI-MAP-v0.1_SHARED_AUTHORITY_r0.7.tex` | `08181fce5348103ac22ab13602c6726f3bb487e3537a4722966a11c149e87d3f` |
| `src/common/notation.tex` | `4dd3d9016eb925b48b4c18172e5b3ea5a31c6c0b2fc898bc8d631dd513be8c3c` |
| `src/common/definitions.tex` | `8b1cc6e2d017d18b00c86a18724fa4c7991e5ad6cd953f47a27cfb5d772a253b` |
| `src/common/equations.tex` | `6ed9803123c413287cc045596d84418427af0886ef0c4a497fb09d53480fc275` |
| `src/common/assumptions.tex` | `bba33b92c4189fe5886ef849caebeeb5400bdc7f2572f58a74de53ca578881de` |
| `src/common/requirements.tex` | `ee22b6c77be292d68baa47e9c9014279f90b9143dcce60729ecb6aa9d5411d71` |
| `src/common/edge_cases.tex` | `b9c1cb6eb1767cb77e55a8580bdfaa427f0b82c160486a060095463fc37e4ce4` |
| `src/scientific-rationale.tex` | `c9a819df09ca1f91a02ba72edd2da2204773e2f4947af1d8276a40dc6e4d3733` |
| `src/formal-scientific-engineering-contract.tex` | `726035536e264a11b3d735fac17af5ecf906ac78b8cfed77ae63498596b50245` |
| `src/engineering-conformance.tex` | `90b49ca40595fbf0f1480a7be3a895988242a155e73b7448936632f0ff509ea9` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `bfcc18eced116309356d9d597a4423881ef67df058df07be30ec80532452dd9b` |

The three views import the same wrapper and ordered modules. The generated
owner register is derived from the listed decision ledger. Prior r0.6 reports
and rendered aliases remain history and are superseded by the r0.7 identities
for this packet; similar names do not establish compatibility. The prior
unversioned shared-wrapper basename is intentionally removed because it would
import the revised common modules and therefore could not preserve historical
bytes.

PDF hashes, metadata, page counts, and all-page render inspection are recorded
in `PDF_VISUAL_QA_R0.7.md`.
