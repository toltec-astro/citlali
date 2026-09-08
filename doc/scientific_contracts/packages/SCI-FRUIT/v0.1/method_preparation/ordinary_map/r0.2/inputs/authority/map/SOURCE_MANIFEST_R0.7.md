# SCI-MAP v0.1/r0.7.1 Exact Source And Digest Manifest

Manifest identity: `SCI-MAP_SOURCE_MANIFEST v0.1/r0.7.1`

Status: deterministic scientific source binding; no implementation conformity,
validation, response fidelity, numerical-route availability, scientific
freeze, readiness, production, or performance claim

Prepared: `2026-08-28`; source base commit
`0387d34eec1e4a7c7a431d2ed3f3ec8a293e15d5`.

The exact bytes of this manifest are externally bound by
`SOURCE_MANIFEST_R0.7.sha256`. Keeping the digest in a companion file avoids
self-reference. The durable verifier recomputes the manifest digest and every
binding below.

## Governing and adjacent authorities

| Authority | Exact path and version/revision | SHA-256 | Compatibility and supersession |
| --- | --- | --- | --- |
| r0.7 owner directive | Codex attachment `7ecbf4ca-489e-4e68-adad-e477d762b629/pasted-text.txt`; SCI-MAP v0.1/r0.7 | `f7747eea28710d524e12c818b872ac3fcc49f413271f83c0644ae129949a8c8c` | Governs r0.7; supersedes the r0.6 targeted directive for this packet |
| r0.7.1 freeze-only errata | `FREEZE_ONLY_ERRATA_R0.7.1.md`; SCI-MAP v0.1/r0.7.1 | `c90008bda0c0168de2c044dd5c7c58722f2e769ea92a299729f1552260ae8d72` | Bounded wording, identity, and source-binding correction only |
| r0.7.1 change log | `CHANGE_LOG_R0.7.1.md` | `c49186113fd8f39ca73ef871a128fd3ac931c6d8d4f1c13f4c4579166e640e81` | Stable IDs and scientific semantics unchanged |
| SCI-PTC freeze | `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md`; v0.1/r0.5 | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` | Frozen and compatible only under exact boundary/profile rules; `PTC-OD-010` remains open |
| SCI-PTC decision ledger | `doc/scientific_contracts/packages/SCI-PTC/v0.1/SCIENTIFIC_OWNER_DECISION_LEDGER.md`; v0.1/r0.5 | `d899d93bfe433762d4b3b06a9fa7aeff08f7dcbfd6f170d614e69cff58d35c6f` | Exact coefficient-owner authority; not superseded by MAP |
| SCI-CAL freeze | `doc/scientific_contracts/packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md`; science r0.5 / engineering r0.4 | `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22` | Compatible quantity/beam/response authority through frozen PTC route only |
| SCI-CAL requirements | `doc/scientific_contracts/packages/SCI-CAL/v0.1/src/common/requirements.tex`; frozen authority | `80054fbd526d6a0878f6724c620024955062d41fca1273b85836ead3ee9b5f74` | Exact quantity, beam, availability, response, and uncertainty meanings |
| SCI-AST manifest | `doc/scientific_contracts/packages/SCI-AST/v0.1/SOURCE_MANIFEST.md`; v0.1/r0.3 | `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601` | Frozen source for both coordinate roles below |
| SCI-AST parent equations | `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/notation.tex`; Equations `align-parent`, `tangent-parent`, `pixel-parent`, and `rtc-parent` | `ba0bd3a366416860d612d1d94b723c4a72553290990c0663294901c2dc1586d7` | ALIGN-grid layered parent supplies original-footprint role; RTC parent supplies signal role; neither substitutes for the other |
| SCI-AST requirements | `doc/scientific_contracts/packages/SCI-AST/v0.1/src/common/requirements.tex`; REQ-073--081 | `47b357dd79136fb3d019f45b1092a2efd88fbd1ed16ab038fc6bf51beaf06f01` | Exact stable-slot, RTC-grid, projection-ownership, and pre-MAP geometry authority |
| SCI-VAL Core freeze | `doc/scientific_contracts/packages/SCI-VAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.3.md`; v0.1/r0.3 | `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6` | Core evaluator mechanics unchanged; does not author MAP policy |
| SCI-VAL Registry | `SCI-VAL_PROFILE_REGISTRY v0.1/r0.3-map-r0.7.1-2026-08-28` | `95b2ddeca2039aa2b3614e3dfede530cb7e024e6db33ca22baed867f19d2709d` | `map_upstream_admission@1` historical; exact `@2` and coadd `@1` source-bound; no alias substitution |
| SCI-VAL Source Binding | `SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-2026-08-28` | `7b91a324f35196a8c8a6e23c8abbbf5322fc601798e36d4ac821907a6090eadf` | Binds exact r0.7.1 source generation; all required rows are compatible |

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
| MAP PTC-to-MAP boundary copy, exact `SCI-PTC_TO_SCI-MAP v0.1/r0.1` | `a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7` | Must remain byte-identical to PTC copy |
| PTC PTC-to-MAP boundary copy, exact `SCI-PTC_TO_SCI-MAP v0.1/r0.1` | `a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7` | Must remain byte-identical to MAP copy |
| Original-footprint-coordinate boundary, exact `SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE v0.1/r0.1` | `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4` | Exact target-MAP-WCS binding; not an RTC-coordinate alias |
| MAP upstream-admission profile, exact `SCI-MAP:map_upstream_admission@2` | `0717476c0a1d177074ee8702c18308f093d45a4913b22933f3fda3d33090a883` | `@1` is historical and incompatible as a substitute |
| MAP coadd profiles, exact `SCI-MAP:observation_coadd_admission@1` and `SCI-MAP:uniform_observation_coadd_coefficient@1` | `d93c04488925931676b02dff433774ff2cda9846fdd1d3f34bff29d76efdd702` | Exact aggregate and coefficient roles |
| `SCI-MAP:one_hot_containing_pixel@1`, shared r0.7.1 authority below | shared digest below | Sole base-v0.1 projection; fractional projection is not a compatible substitute |

## Shared MAP authority and views

The shared-authority SHA-256 is computed over raw bytes of the wrapper followed
in the listed module order. The exact aggregate binding is
`649e1694b2a6353b0e5e8cb42ab73d3f03b1bc5cad671810361e802a87278e9b`.

| Exact r0.7.1 MAP source | SHA-256 |
| --- | --- |
| Shared r0.7.1 authority aggregate | `649e1694b2a6353b0e5e8cb42ab73d3f03b1bc5cad671810361e802a87278e9b` |
| `src/SCI-MAP-v0.1_SHARED_AUTHORITY_r0.7.1.tex` | `08fcc9782cfba806d33dc07652a2363c8bd6540084f54e752e1fa91a5336b6bb` |
| `src/common/notation.tex` | `2b132704dd1ee8da7a56e5bafdc998df98422fe512736ac4f904fad8a693e569` |
| `src/common/definitions.tex` | `740f4a6f1ef0bbb12f721f192b7883144c247d039ab0c9dfa0ffae53cd711b65` |
| `src/common/equations.tex` | `36329f4cd1a103c78fcdcc5ff247a850f40aba92a09156d1aa55a1411a430c04` |
| `src/common/assumptions.tex` | `bba33b92c4189fe5886ef849caebeeb5400bdc7f2572f58a74de53ca578881de` |
| `src/common/requirements.tex` | `68acf81d6c27788495cc680a1819da5f269616493b9aba5a6e0a1d1058ba5fa7` |
| `src/common/edge_cases.tex` | `47022012e79173a1778a4e5bdc6743b4691bfecb27faa3090bcf03458d87e123` |
| `src/scientific-rationale.tex` | `652c94b65e9742f6232c4a97027335536ffa584c2b30222edca1947c80a10cd0` |
| `src/formal-scientific-engineering-contract.tex` | `99fdb64e42c005afb37740170aa69ba41c50e9c53424dbe936a76ee2a7664229` |
| `src/engineering-conformance.tex` | `3e67d0b0984278c3a2f16f6b1d001cbd228639e9c3456e40e2987f5046ce2c42` |
| `SCIENTIFIC_OWNER_DECISION_LEDGER.md` | `bfcc18eced116309356d9d597a4423881ef67df058df07be30ec80532452dd9b` |

## Exact SCI-VAL Registry and source-binding records

Registry revision is exactly `SCI-VAL_PROFILE_REGISTRY
v0.1/r0.3-map-r0.7.1-2026-08-28`. Source-binding revision is exactly
`SCI-VAL_SOURCE_BINDING_REGISTER v0.1/r0.3-map-r0.7.1-2026-08-28`.
Section hashes include the exact Markdown heading and all bytes through, but
not including, the next same-or-higher-level heading. Row hashes include the
complete Markdown table row and terminating newline.

| Bound artifact or exact byte range | SHA-256 |
| --- | --- |
| SCI-VAL Profile Registry | `95b2ddeca2039aa2b3614e3dfede530cb7e024e6db33ca22baed867f19d2709d` |
| Registry record `SCI-MAP:map_upstream_admission@2` | `10e250efeb21eec2c865ee4ce73d23f859f935eb0ab8db6bbcc344381bfd12ab` |
| Registry record `SCI-MAP:observation_coadd_admission@1` | `aa2f43962c79389822b389e9f376a2a712d1c18094671f5a5c730a1f7c7dfdad` |
| SCI-VAL Source-Binding Register | `7b91a324f35196a8c8a6e23c8abbbf5322fc601798e36d4ac821907a6090eadf` |
| SCI-VAL source-binding row `SCI-ALIGN` | `c5d3aec8af0cb0d36e0628f4a00f5e43815e5ebc9d8a4666cdb31e7bcf1c5c24` |
| SCI-VAL source-binding row `SCI-AST` | `2c96e9147c88665a84e3e5b246178c7fd92595aa729c7b0a281fafeafdcbf161` |
| SCI-VAL source-binding row `SCI-RTC` | `020bd916f6d63b2fde58b05ebe55bd85869c76d9c263331d77cc13f04f51adb9` |
| SCI-VAL source-binding row `SCI-CAL` | `819793f480115f8c6c6733626a1c06bc987ad1e2636589a52e45aeb3bbbf623e` |
| SCI-VAL source-binding row `SCI-PTC` | `1856a6dbdefbb4bc22b6c5b211f8c76c61afd01843fb48ba95b891532bfd1183` |
| SCI-VAL source-binding row `Tune/readout and telescope inputs` | `157262375f5ca34f92b13f80459fe352a76d28a834178f28cecfdfc86f1b158a` |
| SCI-VAL source-binding row `SCI-MAP` | `20feff2eced49eb166cb3baeab827d70623e2b460520377d31422026475b3242` |

## Owner and parity reports

| Bound artifact | SHA-256 |
| --- | --- |
| Owner-decision ledger | `bfcc18eced116309356d9d597a4423881ef67df058df07be30ec80532452dd9b` |
| Rationale/formal/ECS parity report | `670904f72769766bff81ab17d37b693f33293e8e136c55ce356a697c5733b325` |
| Owner-decision parity report | `a860da25555a2c28e0e88c6faf212daa9aa398dbc511a32773a72fb4986adb81` |
| Byte-equality/shared-authority report | `52bb7b5c43a0b9440b304431ee4d72589df9a76fa9c6855f143042ea149a2716` |

The three views import the same wrapper and ordered modules. The generated
owner register is derived from the listed decision ledger. Prior r0.5, r0.6,
and r0.7 reports and rendered aliases remain immutable history; similar names
do not establish compatibility. The prior r0.7 shared-wrapper basename is
removed because it would import revised common modules and could not preserve
historical bytes.

PDF hashes, metadata, page counts, and all-page render inspection are recorded
in `PDF_VISUAL_QA_R0.7.1.md`.
