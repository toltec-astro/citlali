# SCI-MAP v0.1/r0.5 Exact Source And Digest Manifest

Status: deterministic source-binding record; no implementation conformity,
validation, performance, readiness, or production claim

Prepared: `2026-08-27`

Base revision before the targeted edit: Git commit
`0e7f45f6350de7bdb8c46dbccead80128a0413ac`.

## Owner Directive

| Authority | SHA-256 |
| --- | --- |
| `SCI-MAP v0.1 r0.5 TARGETED PTC-TO-MAP CROSS-PACKAGE CLOSURE DIRECTIVE` | `210e8beafe26381a7d35cf38bacab9a9d959646055635a7c1179e0729a3cfa9a` |

## Frozen And Continuing Adjacent Authorities

| Authority | Exact identity | SHA-256 |
| --- | --- | --- |
| SCI-PTC freeze | SCI-PTC v0.1/r0.5; promoted source commit `8f0ecccfacbdce0543141c4289ec06c702065f5e` | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` |
| SCI-PTC decision ledger | Frozen r0.5 author decisions and `PTC-OD-010` coefficient gate | `0c165b707765ae67aa775e356ec714bd8f25a96553071fc2a71bf91e155cc285` |
| SCI-CAL freeze | SCI-CAL v0.1 science-rationale r0.5 / engineering-conformance r0.4; promoted source commit `0b3cfb24070c1eda04dbda7633accf40e2e8b852` | `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22` |
| SCI-CAL requirements | Exact frozen requirements carrying beam/template identity and response distinction | `80054fbd526d6a0878f6724c620024955062d41fca1273b85836ead3ee9b5f74` |
| SCI-AST source manifest | SCI-AST v0.1/r0.3 | `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601` |
| SCI-VAL freeze | SCI-VAL v0.1/r0.3 Core freeze | `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6` |
| SCI-VAL continuing Profile Registry | Includes `SCI-MAP:map_upstream_admission@1` | `2e529c544b462f7e70aaf28e20da656527069e4d56ff4c76c66178d45baf14d4` |
| SCI-VAL continuing Source-Binding Register | MAP source binding current through 2026-08-27 | `cdb3047ffa8cd0cf5e5a41d13ae32f55e1304cdbd73a4e978df3ff77e96a121c` |

The continuing VAL registries may add later immutable source/profile bindings
without rewriting frozen VAL Core or prior evaluation decisions.

## WP-7 Authority And Closure

| Artifact | SHA-256 |
| --- | --- |
| Current WP-7 source manifest | `6c55ea528c5a646d34f9ee0b1d7eed3b5d3de7ce53e31c622dd992302c4c0890` |
| Scientific-owner closure, 2026-08-26 | `18133f105fd790ab12f04ed14dabd3d40bcc0e3479c39ebe2831422d02640d14` |
| Scientific-owner disposition, 2026-08-25 | `cee7f2445dd0bbad9b1925e82e6d9f757ed158237a481fbccee1e83263e72833` |
| Repair-authority manifest | `f5f9f903e52a979339c04cd741686d267c739e5530e364ab49e3bb612b37b26f` |
| Approved scientific-authority addendum | `7a0a92a411f4d93f321257fba5cdbc561a4249f5d49cc49b8fe974f87e77d577` |

The closure is contract-level only and preserves its seven stated limitations;
it does not establish implementation conformity or production behavior.

## Exact Cross-Package Boundaries

| Boundary | SHA-256 |
| --- | --- |
| `SCI-ALIGN_TO_SCI-AST v0.1/r0.1` | `04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36` |
| `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY v0.1/r0.1` | `237cc448e6597a207158858cf5e9dbf603a52ec8ad4f0859eb6788274677bb71` |
| `TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY v0.1/r0.1` | `4e2e3cda643a687932dd659c5d8008c7e4865a3368f66121a083eb8657a7dceb` |
| `SCI-PTC_TO_SCI-MAP v0.1/r0.1` | `709a798b8004cd1c9c38fee9d8bef00bc12b7c49ee6dba8745c68e461c6a380a` |

## SCI-MAP r0.5 Authority

The compatibility wrapper retains its historical filename. Its semantic
revision is r0.5. The semantic shared-authority digest is SHA-256 over the
wrapper followed, in order, by notation, definitions, equations, assumptions,
requirements, and edge predictions:

`a8db9d9dab1faa81dc5dd983533820581f6535916efd8dcfa378388edc1594b7`

| Artifact | SHA-256 |
| --- | --- |
| Shared wrapper | `753c5adc2ee46656006d6a697f558f971bdad07352759e405ed8b4017de2df6d` |
| `common/notation.tex` | `8248afa464bf80b0d3b2da3aebb6b257ea1b991f514029d1d6648405dcac748b` |
| `common/definitions.tex` | `268cad03b1871b3bc120d7bbd6d28953b0c4e8531d9584e62c071b566cc70e67` |
| `common/equations.tex` | `3ee0b638e1c309b70cf14066d95331c465536bb85f0f011e12b1a4924cc2fb12` |
| `common/assumptions.tex` | `c30045efbbb780471c840d932ba7b8dbcfd1c8779e4ebf1b670c81a0c41c1161` |
| `common/requirements.tex` | `e470d8abcb72fc3d477c47087820f4a1d2bd3df7656ce462b314e06109112b3a` |
| `common/edge_cases.tex` | `6ca43ea04062ef98d228f06a106116549aff70bff7bba5401b346c22cf4fce61` |
| Science-team rationale source | `c40430acbf20719a4921dd5a957ad2c0e46e62ede0216c8e28e08237d2db3203` |
| Formal contract source | `d2fc3f80abd19c32e52eabd52a4ada58f4cbeb704919aef7b0397851059c2289` |
| ECS source | `79791617874fe8d5e97f2035609ca6343db89b7d44b8cb610c698c6e3ae71cdf` |
| MAP upstream-admission profile | `332bf391cca4bd5769ddba364fffa59d95a0dd3ca7e686212fb637ff92ce1b39` |
| r0.5 owner dispositions | `bce3053a0a09e6860f9d04767d887cb0195e770a13d2d4d106efc3a17131df72` |
| Coadd coefficient/admission profiles | `038ecdc3e5797933490509ffcb14367dcf8a5996dfaf40c53ecfab197a0cd549` |
| Owner-decision ledger | `201348511b26f8ffc047b472ebf3fe28b572ea466229e7c8c6dc5fa6ca280a2b` |
| Crosswalk | `7acb4e5f672d9c126f2b2037491e220d086e4e713a1256380760b8aa597387fe` |

This manifest does not hash itself. PDF hashes and page counts are recorded in
`PDF_VISUAL_QA_R0.5.md` because final rendering follows source binding.
