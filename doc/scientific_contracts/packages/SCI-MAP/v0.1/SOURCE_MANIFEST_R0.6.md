# SCI-MAP v0.1/r0.6 Exact Source And Digest Manifest

Status: deterministic scientific source binding; no implementation conformity,
validation, response fidelity, numerical-route availability, freeze, readiness,
or production claim

Prepared: `2026-08-27`; base revision `4cc6d27248f692c9778b17a7536e49337bd2d62d`.

| Authority | Exact identity | SHA-256 |
| --- | --- | --- |
| r0.6 owner directive | Targeted scientific-closure directive | `d57e90f8ed4407b0f727cd2ac981318e02101ddd9f73abac7e2772b66dac2c84` |
| SCI-PTC freeze | v0.1/r0.5 | `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66` |
| SCI-PTC decision ledger | includes open `PTC-OD-010` coefficient selection | `0c165b707765ae67aa775e356ec714bd8f25a96553071fc2a71bf91e155cc285` |
| SCI-CAL freeze | science r0.5 / engineering r0.4 | `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22` |
| SCI-CAL requirements | quantity/beam/response authority | `80054fbd526d6a0878f6724c620024955062d41fca1273b85836ead3ee9b5f74` |
| SCI-AST source manifest | v0.1/r0.3 RTC-output-grid authority | `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601` |
| SCI-VAL Core freeze | v0.1/r0.3 | `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6` |
| SCI-VAL continuing Registry | includes MAP `@1` history, `@2`, and coadd `@1` | `03add9fe778cf09b39f66d1e2db56e80a7b646cb90d6ab9b6b0b303023865d62` |
| SCI-VAL Source-Binding Register | source-current MAP binding | `ba070b0cf2e82568cf263d1ed7c8b0a48295d6d8c25d6d5bd685da75f6ee413d` |

The WP-7 audit is not scientific authority. Exact owner-approved artifacts
resulting from it are bound instead: source manifest
`6c55ea528c5a646d34f9ee0b1d7eed3b5d3de7ce53e31c622dd992302c4c0890`,
owner closure `18133f105fd790ab12f04ed14dabd3d40bcc0e3479c39ebe2831422d02640d14`,
owner disposition `cee7f2445dd0bbad9b1925e82e6d9f757ed158237a481fbccee1e83263e72833`,
repair manifest `f5f9f903e52a979339c04cd741686d267c739e5530e364ab49e3bb612b37b26f`,
and authority addendum
`7a0a92a411f4d93f321257fba5cdbc561a4249f5d49cc49b8fe974f87e77d577`.

| Boundary/profile | SHA-256 |
| --- | --- |
| `SCI-PTC_TO_SCI-MAP v0.1/r0.1` (both byte-identical packet copies) | `9a8a51a8c2201ea59b5d62cca9d4fad6206ef130e305006187e5a17c9ec522d6` |
| `SCI-MAP:map_upstream_admission@2` scientist-readable record | `4451e0aa35a60a19b2adb273b552691fe655d4e265d944f6189bcaf8a5d27e20` |
| MAP coadd coefficient/admission records | `45b676b1257c71608df36fb9bd9689e150c5ede60af4cd7b39ee4b28c5e8d056` |
| `SCI-ALIGN_TO_SCI-AST v0.1/r0.1` | `04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36` |
| `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY v0.1/r0.1` | `237cc448e6597a207158858cf5e9dbf603a52ec8ad4f0859eb6788274677bb71` |
| `TIMESTREAM_EXPOSURE_LINEAGE_BOUNDARY v0.1/r0.1` | `4e2e3cda643a687932dd659c5d8008c7e4865a3368f66121a083eb8657a7dceb` |

The shared-authority SHA-256 is computed over raw bytes of the wrapper followed
in order by the six modules below: 
`bb8c9601d6bfe828fff2d7193fcca5a297e14b11fd22c13b2afd05377936b72d`.

| Current MAP source | SHA-256 |
| --- | --- |
| shared wrapper | `753c5adc2ee46656006d6a697f558f971bdad07352759e405ed8b4017de2df6d` |
| `common/notation.tex` | `11b53bc15b01e652b704f584884d5aa7fe56cfc3a22be57bdb2e0679522222bc` |
| `common/definitions.tex` | `5c78296b1bdb99fb2e7375f41cd5e23e213a859d0bce76d6db3c62364200aa78` |
| `common/equations.tex` | `95051c2d202323e88ea7146a5680530322cafdac5d1cf990b3c6fdc94ef36f50` |
| `common/assumptions.tex` | `34b081de1606e248ee6420593741119ca127d4dc3a0715c66a9bf5f3a4e58554` |
| `common/requirements.tex` | `be2940bee8fb1ab77fb11043944d834014f067c1c3ab18e5b7733ef5bc16264b` |
| `common/edge_cases.tex` | `6bf5cc7dc90ba34941617221989f2bfe9c0789cdfe6254337af02fa934ffa751` |
| science-team rationale source | `52d28445cdd999c7ec93d8149c4d966320d48310648693ba2c2357db7b3d144f` |
| formal contract source | `6f59d51097ac80b249c9072159c97393feebb6559bd62c3aebd7acf18d94d052` |
| ECS source | `2f044bbd304c6bc2cf63f03d90be975cdfb2c70e2989f14fedd1baea0e0570b4` |
| owner-decision ledger | `eecd868da0292b38b2c517febf7d9f0e9ebf431d1c5f8daf09c72cde37faaa82` |
| crosswalk | `393074bfd66d7101704b4cade617f01f9426e6db2e891df82aa89d7dbe1f5848` |

This manifest does not hash itself. PDF hashes and page counts are recorded in
`PDF_VISUAL_QA_R0.6.md` after final rendering.
