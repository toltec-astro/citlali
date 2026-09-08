# Isolated clean-build report — r0.4

Date: 2026-09-08. Build identity: SCI-FRUIT-DOC-BUILD-R0.4-2026-09-08. Document compilation only.

The clean build copied only the declared source closure plus derived digest bindings to a separate source directory, used a separate output directory and fresh format cache, and read the exact locally cached TeX resource bundle. No network retrieval or inherited document intermediates were used. All three normal/clean PDF pairs are byte-identical with zero checked TeX diagnostics.

| View | Pages | PDF SHA-256 (normal and clean) | Diagnostics |
| --- | ---: | --- | ---: |
| core | 28 | `86ffefab210439d095a0955fc133afa717a8d3b91fad0149d794cb4c256a6687` | 0 |
| rationale | 10 | `ac980a74ad33d6daf5a41dbe5cd138ff5a61ec1ca7b6f35c0ac8e4151075b397` | 0 |
| ecs | 13 | `0805c7ffd9d6165f058bc23775e4713a6c8da1a03261195752176d0190a821c1` | 0 |

The isolated closure contains 15 source/binding files. SOURCE_DATE_EPOCH=1788825600. The 619 local TeX resources have inventory SHA-256 `57e263922b731404210fdae3fa64b6926d749aa6d6c81b31116a6f042cc85514`.

[Exact build record](BUILD_RECORD.json) records source bindings, tools/versions/hashes, resource digest and diagnostics. [Build recipe](../BUILD_RECIPE.md) states local prerequisites; byte reproducibility on unspecified toolchains is not claimed. Build parity does not establish visual inspection or any numerical method/conformance, fidelity, validation, readiness or freeze.
