# Isolated clean-build report — r0.3

Date: 2026-09-08. Build identity: SCI-FRUIT-DOC-BUILD-R0.3-2026-09-08. Document compilation only.

The clean build copied only the declared source closure plus derived digest bindings to a separate source directory, used a separate output directory and fresh format cache, and read the exact locally cached TeX resource bundle. No network retrieval or inherited document intermediates were used. All three normal/clean PDF pairs are byte-identical with zero checked TeX diagnostics.

| View | Pages | PDF SHA-256 (normal and clean) | Diagnostics |
| --- | ---: | --- | ---: |
| core | 28 | `a44acdc1f92dde7280f48c9cd419c14ecf407d6c03dcbb7ec2ec08df6f605b3d` | 0 |
| rationale | 10 | `135897f50b5c202cb38a511aa62d205c48e256df892d389f225a23c2a71860ad` | 0 |
| ecs | 13 | `2bd38635bb9e070657186c61edd0db4e7c6db0bfc13b4b0d7ae65d3a7bfa2ed2` | 0 |

The isolated closure contains 15 source/binding files. SOURCE_DATE_EPOCH=1788825600. The 619 local TeX resources have inventory SHA-256 `57e263922b731404210fdae3fa64b6926d749aa6d6c81b31116a6f042cc85514`.

[Exact build record](BUILD_RECORD.json) records source bindings, tools/versions/hashes, resource digest and diagnostics. [Build recipe](../BUILD_RECIPE.md) states local prerequisites; byte reproducibility on unspecified toolchains is not claimed. Build parity does not establish visual inspection or any numerical method/conformance, fidelity, validation, readiness or freeze.
