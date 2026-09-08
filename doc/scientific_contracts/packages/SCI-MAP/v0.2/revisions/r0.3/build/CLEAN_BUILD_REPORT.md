# Clean document build report — SCI-MAP v0.2/r0.3

Build `SCI-MAP-R03-DOCUMENT-BUILD/candidate-build-01`: **PASS for document compilation and artifact checks**. All three compiler processes completed successfully. The exact sources, generated covers, commands, environment and tools are recorded in [BUILD_ATTEMPT.json](BUILD_ATTEMPT.json); reproduction is described by [the build recipe](BUILD_RECIPE.md). This report grants no scientific approval, freeze, Registry activation or implementation claim.

| View | Pages | PDF SHA256 |
| --- | ---: | --- |
| Formal | 37 | `a091e3d5ad77acc81d22ec6fa36f0cd75c712c9c7f9049b07ab53db9d6520fbd` |
| Rationale | 17 | `7bcd41008daa935d3d4ea43f19bae02a84c9e39caa0c63f42d54d0ef3688320d` |
| Engineering | 30 | `34413f4dcc43f1a755be53a1144d3b6289b2fe320014ac0ce3e71568de5a31e2` |

Each view compiled in a new isolated output directory using cached/untrusted Tectonic with retained logs and intermediates, exact dependency rules, SOURCE_DATE_EPOCH=1788825600 and TZ=UTC. Strict pypdf and Poppler reopening succeeded. Every page rendered independently. Final logs contain no overfull box, missing glyph, unresolved reference or TeX error; underfull spacing notices, if present, remain in raw logs and are assessed visually. Compiler-reported local dependencies resolve to the exact isolated document sources. Cached TeX assets are not asserted to form a portable bundle; no cross-host bit reproducibility is claimed.

- tectonic: `/opt/homebrew/Cellar/tectonic/0.16.9/bin/tectonic`, SHA256 `38eff9059ed622672c9a2590415a8f01c043df4232baa459628a2cd86e512d95`; version evidence in BUILD_ATTEMPT.json.
- pdftoppm: `/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm`, SHA256 `de772e88ab9977ccde25def9b403bf42675d75f5dd82b19fbd7d8123ad183159`; version evidence in BUILD_ATTEMPT.json.
- pdfinfo: `/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo`, SHA256 `fee70ade670fb025343aca2b5c3a2aacacb8ed9edce1b716b233b5de924b6bf5`; version evidence in BUILD_ATTEMPT.json.

The Python executable/version/hash and PDF reader identities are in the exact build/inspection records. Earlier document attempts are preserved in [the attempt history](DOCUMENT_BUILD_ATTEMPT_HISTORY.json); a failed or superseded attempt is not counted as passing. The local document build is not application, Spack, deployment or Unity qualification. No prospective ECS evidence was executed.
