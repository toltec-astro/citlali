# Document build recipe - SCI-FRUIT v0.1/r0.4

Build-record identity: SCI-FRUIT-DOC-BUILD-R0.4-2026-09-08. Scope: local document compilation, rendering and identity checks only.

From the extracted `r0.4` directory, with the declared local runtime and resource prerequisites:

```sh
/Users/gwilson/tolteca/bin/python tools/build_documents.py --clean --render
/Users/gwilson/tolteca/bin/python tools/verify_documents.py
```

The local build expects an existing TeX resource directory. Its original source on this host is `/Users/gwilson/Library/Caches/Tectonic/bundles/data/6ffe055852f8faf66c0acbe1a7fb27f87b869a90bad1204f3bf4d9683f597c7c`. Another location may be passed using `--bundle /absolute/path/to/local/resources`. It must reproduce the delivered resource inventory for this exact build identity. No network retrieval is performed. Runtime executables and the TeX installation/cache are prerequisites, not embedded binaries in the source archive.

The build script creates task-local caches and copies already available TeX resources, then invokes Tectonic with `--bundle`, `--only-cached`, `--keep-logs`, `--keep-intermediates` and an explicit output directory. `SOURCE_DATE_EPOCH=1788825600`, `FORCE_SOURCE_DATE=1`, `MPLBACKEND=Agg`, and task-specific `MPLCONFIGDIR`, `XDG_CACHE_HOME`, `TECTONIC_CACHE_DIR` are set. The clean build copies only the declared source closure plus derived digest bindings to a separate directory with a fresh format cache and no inherited document intermediates.

The six common modules and core body define the complete review-candidate science inventory. Three wrapper/body/shared-layout source inventories define the view sources; generated digest bindings avoid self-reference. See [source identities](SOURCE_IDENTITIES.md). The build recipe/tool is itself a final-manifest payload, distinct from the core's scientific digest.

## Exact tools used

| Tool | Version | Executable path | SHA-256 |
| --- | --- | --- | --- |
| tectonic | Tectonic 0.16.9 | `/opt/homebrew/bin/tectonic` | `38eff9059ed622672c9a2590415a8f01c043df4232baa459628a2cd86e512d95` |
| pdftoppm | pdftoppm version 26.05.0 | `/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdftoppm` | `de772e88ab9977ccde25def9b403bf42675d75f5dd82b19fbd7d8123ad183159` |
| pdfinfo | pdfinfo version 26.05.0 | `/Users/gwilson/.cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/pdfinfo` | `fee70ade670fb025343aca2b5c3a2aacacb8ed9edce1b716b233b5de924b6bf5` |
| Python | 3.13.2 (main, Feb  4 2025, 14:51:09) [Clang 16.0.0 (clang-1600.0.26.6)] | `/Users/gwilson/tolteca/bin/python` | `42f792544842512d02eb8a95a6009062591476558f09f07553c138661a0bca8e` |

Document package versions: pypdf 6.7.2, Pillow 11.1.0.

The 619 cached TeX resource files are individually bound by [TEX_RESOURCES.sha256](identities/TEX_RESOURCES.sha256), whose SHA-256 is `57e263922b731404210fdae3fa64b6926d749aa6d6c81b31116a6f042cc85514`. The local directory-bundle content digest is `a52652bd8f951ad6ceaac8269fbce6e92cbbefa092819ce8df2208b2106a15ba`, computed by sorted filename bytes, NUL and file bytes, excluding its generated SHA256SUM file. The cache is typesetting input only and supplies no scientific authority.

[BUILD_RECORD.json](reports/BUILD_RECORD.json) records exact tool versions/paths/hashes, source bindings, clean-build parity and diagnostics. [Clean-build report](reports/CLEAN_BUILD.md) states the verified scope. [PDF QA](reports/PDF_VISUAL_METADATA.md) and [page record](reports/PAGE_INSPECTION.json) bind inspected final PDFs. Changed sources/PDFs require renewed identity checks and changed-page inspection; rebuilding does not by itself establish visual inspection.

The [packaging tool](tools/package_documents.py) writes the final manifest and archive after document checks pass. It includes every regular delivery payload plus manifest/sidecar, excluding task caches/intermediates/renders and outer archive/digest. Historical archives remain opaque input payloads. Verification rejects symlinks, hard links, absolute/traversal members and extra paths, and compares every archived member byte with its declared file.

The outer archive and digest sidecar are external siblings, excluded from the extracted tree. Verify an extracted delivery against the separately held archive with:

```sh
/Users/gwilson/tolteca/bin/python tools/verify_documents.py
/Users/gwilson/tolteca/bin/python tools/package_documents.py --verify-only --archive /absolute/path/to/SCI-FRUIT-v0.1-stage-b-r0.4-owner-review.tar.gz
```

The supplied archive path is external to the extracted tree; its SHA-256 sidecar must be beside that archive. No outer artifact must be inserted into the extracted tree to pass any link check. Rebuilding changed bytes requires renewed source, mechanical and visual checks and new manifest/archive digests. These document operations do not approve or freeze science.
