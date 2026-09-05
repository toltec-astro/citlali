# Cached PDF build and render receipt

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1**  
Date: **2026-09-05**

## Renderer and network posture

Both PDFs were produced with local `/opt/homebrew/bin/tectonic` using
`--only-cached`. No package installation, network retrieval, or external
scientific source was used. Build logs and visual-QA rasters are confined to
`build/`.

The required create-operation marker was successfully invoked once before
output authoring. A preceding literal invocation containing the manager
message’s sentence-final period was rejected with usage text and created no
marker; the accepted invocation ended at `--output-format pdf`.

## Attempt history

1. The first sandboxed cached call failed before TeX because the local
   renderer’s macOS system-configuration client returned a null object.
2. The first permitted local call found the entrypoint but reported
   `src/common_core.tex` unresolved because Tectonic resolves input paths
   relative to `src/`. All includes were then made output-local and relative.
3. Cached assets lacked the Latin Modern slanted metric
   `lmromanslant10-regular`. A T1 fallback likewise lacked `ecrm1095`.
4. The final entrypoints silently substitute the cached upright Latin Modern
   face for italic/slanted requests. This is typography only; no scientific
   wording, equation, requirement, or decision changed.
5. Both final cached builds completed, reran TeX for references, and wrote
   PDFs. Final logs contain no overfull box, undefined-reference, emergency,
   fatal, or TeX error entry. Narrow-table underfull spacing notices remain and
   do not clip content.

## Final artifacts

| View | Pages | SHA-256 |
| --- | ---: | --- |
| `scientific_rationale.pdf` | 12 | `e46cca90b1bb10c84bb1987bff3582b0fe825f3f8bc58650c7befc7b91f80fcb` |
| `engineering_conformance.pdf` | 11 | `a50d4012816adf107ac9ec6978f85102474b27ecf16b4089a6645e2cfcc49b92` |

Each root PDF byte-matches its corresponding `build/` copy. PDF metadata
contains the full draft contract version and document revision in title and
subject. The entrypoints force the same visible version/revision header on all
standard and special page styles, including longtable continuation pages.
Representative normal and continuation pages in both final views were
rendered at 110 dpi and visually checked for the header, legible equations,
table boundaries, clipping, and overlap.

## Reproduction commands

Run from this output directory:

```sh
mkdir -p build
/opt/homebrew/bin/tectonic --only-cached --keep-logs --outdir build src/scientific_rationale.tex
/opt/homebrew/bin/tectonic --only-cached --keep-logs --outdir build src/engineering_conformance.tex
```
