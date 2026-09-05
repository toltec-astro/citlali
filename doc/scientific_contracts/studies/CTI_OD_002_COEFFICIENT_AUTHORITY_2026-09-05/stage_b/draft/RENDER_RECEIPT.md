# Cached PDF build and render receipt

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.2**  
Date: **2026-09-05**

## Revision and operation record

The r0.2 sibling began as an exact copy of the sealed r0.1 package's 43
final/source artifact files. The old `build/` scratch was excluded. The two
copied root PDFs were replaced by fresh r0.2 builds, and every log and QA
raster under this package's `build/` was created during this revision.

The required PDF artifact marker was successfully invoked exactly once for
this bounded edit operation, immediately before the new sibling directory was
created, with operation kind `edit`, expected output count `2`, and output
format `pdf`.

## Renderer and network posture

Both PDFs were produced with local `/opt/homebrew/bin/tectonic` using
`--only-cached`. No installation, network retrieval, or external scientific
source was used. The existing output-local upright-font substitution remains
unchanged from r0.1 because the cached TeX assets lack the requested slanted
metric; this repair made no typography change.

Both cached builds completed and reran TeX for references. Their final logs
contain no overfull box, undefined-reference, emergency, fatal, or TeX error
entry. Narrow-table underfull spacing notices remain; rendered inspection
found no clipping or overlap.

## Final artifacts

| View | Pages | SHA-256 |
| --- | ---: | --- |
| `scientific_rationale.pdf` | 13 | `3f67fe72c973cc015b062591d7fe9946b67fd6bbee1370ba32ad609b11991e7d` |
| `engineering_conformance.pdf` | 13 | `4083fe401d39b2193a0a420a6fde1af981dbbbc65b72d4106d607a8f3cc24d86` |

Each root PDF byte-matches its corresponding `build/` copy. PDF title and
subject metadata contain `v0.1-draft.1 / r0.2`. Text extraction checked the
visible `v0.1-draft.1/r0.2` running header at the opening of all 13 pages in
each view, including longtable continuation pages.

All 13 pages of each PDF were rendered at 110 dpi. The scientifically affected
scientific pages 3--5, 8--9, and 12--13 and engineering pages 2--5, 8, and
11--12 were visually inspected at full rendered resolution. The structural
gate, UQ-05 rule and restriction, four-axis distinction, failure-scope table,
requirements, predictions, equations, and table boundaries are legible, with
no clipping or overlap. Visible headers were also confirmed on these affected
pages; the all-page extraction check covers the remaining headers.

## Reproduction commands

Run from this output directory:

```sh
mkdir -p build
/opt/homebrew/bin/tectonic --only-cached --keep-logs --outdir build src/scientific_rationale.tex
/opt/homebrew/bin/tectonic --only-cached --keep-logs --outdir build src/engineering_conformance.tex
```
