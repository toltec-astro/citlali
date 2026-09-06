# Cached PDF build and render receipt

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.4**  
Date: **2026-09-06**

## Revision and operation record

The sealed r0.3 manifest was verified at SHA-256
`87eafdbaed29dc8d810a5bba9d4ba818d57a837c04337a4c71c31b06215f0ee9`;
all 47 manifest-listed artifacts passed. Its complete 48-file final/source
inventory, including the manifest and excluding `build/`, was then copied
byte-for-byte into the new `output-r0.4` sibling before revision. All r0.4
source, companion, build, check, and raster work stayed beneath that sibling.
The sealed `output/`, `output-r0.2/`, and `output-r0.3/` packages were
not changed.

The required PDF artifact marker was invoked successfully exactly once for
this bounded r0.4 operation before `output-r0.4` was created, with operation
kind `edit`, expected output count 2, and output format `pdf`.

## Renderer and network posture

Both final PDFs were produced with local `/opt/homebrew/bin/tectonic` and
`--only-cached`. No installation, network retrieval, or external scientific
source was used. The output-local cached/default typography remains unchanged.

The first authoring compile exposed an output-local LaTeX delimiter error:
`Q_{\rm integrity}` had temporarily appeared outside math mode in the new
r0.4 explanatory sentence. That invocation stopped and no PDF or log from it
is a returned identity. The source was corrected to `\(Q_{\rm integrity}\)`
before the successful final pair was built. The retained final logs contain no
overfull box, undefined-control, undefined-reference, emergency, fatal, or TeX
error entry. Underfull table-spacing notices remain; page inspection found no
clipping or overlap.

## Final fixed PDF artifacts

| View | Pages | SHA-256 |
| --- | ---: | --- |
| `scientific_rationale.pdf` | 5 | `22907c022cb91e05364c20f2fb3d982d5a23b0211e652fab277e54ec9188b817` |
| `engineering_conformance.pdf` | 14 | `655192bef6a8db428d9b6bdc192318eeb78b2b983d2e53f0792961a97911d5c3` |

The retained final Tectonic logs are:

| Log | SHA-256 |
| --- | --- |
| `build/scientific_rationale.log` | `2559dbf20ef6e78d554d5c7a02cf311c675814831e155a052c6adee21e4b4260` |
| `build/engineering_conformance.log` | `60331f56ccecd152ac46d0619ba5824e1f318fdf3f2d55b25b4210ce058f0fbc` |

Each root PDF byte-matches its final build copy. PDF title and subject metadata
contain `v0.1-draft.1 / r0.4`, the visible title date is 2026-09-06, and
extracted opening text confirms the view name plus the
`v0.1-draft.1/r0.4` running header on every one of the 19 pages, including
longtable continuation pages.

All 5 scientific and 14 engineering pages were rendered at 110 dpi under
`build/qa-r0.4`. Contact-sheet inspection covered every page; full-resolution
inspection covered scientific page 2 and engineering pages 9, 12, and 13,
which carry the corrected identity/binding versus bound-byte-integrity
explanation, failure-scope table, UFC-REQ-017, UFC-MIX-008, and UFC-PRED-009.
The text is legible with no clipping or overlap. The 25 requirement, 16
prediction, 10 mixed-case, 14 open-proposal, and UFC-Q-001 inventories pass.
No PDF bytes were changed after these checks.

Manager independently rendered and inspected all 19 pages at 110 dpi against
the same two final hashes. That check passed with no clipping, overlap, missing
table, invalid glyph, or header defect; it also confirmed the complete 25/16/10
engineering inventories and the coherence of the bounded source correction.
The separate manager QA receipt was not read or copied into this scientific
packet. The final PDF bytes remained unchanged after the manager pass.

## Reproduction commands

Run from the `output-r0.4` directory:

```sh
mkdir -p build
/opt/homebrew/bin/tectonic --only-cached --keep-logs --outdir build src/scientific_rationale.tex
/opt/homebrew/bin/tectonic --only-cached --keep-logs --outdir build src/engineering_conformance.tex
```
