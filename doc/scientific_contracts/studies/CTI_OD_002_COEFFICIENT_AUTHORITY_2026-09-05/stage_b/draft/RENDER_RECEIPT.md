# Cached PDF build and render receipt

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3**  
Date: **2026-09-06**

## Revision and operation record

The r0.3 sibling began with the 43 final/source files from sealed r0.2; r0.2 build scratch was excluded. The four byte-exact owner-review supplement files and the new owner-response artifact were then added. All r0.3 source, compilation, checks, and rasters stayed beneath output-r0.3. The sealed output and output-r0.2 packages were not changed.

The required PDF artifact marker was invoked successfully exactly once for this bounded r0.3 operation before output-r0.3 was created, with operation kind edit, expected output count 2, and output format pdf.

## Renderer and network posture

Both final PDFs were produced with local /opt/homebrew/bin/tectonic using --only-cached. No installation, network retrieval, or external scientific source was used. The output-local upright-font substitution inherited from the prior draft remains unchanged because the cached TeX assets lack the requested slanted metric; no typography revision was made.

A first final-content invocation inside the restricted process sandbox stopped before TeX with the local macOS system-configuration NULL-object panic. The same two cached commands were rerun with the permitted local process access and succeeded. The final build PDFs and logs replaced the earlier preview outputs. Final logs contain no overfull box, undefined-control, undefined-reference, emergency, fatal, or TeX error entry. Engineering tables retain underfull spacing notices; page inspection found no clipping or overlap.

## Final fixed PDF artifacts

| View | Pages | SHA-256 |
| --- | ---: | --- |
| scientific_rationale.pdf | 5 | efd265d423929dad071a499d3b81a0354d0a3bda2e21ebf7cc2c8ca5b192c266 |
| engineering_conformance.pdf | 14 | f476d16ae997770a7fb9b3c2ebd24937da037fc70354de574b2c4d820e16013c |

The final Tectonic log SHA-256 values are:

| Log | SHA-256 |
| --- | --- |
| build/scientific_rationale.log | 8d27030161177989b4b3b8796a9f190fc81e78123c5881bb2a14b06623f014a0 |
| build/engineering_conformance.log | 67d2b644cec3382d6da37090425502078d18a5c0d31f649fe3a22c0d1dce3674 |

Each root PDF byte-matches its final build copy. PDF title and subject metadata contain v0.1-draft.1 / r0.3, the visible title date is 2026-09-06, and extracted opening text confirms the view name plus v0.1-draft.1/r0.3 running header on every one of the 19 pages, including longtable continuation pages.

All 5 scientific and 14 final engineering pages were rendered at 110 dpi under build/qa-r0.3. Local contact-sheet and full-resolution inspection found all headings, equations, tables, decision register, requirements, mixed cases, predictions, and body text legible without clipping or overlap. The scientific four-axis/finalization/nonexceptionability paragraph and exact requirements 072/090/094/095 citations were visible. The restored formal profile digest, source-promotion rule, aggregation status, lineage, and scope/influence text are legible on engineering page 3. All-page header and metadata checks, plus the 25 requirement, 16 prediction, and 10 mixed-case inventories, pass.

Manager all-page inspection first passed the intermediate efd265d423929dad071a499d3b81a0354d0a3bda2e21ebf7cc2c8ca5b192c266 / 56afc891ab72e36859e07669f0c8b05315532b39775735d0daf88b9435119598 pair. A final preservation check then found two admitted profile fields omitted by the restructuring. That pair is retained only as a labeled superseded QA checkpoint under build/superseded-qa-checkpoint-before-profile-binding-restoration. Restoring the fields left the scientific PDF byte-identical and changed/reflowed extracted engineering text on pages 3--14; pages 1--2 remained text-identical. The root PDFs and ordinary build copies now contain only the final identities in the table above.

Manager final QA then rechecked the fixed pair: scientific bytes remained identical, engineering pages 1--2 were pixel-identical, and changed pages 3--14 were all inspected. The restored digest, aggregation, one-way lineage, and scope/influence clauses were verified in rendered text; all 25/16/10 stable IDs passed again. No clipping, overlap, header, metadata, or inventory defect remained. The final PDF bytes were not changed after this pass.

## Reproduction commands

Run from the output-r0.3 directory:

```sh
mkdir -p build
/opt/homebrew/bin/tectonic --only-cached --keep-logs --outdir build src/scientific_rationale.tex
/opt/homebrew/bin/tectonic --only-cached --keep-logs --outdir build src/engineering_conformance.tex
```
