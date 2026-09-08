# PDF visual QA and metadata — SCI-MAP v0.2/r0.2

Disposition: **PASS for all 75 final rendered pages** at build `SCI-MAP-R02-DOCUMENT-BUILD/candidate-build-05`. This is document-artifact QA only, not scientific-owner acceptance or implementation evidence.

All final pages were rendered through Poppler 26.05.0 at 100 dpi after strict PDF reopening and text extraction. Of these, 71 final page PNGs are byte-identical to satisfactory, visually inspected candidate-build-03 pages. All four changed pages were inspected anew at full size: formal page 1, rationale pages 1 and 2, and ECS page 1. [The exact page comparison](VISUAL_PAGE_COMPARISON.json) records every retained or changed page; [the render report](PDF_BUILD_INSPECTION.json) records every final PNG SHA256 and renderer result.

The final rationale explicitly qualifies admission by selected role/profile, preserves base signal membership for unavailable or basis-incompatible response, publishes unavailable coadd response with its exact cause, and separately reports incomplete covariance. Covers bind the unchanged shared core, final entry sources and build identity. Every final page is free of clipping, overlap, overfull text, missing glyphs and broken tables. Normal paragraph continuation and intentionally separated front matter are retained. No font size or margin was reduced to obtain the page counts. Earlier attempts and their visual-review chain remain preserved in the recorded local evidence directories and the preceding committed packet.

| View | Physical pages | Cover and metadata |
| --- | ---: | --- |
| Formal contract | 34 | Grant Wilson; SCI-MAP v0.2; r0.2; candidate; 2026-09-07 |
| Science-team rationale | 16 | Grant Wilson; SCI-MAP v0.2; r0.2; candidate; 2026-09-07 |
| Engineering conformance | 25 | Grant Wilson; SCI-MAP v0.2; r0.2; candidate; 2026-09-07 |

Every first physical page contains the exact shared-core, original 56-reference manifest, packet addendum, owner directive and raw view-entry SHA256 values. Each names its source and the same build-attempt identity. PDF metadata titles include the correct view, v0.2, r0.2 and candidate status; author is Grant Wilson and creation date is 2026-09-07. Exact metadata and cover checks are in [the identity record](COVER_AND_RENDERED_IDENTITY_CHECKS.json).

Formal and ECS each render exactly 52 canonical requirement headings and 25 prediction headings. Every one of the 52 ECS result fields remains `not_assessed`. The rationale explains the same shared authority and uses the crosswalk instead of duplicating all requirement headings. No stronger scientific or application claim follows from a PDF opening or rendering successfully.

Final build-attempt record SHA256: `7278784119f9b60f6ef43d1352917c1444f092906778e2ece6cc34cb61b871f0`. Mechanical PDF inspection SHA256: `829f94a76422e86337454f92b8e55cb94fb2a21b36f2faebc6194c00a200cb1f`. Rendered PNGs, contact sheets, earlier attempts and their logs remain preserved under the absolute attempt directories recorded in the build evidence; the delivery archive contains the final PDFs, exact sources and reports.
