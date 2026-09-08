# PDF visual QA and metadata — SCI-MAP v0.2/r0.2

Disposition: **PASS for all 75 final rendered pages** at build `SCI-MAP-R02-DOCUMENT-BUILD/candidate-build-03`. This is document-artifact QA only, not scientific-owner acceptance or implementation evidence.

All final pages were rendered through Poppler 26.05.0 at 100 dpi after strict PDF reopening and text extraction. Of these, 63 final page PNGs are byte-identical to satisfactory, visually inspected candidate-build-02 pages. All 12 changed pages were inspected anew: formal pages 1, 27, 28, 29, 31, 32; rationale page 1; ECS pages 1, 16, 17, 18, 22. Full-size checks additionally covered all three final covers and REQ-046 on formal page 28 and ECS page 18. [The exact page comparison](VISUAL_PAGE_COMPARISON.json) records every retained or changed page; [the render report](PDF_BUILD_INSPECTION.json) records every final PNG SHA256 and renderer result.

The final source correction makes reached-stage wording consistent in REQ-036/043/046 and PRED-012. Covers bind the new aggregate and build identity. Every final page is free of clipping, overlap, overfull text, missing glyphs and broken tables. Normal paragraph continuation and intentionally separated front matter are retained. No font size or margin was reduced to obtain the page counts. Earlier document attempts and their visual-review chain remain preserved in the recorded local evidence directories.

| View | Physical pages | Cover and metadata |
| --- | ---: | --- |
| Formal contract | 34 | Grant Wilson; SCI-MAP v0.2; r0.2; candidate; 2026-09-07 |
| Science-team rationale | 16 | Grant Wilson; SCI-MAP v0.2; r0.2; candidate; 2026-09-07 |
| Engineering conformance | 25 | Grant Wilson; SCI-MAP v0.2; r0.2; candidate; 2026-09-07 |

Every first physical page contains the exact shared-core, original 56-reference manifest, packet addendum, owner directive and raw view-entry SHA256 values. Each names its source and the same build-attempt identity. PDF metadata titles include the correct view, v0.2, r0.2 and candidate status; author is Grant Wilson and creation date is 2026-09-07. Exact metadata and cover checks are in [the identity record](COVER_AND_RENDERED_IDENTITY_CHECKS.json).

Formal and ECS each render exactly 52 canonical requirement headings and 25 prediction headings. Every one of the 52 ECS result fields remains `not_assessed`. The rationale explains the same shared authority and uses the crosswalk instead of duplicating all requirement headings. No stronger scientific or application claim follows from a PDF opening or rendering successfully.

Final build-attempt record SHA256: `6db62507ddea8068ff1d82f65364b2d52ea680a7c74ba4d8754ca5149af6cee6`. Mechanical PDF inspection SHA256: `861cfb17f2f7cdad1a17775ad9e416cb98a1217d53749df4eb24f6dbee2614bb`. Rendered PNGs, contact sheets, earlier attempts and their logs remain preserved under the absolute attempt directories recorded in the build evidence; the delivery archive contains the final PDFs, exact sources and reports.
