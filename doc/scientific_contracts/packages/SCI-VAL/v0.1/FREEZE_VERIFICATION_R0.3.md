# SCI-VAL v0.1/r0.3 Freeze Verification

Date: `2026-08-24`

Status: complete; status-only promotion verified

Review boundary: package sources, continuing registries, owner records,
crosswalk, and PDF artifacts only. Citlali implementation, tests, reductions,
validation results, performance evidence, and production state were not
consulted.

## Bound Identity

- Owner-approved content-bound candidate commit:
  `3ad018e97e134a0b0324d3fa2674ef96d5a680d4`.
- Candidate-manifest SHA-256:
  `314823249917e09d36ba76557699c1fbd1ba29171b3604a9b6d74cea8ca5d7f1`.
- Scientific-owner freeze SHA-256:
  `2fc3b3ad329fe3035d442b43d1e564a74fc86ab49f85f56e87322d8553fad9a6`.
- Mechanical verifier SHA-256:
  `b11342d962ed2fd01e881f48cb36824ef0ca971b55107f80e333aca113270fdb`.
- Canonical scientific-rationale PDF: 8 US Letter pages; SHA-256
  `53e32a12ad4b60b4cccaaf05e1c0f9ad248d7e31637fbcfe2b4344992b81359c`.
- Canonical engineering-conformance PDF: 20 US Letter pages; SHA-256
  `e5b353d52303e7f9fd3d10abcd35a4a15eb24021eab4e0663244d414052232fa`.

## Mechanical Verification

The package verifier passes with:

- exact hashes for the four original packet authorities and both r0.2/r0.3
  revision directives;
- the canonical independent-exposure profile, aggregate schema, four
  response/uncertainty roles, and continuing source bindings;
- 49 sequential requirements and 24 sequential predictions;
- exact 73-row crosswalk coverage;
- the standalone science-team rationale and complete six-module engineering
  view kept distinct;
- every formal identifier present in the engineering PDF; and
- expected 8-page and 20-page PDF counts.

An independent PDF structure check confirms both files are unencrypted,
612-by-792-point US Letter throughout, and contain no form fields, widget
annotations, or JavaScript. The six VAL Core modules retained the exact hashes
of the approved candidate; only active status text and associated records were
promoted.

## Visual Verification

All 28 canonical pages were rendered with Poppler at 140 dpi and inspected
through four whole-document contact sheets. Both title pages also received
full-resolution inspection. Title/status blocks, diagrams, tables, equations,
truth tables, long requirement and prediction blocks, appendices, page
numbers, and section transitions show no clipping, overlap, broken glyph,
missing content, or unreadable layout.

## Claim Boundary

This verification establishes artifact identity, internal consistency,
status-clean rendering, and exact owner-approved scientific authority. It does
not establish implementation conformity, observational validation, achieved
performance, production readiness, MAP or coadd availability, or clean-room
audit-finding closure.

The SCI-VAL freeze supplies the authority required by WP-5. `F-003`, `F-004`,
`F-005`, `F-016`, and `F-020` remain open until WP-7 performs the clean-room
re-audit and records the resulting closure dispositions.
