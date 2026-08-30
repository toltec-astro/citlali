# SCI-FLT-FIXED v0.1 Stage B Verification Report

Report identity: `SCI-FLT-FIXED-STAGE-B-VERIFICATION v0.1/draft-r0.2`

Status: PASS; deterministic r0.2 closure verification only; scientific-owner review required

Build binding SHA-256: `5bbb0f5d681307fb99a1b21ce9785273ad6d5885da6059598919c07633cc1955`

Verifier SHA-256: `097d8e52ad0b6e40b019219d7d6f3822330da269e82b7208e98064bc5f375e49`

Poppler: `pdftoppm version 26.05.0`

## Results

- PASS: packet manifest external SHA-256 matches
- PASS: all 17 admitted object SHA-256 values match
- PASS: all authored r0.2 sources are ASCII-clean; the exact owner directive is preserved as UTF-8
- PASS: all 14 r0.2 owner-directive sections are present
- PASS: 44 stable requirement identifiers preserve 001-036 and append 037-044
- PASS: 24 stable prediction identifiers preserve 001-021 and append 022-024
- PASS: engineering-conformance view routes every stable identifier exactly once
- PASS: traceability covers every identifier and only the admitted Stage A packet plus the r0.2 owner directive
- PASS: every traced core, rationale, conformance, Stage A, and r0.2 owner section resolves
- PASS: both views import one shared normative core SHA-256 bbcce3546d895002af8f7bb847f32e734cb23e6b4963132a09e6f54ffe4d6481
- PASS: all three exact typed policy domains and unregistered VAL status are complete
- PASS: equation, requirement, and prediction semantic-change partitions are exact and preserve stable IDs
- PASS: every changed or appended identifier routes to an exact r0.2 owner-directive section
- PASS: launch commit, exact packet bytes, owner directive, r0.2 sources, and build tools match BUILD_BINDING.json
- PASS: all embedded font files and SHA-256 values match BUILD_BINDING.json
- PASS: all three PDF identity blocks, Grant Wilson author metadata, hashes, sizes, and page counts match
- PASS: every PDF page contains extractable text
- PASS: clean temporary rebuild reproduces all three PDF SHA-256 digests exactly
- PASS: Poppler renders every page of all three PDFs to a non-empty PNG
- PASS: VISUAL_QA.md records a PASS observation for every bound PDF page

## Bound PDF outputs

- `SCI-FLT-FIXED-v0.1-NORMATIVE-CORE-draft-r0.2.pdf`: 16 pages; 104874 bytes; SHA-256 `997fedb40edd0e6513d075effbb1067e349939c2bd813de3b863e65ca2f5bcb7`
- `SCI-FLT-FIXED-v0.1-SCIENTIST-RATIONALE-draft-r0.2.pdf`: 6 pages; 75953 bytes; SHA-256 `33d3bd8e98c89a3386c47027777e9c8f22a4b98deec9ba8b4d970fd7c513e29c`
- `SCI-FLT-FIXED-v0.1-ENGINEERING-CONFORMANCE-draft-r0.2.pdf`: 7 pages; 79097 bytes; SHA-256 `d27985b3c0f77a45158320dd02d93f80090f561722ee9363cdc6281747d4ded4`

## Visual review

The exact bound page-by-page visual-QA record was required and verified.

## Nonclaims

This report makes no implementation-conformity, algorithm-change, validation, calibration, achieved-response, achieved-covariance, numerical-adequacy, performance, readiness, scientific-freeze, production, or Unity claim.
