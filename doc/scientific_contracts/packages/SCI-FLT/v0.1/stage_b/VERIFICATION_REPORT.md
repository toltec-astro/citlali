# SCI-FLT-FIXED v0.1 Stage B Verification Report

Report identity: `SCI-FLT-FIXED-STAGE-B-VERIFICATION v0.1/draft-r0.4`

Status: PASS; deterministic r0.4 proposed-freeze preflight only; scientific-owner review required

Build binding SHA-256: `316bdd5e4ad48d3444230002d33b1ef86cee000f2954e6f81240af2cc1af257d`

Verifier SHA-256: `22e58ea6605938745089b4025af3bc247536bb10124ec2d1a0170ad565ac7d89`

Poppler: `pdftoppm version 26.05.0`

## Results

- PASS: packet manifest external SHA-256 matches
- PASS: all 17 admitted object SHA-256 values match
- PASS: all authored r0.4 sources are ASCII-clean; all three exact owner directives are preserved as UTF-8
- PASS: all 14 r0.2, 10 r0.3, and 9 r0.4 owner-directive sections are present
- PASS: 53 stable requirement identifiers preserve 001-051 and append 052-053
- PASS: 30 stable prediction identifiers preserve 001-028 and append 029-030
- PASS: engineering-conformance view routes every stable identifier exactly once
- PASS: traceability covers every identifier and only the admitted Stage A packet plus the exact r0.2/r0.3/r0.4 owner directives
- PASS: every traced core, rationale, conformance, Stage A, and owner-directive section resolves
- PASS: both views import one shared normative core SHA-256 43c1a5f57cb72b03cfc4a99628ae4c86fcabdf0a338cac2cdc12e08774010999
- PASS: parent roles, owner footprint disposition, covariance table, immutable NOI design, policy actors, low-pass convention, response domain, and rationale preflight are closed
- PASS: all three typed policy domains, exact dispositions, actor boundaries, and unregistered VAL status are complete
- PASS: equation, requirement, and prediction semantic-change partitions are exact and preserve stable IDs
- PASS: every r0.4 amended or appended identifier routes to an exact r0.4 owner-directive section
- PASS: launch commit, exact packet bytes, all three owner directives, r0.4 sources, date, and build tools match BUILD_BINDING.json
- PASS: all embedded font files and SHA-256 values match BUILD_BINDING.json
- PASS: all three PDF identity blocks, Grant Wilson author metadata, date metadata, hashes, sizes, and page counts match
- PASS: every PDF page contains extractable text
- PASS: clean temporary rebuild reproduces all three PDF SHA-256 digests exactly
- PASS: Poppler renders every page of all three PDFs to a non-empty PNG
- PASS: VISUAL_QA.md records a PASS observation for every bound PDF page
- PASS: consolidated proposed-freeze authority inventory and external self-binding are complete

## Bound PDF outputs

- `SCI-FLT-FIXED-v0.1-NORMATIVE-CORE-draft-r0.4.pdf`: 21 pages; 120909 bytes; SHA-256 `39655bd46e90d6125251ec15e2a070e1f31af2e4cfec53cc8b53ae5039cb2a28`
- `SCI-FLT-FIXED-v0.1-SCIENTIST-RATIONALE-draft-r0.4.pdf`: 7 pages; 80280 bytes; SHA-256 `95b0d274ba6932ead17c6ead6452887086b16e2cb738078404a9ac9b5c05cc60`
- `SCI-FLT-FIXED-v0.1-ENGINEERING-CONFORMANCE-draft-r0.4.pdf`: 9 pages; 85675 bytes; SHA-256 `b72a5c16c13fc977552037c633bda119f26b945f26e9c61daa2133ca95b9bab9`

## Visual review

The exact bound page-by-page visual-QA record was required and verified.

## Nonclaims

This report makes no implementation-conformity, algorithm-change, validation, calibration, achieved-response, achieved-covariance, numerical-adequacy, performance, readiness, scientific-freeze, production, or Unity claim.
