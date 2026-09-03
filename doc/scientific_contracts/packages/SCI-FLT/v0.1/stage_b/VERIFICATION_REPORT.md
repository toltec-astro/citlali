# SCI-FLT-FIXED v0.1 Freeze-Candidate Verification Report

Report identity: `SCI-FLT-FIXED-STAGE-B-VERIFICATION v0.1/freeze-candidate`

Status: PASS; deterministic conditional-freeze-candidate preflight; owner signature required; no scientific freeze yet established

Build binding SHA-256: `394af9834e6452ca3a17bc56d2d3ce66101fe82338b427be4320e5c2aca256de`

Verifier SHA-256: `22d28dfd10e363bd2389ac3fe9c5ea5cd69ce221744f54f6b91a3e3010fc1956`

Poppler: `pdftoppm version 26.05.0`

## Results

- PASS: packet manifest external SHA-256 matches
- PASS: all 17 admitted object SHA-256 values match
- PASS: all authored freeze-candidate sources are ASCII-clean; all four exact owner directives are preserved as UTF-8
- PASS: all 14 r0.2, 10 r0.3, 9 r0.4, and 9 final owner-directive sections are present
- PASS: all 53 stable requirement identifiers remain exactly 001-053 without renumbering or addition
- PASS: all 30 stable prediction identifiers remain exactly 001-030 without renumbering or addition
- PASS: engineering-conformance view routes every stable identifier exactly once
- PASS: traceability covers every identifier and only the admitted Stage A packet plus the exact r0.2/r0.3/r0.4/final owner directives
- PASS: every traced core, rationale, conformance, Stage A, and owner-directive section resolves
- PASS: both views import one shared normative core SHA-256 7147d242f54d64ca80f6d3a17d309c65f75180457629fed62ce676da93b11089
- PASS: parent roles, owner footprint disposition, covariance table, immutable NOI design, policy actors, low-pass convention, response domain, and rationale preflight are closed
- PASS: all three typed policy domains, exact dispositions, actor boundaries, and unregistered VAL status are complete
- PASS: equation, requirement, and prediction semantic-change partitions are exact and preserve stable IDs
- PASS: every final amended identifier routes to an exact final owner-directive section; no identifier was appended or renumbered
- PASS: launch commit, exact packet bytes, all four owner directives, freeze-candidate sources, date, and build tools match BUILD_BINDING.json
- PASS: all embedded font files and SHA-256 values match BUILD_BINDING.json
- PASS: all three PDF identity blocks, Grant Wilson author metadata, date metadata, hashes, sizes, and page counts match
- PASS: every PDF page contains extractable text
- PASS: clean temporary rebuild reproduces all three PDF SHA-256 digests exactly
- PASS: Poppler renders every page of all three PDFs to a non-empty PNG
- PASS: VISUAL_QA.md records a PASS observation for every bound PDF page
- PASS: single conditional-freeze-candidate authority inventory and external self-binding are complete

## Bound PDF outputs

- `SCI-FLT-FIXED-v0.1-NORMATIVE-CORE-freeze-candidate.pdf`: 22 pages; 126362 bytes; SHA-256 `03390e2a8726867f1257cc50321af2c4f3f8186c8a3e4fa436e49d5df91a37ca`
- `SCI-FLT-FIXED-v0.1-SCIENTIST-RATIONALE-freeze-candidate.pdf`: 7 pages; 81325 bytes; SHA-256 `881a260459a707493303d2130490af2a8f2351e7564758ecb2558b582adbec76`
- `SCI-FLT-FIXED-v0.1-ENGINEERING-CONFORMANCE-freeze-candidate.pdf`: 9 pages; 87340 bytes; SHA-256 `5b1a20ee16059d83f87c0a9004bbec6f9184881c7ea90edf8f9c046b074bb78e`

## Visual review

The exact bound page-by-page visual-QA record was required and verified.

## Nonclaims

This report makes no implementation-conformity, algorithm-change, validation, calibration, achieved-response, achieved-covariance, numerical-adequacy, performance, readiness, scientific-freeze, production, or Unity claim.
