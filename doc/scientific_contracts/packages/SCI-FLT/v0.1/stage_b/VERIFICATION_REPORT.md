# SCI-FLT-FIXED v0.1 Stage B Verification Report

Report identity: `SCI-FLT-FIXED-STAGE-B-VERIFICATION v0.1/draft-r0.3`

Status: PASS; deterministic r0.3 proposed-freeze preflight only; scientific-owner review required

Build binding SHA-256: `578afdad159b4d4330f1cf3e3fb48b302a72c00c3b761467a69b62873a352d72`

Verifier SHA-256: `9ca23c97a7080015f90e1ec3644b81c49cc0a0d9436f32c5aa65cbdc6a29b5b5`

Poppler: `pdftoppm version 26.05.0`

## Results

- PASS: packet manifest external SHA-256 matches
- PASS: all 17 admitted object SHA-256 values match
- PASS: all authored r0.3 sources are ASCII-clean; both exact owner directives are preserved as UTF-8
- PASS: all 14 r0.2 and all 10 r0.3 owner-directive sections are present
- PASS: 51 stable requirement identifiers preserve 001-044 and append 045-051
- PASS: 28 stable prediction identifiers preserve 001-024 and append 025-028
- PASS: engineering-conformance view routes every stable identifier exactly once
- PASS: traceability covers every identifier and only the admitted Stage A packet plus the exact r0.2/r0.3 owner directives
- PASS: every traced core, rationale, conformance, Stage A, and owner-directive section resolves
- PASS: both views import one shared normative core SHA-256 98f3966e9d65ad57efcf7c8ba57344d1f9cb5635a05309eb5d068e4ab62ca9e1
- PASS: parent roles, owner footprint disposition, covariance table, immutable NOI design, policy actors, low-pass convention, response domain, and rationale preflight are closed
- PASS: all three corrected typed policy domains, actor boundaries, and unregistered VAL status are complete
- PASS: equation, requirement, and prediction semantic-change partitions are exact and preserve stable IDs
- PASS: every r0.3 changed or appended identifier routes to an exact r0.3 owner-directive section
- PASS: launch commit, exact packet bytes, both owner directives, r0.3 sources, date, and build tools match BUILD_BINDING.json
- PASS: all embedded font files and SHA-256 values match BUILD_BINDING.json
- PASS: all three PDF identity blocks, Grant Wilson author metadata, date metadata, hashes, sizes, and page counts match
- PASS: every PDF page contains extractable text
- PASS: clean temporary rebuild reproduces all three PDF SHA-256 digests exactly
- PASS: Poppler renders every page of all three PDFs to a non-empty PNG
- PASS: VISUAL_QA.md records a PASS observation for every bound PDF page
- PASS: consolidated proposed-freeze authority inventory and external self-binding are complete

## Bound PDF outputs

- `SCI-FLT-FIXED-v0.1-NORMATIVE-CORE-draft-r0.3.pdf`: 18 pages; 112745 bytes; SHA-256 `ed6d58323b85db7db6c25bebedc1312fec99e8890645e38abdfe0a083556c50b`
- `SCI-FLT-FIXED-v0.1-SCIENTIST-RATIONALE-draft-r0.3.pdf`: 6 pages; 77602 bytes; SHA-256 `923516fcf18cb5ce29a83092a2eb47b43c841fef17de849b1085b13dd5014c0a`
- `SCI-FLT-FIXED-v0.1-ENGINEERING-CONFORMANCE-draft-r0.3.pdf`: 8 pages; 82610 bytes; SHA-256 `2e9af76177a8ae7f189c9495456ab4fde3b64c0891e20997fed09f65d8d49f12`

## Visual review

The exact bound page-by-page visual-QA record was required and verified.

## Nonclaims

This report makes no implementation-conformity, algorithm-change, validation, calibration, achieved-response, achieved-covariance, numerical-adequacy, performance, readiness, scientific-freeze, production, or Unity claim.
