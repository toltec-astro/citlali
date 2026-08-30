# SCI-FLT-FIXED v0.1 Stage B Verification Report

Report identity: `SCI-FLT-FIXED-STAGE-B-VERIFICATION v0.1/draft-r0.1`

Status: PASS; deterministic draft verification only; scientific-owner review required

Build binding SHA-256: `83cb8bfe87405fd5a527357e5e3397ac13bca3d278a7fa4312fe25b3ba0a741d`

Verifier SHA-256: `8a758ccce863e992af664f888cc1968310418c334c5baec92ad3a497c671c8b0`

Poppler: `pdftoppm version 26.05.0`

## Results

- PASS: packet manifest external SHA-256 matches
- PASS: all 17 admitted object SHA-256 values match
- PASS: all Stage B authored sources are ASCII-clean
- PASS: 36 stable requirement identifiers are complete, unique, and ordered
- PASS: 21 stable prediction identifiers are complete, unique, and ordered
- PASS: engineering-conformance view routes every stable identifier exactly once
- PASS: traceability covers every identifier and only admitted Stage A objects
- PASS: every traced core, rationale, conformance, and Stage A section resolves
- PASS: both views import one shared normative core SHA-256 867c48304906c43da1ac6f54893fe5988e7fbbeecde393519cdd7b0ea4f7a97f
- PASS: launch commit, packet, sources, and build tools match BUILD_BINDING.json
- PASS: all embedded font files and SHA-256 values match BUILD_BINDING.json
- PASS: all three PDF identity blocks, metadata, hashes, sizes, and page counts match
- PASS: every PDF page contains extractable text
- PASS: clean temporary rebuild reproduces all three PDF SHA-256 digests exactly
- PASS: Poppler renders every page of all three PDFs to a non-empty PNG
- PASS: VISUAL_QA.md records a PASS observation for every bound PDF page

## Bound PDF outputs

- `SCI-FLT-FIXED-v0.1-NORMATIVE-CORE-draft-r0.1.pdf`: 11 pages; 90085 bytes; SHA-256 `bf96137662124dac55777b3fbc9fb5a407c3810be087f80231bb0a15ebe63853`
- `SCI-FLT-FIXED-v0.1-SCIENTIST-RATIONALE-draft-r0.1.pdf`: 5 pages; 71885 bytes; SHA-256 `f0ec63d27ffbb0d341146fb7ba69a801f81ddfdd0a02eaedd7298a8bdd130b0e`
- `SCI-FLT-FIXED-v0.1-ENGINEERING-CONFORMANCE-draft-r0.1.pdf`: 6 pages; 75500 bytes; SHA-256 `bc83dfb1da2842b03032cf2b7f6cc5621306db8cb46b76f87b5afc464363c532`

## Visual review

The exact bound page-by-page visual-QA record was required and verified.

## Nonclaims

This report makes no implementation-conformity, algorithm-change, validation, calibration, achieved-response, achieved-covariance, numerical-adequacy, performance, readiness, scientific-freeze, production, or Unity claim.
