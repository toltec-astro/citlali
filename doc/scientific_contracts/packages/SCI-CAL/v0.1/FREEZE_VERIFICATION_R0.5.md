# SCI-CAL v0.1/r0.5-r0.4 Freeze Verification

Date: `2026-08-23`

Status: complete; status-only promotion verified

Review boundary: package sources, owner records, crosswalk, and PDF artifacts
only. Citlali implementation, tests, reductions, validation results, and
production evidence were not consulted.

## Bound Identity

- Owner-approved content-bound candidate commit:
  `0b3cfb24070c1eda04dbda7633accf40e2e8b852`.
- Scientific-owner freeze SHA-256:
  `413426f49edf1249f751a05bb8c6e9fd907b11e8da0530fe2da39814885efb22`.
- Mechanical verifier SHA-256:
  `3e09c9c80254936313e75bbf9e0e1f7d15d27cdbc7ba8678c949bb7bf18281fe`.
- Canonical scientific-rationale PDF: 14 US Letter pages; SHA-256
  `fa6a11a359bcc4f54cb75ab5057ba56cf76fa8eef8d28ac3bcb7954963a12034`.
- Canonical engineering-conformance PDF: 26 US Letter pages; SHA-256
  `07dd88895b21ee02eca611bba8d1adcf90f9c37439f791437c4f46e351700101`.

## Mechanical Verification

The package verifier passes with:

- exact frozen hashes for the package authority records, shared sources,
  audience wrappers, and canonical PDFs;
- 11 sequential assumptions, 50 sequential requirements, and 30 sequential
  edge predictions;
- 50 exact crosswalk requirement rows;
- all `SCI-CAL-OWNER-Q01--Q09` identities in both the owner ledger and
  engineering authority;
- exact r0.5/r0.4 revision, shared-authority inclusion, lineage, atmosphere,
  photometric, CAL-before-PTC, closure, and transfer markers;
- active source and PDF status marked frozen with no retained active draft
  marker;
- explicit achieved-performance acceptance terminology; and
- both canonical PDFs unencrypted, US Letter throughout, and free of forms
  and JavaScript.

## Visual Verification

All 40 canonical pages were rendered with Poppler and inspected through
whole-document contact sheets. Title pages, status headers, contents,
equations, flow diagrams, owner-decision tables, uncertainty and claim tables,
long requirement blocks, edge predictions, crosswalk tables, footers, and page
transitions show no clipping, overlap, broken glyphs, missing content, or
unreadable layout. Dense science pages 8 and 13 and engineering pages 19--21
and 24 received additional full-size inspection.

## Claim Boundary

This verification establishes artifact identity, internal consistency,
status-clean rendering, and exact owner-approved scientific authority. It does
not establish implementation conformity, atmosphere-model fidelity,
observational validation, achieved performance, achieved-performance
acceptance, total calibrated uncertainty, science qualification, production
readiness, MAP availability, or clean-room audit-finding closure.

This freeze supplies the CAL artifact required by WP-3. `F-016` remains open
until WP-7 binds the exact source and completes the clean-room re-audit.
