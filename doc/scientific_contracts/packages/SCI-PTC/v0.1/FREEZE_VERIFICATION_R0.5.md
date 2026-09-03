# SCI-PTC v0.1/r0.5 Freeze Verification

Date: `2026-08-23`

Status: complete; status-only promotion verified

Review boundary: package sources, owner records, generated crosswalk, and PDF
artifacts only. Citlali implementation, tests, reductions, validation results,
and production evidence were not consulted.

## Bound Identity

- Approved content-bound candidate commit:
  `8f0ecccfacbdce0543141c4289ec06c702065f5e`.
- Scientific-owner freeze SHA-256:
  `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`.
- Mechanical verifier SHA-256:
  `9ffe2fe9f189d0ef152c11dbc53b0594602fe01602b9f73492ac9068477dcc3b`.
- Canonical scientific-rationale PDF: 13 US Letter pages; SHA-256
  `3c927dbcb631b2033f04d933fa4d69911698b840b7b4642e759ad8c715c16ab6`.
- Canonical engineering-conformance PDF: 26 US Letter pages; SHA-256
  `f0b3bdc28c0997e4960b8231d6113339fe391d331e84dc7c0638912b0d3f7adb`.

The separately marked candidate PDFs retain their exact pre-freeze hashes.
The r0.4 freeze record retains its exact digest, and the superseded r0.4 source
and PDF bytes remain recoverable from immutable Git history.

## Mechanical Verification

The package generator and verifier pass with:

- 45 sequential definitions;
- 25 numbered equations;
- 35 sequential assumptions;
- 99 sequential requirements;
- 60 sequential predictions;
- 159 exact generated crosswalk rows with resolved rationale locators;
- 38 author decisions;
- 15 owner entries: five decided, three open, four known-but-not-supplied, and
  three deferred;
- both canonical PDFs US Letter, unencrypted, and free of forms and
  JavaScript;
- all 159 normative IDs present in the engineering PDF;
- no independent displayed mathematics in the engineering wrapper; and
- exact binding of the r0.5 owner freeze, preserved r0.4 freeze, canonical
  frozen PDFs, and candidate-review history.

## Visual Verification

All 39 canonical pages were rendered with Poppler and inspected page by page.
The title/status blocks, headers, footers, contents, equations, figures,
longtables, formal registers, provenance section, and references show no
clipping, overlap, broken glyphs, missing content, or unreadable layout.
Equation 8 and its finite time-local full-rank guard are clean and legible.

## Claim Boundary

This verification confirms document identity, internal consistency, and
rendering quality for the frozen scientific authority. It does not establish
implementation conformity, numerical or observational validation, achieved
performance, science qualification, production readiness, MAP availability,
or audit-finding closure.
