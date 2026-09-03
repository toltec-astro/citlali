# SCI-MAP v0.1/r0.7.1 Freeze Verification

Date: `2026-08-28`

Status: complete; status-only promotion verified

Review boundary: scientific-contract package sources, exact adjacent
boundaries/Registry records, owner records, source manifest, parity reports,
and PDF artifacts only. Citlali implementation, tests, reductions,
observational validation, performance evidence, and production evidence were
not consulted.

## Bound Identity

- Approved content-bound candidate commit:
  `bd010e20eb8a7901aa677810aa7a5c982a436e07`.
- Scientific-owner freeze SHA-256:
  `91801005ba2f2bce6471a9f6f4ed0b79806c893f498b4f3cca9e81e26df39ce1`.
- Externally bound source-manifest SHA-256:
  `bd3f172f8bb1e17bf95fde034ad76632439c766ef89b0a992361b9e6d79ada0a`.
- Durable verifier SHA-256:
  `68fa1f139e3888e09e941bb4399f2a81f0142955f4abd5d4c7ea1a15e55ba3d0`.
- Canonical formal PDF: 29 US Letter pages; SHA-256
  `4249348517d9be1fe8fe5535987258bffb9ce276046032a8db45575fca75e8b8`.
- Canonical science-team rationale PDF: 14 US Letter pages; SHA-256
  `24f2537ae727db3a024c8df5395f0af47061c0f4066afea8956a1adeac23f14c`.
- Canonical engineering-conformance PDF: 22 US Letter pages; SHA-256
  `60d2a3175b93148098a08a21bf93e0213eb23b0c35e4a7db6b73915cb652b1b8`.

The r0.7.1 revision-bearing aliases remain byte-identical to their canonical
PDFs. The status-only freeze does not rebuild or alter any PDF or scientific
source byte.

## Mechanical Verification

The durable package verifier passes with:

- 52 sequential unique requirement identifiers;
- 25 sequential unique prediction identifiers;
- complete requirement and prediction crosswalk coverage;
- nine owner-decision identifiers, retaining eight open decisions and
  resolved `SCI-MAP-OD-008`;
- exact shared-authority import and formal/ECS inventory parity;
- no independent normative science in the engineering wrapper;
- exact PTC/MAP boundary byte equality;
- valid external SHA-256 binding of `SOURCE_MANIFEST_R0.7.md`;
- exact whole-file, Registry-record, and all seven source-binding-row hashes;
- stable/revision-bearing PDF byte equality; and
- expected PDF page counts and nonempty extracted text on every page.

`git diff --check`, direct PTC/MAP boundary comparison, and the manifest
companion checksum also pass.

## Visual Verification

All 65 canonical pages were rendered with Poppler and inspected page by page
in the candidate review. Title/status blocks, headers, footers, contents,
equations, tables, formal inventories, registers, and references show no
clipping, overlap, broken glyphs, missing content, or blocking layout defect.
The status-only freeze changes no rendered bytes, so the exact r0.7.1 visual-QA
record remains applicable.

## Claim Boundary

This verification confirms document identity, internal consistency, source
binding, and rendering quality for the frozen scientific authority. It does
not establish implementation conformity, representation fidelity, numerical
or observational validation, achieved response fidelity, performance,
science qualification, default activation, production readiness, production
authorization, or numerical-route availability.
