# SCI-FRUIT v0.1 — Focused Owner-Review Archive Checksum Report r0.8

Date: `2026-09-01`

Status: **archive generated and verified**

## Archive Binding

Archive path:
`SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review.tar.gz`

Archive byte count: `28617`

Archive SHA-256:
`5f11836908aa6aeb4f51690209a32dc6e8d4cee4e6b9c223903c6c57033b9b22`

Archive root:
`SCI-FRUIT-v0.1-ODQ-001F-r0.8-owner-review/`

Payload file count: `19`

## Manifest Self-Binding

The archive contains `BUNDLE_MANIFEST.md`. Because a manifest cannot contain
its own final SHA-256 without changing its bytes, this external report supplies
that final binding:

- `BUNDLE_MANIFEST.md` byte count: `5089`;
- `BUNDLE_MANIFEST.md` SHA-256:
  `3cdf1a8729a20f41a1abe665a845a1d9bcaaa6d7bf5be9c7fe010705638b0f4c`.

The included source-byte report has:

- `SOURCE_BYTE_AND_INTERNAL_LINK_REPORT_R0.8.md` byte count: `5327`;
- SHA-256:
  `0579e9f6ec5a388cd4c65b50e7c82b91b9fc5cd32d0bca4d411098b7a54d96f9`.

Together, the in-archive manifest and this external archive/manifest binding
cover every included file without a false self-referential hash claim.

## Verification Results

- Archive listing contains one root directory and exactly `19` regular Markdown
  files.
- Every staged archive member is byte-identical to its canonical package source.
- Every non-manifest payload byte count and SHA-256 matches
  [`BUNDLE_MANIFEST.md`](BUNDLE_MANIFEST.md).
- Every archive-local Markdown link resolves within the focused subset.
- No archive member matches AppleDouble pattern `._*` at any path depth.
- Extended-attribute archiving was disabled during creation.
- The Stage A package verifier and `git diff --check` passed before archive
  creation.

The checksum report is an external companion and is intentionally not included
inside the archive whose digest it records.

## Lifecycle Boundary

This checksum proves archive identity only. It does not approve ODQ-001F,
create a Stage B packet, launch an empirical lane, or authorize any scientific,
implementation, validation, readiness, production, fallback, or Unity action.
