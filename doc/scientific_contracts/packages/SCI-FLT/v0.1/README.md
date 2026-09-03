# SCI-FLT — Map-Domain Filtering Tranche

Status: **SCI-FLT-FIXED v0.1 conditionally scientifically frozen;
implementation conformity and numerical validation not established.**

Version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package follows the
[Scientific Contract Library Program](../../../README.md),
[pilot process](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
It begins from the current frozen/approved SCI-PTC, SCI-MAP, SCI-JINC,
SCI-VAL, RTC, CAL, ALIGN, AST, SCI-BEAM, and SCI-NOI Stage A authority without
modifying them.

Recovery and classification are recorded in [`PRIOR_WORK.md`](PRIOR_WORK.md).
Implementation-informed evidence remains quarantined in
[`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md) and is excluded from future
authorship. Current SCI-NOI Stage B draft material is also excluded; approved
SCI-NOI Stage A controls the transformation/uncertainty boundary.

## Owner-Resolved Package Architecture

- `SCI-FLT` remains the tranche name.
- `SCI-FLT-FIXED` is the first package; `SCI-FLT-DET` is rejected because of
  the detector-namespace collision.
- `SCI-FLT-INF` is a non-authoritative holding tranche only. It has no combined
  Stage B authority.
- Wiener, matched/template-amplitude, source-learned, data-derived selection,
  automatic method selection, and per-member relearning remain separate
  inference-bearing work.

## SCI-FLT-FIXED v0.1 Scope

The first contract contains one scientific object:

\[
  y = J_{\rm full}L_\Theta m,
\]

a strict-linear, same-grid, externally resolved fixed map-domain
transformation with fixed convolution as its concrete family. A fixed-low-pass-
convolution subtype is admitted only as a complete qualified transfer claim.
Full-footprint-only convolution is the sole v0.1 edge/missing method. Affine
offsets, boundary extension, truncation, support renormalization, reprojection,
adaptive/inference-bearing state, and coaddition are excluded.

## Repaired Stage A Artifacts

The owner-review set includes:

- revised [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md);
- package/tranche and operator/product taxonomy in
  [`AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md`](AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md);
- exact MAP, JINC, and NOI boundaries;
- strict-linear operator/convolution specification;
- WCS/kernel/discretization, edge/missing, normalization/unit/beam,
  response/null-space/covariance, and observation/coadd tables;
- FLT-owned VAL profile drafts and the atomic product/lifecycle table;
- content-bound
  [`SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`](SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md);
- [`STAGE_A_CHANGE_LOG.md`](STAGE_A_CHANGE_LOG.md); and
- the exact SHA-bound 17-object
  [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

## Controlling NOI Boundary

SCI-FLT-FIXED owns the exact transformation, local response, support/validity,
deterministic covariance state, lifecycle, and failure. NOI applies the exact
same `J_full L_Theta` to every compatible admitted randomization and owns the
resulting empirical uncertainty. NOI neither chooses nor defines the filter.
Per-member state resolution is a separate inference-bearing method and cannot
mix with fixed-state members.

## Current Gate And Nonclaims

All bounded Stage A scope decisions are resolved. Implementation-blind Stage B
completed, and Grant Wilson conditionally froze the exact manifest-bound
candidate at Git commit
`43f4fe59ab23a591c1c9e17a2ac4b1fed0a9e613` on `2026-08-31`. The approval
record is retained on dedicated-branch tip
`7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5`; `AUTHORITY_MANIFEST.json` has
SHA-256
`69e6766f26396ba843ee29cfb89a48efd91b7e1b517ed90d3d93c87a63e55778`.
Ordinary numerical MAP/JINC parents remain unavailable and the FLT profiles
remain unregistered. No SCI-FLT-INF Stage B authority is created.

The integration reconciliation changes no algorithm or frozen authority and
creates no new scientific-freeze claim. This package makes no
implementation-conformity, validation, calibration, response/covariance
fidelity, performance, readiness, production, or Unity claim.
