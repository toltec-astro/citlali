# SCI-ALIGN-001 phase-zero D005 owner decision — 2026-08-02

Status: `D005-Q1-HOLD` owner-approved for a bounded existing-use compatibility
adapter; the final phase-one sequencing card remains pending; phase one remains
unauthorized

Package: `SCI-ALIGN-001`

## Authority and evidence boundary

The project owner approved the coordinator's compatibility-only disposition of
`D005-Q1-HOLD`. The owner will separately request a strict `Hold` definition
from the responsible telescope engineer. That request is a future producer-
authority path and is not a prerequisite for considering the separate
phase-one sequencing card. It does not authorize repair or permission to infer
a bit meaning before the returned definition is reviewed.

This partial decision binds, but does not rewrite:

- exact governing application SHA
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`;
- phase-zero evidence commit
  `53c7154a3633dfe19dc036cfb5a6250f729a897d`;
- D005 evidence/preregistration commit
  `5a0d64b8f1b9b246b1b5d575c548269823203d22`, whose parent is the phase-zero
  evidence commit and which changes no application source;
- D005 checksum-list digest
  `149ef430af3223562d9e69b7224703b831f6f56629b2f3c513bf44c40a567bbb`;
- D005 `REPORT.md` SHA-256
  `6edf7a7bd79881c3e7f9809d1da36c56a763bb22cd9e7589e91950351536663a`;
- D005 owner brief SHA-256
  `0147b7e6d18e03a76b5cacf18fd955915ba78c9ddf4ce77714e4eebc0f3b8622`;
- D005 preregistration protocol SHA-256
  `b8208693fb19621264fedcb4752688c021c3551e5eb3cb4d91a1af3d05600abd`;
- D005 `Hold` findings SHA-256
  `a6daeb11a8681833ea7545bce1b8089336035ab82a823c4e502c00b82883e95e`;
  and
- the approved D001--D004 records and governing `ALIGN-OD5`/`ALIGN-C001`
  proportionality constraints.

The observed raw words `{0,2,8,10,64,66,72,74}` do not establish one physical
turnaround bit. `0x02` and `0x40` occur independently as well as beside
`0x08`. In Beammap 148670 every tested `Hold`-true row is also outside the
configured map box, so the tested raw predicate and transition-side hypotheses
produce the same composite final state. That is sufficient to bind existing-
use compatibility; it is not sufficient to claim producer semantics.

## D005-Q1-HOLD — compatibility raster segmentation

Decision: owner-approved with the following exact boundary.

1. For the exact previously accepted nonpolarimetric raster profiles, ALIGN
   may use the named `legacy_4x_linear_any_nonzero` view together with the
   separately applied governing outside-map-box condition to control raster
   segmentation. This is a `legacy_inferred` compatibility adapter, not a
   producer-authoritative `Hold`, turn, or telescope-state definition.

2. Preserve the complete finite, nonnegative, integral raw word and its source
   identity. Keep `turnaround_candidate_0x08` diagnostic-only and preserve
   `0x02`, `0x40`, and every other bit without assigning scan, validity,
   hardware, flagging, eligibility, or exposure meaning.

3. If phase one is separately authorized, apply only the already approved OD5
   repairs: use half-open window identity, include the first composite-final-
   state-false sample while keeping ordinary field/sample validity separate,
   preserve final partial support, separate science and context windows, and
   retain stable short, empty, rejected, or unusable identities rather than
   silently deleting or renumbering them. The exact 198 legacy outputs remain
   the governing compatibility baseline. The 241 conditional candidate
   identities are compact internal identities; the additional 43 short/partial
   identities do not automatically become numerical-consumer inputs or new
   science reductions.

4. Do not add routine per-sample `Hold`, transition, or scan-ID products.
   Normal identity remains implicit or compact; expanded forensic transition
   tables remain `as_requested` under `ALIGN-C001`.

5. Stop if implementation changes accepted scan state beyond the named
   boundary, context, partial-support, identity, or status repairs. This
   decision does not supply a nonzero scientific-product tolerance for an
   intentionally changed OD5 record.

Stronger physical raster segmentation, a new profile, or a consumer requiring
producer-authoritative turn state remains unavailable.

## Future telescope-engineer definition

The requested strict definition should, if available, identify the producing
system and field version; logical width and bit assignments; the physical
meaning of each used bit; transition/event timing and left/right or interval
support; validity, reset, and missing-state behavior; and whether map-box
exclusion is upstream state or a separate consumer rule.

When received, the coordinator must preserve the returned artifact and its
author/date/version, compare it against the raw-word evidence and compatibility
adapter, and route any behavioral change through an explicit contract
amendment. It may not silently rewrite this decision. If a repair candidate
has already been inspected or implemented, any changed predicate, transition
placement, or scan result becomes explicit repair/re-audit scope rather than a
post-hoc substitution.

## Explicit non-approvals and next owner gate

This partial decision does not authorize phase one, an application edit,
Unity evidence, a new production profile, physical `Hold` semantics, a nonzero
changed-product tolerance, re-audit, merge, rebase, push, or production
expansion.

The remaining D005 owner card is administrative and bounded. It will ask
whether to authorize the recommended minimal phase-one implementation/evidence
sequence while carrying unavailable native-rate, multi-Beammap, producer-span,
changed-product, and performance evidence as explicit later gates rather than
inventing or prematurely collecting them. Until that choice is approved, the
existing prerequisites and phase-one prohibition remain unchanged.
