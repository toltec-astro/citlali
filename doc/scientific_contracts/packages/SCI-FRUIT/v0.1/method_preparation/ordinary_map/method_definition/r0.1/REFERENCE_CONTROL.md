# Current source and proposed-reference control

The [full source identities](SOURCE_IDENTITIES.json) bind every copied input to
its exact source commit/path or owner attachment, byte count and SHA-256.
The [reference inventory](AUTHOR_REFERENCE_INVENTORY.json) contains exactly the
full rows marked `proposed_author_reference=true`. Neither record is a future
sanitized author packet or dispatch authorization.

| Current source group | Proposed author reference | Authority and use |
| --- | --- | --- |
| `inputs/authority/fruit/`, `inputs/authority/PTC_APPLICATION_REFERENCE.tex`, `inputs/authority/map/` | true | Retained exact scientific/status references from the approved scope. Read only within their stated authority; unavailable states remain unavailable. |
| `inputs/program/README.md` | true | Process-only charter. No numerical science or method selection. |
| `inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md` | true | Process-only accepted workflow. No numerical science or method selection. |
| `inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md` | true | Process-only sequencing. No numerical science or method selection. |
| `inputs/manager/` | false | Additional frozen PTC/CAL boundary recovery, exact MAP source-lock copies and prior internal dossier/scope/decision material. Manager review only; not automatically admitted to future independent authorship. |
| `inputs/owner/` | false | Exact owner direction governing this manager task; later author-control contents need separate exact approval. |

This repairs the three process-reference flags in the current successor
controls. The reviewed r0.2 packet and tar.gz retain their original bytes,
including the earlier mismatch; their pending-scope wording and conflicting
permission flags are superseded by this control and the
[scope approval](OWNER_SCOPE_APPROVAL_2026-09-08.md). They are historical records,
not a second active source-permission authority.

MAP is bound only to frozen v0.1/r0.7.1, promoted at
`bd010e20eb8a7901aa677810aa7a5c982a436e07`:

- Freeze SHA-256: `91801005ba2f2bce6471a9f6f4ed0b79806c893f498b4f3cca9e81e26df39ce1`.
- Source manifest SHA-256: `bd3f172f8bb1e17bf95fde034ad76632439c766ef89b0a992361b9e6d79ada0a`.
- Shared wrapper plus six modules, manifest order, aggregate SHA-256:
  `649e1694b2a6353b0e5e8cb42ab73d3f03b1bc5cad671810361e802a87278e9b`.

PTC is bound to its frozen r0.5 authority, promoted at
`8f0ecccfacbdce0543141c4289ec06c702065f5e`; CAL science is bound to frozen r0.5,
promoted at `0b3cfb24070c1eda04dbda7633accf40e2e8b852`.
The FRUIT seven-source inventory SHA-256 remains
`3dc912d9f6a58d9f92561ad10b9442485c5b47232e5327389d2fb3b89c296e92`.
All individual source bytes and review-copy locations are in the source records.

The proposed BC-IN/BC-OUT and coefficient/QC changes are owner-facing text only.
No controlled successor is adopted and no source lock/profile is modified.
Copied source context links are non-transitive: they do not admit implementation,
empirical material, referenced repositories or an unapproved candidate revision.
The two opaque untracked archives and reduction products are not packet inputs.
