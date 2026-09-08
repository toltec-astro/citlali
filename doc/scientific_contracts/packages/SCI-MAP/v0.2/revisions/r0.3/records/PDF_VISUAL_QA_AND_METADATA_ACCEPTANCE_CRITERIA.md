# SCI-MAP v0.2/r0.3 PDF visual-QA and metadata acceptance criteria

Status: **scientific-author acceptance criteria only**. Actual compilation,
page rendering, inspection, metadata extraction, PDF hashing, and report
identity belong to the manager-owned build packet.

Each stable PDF must:

- be built from the exact sealed entry source and shared-core digest recorded
  in `bindings/SCI-MAP_SOURCE_BINDING_v0.2_r0.3_CANDIDATE.md`;
- place its full identity block on the physical front cover after the title;
- show scientific owner Grant Wilson, contract v0.2, document r0.3, date
  2026-09-08, and status `Candidate for scientific-owner disposition`;
- state the same document-role hierarchy: shared source sole normative
  authority, formal PDF canonical rendered normative view, rationale
  explanatory, and ECS prospective procedure;
- carry PDF title/author/subject metadata consistent with the visible cover;
- resolve all references and transitive inputs with no missing glyph,
  overfull/clipped content, overlapping text, broken table, or unreadable
  typography;
- reopen and render every page through an independent renderer; and
- use these stable paths:
  `pdf/SCI-MAP-FORMAL-SCIENTIFIC-ENGINEERING-CONTRACT-v0.2.pdf`,
  `pdf/SCI-MAP-SCIENTIFIC-RATIONALE-v0.2.pdf`, and
  `pdf/SCI-MAP-ENGINEERING-CONFORMANCE-v0.2.pdf`.

The manager report must identify exact build recipe/tool bytes, generated
cover fragments, build attempt, entry/core hashes, PDF hashes, page counts,
all-page render inventory, metadata extraction, and every material finding.
A clean visual/build result establishes document integrity only. It does not
establish scientific acceptance, implementation conformity, fidelity,
validation, performance, readiness, or production authorization.
