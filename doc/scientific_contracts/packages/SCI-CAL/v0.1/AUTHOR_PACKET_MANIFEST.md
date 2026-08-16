# SCI-CAL v0.1 — Author Packet Manifest

Status: owner-approved, content-bound author packet

Scientific owner: Grant Wilson

Approval date: `2026-08-16`

## Allowed Inputs

The fresh implementation-blind scientific author may open only these four
packet items:

1. [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md)
2. the pair consisting of
   [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md) and the exact
   independent core
   `27b0916e725696597c3ba84fb6a82bf6cf0ea356:doc/audits/packages/SCI-CAL-001_INDEPENDENT_CORE.tex`
3. the exact instrument manifest
   `8c581bfb26f01b187f4f1e0565f4457bcc25f099:doc/audits/packages/SCI-CAL-001_PASSBAND_AUTHORITY_001.json`
4. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)

| Logical item | Exact source | Content SHA-256 |
| --- | --- | --- |
| 1 — approved Scope Brief | `SCOPE_BRIEF.md` | `a3d6332dcc98bfbb638a45aea9be830edca693a4fe7e65d831b5880603c16578` |
| 2a — supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `57dba2d9fdc837902cf0768a20a9680462929e647a6649c1cb51676fad4638b2` |
| 2b — independent core | exact Git object named above | `106755520b048f601bc60fd04e7b6020e6fa470480ac3105fa7ba269c730a4fe` |
| 3 — passband authority | exact Git object named above | `2756908181cc466550399ec0a869e6671de7912bd3a935f9aeebf63e3e826617` |
| 4 — conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `7e9a630fd183ca04bc3d8bbd21b5e801776b9aad7bd084d48fa7f2c572766520` |

The hashes identify the exact bytes admitted to the author task. A content
change requires owner review and a new manifest rather than silent packet
drift.

## Prohibited Inputs

The author must not open:

- [`PRIOR_WORK.md`](PRIOR_WORK.md) or
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md);
- any Citlali implementation, executable contract, current interface, test,
  generated product, or source-specific explanation;
- the historical SCI-CAL audit or any finding, repair, re-audit, numerical
  execution, Unity, validation, conformity, or production-status material;
- raw owner-decision files already sanitized into the Scope Brief and cover;
- active ALIGN B3c material; or
- any unlisted repository, local file, web source, or model-memory substitute.

If the allowed packet is insufficient, the author must return one precise
scientific question to the manager. It may not search for an answer.

## Author Deliverables

The author writes only within this package's `src/`, `pdf/`, `CROSSWALK.md`,
and a draft scientific-substance section appended to `DECISION_LOG.md`. It
must produce:

- shared canonical LaTeX modules for notation, definitions, equations,
  assumptions, requirements, and edge cases as needed;
- a scientist-facing *Scientific Rationale and Contract*;
- an engineering-facing *Engineering Conformance Specification* expressing
  the same authority without implementation-specific mappings;
- stable `SCI-CAL-REQ-NNN` requirements and a complete crosswalk;
- rendered PDFs for both views;
- explicit unresolved owner decisions; and
- scientifically meaningful validation predictions, without running
  scientific validation.

The draft is not frozen merely because it compiles.
