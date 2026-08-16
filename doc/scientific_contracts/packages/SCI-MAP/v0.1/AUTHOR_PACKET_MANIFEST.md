# SCI-MAP v0.1 — Author Packet Manifest

Status: owner-approved, content-bound author packet

Scientific owner: Grant Wilson

Approval date: `2026-08-16`

## Allowed Inputs

The fresh implementation-blind scientific author may open only these three
logical packet items:

1. [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md)
2. the pair consisting of
   [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md) and the exact
   independent core
   `c28f18ed089657dae278caba2d6d6d65c7ec72f4:doc/audits/packages/SCI-MAP-001_INDEPENDENT_CORE.tex`
3. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)

| Logical item | Exact source | Content SHA-256 |
| --- | --- | --- |
| 1 — approved Scope Brief | `SCOPE_BRIEF.md` | `e2a9eb51edb5956191813b4cdbd23866e875d52cdf89cd8b6c272988b4f26674` |
| 2a — supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `8ea283525f18199d9760c3f672d145d71f0db87b320ab2e88a5c6635ef3d4aa0` |
| 2b — independent core | exact Git object named above | `13dd5922bd492e381afcc3b015284216dde1ccc2199ece3d070ee577c7324381` |
| 3 — conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `2d478cb6c5e897308d19614b8b01663318744971850c67459f84c7ddcd57c5c9` |

The hashes identify the exact bytes admitted to the author task. A content
change requires owner review and a new manifest rather than silent packet
drift.

## Prohibited Inputs

The author must not open:

- [`PRIOR_WORK.md`](PRIOR_WORK.md),
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md), or
  [`DECISION_LOG.md`](DECISION_LOG.md);
- any Citlali implementation, executable product/config contract, current
  interface, test, generated product, or source-specific explanation;
- any SCI-MAP-001/002/003 audit, finding, repair, re-audit, numerical
  execution, Unity, validation, conformity, integration, or production-status
  material other than the exact independent core admitted above;
- the later weighted-method note, raw ADR, owner amendment, implementation
  records, or active ALIGN material; or
- any unlisted repository, local file, web source, or model-memory substitute.

If the allowed packet is insufficient, the author must return one precise
scientific question to the manager. It may not search for an answer.

## Author Deliverables

The author writes only within this package's `src/`, `pdf/`, `CROSSWALK.md`,
`AUTHOR_DRAFT_DECISIONS.md`, and
`SCIENTIFIC_OWNER_DECISION_LEDGER.md`. It must produce:

- shared canonical LaTeX modules for notation, definitions, equations,
  assumptions, requirements, and edge cases as needed;
- a scientist-facing *Scientific Rationale and Contract*, with the physical
  model first and the main narrative ordinarily limited to eight to twelve
  pages before formal appendices;
- an engineering-facing *Engineering Conformance Specification* expressing
  the same authority without implementation-specific mappings;
- stable `SCI-MAP-REQ-NNN` requirements and a complete crosswalk;
- a scientific-owner decision ledger containing every unresolved question,
  its state, authority, blocked claim, evidence needed, and affected artifact;
- rendered PDFs for both views, with contract version `v0.1` and initial
  document revision `r0.1` kept distinct;
- scientifically meaningful falsifiable predictions without running
  scientific validation; and
- explicit separation of engineering conformance, representation fidelity,
  observational performance, and production readiness.

The author must reuse and consolidate the admitted independent derivation; it
must not repeat that derivation merely to appear new. The draft is not frozen
merely because it compiles.
