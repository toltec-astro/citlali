# SCI-BEAM v0.1 — Author Packet Manifest

Status: owner-approved, content-bound author packet

Scientific owner: Grant Wilson

Approval date: `2026-08-16`

## Allowed Inputs

The fresh implementation-blind scientific author may open this manifest and
only these three packet items:

1. [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md)
2. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)
3. [`AUTHOR_PRIMARY_REFERENCE_BOUNDARY.md`](AUTHOR_PRIMARY_REFERENCE_BOUNDARY.md)

| Logical item | Exact source | Content SHA-256 |
| --- | --- | --- |
| 1 — approved Scope Brief | `SCOPE_BRIEF.md` | `1a35a6c15c27769461a438548842828365f16cd76ba5f0b7c768067fad6a931f` |
| 2 — conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `db8b19cb0eed212c30e38c93b577f21f8b04e9d4356ae9355814591d25f47bec` |
| 3 — primary-reference boundary | `AUTHOR_PRIMARY_REFERENCE_BOUNDARY.md` | `7f9ff1486f80e781efb6c11a687725c859c1109f7947d05bd948befb496c51e1` |

The hashes identify the exact bytes admitted to the author task. A content
change requires owner review and a new manifest rather than silent packet
drift. The paper identifiers in item 3 are citation identities; the author may
use only the bounded paraphrased claims in item 3 and may not open the full
papers or other web sources.

## Prohibited Inputs

The author must not open:

- [`PRIOR_WORK.md`](PRIOR_WORK.md),
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md),
  [`DECISION_LOG.md`](DECISION_LOG.md),
  [`CROSSWALK.md`](CROSSWALK.md), or the existing owner ledger;
- any Citlali, TolAPT, TolProj, TolTECA, `toltec_beammap`, or other repository
  implementation, executable contract, current interface, test, generated
  product, source-specific explanation, local analysis, or status document;
- the historical SCI-BEAM audit inventory or any incoming handoff, finding,
  repair, re-audit, numerical execution, A/B result, Unity evidence,
  validation, conformity, integration, or production-status material;
- active ALIGN/AST material, tracked prior catalogs, historical APTs, full CAL
  or MAP packages, or analogue-instrument methodology papers; or
- any unlisted local file, repository, web source, or model-memory substitute.

If the allowed packet is insufficient, the author must return one precise
scientific question to the manager. It may not search for an answer.

## Author Deliverables

The author writes only within this package's `src/`, `pdf/`, and new versions
of `CROSSWALK.md` and `AUTHOR_DRAFT_DECISIONS.md`. It must not edit the approved
Scope Brief, packet manifest, sanitized references, prior-work record, internal
dossier, decision log, owner ledger, package README, program index, or living
project status.

It must produce:

- shared canonical LaTeX modules for notation, definitions, equations,
  assumptions, requirements, and edge cases;
- a scientist-facing *Scientific Rationale and Contract* with a compact
  input/output/equation/source/status opening and a physical-model-first main
  narrative ordinarily limited to eight to twelve pages before appendices;
- an engineering-facing *Engineering Conformance Specification* expressing
  the same shared authority without implementation-specific mappings or
  independent science;
- stable sequential `SCI-BEAM-REQ-NNN` requirements and stable falsifiable
  prediction identifiers;
- a complete crosswalk from every requirement and prediction to its shared
  source, scientific rationale location, engineering location, owner decision,
  and dependency;
- `AUTHOR_DRAFT_DECISIONS.md` containing every new owner question, author
  choice, scientific inconsistency, unavailable claim, and consequence;
- canonical PDFs
  `SCI-BEAM-SCIENTIFIC-RATIONALE-v0.1.pdf` and
  `SCI-BEAM-ENGINEERING-CONFORMANCE-v0.1.pdf`, with contract version `v0.1`
  and initial document revision `r0.1` kept distinct;
- clean LaTeX compilation, mechanical coverage checks, Poppler rendering, and
  page-by-page visual inspection of both final PDFs; and
- explicit separation of algebraic contract correctness, implementation
  conformance, representation/response fidelity, observational performance,
  and production readiness.

The author must derive genuine BEAM science rather than infer current behavior.
It must not invent production thresholds, promote calibration candidates,
claim sensitivity authority, or close upstream dependencies. A compiling draft
is not automatically accepted or frozen.
