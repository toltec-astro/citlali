# SCI-RTC v0.1 — Proposed Author Packet Manifest

Status: proposed, content-bound packet; scientific-owner approval pending

Scientific owner: Grant Wilson

Prepared: `2026-08-17`

## Proposed Allowed Inputs

A fresh implementation-blind scientific author would be allowed to open only
these three logical packet items:

1. [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md)
2. the pair consisting of
   [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md) and exact
   independent core
   `3319d7424c732c1c9fc300c336e4d428e6f91068:doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex`
3. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)

| Logical item | Exact source | Content SHA-256 |
| --- | --- | --- |
| 1 — draft Scope Brief | `SCOPE_BRIEF.md` | `aac4149f14cbad49f49c3d42af6b76bce5c4e5270aa9a0a7020a0249ec0cbc09` |
| 2a — proposed supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `1923b48baa0256c851b45fc302eee22c32a1e5300ed5ded1860d5e0746f434a9` |
| 2b — independent core | exact Git object named above | `d6cf49d1a5e17754c55cc4f2c8f4b4f5e276755f247496df888581d890be80b7` |
| 3 — proposed conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `c2046fd00e135abcc1830b80c8715f273e5367ac7b9ae25681d3949b14da366f` |

The local-file hashes identify the exact Stage A bytes proposed for approval.
Owner approval would admit those exact bytes; any later content change
requires a new review and manifest rather than silent packet drift.

## Prohibited Inputs

The author must not open:

- [`PRIOR_WORK.md`](PRIOR_WORK.md),
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md), or
  [`DECISION_LOG.md`](DECISION_LOG.md);
- any Citlali implementation, current interface, executable config/product
  contract, test, generated product, or source-specific explanation;
- any RTC/CAL/ALIGN/AST/PTC/MAP/BEAM audit, finding, raw handoff, repair,
  re-audit, execution, validation, Unity, conformity, integration, or
  production-status material other than the exact independent core admitted
  above;
- the raw learned-sampling plan, ADR, implementation, metrics, or successor
  history, whose approved scientific substance is instead consolidated in the
  proposed supersession cover;
- active ALIGN work or any inferred physical timing solution; or
- any unlisted repository, local file, web source, or model-memory substitute.

If the proposed packet is insufficient, the author must return one precise
scientific question. It may not search for an answer.

## Expected Author Deliverables After Approval

The author would write only within this package's `src/`, `pdf/`,
`CROSSWALK.md`, `AUTHOR_DRAFT_DECISIONS.md`, and
`SCIENTIFIC_OWNER_DECISION_LEDGER.md`, producing:

- the six shared canonical LaTeX modules;
- a scientist-facing rationale with an eight-to-twelve-page main narrative;
- an engineering conformance view of the same authority;
- stable `SCI-RTC-REQ-NNN` requirements and falsifiable predictions;
- a complete crosswalk and exact owner-decision register;
- rendered v0.1/r0.1 PDFs after compilation and full visual QA; and
- explicit separation of contract, conformity, representation fidelity,
  observational performance, and production readiness.

The author must reuse the independent derivation rather than repeat it. A
compilable document is not automatically scientifically approved or frozen.
