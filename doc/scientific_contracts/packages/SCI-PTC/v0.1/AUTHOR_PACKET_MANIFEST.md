# SCI-PTC v0.1 — Author Packet Manifest

Status: scientific-owner approved content-bound Stage B author packet

Scientific owner: Grant Wilson

Prepared date: `2026-08-20`

## Allowed Inputs

The implementation-blind scientific author may open this manifest and only
these five owner-approved packet items:

1. [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md)
2. [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md)
3. the exact frozen `SCI-PTC-001_INDEPENDENT_CORE.tex` named below
4. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)
5. [`AUTHOR_METHOD_REFERENCE_BOUNDARY.md`](AUTHOR_METHOD_REFERENCE_BOUNDARY.md)

| Logical item | Exact source | Content SHA-256 |
| --- | --- | --- |
| 1 — approved Scope Brief | `SCOPE_BRIEF.md` | `8aa05920589b67cb7634003f466161769101e3013cf82573260e94b257532bed` |
| 2 — binding supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `2a13d3984c2334ccd1886021d2d869bb71363abd3a06bb7f9fbf536614d9ee3e` |
| 3 — reusable independent core | `01ee247461d6c19bc4db81ccac4fec21af162c88:doc/audits/packages/SCI-PTC-001_INDEPENDENT_CORE.tex` | `82c0835f51ea9b1fa8a37489f289be89a8018a0b2700e84b1e25c2e4d2a013c2` |
| 4 — conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `568b35ff3da16c8ed6902d3bb0d845e01eec38e5374c6e89e75823f1f8ecabe6` |
| 5 — bounded method-reference context | `AUTHOR_METHOD_REFERENCE_BOUNDARY.md` | `d5d33180c9e40958237916ec6dd98ba655d161bc984a3b694197a1a90d78be61` |

Items 2 and 3 are inseparable. The author may not read or use the frozen core
without applying every limitation and successor rule in the cover. Any packet
content change requires owner review and a new manifest rather than silent
drift.

## Prohibited Inputs

The author must not open:

- [`PRIOR_WORK.md`](PRIOR_WORK.md),
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md),
  [`DECISION_LOG.md`](DECISION_LOG.md),
  [`SCOPE_REVIEW_R0.1.md`](SCOPE_REVIEW_R0.1.md),
  [`SCOPE_REVIEW_R0.2.md`](SCOPE_REVIEW_R0.2.md),
  [`CROSS_PACKAGE_FOLLOWUP.md`](CROSS_PACKAGE_FOLLOWUP.md), the three raw scope
  reviews, the program registry, or other package
  files not listed above;
- any Citlali or adjacent-repository implementation, executable contract,
  interface, configuration, source-specific explanation, test, generated
  product, or status document;
- the historical PTC audit report, evidence, finding, proposal, raw owner
  amendment, handoff, repair, re-audit, reduction, numerical execution, Unity,
  validation, conformity, integration, or production material;
- the raw model-protected-notch note, optional transfer-characterization
  execution plan, full SCI-RTC/SCI-CAL/SCI-MAP package, or active adjacent
  scientific work; or
- the full papers cited in the bounded method-reference record; or
- any unlisted local file, repository, external source, web source, or model-
  memory substitute.

If the allowed packet is insufficient, the author must return one precise
scientific question to the manager. It may not search for an answer.

## Author Deliverables

Once authorized, the Stage B author writes only within this package's `src/`,
`pdf/`, and new versions of `CROSSWALK.md` and
`AUTHOR_DRAFT_DECISIONS.md`. It must not edit the approved Scope Brief, packet
manifest, supersession cover, sanitized conventions, prior-work record,
internal dossier, decision log, package README, program index, or living
project status.

It must produce:

- one shared canonical LaTeX core for notation, definitions, equations,
  assumptions, requirements, and edge cases;
- a scientist-facing *Scientific Rationale and Contract* with a compact
  input/output/equation/source/status opening and a physical-model-first main
  narrative ordinarily limited to eight to twelve pages before appendices,
  organized through estimand/null space, centering/grouping/gauge, source
  protection, estimator families, conjunctive least-aggressive mode selection,
  coefficient taxonomy, response,
  covariance/support, and iteration/recurrence/products;
- an engineering-facing *Engineering Conformance Specification* expressing
  the exact same shared authority without implementation mappings or
  independent science;
- stable sequential `SCI-PTC-REQ-NNN` requirements and stable falsifiable
  prediction identifiers;
- a complete crosswalk from every requirement and prediction to shared source,
  rationale location, engineering location, owner decision, and dependency;
- `AUTHOR_DRAFT_DECISIONS.md` containing every new owner question, author
  choice, scientific inconsistency, unavailable claim, and consequence;
- canonical PDFs
  `SCI-PTC-SCIENTIFIC-RATIONALE-v0.1.pdf` and
  `SCI-PTC-ENGINEERING-CONFORMANCE-v0.1.pdf`, with contract version `v0.1`
  distinct from initial document revision `r0.1`;
- clean LaTeX compilation, mechanical identifier/coverage checks, Poppler
  rendering, and page-by-page visual inspection of both PDFs; and
- explicit separation of algebraic contract correctness, implementation
  conformity, response/representation fidelity, observational performance,
  and production readiness.

The author must consolidate the reusable core rather than repeat it. It must
not redesign mature PTC numerics, infer defaults or thresholds, repair CAL,
authorize raw Beammap PTC, promote scalar coefficients to precision, claim
complete covariance or response without authority, or close upstream
dependencies. It must treat `PTC-OWNER-Q001` according to the owner's launch
disposition: base v0.1 `r` analysis is diagnostic-only, inert or advisory, and
may not control calibrated `x` or provide cross-channel subtraction. A
compiling draft is not automatically accepted or frozen.

## Authorization State

Grant Wilson approved the Scope Brief, `PTC-SCOPE-D001--D017`, and this exact
packet on `2026-08-19`, launching implementation-blind Stage B authorship.
`PTC-OWNER-Q001` is resolved diagnostic-only for the first implementation/base
v0.1. On `2026-08-20`, the owner directed the r0.4 support-ownership,
conservative composition, nonrestoring-centering, and traceability amendments;
items 1 and 4 were re-bound above rather than changed silently. Approval does
not establish implementation conformity, validation, scientific freeze, or
production readiness.

The subsequent owner statement recorded in
[`SCIENTIFIC_OWNER_FREEZE_R0.4.md`](SCIENTIFIC_OWNER_FREEZE_R0.4.md) freezes the
completed v0.1/r0.4 scientific authority. That later status decision does not
retroactively broaden the author packet or establish implementation
conformity, validation, or production readiness.
