# SCI-JINC v0.1 — Proposed Author Packet Manifest

Status: content-bound Stage A candidate; awaiting scientific-owner approval

Scientific owner: Grant Wilson

Prepared: `2026-08-28`

Starting authority:
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`

No Stage B author is commissioned by this manifest. The packet becomes usable
only after the scientific owner approves these exact bytes or an explicitly
versioned successor manifest.

## Proposed Allowed Inputs

A future fresh implementation-blind scientific author may open this manifest
and only these three logical packet items:

1. [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md)
2. the pair consisting of
   [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md) and the exact
   frozen independent core
   `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex`
3. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)

| Logical item | Exact source | Content SHA-256 |
| --- | --- | --- |
| 1 — proposed Scope Brief | `SCOPE_BRIEF.md` | `bf8b9085e0555cdc8effd7742c24309daea486a10300ea22f61806454bb6846e` |
| 2a — proposed supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `e01fc243a9e9791e653a52a0d2edcb20c3994c82f50e10f3df2d480fe49018be` |
| 2b — frozen independent core | exact Git object named above | `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24` |
| 3 — proposed conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `40ace565d78d34b6da8e268e71f3d53838eed8e38dccd99f59daa919b7a8b676` |

These hashes bind the exact proposed bytes. Any content change requires
recomputed hashes and renewed owner review; it cannot drift silently.

## Prohibited Inputs

The future author must not open:

- [`PRIOR_WORK.md`](PRIOR_WORK.md),
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md),
  [`DECISION_LOG.md`](DECISION_LOG.md),
  [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md),
  or the package README;
- the raw D003 owner-decision files, third-successor owner-acceptance record,
  audit ledger, cross-audit handoffs, or any other historical SCI-MAP-002
  coordination material;
- any Citlali implementation, executable configuration/product contract,
  interface, class/function path, test, generated product, source trace,
  source-specific explanation, or current status document;
- any audit, finding, repair, re-audit, numerical execution, reduction, Unity,
  comparison, validation, achieved-performance, conformity, integration,
  readiness, or production-status material;
- the March JINC memo-alignment note, its unrecovered underlying memo, the
  internal draft noise memo, or historical parameter-tuning recommendations;
- the full frozen SCI-ALIGN, SCI-AST, SCI-RTC, SCI-CAL, SCI-PTC, SCI-VAL, or
  SCI-MAP packages; the ordinary PTC-to-MAP boundary; or later NOI/FLT/BEAM/
  SRC/MODE/FRUIT material; or
- any unlisted local file, repository, web source, external paper, or model-
  memory substitute.

If the allowed packet is insufficient, the author must return one precise
scientific question to the manager. It may not search for an answer.

## Future Author Deliverables After Approval Only

Only after explicit scientific-owner approval and a separate Stage B launch,
the author may write within this package's `src/`, `pdf/`, `CROSSWALK.md`, and
new author-draft decision artifacts. It must not edit the approved Stage A
controls.

The future deliverables are:

- shared canonical LaTeX modules for notation, definitions, equations,
  assumptions, requirements, and edge cases;
- a scientist-facing *Scientific Rationale and Contract* with a compact
  input/output/equation/source/status opening and a physical-model-first main
  narrative ordinarily limited to eight to twelve pages before appendices;
- an engineering-facing *Engineering Conformance Specification* expressing
  the same shared authority without implementation-specific mappings or
  independent science;
- stable sequential `SCI-JINC-REQ-NNN` requirements and falsifiable prediction
  identifiers with a complete crosswalk;
- an author-draft decision record returning every new owner question,
  inconsistency, unavailable claim, and consequence without resolving it from
  excluded context;
- canonical PDFs `SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` and
  `SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf`, keeping contract version `v0.1`
  distinct from document revision `r0.1`;
- clean compilation, mechanical identifier/coverage checks, Poppler rendering,
  and page-by-page visual inspection; and
- explicit separation of algebraic correctness, implementation conformity,
  representation/response fidelity, numerical and observational validation,
  achieved performance, readiness, and production authorization.

The author must reuse and reconcile the recovered signed-estimator science.
It must not repeat the derivation, infer missing upstream authority, invent an
analytic memo, import ordinary MAP semantics, choose numerical parameters, or
claim validation. A compiling draft would remain a draft until subsequent
manager and scientific-owner review.

## Owner Approval Gate

Approval must explicitly cover:

1. the exact four hashes in the table;
2. the observation-level scientific boundary and exclusions;
3. the recovered decisions and supersessions;
4. the open-question disposition that determines the author task; and
5. the information firewall.

Until then, `SCI-JINC-STAGE-A-Q001` remains open and Stage B is prohibited.
