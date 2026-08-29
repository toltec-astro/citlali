# SCI-NOI v0.1 — Proposed Implementation-Blind Author Packet

Status: proposed and content-bound for scientific-owner review; not approved

The future implementation-blind author may open this manifest and only these
four logical packet items:

1. [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md)
2. the pair consisting of
   [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md) and both
   exact recovered mathematical cores
3. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)
4. [`AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md`](AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md)

| Logical item | Exact source | Content SHA-256 |
| --- | --- | --- |
| 1 — proposed Scope Brief | `SCOPE_BRIEF.md` | `9be5b3d1945592bf8515a6b473a8866fb0922404136e844d1c31c0b12039b2b4` |
| 2a — proposed supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `eb86b58f01aef08868234f55c90a03d56dd44fe8785c26cf226d3bb5747f1e02` |
| 2b — NOI-001 R3 mathematical core | `5a027c94ef9fc9c4a6e6cadc84af1c8a550d3508:doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE_R3.tex` | `27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da` |
| 2c — NOI-002 mathematical core | `4f1fec36f7802f3b5e8ac067377679946930983c:doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex` | `36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d` |
| 3 — proposed conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `8878c2cea09a732ec65b97630608a6d420e7995e3077df4d3303637ed830f88d` |
| 4 — proposed taxonomy | `AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md` | `4fb96173e7d9f649682019c7ebacd4fad6029c3c434c99e9cdd802e74327f478` |

Any content change requires recomputed hashes and renewed owner review.

## Prohibited Inputs

The future author must not open:

- [`README.md`](README.md), [`PRIOR_WORK.md`](PRIOR_WORK.md),
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md),
  [`OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md`](OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md),
  [`DECISION_LOG.md`](DECISION_LOG.md),
  [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md),
  or the raw owner launch record;
- historical NOI owner/coordinator briefs, audits, findings, repairs,
  re-audits, integration records, cross-audit handoffs, and evidence results;
- current or historical Citlali implementation, schemas, product contracts,
  configuration authority, tests, generated products, accepted runs,
  validation, reductions, Unity, achieved performance, or status records;
- `doc/citlali_noise_estimation_plan.tex`, the Convolve audit/contract,
  historical counts/defaults, or implementation-specific vocabulary;
- the full frozen MAP, JINC, RTC, CAL, PTC, AST, VAL, BEAM, or other package;
- FLT, SRC/MODE, or FRUIT source/audit material; or
- any unlisted local file, repository, web source, external paper, or model-
  memory substitute.

If the allowed packet is insufficient, the author must return one precise
scientific question. It may not search for an answer.

## Future Deliverables After Approval Only

Only after explicit scientific-owner Stage A approval and a separate Stage B
launch may a fresh implementation-blind author write the shared `src/common/`
core, the two views, `CROSSWALK.md`, author decision records, and PDFs.

The future draft must:

- reuse and reconcile the recovered mathematics rather than repeat it;
- maintain the Family G/U/Z typed interfaces;
- use stable sequential `SCI-NOI-REQ-NNN` and prediction identifiers;
- return every unresolved owner question without choosing an implementation
  default;
- make unavailable and limited products explicit;
- keep algebraic, conformity, validation, performance, readiness, and
  production claims separate; and
- compile, pass mechanical checks, render through Poppler, and receive
  page-by-page visual inspection before delivery.

## Owner Approval Gate

Approval must explicitly cover the six content hashes in the table, the
scientific boundary, the supersessions, the owner-question disposition, and
the information firewall. Until then, `SCI-NOI-STAGE-A-Q001` remains open.
