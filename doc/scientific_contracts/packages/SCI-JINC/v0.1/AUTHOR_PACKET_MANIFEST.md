# SCI-JINC v0.1 — Proposed Sanitized Author Packet Manifest

Status: exact-byte Stage A candidate; not owner-approved; not launchable

Scientific owner: Grant Wilson

Prepared: `2026-08-28`

Starting authority:
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`

No Stage B author is commissioned by this manifest. Even after byte approval,
dispatch remains blocked until the scientific owner closes or explicitly
types the hard blockers below and authorizes a successor VAL registry binding.

## Exact Proposed Allowed Inputs

A future fresh implementation-blind scientific author may open this manifest
and only the exact objects in the table. Covers and their paired sources are
single logical inputs; neither source may be read without its cover.

| # | Logical input | Exact source | Content SHA-256 |
| --- | --- | --- | --- |
| 1 | Repaired Scope Brief | `SCOPE_BRIEF.md` | `71ba6e07d18151685e4f2fc34cdf1c8821307b816beee44314644897c834be3f` |
| 2a | Frozen-core supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `25c9127fbfbb82b9d44278f850effb61b1e08875e236a64523a3495a0e85a346` |
| 2b | Frozen signed-estimator core, readable only with 2a | `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex` | `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24` |
| 3a | LMT-method reference cover | `AUTHOR_LMT_JINC_REFERENCE_COVER.md` | `f0507b18edb96034ad377a2a4f6bbed96b36fda845ffd87bc562406fd5262e4b` |
| 3b | Page-exact Schloerb method excerpt, original pages 15--19, readable only with 3a | `references/LMT_JINC_OTF_MAPMAKING_MEMO/Schloerb_JINC_memo_v1.1_METHOD_EXCERPT_pp15-19.pdf` | `a065843b4b83c21aabb25233c588817e998773a5d6a7bd389874eab50c9a88e9` |
| 4 | Recovered decisions and ownership | `AUTHOR_DECISIONS_AND_OWNERSHIP.md` | `3e9daf3c170b6d7b8aebe7f85af0855dbd1d49d8c5d9bf6ca127249c2772bdca` |
| 5 | Conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `d330da218a30c6a9267f592c217b8aaf8a2618245afac87366bd47fe636b60f0` |
| 6 | Generic analytic identity and TolTEC gap | `ANALYTIC_JINC_IDENTITY.md` | `e18c611c6e7ebbfcaaed2c8a41f070fcc1f784556bd1bc605517f46df86ea694` |
| 7 | PTC-to-JINC boundary candidate | `SCI-PTC_TO_SCI-JINC_BOUNDARY.md` | `13f3122867f1194c7f2669e9f893b5ba66651774e126f64b3b765da35dc276b5` |
| 8 | AST-to-JINC boundary candidate | `SCI-AST_TO_SCI-JINC_BOUNDARY.md` | `32ce6acab7c7c9c5efc3a72d304cddbdae4f987c6b03f6e2118fc4187c6a7b05` |
| 9 | JINC upstream-admission profile candidate | `SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md` | `d50014f86325ac91a2741460b452a1bdb7e52fbc8a3314dbdcc5a1c809da435b` |
| 10 | Collision-free notation and units | `NOTATION_AND_UNITS.md` | `8ca63d283908ea5cc34c082cefb4293c052cc3b6532c750c36681eb4ab06ddc8` |
| 11 | Geometry decision table | `GEOMETRY_DECISION_TABLE.md` | `90abbb5b7c6cdd9a2d75392b2879fdb731e5c7e3102f7e019af29ad9470f459e` |
| 12 | Grouping and product roles | `GROUPING_AND_PRODUCT_ROLES.md` | `bb55dd843d533137c4c8a24974b98affad07ea44f4a62ffb0cdbadc2525eeced` |
| 13 | Response and covariance families | `RESPONSE_AND_COVARIANCE_FAMILIES.md` | `6540a6497f520adad4b478821364fb42e0c64cd6fc3de361e4e666a911972dc1` |

These digests bind the exact proposed bytes. This manifest is the control
document and is intentionally not self-hashed. Any allowed-object change
requires all affected hashes to be recomputed and the complete successor
manifest to receive a new exact-byte owner review.

## Hard Blockers Before Stage B Dispatch

Exact-byte approval of this candidate is necessary but not sufficient. No
author may be commissioned until the owner resolves or explicitly types:

1. `SCI-JINC-ODQ-101`: exact PTC signal and JINC-facing coefficient family,
   units, normalization, support, QC, generation and covariance meaning;
2. `SCI-JINC-ODQ-102B`: exact effective TolTEC radial scale and parameter
   source/value state for `a1100`, `a1400`, and `a2000`;
3. the versioned VAL Registry successor binding for
   `SCI-JINC:upstream_admission@1`;
4. `SCI-JINC-ODQ-109`: exact numerical tie/bin/cache/summation-error policy;
5. `SCI-JINC-ODQ-110`: the outside-center but overlapping-square edge rule;
   and
6. `SCI-JINC-STAGE-A-Q001`: approval of every exact object and digest in this
   manifest plus its information firewall.

The separate downstream derivation of optimum parameters for TolTEC's three
bands is not a prerequisite to describe a typed no-numerical-route v0.1, but
no per-band numerical JINC route, default, recommendation or optimum may be
authored without separate owner authorization and evidence.

## Prohibited Inputs

The future author must not open:

- [`PRIOR_WORK.md`](PRIOR_WORK.md),
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md),
  [`DECISION_LOG.md`](DECISION_LOG.md),
  [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md),
  [`STAGE_A_CHANGE_LOG.md`](STAGE_A_CHANGE_LOG.md), the package README, the
  reference intake record, or this package's raw owner-feedback material;
- the full `Schloerb_JINC_memo_v1.1.pdf`; its 3-mm receiver context, FCRAO
  values, 86-GHz simulations, numerical examples, tuning, optimization and
  performance implications; the March JINC alignment note; or the unnamed
  historical memo behind that note;
- raw D003 owner-decision files, the third-successor acceptance record, audit
  ledger, cross-audit handoffs, or other historical SCI-MAP-002 coordination
  material;
- any Citlali implementation, executable configuration/product contract,
  interface, class/function path, test, generated product, source trace,
  source-specific explanation or current status document;
- any audit, finding, repair, re-audit, numerical execution, reduction, Unity,
  comparison, validation, achieved-performance, conformity, integration,
  readiness or production-status material;
- the internal draft noise memo or historical parameter recommendations;
- full frozen SCI-ALIGN, SCI-AST, SCI-RTC, SCI-CAL, SCI-PTC, SCI-VAL or SCI-MAP
  packages; the ordinary PTC-to-MAP boundary; or later NOI/FLT/BEAM/SRC/MODE/
  FRUIT material; and
- any unlisted local file, repository, web source, external paper or model-
  memory substitute.

If the allowed packet is insufficient, the author must return one precise
scientific question to the manager. It may not search for an answer.

## Future Author Deliverables After Separate Approval And Launch Only

Only after the blocker gate is closed and a separate Stage B launch, a fresh
implementation-blind author may write within this package's `src/`, `pdf/`,
`CROSSWALK.md`, and new author-draft decision artifacts. It must not edit the
approved Stage A controls.

The future deliverables are:

- shared canonical LaTeX modules for notation, definitions, equations,
  assumptions, requirements and edge cases;
- a scientist-facing *Scientific Rationale and Contract* with a compact
  input/output/equation/source/status opening and a physical-model-first main
  narrative ordinarily limited to eight to twelve pages before appendices;
- an engineering-facing *Engineering Conformance Specification* expressing
  the same shared authority without implementation mappings or independent
  science;
- stable sequential `SCI-JINC-REQ-NNN` requirements and falsifiable prediction
  identifiers with a complete crosswalk;
- an author-draft decision record returning every new owner question,
  inconsistency, unavailable claim and consequence without resolving it from
  excluded context;
- canonical PDFs `SCI-JINC-SCIENTIFIC-RATIONALE-v0.1.pdf` and
  `SCI-JINC-ENGINEERING-CONFORMANCE-v0.1.pdf`, keeping contract version `v0.1`
  distinct from document revision `r0.1`;
- clean compilation, mechanical identifier/coverage checks, Poppler rendering
  and page-by-page visual inspection; and
- explicit separation of algebraic correctness, implementation conformity,
  representation/response fidelity, numerical and observational validation,
  achieved performance, readiness and production authorization.

The author must reuse and reconcile the recovered signed-estimator science. It
must not repeat the derivation, infer missing upstream authority, import
ordinary MAP semantics, select TolTEC numerical parameters, or claim
validation. A compiling draft remains a draft until later manager and
scientific-owner review.

## Exact-Byte Owner Gate

Approval must explicitly cover:

1. every exact source and content digest in the table;
2. the observation-level scientific boundary and exclusions;
3. the recovered decisions and supersessions;
4. the hard-blocker disposition that determines the eventual author task;
5. the Schloerb excerpt's generic-method-only status and TolTEC exclusions;
   and
6. the complete information firewall.

Until the owner records that approval and the dispatch blockers are closed,
`SCI-JINC-STAGE-A-Q001` remains open and Stage B is prohibited.
