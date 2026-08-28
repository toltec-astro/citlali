# SCI-JINC v0.1 — Proposed Sanitized Author Packet Manifest

Status: ODQ-101/102B/103/104/105/106 exact-byte Stage A successor candidate; predecessor
manifest owner-approved; successor bytes not owner-approved; not launchable

Scientific owner: Grant Wilson

Prepared: `2026-08-28`

Starting authority:
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`

Approved predecessor packet commit:
`6639bff3d94b92ace8faf3e407ccaefd5a38ea1f`

Controlled ODQ-101 successor source:
`54475956f6aefb839d43b2f0fb019a142cb64310:doc/scientific_contracts/packages/SCI-MAP/v0.1/POST_FREEZE_SCIENTIFIC_OWNER_DECISIONS_2026-08-28.md`,
SHA-256
`4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c`

No Stage B author is commissioned by this manifest. Even after byte approval,
dispatch remains blocked until the scientific owner closes or explicitly
types the hard blockers below and authorizes a successor VAL registry binding.

## Exact Proposed Allowed Inputs

A future fresh implementation-blind scientific author may open this manifest
and only the exact objects in the table. Covers and their paired sources are
single logical inputs; neither source may be read without its cover.

| # | Logical input | Exact source | Content SHA-256 |
| --- | --- | --- | --- |
| 1 | ODQ-101/102B/103/104/105/106 successor Scope Brief | `SCOPE_BRIEF.md` | `9134f92518f2d0d906de0ae98abae753f8d98076d3f4f1236e51654e89d57275` |
| 2a | Frozen-core supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `f504ee3ea3c25f50b1013119d19623c36a6798a823ccbdf8b0a7c5aa4474fe5b` |
| 2b | Frozen signed-estimator core, readable only with 2a | `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex` | `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24` |
| 3a | LMT-method reference cover | `AUTHOR_LMT_JINC_REFERENCE_COVER.md` | `bdbd4e37c86c9b77ccedd88285b2ec77e7cc1f1bf1e3425ebd2f7a0d5a6d4abd` |
| 3b | Page-exact Schloerb method excerpt, original pages 15--19, readable only with 3a | `references/LMT_JINC_OTF_MAPMAKING_MEMO/Schloerb_JINC_memo_v1.1_METHOD_EXCERPT_pp15-19.pdf` | `a065843b4b83c21aabb25233c588817e998773a5d6a7bd389874eab50c9a88e9` |
| 4 | Recovered decisions and ownership | `AUTHOR_DECISIONS_AND_OWNERSHIP.md` | `3e9daf3c170b6d7b8aebe7f85af0855dbd1d49d8c5d9bf6ca127249c2772bdca` |
| 5 | Conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `db6fb91371e89fff7a2a4fd30803c9c9ed9f8adb1de1b01391a4607e7ec1df2f` |
| 6 | Generic analytic identity and TolTEC numerical-unavailability semantics | `ANALYTIC_JINC_IDENTITY.md` | `f35116f3bb525bb42941c58d51b0679aedc4feab71c29945b41e3f84d13239ef` |
| 7 | PTC-to-JINC r0.3 successor boundary | `SCI-PTC_TO_SCI-JINC_BOUNDARY.md` | `e8b70a2169fc990ede8beb9072f899b0ea8d295edeaa05a5fddf12cff8bf067a` |
| 8 | AST-to-JINC r0.2 successor boundary | `SCI-AST_TO_SCI-JINC_BOUNDARY.md` | `505b30f5f3ead2417289c27d002cd4e3c2d28b01ea99f802d4be7341ab321576` |
| 9 | JINC map-contribution admission profile candidate | `SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md` | `63b841f06fef520cec7ded35cf6f8e9f530a1eb195caaf0d50011b2addc52481` |
| 10 | Collision-free notation and units | `NOTATION_AND_UNITS.md` | `62eb12c90da38eb4518fe964231e9f44cd70803c6f2adf97935c843bafe2a831` |
| 11 | Geometry decision table | `GEOMETRY_DECISION_TABLE.md` | `b80c3de790bb8fda85a46a0b6cb2d441c2c6c9dcb2482a9119b906338cee5a21` |
| 12 | Grouping and product roles | `GROUPING_AND_PRODUCT_ROLES.md` | `718f2c79f0039d3340bca2cc16feb6965dab38928452d698e04d6e9d8c2b9f24` |
| 13 | Response and covariance families | `RESPONSE_AND_COVARIANCE_FAMILIES.md` | `6540a6497f520adad4b478821364fb42e0c64cd6fc3de361e4e666a911972dc1` |
| 14a | PTC coefficient-registry successor cover | `AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md` | `199c991d989928671ee19df3dab746feae3fe9593cb83cc07be0544c98b7a909` |
| 14b | Exact post-freeze PTC registry predecessor, readable only with 14a and only in the admitted sections | `54475956f6aefb839d43b2f0fb019a142cb64310:doc/scientific_contracts/packages/SCI-MAP/v0.1/POST_FREEZE_SCIENTIFIC_OWNER_DECISIONS_2026-08-28.md` | `4d2b857b7ec9efe489fe065f464df4ecd23b57a4c1320cda6a10a56592825d1c` |

These digests bind the exact proposed bytes. This manifest is the control
document and is intentionally not self-hashed. Any allowed-object change
requires all affected hashes to be recomputed and the complete successor
manifest to receive a new exact-byte owner review.

## Hard Blockers Before Stage B Dispatch

Exact-byte approval of this candidate is necessary but not sufficient. No
author may be commissioned until the owner resolves or explicitly types:

1. the versioned VAL Registry successor binding for
   `SCI-JINC:jinc_map_contribution@1`;
2. `SCI-JINC-ODQ-109`: exact numerical tie/bin/cache/summation-error policy;
3. `SCI-JINC-ODQ-110`: the outside-center but overlapping-square edge rule;
   and
4. `SCI-JINC-STAGE-A-Q002`: approval of every exact successor object and
   digest in this manifest plus its information firewall.

`SCI-JINC-ODQ-101` is resolved for registry ownership, consumer permission,
selection lifecycle and fail-closed behavior. `SCI-JINC-ODQ-102B` is resolved
for generic parameter semantics and a typed no-numerical-route state; it
authorizes no TolTEC numerical values. At least one exact
JINC-permitted family must still be registered, selected and realized before a
numerical route exists; that source-availability prerequisite is typed
unavailable and is not silently filled by the Stage B author.

`SCI-JINC-ODQ-103` is resolved for exact scientific sample-coordinate
association, AST/JINC ownership, the single JINC-owned
`SCI-JINC:jinc_map_contribution@1` profile, sample-admission/support
separation, coupled-accumulator identity and cause policy. It authorizes no
implementation-specific join architecture, ordinary MAP validity inheritance,
or per-contribution provenance machinery.

`SCI-JINC-ODQ-104` is resolved: `jinc_coefficient_squared_time` is the sole
base-v0.1 time-support product. A distinct physical-exposure product is
deferred until an identified scientific use separately authorizes its exact
original-occurrence lineage and semantics; no exposure-based claim follows.

`SCI-JINC-ODQ-105` is resolved: base v0.1 defines one complete observation
bundle, permits same-observation incremental accumulation under one exact
realization/bundle identity, and authorizes no cross-observation combination.
Any future JINC coadd requires a separately authorized boundary over complete
observation bundles; no ordinary MAP or inferred accumulator/map rule follows.

`SCI-JINC-ODQ-106` is resolved: one observation may produce zero through three
independent bundles, with at most one per stable array admitted/requested under
the exact JINC realization and destination geometry. Missing, unavailable or
unrequested arrays create no placeholder and do not invalidate another
bundle. Different array/destination contributions never merge; no additional
per-contribution provenance is required.

The separate downstream derivation of optimum parameters for TolTEC's three
bands is explicitly deferred. It is not a prerequisite to describe a typed
no-numerical-route v0.1, but no per-band numerical JINC route, default,
recommendation or optimum may be authored without separate owner authorization
and evidence.

## Prohibited Inputs

The future author must not open:

- [`PRIOR_WORK.md`](PRIOR_WORK.md),
  [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md),
  [`DECISION_LOG.md`](DECISION_LOG.md),
  [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md),
  [`SCIENTIFIC_OWNER_ODQ_102B_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_102B_DECISION_2026-08-28.md),
  [`SCIENTIFIC_OWNER_ODQ_103_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_103_DECISION_2026-08-28.md),
  [`SCIENTIFIC_OWNER_ODQ_104_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_104_DECISION_2026-08-28.md),
  [`SCIENTIFIC_OWNER_ODQ_105_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_105_DECISION_2026-08-28.md),
  [`SCIENTIFIC_OWNER_ODQ_106_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_106_DECISION_2026-08-28.md),
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
- the MAP-local sections of the post-freeze decision record admitted as item
  14b; only the sections permitted by item 14a may be used;
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

Until the owner records renewed approval of this successor manifest and the
remaining dispatch blockers are closed, `SCI-JINC-STAGE-A-Q002` remains open
and Stage B is prohibited.
