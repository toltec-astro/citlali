# SCI-JINC v0.1 — Proposed Sanitized Author Packet Manifest

Status: ODQ-101/102B/103/104/105/106/107/109 exact-byte Stage A successor candidate; predecessor
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
| 1 | ODQ-101/102B/103/104/105/106/107/109 successor Scope Brief | `SCOPE_BRIEF.md` | `ae7878d5452df0261e910c80dd6dad0d229a1629e83c7fb9fb85a3122e57aedd` |
| 2a | Frozen-core supersession cover | `AUTHOR_SUPERSESSION_COVER.md` | `2718af09e03ead6a5aab2c6a2024ef4d84b9783604a5e4b4ddd16a1425bfe906` |
| 2b | Frozen signed-estimator core, readable only with 2a | `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex` | `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24` |
| 3a | LMT-method reference cover | `AUTHOR_LMT_JINC_REFERENCE_COVER.md` | `9b32095fc7e1773e13e70b4c21d4f402b0c7376aff0b776d41fa9f5a263b7c4f` |
| 3b | Page-exact Schloerb method excerpt, original pages 15--19, readable only with 3a | `references/LMT_JINC_OTF_MAPMAKING_MEMO/Schloerb_JINC_memo_v1.1_METHOD_EXCERPT_pp15-19.pdf` | `a065843b4b83c21aabb25233c588817e998773a5d6a7bd389874eab50c9a88e9` |
| 4 | Recovered decisions and ownership | `AUTHOR_DECISIONS_AND_OWNERSHIP.md` | `2a006ba6e3490fb825c6e2cb2c76bedd89f99fb7190c97b97b5bcc97003e4ad8` |
| 5 | Conventions and ownership | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `b1d04c38b617a41fd8692f41a379c95a35e2713268161a0e492588c691a6a2bb` |
| 6 | Generic analytic identity and TolTEC numerical-unavailability semantics | `ANALYTIC_JINC_IDENTITY.md` | `537a8c18fe5cb2b36094b5b46aabea8d5c3a756b5cd13d8e37769a17b8c48da6` |
| 7 | PTC-to-JINC r0.3 successor boundary | `SCI-PTC_TO_SCI-JINC_BOUNDARY.md` | `f9d9d28825d7d2ff2fa94d91124e60d916217b6574d9e83d6ceae924a9d5d313` |
| 8 | AST-to-JINC r0.2 successor boundary | `SCI-AST_TO_SCI-JINC_BOUNDARY.md` | `04521b5953b33aa2e092f752542a8d054a617bb28b7e8a95cdc8ee974ecbeede` |
| 9 | JINC map-contribution admission profile candidate | `SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md` | `f634621eee1d04fb6f6a56ae2318f4144577e0308ad4bb98e5fea45db14dd8a8` |
| 10 | Collision-free notation and units | `NOTATION_AND_UNITS.md` | `7fb91e87921d562a1d40005872588ef88805a6f4ad720b36e0851b803bdc2ab5` |
| 11 | Geometry decision table | `GEOMETRY_DECISION_TABLE.md` | `d5e285b0aa599239c39440a700d1553f883a370000e730f45a323b21b86136bb` |
| 12 | Fixed grouping and product roles | `GROUPING_AND_PRODUCT_ROLES.md` | `e8b1771ffabfa398d9513d013143aed9f1b8c12fdfe942d2ac053eebacfb4424` |
| 14a | PTC coefficient-registry successor cover | `AUTHOR_PTC_COEFFICIENT_REGISTRY_COVER.md` | `941671d0f9113c94a15bf2de6b69bd9b21a528b41d745b6bbdb936e8e8d8646f` |
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
2. `SCI-JINC-ODQ-110`: the outside-center but overlapping-square edge rule;
   and
3. `SCI-JINC-STAGE-A-Q002`: approval of every exact successor object and
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

`SCI-JINC-ODQ-107` is resolved: every produced bundle contains exactly
required `N_p`, `C_p`, `Q_p`, derived `m_p` with local support/validity, and
`jinc_coefficient_squared_time`. Whole-product failure suppresses the bundle;
local invalid pixels remain ordinary content. No generic availability,
optional/conditional-role, detailed-cause, diagnostic or provenance framework
is authorized. ODQ-108 response/covariance products and every other role are
deferred pending a concrete scientific use. The recovered response/covariance
table is therefore removed from allowed inputs.

`SCI-JINC-ODQ-109` is resolved: numerical error from finite arithmetic,
accumulation/reduction order, function evaluation, phase quantization and
cache/index realization must be negligible compared with the approximately
`10^-3` relative fidelity relevant to the instrument. Finite-state,
`Q_p>0`, `C_p!=0`, exact-cancellation, finite-negative normalization, common-
scale invariance and dimensionless `rho_p` conditioning semantics remain.
No prescribed summation algorithm, contributor-count/machine-epsilon formula,
universal `rho_p` cutoff, exact adequate tie/bin/cache choice, bitwise
reproducibility or stronger precision is a scientific requirement. Adequate
realization and test design belong to the future Engineering Conformance
Specification and do not establish achieved fidelity here.

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
  [`SCIENTIFIC_OWNER_ODQ_107_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_107_DECISION_2026-08-28.md),
  [`SCIENTIFIC_OWNER_ODQ_109_DECISION_2026-08-28.md`](SCIENTIFIC_OWNER_ODQ_109_DECISION_2026-08-28.md),
  [`RESPONSE_AND_COVARIANCE_FAMILIES.md`](RESPONSE_AND_COVARIANCE_FAMILIES.md),
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
