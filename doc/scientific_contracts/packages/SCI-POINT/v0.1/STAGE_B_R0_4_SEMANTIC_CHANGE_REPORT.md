# SCI-POINT v0.1 r0.3-to-r0.4 Semantic-Change Report

Date: 2026-09-03

Result: **editorial/document-architecture change only; no substantive
scientific change**.

## Controlling Authorities

- Accepted scientific basis:
  `SCI-POINT-v0.1-r0.3-final-targeted-stage-b-owner-directive@2026-09-03`,
  SHA-256
  `9f445480fd42311ebe21f4da772e4e32db400c2da08a119b699f6bd8e13a54d4`.
- Presentation directive:
  `SCI-POINT-v0.1-r0.4-view-separation-directive@2026-09-03`, SHA-256
  `3f2331d4dc2a926ebd840cddb9bd85c9bc7d2a88be28718074313e262c259193`.
- Accepted machine-readable scientific record: `STAGE_B_R0_3_RECORDS.json`,
  SHA-256
  `b80094ec2af23dac256a4ecac488183229c6ea22dc2c24d7660e48bcc8ff57c4`.

## Common-Core Byte Result

The six accepted scientific source files did **not** change by one byte. Their
ordered, separator-free SHA-256 remains:

`c0ca71bd457b8e6d37a425eb3ead76400dba3a5e29c869420807928201cdcdbd`.

The per-file hashes also match r0.3. The new normative-core TeX file is only a
publication wrapper around those exact files; its cover, table of contents,
headers, and reading rule are presentation material.

## Exact Modification Map

| Modification | Classification | Scientific effect |
|---|---|---|
| Add a standalone normative-core wrapper and PDF around the exact six common files. | relocation | None. The accepted normative text is centralized rather than duplicated. |
| Remove the six full common-file imports from both concise view sources. | deduplication | None. Both views bind to the core identity and exact digest. |
| Retain selected rationale explanations, equations, and consolidated tables while removing verbatim complete registers. | editorial compression | None. The complete registers remain in the core. |
| Replace rationale formal appendices with compact REQ/PRED ranges and the controlling core identity/digest. | cross-reference replacement | None. The cited identifiers and source are unchanged. |
| Replace ECS requirement and prediction prose with complete compact evidence and fixture indexes. | cross-reference replacement | None. Each exact ID remains indexed to the normative core. |
| State the eight ECS logical stages once and refer to their letters elsewhere. | deduplication | None. Stage identity, ordering, fields, and terminal rules are preserved. |
| State method-record acceptance fields once and refer to that subsection from dependent obligations. | deduplication | None. All three authority gates and acceptance obligations remain. |
| Use top-level-only contents, natural section flow, and substantive shared final pages. | pagination/typesetting | None. No body/table font, line spacing, or margin was reduced merely to hit a page target. |
| Add r0.4 view, source, build, parity, QA, and delivery records. | cross-reference replacement | None. These establish provenance and presentation parity only. |
| Change the scientific estimand, equation, role, owner, lifecycle, state, gate, response meaning, requirement, prediction, unavailable state, source, digest, or nonclaim. | substantive scientific change | **No such change was made.** |

Editorial compression exposed no scientific contradiction.

## Semantic Parity

| Surface | r0.3 | r0.4 result |
|---|---|---|
| Requirements | `SCI-POINT-REQ-001--038` | Same 38 IDs and exact normative prose in the core; all 38 appear in the ECS evidence index. |
| Predictions | `SCI-POINT-PRED-001--032` | Same 32 IDs and exact normative prose in the core; all 32 appear in the ECS fixture index. |
| Unavailable states | `SCI-POINT-UNAV-001--023` | Same 23 IDs and exact normative prose in the core. |
| Equations | Accepted model, displacement, diagnostic, response, bias, and covariance equations | Exact common equation bytes unchanged; selected equations copied into the rationale retain their meanings. |
| Product roles and claim ceilings | Accepted r0.3 role set | Exact common definitions unchanged; rationale consolidates presentation only. |
| Lifecycle and state tokens | 13 producer tokens; eight logical stages; three terminal early-stop states | Exact common text and r0.3 machine record unchanged; ECS retains all eight stages once. |
| SCI-VAL | Four independent axes and absence-of-proposition behavior | Exact crosswalk unchanged in the core; views do not redefine it. |
| Responses | fixed, POINT-full-procedure/parent-fixed, and whole-chain meanings remain distinct | Exact common equations and definitions unchanged; rationale displays the three families together. |
| Source/build binding | Stage A, scientific directive, common core, source, build | Preserved and extended with the presentation directive and three source/PDF identities. |

## Removed-Duplication Inventory

- Rationale: removed the complete 38-row requirement register, 32-row
  prediction register, 23-row unavailable-state register, complete edge-case
  matrix, complete engineering record inventories, complete method-record
  templates, and repeated source-closure prose.
- ECS: removed the complete scientific narrative, verbatim requirement and
  prediction prose, repeated eight-stage field lists, and repeated
  method-record field lists.
- The exact normative material now appears once, in the normative core. The
  concise views carry only their audience-specific material and exact
  cross-references.

## Page And Word Comparison

Words are counted from `pypdf` extracted text with Unicode word tokens matching
`\b[\w@./+\-]+\b`.

| Artifact | Revision | Pages | Words | Change from same r0.3 view |
|---|---:|---:|---:|---:|
| Scientific rationale | r0.3 | 21 | 9,335 | baseline |
| Scientific rationale | r0.4 | 8 | 2,041 | -13 pages; -7,294 words |
| Engineering conformance | r0.3 | 22 | 9,835 | baseline |
| Engineering conformance | r0.4 | 9 | 2,608 | -13 pages; -7,227 words |
| Normative core | r0.4 | 18 | 8,346 | new centralized artifact |

The two primary audience views fell from 43 to 17 pages (60.5 percent). The
three-artifact r0.4 publication totals 35 pages while carrying the exact full
normative content only once. The 18-page core is two pages above its soft
target because exact source-byte preservation and readable typography control.
The 9-page ECS is one page below its soft target because repetition was removed
and content was allowed to flow naturally; it was not shrunk to obtain that
count.

## Evidence Boundary

This report establishes editorial and semantic parity only. It makes no claim
of a numerical route, implementation conformity, response or covariance
fidelity, uncertainty coverage, observational pointing accuracy, validation,
performance, readiness, production suitability, production authorization, or
Unity activity.
