# SCI-RTC v0.1/r0.10 candidate consistency report

Date: 2026-08-21

Status: implementation-blind candidate-source consistency and PDF verification
complete; scientific-owner review and freeze remain pending.

## Scope and claim boundary

This review covers only the bounded conditioned-$r$ reopening authorized in
`SCIENTIFIC_OWNER_REOPENING_DIRECTIVE_R0.10.md` and the resulting contract
artifacts. It inspected no implementation, tests, configuration, generated
science products, audit/repair history, external literature, or web content.
It establishes no implementation conformity, validation, observational
performance, science qualification, or production readiness.

The candidate starts from exact commit
`2ad12caeabc4a1f84b6748cd7a4cf5683202c51d`, which is a descendant of
`9564bcca0323dacb8bea13a5ec4bbbf3b908de8f` and an ancestor of
`codex/scientific-contract-library`. The frozen r0.9 baseline remains available
unchanged at that starting commit.

## Six-decision closure

| Decision | Normative closure | Falsifier | Ledger |
| --- | --- | --- | --- |
| R10-D01 role/optionality | REQ-109 | PRED-072 | OWNER-084 |
| R10-D02 pair-coherent artifacts/no invented repair | REQ-110 | PRED-073 | OWNER-085 |
| R10-D03 exact grid/failure isolation | REQ-111 | PRED-074 | OWNER-086 |
| R10-D04 response/covariance | REQ-112 and EQ-036 | PRED-075 | OWNER-087 |
| R10-D05 leakage/source protection | REQ-113 | PRED-076 | OWNER-088 |
| R10-D06 consumer handoff | REQ-114 | PRED-077 | OWNER-089 |

The prior conditioned-$r$ reservation in DEF-018, REQ-003/108, and
OWNER-071/074 is explicitly superseded only to the extent required by these
six decisions. Every unrelated owner state remains unchanged.

## Conditioned-x preservation

A direct extraction-and-comparison against the frozen r0.9 source confirmed
that `SCI-RTC-EQ-005` and `SCI-RTC-EQ-006` are byte-identical. Thus the
conditioned-$x$ complete numerical operator and its zero fixed-state
$x\leftarrow r$ branch were not modified. New EQ-036 adds only the optional
conditioned-$r$ companion, coordinate-diagonal pair response, and admitted
joint-covariance propagation. SCI-CAL remains x-only.

## Mechanical verification

`src/verify_contract.py` completed successfully with the following exact
inventory:

- 39 definitions, sequential `DEF-001`--`039`;
- 38 displayed equation tags, including `016a`, `016b`, `020a`, `020b`, and
  new `EQ-036`;
- 12 assumptions, sequential `ASM-001`--`012`;
- 114 requirements, sequential `REQ-001`--`114`;
- 77 predictions, sequential `PRED-001`--`077`;
- 24 author decisions;
- 89 owner entries: 63 open, 1 conditional, 20 resolved, and 5 deferred.

The check also confirmed exact approved-input hashes, the retained-core hash,
the unchanged r0.9 freeze hash, the r0.10 reopening-directive hash, complete
crosswalk rows, one-time six-file core inclusion in the engineering view, the
twelve-section rationale structure, no independent displayed mathematics in
either wrapper, and exact candidate PDF hashes. `git diff --check` reported no
whitespace errors.

## PDF build and all-page inspection

Both PDFs were rebuilt with Tectonic from the candidate TeX sources. The final
logs contain no overfull box, underfull box, or engine warning. Poppler reports
US Letter, PDF 1.5, unencrypted, no forms, no JavaScript, and no suspect
metadata for both artifacts.

| Artifact | Pages | SHA-256 | Visual disposition |
| --- | ---: | --- | --- |
| `pdf/SCI-RTC-SCIENTIFIC-RATIONALE-v0.1.pdf` | 14 | `b09efeb698c736917c159bf5295e0281b21d7ee90f0deea81aca2737ea042e87` | All pages rendered at 110 dpi and inspected; title/status, diagrams, tables, section flow, margins, footer clearance, and final page are clean. |
| `pdf/SCI-RTC-ENGINEERING-CONFORMANCE-v0.1.pdf` | 48 | `ce474dd5f9aa64ddcd664ef21a509fa3de2d53b9c7a6055b1ab0596813dfed49` | All pages rendered at 110 dpi and inspected; EQ-036, REQ-108--114, PRED-072--077, split requirement tables, margins, and end-of-core marker are legible and unclipped. |

All 62 final pages were inspected. No clipping, overlap, accidental blank page,
unreadable equation, table overflow, stale r0.9 status, or audience-view
authority conflict was observed.

## Candidate disposition

The source, crosswalk, ledger, revision records, and two canonical PDFs form a
tightly bounded v0.1/r0.10 candidate for scientific-owner review. They are
content-bound by the hashes above but are not the frozen scientific authority
until Grant Wilson explicitly approves and freezes the candidate.
