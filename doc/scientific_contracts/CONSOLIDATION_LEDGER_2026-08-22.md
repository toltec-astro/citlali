# Scientific Contract Library Consolidation Ledger — 2026-08-22

Status: canonical pre-audit inventory and provenance record

Canonical branch: `codex/scientific-contract-library`

Scientific owner: Grant Wilson

## Purpose And Boundary

This ledger records the consolidation of the packaged Citlali scientific
rationales, engineering conformance contracts, Stage A/Stage B control
artifacts, decision records, crosswalks, revision history, frozen PDFs, and
horizontal contract reviews into one branch and one canonical documentation
root:

`doc/scientific_contracts/`

It is a branch and artifact inventory, not a new scientific authority. Package
status remains governed by each package's owner ledger and freeze record. No
implementation conformity, observational validation, readiness, or production
claim follows from consolidation.

Implementation-audit and repair packages such as `SCI-ALIGN-001` and
`SCI-AST-001` are not silently promoted into this implementation-independent
library. They remain separately governed evidence for a later conformity
exercise unless a package's approved prior-work record already cites a
sanitized scientific input.

## Canonical Package Inventory

| Package | Canonical path | Consolidated scientific state |
| --- | --- | --- |
| SCI-CAL | [`packages/SCI-CAL/v0.1/`](packages/SCI-CAL/v0.1/) | r0.5 rationale and r0.4 engineering view; Q01--Q09 decided; validation evidence and final owner acceptance pending; not frozen |
| SCI-MAP | [`packages/SCI-MAP/v0.1/`](packages/SCI-MAP/v0.1/) | r0.3 house version retained; scientific authority not frozen |
| SCI-BEAM | [`packages/SCI-BEAM/v0.1/`](packages/SCI-BEAM/v0.1/) | v0.1/r0.3 scientific authority frozen; implementation conformity unassessed |
| SCI-RTC | [`packages/SCI-RTC/v0.1/`](packages/SCI-RTC/v0.1/) | v0.1/r0.12 scientific authority frozen; implementation conformity unassessed |
| SCI-PTC | [`packages/SCI-PTC/v0.1/`](packages/SCI-PTC/v0.1/) | v0.1/r0.4 scientific authority frozen; implementation conformity unassessed |
| SCI-VAL | [`packages/SCI-VAL/v0.1/`](packages/SCI-VAL/v0.1/) | manager-reviewed r0.3 pair; scientific authority not frozen |
| SCI-ALIGN | [`packages/SCI-ALIGN/v0.1/`](packages/SCI-ALIGN/v0.1/) | v0.1/r0.3 scientific authority frozen; implementation conformity unassessed |
| SCI-AST | [`packages/SCI-AST/v0.1/`](packages/SCI-AST/v0.1/) | v0.1/r0.3 scientific authority frozen; implementation conformity unassessed |

The library index remains the governing concise status view. This table makes
no inference beyond the package records.

## Commit And Branch Provenance

| Material | Provenance retained in the consolidated history |
| --- | --- |
| Program charter and package framework | began at `f86cd2523` |
| SCI-CAL original package lineage | through `b237c5f0b`; later repair commits `26c1fc897`, `3313bced3`, and `b1835675b` merged by `327727162` |
| SCI-MAP | frozen house revision at `82bde1886` |
| SCI-BEAM | frozen v0.1/r0.3 at `c2a01842e` |
| SCI-RTC | frozen v0.1/r0.12 at `9c1107b11`, with its branch authority merged at `9d516ba32` |
| SCI-PTC | frozen v0.1/r0.4 at `9564bcca0` |
| RTC--CAL--PTC horizontal coherence profile | `2ad12caea` |
| SCI-VAL | r0.3 contract pair at `e25d1e569` |
| SCI-ALIGN and SCI-AST exact r0.3 freeze packet | original content commit `353b11887`; documentation-only cherry-pick `2686e2794`; normalized here into the canonical package root without changing manifest-bound bytes |

The original ALIGN/AST commit was cherry-picked rather than merged because
its source branch also contains separately governed implementation work. This
ledger preserves both the original identity and the consolidated-history
identity instead of implying ancestry that is not present.

## Retained ALIGN/AST Revision History

The canonical r0.3 package manifests have the following SHA-256 identities:

- SCI-ALIGN: `26285329635c722cb9161d383ad1b95f56a03b782c101bcd89d8785a3575faac`
- SCI-AST: `b54b6013750540f28aad02339a60bf36078980dc53b132beab73069d66ef3601`

The exact r0.2 predecessor packets were recovered from surviving workspace
and turn-checkpoint artifacts and retained under each package's `history/r0.2/`
directory. Their manifests are internally hash-consistent:

- SCI-ALIGN r0.2 manifest: `7d4aa3741e29c83b2c9c30181a2a79917879373dab1dc02f6c2cad38f4761aa7`
- SCI-AST r0.2 manifest: `aca842135a28b9202c39e34f75212359830008cbb18a72edd76e2237fbb22011`
- joint r0.2 horizontal audit: `cb3d2f4575050cad3af99e30a2d8a757adc3fe591845426ef0cb8012c5bb4044`

Available initial PDF-only drafts are retained under
`history/r0.1-pdf-only/`. They are historical artifacts, not current
authority.

## Known Recovery Gap

The original standalone ALIGN and AST Stage A Scope Brief, internal dossier,
and author-packet files were not found in reachable branch history or the
surviving workspace checkpoints. They have not been reconstructed and are not
claimed as recovered.

The approved scientific decisions, typed owner questions, revision authority,
and exact frozen Stage B content do survive in each package's owner register,
author decisions, change records, and source manifest. Each canonical package
therefore carries a clearly labeled retrospective scope-control note that
routes readers to those exact authorities without inventing missing Stage A
text. This is the principal documented provenance gap for the next audit.

## Consolidation Rule

Future scientific-contract work starts from this branch and canonical root,
links to the program charter and prior-work registry, and records its package
in the index. Existing frozen content is never silently edited. A scientific
change requires its package's stated successor or reopening procedure; a
branch merge alone cannot change scientific authority.
