# Final package inventory

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.4**  
Revision date: **2026-09-06**

The package contains **48 final/source files** outside `build/`. They are the
complete 48-file sealed r0.3 inventory copied before revision and then revised
only within the new sibling. `DRAFT_ARTIFACT_MANIFEST.sha256` contains 47
artifact entries and excludes itself and all build scratch.

## Authored final artifacts

- `README.md` — program/recovery opening, r0.4 correction status, contract
  synopsis, artifact map, and open owner action.
- `scientific_rationale.pdf` — 5-page explanatory scientist-facing view.
- `engineering_conformance.pdf` — 14-page complete engineering
  specification.
- `src/scientific_rationale.tex` and `src/engineering_conformance.tex` —
  view entrypoints with exact metadata, revision date, and all-page headers.
- `src/common_core.tex` — thin composition wrapper.
- `src/common/notation.tex`, `definitions.tex`, `equations.tex`,
  `assumptions.tex`, `requirements.tex`, and `edge_cases.tex` — the six
  canonical singly sourced modules.
- `OWNER_REVIEW_1_RESPONSE.md` — point-by-point OR1-01--05, OR1-S1--S3,
  OR1-A1 dispositions, and the bounded r0.4 within-round correction.
- `CROSSWALK.md` — exact three-column mapping for all 25 requirements.
- `PREDICTION_CROSSWALK.md` — all 16 predictions and 10 mixed cases with
  discriminating evidence.
- `OWNER_DECISION_LEDGER.md` — eight recovered constraints, owner-review
  evidence, fourteen open proposals, one precise open question, and five
  deferred items.
- `PROPOSED_REGISTRY_AND_BOUNDARY_RECORDS.md` — proposed exact family,
  profile, MAP, JINC, and NOI records, including the corrected structural and
  payload-integrity partition.
- `BLOCKED_DEFINITIONS_AND_QUESTIONS.md` — no blocked mathematical definition
  and one exact actual-owner question.
- `INPUT_INTEGRITY_AND_READ_RECORD.md` — both input preflights, content
  hashes/read order, firewall, predecessor verification, and revision
  provenance.
- `DECISION_REGISTER_CHECK.md` — PDF-register/ledger, owner-response,
  correction, requirement, prediction, and mixed-case traceability checks.
- `LINK_AND_COPY_CHECK.md` — local-link, immutable-copy, and all predecessor
  manifest checks.
- `RENDER_RECEIPT.md` — marker, cached build, metadata, hash, header, and
  visual-QA receipt.
- `PACKAGE_INVENTORY.md` — this inventory.
- `DRAFT_ARTIFACT_MANIFEST.sha256` — exact SHA-256 binding for every returned
  artifact except itself; it grants no scientific authority.

## Immutable local input copies

- `PRIOR_WORK.md`, `SCOPE_BRIEF.md`, and the seven files under
  `references/` are byte-exact local copies from the original released
  packet, retained so approved links resolve.
- `inputs/` contains the complete byte-exact eleven-file original packet,
  including its `MANIFEST.json` and `README.md`.
- `owner-review-1-input/` contains the complete byte-exact four-file owner
  supplement, including its manifest, cover, verbatim assessment, and admitted
  frozen-clause extract.

## Scratch and final render evidence

`build/` is non-authoritative scratch and is excluded from the artifact
manifest. It contains:

- final byte-identical build copies of both PDFs and their final Tectonic logs;
- 19 page rasters at 110 dpi under `build/qa-r0.4/` plus four contact sheets;
- `check_package.py`, `check_links_and_copies.py`,
  `check_pdf_text.py`, and their PASS outputs where applicable;
- `make_contact_sheets.py`, used only for local visual inspection.

No predecessor build directory, PDF, log identity, raster, or superseded r0.4
PDF is retained in this package.
