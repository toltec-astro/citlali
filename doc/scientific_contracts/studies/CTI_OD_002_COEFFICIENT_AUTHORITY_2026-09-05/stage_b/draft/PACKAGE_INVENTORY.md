# Final package inventory

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3**  
Revision date: **2026-09-06**

The package contains **48 final/source files** outside build: the 43 r0.2 final/source files copied byte-for-byte before revision, four byte-exact owner-review supplement files, and the new OWNER_REVIEW_1_RESPONSE.md. DRAFT_ARTIFACT_MANIFEST.sha256 contains 47 artifact entries and excludes itself and all build scratch.

## Authored final artifacts

- README.md — required program/recovery opening, revision status, contract synopsis, artifact map, and open owner action.
- scientific_rationale.pdf — 5-page explanatory scientist-facing view.
- engineering_conformance.pdf — 14-page complete engineering specification.
- src/scientific_rationale.tex and src/engineering_conformance.tex — view entrypoints with exact metadata, revision date, and all-page headers.
- src/common_core.tex — thin composition wrapper.
- src/common/notation.tex, definitions.tex, equations.tex, assumptions.tex, requirements.tex, and edge_cases.tex — the six canonical singly sourced modules.
- OWNER_REVIEW_1_RESPONSE.md — point-by-point OR1-01--05, OR1-S1--S3, and OR1-A1 dispositions.
- CROSSWALK.md — exact three-column mapping for all 25 requirements.
- PREDICTION_CROSSWALK.md — all 16 predictions and 10 mixed cases with discriminating evidence.
- OWNER_DECISION_LEDGER.md — eight recovered constraints, owner-review evidence, fourteen open proposals, one precise open question, and five deferred items.
- PROPOSED_REGISTRY_AND_BOUNDARY_RECORDS.md — proposed exact family, profile, MAP, JINC, and NOI records.
- BLOCKED_DEFINITIONS_AND_QUESTIONS.md — no blocked mathematical definition and one exact actual-owner question.
- INPUT_INTEGRITY_AND_READ_RECORD.md — both input preflights, content hashes/read order, firewall, and revision provenance.
- DECISION_REGISTER_CHECK.md — PDF-register/ledger, owner-response, requirement, prediction, and mixed-case traceability checks.
- LINK_AND_COPY_CHECK.md — local-link, immutable-copy, and predecessor-manifest checks.
- RENDER_RECEIPT.md — marker, cached build, metadata, hash, header, and visual-QA receipt.
- PACKAGE_INVENTORY.md — this inventory.
- DRAFT_ARTIFACT_MANIFEST.sha256 — exact SHA-256 binding for every returned artifact except itself; it grants no scientific authority.

## Immutable local input copies

- PRIOR_WORK.md, SCOPE_BRIEF.md, and the seven files under references are byte-exact local copies from the original released packet, retained so approved links resolve.
- inputs contains the complete byte-exact eleven-file original packet, including its MANIFEST.json and README.md.
- owner-review-1-input contains the complete byte-exact four-file owner supplement, including its MANIFEST.json, cover, verbatim assessment, and admitted frozen-clause extract.

## Scratch and final render evidence

build is non-authoritative scratch and is excluded from the artifact manifest. It contains:

- final byte-identical build copies of both PDFs and their final Tectonic logs;
- 19 page rasters at 110 dpi under build/qa-r0.3 plus four contact sheets;
- check_pdf_text.py, check_links_and_copies.py, check_package.py, and their PASS output where applicable;
- make_contact_sheets.py used only for local visual inspection.
- build/superseded-qa-checkpoint-before-profile-binding-restoration contains the clearly labeled intermediate r0.3 PDF pair retained at the manager's request; it is excluded from final artifact identities and the manifest.

No old r0.2 build directory, PDF, log identity, or raster was copied into this package.
