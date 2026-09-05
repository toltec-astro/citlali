# Final package inventory

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1**

## Final author artifacts

- `README.md` — package opening, status, artifact map, and decision headline.
- `scientific_rationale.pdf` — 12-page scientist-facing view.
- `engineering_conformance.pdf` — 11-page engineering-facing view.
- `src/scientific_rationale.tex` and
  `src/engineering_conformance.tex` — view entrypoints with exact PDF
  metadata and forced version/revision headers.
- `src/common_core.tex` — thin composition wrapper.
- `src/common/notation.tex`, `definitions.tex`, `equations.tex`,
  `assumptions.tex`, `requirements.tex`, and `edge_cases.tex` — the six
  canonical, singly sourced scientific modules.
- `CROSSWALK.md` — three-column mapping for all 25 requirements.
- `PREDICTION_CROSSWALK.md` — all 16 predictions, discriminating
  observations, and evidence layers.
- `OWNER_DECISION_LEDGER.md` — seven recovered decisions, fourteen open
  author proposals, one precise open question, and five deferred items.
- `PROPOSED_REGISTRY_AND_BOUNDARY_RECORDS.md` — proposed exact family,
  profile, MAP, JINC, and NOI bindings.
- `BLOCKED_DEFINITIONS_AND_QUESTIONS.md` — no blocked mathematical
  definition; one exact actual-owner question for Registry activation.
- `INPUT_INTEGRITY_AND_READ_RECORD.md` — input digest, inventory, content
  hashes, read order, and firewall record.
- `DECISION_REGISTER_CHECK.md` — explicit PDF-register/ledger and
  crosswalk-count check.
- `LINK_AND_COPY_CHECK.md` — local Markdown-link and byte-exact source-copy
  check.
- `RENDER_RECEIPT.md` — cached build attempts, final build status, metadata,
  visual QA, page counts, and PDF hashes.
- `DRAFT_ARTIFACT_MANIFEST.sha256` — exact hashes of the returned final and
  preserved-source artifacts; it excludes itself and scratch.
- `PACKAGE_INVENTORY.md` — this inventory.

## Byte-exact local source copies

The following are preserved inputs rather than new author content:

- `PRIOR_WORK.md`, `SCOPE_BRIEF.md`, and the seven files under
  `references/` make all links in the required package opening resolve.
- `inputs/` is a labeled byte-exact copy of all eleven released packet files,
  including `inputs/MANIFEST.json` and `inputs/README.md`.

The root source copies and their counterparts under `inputs/` have no byte
differences. The packet manifest and all ten declared content hashes remain
unchanged.

## Scratch and render logs

`build/` is non-authoritative scratch:

- `build/scientific_rationale.pdf` and
  `build/engineering_conformance.pdf` are byte-identical build copies of the
  root PDFs.
- `build/scientific_rationale.log` and
  `build/engineering_conformance.log` are the final Tectonic logs.
- `build/check_markdown_links.py` is the local link checker.
- `build/qa/`, `build/qa-final/`, and `build/qa-header/` contain
  110-dpi page rasters used for visual inspection.

Scratch files are intentionally omitted from the artifact manifest.
