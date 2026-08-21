# SCI-RTC v0.1/r0.11 consistency report

Date: 2026-08-21

Status: Complete candidate implementation-blind verification record for owner
review; this is not an authority freeze or implementation-conformity result.

## Authority and scope

- Owner authority: `SCIENTIFIC_OWNER_REVISION_DIRECTIVE_R0.11.md`, exact
  supplied-text SHA-256
  `89eb07832fa064238045c6c765c019f3b6fe74c3e5d1c6f163de5f5ebd20e9d8`.
- Comparison baseline: sealed v0.1/r0.10 commit
  `326ec554998a124202d746f435bec8180e875fa1`.
- Inspected domain: SCI-RTC contract sources and generated PDFs only.
- Excluded: implementation, tests, configuration, generated science products,
  audit/repair history, external literature, and web sources.

## Semantic checks

- [x] Fixed-state response is exactly `I2 tensor L_Pi`, with identical
  ordinary masks/state/phase/grid and zero numerical cross branches.
- [x] $E_x$, $E_r$, and $E_{xr}$ retain origin and direct/inferred causes.
- [x] Only accepted hard action support is unioned; spectral candidates do not
  automatically subtract from $x$.
- [x] Pair event/action state is distinct from raw coordinate validity and
  conditioned coordinate availability.
- [x] Level shifts share event/support/reset but not correction amplitudes.
- [x] The $x$ donor exception creates no $r$ value and propagates honest local
  $r$ unavailability over full causal influence on the common grid.
- [x] CAL, PTC, and SCI-VAL boundaries remain explicit.
- [x] No implementation, validation, performance, qualification, or production
  claim appears.

## Mechanical checks

- [x] `src/verify_contract.py` passes the approved-input hashes, exact shared-
  core inclusion, no independent displayed mathematics in either wrapper,
  sequential inventories, exhaustive crosswalk, ledger counts, baseline
  preservation, and canonical PDF hashes.
- [x] Inventory: 51 definitions, 43 equation tags, 12 assumptions, 138
  requirements, 103 predictions, 24 author decisions, and 96 owner entries.
- [x] Owner states: 63 open, one conditional, 27 resolved, five deferred.
- [x] Rationale retains exactly 12 numbered scientific sections and no
  displayed normative equation.
- [x] Engineering imports each of the six shared-core files exactly once and
  contains no independent displayed equation in the wrapper.
- [x] `git diff --check` reports no whitespace error.

## Build, metadata, text, and visual checks

- [x] Both PDFs built with Tectonic without warning, overfull/underfull box, or
  unresolved-reference message.
- [x] Scientific rationale: 15 US-Letter pages, unencrypted, no form or
  JavaScript; SHA-256
  `f92cefdd064a250466d75be7b1aafb9725c22ff2930a8fecef5a9e1db7315dbd`.
- [x] Engineering/formal view: 60 US-Letter pages, unencrypted, no form or
  JavaScript; SHA-256
  `b11dbf3bfc835f7bf144d4f6088960b3b3a7ff0409a3d93ddcd5514ff8bc24d5`.
- [x] Poppler extraction finds r0.11 in both PDFs and finds EQ-040, REQ-138,
  PRED-103, and the r0.11 end-of-core marker in the engineering PDF.
- [x] All 75 final pages were rendered with Poppler and inspected in contact
  sheets; representative title, rationale closing, equation, requirement,
  prediction, end-of-core, routing, and checklist pages were inspected at
  full-page scale.
- [x] No clipping, overlap, blank spill page, malformed glyph, broken table,
  equation collision, inconsistent orientation, or footer collision remains.

## Claim disposition

The checks above establish only source consistency, deterministic contract-PDF
construction, metadata identity, text presence, and visual integrity. They do
not establish implementation conformity, representation fidelity in an
implementation, validation, performance, science qualification, or production
readiness.
