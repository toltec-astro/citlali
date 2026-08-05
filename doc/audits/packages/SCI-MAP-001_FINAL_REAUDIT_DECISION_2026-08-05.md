# SCI-MAP-001 final re-audit decision — 2026-08-05

## Decision identity

- Exact repair candidate and verified repair-branch tip:
  `af0c849ce59a5f80e5efc8db435bb6662863052f`.
- Parent repair candidate:
  `f84b9fd7d7364f9d35317fc6c15b55d2a30e89f7`.
- Second-cycle independent audit:
  `fc26e24e6543d1102f9fcc9bf4e849369b39dd04`.
- Owner amendment:
  `6409a36d324072c9b29145c620d01a0686275870`, SHA-256
  `52be19700b73659ba1847012d4cb0766407399cda5899570acb79bf5b45221f3`.
- Independent audit branch: `codex/final-reaudit-sci-map-001`.
- Governing report:
  `handoff/SCI-MAP-001_FINAL_NARROW_INDEPENDENT_REAUDIT_2026-08-05.md`.

## Decision

The stage-aware bookkeeping repair conforms. Observation and coadd products
now use their respective output-stage counts. The coadd-plus-filter
three-map/two-realization state records exactly 9 empirical product maps and
18 realization writes, while the filtering-disabled and non-coadd policy
states remain unchanged.

The proposed bounded package disposition is:

- contract status: `approved`;
- implementation status: `conformant` within SCI-MAP-001 scope;
- validation status: `complete` within the local and owner-accepted bounded
  evidence scope;
- production status: `existing_use_only`;
- verdict: `accept` for the bounded MAP scientific contract;
- re-audit status: `complete`;
- MAP repair/re-audit work: complete and ready for coordinator ledger
  integration; and
- application integration and production expansion: not authorized by this
  audit.

| Finding | Proposed disposition |
| --- | --- |
| F001 | `closed` — prior scoped closure not reopened |
| F002 | `closed` — prior scoped closure not reopened |
| F003 | `closed` — prior scoped closure not reopened |
| F004 | `closed` — exact persistence/cardinality contract now conforms |
| F005 | `closed` — second-cycle closure confirmed |
| F006 | `closed` — prior scoped closure not reopened |
| F007 | `closed` — second-cycle closure confirmed |
| F008 | `closed` — prior scoped closure not reopened |
| F009 | `closed` — prior scoped closure not reopened |
| F010 | `closed` — second-cycle closure confirmed |
| F011 | `closed` — missing exact state added; complete gates pass |
| F012 | `closed_bounded_owner_accepted` — named limitations retained |
| F013 | `open_conditioned` — ALIGN/CAL/AST/PTC/VAL remain open |

## Decisive evidence and limitation

The exact three-state selection passes 3/3. All six required targets build;
593/593 enabled CTests, 31 focused truth/provenance tests, nine TSan tests, 22
production FITS tests, 147 baseline tests, and the full 127-test config
preflight pass. The final diff changes only realized cardinality bookkeeping
and its regression; it changes no numerical, noise-generation, mapmaking,
coadd, WCS, threshold, or output-routing behavior.

F012 proves only the owner-amended exact-`ed28dafb` external
execution/completion, product/inventory, visible observation/coadd, and
SEQ/OMP claims. Missing raw/sample, pre-normalization, operational-chain, and
same-case S-X external files remain limitations. No corpus or Unity access
occurred.

MAP closure does not establish upstream production eligibility. F013 remains
conditioned on SCI-ALIGN-001, SCI-CAL-001, SCI-AST-001, SCI-PTC-001, and
SCI-VAL-001, and production remains `existing_use_only`. Only the coordinator
may integrate this proposal into canonical state or authorize later
integration/production changes.
