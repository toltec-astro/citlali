# SCI-MAP-001 second-cycle re-audit decision — 2026-08-05

## Decision identity

- Verified clean starting HEAD and exact repair candidate:
  `f84b9fd7d7364f9d35317fc6c15b55d2a30e89f7`.
- Required repair branch tip: `codex/repair-sci-map-001`.
- Independent audit branch: `codex/second-cycle-reaudit-sci-map-001`.
- Historical re-audit: `851035e67f63bdb2bacc122b17566877a9e6db97`.
- Owner amendment: `6409a36d324072c9b29145c620d01a0686275870`;
  artifact SHA-256
  `52be19700b73659ba1847012d4cb0766407399cda5899570acb79bf5b45221f3`.
- Governing report:
  `handoff/SCI-MAP-001_SECOND_CYCLE_INDEPENDENT_REAUDIT_2026-08-05.md`.
- Authority: independent re-auditor; coordinator review is required for any
  canonical update.

## Decision

The amended contract is approved, and the repairs for F005, F007, and F010
conform. The repair is not complete because the coadd-plus-filter output state
overstates realized product and realization-write cardinality. The
package-level proposal is:

- contract status: `approved`;
- implementation status: `nonconformant`;
- validation status: `in_progress`;
- production status: `existing_use_only`;
- verdict: `amend`;
- re-audit status: `complete` for `f84b9fd7`;
- F012 evidence: `accepted_bounded_with_limitations`;
- integration and production expansion: not authorized.

| Finding | Proposed disposition |
| --- | --- |
| F001 | `closed` — not reopened |
| F002 | `closed` — not reopened |
| F003 | `closed` — not reopened |
| F004 | `open` |
| F005 | `closed` |
| F006 | `closed` — not reopened |
| F007 | `closed` |
| F008 | `closed` — not reopened |
| F009 | `closed` — not reopened |
| F010 | `closed` |
| F011 | `open` |
| F012 | `closed_bounded_owner_accepted` |
| F013 | `open_conditioned` |

## Decisive evidence

- Checked floating and signed-count aggregation plus finite projected-index
  range rejection occurs before live bundle mutation; normal accepted
  arithmetic order remains exact.
- The production writer passes the amended `0.1 arcsec` WCS boundary, exact
  sign/orientation/integer-centering relations, and the exact-sidecar plus
  finite/unit/identity/alias/`rtol=1e-12` threshold-card contract.
- Required raw observation realization files now persist with coaddition, and
  missing required output is rejected before the first primary HDU or legacy
  WCS mutation.
- With filtering and coaddition together, observation realizations have one
  output stage and coadd realizations have two. Noise completion instead
  multiplies both by two. In the inspected three-map/two-realization example,
  actual writes are 18 while provenance records 24; product maps are 9 versus
  12. The coadd cardinality test passes `filtered_maps_enabled=false`, so the
  failing state has no regression test.

The smallest remaining work is stage-aware noise product/write cardinality
using the existing separate observation and coadd stage rules, plus one exact
coadd-plus-filter provenance test. No owner decision, numerical algorithm
change, Unity rerun, or new framework is needed.

## Gates and limitations

The exact-candidate Release CLI build, 31 focused truth tests, nine TSan tests,
22 production FITS tests, all 592 enabled CTests, 147 baseline-tool tests, and
the full 127-test config preflight pass. Validation ledger/profile checks pass.
The pinned campaign package verifies 21 members and passes its driver
self-check against exact `ed28dafb`; it correctly rejects the newer `f84`
registry as not byte-identical to the campaign pin.

The bounded before/after ordinary-fixture measurement was 0.35 s versus 0.36 s
median over 500 repetitions, with identical measured user CPU. No dramatic
regression was observed. Sparse aggregation moved outside the mutex, while
live preflight and an extra noise read pass occur inside; production-scale
lock contention remains unmeasured because no existing fixture exposes it.

F012 closure proves only the owner-amended exact-`ed28dafb`
execution/product/inventory/visible-coadd/SEQ-OMP claims. Missing raw/sample
ledgers, pre-normalization traces, operational chain, and S-X observation
realization files remain limitations. F013 remains conditioned on
SCI-ALIGN-001, SCI-CAL-001, SCI-AST-001, SCI-PTC-001, and SCI-VAL-001.

This record launches no repair or campaign and changes no canonical ledger,
coordination line, application source, test, configuration, original artifact,
or external corpus.
