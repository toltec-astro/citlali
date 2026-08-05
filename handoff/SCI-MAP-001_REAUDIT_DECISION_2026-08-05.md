# SCI-MAP-001 re-audit decision — 2026-08-05

## Decision identity

- Verified starting HEAD: `02b9eb303037eb3f3a7bb90838b478bb5262e346`.
- Exact application candidate: `ed28dafb37f9113c0d3c95297148157129a90886`.
- Independent audit branch: `codex/reaudit-sci-map-001`.
- Governing report:
  `handoff/SCI-MAP-001_INDEPENDENT_REAUDIT_2026-08-05.md`.
- Decision authority: independent re-auditor; coordinator review is required
  before any ledger or coordination-line change.

## Decision

The approved mathematical contract is retained, but the candidate is not
conformant and the returned evidence is not sufficient to close SCI-MAP-001.
The package-level disposition is:

- contract status: `approved`;
- implementation status: `nonconformant`;
- validation status: `in_progress`;
- production status: `existing_use_only`;
- verdict: `amend`;
- F012 evidence: `insufficient`.

Proposed finding dispositions are:

| Finding | Proposed disposition |
| --- | --- |
| F001 | `closed` |
| F002 | `closed` |
| F003 | `closed` |
| F004 | `open` |
| F005 | `open` |
| F006 | `closed` |
| F007 | `open` |
| F008 | `closed` |
| F009 | `closed` |
| F010 | `addressed_pending_reaudit` |
| F011 | `open` |
| F012 | `open` |
| F013 | `open` |

The closure proposals accept only the scoped repaired behavior. They do not
authorize a downstream consumer, a precision/covariance claim, production
expansion, or closure of any upstream dependency.

## Decisive evidence

- The candidate implements one staged ordinary primitive, finite-positive
  support, strict typed two-phase coadd admission, centered integer embedding,
  `L=I`, the approved nonprecision mean, the eight F010 facts, and lossless
  sidecar provenance. All executed local gates pass.
- Production output narrows typed binary64 WCS values through a binary32
  adapter. A representative all-pixel comparison misses the registered
  `1e-12 degree` bound by more than seven orders of magnitude.
- All 18 sampled coadd threshold FITS cards fail exact binary64 equality with
  their realized sidecar values. No display-rounding exception is registered.
- The ordinary sparse merge can overflow finite floating and integer
  aggregates without a pre-commit check, and finite projected coordinates are
  rounded before their integer representability is established.
- The seven-case corpus is authentically bound to `ed28dafb`, has coherent
  F010 products, strong sequential/OpenMP agreement, and exact reconstructible
  serialized coadds. It lacks the independent raw/trace authority and frozen
  wrapper/Slurm chain. S-X-SEQ lacks same-case observation-noise serialization;
  the S-E sibling is supporting evidence only.
- Typed `stokes_identity=0`, label `I`, conforms to the governing zero-based
  component convention. The frozen analyzer assertions requiring typed value
  `1` are verifier defects. Rejecting them does not cure another failed or
  absent evidence lane.

## Gates and limits

The local Release CLI build, 29 focused truth tests, seven TSan-focused tests,
all 588 enabled CTests, 147 baseline-tool tests, full 127-test config preflight
and its mode/compatibility/authority audits, and the current frozen-package
verifier pass. Those gates omit production typed-to-FITS WCS, aggregate
overflow/index range, concurrent realization merging, realization gamma
bounds, and one complete legacy-WCS atomicity assertion, so F011 remains open.

F013 remains conditioned on SCI-ALIGN-001, SCI-CAL-001, SCI-AST-001,
SCI-PTC-001, and SCI-VAL-001. MAP evidence closes none of them.

No integration, production expansion, Unity action, new evidence request,
repair, or coordination-line change is authorized by this record. The
machine-readable companion is a proposal for later coordinator review, not a
canonical ledger update.
