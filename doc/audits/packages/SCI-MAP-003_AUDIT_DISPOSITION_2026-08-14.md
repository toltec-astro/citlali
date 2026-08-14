# SCI-MAP-003 completed-audit disposition registration

Record ID: `SCI-MAP-003-DISP-001`

Date: 2026-08-14

Status: Phase 1 audit disposition registered; Phase 2 handoffs prepared but
not dispatched

This documentation-only record registers the completed `SCI-MAP-003`
scientific-contract audit without changing its scientific content or frozen
bytes. The audit branch is immutable post-core evidence. It is not the
canonical registry base and is not promoted to pre-core authority.

## Canonical registration authority

- Registration branch: `codex/register-sci-map-003-audit-disposition`
- Canonical base ref:
  `origin/codex/coordinate-sci-map-003-registration`
- Canonical base commit:
  `8fb893639bdf504968186867790cf1b140b2ff54`
- Canonical base parent:
  `b3e500117ab273d38d5055e82683a49015d028cf`
- Canonical base tree:
  `ba91a1cf619401b165060a5b5e31600ed0844d24`
- Registration authority: coordinator amendment for
  `SCI-MAP-003-DISP-001`

The prior registration queue position remains unchanged: after
`SCI-MAP-002` and before any `SCI-MODE-001` LMTOOF consumer admission. This
record does not preempt or launch another package.

## Immutable completed audit and core identities

The completed audit is referenced at its exact pushed evidence identity; none
of these artifacts is recreated, imported, merged, cherry-picked, rebased, or
edited by this registration.

| Field | Exact identity |
| --- | --- |
| Audit ref | `origin/codex/audit-sci-map-003` |
| Audit commit | `65d1abb88b6d8fa0c3235a3d62ef9a2ab3122839` |
| Audit parent | `c0bd689598c8bc1c20114248a45822b13d6ba2f2` |
| Audit tree | `782a3fbed69a3b729ec20fafa6dfa6d4c20a283b` |
| Standard binary audit patch SHA-256 | `d0709105113d9604d15b2940b2a6af087280fa95d3f2c068a2ab7f940207792e` |
| Audit report object | `65d1abb88b6d8fa0c3235a3d62ef9a2ab3122839:doc/audits/packages/SCI-MAP-003_SCIENTIFIC_CONTRACT_AUDIT.tex` |
| Audit report SHA-256 | `2cbad1fd11ff8851b66202197f881c48f8c0f5cc5d08f376256d6dfdb8eb1764` |
| Frozen predecessor core object | `3eecd9f4c6ccf9c9a5c509908b2689a8a0e1a80a:doc/audits/packages/SCI-MAP-003_INDEPENDENT_CORE.tex` |
| Frozen predecessor core SHA-256 | `d6eac8499191f65396d3bc743d5aea70e055ad6342dd812e67d68d31d436073e` |
| Operative successor core object | `c0bd689598c8bc1c20114248a45822b13d6ba2f2:doc/audits/packages/SCI-MAP-003_INDEPENDENT_CORE_SUCCESSOR_001.tex` |
| Operative successor core SHA-256 | `5fb256b18566d725267d25737163f97dd1d5bbd463ba77ed33960e977a5dfed4` |

The predecessor remains frozen and preserved. Successor 001 remains the
operative independent core. The audit report and all implementation
observations made after the core freeze remain post-core evidence.

## Registered disposition axes

The exact completed-audit axes are:

```text
contract_status: proposed
implementation_status: nonconformant
validation_status: not_started
production_status: existing_use_only
verdict: amend
```

The audit is complete as an audit report. The scientific package is not done,
no finding is closed, and no validation or production adoption is implied.

## Open findings

Exactly ten findings remain open. The following titles, owners, priorities,
classes, and closure gates preserve the frozen audit disposition.

### SCI-MAP-003-F001 — No exact discrete g

- Class: `contract_gap`
- Priority: `P1`
- Owner: SCI-RTC-001 and SCI-MAP-003 owners
- Closure gate: Freeze every APT row/weight, Gaussian parameter,
  amplitude/unit, truncation, center, pixelization, WCS, mask, crop, and
  digest; two independent reconstructions meet a preregistered byte or
  arithmetic identity.

### SCI-MAP-003-F002 — No admissible complete final k parent

- Class: `dependency_gap`
- Priority: `P0`
- Owners: SCI-RTC-001, SCI-PTC-001, SCI-MAP-002, SCI-VAL-001, and
  SCI-FRUIT-001 owners
- Closure gate: Complete-response-or-unavailable, eligibility, and
  terminal-pass contracts close; the exact serial pre-crop normalized kernel
  and once-cropped k have immutable parent/stage/digest identity at the
  governing application SHA.

### SCI-MAP-003-F003 — Parallel JINC shared-array data races

- Class: `implementation_defect`
- Priority: `P0`
- Owner: SCI-MAP-002 / Citlali mapmaker owner
- Closure gate: An authorized repair provides race-free private/deterministic
  reduction or proved disjoint writes; thread-sanitizer evidence and
  preregistered serial-versus-parallel byte/tolerance equivalence pass for all
  accumulators, then a fresh independent audit accepts the exact repair SHA.

### SCI-MAP-003-F004 — Odd grid, transform, quotient, and domain absent

- Class: `contract_gap`
- Priority: `P1`
- Owner: SCI-MAP-003 with SCI-MAP-001 and SCI-AST-001 interfaces
- Closure gate: The exact owner crop and aligned planes are implemented and
  persisted; Cases 1--4 and 6--11 pass after ND-GRID and ND-DEN are frozen.

### SCI-MAP-003-F005 — Transfer product schema and provenance absent

- Class: `contract_gap`
- Priority: `P1`
- Owners: SCI-MAP-003, SCI-MAP-001, and SCI-MODE-001 owners
- Closure gate: A versioned product contract defines representation,
  cardinality, exact identity/provenance, aliases, association,
  required/optional behavior, atomic failure, and digest; deliberate
  mismatch/publication tests pass.

### SCI-MAP-003-F006 — Response uncertainty and covariance unavailable

- Class: `contract_gap`
- Priority: `P1`
- Owner: SCI-MAP-003 with RTC/PTC/VAL and later LMTOOF owners
- Closure gate: An approved estimator and persistence contract supplies or
  explicitly marks unavailable every formal, empirical,
  calibration/systematic, and response uncertainty term; named
  coverage/calibration tests satisfy a frozen protocol.

### SCI-MAP-003-F007 — Remaining grid/domain/synthesis policy open

- Class: `scientific_policy_decision`
- Priority: `P1`
- Owner: SCI-MAP-003 owner, with AST/VAL interfaces
- Closure gate: One owner decision freezes every operator and boundary value
  before implementation or Case 7/8 execution; at/below/above threshold and
  operator identity cases then pass unchanged.

### SCI-MAP-003-F008 — Amplitude, uncertainty, publication, and consumer admission open

- Class: `scientific_policy_decision`
- Priority: `P1`
- Owners: SCI-MAP-003, SCI-MODE-001, SCI-VAL-001, and LMTOOF product owners
- Closure gate: Separate owner decisions freeze optical units/nuisance
  amplitude, uncertainty, product schema, OOF association, fixed/recompute
  lifecycle, allowed consumer version, identity rejection, and failure
  policy.

### SCI-MAP-003-F009 — VAL, FRUIT, and MODE packages not started

- Class: `dependency_gap`
- Priority: `P0`
- Owners: SCI-VAL-001, SCI-FRUIT-001, and SCI-MODE-001 owners
- Closure gate: Each package produces an owner-approved bounded contract and
  accepted implementation/evidence result for its named interface; a dated
  re-audit admits it without rewriting either frozen core.

### SCI-MAP-003-F010 — No validation or adoption evidence

- Class: `evidence_gap`
- Priority: `P1`
- Owners: SCI-MAP-003 validation owner and separately authorized
  external/telescope owners
- Closure gate: After contracts and implementation blockers close,
  separately authorized preregistered studies pass unchanged under
  FRAMEWORK-NUM-001 controls, followed by independent review and an explicit
  production decision.

## Dependency disposition

The exact dependency states and boundaries remain:

| Package | State | Registered closure condition |
| --- | --- | --- |
| `SCI-MAP-001` | `open` | Exact OOF-cell parent, WCS/unit/validity, required-output, schema, and implementation conformance are accepted. |
| `SCI-MAP-002` | `conditioned` | Race-free exact governing implementation conforms to D003 and all parent/VAL gates, with fresh audit. |
| `SCI-RTC-001` | `open` | Exact g and every response-changing stage are represented with immutable parentage, or response is honestly unavailable. |
| `SCI-PTC-001` | `conditioned` | Exact selection/realization response family and needed uncertainty/validity interfaces conform at the governing SHA. |
| `SCI-AST-001` | `conditioned` | Common retained WCS/crop and dual-frequency axes pass orientation and identity fixtures without silent recentering. |
| `SCI-VAL-001` | `open`; package `not_started` | Owner-approved VAL contract and conforming fail-closed implementation. |
| `SCI-FRUIT-001` | `open`; package `not_started` | Owner-approved terminal/restart/parent-state contract and conforming evidence. |
| `SCI-MODE-001` | `open`; package `not_started` | Approved association, units/amplitude, fixed/recompute lifecycle, version, identity checks, and failure policy. |

Open and conditioned dependencies do not prevent registration of the
completed audit. They keep transfer construction, publication, LMTOOF use,
and production fail closed.

## Inbound handoff dispositions

The completed audit proposed these exact dispositions. This registration
records them without editing the eight immutable inbound handoff files and
without allowing any handoff to close a finding or dependency:

| Inbound ID | Audit disposition | Review class retained |
| --- | --- | --- |
| `SCI-MAP-003-XAUD-001` | `accept_bounded_pre_core_authority` | `pre_core_authority` |
| `SCI-MAP-003-XAUD-002` | `accept_bounded_pre_core_authority` | `pre_core_authority` |
| `SCI-MAP-003-XAUD-003` | `accept_bounded_pre_core_authority` | `pre_core_authority` |
| `SCI-MAP-003-XAUD-004` | `accept_bounded_pre_core_authority` | `pre_core_authority` |
| `SCI-MAP-003-XAUD-005` | `accept_bounded_pre_core_authority` | `pre_core_authority` |
| `SCI-MAP-003-XAUD-006` | `accept_post_core_dependency_evidence` | `post_core_evidence` |
| `SCI-MAP-003-XAUD-007` | `accept_post_core_dependency_evidence` | `post_core_evidence` |
| `SCI-MAP-003-XAUD-008` | `accept_post_core_dependency_evidence` | `post_core_evidence` |

The first five records retain only their already-approved bounded authority.
The last three remain post-core dependency evidence and supply no VAL, FRUIT,
or MODE contract fact.

## Phase 2 administrative handoff mapping

The frozen audit's source-prefixed labels remain immutable proposal labels in
the audit evidence. They are not canonical IDs or schema aliases. The
coordinator-approved one-to-one mapping is:

| Frozen audit proposal label | Canonical target-prefixed handoff ID | Target |
| --- | --- | --- |
| `SCI-MAP-003-XAUD-009` | `SCI-MAP-002-XAUD-002` | `SCI-MAP-002` |
| `SCI-MAP-003-XAUD-010` | `SCI-RTC-001-XAUD-004` | `SCI-RTC-001` |
| `SCI-MAP-003-XAUD-011` | `SCI-PTC-001-XAUD-007` | `SCI-PTC-001` |
| `SCI-MAP-003-XAUD-012` | `SCI-MAP-001-XAUD-005` | `SCI-MAP-001` |
| `SCI-MAP-003-XAUD-013` | `SCI-AST-001-XAUD-003` | `SCI-AST-001` |
| `SCI-MAP-003-XAUD-014` | `SCI-VAL-001-XAUD-011` | `SCI-VAL-001` |
| `SCI-MAP-003-XAUD-015` | `SCI-FRUIT-001-XAUD-001` | `SCI-FRUIT-001` |
| `SCI-MAP-003-XAUD-016` | `SCI-MODE-001-XAUD-003` | `SCI-MODE-001` |

Only the target-prefixed values are canonical handoff IDs. Phase 2 is
`prepared_not_dispatched`: no handoff file, registry entry, message, recipient
review, recipient package launch, or downstream task is created by Phase 1.

## Product and authority boundary

- Existing OOF remains `existing_use_only` under its existing contract.
- Transfer construction and publication remain fail closed and unavailable.
- LMTOOF remains fail closed and unauthorized.
- All ten findings remain open.
- Validation remains `not_started`; no numerical result is registered.
- The audit report is evidence for this disposition, not a new scientific
  authority and not an application dependency.
- No post-core implementation observation is promoted to pre-core authority.

## Phase-1 ceiling and next gate

Phase 1 is limited to this disposition, its machine-readable ledger proposal,
`doc/audits/README.md`, and `doc/audits/audit-ledger.yaml`. Phase 2 requires a
separate coordinator authorization and must create each target-prefixed
handoff from the immutable source claim without changing its finding, owner,
dependency, priority, closure gate, scientific content, or evidence class.

This record does not authorize application, test, configuration, validation-
product, audit, re-audit, repair, numerical, evidence-request, external,
Unity, reduction, integration, production, merge, rebase, push, or downstream
launch activity.
