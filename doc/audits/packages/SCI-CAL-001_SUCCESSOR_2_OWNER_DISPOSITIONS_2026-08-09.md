# SCI-CAL-001 successor-2 owner dispositions

Date: 2026-08-09

Record ID: `SCI-CAL-001-SUCCESSOR-2-OWNER-D001`

Status: project-owner approved; documentation-only authority for preparing a
bounded successor repair handoff; repair not launched

## Exact context and unchanged disposition

- Canonical application mainline:
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.
- Rejected-as-complete-closure repair candidate:
  `7894346a91fa78ceb2a8b3d625335f466e5e1756` (parent
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`, tree
  `991f96c64e4d2d973ed5fc02630bfe29149109d9`).
- Exact re-audit: `4140923de4ae33d36224493b5937e291bd552d30`
  (parent `7894346a91fa78ceb2a8b3d625335f466e5e1756`, tree
  `2f051f6831b25e0b297fb61905f40e5e17c6a925`).
- Immutable re-audit report SHA-256:
  `7a9eeae603871f3e2c157b123c15970dd2b2e472257479d100b02bea43101d34`.
- Immutable re-audit ledger-proposal SHA-256:
  `47a63a5c2a2fcc1000547dd5cdc64d24382818666e299b6629e92afff28e9ee2`.
- Coordination snapshot used for this decision consolidation:
  `9e175f3bed43366e81267011d7063ca4f39d9176`.

The candidate is not complete CAL closure. The exact fixed atmosphere operator
is retained as successor material, and F002 keeps its narrow structural
closure. No other finding is closed merely because the candidate or local
tests exist.

The canonical axes remain:

- contract: `approved`;
- implementation: `nonconformant`;
- validation: `in_progress`;
- production: `fail_closed`; and
- verdict: `amend`.

## Owner-approved successor-2 finding dispositions

### F001 — conditioned external dependencies

`SCI-CAL-001-F001` remains open P0 and conditioned. Kernel/response and
downstream weight propagation beyond the existing conditional
variance/weight recipients covered by F005, accepted ALIGN/AST authority,
exact-successor-SHA Unity evidence, astronomical-standard evidence, and
empirical response fidelity remain external dependencies.

They must not broaden the bounded CAL implementation repair. CAL may reach
local implementation conformance while production precision/accuracy and
response-fidelity claims remain explicitly fail closed until those
dependencies return and are accepted.

### F002 — narrow structural closure retained

`SCI-CAL-001-F002` remains closed only for the fixed atmosphere operator's
structural low-opacity behavior. The operator removes the finite-positive
unity plateau and interpolates LOS optical depth continuously from the analytic
zero anchor through the first nonzero anchor with the accepted node, endpoint,
monotonicity, and seam behavior.

This does not establish atmosphere truth, model fidelity, observational
calibration, response fidelity, or production authority. The successor repair
must preserve the exact operator and focused regression coverage without
redesigning it.

### F003 — configuration, admission, and cause-specific failure

`SCI-CAL-001-F003` remains open P0. A requested calibration output unit outside
the approved supported set must fail during initial configuration/startup
validation. No observation analysis, APT processing, TOD mutation, output
creation, or scientific publication may begin.

A malformed or incompatible APT/raw acquisition binding fails after the exact
inputs are identified, but still before calibration, TOD mutation, output
creation, or scientific publication. Explicit RTC-only or intentionally
uncalibrated modes retain only their approved skip semantics and must never be
labeled or published as calibrated.

Every failure is typed, persistent, cause-specific, and propagated to the
engine/CLI boundary. A failure need not be forced into an already constructed
`CalibrationProduct`; the correct boundary and zero-scientific-output result
are authoritative. Complete factor/product admission remains atomic before
calibration mutation or publication, and a tau/extinction validity bit is not
complete calibration validity.

### F004 — reuse available APT lineage

`SCI-CAL-001-F004` remains open P0. The phrase “matched-source lineage absent”
is superseded as too broad. Existing APT lineage is available upstream but is
not yet fully consumed, validated, and propagated by Citlali.

Legacy matched APT ECSV headers can carry observation identity and
`obsnum_matched`; row metadata can carry `det_id`, `det_id_right`, `uid`,
network, and tone frequency. Modern TolAPT additionally records exact input
path/hash/run provenance in `manifest.yaml` and row association in ECSV
tables, when those details are present.

The bounded repair reuses those existing facts, preserves the selected source
association and applicable validity/eligibility state, and validates either
approved target-row order or an explicit keyed acquisition join. It must not
invent a lineage system, duplicate APT extensions, require perfect
design-detector identity for ordinary measured Beammap calibration quantities,
or claim unavailable optional modern details.

### F005 — existing conditional variance/weight recipients only

`SCI-CAL-001-F005` remains open P1. For every existing production signal
recipient multiplied by a valid calibration factor `a`, its existing
conditional measurement variance and inverse-variance weight transform as

\[
V' = a^2 V,
\qquad
W' = \frac{W}{a^2}.
\]

The same admitted factor, recipient, support, and realized stage must be used.
The repair implements this only for production variance/weight products that
already exist; it creates no new uncertainty product solely for CAL.

Calibration-model, atmosphere, beam/response, donor-target, common-mode,
cross-detector, and other nuisance covariance remain explicitly unavailable
unless separately measured and approved. Metadata distinguishes conditional
measurement variance/weight from total calibrated uncertainty. Conditional
weight must never be represented as total uncertainty or significance.

### F006 — approved mJy/beam configuration boundary only

`SCI-CAL-001-F006` is closed only as an owner scientific-policy decision for
the approved top-of-atmosphere `mJy/beam` point-source configuration boundary.
The repair does not implement or authorize `MJy/sr`, `Jy/pixel`, temperature,
extended-source, or integrated-photometry modes. Unsupported requested units
fail at the F003 configuration boundary.

This narrow contract closure does not establish implementation conformity,
response fidelity, uncertainty completeness, or production authority.

### F007 — package-level reconstructibility

`SCI-CAL-001-F007` remains open P1. Calibration reconstructibility is a
reduction-package contract, not a requirement that every individual file
duplicate all lineage and per-detector data.

Each package contains one canonical calibration-lineage record, a package-local
copy of the exact selected APT with existing ECSV lineage metadata preserved
and its digest recorded, the exact raw-observation identity, the applied
calibration-factor definitions, and stable package/calibration identities and
joins.

Individual FITS, TOD, and Beammap products need only the package/calibration
identity, calibration validity, target unit, and linkage required to resolve
the canonical record. Do not duplicate APT extensions or complete per-detector
factor tables in every product. Missing, stale, conflicting, or ambiguous
joins fail closed.

### F008 — once-only algebra and realized response identity

`SCI-CAL-001-F008` remains open P1. CAL preserves the exact once-only
composition of every applied calibration factor and records each factor's
definition, role, recipient/support, application stage, and exact composed
multiplier.

The canonical lineage also records the precise realized response basis:
selected APT beam identity, mapmaker/kernel class, and relevant filtering
state. This is response-basis provenance, not empirical response validation.
Existing production variance/weight recipients follow F005.

CAL must not claim verified absolute response fidelity, total calibrated
uncertainty, donor-target covariance, or common-mode covariance. Those claims
remain conditioned on later MAP/BEAM contracts and exact-SHA observational
validation and must not broaden the bounded repair.

### F009 — bounded validation lane

`SCI-CAL-001-F009` remains open P1. The exact successor must pass local
engine-level success and failure fixtures, including actual writer/reopen
checks for canonical package provenance and every existing uncertainty/weight
recipient affected by calibration. It also passes focused tests, full CTest,
configuration preflight, and applicable deterministic authority, baseline,
and product-contract gates.

Do not run local science reductions solely to prove code mechanics. Only after
owner integration and push of an accepted successor may the coordinator
prepare a small exact-SHA human-run Unity reduction for operational
confirmation. This record does not request or authorize it.

Astronomical response-fidelity validation remains a later MAP/BEAM
responsibility and is not a bounded CAL implementation gate.

### F010 — conditioned external dependency

`SCI-CAL-001-F010` remains open P1 and conditioned on accepted ALIGN/AST
authority and exact downstream evidence. The bounded repair may implement the
approved abstract identity, time, support, and eligibility interfaces, but it
must not choose or rederive ALIGN/AST science or absorb their evidence work.

CAL may reach local implementation conformance while F010 and the associated
production precision/accuracy claims remain explicitly fail closed.

## Closure accounting and scope split

- Narrowly closed: F002 structural atmosphere behavior; F006 owner-approved
  `mJy/beam` configuration boundary.
- Bounded implementation repair: F003, F004, F005, F007, F008, and the local
  implementation portion of F009.
- Conditioned external dependencies outside repair expansion: F001, F010,
  later exact-SHA Unity operational confirmation, astronomical-standard
  evidence, MAP/BEAM empirical response fidelity, and unavailable nuisance
  covariance.

Local implementation conformity does not imply validation completion,
production approval, or precision/accuracy authority. The current production
state remains `fail_closed` until separately accepted evidence and a fresh
independent re-audit support a successor disposition.

## Non-authorization and stop

This record authorizes only preparation of a documentation-only bounded repair
handoff. It does not authorize application/config/test edits, a repair branch
or worktree, re-audit, Unity request or access, local or external reduction,
evidence execution, downstream work, production change, merge, or push.
