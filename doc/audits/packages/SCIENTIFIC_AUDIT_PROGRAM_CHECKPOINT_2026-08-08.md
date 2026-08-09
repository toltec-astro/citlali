# Scientific-audit program checkpoint — 2026-08-08

Record ID: `SCIENTIFIC-AUDIT-PROGRAM-CHECKPOINT-2026-08-08`

Status: current coordination record; documentation and queue control only

## Exact authority

- Coordination entry was clean at `codex/scientific-audit-framework` commit
  `dd5894679bf12bf4a5fb551e871b3c6010ef9b9b`, parent
  `44d9765e1d93f9216299e8ace17bcd531d8696ba`, tree
  `e87b507a6dc5246da0f65e563d96b94824e61ba1`.
- Pushed canonical application mainline is
  `origin/codex/refactor-mainline` at
  `46ad23888a40f5102cdfd50c06e49a549bdf8a20`, parent
  `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, tree
  `ab230a93b8fb310d58aefd7ac5da92e5d5e0f408`.
- `SCI-MAP-001` is integrated only for accepted application source
  `af0c849ce59a5f80e5efc8db435bb6662863052f` and the bounded axes and
  limitations recorded on application mainline. Production remains
  `existing_use_only`; F013 remains conditioned on ALIGN, CAL, AST, PTC, and
  VAL.
- `SCI-NOI-002` is integrated only for accepted application source
  `5b29e13548a6fec884c67b192dec20c92f0bbb62` and the bounded axes and
  limitations recorded on application mainline. Production remains
  `existing_use_only`; F005/RA-B004 remain conditioned under FLT and F006
  remains held under FRUIT.

No broader MAP, uncertainty, significance, covariance, consumer, production,
or dependency claim is inferred from those integrations.

## Active and queued work

| Lane | Current bounded state | Next gate |
| --- | --- | --- |
| `SCI-CAL-001` | Repair active from exact base `46ad2388...`; no active-task result is admitted here. | Accepted, independently re-audited, and integrated successor. |
| `SCI-MAP-002` | Third successor `86f1582...` independently re-audited at `a70424a...` and owner-accepted with axes `approved` / `conformant` / `complete` / `existing_use_only`, verdict `accept`; bounded findings only are closed and no merge is authorized. | Separate owner decision for any application integration or production change; BEAM and downstream work remain held. |
| `SCI-ALIGN-001` | Owner-managed ALIGN/3C273 evidence remains active and deliberately slow. | Frozen returned evidence and coordinator review. |
| `SCI-TEL-INPUT-001` | Audit `518e0ccb...` owner-accepted with axes `proposed` / `nonconformant` / `bounded_incomplete` / `existing_use_only`, verdict `amend`; F001--F010 remain open and TolTECA remains read-only. | Separate later owner decision on repair scope and ownership; no repair or re-audit is authorized. |
| `SCI-AST-001` | Repair remains held. | Accepted and re-audited ALIGN successor, then exact AST repair-base selection. |
| `SCI-RTC-001` | Audit `3319d742...` integrated; D001--D004 approved; axes remain `proposed` / `nonconformant` / `in_progress` / `existing_use_only`, verdict `amend`. | Upstream closure, separately authorized repair, focused exact-successor evidence, then independent re-audit. |
| `SCI-PTC-001` | Audit `01ee2474...` integrated; axes are `proposed` / `nonconformant` / `in_progress` / `existing_use_only`, verdict `amend`; D001--D005 approved and D006 awaits owner choice. | Complete the remaining owner decision; no repair, optional transfer characterization, or re-audit is launched. |
| `SCI-VAL-001` | Not launched; new PTC handoff `SCI-VAL-001-XAUD-008` is pending recipient review. | Later explicit sequencing decision after PTC owner review; current restrictions remain. |
| `SCI-BEAM-001` | Queued after PTC/VAL. | Frozen PTC and VAL interfaces; existing upstream restrictions remain. |
| `SCI-FLT-001` | Successor-repair dispatch readiness only; not authorized. | Exact repair base remains unresolved until CAL has an accepted, re-audited, integrated successor. |
| `ENG-STATE-001` | Static Tier C package accepted into coordination; F001/F002 remain open P1 evidence gaps and all axes remain unchanged. | Fresh static-only scope checkpoint before any further ENG work. |

This queue is not a blanket ban on new audits. New audits remain admissible
when their dependency facts, role separation, frozen manifests, resource
choice, and checkpoint scope are explicit. Repair debt, consumer restrictions,
cost, and concurrent scope remain actively controlled.

`SCI-TEL-INPUT-001` was registered by
`doc/audits/packages/SCI-TEL-INPUT-001_PRODUCT_REGISTRATION_2026-08-08.md`.
Its paired planning sources are Citlali
`46ad23888a40f5102cdfd50c06e49a549bdf8a20` and operational TolTECA
`origin/main` commit `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`.
The completed audit preserved ALIGN observations as post-core evidence and
did not admit them as physical timing authority.

## Accepted SCI-MAP-002 third-successor disposition — 2026-08-09

The owner accepted application candidate
`86f1582fad92bdd0453bca3264ce39478b00c227` (parent
`02f443bfeb85f3b2e12a6eff60f3a77e77fe342c`, tree
`f655e96daa578bd77c9b16528c3aaadf882ee80d`) after independent re-audit
commit `a70424a69365d7ed20fb39c45bc6334cc9e7bafe` (parent equal to the
candidate, tree `35ff97e9d2a615d85145249f5782fc16fcb62e65`). The exact imported
[re-audit report](SCI-MAP-002_THIRD_SUCCESSOR_REAUDIT_2026-08-09.md) has
SHA-256
`fe07901504ad26916cab2c5589f452b1873c11ef215239653ed938f45eefd4d5`;
the [owner acceptance](SCI-MAP-002_THIRD_SUCCESSOR_OWNER_ACCEPTANCE_2026-08-09.md)
records the canonical disposition.

RA-001, RA-003, and RA-004 are closed and preserved; RA-002 and RA-005 are
closed. Contract is `approved`, implementation `conformant`, validation
`complete`, production `existing_use_only`, and verdict `accept`. This closes
only the bounded SCI-MAP-002 repair findings. It does not merge or push the
candidate, expand production, authorize Unity or reductions, launch BEAM or
other downstream work, or authorize an external campaign.

## ALIGN-deferred compatibility amendment — 2026-08-09

Owner policy `ALIGN-ASSIGNED-TIME-COMPAT-001` adopts the existing assigned-time
grid only as an exact identified compatibility interface. Every affected
contract/product binds that ALIGN/assigned-grid identity and records
`physical_event_semantics: unavailable` or equivalent. No half-/whole-sample
correction, physical centroid, detector-time absolute oracle, absolute timing,
sub-sample placement, timing-sensitive source-mask fidelity, or correction is
approved. When producer authority returns, only materially timing-sensitive
seams and consumers are re-audited.

Queue consequences are bounded:

- CAL and MAP continue only under their already approved scopes.
- The High-effort `SCI-TEL-INPUT-001` structural audit completed and is
  owner-accepted. ALIGN `92cfa670...` and `08f0a673...` remain post-core
  evidence and physical event meaning remains an unavailable dependency.
- A phase-independent RTC repair handoff is ready against proposed application
  base `46ad2388...` under D001--D004. It covers eligibility/influence,
  complete response parity on the assigned grid, filter/edge/support, stage
  identity, provenance, and local production tests. RTC repair is not launched.
- AST coordinate mathematics and PTC internal-estimator work may proceed
  against the identified assigned grid under a later separate launch, while
  physical timing and absolute placement remain unavailable. This checkpoint
  launches neither.
- BEAM remains held.

The owner-mediated producer-authority evidence request remains read-only and
pending return. `08f0a673...` is not physical integration-event
identification and supplies no timing correction or prior.

## Completed TEL-INPUT audit disposition

The owner accepted the immutable structural audit at final commit
`518e0ccb2fa3b54fa99212a05d33286506b59f80`, parent/core
`e8f43c678fa001d6369d4ebd45985bb820e129b3`, and tree
`d67cc7308f6c24a3c10666fb361a123f9eec5b83`. The accepted axes are contract
`proposed`, implementation `nonconformant`, validation `bounded_incomplete`,
production `existing_use_only`, and verdict `amend`. Findings F001--F010
remain open.

The exact owner acceptance is
`doc/audits/packages/SCI-TEL-INPUT-001_OWNER_ACCEPTANCE_2026-08-09.md`.
`physical_event_semantics` remains unavailable, and all Tier-A stops for
coordinate alias authority/remedy, ALIGN interpolation response, event
meaning, physical displacement/timing correction, uncertainty, and
astrometric response remain in force. Outbound handoffs
`SCI-ALIGN-001-XAUD-001`, `SCI-AST-001-XAUD-002`,
`ENG-STATE-001-XAUD-001`, and `SCI-VAL-001-XAUD-010` are bounded inputs
pending recipient review and launch no recipient work.

No repair scope or ownership is selected. TolTECA remains read-only, and any
future TolTECA modification requires separate maintainer opt-in.

Remote preservation verified on 2026-08-09:

- `origin/codex/repair-sci-cal-001-successor-2` is exactly
  `8b1534807f5abe4d80be2fbd45ed3838ed351509` (parent
  `7894346a91fa78ceb2a8b3d625335f466e5e1756`, tree
  `ae205d935454b869412b214f34744224a31f8e7b`). This is the already completed
  and re-audited CAL successor-2 candidate. The ref preservation is not a new
  post-decision successor, audit launch, closure, integration, or production
  authority.
- `origin/codex/sci-align-001-producer-event-semantics-request` is exactly
  `1d682ee78ca5d85bd30673783a978265bd01048c` (parent
  `08f0a6733d1cb523ae78ccf9348ac6832b834e52`, tree
  `58390255aff30492cd58c753ded87cc03130a485`). This preserves the authorized
  producer-authority request remotely; it is not returned producer evidence,
  physical-state identification, a correction, or acceptance.

## Exact post-core PTC/VAL routing

Queue ID: `ALIGN-3C273-PTC-VAL-POSTCORE-2026-08-08`

Source task: `019fc822-9f45-72b1-91e9-775e80768d2a`

Status: canonical PTC handoff integrated; validation and recipient disposition pending

The ALIGN/3C273 task froze repair commit
`5c6309125fef15f7c98e70a62b591c663944b130`, parent
`2ee6d5116a1acd6230694715d0d1d39bac2f9a77`, tree
`247c04b4cd5f76005dc07fa63480008568c51ea4`. Its handoff object is
`5c6309125fef15f7c98e70a62b591c663944b130:handoff/SCI_ALIGN_001_PTC_SCAN_METADATA_DEFECT_2026-08-08.md`,
SHA-256
`958d4494de801a08361945d046c92daec4b94e907c8317ad506cb1653c026553`.
The independently verified local bundle
`/Users/gwilson/GitHub/citlali-refactor/sci_align_001_ptc_scan_metadata_from_2ee6_5c630912.bundle`
has SHA-256
`b73aff50c7aeb14fd9af5b2e3055a92a3b2d2923443ce32de8d5f7f5c78686a5`.

The bounded root cause is an engineering metadata defect: processed PTC
variable-length chunks were appended correctly, but the scan-metadata path
repeatedly shifted a prior 606-sample window. The repair derives bounds from
the pre-append sample count and actual appended chunk length. Focused local
checks are reported passing, while the named owner-run Unity validation
remains unsupplied. This record does not establish integration, external
validation, scientific effect, or production authorization.

The object is routed to `SCI-PTC-001` as canonical handoff
`SCI-PTC-001-XAUD-006`, with current canonical SHA-256
`23ae2b52d0347006c9fc8362299c6ea9eae98b803e3a7698d2e2fb48ae336a7f`.
It remains queued for later VAL routing. It is absent from the frozen RTC
manifest, changes no RTC derivation or scope, and cannot be opened before a
recipient's independent core is frozen.

## Completed PTC audit disposition

PTC audit commit `01ee247461d6c19bc4db81ccac4fec21af162c88`, parent/core
`66e8d6f98c3e22da74de4eea84e568a0b4cc6310`, and tree
`e6685c920ff37f1d4e51d27ecf23b73ac16087b5` were independently verified and
integrated. The audit preserved the frozen pre/post-core partition and did not
promote RTC or ALIGN evidence.

The exact frozen launch identities are prompt
`doc/audits/prompts/SCI_PTC_001_AUDIT_PROMPT.md` at SHA-256
`a34d2444c3f66a2a1056ebe5c552d16a31bd9f69035b5c6f34ddbb9bc90fd24b`
and manifest
`doc/audits/handoffs/SCI-PTC-001/SCI-PTC-001_INBOX_AUTHORITY_MANIFEST_2026-08-08.yaml`
at SHA-256
`23913b37dfac7a106bcb281f9a1870616c99acf912a5a7aad59aed39e6bd67d3`,
both frozen in dispatch-content commit
`70d191f3207d37cbdca3a77392668e93fb68c4fc`. The completed audit does not
authorize repair, re-audit, or downstream launch.

The coordinator accepted verdict `amend` and the four status axes without
relaxation. D001 and D002 remain approved under their prior exact records. The
owner approved D003 with the successor amendment
`doc/audits/packages/SCI-PTC-001_D003_OWNER_AMENDMENT_2026-08-09.md` and D004
with
`doc/audits/packages/SCI-PTC-001_D004_OWNER_AMENDMENT_2026-08-09.md`. The owner
rejected and replaced the original D005 proposal with
`doc/audits/packages/SCI-PTC-001_D005_OWNER_REPLACEMENT_2026-08-09.md`. D006
remains ready for owner review in
`doc/audits/packages/SCI-PTC-001_COORDINATOR_DECISION_BRIEF_2026-08-08.md`.

D003 identifies the current stored kernel as the collaboration's **estimated
map-center point-source response** of the instrument after the declared
RTC/PTC/analysis chain. It remains conditioned on exact band,
analysis/configuration, detector/mask/selection, upstream, parent, and
realization state, with its calibration, validation, domain, and uncertainty
status stated honestly. It does not alone establish off-center, spatially
varying, extended-source, arbitrary morphology/amplitude, cross-band, or
cross-mode response. Expected band/mode differences are explicit.

The optional plan at
`doc/audits/packages/SCI-PTC-001_OPTIONAL_TRANSFER_CHARACTERIZATION_PLAN_2026-08-09.md`
would extend the current estimate across position, scale/morphology,
amplitude, band/mode, and realization uncertainty. It is neither an ordinary
F005 repair gate nor authorized evidence work. A consumer requesting a
stronger measured claim requires separately authorized evidence for the exact
declared domain.

D004 preserves existing detector-weight families as scalar analysis and
gridding coefficients. Each family must publish its identity, units,
normalization scope, lifecycle, and applied factors. Formal precision,
inverse-variance, significance, and independent-noise interpretations require
independent proof of the complete stronger conditions. Full covariance is not
mandatory, but unavailable covariance must be explicit. Existing
coefficient-weighted mapmaking remains permitted under these semantics and the
unchanged `existing_use_only` production boundary.

D005 identifies the PTC-transformed timestream in memory as the authoritative
intermediate at the PTC-to-mapmaking interface, not an independent sky
estimator. Final science products record material RTC/PTC processing state.
Persisted PTC timestreams declare either diagnostic or requested-derived role;
persistence alone grants no science-product status. Diagnostic identity must
prevent ambiguity but need not provide archival replay. Requested-derived
products add parent and consumer-required material-state binding. Exhaustive
state serialization is required only by a declared consumer or explicit
reproducibility claim. Required-output failures propagate, declared diagnostic
best effort is permitted, and no partial artifact may be represented as
complete.

Outgoing `SCI-VAL-001-XAUD-008`, `SCI-MAP-001-XAUD-004`, and
`SCI-NOI-001-XAUD-001` are registered as submitted and pending recipient
review. No recipient audit was launched.

## Explicit non-authorizations

This checkpoint does not authorize application or TolTECA edits, audit science,
repair, re-audit, broad or costly execution, local Citlali reductions, Unity
activity, coordinator external contact, production change, or RTC/AST/PTC/
BEAM launch, optional PTC transfer characterization, or TEL-INPUT
repair/re-audit. The separately recorded
owner-mediated ALIGN producer request is the sole evidence-acquisition
authority. Every frozen packet or repair handoff still requires its named
separate owner launch.
