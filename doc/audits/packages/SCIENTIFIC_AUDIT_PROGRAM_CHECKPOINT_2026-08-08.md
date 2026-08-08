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
| `SCI-MAP-002` | Repair active from exact base `46ad2388...`; no active-task result is admitted here. | Bounded repair return and fresh re-audit. |
| `SCI-ALIGN-001` | Owner-managed ALIGN/3C273 evidence remains active and deliberately slow. | Frozen returned evidence and coordinator review. |
| `SCI-AST-001` | Repair remains held. | Accepted and re-audited ALIGN successor, then exact AST repair-base selection. |
| `SCI-RTC-001` | Audit `3319d742...` integrated; D001--D004 approved; axes remain `proposed` / `nonconformant` / `in_progress` / `existing_use_only`, verdict `amend`. | Upstream closure, separately authorized repair, focused exact-successor evidence, then independent re-audit. |
| `SCI-PTC-001` | Next independent audit at Sol Ultra against exact application SHA `46ad2388...`; packet `70d191f3...` is frozen and not launched. | Owner/coordinator launch of the exact packet; RTC and ALIGN evidence remains post-core and undispositioned. |
| `SCI-VAL-001` | Queued after PTC. | Frozen RTC and PTC interfaces. |
| `SCI-BEAM-001` | Queued after PTC/VAL. | Frozen PTC and VAL interfaces; existing upstream restrictions remain. |
| `SCI-FLT-001` | Successor-repair dispatch readiness only; not authorized. | Exact repair base remains unresolved until CAL has an accepted, re-audited, integrated successor. |
| `ENG-STATE-001` | Static Tier C package accepted into coordination; F001/F002 remain open P1 evidence gaps and all axes remain unchanged. | Fresh static-only scope checkpoint before any further ENG work. |

This queue is not a blanket ban on new audits. New audits remain admissible
when their dependency facts, role separation, frozen manifests, resource
choice, and checkpoint scope are explicit. Repair debt, consumer restrictions,
cost, and concurrent scope remain actively controlled.

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

## PTC launch sequencing conclusion

The completed RTC audit brief's conservative dependency sequence governs
acceptance or closure of RTC's outgoing handoffs and downstream scientific or
production use. It does not prohibit launching a fresh PTC independent audit.
PTC may launch against `46ad2388...` with `SCI-PTC-001-XAUD-001` as its only
pre-core handoff, the approved RTC D001--D004 owner record as pre-core
contract authority, and all observational or implementation records—including
RTC `SCI-PTC-001-XAUD-005` and ALIGN `SCI-PTC-001-XAUD-006`—quarantined until
the independent core is frozen. Launch does not disposition either handoff,
authorize repair, or change production status.

The exact frozen launch identities are prompt
`doc/audits/prompts/SCI_PTC_001_AUDIT_PROMPT.md` at SHA-256
`a34d2444c3f66a2a1056ebe5c552d16a31bd9f69035b5c6f34ddbb9bc90fd24b`
and manifest
`doc/audits/handoffs/SCI-PTC-001/SCI-PTC-001_INBOX_AUTHORITY_MANIFEST_2026-08-08.yaml`
at SHA-256
`23913b37dfac7a106bcb281f9a1870616c99acf912a5a7aad59aed39e6bd67d3`,
both frozen in dispatch-content commit
`70d191f3207d37cbdca3a77392668e93fb68c4fc`. This coordination task did not
launch it.

## Explicit non-authorizations

This checkpoint does not authorize application edits, audit science, repair,
re-audit, broad or costly execution, local Citlali reductions, Unity activity,
external contact, production change, or PTC/BEAM launch. It records only that
a separately frozen PTC independent-audit packet may be owner-launched.
