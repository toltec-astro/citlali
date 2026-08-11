# SCI-RTC-001 learned-sampling Stage A successor-2 owner acceptance

Date: 2026-08-11

Record ID: `SCI-RTC-001-STAGE-A-SUCCESSOR-2-OWNER-ACCEPTANCE-001`

Status: owner accepted `RETURN FOR REPAIR`; successor-3 repair handoff
authorized; repair not launched

## Exact identities

- Coordination authority:
  `3fe0aa30eaa0d8848dbb39eb720457326c0b43ba` (parent
  `3132d5d8c001ef32f185d4ece2038aa6d7ce1b5c`, tree
  `d753f3ed7445e5a54ae792583c50af08aa88c7ae`).
- Audited application candidate:
  `cbb2fd767e0676906d1413ae84022270bee1a667` (parent
  `66c96757164af2c83ee1449d00fea30d131a7e3f`, tree
  `4727864c7ca4f078649fcf6473a7225d5d3aa9f8`).
- Candidate binary-patch SHA-256:
  `d1521fbc0a5afdfcfa61b41c57ba483b1d69969a45115829d2f8d973a51c9c39`.
- Independent re-audit:
  `f935f25c48b80861eb81a61133b20fc8e4fa4cf0` (parent exact candidate,
  tree `be21aff93642a6a419bf6a1d1b21f8ceff015257`) on
  `codex/reaudit-rtc-learned-sampling-stage-a-successor-2-20260811`.

The exact imported audit artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| [`RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_2026-08-11.md`](RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_2026-08-11.md) | `6ad44cd732215041294e6514a1900daac64d7713543b655e625d450b535eec42` |
| [`RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_EVIDENCE_2026-08-11.md`](RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_EVIDENCE_2026-08-11.md) | `8e01ed71e8e043888efaec0e81fe22477928e8983b7fef2538a0c1ead1f36bfe` |
| [`RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_FINDINGS_2026-08-11.csv`](RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_FINDINGS_2026-08-11.csv) | `095a731124974d8a53c298140073d4c48ac2e1a6f30df81b133260f04d5f89c4` |

## Accepted disposition

The owner accepts the independent disposition **RETURN FOR REPAIR**. The
successor-2 candidate is not accepted and closes none of SRA-001 through
SRA-009. The finding register contains exactly 15 open records,
`S2RA-001` through `S2RA-015`: 11 P1 and four P2.

The audit found technical departures from already frozen authority. It
requires no new scientific decision. Existing positive controls and all prior
Stage A scientific/product decisions remain binding unless an exact successor
record says otherwise.

The owner authorizes a tightly bounded successor-3 handoff and a subsequent
role-separated repair launch against exact candidate `cbb2fd767e...`. This
record and the handoff do not create or launch that repair task.

## Program separation and sequence

Learned-sampling Stage A remains distinct from the phase-independent core RTC
repair. The core line retains its own authority, including the separate
phase-zero downsampling amendment recorded on 2026-08-11. No Stage A finding
may broaden or silently amend core RTC science.

The application integration sequence remains MAP followed by accepted CAL
consolidation. Stage A remains separate from that sequence until an exact
successor is independently re-audited and accepted.

The overall `SCI-RTC-001` axes remain `proposed`, `nonconformant`,
`in_progress`, and `existing_use_only`, with verdict `amend`.

## Non-authorization

This acceptance does not merge or accept the application candidate, change
production status, authorize Stage B, launch repair or re-audit, access Unity,
run reductions, contact external parties, launch downstream work, or alter
PTC D001--D006, MAP, TEL, ALIGN, CAL, or prior RTC authority.
