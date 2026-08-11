# SCI-CAL-001 successor-3 owner acceptance

Date: 2026-08-11

Record ID: `SCI-CAL-001-SUCCESSOR-3-OWNER-ACCEPTANCE-001`

Status: owner accepted `AMEND` / return for bounded technical repair;
successor-4 scope authorized; repair not launched

## Exact identities

- Coordination authority:
  `7d586ad06f136264fcfa7f5cffe3f2041d7438fb` (parent
  `3fe0aa30eaa0d8848dbb39eb720457326c0b43ba`, tree
  `75183134de8ef0eed8979bd0ae3ade1aa73416df`).
- Audited application candidate:
  `3af6faf996fa002b2647adca8f33991002d49ff1` (parent
  `8b1534807f5abe4d80be2fbd45ed3838ed351509`, tree
  `16130eb6deba3f9d8b5a8f1d1fae126084b63c95`) on
  `codex/repair-sci-cal-001-successor-3`.
- Candidate binary-patch SHA-256:
  `4558d541a82b2a1f5c4406825c277a2f7317b6d4c788f4bbaf699a385d471bdf`.
- Independent re-audit:
  `0b597ec5eaaf52477c995af6f2f4fa3eddb2a5de` (parent exact candidate,
  tree `1766cdea7d03fc8259caffe5604dbbcbb28ac038`) on
  `codex/reaudit-sci-cal-001-successor-3-20260811`.

The exact imported audit artifacts are:

| Artifact | SHA-256 |
| --- | --- |
| [Successor-3 re-audit](../SCI-CAL-001_SUCCESSOR_3_REAUDIT_2026-08-11.md) | `ee0f8c40e31300fd5c547b45d086a5e97f7be52e45d453016c0ade28c014e59a` |
| [Local evidence](../evidence/SCI-CAL-001_SUCCESSOR_3_LOCAL_EVIDENCE_2026-08-11.yaml) | `7245872f044fc15a7cdb631ea02b70a9746cba1351bc942c1dde8bffa2b25a6f` |

## Accepted disposition

The owner accepts verdict `amend` and return for a bounded technical repair.
The successor-3 candidate is not complete CAL closure and is not accepted for
integration or production.

The controlled axes remain:

- contract: `approved`;
- implementation: `nonconformant`;
- validation: `in_progress`;
- production: `fail_closed`; and
- verdict: `amend`.

No new scientific decision or algorithmic redesign is required or authorized.
The current finding authority is:

| Finding | Accepted state |
| --- | --- |
| F001 | Open and conditioned, unchanged. |
| F002 | Previously accepted narrow structural closure retained. |
| F003 | Closed and conformant at the approved startup/admission boundary. |
| F004 | Closed and conformant within the accepted structural APT-association claim. |
| F005 | Open; bounded exactly-once recipient proof and truthful inventory only. |
| F006 | Previously accepted `mJy/beam` configuration-boundary closure retained. |
| F007 | Open; bounded canonical identity and package/product joins only. |
| F008 | Open; bounded complete realized response-state identity only. |
| local F009 | Open; bounded v4/package-copy executable-contract synchronization only. |
| F010 | Open and conditioned, unchanged. |

The positive F003/F004 results and prior F002/F006 closures must be preserved.
F001/F010 remain external evidence conditions and must not broaden successor-4.

## Successor-4 authority

The owner authorizes preparation and later role-separated launch of the exact
bounded repair defined by
[`SCI-CAL-001_SUCCESSOR_4_BOUNDED_REPAIR_HANDOFF_2026-08-11.md`](SCI-CAL-001_SUCCESSOR_4_BOUNDED_REPAIR_HANDOFF_2026-08-11.md)
and its machine-readable finding ledger. The proposed branch is
`codex/repair-sci-cal-001-successor-4` from exact pushed base
`3af6faf996fa002b2647adca8f33991002d49ff1`.

This coordination record and handoff do not create a branch or worktree and do
not launch the repair. A separate role-separated task must first pass the
handoff's mandatory READY checkpoint.

## Non-authorization

This acceptance does not merge or accept the candidate, modify application,
configuration, test, build, or validation-product code, change production,
launch repair or re-audit, access Unity, run reductions, contact external
parties, or launch RTC, PTC, MAP, BEAM, TEL, ALIGN, FLT, or other downstream
work. It creates no covariance, empirical response-fidelity, uncertainty, or
scientific-validity claim.
