# SCI-CAL-001 successor-5 owner acceptance

Date: 2026-08-12

Record ID: `SCI-CAL-001-SUCCESSOR-5-OWNER-ACCEPTANCE-001`

Status: owner accepted return for bounded successor-6 repair; repair not launched

## Exact accepted identities

The owner accepts the independent successor-5 re-audit at exact commit
`900b7f8b3dac59246f7628a116923de07f678b16` on branch
`codex/reaudit-sci-cal-001-successor-5-20260812`:

- parent: exact pushed candidate
  `5dfc414a13fe69e6b063608906d87e3b30491ec7`;
- tree: `880a632c6f51e919176bc78dc9a1d1a0361d8fa9`;
- documentation patch SHA-256:
  `476f8fb872c9d10ef6f2f040e271c01d6a1788c26248cd287346806bcd3d510c`;
- immutable [report](../SCI-CAL-001_SUCCESSOR_5_REAUDIT_2026-08-12.md),
  SHA-256
  `7f3a484cf5d446647313659a3d6d3103805837ecbb3a9d77034d49bb5762234a`;
- immutable [local evidence](../evidence/SCI-CAL-001_SUCCESSOR_5_LOCAL_EVIDENCE_2026-08-12.yaml),
  SHA-256
  `e6d8d721c02d22683a0ca8500efcd66d1e53c00b85d9529f92bd1c7ccbc64206`.

The accepted candidate is exact pushed commit
`5dfc414a13fe69e6b063608906d87e3b30491ec7` on
`origin/codex/repair-sci-cal-001-successor-5`:

- parent: `693f1b107855e3ae9b36617323ca14aac868f304`;
- tree: `72e4df08bc3677290b03d1c39457ea049f8db813`; and
- parent-to-candidate binary-patch SHA-256:
  `1c9e634c574da60c40cf7e2808b1ec1ac25d1fa8f80cd4de7cb31230365cf7d8`.

## Accepted disposition

The verdict remains `amend`. The controlled axes remain:

- contract: `approved`;
- implementation: `nonconformant`;
- validation: `in_progress`; and
- production: `fail_closed`.

F002, F003, F004, and F006 retain only their previously accepted bounded
closures. F001 and F010 remain open and conditioned without change. F005,
F007, F008, and the local implementation portion of F009 remain open and are
returned for the bounded successor-6 technical repair defined by the
[repair handoff](SCI-CAL-001_SUCCESSOR_6_BOUNDED_REPAIR_HANDOFF_2026-08-12.md)
and [finding ledger](../proposals/SCI-CAL-001_SUCCESSOR_6_REPAIR_FINDING_LEDGER_2026-08-12.yaml).

## Authority-sensitive boundaries

The accepted audit establishes an active v1--v3 product-contract
compatibility contradiction. The bounded repair may not decide it silently.
If accepted legacy behavior cannot coexist with the v4 requirement, the task
must stop for an owner choice among:

1. preserving the legacy epoch;
2. creating a successor contract/profile epoch; or
3. explicitly superseding the affected authority.

Per-artifact atomic publication remains mandatory. The repair may not weaken
that requirement, represent a partially linked artifact as complete, or
replace it with an unauthorized global transaction architecture.

## Non-authorization

This acceptance does not launch successor-6, create its branch or worktree,
close SCI-CAL-001, authorize production, alter F001/F010 dependencies, or
authorize application/configuration/test/build/validation-product edits by
this coordination task. It authorizes no science or arithmetic redesign,
cross-package expansion, Unity access or request, reduction, external
contact, downstream work, re-audit, merge, or push.
