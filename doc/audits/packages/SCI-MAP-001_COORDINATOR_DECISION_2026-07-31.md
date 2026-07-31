# SCI-MAP-001 coordinator integration and repair-base decision — 2026-07-31

## Integrated immutable records

The audit coordinator reviewed and accepted the completed SCI-MAP-001 audit
and its separate PTC/VAL handoff proposal. The source identities remain
immutable and distinct:

- scientific-audit framework and cross-audit protocol:
  `fba342020e5c241fb06320e3c929d4c4bb050a2f`;
- final SCI-MAP-001 audit decision:
  `b9e1e9a9b2fe492c402d8c7b0cf7e5a36c136a53`;
- final audit artifact SHA-256:
  `6c8decef93f5607bc9e8dfc84e31aee67f45fa5c695fc80563c7e7064f78d556`;
- audit-history integration merge:
  `4d75be02ec31b1ee622e9ede7498fb0c1276155f`;
- separate cross-audit proposal:
  `a675c2a54a50ed0c67b077e9c5d420933fa11ab0`; and
- cross-audit proposal integration merge:
  `f230e8bf0a130ea5834aa427e6eacfc318da924b`.

The four accepted outbound records retain `arrival: before_dispatch`,
`status: submitted`, and a pending recipient disposition. Their bytes and
digests match the reviewed proposal. No PTC or VAL audit is dispatched by
this integration, and no recipient disposition is simulated. The coordinator
will freeze a target inbox manifest only when the corresponding audit is
actually dispatched.

## Repair-base decision

The project owner authorized the coordinator to make and record the pending
repair-base decision. The selected application base is:

```text
9aae0e669384c5c0c0dda93debc194d6b8dac787
```

The convolve/noise candidate does **not** land first. In particular,
`02a198cbfb379eaf6ab279c5a3d44ee73ff90435` remains an audited but unapproved
candidate with an `amend` verdict, `fail_closed` production status, and a
required re-audit. It also changes MAP-adjacent files including
`src/citlali/core/mapmaking/map.cpp`, map-image output helpers, scientific
conventions, metadata, and product contracts. Basing SCI-MAP-001 repair on the
audited `9aae0e...` application isolates the repair delta and the later
same-SHA evidence from those unresolved convolve changes.

This decision does not reject or supersede the convolve work. A separately
approved convolve successor may be integrated later through its own repair,
validation, and re-audit gates, followed by compatibility validation against
the accepted MAP successor.

## Authorized next stage and limits

A fresh `codex/repair-sci-map-001` worktree may now be created from the exact
selected application SHA. The repair must follow
`doc/audits/packages/SCI-MAP-001_BOUNDED_REPAIR_REAUDIT_HANDOFF_2026-07-31.md`
and must not be made on the audit or coordination branches.

This coordination record does not create the repair branch, change
application code, close any finding, expand production use, dispatch Unity
work, or satisfy the required fresh re-audit. Until those stages succeed,
implementation remains `nonconformant`, validation remains `in_progress`,
production remains `existing_use_only`, verdict remains `amend`, and re-audit
remains `required`.
