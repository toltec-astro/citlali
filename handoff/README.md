# Citlali Handoff Notes

This directory holds persistent handoff notes for Citlali work. The intent is
to make work portable between machines and Codex instances without relying on
chat history.

When editing Citlali, add or append to the current day's handoff note before
ending the session.

Use this filename pattern:

```text
HANDOFF_YYYY-MM-DD.md
```

Each entry should start with the local timestamp and include, as relevant:

- current branch and commit state,
- files changed,
- tests or Unity jobs running,
- important decisions,
- next steps,
- open questions.

For the current accepted baseline, start with [current state](../doc/CURRENT_STATE.md)
and the [workspace map](../doc/WORKSPACE_MAP.md). Handoffs below preserve dated
continuity; they do not override those entry points or later owner acceptance.

The scientific-contract program manager transition recorded on 2026-09-04 is
[`HANDOFF_2026-09-04_SCIENTIFIC_CONTRACT_PROGRAM.md`](HANDOFF_2026-09-04_SCIENTIFIC_CONTRACT_PROGRAM.md).
Its companion JSON manifest and verifier bind the exact repository snapshot,
the pending conformance-review candidate, and the protected independent FRUIT
and historical ALIGN states.
