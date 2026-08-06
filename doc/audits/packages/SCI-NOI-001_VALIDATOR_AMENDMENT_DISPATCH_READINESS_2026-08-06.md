# SCI-NOI-001 validator/tooling amendment dispatch readiness — 2026-08-06

Status: `ready_pending_launch`; no task, worktree, branch, edit, evidence,
re-audit, integration, push, or production action is authorized by this record.

The completed independent re-audit at `159a1a8f46e2197c6faee253d40e1ca7242bfb22`
accepts repair `38ef72860743636f59d226c9e1ff5ff776d0e9c0` for F001, bounded
current-mode F003, and explicit-opt-in F008. F002 and F005 remain open only at
the required validation boundary: the active baseline auditor is v1-only and
rejects v2 deterministic identity/available-zero disabled records, while its
standard FITS summary/comparison inventories omit all nine identity cards.

The fresh proposed tooling branch is `codex/amend-sci-noi-001-validator`, from
exact repair SHA `38ef72860743636f59d226c9e1ff5ff776d0e9c0` only. Its frozen
[prompt](../prompts/SCI_NOI_001_VALIDATOR_AMENDMENT_PROMPT.md) and
[authority manifest](../handoffs/SCI-NOI-001/SCI-NOI-001_VALIDATOR_AMENDMENT_AUTHORITY_MANIFEST_2026-08-06.yaml)
permit existing baseline auditor/tests and existing FITS summary/comparison
inventories only. The required result admits/validates v2 while preserving
intentionally supported v1, validates compact joins and available-zero
disabled state, and retains/compares the nine cards. It may not change Citlali
algorithms, configuration/defaults, evidence state, Unity state, or production.

Recommended implementation resource profile is Terra High, serial with no
delegation, because this is bounded tooling compatibility. The eventual fresh
narrow re-audit remains Sol Ultra and is separately authorized only after this
amendment returns clean with its local gates.

Standard Beammap remains disabled, effective zero, and no-work by default; the
optional v4-compatible capability is relevant only to the already bounded F008
explicit-opt-in lifecycle and is not an evidence or count-adequacy request.
