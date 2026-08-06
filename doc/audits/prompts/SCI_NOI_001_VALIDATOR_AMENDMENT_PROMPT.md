# SCI-NOI-001 bounded validator/tooling amendment — frozen dispatch prompt

## Authority and starting state

This is a coordination-authorized tooling amendment only. Start a fresh,
isolated worktree at exact repair commit
`38ef72860743636f59d226c9e1ff5ff776d0e9c0` (parent
`d5015fe716971bf8ea617e8a187311bf5af05185`), never from an audit or
coordination branch. Proposed branch: `codex/amend-sci-noi-001-validator`.
Create it only if absent and only from a clean exact starting worktree.

Read only the frozen authority manifest before implementation. Return a scope
checkpoint before opening or editing a file. The task is `gpt-5.6-terra`, High,
serial, with no delegation or subagents. Implement, test, commit the bounded
tooling amendment, and stop for coordinator review. Do not push.

## Sole purpose and allowed paths

Resolve the re-audit validation-boundary blocker for SCI-NOI-001 F002/F005:
the repaired application emits compact provenance v2, versioned deterministic
identity, and disabled available-zero state, but the active baseline tooling
only admits retired v1 semantics and its standard FITS inventories omit the
nine new identity cards.

The permitted surface is limited to existing baseline audit tooling and its
tests, plus the existing FITS summary/comparison identity inventories:

- `tools/baseline/audit_reduction_run.py` and directly associated tests;
- the existing output-summary and product-comparison tools and directly
  associated tests that own their FITS header-key inventories; and
- existing documentation/provenance fixtures strictly needed to exercise those
  same v1/v2 and enabled/disabled contracts.

Admit and semantically validate
`citlali-noise-products-provenance-v2` while preserving every intentionally
supported v1 lane. Require compact v2 policy/version, stable partition/order,
completed realization IDs, ensemble mode, assignment/reconstruction and
product digest joins. Require the approved disabled available-zero state,
suppressed promises, and zero-work representation. Retain and compare all nine
new realization identity FITS cards in the existing summary/comparison
inventories. Do not add a new framework, schema, wrapper, verifier, or tool.

## Explicit exclusions

Do not modify Citlali application source, application tests, runtime
configuration, algorithms, count/default policy, realization generation,
metadata producer, MAP/JINC/FLT/FRUIT/RTC/PTC behavior, or production status.
Do not run a reduction, contact or use Unity, request or execute evidence,
launch a re-audit, integrate an application branch, or infer a scientific,
numerical, or production conclusion. F001/F003/F008 are re-audit-closed only in
their bounded scope; F004/F006/F007 are outside this task.

## Required gates and return

Add focused v1-compatibility and v2 admission/semantic fixtures, including
enabled and disabled available-zero cases and all nine FITS identity cards.
Run proportionate existing tooling tests, the existing summary/comparison
tests, relevant FITS tests, config preflight, and any focused CTest/build gate
needed by touched files. Report exact paths, commit, digest-bound changed
artifacts, gate results, intentional v1 compatibility, and clean state.
Stop before re-audit. Sol Ultra is reserved for the later fresh narrow re-audit;
Terra High is sufficient for this bounded tooling compatibility work.
