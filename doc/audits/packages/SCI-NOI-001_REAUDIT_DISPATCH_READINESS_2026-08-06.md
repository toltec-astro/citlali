# SCI-NOI-001 narrow independent re-audit dispatch readiness — 2026-08-06

Status: `completed_with_bounded_validator_closure_blocker`. The frozen prompt
and authority manifest remain immutable dispatch authorities. The independent
re-audit completed at `159a1a8f46e2197c6faee253d40e1ca7242bfb22`; no evidence
execution, Unity action, local reduction, integration, or production change
occurred.

## Repair completion accepted for re-audit

Repair commit `38ef72860743636f59d226c9e1ff5ff776d0e9c0` on
`codex/repair-sci-noi-001` is clean, has exact parent
`d5015fe716971bf8ea617e8a187311bf5af05185`, and is bounded to the approved
F001/F002/F003/F005/F008 noise identity/configuration/provenance/output/test
surfaces. Its reported gates are recorded in the frozen authority manifest;
the root reran seven noise-realization contract tests and two FITS contract
tests successfully. The repair remains unintegrated, all findings remain open
pending this independent re-audit, and production remains `existing_use_only`.

## Frozen entry

The future re-auditor starts in a fresh isolated worktree at exact repair commit
`38ef72860743636f59d226c9e1ff5ff776d0e9c0`, never the repair or prior audit
worktree. Its proposed branch is `codex/reaudit-sci-noi-001`, created only if
absent and clean. The repair parent must verify as exact d501.

The required authority manifest is
`doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001_REAUDIT_AUTHORITY_MANIFEST_2026-08-06.yaml`,
SHA-256 `fbe4cf5c4c783f6bc878ef1037aae7806386294124af1689dd071e04d9abb49e`.
The re-audit prompt is `doc/audits/prompts/SCI_NOI_001_REAUDIT_PROMPT.md`.

## Scope and stop rule

Recommended execution is Sol Ultra, serial/no delegation, because the
deterministic realization key, concurrency, Beammap lifecycle, and provenance
must be reconciled across paths. Scope is only F001/F002/F003/F005/F008; F007
remains a held no-current-evidence policy. F004/F006 and all filter, estimator,
residual, Unity, reduction, integration, and production work remain excluded.
The re-auditor stops after documentation-only report/proposal commit for
coordinator review.

## Live owner clarification: optional Beammap capability

The frozen prompt and authority manifest remain byte-for-byte unchanged. The
live coordination record clarifies that the v4-compatible Beammap
jackknife/noise-map capability is retained only as an explicit opt-in. Standard
Beammap configurations remain disabled by default and then perform zero
realization/noise-product work; any disabled-mode configured count (commonly
10) is inert and resolves to effective count zero, not a requirement, adequacy
threshold, operational expectation, or evidence request. Re-audit F008 only
for deterministic pass/active-map assignment correctness in an explicitly
enabled capability; this authorizes no generation, validation campaign,
default, application change, or production expansion.

## Completed independent disposition

The re-audit closes F001, the bounded `source_imprinted_current` portion of
F003, and explicit-opt-in F008 after coordinator review. F002 and F005 remain
open only because the active baseline auditor and FITS summary/comparison
inventories retain the retired v1/unavailable-disabled-count contract and omit
all nine v2 identity cards. This is a bounded validation-tool compatibility
defect, not a scientific or numerical defect. F004/F006 remain excluded and
open; F007 remains held with no current evidence execution. The successor is a
separate unlaunched validator/tooling amendment from exact repair `38ef`, then
a fresh Sol-Ultra re-audit.
