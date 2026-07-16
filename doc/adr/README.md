# Citlali Architecture Decision Records

This directory contains durable decisions that are consequential, easy to
misunderstand, and expensive to reconstruct from Git history. The current
software map remains [`../ARCHITECTURE.md`](../ARCHITECTURE.md), scientific
semantics remain [`../SCIENTIFIC_CONVENTIONS.md`](../SCIENTIFIC_CONVENTIONS.md),
and phase sequencing remains [`../REFACTOR_STATUS.md`](../REFACTOR_STATUS.md).

An ADR records why a decision exists and what would supersede it. It does not
duplicate changing implementation inventories or validation snapshots.

## Records

| ADR | Status | Decision |
| --- | --- | --- |
| [0001](0001-config-state-transitions.md) | Accepted | Immutable request to effective plan to observation-resolved and realized state, with a one-way legacy adapter |
| [0002](0002-reduction-result-and-required-output-failures.md) | Accepted | Structured reduction result, required-output failure propagation, and CLI-only process exit |
| [0003](0003-session-lifecycle-and-engine-compatibility.md) | Accepted | Sequential session lifecycle and `Engine` as a frozen compatibility boundary |
| [0004](0004-compiled-boundary-and-header-policy.md) | Accepted | Evidence-driven compiled boundaries and header/hot-path policy |
| [0005](0005-defer-measured-r-channel-execution.md) | Accepted | Preserve measured R-channel structure while deferring execution until its contract is approved |

Numbers are never reused. A materially different decision adds a new ADR and
marks the old record superseded; do not rewrite the historical rationale.
