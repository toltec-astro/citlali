# SCI-NOI-001 bounded repair dispatch readiness — 2026-08-06

Status: `ready_pending_launch`. This is a frozen coordination package only;
no repair worktree, repair branch, source edit, test run, evidence request,
Unity action, re-audit, integration, push, or production change has occurred.

## Exact starting state

The eventual repair task must use a fresh isolated worktree at exact
`d5015fe716971bf8ea617e8a187311bf5af05185` from
`origin/codex/refactor-mainline`, never an audit or coordination branch. Its
proposed branch is `codex/repair-sci-noi-001`; creation is permitted only on a
clean worktree when that branch is absent. The authority baseline is
coordination commit `c21ff272ff09160fb004536504f1820e9dbd08d5`.

## Frozen package hashes

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/prompts/SCI_NOI_001_REPAIR_PROMPT.md` | `83996bd6ae5ffbf609f14354d237d23465f3ecb1ec6a3d161447b162bfeaa024` |
| `doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001_REPAIR_AUTHORITY_MANIFEST_2026-08-06.yaml` | `44c6ac459f058ba39e18d01f35e64bfb37bcfec9c116a12a2839f64beac71bae` |

## Approved boundary

The repair covers only F001, F002, F003, F005, and F008 under owner decisions
D001--D004: deterministic observation/pass/realization identity, compact
reconstruction provenance, truthful `source_imprinted_current` metadata,
enabled-versus-disabled admission/no-work semantics, and Beammap named-pass
active-slot invariance. It preserves ordinary naive-map operator behavior
except intentional sign-identity changes; JINC conclusions stay conditioned on
SCI-MAP-002.

F004, F006, and F007 remain excluded and open. In particular, there is no
filter-edge resolution; finite-N/estimator/weight/S/N/significance/threshold/
feedback/aperture/count policy remains NOI-002; and no astronomical evidence
or residual FRUIT implementation is admitted. The repair changes no production
default or status.

## Required execution profile and return

The recommended task is `gpt-5.6-sol`, Ultra, serial with no delegation. The
written Ultra trigger is the reconciliation of RNG namespace, concurrency
determinism, Beammap lifecycle, and provenance across distinct paths. The
repairer must first return a scope checkpoint, then only after coordinator
continuation implement/test/commit the bounded repair and stop for review.

Focused gates are deterministic sequential/OpenMP/scheduling repeatability,
distinct observation namespaces, Beammap pass/active-map invariance,
positive-count/disabled-zero-work behavior, provenance round trip and required
write failure propagation, focused CTest, config preflight, and proportionate
`citlali_cli` build. No full reduction, Unity, evidence execution, re-audit,
push, or integration is authorized.
