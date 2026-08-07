# SCI-NOI-002 Cycle 4 bounded repair dispatch readiness — 2026-08-07

Status: `ready_pending_coordination_commit_push_and_existing_task_continuation`.
The pushed Cycle 3 independent re-audit is byte-preserved on the coordination
line, the three bounded engineering findings require no new owner decision,
and Cycle 4 repair authority is frozen before implementation. This readiness
record performs no application edit, push, integration, Unity or astronomical
action, production change, canonical-ledger mutation, or re-audit launch.

## Exact repair continuation

Continue the existing repair task and branch `codex/repair-sci-noi-002` from
exact clean application commit
`390edf4f8c696551921c615f2439e956d240ec1d`, parent
`63efd8b08a599d2d56a1716e3cbb2d3686d62b9f`, tree
`a82cdad542494c261d9095105813e157436766c8`. The locally stored
`origin/codex/repair-sci-noi-002` must match. The successor must be a child of
that candidate. Never use audit commit
`b45da53708dcb05e22f284d6a815bab47caefa40`, the audit branch, or this
coordination branch as the application base.

## Frozen artifacts

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/packages/SCI-NOI-002_CYCLE4_COORDINATOR_DISPOSITION_AND_REPAIR_HANDOFF_2026-08-07.md` | `bb8060f704fbc2abccffdf1fb1b5388364568653bed0b82bbbad242f213aec10` |
| `doc/audits/prompts/SCI_NOI_002_CYCLE4_REPAIR_PROMPT.md` | `2beda29726d3da4f13da83e277e26200431536ae11199f27231cf6e7c6896a27` |
| `doc/audits/handoffs/SCI-NOI-002/SCI-NOI-002_CYCLE4_REPAIR_AUTHORITY_MANIFEST_2026-08-07.yaml` | `ed3b1cb2776d484670f21b4fe57d509583701b3f1cb4fc9feb746ba638d43b21` |
| `doc/audits/packages/SCI-NOI-002_CYCLE3_INDEPENDENT_REAUDIT_2026-08-07.md` | `cef029918bc9923d2f20e479a1bfcda02027c658359c186835ceff2643b6a139` |
| `doc/audits/results/SCI-NOI-002_CYCLE3_REAUDIT_RESULT_2026-08-07.yaml` | `e801fb991c146d0af3f522edbbee3dcc45f22975b4a08868546b0df11dbe4ecf` |
| `doc/audits/proposals/SCI-NOI-002_CYCLE3_REAUDIT_LEDGER_UPDATE_PROPOSAL_2026-08-07.yaml` | `f46ee2c6f90bab5f81ba34a1a0c3d5a91badaa1b11031ce1e46dd0f964e13451` |

The containing coordination commit is supplied at dispatch because an
artifact cannot contain the hash of the Git commit that contains itself. The
Cycle 3 re-audit source commit is
`b45da53708dcb05e22f284d6a815bab47caefa40`; its coordination-line
documentation-only integration commit is
`afedbad3a0d7f8829555f84bc8a21974992a5dfc`.

## Authorized outcome

Cycle 4 repairs only:

- exact compact ECSV/NetCDF missingness parity;
- mode-aware successor-coadd NOI identity/cardinality accounting while
  preserving the no-empirical-companion output policy;
- package reconciliation for the existing split detector-group Beammap
  multi-map-per-array-file layout, including exclusion of unused files from
  NOI membership; and
- deterministic production-writer-to-final-package fixtures for those actual
  shapes.

It does not change estimator, coadd, Beammap, mapmaking, filtering, FRUIT, or
other numerical behavior; output selection/order or file partitioning;
configured counts/defaults; scientific claims; or production status.

## Return gate

The task must verify these exact frozen bytes and return a scope checkpoint
before editing. After coordinator continuation it may make only allowed-path
changes, run the frozen local gates, and create one coherent repair commit. It
must stop before push, integration, Unity, astronomical evidence, production
action, canonical-ledger update, or fresh re-audit.

Implementation remains nonconformant and production remains
`existing_use_only` until a clean Cycle 4 repair and fresh independent re-audit
resolve the three P1 findings.
