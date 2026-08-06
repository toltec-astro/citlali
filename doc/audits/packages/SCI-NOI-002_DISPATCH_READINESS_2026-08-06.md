# SCI-NOI-002 independent audit dispatch readiness — 2026-08-06

Status: Phase 0 and Phase 1 `completed_accepted`; Phase 2
`authorized_not_started`. The frozen prompt and inbox manifest are unchanged.
No Phase 2 source/test/diff or sealed-XAUD-001 exposure, evidence request,
reduction, Unity action, repair, integration, push, or production change has
occurred.

## Exact starting state and phase profile

The coordinator may create an isolated Phase 0 execution worktree at exact
`d5015fe716971bf8ea617e8a187311bf5af05185` from
`origin/codex/refactor-mainline`; it is infrastructure only. Phase 0 is Terra
High and stops after verifying worktree/HEAD/clean state, prompt and manifest
bytes, and quarantine. The future audit branch
`codex/audit-sci-noi-002` may be created only after explicit Phase 1
authorization, from exact d501 in that clean worktree if the branch is absent.

## Accepted Phase 0 and Phase 1 authorization

Task `019fd4e1-8add-7b52-aa2f-3446e980cd4b` used isolated worktree
`/Users/gwilson/.codex/worktrees/e801/citlali-refactor` at clean detached
exact d501, with `origin/codex/refactor-mainline` resolving to the same SHA.
It verified the frozen prompt, inbox manifest, readiness record, and
SCI-NOI-002-XAUD-002 digests; SCI-NOI-002-XAUD-001 remained sealed and no
prohibited exposure or write occurred. `codex/audit-sci-noi-002` remains
absent and untouched.

The same task/worktree was authorized only for Sol-Ultra Phase 1, serial with
no delegation. It created `codex/audit-sci-noi-002` from exact d501, read only
repository-level authorities plus SCI-NOI-002-XAUD-002/R3, derived and committed
only `doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex`, and stopped before
every quarantined source/test/diff or post-core handoff.

## Completed Phase 1 and Phase 2 boundary

Phase 1 completed on `codex/audit-sci-noi-002` at commit
`f08a6da2ceebff03f498386f374980d13c5146a6`, exact parent d501, with the sole
added file `doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex` (1,031 lines,
SHA-256 `36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d`).
Root structural review reports 50 unique labels and all 12 references resolved;
no TeX-engine compile evidence is available. The clean task trace reports no
source/test/diff or sealed-XAUD-001 exposure before the core freeze.

Phase 2 is now authorized only for the same Sol-Ultra serial/no-delegation
task. It may open SCI-NOI-002-XAUD-001 and exact-d501 source/tests/diffs to map
the implementation against the independent core. It must not inspect repair
commit `38ef72860743636f59d226c9e1ff5ff776d0e9c0` or use repaired behavior as
the audit target; no repair, evidence, Unity, integration, or production action
is authorized.

## Frozen package hashes

| Artifact | SHA-256 |
| --- | --- |
| `doc/audits/prompts/SCI_NOI_002_AUDIT_PROMPT.md` | `74cd002580f1bd92eb6c5030eb8a9fdf19711db7fb6f9359f26ac21245a12220` |
| `doc/audits/handoffs/SCI-NOI-002/SCI-NOI-002_INBOX_MANIFEST_2026-08-06.yaml` | `1de9dbc5f9aca42f5a9f9b1f05b7b14c092ee9b443946d0c52eeb27c8da117b0` |
| `SCI-NOI-002-XAUD-002` pre-core authority | `9eb6c778409d344ce73387f44ac4a5429a89d43b8e768c19bf3da1ed6967c1e5` |
| `SCI-NOI-002-XAUD-001` post-core evidence | `dfcd59e9d59395ba84f7dfed1656690daae694872c2a1a40bf4f5c79f6abed3a` |

Phase 1 is Sol Ultra, serial/no delegation, and may commit only
`doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex` before stopping. Phase
2 is separately held for coordinator acceptance of the frozen core. The Ultra
trigger is the required synthesis of independent finite-N normalization,
covariance/weight, consumer, aperture, and production-count investigations.

## Frozen authority partition

R3 is the only pre-core package authority, via `SCI-NOI-002-XAUD-002`.
`SCI-NOI-002-XAUD-001` is source-derived post-core evidence and stays sealed
with the NOI-001 final audit, repair material/diffs, and all implementation/
test/diff content until the independently derived core is frozen and accepted.
The partition preserves independent estimator reasoning; it does not resolve
any NOI-002 scientific question.

## Exclusive scope and retained boundaries

NOI-002 alone owns finite-N/sample-variance normalization, empirical
variance/covariance and calibrated weight formulas, S/N/significance/product
labels, thresholds/source-finding/feedback, aperture uncertainty, use-specific
ensemble adequacy, and production realization-count/default policy. It may
condition on R3 but cannot repair NOI-001. Count 64 remains optional validation
capacity only. MAP `weight_I` is nonprecision; JINC remains SCI-MAP-002-
conditioned; MAP remains PTC/VAL-conditioned and `existing_use_only`; FLT and
FRUIT remain separate and unopened by this readiness record.
