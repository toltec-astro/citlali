# Repository Ref And Worktree Inventory - 2026-08-30

## Scope And Reference Point

This is the post-integration, pre-retirement census for application mainline
`f0f423827ab321640e0cbcb003f7bf015368f694`. The local
`codex/refactor-mainline` branch and `origin/codex/refactor-mainline` were equal
at that commit and the current worktree was clean. The remote-tracking census
uses the locally recorded `origin/*` refs; it does not query or mutate Unity.

The inventory supersedes the cleanup classifications in the dated
[`2026-08-26 census`](REPOSITORY_REF_WORKTREE_INVENTORY_2026-08-26.md) without
rewriting that historical evidence. No tag, stash, divergent branch,
Codex-managed ref, reachable file, or user-owned untracked file is a deletion
target in this cycle.

## Integrated Application State

The final integration history preserves four exact operational identities:

| Tag | Peeled commit | Meaning |
| --- | --- | --- |
| `stage7-ngc4449-152390-3ebc2a67f` | `3ebc2a67fc32bad69759ff45638484efabf91773` | Exact Unity Stage 7 science identity |
| `wp7-timestream-integration-20260826` | `a36abaebfb82d503b113de0cf4c1c6e0f6dcffc3` | First WP-7 integration candidate |
| `v2-ngc4449-memory-repair-187df04b` | `187df04b21e942701cf41e6d9c50883922fd65aa` | Exact Unity-tested memory-repair identity; not a successor science baseline |
| `wp7-native-memory-integration-20260830` | `f0f423827ab321640e0cbcb003f7bf015368f694` | Integrated native-memory repair and acceptance record |

The source-disposition cleanup is complete. The 13 census-selected obsolete
sources, stale commented CMake entries, and obsolete audit allowlist entries
are absent. Header-defined implementations remain reachable and retained.

## Ref Census Before This Cleanup

| Ref family | Count | Relation to `f0f423827` | Disposition |
| --- | ---: | --- | --- |
| Local branches | 133 | 1 equal, 26 ancestors, 106 divergent | Retire only the guarded 21-branch set below |
| `origin/*` refs | 112 including symbolic `origin/HEAD` | 1 equal, 32 ancestors, 78 divergent among 111 direct refs | User may retire the guarded 19-branch remote set; retain the rest |
| Tags | 13 | Nine release/scientific-history tags plus four operational tags above | Retain all |
| `refs/codex/snapshots` | 74 | One snapshot-only tip remains | Codex-managed; do not remove manually |
| `refs/codex/turn-diffs` | 22 | Codex-managed checkpoints | Do not remove manually |
| Stashes | 3 | User work outside ordinary branch reachability | Retain all |
| Registered worktrees | 26 | 1 stale administrative entry; 25 live paths | Prune only the stale entry in this cycle |

Local/remote name topology is 81 identical tips, eight shared names with
different tips, 44 local-only names, and 22 remote-only names. The shared names
with different tips are:

| Branch | Local | Remote |
| --- | --- | --- |
| `codex/build-adaptation` | `d9843e85ed87ba9ac8c42d8cc21f997dacbe1046` | `4f9c7e55ab41c2d7ca88ffee86bc9e60e8c5727b` |
| `codex/coherent-iq-sidecar-validation` | `9e739db6d17410497aceaf33a4fd22e9d62ff793` | `762e6c1f131ba53392646424681f60dafe66bf29` |
| `codex/fruit-loop-calibration-reference` | `b02fef613cc7e632828ec762fced6a428906c502` | `f70701ad488444f3e2528c6bbe3e798863c9e301` |
| `codex/integrate-sci-noi-002` | `46ad23888a40f5102cdfd50c06e49a549bdf8a20` | `4846fa4db39bd2f7d4ddc41f693836834cbc5ff4` |
| `codex/scientific-audit-framework` | `dd5894679bf12bf4a5fb551e871b3c6010ef9b9b` | `192e0d9b5e3be4eb20522d3319cae346168c4bce` |
| `codex/scientific-contract-library` | `54475956f6aefb839d43b2f0fb019a142cb64310` | `5f206cf46bb2868aadb00f37dbbbc3944ac4ec8c` |
| `codex/structural-refactor` | `17148719663dd9f6aca1234f78ac4877f8535507` | `c495c3a4840af50072ed6ebdcd297045a5739fe1` |
| `v4.x` | `bbf14b57f22e9a1e30f2c156f66b199d64d52f95` | `61dfdc0492ef56bc88e572e295369f5b63f7d91d` |

Except for the completed `integrate-sci-noi-002` integration lane, these
different-tip names are retained. In particular, the clean attached local
`build-adaptation` worktree is not retired merely because its older remote tip
is now contained by mainline.

## Completed Local Retirement Set

Following explicit owner confirmation, the following 21 branches were deleted
locally. Each was proven to be an ancestor of both `f0f423827` and
administrative descendant `9ef0b52d6`, each had no live attached worktree, and
none lost commit reachability. `repair-native-consumer-extinction` is
additionally pinned by the exact Stage 7 tag.

```text
codex/converge-apt-align-jinc
codex/fruitloop-feedback-estimator
codex/integrate-sci-map-001
codex/integrate-sci-map-002
codex/integrate-sci-noi-002
codex/issue-observation-apt-v2
codex/repair-apt-prod-001-canonical-baseline-v1
codex/repair-apt-prod-002-observation-contract
codex/repair-apt-prod-003-compact-v2
codex/repair-apt-v2-typed-null-policy
codex/repair-build-git-provenance-refresh
codex/repair-native-consumer-extinction
codex/repair-native-explicit-mjd-pointing-support
codex/repair-sci-align-native-cohorts
codex/repair-sci-map-001
codex/repair-sci-map-002
codex/repair-sci-map-002-successor
codex/repair-sci-map-002-successor-2
codex/repair-sci-map-002-successor-3
codex/repair-sci-noi-002
codex/restore-legacy-apt-admission
```

The guarded deletion completed with 112 local branches remaining: one current
mainline, five retained ancestors, and 106 divergent branches. The five
retained ancestor branches are the two attached lanes and three intentional
historical/forensic pointers listed below.

Retain these other merged local refs:

- `codex/refactor-mainline` is the integrated authority;
- `codex/build-adaptation` has a clean attached worktree and a different local
  and remote tip;
- `codex/integrate-apt-v2-sci-align-native` has an attached clean Codex
  worktree;
- `codex/fruit-loop-calibration-reference`, `codex/structural-refactor`, and
  `v4.x` are intentional historical/forensic pointers.

## User-Controlled Remote Retirement Set

Nineteen of the local retirement names also have contained remote refs. The
user may remove exactly these after verifying that remote mainline equals the
current local mainline and still contains the validated integration identity:

```bash
validated=f0f423827ab321640e0cbcb003f7bf015368f694
local_mainline="$(git rev-parse refs/heads/codex/refactor-mainline)"
remote_mainline="$(git ls-remote --heads origin refs/heads/codex/refactor-mainline | awk '{print $1}')"
test "$remote_mainline" = "$local_mainline"
git merge-base --is-ancestor "$validated" "$remote_mainline"

git push origin --delete \
  codex/converge-apt-align-jinc \
  codex/integrate-sci-map-001 \
  codex/integrate-sci-map-002 \
  codex/integrate-sci-noi-002 \
  codex/issue-observation-apt-v2 \
  codex/repair-apt-prod-001-canonical-baseline-v1 \
  codex/repair-apt-prod-002-observation-contract \
  codex/repair-apt-prod-003-compact-v2 \
  codex/repair-apt-v2-typed-null-policy \
  codex/repair-native-consumer-extinction \
  codex/repair-native-explicit-mjd-pointing-support \
  codex/repair-sci-align-native-cohorts \
  codex/repair-sci-map-001 \
  codex/repair-sci-map-002 \
  codex/repair-sci-map-002-successor \
  codex/repair-sci-map-002-successor-2 \
  codex/repair-sci-map-002-successor-3 \
  codex/repair-sci-noi-002 \
  codex/restore-legacy-apt-admission

git fetch --prune origin
```

`fruitloop-feedback-estimator` and `repair-build-git-provenance-refresh` are
local-only. The other merged remote refs are retained release lines,
validated-run authorities, differently tipped shared branches, or branches
with attached worktrees.

## Divergent Retained Lanes

All 106 divergent local branches and all 78 divergent direct remote refs are
excluded from mechanical cleanup. They preserve ongoing or forensic work in
these families:

- scientific-contract, audit, re-audit, registration, and freeze lanes;
- SCI-ALIGN, SCI-AST, SCI-CAL, SCI-MAP, SCI-NOI, RTC, PTC, and JINC evidence;
- build/performance experiments and established `gw_dev*` history;
- WP-7 RTC timing, fixed-decimation, and successor-baseline work; and
- legacy release/development histories that are not ancestors of application
  mainline.

Ancestry containment is necessary but not sufficient for deletion. A branch
with a live worktree, dirty state, different local/remote tip, validation role,
or historical authority remains retained.

## Worktrees And User Material

Before cleanup there are 26 registered worktrees. Exactly one entry is stale:

```text
/private/tmp/citlali-legacy-apt-dispatch
  branch: codex/repair-native-consumer-extinction
  reason: gitdir file points to non-existent location
```

`git worktree prune --dry-run` reports only that entry. Pruning it removes
administrative metadata; it does not delete a live directory. The two
untracked files formerly associated with that worktree were archived during
the August 26 preservation cycle.

The guarded cleanup completed with
`git worktree prune --expire now`; the stale entry is gone and 25 live
worktrees remain. No live path or file was removed. Following a separate
explicit owner confirmation, the 21 local branches classified above were
deleted. The 19 corresponding remote branches remain unchanged and
user-controlled.

The 25 live paths include 12 detached Codex worktrees and 13 branch worktrees.
Ten detached Codex worktrees are clean; close or archive their app tasks before
manual removal. The following live worktrees contain material that must not be
deleted or reset:

| Worktree | Tracked changes | Untracked files | Disposition |
| --- | ---: | ---: | --- |
| `/private/tmp/citlali-contracts-consolidated` | 56 | 0 | Active scientific-contract work; retain |
| `/private/tmp/citlali-wp7-successor-baseline` | 1 | 2 | Active WP-7 work; retain |
| `~/.codex/worktrees/9448/citlali-refactor` | 0 | 15 | SCI-NOI contract material; retain |
| `~/.codex/worktrees/c9be/citlali-refactor` | 0 | 3 | Independent WP-7 audit evidence; retain |
| `~/GitHub/citlali-perf` | 0 | 1 | Performance evidence; retain |
| `~/GitHub/citlali-refactor` | 0 | 464 | Bundles, contract output, and transfer evidence; retain |

The total is 57 tracked changes and 485 untracked files across these six
worktrees. The August 26 archive preserves the then-current 464-file transfer
set, but the live files remain user-owned and are not removed merely because an
archive exists.

## Tags, Stashes, Codex Refs, And Unreachable Objects

Retain all 13 tags: the four operational tags listed above; historical release
tags `v1.0.0`, `v1.1.0`, `v1.2.0`, `v1.2.1`, `v2.0.0`, `v3.0.0`, `v3.1.0`,
and `v4.0.0`; and scientific freeze tag `sci-jinc-v0.1-r0.3`.

Retain all three stashes:

```text
stash@{0} 75cc620256a460d6002ad367cecf17f7df1dbd1b perf naive_mm work
stash@{1} 1c98f93376776b4dd51494f230c559cf1e031a5b gw_dev RTC work
stash@{2} b27271fc76ffea063341465a5a62112167f50e01 gw_dev JINC/pointing work
```

Do not manually prune `refs/codex/*`. Snapshot
`refs/codex/snapshots/7b0194474e74fa2450c74c7e31c555e96ac50765`
at `2aa5c12ba20f5ca043609b9b75713cf014394065` remains snapshot-only.

`git fsck --unreachable` finds no unreachable commit when reflogs are included.
With reflogs excluded, two commits are visible and therefore remain protected
by reflog history:

```text
2038590de1dba0ba2479e0d78a7dae53cae6ddc7 test: add local WP-7 identity RTC acceptance gate
ca075fb7f7bd1739c65d71050a7682b8e310b31a docs(science): resolve SCI-JINC ODQ-109
```

No manual object pruning, reflog expiration, stash deletion, tag deletion, or
garbage collection is authorized.

## Cleanup Boundary

This cleanup is repository administration, not another source-refactor phase.
Do not delete additional headers or sources, compact the living status history,
remove accepted executable snapshots or validation reports, or reopen mature
RTC/PTC/JINC/Wiener algorithms. The next application work is the governed
audit/repair/re-audit cycle or an explicitly frozen same-SHA four-mode
campaign.
