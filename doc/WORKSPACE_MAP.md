# Refactor workspace map

Verified on 2026-10-31. A **branch** names a line of commits; a **working folder**
is a checkout that may also contain builds, results or files not recorded in Git.
A saved checkout without a branch name is frozen at its recorded commit; it is
not missing work. Counts are inventories, not counts of active projects.

## Where to work

Use `codex/refactor-mainline` for accepted development. Its current working folder is:

`/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor`

The old candidate name is only a directory name. Do not create another mainline
folder to compensate for it. A future rename needs a build/runtime path audit:
Spack stores, evidence and executable bindings can contain absolute paths.

The similarly named `/Users/gwilson/GitHub/citlali-refactor` is an older protected
scientific workspace, **not** the current mainline. All these checkouts share Git
storage at `/Users/gwilson/GitHub/citlali/.git`; do not move or remove that store.

Published mainline `8078ba4567b3096b74ee378a99acfca270e27c3f` passed
[GitHub checks](https://github.com/toltec-astro/citlali/actions/runs/41201517670).
A cached `origin/...` label can lag the live GitHub branch: the old “ahead 683”
message at this audit was a stale local comparison, not 683 unpublished commits.
Local and GitHub branch inventories are different sets, not two copies of every
branch; ongoing pointing/handoff names were local-only at this snapshot.

## Nine retained working folders

Seven are refactor-related; two are protected legacy/performance folders.
The approved integration reuses the mainline folder on one candidate branch
during qualification, then switches it back to mainline. No new folder is added.
Paths below are absolute locators on this Mac, not portable deployment settings.

| Folder | Branch or frozen state | Role / next disposition |
| --- | --- | --- |
| `/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor` | `codex/refactor-mainline` | Primary accepted development checkout; retain |
| `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor` | `codex/configurable-point-20261028` at `6969dcf81` | Pointing assessments/planning; approved evidence cutoff included; protect any later work |
| `/Users/gwilson/.codex/worktrees/point-handoff-20261030/citlali-refactor` | `codex/point-handoff-20261030` at `1f3bf6db7` | Approved correction/table handoff source; retained separately after integration |
| `/Users/gwilson/.codex/worktrees/collaborator-pilot-20261027/citlali-refactor` | `codex/collaborator-pilot-20261027` at `16d1b1aae` | Earlier collaborator workflow; compare separate work before retirement |
| `/Users/gwilson/.codex/worktrees/citlali-engineering-20261029/citlali-refactor` | Frozen `5c4620105`; branch retired | Committed work incorporated; folder-retirement candidate after build/evidence inventory and approval |
| `/Users/gwilson/.codex/worktrees/884f/citlali-refactor` | Frozen `ae953ed4d` | Scientific-contract program handoff; retain pending scientific-workspace reconciliation |
| `/Users/gwilson/GitHub/citlali-refactor` | `codex/sci-align-001-lissajous-timestream-fit` at `dd4a35992` | Older scientific work and evidence; 465 modified/untracked paths at audit; preserve and sort separately |
| `/Users/gwilson/GitHub/citlali` | `gw_dev` at `ffc6b9070` | Protected legacy checkout and shared Git store |
| `/Users/gwilson/GitHub/citlali-perf` | `codex/perf-map-accumulation-noise-lifecycle` at `d01ffa598` | Protected legacy/performance checkout; one untracked command note |

The other seven working folders had no staged, unstaged or untracked changes.
That check excludes ignored builds/products; a clean Git status alone does not
make a folder disposable.

## Eleven local branch names after candidate preparation

After the gated mainline move, seven local branch names are attached to folders
in the table. Four have no working folder:

| Branch | Role / disposition |
| --- | --- |
| `codex/pointing-handoff-integration-20261031` | Retained exact integration candidate; mainline receives it only after gates and review; later retirement requires its own disposition |
| `codex/baseline-before-successor-default-2026-09-17` at `0f52e1a42` | Frozen comparison checkpoint; retained despite being an ancestor of mainline |
| `codex/scientific-contract-library` at `54475956f` | Scientific-specification reference; one local owner-decision commit outside mainline; retain pending disposition |
| `codex/wp7-rtc-fixed-decimation-authority` at `49fe73e75` | Divergent scientific evidence; preserve, do not resume merely because the branch exists |

GitHub has 29 branches, mostly legacy releases and historical development.
`codex/structural-refactor` remains a protected GitHub reference. All Citlali
4.x/5.x branches stay protected. Neither an old name nor complete ancestry alone
authorizes retirement of an owner-retained comparison or scientific reference.

## Saved edits and recovery

All three stashes belong to legacy/performance work: a performance `naive_mm`
change, legacy RTC work, and legacy JINC/pointing edits. Keep them untouched.

The latest branch retirement removed only local
`codex/citlali-engineering-20261029`, local
`codex/pointing-engineering-candidate-20261030`, and GitHub
`codex/citlali-engineering-20261029`. Their history remains in mainline and archive
refs; no working folder was deleted. Recovery is under:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/branch-retirement-20261031.iVqkWr`

It includes a verified bundle, independently restored repository, branch logs,
and before/after inventories. Older recovery packets and frozen pointing runtimes
remain protected. This map is not authorization for cleanup.

## Intended consolidation

Aim for 3–5 refactor working folders plus the two protected legacy/performance
folders: mainline, ongoing pointing, and the scientific workspace(s) still needed.
Complete and record the [approved pointing integration](POINTING_HANDOFF_INTEGRATION_2026-10-31.md).
Then consider engineering and pilot folder retirement in a separate approved
batch. Sort the older scientific workspace before deciding which scientific
folders can close. Preserve current work and evidence independently of how many
branch names or folders remain.
