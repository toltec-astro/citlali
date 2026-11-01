# Refactor workspace map

Verified on 2026-11-01. A **branch** names a line of commits; a **working folder**
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

GitHub mainline was last verified at
`f089a5d665877dde4e32f006cb81717d28f647aa`, with
[passing Python 3.11/3.13 checks](https://github.com/toltec-astro/citlali/actions/runs/41311399974).
The subsequent pilot-retirement documentation is local until owner publication.
A cached `origin/...` label can lag the live GitHub branch: the old “ahead 683”
message at this audit was a stale local comparison, not 683 unpublished commits.
Local and GitHub branch inventories are different sets, not two copies of every
branch; ongoing pointing/handoff names were local-only at this snapshot.

## Six current working folders

Four are refactor-related; two are protected legacy/performance folders.
After [A+B retirement](WORKSPACE_RETIREMENT_2026-10-31.md), the owner approved
closing the obsolete collaborator pilot. No replacement working folder was added.
Start new development from mainline; keep ongoing pointing work in its named folders.
Paths below are absolute locators on this Mac, not portable deployment settings.

| Folder | Branch or frozen state | Role / next disposition |
| --- | --- | --- |
| `/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor` | `codex/refactor-mainline` | Primary accepted development checkout; retain |
| `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor` | `codex/configurable-point-20261028` at `6969dcf81` | Pointing assessments/planning; approved evidence cutoff included; protect any later work |
| `/Users/gwilson/.codex/worktrees/point-handoff-20261030/citlali-refactor` | `codex/point-handoff-20261030` at `1f3bf6db7` | Approved correction/table handoff source; retained separately after integration |
| `/Users/gwilson/GitHub/citlali-refactor` | `codex/sci-align-001-lissajous-timestream-fit` at `dd4a35992` | Retain: 227 changed/untracked paths remain after batch C; all 465 original paths are cataloged and backed up; retained files and live build dependencies protected |
| `/Users/gwilson/GitHub/citlali` | `gw_dev` at `ffc6b9070` | Protected legacy checkout and shared Git store |
| `/Users/gwilson/GitHub/citlali-perf` | `codex/perf-map-accumulation-noise-lifecycle` at `d01ffa598` | Protected legacy/performance checkout; one untracked command note |

The other four working folders had no staged, unstaged or untracked changes.
That check excludes ignored builds/products; a clean Git status alone does not
make a folder disposable.

## Ten local branch names

Six local branch names are attached to folders
in the table. Four have no working folder:

| Branch | Role / disposition |
| --- | --- |
| `codex/pointing-handoff-integration-20261031` | Retained exact integrated candidate at `f2ad2913a`; later branch retirement requires its own disposition |
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

An earlier branch retirement removed only local
`codex/citlali-engineering-20261029`, local
`codex/pointing-engineering-candidate-20261030`, and GitHub
`codex/citlali-engineering-20261029`. Their history remains in mainline and archive
refs; no working folder was deleted in that earlier branch-retirement operation. Recovery is under:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/branch-retirement-20261031.iVqkWr`

It includes a verified bundle, independently restored repository, branch logs,
and before/after inventories. Older recovery packets and frozen pointing runtimes
remain protected. This map is not authorization for cleanup.

## Collaborator pilot closed — 2026-11-01

The owner approved retiring the pilot folder and local
`codex/collaborator-pilot-20261027` branch at `16d1b1aaeaed3526cb6383672d47f1fc770e78c4`.
All 11 separate commits remain in the previously verified `pilot-science-history.bundle`
and its independent restored repository under `citlali-workspace-consolidation-20261031`.
The disposable build tree and generated test-discovery file were discarded with
the clean worktree; no new bulk build archive or runtime migration was made.

Five legacy pilot entry points in `COM01-local-trial-20261027` now exit 64:
`pointings/02_redu.sh`, `pointings-152389/02_redu.sh`,
`refactor/NGC4449/02_redu.sh`, `software/preparation-20261028/bin/prepare-point-152389`
and `pointings-152389/.tolproj/citlali-launcher`. Other pointing directories in
that shared project remain active and unchanged. Input links, APTs, products,
frozen executables and existing backups remain. Original scripts total less than
6 KB and are recorded with the operation; old source locators remain provenance.

Approval, bounded process-check reassessment, independent reviews, exact script
changes, worktree removal, branch deletion and before/after checks:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/citlali-pilot-retirement-20261101`

This closes the experiment without adopting its old proposals or claiming that
the collaborator release is complete. Future work starts from current mainline.

## Scientific workspace: batch C completed

The owner approved [batch C](SCIENTIFIC_WORKSPACE_CLEANUP_2026-11-01.md).
Exactly 238 archived historical document working copies were removed after fresh
checks and independent script review, reducing the changed/untracked count from
465 to 227. All 41 original bundles, the local instructions edit, two audit
comparisons, 183 pointing-analysis artifacts, tracked source and ignored build
dependencies remain unchanged. No folder or branch was retired. Recovery
instructions and exact execution evidence are linked from the cleanup record.

## Consolidation reached

The target of 3–5 refactor working folders is met at four, plus the two protected
legacy/performance folders. Mainline, both ongoing pointing/handoff folders and
the older scientific workspace remain. Ten local branches, three stashes and all
prior recovery packets remain.

Engineering and the parked `884f` contract handoff can be restored from the
verified archives described in [the retirement record](WORKSPACE_RETIREMENT_2026-10-31.md).
The owner accepted that the parked contract chat's old working folder must be
restored before resuming there. Its scientific program is not declared complete;
no chat was archived or altered. The scientific catalog separates local work,
accepted-authority routes, historical evidence and recovery archives. Batch C
subsequently removed only its 238 approved working copies; all 465 original paths
remain recoverable and the other 227 remain in place. Further cleanup requires a
new exact disposition.
