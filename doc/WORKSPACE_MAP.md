# Refactor workspace map

Verified on 2026-11-02 after owner publication, remote retirement and cache refresh. A **branch** names a line of commits; a **working folder**
is a checkout that may also contain builds, results or files not recorded in Git.
A saved checkout without a branch name is frozen at its recorded commit; it is
not missing work. Counts are inventories, not counts of active projects.

## Where to work

Use `codex/refactor-mainline` for accepted development. Its current working folder is:

`/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor`

The old candidate name is only a directory name. Do not create another mainline
folder to compensate for it. A future rename needs a build/runtime path audit:
Spack stores, evidence and executable bindings can contain absolute paths.

The similarly named `/Users/gwilson/GitHub/citlali-refactor` was the older
SCI-ALIGN workspace and is now retired. The Codex project points to mainline.
Retained checkouts share Git storage at `/Users/gwilson/GitHub/citlali/.git`;
do not move or remove that store.

GitHub mainline was verified at
`8ed25efb416af40d5f9987b41578041638527b34`, with
[passing Python 3.11/3.13 checks](https://github.com/toltec-astro/citlali/actions/runs/41551997054).
Grant performed publication and the separately approved SCI-ALIGN remote deletion.
The local GitHub cache was refreshed: 92 obsolete branch labels were removed,
one existing tip was updated and one missing branch label was added. A follow-up
refresh removed the now-deleted SCI-ALIGN label and advanced the cached mainline
to `8ed25efb4`. The cache represents **28 live branches** plus Git's default-branch
pointer. Thirty historical
tips needing an explicit retained anchor were saved under
`refs/archive/final-cleanup-20261102/origin/`; an additional archive reference
preserves the SCI-ALIGN remote tip. Archive references are recovery locators,
not working branches or new projects.
Local and GitHub branch inventories are different sets. Ongoing pointing/handoff
names remain local-only in this snapshot.

## Five current working folders

Three are refactor-related; two are protected legacy/performance folders.
After [A+B retirement](WORKSPACE_RETIREMENT_2026-10-31.md), the owner approved
closing the obsolete collaborator pilot and SCI-ALIGN checkout. The temporary
metadata repair folder was subsequently created, qualified and retired after its
application was integrated; the five folders below are the current inventory.
Start new development from mainline; keep ongoing pointing work in its named folders.
Paths below are absolute locators on this Mac, not portable deployment settings.

| Folder | Branch or frozen state | Role / next disposition |
| --- | --- | --- |
| `/Users/gwilson/.codex/worktrees/pointing-engineering-candidate-20261030/citlali-refactor` | `codex/refactor-mainline` | Primary accepted development checkout; retain |
| `/Users/gwilson/.codex/worktrees/configurable-point-20261028/citlali-refactor` | `codex/configurable-point-20261028` at `6969dcf81` | Pointing assessments/planning; approved evidence cutoff included; protect any later work |
| `/Users/gwilson/.codex/worktrees/point-handoff-20261030/citlali-refactor` | `codex/point-handoff-20261030` at `1f3bf6db7` | Approved correction/table handoff source; retained separately after integration |
| `/Users/gwilson/GitHub/citlali` | `gw_dev` at `ffc6b9070` | Protected legacy checkout and shared Git store |
| `/Users/gwilson/GitHub/citlali-perf` | `codex/perf-map-accumulation-noise-lifecycle` at `d01ffa598` | Protected legacy/performance checkout; one untracked command note |

The other four working folders have no staged, unstaged or untracked changes.
That check excludes ignored builds/products; a clean Git status alone does not
make a folder disposable.

## Eight local branch names

Five local branch names are attached to folders
in the table. Three have no working folder:

| Branch | Role / disposition |
| --- | --- |
| `codex/baseline-before-successor-default-2026-09-17` at `0f52e1a42` | Frozen comparison checkpoint; retained despite being an ancestor of mainline |
| `codex/scientific-contract-library` at `54475956f` | Scientific-specification reference; one local owner-decision commit outside mainline; retain pending disposition |
| `codex/wp7-rtc-fixed-decimation-authority` at `49fe73e75` | Divergent scientific evidence; preserve, do not resume merely because the branch exists |

The live GitHub inventory has **28 branches**, mostly legacy releases and
historical development. Grant deleted only
`codex/sci-align-001-lissajous-timestream-fit`, formerly at
`353b11887ff04dfd7bca12915917495f81a587fa`. Live verification confirmed its absence
and that all other branch tips were unchanged except the published mainline.
The retired tip remains under `refs/archive/final-cleanup-20261102/github-sci-align`
and in the prior independent SCI-ALIGN history restoration. Codex performed no
remote write; old pending-deletion receipts retain their historical scope.
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

## SCI-ALIGN retired — 2026-11-01

The owner approved closing `/Users/gwilson/GitHub/citlali-refactor` and deleting
local `codex/sci-align-001-lissajous-timestream-fit` at
`dd4a3599279a0d09eeccaf3c63a4ac2782198d57`. All 78 separate commits remain in the
verified `pilot-science-history.bundle` and independent restored repository under
`citlali-workspace-consolidation-20261031`. Its full physical archive and restored
copy cover the checkout's source and ignored build contents. Prior batch C's
238 removed document copies remain recoverable there too.

All remaining 227 changed/untracked originals are preserved: 226 untracked files
were moved, and the modified `AGENTS.md` was copied before its exact restore.
These include all 41 original bundles, 183 analysis files and two audit reports.
The exact manifests, hashes, checks, reviews and original files are in:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/citlali-sci-align-retirement-20261101`

Original files are under `preserved-workspace-files`, retaining relative paths.
The separate sealed assessment packet, `citlali-sci-align-retirement-assessment-20261101`,
records commit dispositions, canonical contract matches and two output-metadata
repair proposals. Those proposals and the later Beammap final-header defect were
resolved in integrated application `012cc7adc`; no old experiment was adopted.

Pointing's 18 dependency-source trees were verified and its build configuration
changed to its own copies. A separate AppleClang/Homebrew build, native tests and
configuration preflight passed. Its original executable is byte-identical; frozen
project executables, inputs and products remain untouched. This supplemental
relocation check does not reproduce the accepted Spack-backed observing campaign.
Dormant generated dependency-dashboard files retain historical source locators;
they are not dependencies of the current build.

The owner changed the Codex project's primary folder to mainline, independently
confirmed through the app. Existing chats were not redirected, archived or
unpinned. Before resuming a chat bound to SCI-ALIGN or another retired folder,
continue explicitly from mainline or restore its authenticated old checkout.

## Completed candidates retired — 2026-11-02

The owner approved retiring local `codex/pointing-handoff-integration-20261031`
at `f2ad2913acf5fee2d7ca8c40545a385ae07aab31` and local
`codex/ptc-metadata-20261101` at `012cc7adc5b50bd3491c47021744310735de3a8a`.
Both exact tips are ancestors of accepted mainline with no unique commits.
The metadata folder
`/Users/gwilson/.codex/worktrees/ptc-metadata-20261101/citlali-refactor`
was removed only after its complete ignored build was archived and its 396
regular-file/symlink entries matched by content, mode and link target. Its
qualified executable is preserved, not installed into another runtime.
The other candidate branch had no working folder.

The operation, archive, build manifest, Git administration copy, preservation
receipts and independent review are under:

`/Users/gwilson/.codex/visualizations/2026/10/29/01a188f2-4d30-7e60-88a8-956b50bb08cf/citlali-final-cleanup-20261102`

`RESTORE.txt` describes restoration. Continue old candidate chats from mainline
for new work. Historical run scripts remain sealed and also bind old mainline
`7a8e2cd8`; restoring candidate `012cc7adc` alone does not make them reusable.
Prepare a fresh owner-reviewed run packet or separately restore the exact old
environment. Never reset current mainline to satisfy an old runner. Restored
builds require the original source path, retained mainline `build/spack-sources`
and the separate `build/spack-candidate` Spack store/environment. Grant alone
executes builds, tests, probes and reductions. No runtime was launched here.

## Consolidation reached

There are now **three refactor working folders plus two protected folders**,
**eight local branches**, and **three preserved stashes**. New development starts
from mainline. Both ongoing pointing/handoff folders, Citlali 4.x/5.x and all prior
recovery packets remain protected. The GitHub SCI-ALIGN branch is retired,
leaving 28 live GitHub branches. The repository-consolidation batch is complete.

Engineering and the parked `884f` contract handoff can be restored from the
verified archives described in [the retirement record](WORKSPACE_RETIREMENT_2026-10-31.md).
The scientific program is not declared complete by closing its old working
folders. This map records completed cleanup and grants no further deletion authority.
